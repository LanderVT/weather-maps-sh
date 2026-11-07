"""Cloud runner: fresh Hybrid + ECMWF + ICON-D2 forecasts -> GitHub Pages.

Runs the original science scripts in an isolated HOME, never uploads GRIB/NetCDF
or intermediate arrays. Every publish is a new self-contained website; there is
no carryover of old maps when one of the models fails.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from html import escape
import importlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
WORK = ROOT / "_run"
MAPS = WORK / "maps"
SITE = ROOT / "site"
SOURCES = (
    "SnowHybrid.py", "SnowDays_AROME.py", "SnowDays_AROME-AT.py",
    "SnowDays_ICON-CH1.py", "SnowDays_ICON-CH2.py",
    "SnowDays_ICON-D2.py", "SnowDays_ECMWF.py", "run_ecmwf.py",
    "SnowAll_ICON-D2.py", "run_icon_d2.py",
)
MODELS = ("hybrid", "ecmwf", "icon_d2")
MODEL_NAMES = {
    "hybrid": "Hybrid Model", "ecmwf": "ECMWF Model", "icon_d2": "ICON-D2 Model"
}
# Five official Alpine webcam pages. These are regular links, not live embeds,
# so the images remain lightweight and the website requires no third-party JS.
# To add/remove a camera later, edit only this list.
HYBRID_WEBCAMS = (
    {"name": "Chamonix", "lon": 6.870, "lat": 45.924,
     "url": "https://www.chamonix.com/webcams"},
    {"name": "Zermatt", "lon": 7.748, "lat": 46.020,
     "url": "https://zermatt.swiss/en/info/webcams"},
    {"name": "Engelberg / Titlis", "lon": 8.407, "lat": 46.819,
     "url": "https://www.engelberg.ch/webcams/"},
    {"name": "St. Anton", "lon": 10.264, "lat": 47.129,
     "url": "https://www.stantonamarlberg.com/en/webcams"},
    {"name": "Val Gardena", "lon": 11.719, "lat": 46.575,
     "url": "https://www.valgardena.it/en/webcams/"},
)

GROUPS = (
    "Sneeuwval per dag", "Cumulatieve sneeuwval", "Nieuwe sneeuw op de grond",
    "Sneeuwgrens", "Vorstgrens", "Kans op sneeuw", "Modelonenigheid",
)


def ensure_logo(tmp_home: Path) -> None:
    """Use the owner's logo or an unbranded fallback for both model panels."""
    target = tmp_home / "logo-sneeuwhoogte.png"
    original = ROOT / "logo-sneeuwhoogte.png"
    if original.is_file():
        shutil.copy2(original, target)
        print("Logo: original logo-sneeuwhoogte.png", flush=True)
        return

    from PIL import Image, ImageDraw, ImageFont
    image = Image.new("RGBA", (420, 112), (255, 255, 255, 0))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((2, 2, 418, 109), radius=16, fill=(16, 57, 108, 235))
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 47)
    except OSError:
        font = ImageFont.load_default()
    draw.text((18, 27), "SNEEUW", font=font, fill="white")
    image.save(target)
    print("Logo: generic SNEEUW placeholder; upload your own if desired", flush=True)


def map_group(name: str) -> str:
    stem = name.lower()
    if "vorstgrens" in stem:
        return "Vorstgrens"
    if "sneeuwgrens" in stem:
        return "Sneeuwgrens"
    if "kans" in stem:
        return "Kans op sneeuw"
    if "onenigheid" in stem:
        return "Modelonenigheid"
    if "grond" in stem:
        return "Nieuwe sneeuw op de grond"
    if "cumulatief" in stem:
        return "Cumulatieve sneeuwval"
    return "Sneeuwval per dag"


def display_title(name: str) -> str:
    """Friendly times in Belgium/Swiss local time; model timestamps stay UTC."""
    from zoneinfo import ZoneInfo

    stem = Path(name).stem
    match = re.search(r"_(alps|full)_(\d{8})_(\d{2})z_f(\d{3})(?:-(\d{3}))?", stem)
    if not match:
        return stem.replace("_", " ")
    area, ymd, hour, start, end = match.groups()
    region = "Alpen" if area == "alps" else "West-Europa"
    run = datetime.strptime(ymd + hour, "%Y%m%d%H").replace(tzinfo=timezone.utc)
    tz = ZoneInfo("Europe/Brussels")
    days = ("ma", "di", "wo", "do", "vr", "za", "zo")

    def label(dt):
        local = dt.astimezone(tz)
        return f"{days[local.weekday()]} {local:%d/%m %H:%M}"

    if end is None:
        return f"{region} · {label(run + timedelta(hours=int(start)))}"
    return (f"{region} · {label(run + timedelta(hours=int(start)))} "
            f"- {label(run + timedelta(hours=int(end)))}")


# The fallback box is calibrated for the fixed Alpine map layout rendered
# by SnowDays_ICON-CH2.py (13.5 x 8.1 in, PlateCarree, cropped PNG).  It is
# only used if the optional sidecar written by SnowHybrid.py is missing or
# malformed.  Prefer precise georeferencing whenever possible.
HYBRID_ALPS_FALLBACK_BOX = (0.7, 5.5, 98.6, 80.0)
HYBRID_ALPS_EXTENT = (5.0, 16.8, 43.6, 49.0)


def webcam_pin_links(source: Path) -> str:
    """Create five clickable camera markers on the Hybrid PNG.

    An exact crop calibration is read from each *.png.geo.json sidecar.
    Previous versions silently returned zero markers if the sidecar was
    missing. Fall back to the known fixed Alpen map layout instead, while
    logging the issue so it is diagnosable from GitHub Actions.
    """
    meta_path = Path(str(source) + '.geo.json')
    calibration = None
    source_type = 'exact'

    if meta_path.is_file():
        try:
            data = json.loads(meta_path.read_text(encoding='utf-8'))
            left, top, width, height = map(float, data['plot_box_pct'])
            west, east, south, north = map(float, data['extent_lonlat'])
            if not (east > west and north > south and width > 0 and height > 0
                    and 0 <= left <= 100 and 0 <= top <= 100
                    and left + width <= 100.5 and top + height <= 100.5):
                raise ValueError('Invalid crop calibration')
            calibration = (left, top, width, height, west, east, south, north)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            print(f'WAARSCHUWING: ongeldige webcam-georeferentie voor {source.name}: {exc}', flush=True)

    if calibration is None:
        if '_alps_' not in source.name.lower():
            print(f'WAARSCHUWING: geen webcam-georeferentie voor {source.name}', flush=True)
            return ''
        left, top, width, height = HYBRID_ALPS_FALLBACK_BOX
        west, east, south, north = HYBRID_ALPS_EXTENT
        calibration = (left, top, width, height, west, east, south, north)
        source_type = 'fallback'
        print(f'Webcams: {source.name}: geen bruikbare .geo.json; '
              'benaderde Alpenpositie gebruikt', flush=True)

    def render(cal):
        left, top, width, height, west, east, south, north = cal
        links = []
        for cam in HYBRID_WEBCAMS:
            if not (west <= cam['lon'] <= east and south <= cam['lat'] <= north):
                continue
            x = left + (cam['lon'] - west) / (east - west) * width
            y = top + (north - cam['lat']) / (north - south) * height
            if not (0 <= x <= 100 and 0 <= y <= 100):
                continue
            url = escape(cam['url'], quote=True)
            name = escape(cam['name'], quote=True)
            links.append(
                f'<a class="webcam-pin" href="{url}" target="_blank" '
                f'rel="noopener noreferrer" title="Webcam {name}" '
                f'aria-label="Webcam {name} openen in nieuw tabblad" '
                f'style="left:{x:.4f}%;top:{y:.4f}%">'
                f'<span aria-hidden="true">📷</span></a>'
            )
        return links

    links = render(calibration)
    # A sidecar may be valid structurally but have the wrong geographic
    # extent, leaving no camera visible. Recover on known Alpine maps.
    if not links and source_type == 'exact' and '_alps_' in source.name.lower():
        left, top, width, height = HYBRID_ALPS_FALLBACK_BOX
        west, east, south, north = HYBRID_ALPS_EXTENT
        links = render((left, top, width, height, west, east, south, north))
        print(f'Webcams: {source.name}: zijbestand bevat geen camerapunten; '
              'benaderde Alpenpositie gebruikt', flush=True)
    print(f'Webcams {source.name}: {len(links)} symbolen', flush=True)
    return ''.join(links)


def collect_models(errors: dict[str, str]) -> dict[str, dict]:
    """Validate each run separately before copying any maps to the site."""
    output = {}
    for model in MODELS:
        folder = MAPS / model
        images = sorted(
            p for p in folder.glob("*.png")
            if p.is_file() and p.stat().st_size > 100
        )
        meta_path = folder / "forecast_metadata.json"
        meta = {}
        reason = errors.get(model)
        if reason is None:
            if not meta_path.is_file():
                reason = "Ontbrekende forecast-metadata"
            else:
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                except (OSError, ValueError) as exc:
                    reason = f"Metadata onleesbaar: {exc}"
        if reason is None and not images:
            reason = "Geen nieuwe kaarten gegenereerd"
        if reason is None and int(meta.get("map_count", -1)) != len(images):
            reason = f"Aantal kaarten ({len(images)}) wijkt af van model-uitvoer"
        output[model] = {
            "available": reason is None,
            "error": reason,
            "metadata": meta if reason is None else {},
            "images": images if reason is None else [],
        }
    return output


def make_site(errors: dict[str, str] | None = None) -> str:
    """Publish three independent model tabs (or explicit error panels)."""
    errors = errors or {}
    results = collect_models(errors)
    if SITE.exists():
        shutil.rmtree(SITE)
    SITE.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)

    total_bytes = sum(
        p.stat().st_size for m in results.values() for p in m["images"]
    )
    if total_bytes > 750_000_000:
        # Avoid attempting to publish an oversized GitHub Pages artifact.
        for model, result in results.items():
            result["available"] = False
            result["error"] = "Kaarten samen >750 MB: verlaag resolutie/DPI"
            result["images"] = []
            result["metadata"] = {}

    for model, result in results.items():
        if result["available"]:
            destination = SITE / "maps" / model
            destination.mkdir(parents=True, exist_ok=True)
            for source in result["images"]:
                shutil.copy2(source, destination / source.name)

    successes = sum(int(r["available"]) for r in results.values())
    overall = "ok" if successes == len(MODELS) else "partial" if successes > 0 else "error"
    status = {
        "status": overall,
        "updated_utc": now.isoformat(timespec="seconds"),
        "fresh_until_utc": (now + timedelta(hours=6)).isoformat(timespec="seconds"),
        "models": {},
    }
    for model, result in results.items():
        meta = result["metadata"]
        status["models"][model] = {
            "status": "ok" if result["available"] else "error",
            "map_count": len(result["images"]),
            "error": result["error"],
            "forecast_start_utc": meta.get("forecast_start_utc"),
            "forecast_end_utc": meta.get("forecast_end_utc"),
            "model_runs_utc": meta.get("model_runs_utc", {}),
            "areas": meta.get("areas", ["alps"]),
            "days": meta.get("days"),
        }
    (SITE / "status.json").write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")

    panels = []
    for model in MODELS:
        result = results[model]
        meta = result["metadata"]
        good = result["available"]
        groups = {name: [] for name in GROUPS}
        if good:
            for source in result["images"]:
                name = escape(source.name, quote=True)
                label = escape(display_title(source.name))
                url = f"maps/{model}/{name}"
                cameras = webcam_pin_links(source) if model == "hybrid" else ""
                groups[map_group(source.name)].append(
                    f'<article class="map-card"><div class="map-frame">'
                    f'<a class="map-image-link" href="{url}" target="_blank" rel="noopener">'
                    f'<img loading="lazy" src="{url}?v={int(now.timestamp())}" alt="{label}"></a>'
                    f'{cameras}</div><p>{label}</p></article>'
                )
        if good:
            sections = "".join(
                f'<section class="map-group"><h2>{escape(group)}</h2>'
                f'<div class="grid">{"".join(cards)}</div></section>'
                for group, cards in groups.items() if cards
            )
            model_runs = meta.get("model_runs_utc", {})
            runs = ", ".join(
                f"{escape(name)}: {escape(str(timestamp).replace('T', ' ')[:16])} UTC"
                for name, timestamp in model_runs.items()
            )
            if model == "hybrid":
                intro = f"48 uur · {len(result['images'])} kaarten · {len(model_runs)} modellen"
            elif model == "ecmwf":
                regions = ", ".join("Alpen" if a == "alps" else "West-Europa" for a in meta.get("areas", []))
                intro = f"{escape(str(meta.get('days', 7)))} dagen · {regions} · {len(result['images'])} kaarten"
            else:
                regions = ", ".join("Alpen" if a == "alps" else "West-Europa" for a in meta.get("areas", []))
                intro = (f"ICON-D2 · 48 uur · {regions} · {len(result['images'])} kaarten "
                         f"· sneeuw- en vorstgrens om de 6 uur")
            detail = f'<p class="detail">{runs}</p>' if runs else ""
            webcam_toolbar = (
                '<div class="webcam-toolbar">'
                '<label><input type="checkbox" id="toggle-hybrid-webcams">'
                ' 📷 Toon 5 webcams op de Hybrid-kaarten</label>'
                '<span>Klik op een camera voor de officiële livebeelden.</span>'
                '</div>'
            ) if model == "hybrid" else ""
            markup = (
                f'<p class="panel-intro">{intro}</p>{detail}'
                f'<div class="model-status good">Nieuwe kaarten beschikbaar</div>'
                f'{webcam_toolbar}{sections}'
            )
        else:
            markup = (
                '<div class="model-status bad">Geen actuele kaarten voor dit model in deze verwerking. '
                'De vorige kaarten worden niet opnieuw getoond. '
                'Controleer de GitHub Actions-log.</div>'
            )
        panels.append(
            f'<div id="panel-{model}" class="model-panel" role="tabpanel" '
            f'aria-labelledby="tab-{model}" {"" if model == "hybrid" else "hidden"}>'
            f'{markup}</div>'
        )

    now_str = now.strftime("%d-%m-%Y %H:%M UTC")
    html = f'''<!doctype html>
<html lang="nl">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate">
<title>Sneeuwhoogte · Hybrid, ECMWF & ICON-D2</title>
<style>
:root {{ color-scheme:light; font-family:system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif; }}
* {{ box-sizing:border-box; }}
body {{ background:#f4f7fb; color:#16283d; margin:0; }}
header {{ color:white; padding:30px max(18px,calc((100vw - 1240px)/2)); background:linear-gradient(115deg,#10396c,#1a638a); }}
h1 {{ font-size:clamp(1.65rem,3vw,2.55rem); margin:0 0 8px; }}
header p {{ color:#dcecf9; margin:5px 0; }}
main {{ max-width:1240px; margin:0 auto; padding:22px 18px 60px; }}
.tabs {{ display:flex; flex-wrap:wrap; gap:10px; margin:4px 0 24px; }}
.tab {{ appearance:none; border:1px solid #c5d3e3; background:white; color:#25415b; font-weight:750; font-size:1rem; padding:13px 23px; border-radius:10px; cursor:pointer; }}
.tab:hover {{ background:#e8f0f9; }}
.tab[aria-selected="true"] {{ background:#10396c; color:white; border-color:#10396c; }}
.tab:focus-visible {{ outline:3px solid #3092c3; outline-offset:3px; }}
.model-panel[hidden],.map-group[hidden] {{ display:none; }}
.panel-intro {{ font-size:1.08rem; font-weight:700; margin:0 0 4px; }}
.detail {{ margin:0 0 16px; color:#5d6d82; font-size:.85rem; line-height:1.6; }}
.model-status {{ padding:15px 18px; border-radius:9px; margin:18px 0; font-weight:650; }}
.model-status.good {{ background:#eaf6ef; color:#17633b; border:1px solid #b7dfc6; }}
.model-status.bad {{ background:#fff0ec; color:#8a3026; border:1px solid #e7b8b0; }}
h2 {{ font-size:1.2rem; margin:34px 0 15px; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(min(100%,380px),1fr)); gap:18px; }}
.map-card {{ background:white; border-radius:10px; overflow:hidden; border:1px solid #e0e7ef; box-shadow:0 1px 4px #0000000d; }}
.map-card img {{ display:block; width:100%; height:auto; }}
.map-frame {{ position:relative; }}
.map-image-link {{ display:block; }}
.webcam-pin {{ position:absolute; z-index:3; transform:translate(-50%,-50%); width:28px; height:28px; display:flex; align-items:center; justify-content:center; text-decoration:none; background:#fffef4; border:2px solid #135e88; border-radius:50%; box-shadow:0 1px 4px #0008; font-size:15px; line-height:1; }}
.webcam-pin:hover {{ background:#d5f0ff; transform:translate(-50%,-50%) scale(1.16); z-index:4; }}
.webcam-pin:focus-visible {{ outline:3px solid #ffa400; outline-offset:2px; }}
#panel-hybrid.hide-webcams .webcam-pin {{ display:none; }}
.webcam-toolbar {{ display:flex; gap:10px 20px; align-items:center; flex-wrap:wrap; margin:18px 0 10px; background:#fff; padding:12px 15px; border:1px solid #d8e2ee; border-radius:10px; }}
.webcam-toolbar label {{ font-weight:750; cursor:pointer; }}
.webcam-toolbar input {{ width:17px; height:17px; vertical-align:middle; accent-color:#10396c; }}
.webcam-toolbar span {{ color:#516278; font-size:.84rem; }}
.map-card p {{ padding:0 14px; font-size:.83rem; color:#516278; overflow-wrap:anywhere; }}
footer {{ max-width:1240px; margin:0 auto; color:#65758b; padding:0 18px 28px; font-size:.82rem; line-height:1.6; }}
@media(max-width:600px) {{ .tabs .tab {{ flex:1; min-width:140px; }} }}
</style>
</head>
<body>
<header><h1>Sneeuwverwachting Alpen</h1>
<p>Hybride voorspelling, ECMWF en ICON-D2 als afzonderlijke modellen</p>
<p>Laatst verwerkt: {now_str}</p></header>
<main>
<nav class="tabs" role="tablist" aria-label="Kies het weermodel">
<button type="button" id="tab-hybrid" class="tab" role="tab" aria-controls="panel-hybrid" aria-selected="true" data-model="hybrid">Hybrid Model</button>
<button type="button" id="tab-ecmwf" class="tab" role="tab" aria-controls="panel-ecmwf" aria-selected="false" data-model="ecmwf">ECMWF Model</button>
<button type="button" id="tab-icon_d2" class="tab" role="tab" aria-controls="panel-icon_d2" aria-selected="false" data-model="icon_d2">ICON-D2 Model</button>
</nav>
{''.join(panels)}
</main>
<footer>Experimentele sneeuwverwachting, geen officiële waarschuwing. Bronnen: Météo-France, GeoSphere Austria, MeteoSwiss, DWD en ECMWF. Controleer de modelrun en het actualiteitstijdstip.</footer>
<script>
function selectModel(model) {{
  if (!['hybrid','ecmwf','icon_d2'].includes(model)) model='hybrid';
  for (const key of ['hybrid','ecmwf','icon_d2']) {{
    document.getElementById('panel-'+key).hidden = key !== model;
    document.getElementById('tab-'+key).setAttribute('aria-selected', String(key === model));
  }}
  if (location.hash !== '#'+model) history.replaceState(null, '', '#'+model);
}}
for (const tab of document.querySelectorAll('button[data-model]')) {{
  tab.addEventListener('click', () => selectModel(tab.dataset.model));
}}
window.addEventListener('hashchange', () => selectModel(location.hash.slice(1)));
selectModel(location.hash.slice(1));
const webcamCheckbox = document.getElementById('toggle-hybrid-webcams');
if (webcamCheckbox) {{
  document.getElementById('panel-hybrid').classList.add('hide-webcams');
  webcamCheckbox.addEventListener('change', () => {{
    document.getElementById('panel-hybrid').classList.toggle(
      'hide-webcams', !webcamCheckbox.checked
    );
  }});
}}
const updated = Date.parse({json.dumps(now.isoformat())});
if (Date.now() - updated > 6*60*60*1000) {{
  for (const panel of document.querySelectorAll('.model-panel')) {{
    panel.querySelectorAll('.map-group').forEach(group => group.hidden = true);
    panel.querySelectorAll('.model-status.good').forEach(status => status.hidden = true);
    const msg = document.createElement('div');
    msg.className = 'model-status bad';
    msg.textContent = 'De laatste verwerking is ouder dan 6 uur. De kaarten worden verborgen tot een nieuwe update beschikbaar is.';
    panel.prepend(msg);
  }}
}}
</script>
</body></html>'''
    (SITE / "index.html").write_text(html, encoding="utf-8")
    webcam_total = html.count('class="webcam-pin"')
    hybrid_maps = len(results['hybrid']['images'])
    print(f"WEB-CAM CONTROLE: {webcam_total} klikbare symbolen "
          f"op {hybrid_maps} Hybrid-kaarten", flush=True)
    if hybrid_maps and webcam_total == 0:
        print('::warning::Webcams ontbreken op alle Hybrid-kaarten; '
              'controleer de geo-metadata en run_forecast.py', flush=True)
    print(
        f"Website: {overall}; " + ", ".join(
            f"{MODEL_NAMES[m]} {len(results[m]['images'])} PNG" for m in MODELS
        ) + f"; total {total_bytes / 1e6:.1f} MB", flush=True,
    )
    return overall


def check_sources() -> None:
    missing = [name for name in SOURCES if not (ROOT / name).is_file()]
    if missing:
        raise FileNotFoundError("Ontbrekende Python-bestanden: " + ", ".join(missing))

    for module_name in ("numpy", "matplotlib", "scipy", "requests", "xarray", "netCDF4", "eccodes", "cartopy", "cfgrib", "PIL"):
        importlib.import_module(module_name)
        print(f"  Dependency OK: {module_name}", flush=True)
    for path in sorted(ROOT.glob("SnowDays_*.py")):
        module_name = "check_" + path.stem.replace("-", "_")
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError("Cannot import " + path.name)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        print(f"  Model script OK: {path.name}", flush=True)
    for path in (ROOT / "SnowAll_ICON-D2.py", ROOT / "run_icon_d2.py"):
        spec = importlib.util.spec_from_file_location("check_" + path.stem.replace("-", "_"), path)
        if spec is None or spec.loader is None:
            raise RuntimeError("Cannot import " + path.name)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        print(f"  Model script OK: {path.name}", flush=True)
    print("SOURCE AND DEPENDENCY CHECK OK (no downloads)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Only check sources and libraries")
    options = parser.parse_args()

    missing = [name for name in SOURCES if not (ROOT / name).is_file()]
    if missing:
        raise FileNotFoundError("Ontbrekende bestanden: " + ", ".join(missing))
    if options.check:
        check_sources()
        return

    if WORK.exists():
        shutil.rmtree(WORK)
    if SITE.exists():
        shutil.rmtree(SITE)
    for folder in (MAPS / "hybrid", MAPS / "ecmwf", MAPS / "icon_d2", WORK / "home", WORK / "tmp", WORK / "matplotlib"):
        folder.mkdir(parents=True, exist_ok=True)
    errors: dict[str, str] = {}
    try:
        ensure_logo(WORK / "home")
        env = os.environ.copy()
        env.update({
            "HOME": str(WORK / "home"),
            "TMPDIR": str(WORK / "tmp"),
            "MPLCONFIGDIR": str(WORK / "matplotlib"),
            "XDG_CACHE_HOME": str(WORK / "cache"),
            "HYBRID_OUTPUT_DIR": str(MAPS / "hybrid"),
            "ECMWF_OUTPUT_DIR": str(MAPS / "ecmwf"),
            "ICON_D2_OUTPUT_DIR": str(MAPS / "icon_d2"),
            "PYTHONUNBUFFERED": "1",
            "MPLBACKEND": "Agg",
        })
        for model, script in (("hybrid", "SnowHybrid.py"), ("ecmwf", "run_ecmwf.py"), ("icon_d2", "run_icon_d2.py")):
            print(f"\n=== RUNNING {MODEL_NAMES[model]} ===", flush=True)
            try:
                subprocess.run([sys.executable, str(ROOT / script)], cwd=ROOT, env=env, check=True)
            except subprocess.CalledProcessError as exc:
                errors[model] = f"{script} returned exit code {exc.returncode}"
                print(f"ERROR: {MODEL_NAMES[model]} failed: {exc}", file=sys.stderr, flush=True)
    except Exception as exc:
        for model in MODELS:
            errors.setdefault(model, f"{type(exc).__name__}: {exc}")
        print(f"RUNNER ERROR: {exc}", file=sys.stderr, flush=True)
    finally:
        try:
            make_site(errors)
        finally:
            shutil.rmtree(WORK, ignore_errors=True)
            print("All temporary model downloads and working maps removed.", flush=True)


if __name__ == "__main__":
    main()
