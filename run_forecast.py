"""Cloud runner: new model data -> fresh maps -> one replaceable GitHub Pages site.

Everything downloaded is written under _run/, removed in finally. The website
contains only newly generated PNG images and metadata (never model GRIB/NetCDF).
A failed forecast gets a replacement error-only site, not outdated maps.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone, timedelta
from html import escape
import json
import os
from pathlib import Path
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
    "SnowDays_ICON-D2.py", "SnowDays_ECMWF.py",
)


def ensure_logo(tmp_home: Path) -> None:
    """Supply an actual user logo, or create a clearly generic fallback."""
    target = tmp_home / "logo-sneeuwhoogte.png"
    original = ROOT / "logo-sneeuwhoogte.png"
    if original.exists():
        shutil.copy2(original, target)
        print("Logo: logo-sneeuwhoogte.png uit repository", flush=True)
        return

    from PIL import Image, ImageDraw, ImageFont
    image = Image.new("RGBA", (420, 112), (255, 255, 255, 0))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((2, 2, 418, 109), radius=16, fill=(16, 57, 108, 235))
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 53)
    except OSError:
        font = ImageFont.load_default()
    draw.text((23, 24), "HYBRID", font=font, fill="white")
    image.save(target)
    print("Logo: tijdelijk neutraal HYBRID-label (eigen logo nog niet geupload)", flush=True)


def make_site(error: str | None = None) -> bool:
    """Replace entire site; only maps from this execution will be included."""
    if SITE.exists():
        shutil.rmtree(SITE)
    SITE.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    metadata_file = MAPS / "forecast_metadata.json"
    metadata = {}
    if metadata_file.exists():
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))

    found = sorted(p for p in MAPS.glob("*.png") if p.is_file() and p.stat().st_size > 100)
    if error is None and not found:
        error = "Geen nieuwe PNG-kaarten gegenereerd"
    if error is None and not metadata:
        error = "Geen forecast-metadata (de run is niet volledig afgerond)"
    if error is None and int(metadata.get("map_count", -1)) != len(found):
        error = f"Aantal PNG-kaarten ({len(found)}) komt niet overeen met de modeluitvoer ({metadata.get('map_count')})"
    # GitHub Pages artifacts have practical size limits; keep the dashboard lean.
    if error is None and sum(p.stat().st_size for p in found) > 750_000_000:
        error = "De kaarten zijn samen groter dan 750 MB; verlaag HYBRID_DPI"

    good = error is None
    images = found if good else []
    if good:
        (SITE / "maps").mkdir()
        for p in images:
            shutil.copy2(p, SITE / "maps" / p.name)

    status = {
        "status": "ok" if good else "error",
        "updated_utc": now.isoformat(timespec="seconds"),
        "fresh_until_utc": (now + timedelta(hours=6)).isoformat(timespec="seconds"),
        "map_count": len(images),
        "forecast_start_utc": metadata.get("forecast_start_utc") if good else None,
        "forecast_end_utc": metadata.get("forecast_end_utc") if good else None,
        "model_runs_utc": metadata.get("model_runs_utc", {}) if good else {},
        "grid_step_deg": metadata.get("grid_step_deg") if good else None,
    }
    (SITE / "status.json").write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
    groups = {
        "Sneeuwval per dag": [],
        "Cumulatieve sneeuwval": [],
        "Nieuwe sneeuw op de grond": [],
        "Kans op sneeuw": [],
        "Modelonenigheid": [],
    }
    if good:
        for img in images:
            name = img.stem.lower()
            if "kans" in name:
                group = "Kans op sneeuw"
            elif "onenigheid" in name:
                group = "Modelonenigheid"
            elif "grond" in name:
                group = "Nieuwe sneeuw op de grond"
            elif "cumulatief" in name:
                group = "Cumulatieve sneeuwval"
            else:
                group = "Sneeuwval per dag"
            filename = escape(img.name, quote=True)
            label = escape(img.stem.replace("_", " "))
            groups[group].append(
                f'<article><a href="maps/{filename}" target="_blank" rel="noopener"><img loading="lazy" src="maps/{filename}?v={int(now.timestamp())}" alt="{label}"></a><p>{label}</p></article>'
            )

    cards = "".join(f'<section><h2>{escape(group)}</h2><div class="grid">{"".join(items)}</div></section>' for group, items in groups.items() if items)
    if not good:
        cards = '<div class="notice">De laatste modelberekening is mislukt. Er worden geen oude kaarten als nieuwe voorspelling getoond. Bekijk de foutdetails in GitHub Actions.</div>'
    used_runs = metadata.get("model_runs_utc", {}) if good else {}
    model_text = ", ".join(escape(str(name)) for name in used_runs) if used_runs else "geen nieuwe modeldata"
    produced = now.strftime("%d-%m-%Y %H:%M UTC")
    forecast_start = str(metadata.get("forecast_start_utc", ""))[:16].replace("T", " ") if good else ""
    error_label = "Voorspelling niet beschikbaar" if not good else "Nieuwe kaarten beschikbaar"

    html = f'''<!doctype html>
<html lang="nl">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate">
<title>Hybrid sneeuwkaarten Alpen</title>
<style>
:root {{ color-scheme:light; font-family:system-ui,-apple-system,Segoe UI,Arial,sans-serif; }}
body {{ margin:0; background:#f4f7fa; color:#12253d; }}
header {{ background:#10396c; color:white; padding:32px max(18px,calc((100vw - 1250px)/2)); }}
h1 {{ margin:0 0 8px; font-size:clamp(1.7rem,3vw,2.7rem); }}
header p {{ color:#d6e7f7; margin:5px 0; }}
main {{ max-width:1250px; margin:0 auto; padding:24px 18px 60px; }}
#status {{ padding:16px 20px; border:1px solid #aaccca; background:#e9f7f0; border-radius:10px; margin-bottom:22px; font-weight:600; }}
#status.bad,.notice {{ border-color:#ffc1b7; background:#fff0ec; color:#87291c; }}
.notice {{ padding:24px; border:1px solid; border-radius:10px; }}
h2 {{ margin:36px 0 14px; font-size:1.25rem; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(min(100%,380px),1fr)); gap:18px; }}
article {{ background:white; border:1px solid #dce4ed; border-radius:10px; overflow:hidden; box-shadow:0 1px 4px #00000010; }}
article img {{ display:block; width:100%; height:auto; }}
article p {{ margin:8px 14px 13px; color:#52647b; font-size:.8rem; overflow-wrap:anywhere; }}
footer {{ color:#627287; font-size:.85rem; max-width:1250px; margin:auto; padding:8px 18px 30px; }}
</style></head><body>
<header><h1>Hybride sneeuwverwachting Alpen</h1>
<p>AROME · AROME-AT · ICON-CH1 · ICON-CH2 · ICON-D2 · ECMWF</p>
<p>Verwerkt: {produced} · {len(images)} kaarten · modellen: {model_text}</p>
<p>{f'Prognose vanaf: {forecast_start} UTC' if forecast_start else ''}</p></header>
<main><div id="status" class="{'bad' if not good else ''}">{error_label}</div>{cards}</main>
<footer>Experimentele sneeuwkaarten, geen operationele waarschuwingen. Data: Meteo-France, GeoSphere Austria, MeteoSwiss, DWD en ECMWF. Controleer het actualiteitstijdstip.</footer>
<script>
const updated=Date.parse({json.dumps(now.isoformat())});
if(Date.now()-updated>6*60*60*1000){{
 const s=document.getElementById('status');
 s.textContent='Geen actuele voorspelling: laatste verwerking is ouder dan 6 uur.';
 s.classList.add('bad');
 document.querySelectorAll('main section').forEach(x=>x.hidden=true);
}}
</script></body></html>'''
    (SITE / "index.html").write_text(html, encoding="utf-8")
    print(f"Website: {'OK' if good else 'ERROR'}, {len(images)} maps, {sum(p.stat().st_size for p in images)/1e6:.1f} MB", flush=True)
    return good


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Only verify source files and import dependencies")
    options = parser.parse_args()
    missing = [name for name in SOURCES if not (ROOT / name).is_file()]
    if missing:
        raise FileNotFoundError("Ontbrekende Python-bestanden: " + ", ".join(missing))
    if options.check:
        import importlib
        import importlib.util
        for module_name in ("numpy", "matplotlib", "scipy", "requests", "xarray", "netCDF4", "eccodes", "cartopy", "PIL"):
            importlib.import_module(module_name)
            print(f"  Imported dependency: {module_name}")
        for path in sorted(ROOT.glob("SnowDays_*.py")):
            module_name = "check_" + path.stem.replace("-", "_")
            spec = importlib.util.spec_from_file_location(module_name, path)
            if spec is None or spec.loader is None:
                raise RuntimeError("Cannot import " + path.name)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            print(f"  Imported source: {path.name}")
        print("SOURCE AND DEPENDENCY CHECK OK (no weather data downloaded)")
        return

    if WORK.exists():
        shutil.rmtree(WORK)
    if SITE.exists():
        shutil.rmtree(SITE)
    MAPS.mkdir(parents=True)
    home = WORK / "home"
    tmp = WORK / "tmp"
    mpl = WORK / "matplotlib"
    for directory in (home, tmp, mpl):
        directory.mkdir(parents=True, exist_ok=True)

    try:
        ensure_logo(home)
        env = dict(os.environ)
        env.update({
            "HOME": str(home),
            "TMPDIR": str(tmp),
            "MPLCONFIGDIR": str(mpl),
            "XDG_CACHE_HOME": str(WORK / "cache"),
            "HYBRID_OUTPUT_DIR": str(MAPS),
            "PYTHONUNBUFFERED": "1",
            "MPLBACKEND": "Agg",
        })
        print("Modeldata: nieuwe downloads; geen vorige caches", flush=True)
        subprocess.run([sys.executable, str(ROOT / "SnowHybrid.py")], cwd=ROOT, env=env, check=True)
        make_site()
    except Exception as exc:
        print(f"FORECAST FAILED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        make_site(error=f"{type(exc).__name__}: {exc}")
    finally:
        shutil.rmtree(WORK, ignore_errors=True)
        print("Alle tijdelijke modeldownloads en kaart-werkbestanden verwijderd.", flush=True)


if __name__ == "__main__":
    main()
