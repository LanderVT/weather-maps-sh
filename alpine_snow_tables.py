"""Regional Alpine snowfall tables for the SneeuwhoogteNL GitHub Pages site.

The optional matplotlib capture hook records numerical snowfall fields *during*
existing map rendering. This avoids OCR / inferring values from map colours and
avoids downloading GRIB files a second time. Post-processing builds a standalone
JSON data file plus three HTML tables (1000, 1500, 2000 m).

Model fields and elevations must be aligned. If this cannot be demonstrated,
we publish missing data, never guessed numerical snowfall.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone, time as clocktime
from functools import lru_cache
from pathlib import Path
from zoneinfo import ZoneInfo
import argparse
import json
import math
import os
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
CACHE = Path(os.getenv("SNOW_REGION_CACHE", str(ROOT / "_regional_snow_snapshots")))
SITE = ROOT / "site"
LOCAL_TZ = ZoneInfo("Europe/Brussels")
ALTITUDES = (1000, 1500, 2000)
ALT_HALF_WIDTH = 100.0  # m
MIN_PIXELS = 15

# Approximate *mountain subregions*, intentionally not administrative borders.
# Geographic polygon coordinates in WGS84 lon/lat. They can be refined later
# without changing the capture / publishing infrastructure.
REGIONS = (
    ("n_fr", "Noord-Franse Alpen", ((5.60,45.30),(6.28,45.27),(7.11,45.75),(6.95,46.48),(6.22,46.50),(5.65,45.95))),
    ("s_fr", "Zuid-Franse Alpen", ((5.55,43.58),(6.50,43.60),(7.43,44.02),(7.32,45.35),(6.55,45.40),(5.50,44.65))),
    ("valais", "Wallis / Valais", ((6.75,45.86),(7.80,45.82),(8.55,46.23),(8.34,46.71),(7.14,46.59),(6.60,46.20))),
    ("bern", "Berner Oberland", ((7.00,46.35),(8.37,46.35),(8.55,46.85),(7.92,47.04),(7.12,46.96))),
    ("central_ch", "Centraal-Zwitserland", ((8.02,46.50),(9.12,46.46),(9.30,47.18),(8.63,47.30),(8.07,47.02))),
    ("graubunden", "Graubünden", ((8.77,46.17),(10.56,46.11),(10.70,46.87),(9.88,47.20),(8.91,47.10))),
    ("aosta", "Aostadal / Piemonte", ((6.65,44.71),(7.85,44.73),(8.43,45.55),(8.06,46.09),(7.00,46.01))),
    ("lombardy", "Lombardische Alpen", ((8.36,45.78),(9.25,45.76),(10.69,46.08),(10.45,46.60),(9.29,46.69),(8.49,46.27))),
    ("vorarlberg", "Vorarlberg / Arlberg", ((9.46,46.94),(10.54,46.81),(10.63,47.25),(10.10,47.64),(9.49,47.59))),
    ("n_tyrol", "Noord-Tirol", ((10.52,46.80),(12.12,46.81),(12.36,47.23),(11.75,47.63),(10.55,47.63))),
    ("e_tyrol", "Oost-Tirol / Karinthië", ((12.16,46.35),(14.48,46.30),(14.81,46.87),(13.96,47.24),(12.20,47.13))),
    ("salzburg", "Salzburgerland / Steiermark", ((12.55,47.08),(15.65,47.01),(15.82,47.72),(14.03,47.90),(12.67,47.66))),
    ("s_tyrol", "Zuid-Tirol / Ortler", ((10.22,46.18),(11.24,46.19),(11.26,46.91),(10.24,46.90))),
    ("dolomites", "Dolomieten", ((11.15,46.23),(12.64,46.23),(12.79,46.68),(12.12,46.96),(11.22,46.83))),
    ("julian", "Julische Alpen", ((13.28,46.10),(14.36,46.12),(14.30,46.72),(13.37,46.75))),
)
MODEL_LABELS = {"hybrid": "Hybrid", "icon_d2": "ICON-D2", "ecmwf": "ECMWF", "icon_ch2": "ICON-CH2"}
NAME_PATTERN = re.compile(
    r"_(?:alps)_(\d{8})_(\d{2})z_f(\d{3})-(\d{3})", re.I
)
DEM_PATTERN = re.compile(
    r"dem_z\d+_([-\d.]+)_([-\d.]+)_([-\d.]+)_([-\d.]+)_([-\d.]+)\.npy$"
)


def point_inside_polygon(x: np.ndarray, y: np.ndarray, poly: tuple) -> np.ndarray:
    """Vectorized even/odd test. Includes neither a large shapely dependency nor a CRS change."""
    inside = np.zeros(np.broadcast_shapes(x.shape, y.shape), dtype=bool)
    x0, y0 = poly[-1]
    for x1, y1 in poly:
        if y0 != y1:
            cross = ((y0 > y) != (y1 > y)) & (
                x < (x1 - x0) * (y - y0) / (y1 - y0) + x0
            )
            inside ^= cross
        x0, y0 = x1, y1
    return inside


def _coordinates(x, y, shape):
    ny, nx = shape
    x, y = np.asarray(x), np.asarray(y)
    if x.ndim == 1 and y.ndim == 1:
        if x.size == nx + 1: x = (x[:-1] + x[1:]) / 2
        if y.size == ny + 1: y = (y[:-1] + y[1:]) / 2
        if x.size != nx or y.size != ny: return None
        xx, yy = np.broadcast_arrays(x[None, :], y[:, None])
    elif x.ndim == 2 and y.ndim == 2:
        if x.shape == (ny + 1, nx + 1) and y.shape == x.shape:
            x = (x[:-1, :-1] + x[1:, 1:]) / 2
            y = (y[:-1, :-1] + y[1:, 1:]) / 2
        if x.shape != shape or y.shape != shape: return None
        xx, yy = x, y
    else:
        return None
    if not (np.isfinite(xx).all() and np.isfinite(yy).all()): return None
    if min(np.min(xx), np.min(yy)) < -180 or max(np.max(xx), np.max(yy)) > 180:
        return None
    if not (4.0 < np.nanmean(xx) < 18.0 and 43.0 < np.nanmean(yy) < 50.0):
        return None
    return xx, yy


def _dem_candidates(home: Path):
    # The science scripts use ~/Documents/SneeuwhoogteNL/data/dem_*.npy.
    root = home / "Documents" / "SneeuwhoogteNL" / "data"
    if not root.is_dir():
        return []
    return list(root.glob("dem*.npy")) + list(root.glob("**/dem*.npy"))


@lru_cache(maxsize=16)
def _load_dem(path: str):
    return np.load(path, mmap_mode="r", allow_pickle=False)


def find_matching_dem(shape, xx, yy):
    """Only accept a DEM whose shape AND bounds agree with the map grid."""
    candidates = []
    for path in _dem_candidates(Path.home()):
        match = DEM_PATTERN.search(path.name)
        if not match:
            continue
        step, west, east, south, north = (float(k) for k in match.groups())
        # Avoid loading the wrong regional DEM even if dimensions match.
        if not (abs(float(np.min(xx)) - west) < step * 2.5 and
                abs(float(np.max(xx)) - east) < step * 2.5 and
                abs(float(np.min(yy)) - south) < step * 2.5 and
                abs(float(np.max(yy)) - north) < step * 2.5):
            continue
        candidates.append(path)
    for path in candidates:
        try:
            dem = _load_dem(str(path))
            if dem.shape == shape:
                return dem
        except (OSError, ValueError):
            continue
    return None


def _model_from_filename(name: str) -> str | None:
    lower = name.lower()
    for prefix, model in (("hybrid_", "hybrid"), ("ecmwf_", "ecmwf"),
                          ("icon_d2_", "icon_d2"), ("icon_ch2_", "icon_ch2")):
        if lower.startswith(prefix):
            return model
    return None


def select_snow_layer(fig):
    """Pick the numerical snowfall contour/pixel mesh, not the relief image."""
    candidates = getattr(fig, "_snow_region_layers", ())
    best = None
    best_score = -1e9
    for layer in candidates:
        values = layer["z"]
        if values.ndim != 2 or min(values.shape) < 80:
            continue
        coordinates = _coordinates(layer["x"], layer["y"], values.shape)
        if coordinates is None:
            continue
        xx, yy = coordinates
        # Negative relief/shading values are not snowfall.
        numeric = np.asarray(np.ma.filled(values, 0), dtype=np.float32)
        if not np.isfinite(numeric).any():
            continue
        level_array = layer.get("levels")
        bonus = 0
        if level_array is not None:
            try:
                levels = np.asarray(level_array, dtype=float)
                if levels.size >= 8 and levels.min() <= 1 and levels.max() >= 100:
                    bonus = 100
            except (TypeError, ValueError):
                pass
        score = bonus + np.log1p(values.size) + float(layer.get("zorder", 0)) / 100
        if score > best_score:
            best = (numeric, xx, yy)
            best_score = score
    return best


def capture_figure(fig, output):
    """Called *before* saving, where the original model arrays still exist."""
    path = Path(str(output))
    name = path.name.lower()
    if (not name.endswith(".png") or "_sneeuwval" not in name or
            "_alps_" not in name or "cumulatief" in name):
        return False
    model = _model_from_filename(name)
    match = NAME_PATTERN.search(name)
    if model is None or match is None:
        print("Alpen tabel: onherkend bestandsformaat:", path.name, flush=True)
        return False
    selected = select_snow_layer(fig)
    if selected is None:
        print("Alpen tabel: geen bruikbaar numeriek sneeuwveld:", path.name, flush=True)
        return False
    data, xx, yy = selected
    dem = find_matching_dem(data.shape, xx, yy)
    if dem is None:
        print("Alpen tabel: geen passende DEM gevonden:", path.name, flush=True)
        return False
    values = np.where(np.isfinite(data), np.maximum(data, 0), 0)
    result = {}
    for region_id, _label, polygon in REGIONS:
        # Limit computation to each region's geographic bounding box first.
        west, east = min(p[0] for p in polygon), max(p[0] for p in polygon)
        south, north = min(p[1] for p in polygon), max(p[1] for p in polygon)
        coarse = ((xx >= west) & (xx <= east) & (yy >= south) & (yy <= north))
        if not np.any(coarse):
            continue
        index = np.where(coarse)
        ri, ci = index[0], index[1]
        submask = point_inside_polygon(xx[ri, ci], yy[ri, ci], polygon)
        region_values = {}
        for alt in ALTITUDES:
            mask = (submask & np.isfinite(dem[ri, ci]) &
                    (np.abs(dem[ri, ci] - alt) <= ALT_HALF_WIDTH))
            if int(np.count_nonzero(mask)) < MIN_PIXELS:
                continue
            selected_values = values[ri[mask], ci[mask]]
            q25, q50, q75 = np.percentile(selected_values, (25, 50, 75))
            region_values[str(alt)] = {
                "q25": round(float(q25), 2),
                "median": round(float(q50), 2),
                "q75": round(float(q75), 2),
                "pixels": int(selected_values.size),
            }
        if region_values:
            result[region_id] = region_values
    ymd, hh, lead_a, lead_b = match.groups()
    run = datetime.strptime(ymd + hh, "%Y%m%d%H").replace(tzinfo=timezone.utc)
    start = run + timedelta(hours=int(lead_a))
    end = run + timedelta(hours=int(lead_b))
    if end <= start:
        return False
    payload = {
        "model": model,
        "run_utc": run.isoformat(),
        "start_utc": start.isoformat(),
        "end_utc": end.isoformat(),
        "file": path.name,
        "source": "numerical snowfall field before PNG rendering",
        "statistics": result,
    }
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / (path.stem + ".json")).write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"Alpen tabel: {model}, {len(result)} regio's uit {path.name}", flush=True)
    return True


def install_matplotlib_capture():
    """Non-invasive, process-local capture of already computed numerical maps.

    This hook never changes what the science model plots. If extraction fails,
    it logs the problem and leaves normal map generation untouched.
    """
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    if getattr(Figure.savefig, "_snow_tables_capture", False):
        return

    original_contourf = Axes.contourf
    original_pcolormesh = Axes.pcolormesh
    original_savefig = Figure.savefig

    def remember(self, kind, args, kwargs):
        try:
            if len(args) < 3:
                return
            x, y, z = args[:3]
            z = np.ma.asarray(z)
            if z.ndim != 2 or min(z.shape) < 80:
                return
            entry = {
                "kind": kind,
                "x": np.asarray(x),
                "y": np.asarray(y),
                "z": z,
                "levels": kwargs.get("levels", args[3] if kind == "contourf" and len(args) > 3 else None),
                "zorder": kwargs.get("zorder", 0),
            }
            layers = getattr(self.figure, "_snow_region_layers", None)
            if layers is None:
                layers = []
                setattr(self.figure, "_snow_region_layers", layers)
            layers.append(entry)
        except Exception:
            pass

    def contourf(self, *args, **kwargs):
        remember(self, "contourf", args, kwargs)
        return original_contourf(self, *args, **kwargs)

    def pcolormesh(self, *args, **kwargs):
        remember(self, "pcolormesh", args, kwargs)
        return original_pcolormesh(self, *args, **kwargs)

    def savefig(self, fname, *args, **kwargs):
        try:
            capture_figure(self, fname)
        except Exception as exc:
            print(f"::warning::Alpen tabel capture overgeslagen: {exc}", flush=True)
        return original_savefig(self, fname, *args, **kwargs)

    setattr(savefig, "_snow_tables_capture", True)
    Axes.contourf = contourf
    Axes.pcolormesh = pcolormesh
    Figure.savefig = savefig


def _local_hours_overlap(start, end, day):
    local_start = datetime.combine(day, clocktime.min, tzinfo=LOCAL_TZ)
    local_end = datetime.combine(day + timedelta(days=1), clocktime.min, tzinfo=LOCAL_TZ)
    return max(0.0, (min(end, local_end.astimezone(timezone.utc)) -
                     max(start, local_start.astimezone(timezone.utc))).total_seconds() / 3600)


def model_daily_stats(records, date, region_id, altitude):
    """Time-weighted q25/q75 within a local day, using source plot intervals.

    When an original plot covers multiple local dates, snowfall is allocated
    proportionally to interval overlap (explicit approximation).
    """
    lo = hi = coverage = 0.0
    intervals = []
    for row in records:
        stats = row.get("statistics", {}).get(region_id, {}).get(str(altitude))
        if not stats:
            continue
        start = datetime.fromisoformat(row["start_utc"])
        end = datetime.fromisoformat(row["end_utc"])
        length_h = (end - start).total_seconds() / 3600
        if length_h <= 0:
            continue
        hours = _local_hours_overlap(start, end, date)
        if hours <= 0:
            continue
        frac = hours / length_h
        lo += stats["q25"] * frac
        hi += stats["q75"] * frac
        coverage += hours
        intervals.append(row["file"])
    if coverage < 8 or coverage > 26:
        return None
    return {"q25": lo, "q75": hi, "coverage_hours": round(coverage, 1), "files": intervals}


def _range_text(lo, hi):
    if lo is None or hi is None:
        return "–"
    if hi < 1:
        return "0"
    step = 5 if hi <= 50 else 10
    a = max(0, math.floor(max(lo, 0) / step) * step)
    b = max(step, math.ceil(max(hi, 0) / step) * step)
    if a >= b:
        b = a + step
    return f"{a}–{b}"


def build_summary(now_utc: datetime | None = None):
    now_utc = now_utc or datetime.now(timezone.utc)
    today = now_utc.astimezone(LOCAL_TZ).date()
    dates = [today + timedelta(days=i) for i in range(1, 6)]
    by_model = {name: [] for name in MODEL_LABELS}
    for file in sorted(CACHE.glob("*.json")) if CACHE.exists() else ():
        try:
            row = json.loads(file.read_text(encoding="utf-8"))
            model = row.get("model")
            if model in by_model:
                by_model[model].append(row)
        except (OSError, ValueError):
            continue
    # Prevent accidental mixing of two runs of the same model.
    for model, rows in by_model.items():
        if rows:
            recent = max(x["run_utc"] for x in rows)
            by_model[model] = [x for x in rows if x["run_utc"] == recent]
    summary = {
        "generated_at_utc": now_utc.isoformat(),
        "dates": [x.isoformat() for x in dates],
        "altitudes_m": list(ALTITUDES),
        "regions": [{"id": id, "name": label} for id, label, _ in REGIONS],
        "method": (
            "Ruimtelijke 25e–75e percentielen binnen +/-100 m van de aangegeven hoogte. "
            "Beschikbare modelranges worden gecombineerd zonder Hybrid dubbel te wegen. "
            "Bij grensoverschrijdende 24-uursplots wordt sneeuw naar overlapuren verdeeld; "
            "dit is een benadering, geen probabilistisch betrouwbaarheidsinterval. "
            "Regiogrenzen zijn benaderde Alpenpolygonen."
        ),
        "models_found": {m: len(v) for m, v in by_model.items()},
        "data": {},
    }
    for region_id, _label, _poly in REGIONS:
        summary["data"][region_id] = {}
        for alt in ALTITUDES:
            cells = []
            for index, day in enumerate(dates):
                # First 48 h: include Hybrid + ICON-D2 where available; ECMWF
                # and ICON-CH2 fill any missing lead times. Later: global
                # ECMWF and Swiss ICON-CH2 only.
                preferred = ("hybrid", "icon_d2", "ecmwf", "icon_ch2") if index < 2 else ("ecmwf", "icon_ch2")
                available = []
                for model in preferred:
                    entry = model_daily_stats(by_model[model], day, region_id, alt)
                    if entry:
                        available.append((model, entry))
                if not available:
                    cells.append({"range_cm": "–", "models": [], "partial": False})
                    continue
                lower = min(x["q25"] for _, x in available)
                upper = max(x["q75"] for _, x in available)
                cells.append({
                    "range_cm": _range_text(lower, upper),
                    "low_cm": round(lower, 1),
                    "high_cm": round(upper, 1),
                    "models": [m for m, _ in available],
                    "partial": all(x["coverage_hours"] < 21 for _, x in available),
                })
            summary["data"][region_id][str(alt)] = cells
    return summary


HTML_STYLES = r"""
<style>
#alpine-snow-tables { color:#e9eef5; margin:24px auto; max-width:1380px; padding:0 16px; font-family:system-ui,-apple-system,'Segoe UI',sans-serif; }
#alpine-snow-tables .as-panel { background:#172232; border:1px solid #35445c; border-radius:16px; padding:20px; box-shadow:0 12px 32px rgba(0,0,0,.16); }
#alpine-snow-tables h2 {font-size:clamp(1.24rem,3vw,1.8rem); margin:0 0 8px; color:#fff; letter-spacing:-.02em}
#alpine-snow-tables p {line-height:1.5;margin:5px 0 14px;color:#bdc9db;font-size:.91rem}
#alpine-snow-tables .as-alt {font-size:1.08rem; margin:22px 0 10px;font-weight:700;color:#f3f7ff}
#alpine-snow-tables .as-scroll {width:100%;overflow:auto;scrollbar-color:#526584 #152030;border-radius:10px;border:1px solid #35445c}
#alpine-snow-tables table {border-collapse:separate;border-spacing:0;min-width:780px;width:100%;font-size:.87rem;font-variant-numeric:tabular-nums}
#alpine-snow-tables th,#alpine-snow-tables td {padding:9px 11px;border-bottom:1px solid #2c3a50;white-space:nowrap;text-align:center}
#alpine-snow-tables th {background:#243248;position:sticky;top:0;color:#e6eefc;font-weight:700}
#alpine-snow-tables th:first-child,#alpine-snow-tables td:first-child{text-align:left;position:sticky;left:0;z-index:1;background:#1a293c;min-width:185px}
#alpine-snow-tables th:first-child {background:#243248;z-index:2}
#alpine-snow-tables tr:last-child td {border-bottom:0}
#alpine-snow-tables tbody tr:hover td {background:#263b54}
#alpine-snow-tables td.as-none {color:#8391a6}
#alpine-snow-tables td.as-light {color:#c6ddf3}
#alpine-snow-tables td.as-medium {color:#b9e7e0;font-weight:650}
#alpine-snow-tables td.as-heavy {color:#ffdf91;font-weight:750}
#alpine-snow-tables .as-foot {font-size:.78rem;color:#a7b6c9;margin-top:14px}
#alpine-snow-tables .as-badge {display:inline-block;border:1px solid #49617d;border-radius:7px;color:#c9daea;padding:3px 7px;font-size:.72rem;margin-left:5px;vertical-align:middle}
</style>
"""
HTML_SECTION = r"""
<section id="alpine-snow-tables" aria-label="Sneeuwverwachting per Alpenregio">
 <div class="as-panel">
  <h2>❄️ Sneeuwverwachting per Alpenregio <span class="as-badge">5 dagen</span></h2>
  <p>Verwachte <b>nieuwe sneeuwval</b> op 1000, 1500 en 2000 meter. De getallen zijn regionale ranges (cm), geen sneeuwhoogte op de grond.</p>
  <div id="as-dynamic"><p>Regionale voorspellingen worden geladen…</p></div>
  <p class="as-foot">Ranges zijn gebaseerd op de ruimtelijke 25e–75e percentielen per hoogteband (±100 m), met verschillen tussen beschikbare modellen. Geen betrouwbaarheidsinterval. De regiogrenzen zijn indicatief. Bij ‘–’ zijn er onvoldoende passende modelgegevens.</p>
 </div>
</section>
"""
JS_TABLE = r"""(() => {
  const target = document.querySelector('#as-dynamic');
  if (!target) return;
  const esc = x => String(x).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const dates = new Intl.DateTimeFormat('nl-BE',{weekday:'short',day:'2-digit',month:'2-digit',timeZone:'Europe/Brussels'});
  function fill(data){
    const headings = data.dates.map(x => dates.format(new Date(x+'T12:00:00Z')));
    const rows = data.regions || [];
    const keys = {'hybrid':'Hybrid','icon_d2':'ICON-D2','ecmwf':'ECMWF','icon_ch2':'ICON-CH2'};
    const markup = (data.altitudes_m||[]).map(alt => {
      const body = rows.map(reg => {
        const cells = ((data.data||{})[reg.id]||{})[String(alt)]||[];
        return '<tr><td>'+esc(reg.name)+'</td>'+
          headings.map((_,i) => {
            const c = cells[i] || {range_cm:'–',models:[]};
            const text = c.range_cm || '–';
            const strength = c.high_cm >= 30 ? 'as-heavy' : c.high_cm >= 10 ? 'as-medium' : text === '–' ? 'as-none' : 'as-light';
            const source = (c.models||[]).map(k => keys[k]||k).join(', ');
            const title = source ? 'Modellen: '+source + (c.partial ? ' · gedeeltelijke dag' : '') : 'Geen gegevens';
            return '<td class="'+strength+'" title="'+esc(title)+'">'+esc(text)+'</td>';
          }).join('')+'</tr>';
      }).join('');
      return '<h3 class="as-alt">'+alt+' m hoogte · cm per dag</h3><div class="as-scroll">'+
       '<table><thead><tr><th scope="col">Regio</th>'+headings.map(h => '<th scope="col">'+esc(h)+'</th>').join('')+
       '</tr></thead><tbody>'+body+'</tbody></table></div>';
    }).join('');
    const last = data.generated_at_utc ? new Date(data.generated_at_utc).toLocaleString('nl-BE',{
      day:'2-digit',month:'2-digit',year:'numeric',hour:'2-digit',minute:'2-digit',timeZone:'Europe/Brussels'
    }) : '';
    const available = Object.entries(data.models_found||{}).filter(([_,n]) => n>0).map(([k,_]) => keys[k]||k).join(' · ');
    const info = '<p style=\"margin-bottom:10px\">Bijgewerkt: '+esc(last)+
      ' · Modelvelden: '+esc(available||'geen beschikbaar')+'</p>';
    target.innerHTML = info + (markup || '<p>Geen regionale gegevens beschikbaar.</p>');
  }
  fetch('regional_snowfall.json', {cache:'no-store'}).then(r=>{if(!r.ok)throw Error('Geen JSON');return r.json()})
    .then(fill).catch(()=>target.innerHTML='<p>Regionale sneeuwverwachting tijdelijk niet beschikbaar. De sneeuwkaarten blijven zichtbaar.</p>');
})();
"""


def publish_site(summary, site=SITE):
    site.mkdir(parents=True, exist_ok=True)
    (site / "regional_snowfall.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (site / "alpine_snow_tables.js").write_text(JS_TABLE, encoding="utf-8")
    index = site / "index.html"
    if not index.exists():
        raise FileNotFoundError(f"Websitebestand ontbreekt: {index}")
    page = index.read_text(encoding="utf-8")
    if 'id="alpine-snow-tables"' not in page:
        block = HTML_STYLES + HTML_SECTION + '<script src="alpine_snow_tables.js" defer></script>\n'
        if "</main>" in page:
            page = page.replace("</main>", block + "</main>", 1)
        elif "</body>" in page:
            page = page.replace("</body>", block + "</body>", 1)
        else:
            page += block
        index.write_text(page, encoding="utf-8")
    return len(summary["regions"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build", "self-test"))
    args = parser.parse_args()
    if args.command == "self-test":
        self_test()
        return
    summary = build_summary()
    nr = publish_site(summary)
    if not any(summary["models_found"].values()):
        print("::warning::Geen Alpen-sneeuwvelden onderschept; tabellen blijven leeg (geen schattingen)")
    print(f"Alpen-tabellen: {nr} regio's x 3 hoogtes x 5 dagen, "
          f"captures: {summary['models_found']}")


def self_test():
    from tempfile import TemporaryDirectory
    from unittest.mock import patch
    lon = np.linspace(6.0, 15.0, 1000)
    lat = np.linspace(44.0, 49.0, 500)
    xx, yy = np.broadcast_arrays(lon[None,:], lat[:,None])
    assert point_inside_polygon(np.array([7.5]), np.array([46.2]), REGIONS[2][2])[0]
    assert not point_inside_polygon(np.array([13.5]), np.array([46.2]), REGIONS[2][2])[0]
    assert _coordinates(lon, lat, (500,1000)) is not None
    assert _coordinates(lon[:-10], lat, (500,1000)) is None
    assert _range_text(11,19)=="10–20"
    assert _range_text(.2,.5)=="0"
    assert _range_text(0,3)=="0–5"
    with TemporaryDirectory() as td:
        root = Path(td)
        fake = root / 'site'; fake.mkdir()
        (fake / 'index.html').write_text('<html><main><p>Maps</p></main></html>')
        sample = {'regions':[], 'data':{}, 'dates':[], 'altitudes_m':[]}
        publish_site(sample, fake)
        page = (fake/'index.html').read_text()
        assert 'alpine-snow-tables' in page and 'Maps' in page
        assert (fake/'regional_snowfall.json').is_file()
    print('Alpen-tabellen self-test OK')


if __name__ == "__main__":
    main()
