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
from html import escape
from pathlib import Path
from zoneinfo import ZoneInfo
import argparse
import inspect
import functools
import types
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
DEM_REGISTRY = []  # DEMs already loaded by the original model process.

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
MODEL_LABELS = {"hybrid": "Hybrid", "icon_d2": "ICON-D2", "ecmwf": "ECMWF"}
MODEL_MAX_DISPLAY_DAYS = {"hybrid": 2, "icon_d2": 2, "ecmwf": 7}
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


def register_dem(x, y, dem):
    """Register the *actual* DEM used by the model (avoids filename/path guesses)."""
    try:
        elev = np.asanyarray(dem)
        if elev.ndim != 2:
            return False
        coords = _coordinates(x, y, elev.shape)
        if coords is None:
            return False
        xx, yy = coords
        # Two DEM grids are sufficient because each model runs separately.
        DEM_REGISTRY[:] = [(elev, xx, yy)] + DEM_REGISTRY[:1]
        print(f"Alpen tabel: DEM geregistreerd {elev.shape}", flush=True)
        return True
    except Exception as exc:
        print(f"::warning::DEM registratie overgeslagen: {exc}", flush=True)
        return False


def find_matching_dem(shape, xx, yy):
    """Prefer the live model DEM; then fall back to the on-disk DEM cache."""
    for dem, dem_xx, dem_yy in DEM_REGISTRY:
        if dem.shape != shape:
            continue
        if (np.allclose([float(np.min(xx)), float(np.max(xx)), float(np.min(yy)), float(np.max(yy))],
                        [float(np.min(dem_xx)), float(np.max(dem_xx)), float(np.min(dem_yy)), float(np.max(dem_yy))],
                        rtol=0, atol=0.02)):
            return dem
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
                          ("icon_d2_", "icon_d2")):
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
        numeric = np.asarray(np.ma.filled(values, np.nan), dtype=np.float32)
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
    # Direct plot_snowfall hook can run before matplotlib's savefig hook;
    # never recapture or overwrite the same model interval.
    if (CACHE / (path.stem + ".json")).exists():
        return True
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
    # Missing model cells are not zero snowfall: keep their mask.
    valid_data = np.isfinite(data)
    values = np.where(valid_data, np.maximum(data, 0), np.nan)
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
                    valid_data[ri,ci] &
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
    if not result:
        print(f'::warning::Alpen tabel: 0 hoogtezones in {path.name}; controleer DEM/gebied', flush=True)
    return True


def capture_numeric_field(data, x, y, area, run, start, end, model):
    """Capture a snowfall map directly BEFORE Matplotlib masks / plots it.

    This is the reliable source of numbers. It does not modify the source data
    or the original plotting function in any way.
    """
    if str(area).lower() != 'alps' or model not in MODEL_LABELS:
        return False
    try:
        if not isinstance(run, datetime):
            return False
        if run.tzinfo is None:
            run = run.replace(tzinfo=timezone.utc)
        run = run.astimezone(timezone.utc)
        start = int(start)
        end = int(end)
        if end <= start:
            return False
        data = np.ma.asarray(data)
        if data.ndim != 2:
            return False
        # Give capture_figure a single exact numeric layer.  This retains
        # the existing polygon / altitude / percentile logic and file format.
        fake_fig = types.SimpleNamespace(_snow_region_layers=[{
            'z': data, 'x': np.asarray(x), 'y': np.asarray(y),
            'levels': None, 'zorder': 100,
        }])
        filename = (
            f'{model}_sneeuwval_alps_{run:%Y%m%d_%H}z_'
            f'f{start:03d}-{end:03d}.png'
        )
        return capture_figure(fake_fig, filename)
    except Exception as exc:
        print(f'::warning::Alpen tabel direct capture: {exc}', flush=True)
        return False


def install_module_capture(namespace):
    """Instrument model-local load_dem() and plot_snowfall() functions.

    Called after model functions are defined and before main() starts.
    Works even if a model uses a custom plotting backend / Figure.savefig.
    A figure-level fallback is installed for nonstandard models.
    """
    module = str(namespace.get('__file__', 'unknown')).split('/')[-1]
    raw_label = str(namespace.get('FILE_PREFIX') or namespace.get('MODEL_LABEL') or module).lower()
    if 'hybrid' in raw_label:
        model = 'hybrid'
    elif 'ecmwf' in raw_label:
        model = 'ecmwf'
    elif 'ch2' in raw_label:
        model = None  # ICON-CH2 is disabled
    elif 'd2' in raw_label:
        model = 'icon_d2'
    else:
        model = None

    dem_func = namespace.get('load_dem')
    if callable(dem_func) and not getattr(dem_func, '_snow_v2_dem', False):
        @functools.wraps(dem_func)
        def dem_wrapper(*args, **kwargs):
            result = dem_func(*args, **kwargs)
            if isinstance(result, (tuple, list)) and len(result) >= 3:
                register_dem(result[0], result[1], result[2])
            return result
        dem_wrapper._snow_v2_dem = True
        namespace['load_dem'] = dem_wrapper
        print(f'Alpen tabel: {module} DEM-hook actief', flush=True)

    # ICON-D2 draws its snowline using the model surface height (HSURF).
    # That field is the terrain on the same native x/y grid as snowfall.
    # Capture it before calculating regional height bands; unlike ECMWF,
    # SnowAll_ICON-D2.py has no load_dem() function.
    alt_func = namespace.get('plot_altitude')
    if model == 'icon_d2' and callable(alt_func) and not getattr(alt_func, '_snow_v3_alt', False):
        alt_signature = inspect.signature(alt_func)
        @functools.wraps(alt_func)
        def altitude_wrapper(*args, **kwargs):
            try:
                bound = alt_signature.bind_partial(*args, **kwargs)
                bound.apply_defaults()
                a = dict(bound.arguments)
                # Dark/slate adapter may replace plot_altitude with *args/**kwargs.
                # Positional layout is defined by SnowAll_ICON-D2.py.
                if 'x' not in a or 'y' not in a or 'kind' not in a:
                    fields = ('field','x','y','extent','area','run','step','kind','hsurf')
                    a = dict(zip(fields,args))
                    a.update(kwargs)
                if a.get('kind') == 'snowlmt' and a.get('hsurf') is not None:
                    register_dem(a['x'], a['y'], a['hsurf'])
            except Exception as exc:
                print(f'::warning::ICON-D2 HSURF registratie: {exc}', flush=True)
            return alt_func(*args, **kwargs)
        altitude_wrapper._snow_v3_alt = True
        namespace['plot_altitude'] = altitude_wrapper
        print('Alpen tabel: ICON-D2 HSURF hoogte-hook actief', flush=True)

    plot_func = namespace.get('plot_snowfall')
    if callable(plot_func) and model and not getattr(plot_func, '_snow_v3_plot', False):
        signature = inspect.signature(plot_func)
        @functools.wraps(plot_func)
        def plot_wrapper(*args, **kwargs):
            try:
                bound = signature.bind_partial(*args, **kwargs)
                bound.apply_defaults()
                p = dict(bound.arguments)
                # A style adapter may expose only (*args, **kwargs). Extract
                # original positional parameters from the verified source APIs.
                needed = {'data','x','y','area','run','start','end'}
                if not needed.issubset(p):
                    if model == 'icon_d2':
                        fields = ('data','x','y','extent','area','run','name','start','end')
                    else:
                        fields = ('data','x','y','extent','area','run','start','end','rain','total')
                    p = dict(zip(fields,args))
                    p.update(kwargs)
                if needed.issubset(p):
                    # Exclude ground-snow and cumulative plots.
                    is_ground = model != 'icon_d2' and p.get('rain', None) is None
                    cumulative_d2 = (model == 'icon_d2' and
                                     str(p.get('name','')).lower().startswith('0-48'))
                    if not is_ground and not p.get('total', False) and not cumulative_d2:
                        capture_numeric_field(p['data'],p['x'],p['y'],
                            p['area'],p['run'],p['start'],p['end'],model)
                else:
                    print(f'::warning::Alpen tabel {model}: onbekende plot-signature', flush=True)
            except Exception as exc:
                print(f'::warning::Alpen tabel plot wrapper: {exc}', flush=True)
            return plot_func(*args, **kwargs)
        plot_wrapper._snow_v3_plot = True
        namespace['plot_snowfall'] = plot_wrapper
        print(f'Alpen tabel: {module} directe sneeuwval-hook actief ({model})', flush=True)

    # Catch plots made by models without a matching plot_snowfall function.
    install_matplotlib_capture()


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
    # Partial forecast days are allowed. Mark their actual hourly coverage.
    if coverage < 3 or coverage > 26:
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
    """Build distinct per-model tables for their available forecast dates.

    Hybrid and ICON-D2 show at most two calendar dates; ECMWF up to seven.
    The date window starts today, and empty future days are never fabricated.
    ICON-CH2 is never included.
    """
    now_utc = now_utc or datetime.now(timezone.utc)
    today = now_utc.astimezone(LOCAL_TZ).date()
    # Seven local calendar dates, INCLUDING today. Previous versions silently
    # discarded today's forecasts and capped ECMWF at five days.
    dates = [today + timedelta(days=i) for i in range(7)]
    by_model = {name: [] for name in MODEL_LABELS}
    for file in sorted(CACHE.glob("*.json")) if CACHE.exists() else ():
        try:
            row = json.loads(file.read_text(encoding="utf-8"))
            model = row.get("model")
            if model in by_model:
                by_model[model].append(row)
        except (OSError, ValueError):
            continue
    # Do not combine two forecast initialization times for one model.
    for model, rows in by_model.items():
        if rows:
            latest = max(x["run_utc"] for x in rows)
            by_model[model] = [x for x in rows if x["run_utc"] == latest]
    summary = {
        "generated_at_utc": now_utc.isoformat(),
        "dates": [x.isoformat() for x in dates],
        "altitudes_m": list(ALTITUDES),
        "regions": [{"id": rid, "name": label} for rid, label, _ in REGIONS],
        "model_labels": MODEL_LABELS,
        "model_runs": {m: (v[0]["run_utc"] if v else None) for m, v in by_model.items()},
        "models_found": {m: len(v) for m, v in by_model.items()},
        "method": (
            "Afzonderlijke modellen (Hybrid, ICON-D2 en ECMWF); geen menging. "
            "Ruimtelijke 25e–75e percentielen binnen +/-100 m van de aangegeven hoogte. "
            "Dagtotalen zijn benaderingen indien modelkaart-intervallen niet met kalenderdagen "
            "samenvallen. Gedeeltelijke dagen zijn als zodanig aangeduid; geen volledige 24u-som. "
            "Regiogrenzen zijn benaderde Alpenpolygonen."
        ),
        "data": {},
        "available_day_indices": {},
    }
    for model in MODEL_LABELS:
        model_data = {}
        day_available = set()
        for region_id, _label, _poly in REGIONS:
            model_data[region_id] = {}
            for alt in ALTITUDES:
                cells = []
                for i, day in enumerate(dates):
                    val = model_daily_stats(by_model[model], day, region_id, alt)
                    if val is None:
                        cells.append({"range_cm": "–", "available": False})
                        continue
                    day_available.add(i)
                    cells.append({
                        "range_cm": _range_text(val["q25"], val["q75"]),
                        "low_cm": round(float(val["q25"]), 1),
                        "high_cm": round(float(val["q75"]), 1),
                        "coverage_hours": val["coverage_hours"],
                        "partial": val["coverage_hours"] < 23.5,
                        "available": True,
                    })
                model_data[region_id][str(alt)] = cells
        summary["data"][model] = model_data
        # Only display the first two available calendar dates for the
        # short-range models, and at most seven for ECMWF. This is a display
        # constraint, not a fabricated extension of the model forecast.
        indices = sorted(day_available)[:MODEL_MAX_DISPLAY_DAYS[model]]
        summary["available_day_indices"][model] = indices
    return summary


HTML_STYLES = r"""
<style>
#alpine-snow-tables { color:#e9eef5; margin:24px auto; max-width:1380px; padding:0 16px; font-family:system-ui,-apple-system,'Segoe UI',sans-serif; }
#alpine-snow-tables .as-panel {background:#172232;border:1px solid #35445c;border-radius:16px;padding:20px;box-shadow:0 12px 32px rgba(0,0,0,.16)}
#alpine-snow-tables h2 {font-size:clamp(1.25rem,3vw,1.8rem);margin:0 0 8px;color:#fff;letter-spacing:-.02em}
#alpine-snow-tables p {line-height:1.5;margin:5px 0 14px;color:#bdc9db;font-size:.91rem}
#alpine-snow-tables .as-tabs {display:flex;flex-wrap:wrap;gap:9px;margin:12px 0 6px}
#alpine-snow-tables .as-tab {background:#243248;color:#e5eef9;border:1px solid #566782;padding:9px 18px;border-radius:10px;font-size:.95rem;cursor:pointer}
#alpine-snow-tables .as-tab:hover {background:#30445f}
#alpine-snow-tables .as-tab[aria-selected="true"] {background:#e3edf8;color:#182639;font-weight:750}
#alpine-snow-tables .as-alt {font-size:1.08rem;margin:22px 0 10px;font-weight:700;color:#f3f7ff}
#alpine-snow-tables .as-scroll {width:100%;overflow:auto;scrollbar-color:#526584 #152030;border-radius:10px;border:1px solid #35445c}
#alpine-snow-tables table {border-collapse:separate;border-spacing:0;min-width:600px;width:100%;font-size:.87rem;font-variant-numeric:tabular-nums}
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
/* Native model selection: no JavaScript, no fetch, no stale JS cache. */
#alpine-snow-tables .as-switcher {display:flex;flex-wrap:wrap;gap:9px;margin-top:14px;align-items:center}
#alpine-snow-tables .as-model-radio {position:absolute;width:1px;height:1px;opacity:0;overflow:hidden;clip-path:inset(50%)}
#alpine-snow-tables .as-switcher label.as-tab {display:inline-flex;align-items:center;justify-content:center;min-height:42px;user-select:none;cursor:pointer}
#alpine-snow-tables .as-model-radio:checked + label.as-tab {background:#e3edf8;color:#182639;border-color:#dbe8f8;font-weight:750}
#alpine-snow-tables .as-model-radio:focus-visible + label.as-tab {outline:3px solid #8accff;outline-offset:2px}
#alpine-snow-tables .as-model-view {display:none;flex:0 0 100%;width:100%;min-width:0;margin:8px 0 0}
#alpine-snow-tables #as-choice-hybrid:checked ~ .as-model-hybrid,
#alpine-snow-tables #as-choice-icon_d2:checked ~ .as-model-icon_d2,
#alpine-snow-tables #as-choice-ecmwf:checked ~ .as-model-ecmwf {display:block}
#alpine-snow-tables .as-empty {padding:14px 16px;background:#25344b;border:1px solid #485b76;border-radius:10px}
/* Hide any legacy ICON-CH2 panels with predictable IDs or data-model attributes. */
#tab-icon_ch2,#tab-icon-ch2,#panel-icon_ch2,#panel-icon-ch2,[data-model="icon_ch2"],[data-model="icon-ch2"]{display:none!important}
</style>
"""
def render_table_section(summary):
    """Render each model table into HTML at build time.

    Native radio buttons switch models without JavaScript.  In particular,
    they keep working when GitHub Pages/browser caches stale JS, when fetch()
    is unavailable, or when only one model supplied usable data.  Three
    choices are *always* visible; missing models explicitly say so.
    """
    names = MODEL_LABELS
    order = ("hybrid", "icon_d2", "ecmwf")
    date_list = summary.get("dates") or []
    regions = summary.get("regions") or []
    altitudes = summary.get("altitudes_m") or list(ALTITUDES)
    available = summary.get("available_day_indices") or {}
    table_data = summary.get("data") or {}

    def short_date(ymd):
        day = datetime.strptime(ymd, "%Y-%m-%d")
        weekdays = ("ma", "di", "wo", "do", "vr", "za", "zo")
        return f"{weekdays[day.weekday()]} {day:%d/%m}"

    if summary.get("generated_at_utc"):
        try:
            updated = datetime.fromisoformat(summary["generated_at_utc"])
            updated_str = updated.astimezone(LOCAL_TZ).strftime("%d/%m %H:%M")
        except (ValueError, TypeError):
            updated_str = "onbekend"
    else:
        updated_str = "onbekend"

    # Keep the native input and its visible label as siblings of the output
    # elements, making the CSS sibling selectors reliable in every browser.
    selected_model = next((m for m in order if available.get(m)), order[0])
    choices = []
    for model in order:
        count = len(available.get(model, []))
        label = f"{names[model]}" + (f" ({count} d)" if count else " (geen data)")
        choices.append(
            f'<input class="as-model-radio" type="radio" name="as-active-model" '
            f'id="as-choice-{model}" value="{model}"'
            + (' checked' if model == selected_model else '') + '>\n'
            + f'<label class="as-tab" for="as-choice-{model}">{escape(label)}</label>'
        )

    views = []
    for model in order:
        indices = [i for i in available.get(model, [])
                   if isinstance(i, int) and 0 <= i < len(date_list)]
        indices = indices[:MODEL_MAX_DISPLAY_DAYS[model]]
        if not indices:
            body = (
                f'<p class="as-empty">Voor {escape(names[model])} zijn in deze run '
                'geen regionale sneeuwvelden beschikbaar. De andere modellen '
                'kunnen wel gegevens bevatten.</p>'
            )
        else:
            horizon = 7 if model == 'ecmwf' else 2
            body = (
                f'<p><strong>{escape(names[model])}</strong> · '
                f'{len(indices)} beschikbare dag(en) (maximaal {horizon}) · '
                f'bijgewerkt {escape(updated_str)}. '
                'Een * duidt een gedeeltelijke dag aan.</p>'
            )
            for altitude in altitudes:
                col_heads = ''.join(
                    f'<th scope="col">{escape(short_date(date_list[i]))}</th>'
                    for i in indices
                )
                rows = []
                for region in regions:
                    values = (table_data.get(model, {}).get(region['id'], {})
                              .get(str(altitude), []))
                    cols = []
                    for i in indices:
                        cell = values[i] if i < len(values) else {}
                        present = bool(cell.get("available"))
                        amount = cell.get("range_cm", "–") if present else "–"
                        high = cell.get("high_cm")
                        css_class = ("as-none" if not present else
                                     "as-heavy" if isinstance(high, (int,float)) and high >= 30 else
                                     "as-medium" if isinstance(high, (int,float)) and high >= 10 else
                                     "as-light")
                        partial = bool(present and cell.get("partial"))
                        tooltip = (
                            ("Gedeeltelijke dag" if partial else "Dagberekening")
                            + (f" · {cell['coverage_hours']} uur" if cell.get("coverage_hours") is not None else "")
                        ) if present else "Geen data voor deze regio/hoogte"
                        cols.append(
                            f'<td class="{css_class}" title="{escape(tooltip, quote=True)}">'
                            f'{escape(str(amount))}{"*" if partial else ""}</td>'
                        )
                    rows.append(
                        f'<tr><td>{escape(str(region["name"]))}</td>{"".join(cols)}</tr>'
                    )
                body += (
                    f'<h3 class="as-alt">{escape(str(altitude))} m hoogte · cm per dag</h3>'
                    '<div class="as-scroll"><table><thead><tr>'
                    f'<th scope="col">Regio</th>{col_heads}</tr></thead>'
                    f'<tbody>{"".join(rows)}</tbody></table></div>'
                )
        views.append(
            f'<div class="as-model-view as-model-{model}" data-content-model="{model}">'
            f'{body}</div>'
        )

    return (
        '<section id="alpine-snow-tables" aria-label="Sneeuwverwachting per Alpenregio">'
        '<div class="as-panel">'
        '<h2>❄️ Sneeuwverwachting per Alpenregio '
        '<span class="as-badge">ECMWF tot 7 dagen</span></h2>'
        '<p>Kies hieronder het model. Hybrid en ICON-D2 tonen maximaal 2 '
        'beschikbare kalenderdagen; ECMWF maximaal 7 dagen. '
        'De drie hoogtes staan apart en gebruiken echte modelgegevens.</p>'
        '<div class="as-switcher" role="group" aria-label="Kies een weermodel">'
        + ''.join(choices) + ''.join(views) + '</div>'
        '<p class="as-foot">Ranges: 25e–75e percentiel van de sneeuwval '
        'in de regio op de gekozen hoogte (±100 m). Een * betekent '
        'gedeeltelijke dag; – betekent onvoldoende gegevens, niet 0 cm. '
        'Regiogrenzen zijn indicatief.</p>'
        '</div></section>'
    )


# Legacy website controls may still contain an ICON-CH2 navigation entry,
# since run_forecast.py itself is not changed in the user's repository.
# This script hides the obsolete button/tab without changing other models.
JS_HIDE_CH2 = r"""(() => {
  const match = /\bICON[\s_-]*CH2\b/i;
  const selector = 'button,a,[role="tab"],option,[data-model]';
  function hide() {
    for (const el of document.querySelectorAll(selector)) {
      if (el.closest('#alpine-snow-tables')) continue;
      const attr = [el.getAttribute('data-model'),el.getAttribute('href'),el.getAttribute('aria-controls'),el.getAttribute('id')].filter(Boolean).join(' ');
      const label = (el.textContent||'').trim();
      if (match.test(label) || /icon[-_]?ch2/i.test(attr)) {
        if (el.tagName==='OPTION') el.remove();
        else if (el.dataset.ch2Disabled !== '1') {
          el.dataset.ch2Disabled='1';
          el.style.setProperty('display','none','important');
        }
      }
    }
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', hide);
  else hide();
  const obs = new MutationObserver(hide);
  obs.observe(document.documentElement,{childList:true,subtree:true});
})();
"""


def publish_site(summary, site=SITE):
    """Publish snow tables in their own fourth TOP-LEVEL website tab.

    The three existing main tabs are for maps only. This function modifies
    the generated index.html, not run_forecast.py or any numerical model.
    The site's selectModel() JavaScript already drives tab selection; extend
    its list to include a fourth panel instead of placing a visible section
    outside the panels (the cause of the old repeated table display).
    """
    site.mkdir(parents=True, exist_ok=True)
    (site / "regional_snowfall.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (site / "hide_icon_ch2.js").write_text(JS_HIDE_CH2, encoding="utf-8")

    index = site / "index.html"
    if not index.exists():
        raise FileNotFoundError(f"Websitebestand ontbreekt: {index}")
    page = index.read_text(encoding="utf-8")

    if 'id="tab-tables"' not in page:
        # The runner is patched by enable_alpine_capture.py to use just
        # Hybrid / ECMWF / ICON-D2. Two JavaScript arrays must be extended:
        # one for validating the selected tab and one for toggling panels.
        existing = "['hybrid','ecmwf','icon_d2']"
        updated = "['hybrid','ecmwf','icon_d2','tables']"
        instances = page.count(existing)
        if instances != 2:
            raise RuntimeError(
                'Kan website-tabs niet veilig uitbreiden: '
                f'verwacht 2 modelkeuzelijsten, gevonden {instances}. '
                'Controleer of enable_alpine_capture.py eerst is uitgevoerd.'
            )
        if page.count('</nav>') != 1:
            raise RuntimeError(
                'Kan Sneeuwtabellen-tabblad niet invoegen: '
                'geen unieke afsluiting van de modelnavigatie gevonden'
            )
        if 'id="panel-hybrid"' not in page:
            raise RuntimeError('Hybrid-paneel niet gevonden in website-HTML')

        # Make the fourth tab exactly like the three existing main tabs.
        button = (
            '<button type="button" id="tab-tables" class="tab" role="tab" '
            'aria-controls="panel-tables" aria-selected="false" '
            'data-model="tables">❄️ Sneeuwtabellen</button>'
        )
        panel = (
            '<div id="panel-tables" class="snow-table-panel" role="tabpanel" '
            'aria-labelledby="tab-tables" hidden>\n'
            + render_table_section(summary) + '\n</div>\n'
        )
        # Important: no tables under individual map panels.
        page = page.replace('</nav>', button + '\n</nav>\n' + panel, 1)
        page = page.replace(existing, updated)

        # Hidden must override any author styles; the old page's stale-map
        # check targets .model-panel, so exclude this dedicated data panel.
        css = HTML_STYLES + '\n<style>#panel-tables[hidden]{display:none!important}</style>\n'
        if '</head>' in page:
            page = page.replace('</head>', css + '</head>', 1)
        else:
            page = css + page
        scripts = (
            '<script src="hide_icon_ch2.js" defer></script>\n'
        )
        if '</body>' in page:
            page = page.replace('</body>', scripts + '</body>', 1)
        else:
            page += scripts
        index.write_text(page, encoding="utf-8")

    # Keep old disabled-model status consistent when present.
    status_file = site / "status.json"
    if status_file.is_file():
        try:
            status = json.loads(status_file.read_text(encoding="utf-8"))
            if "icon_ch2" in status.get("models", {}):
                status["models"]["icon_ch2"]["status"] = "disabled"
                status["models"]["icon_ch2"]["map_count"] = 0
                status_file.write_text(
                    json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
        except (OSError, ValueError, TypeError) as exc:
            print(f'::warning::ICON-CH2 status metadata not updated: {exc}', flush=True)
    return len(summary["regions"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build", "self-test"))
    args = parser.parse_args()
    if args.command == "self-test":
        self_test()
        return
    summary = build_summary()
    available = summary["available_day_indices"]
    print("Alpen-tabellen: numerieke velden per model:", summary["models_found"], flush=True)
    print("Alpen-tabellen: beschikbare dagen per model:",
          {m: [summary["dates"][i] for i in indices] for m, indices in available.items()},
          flush=True)
    # Different model snapshots must stay separate even when rounded 5-cm
    # ranges happen to coincide. Log a concrete example for diagnostics.
    for model in MODEL_LABELS:
        example = next((
            (reg["name"], altitude, summary["dates"][i], cell["range_cm"])
            for reg in summary["regions"]
            for altitude in ALTITUDES
            for i in available.get(model, [])
            for cell in [summary["data"][model][reg["id"]][str(altitude)][i]]
            if cell.get("available")
        ), None)
        print(f"Alpen-tabellen: voorbeeld {model}: {example}", flush=True)
    if not any(available.values()):
        raise SystemExit(
            'ERROR: geen enkele beschikbare dag met echte sneeuwvelden! '
            'De website wordt niet overschreven met lege tabellen. '
            'Controleer de regels die beginnen met "Alpen tabel:" in de model-log.'
        )
    nr = publish_site(summary)
    print(f"Alpen-tabellen: {nr} regio's; modellen met gegevens: "
          + ", ".join(m for m, dates in available.items() if dates), flush=True)


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
    # Real extraction smoke test: a synthetic numeric field + the same DEM
    # must yield a JSON snapshot with a nonempty 1500-m Dolomites result.
    with TemporaryDirectory() as td:
        global CACHE
        old_cache = CACHE
        CACHE = Path(td) / 'snapshots'
        DEM_REGISTRY.clear()
        lon = np.linspace(5.0, 16.8, 590)
        lat = np.linspace(43.6, 49.0, 270)
        dem = np.full((lat.size, lon.size), 1500.0, dtype='float32')
        assert register_dem(lon, lat, dem)
        snow = np.full(dem.shape, 15.0, dtype='float32')
        date = datetime(2026, 11, 19, 12, tzinfo=timezone.utc)
        assert capture_numeric_field(snow, lon, lat, 'alps', date,
                                     12, 36, 'ecmwf')
        files = list(CACHE.glob('*.json'))
        assert len(files) == 1
        snapshot = json.loads(files[0].read_text())
        assert snapshot['statistics']['dolomites']['1500']['median'] == 15.0
        assert snapshot['statistics']['dolomites']['1500']['pixels'] >= MIN_PIXELS
        print('Alpen-tabellen: echte capturetest OK (Dolomieten 1500 m = 15 cm)')
        CACHE = old_cache
        DEM_REGISTRY.clear()
    # Partial horizons: one day ICON-D2, two Hybrid, seven ECMWF.
    with TemporaryDirectory() as td:
        previous = CACHE
        CACHE = Path(td) / 'synthetic_records'
        CACHE.mkdir()
        today = datetime(2026, 11, 19, 12, tzinfo=timezone.utc)
        for model, days in (("icon_d2", 1), ("hybrid", 2), ("ecmwf", 7)):
            for n in range(days):
                start = (datetime.combine(today.astimezone(LOCAL_TZ).date(),
                           clocktime.min, tzinfo=LOCAL_TZ)
                         + timedelta(days=n)).astimezone(timezone.utc)
                end = start + timedelta(hours=24)
                row = {
                    "model": model, "run_utc": today.isoformat(),
                    "start_utc": start.isoformat(), "end_utc": end.isoformat(),
                    "file": f"{model}-{n}.png",
                    "statistics": {"dolomites": {"1500": {
                        "q25": {"icon_d2": 0.0, "hybrid": 10.0, "ecmwf": 25.0}[model],
                        "median": {"icon_d2": 2.0, "hybrid": 15.0, "ecmwf": 35.0}[model],
                        "q75": {"icon_d2": 5.0, "hybrid": 20.0, "ecmwf": 45.0}[model],
                        "pixels": 150
                    }}}
                }
                (CACHE / f'{model}-{n}.json').write_text(json.dumps(row))
        summary = build_summary(datetime(2026, 11, 19, 17, tzinfo=timezone.utc))
        assert len(summary['available_day_indices']['icon_d2']) == 1
        assert len(summary['available_day_indices']['hybrid']) == 2
        assert len(summary['available_day_indices']['ecmwf']) == 7
        assert set(summary['models_found']) == {'hybrid','icon_d2','ecmwf'}
        assert summary['data']['icon_d2']['dolomites']['1500'][0]['available']
        assert summary['data']['ecmwf']['dolomites']['1500'][6]['available']
        assert summary['data']['icon_d2']['dolomites']['1500'][0]['range_cm'] == '0–5'
        assert summary['data']['hybrid']['dolomites']['1500'][0]['range_cm'] == '10–20'
        assert summary['data']['ecmwf']['dolomites']['1500'][0]['range_cm'] == '25–45'
        # Explicitly verify that even accidentally longer source series
        # cannot add extra columns to the short-range model tables.
        for model, n in (('icon_d2', 1), ('hybrid', 2), ('ecmwf', 7)):
            later = (datetime.combine(today.astimezone(LOCAL_TZ).date(),
                     clocktime.min, tzinfo=LOCAL_TZ)
                     + timedelta(days=n)).astimezone(timezone.utc)
            record = {
                'model': model, 'run_utc': today.isoformat(),
                'start_utc': later.isoformat(),
                'end_utc': (later+timedelta(hours=24)).isoformat(),
                'file': f'{model}-extra.png',
                'statistics': {'dolomites': {'1500': {
                    'q25': 10., 'median': 15., 'q75': 20., 'pixels': 150}}}
            }
            (CACHE/f'{model}-extra.json').write_text(json.dumps(record))
        capped = build_summary(datetime(2026, 11, 19, 17, tzinfo=timezone.utc))
        assert len(capped['available_day_indices']['icon_d2']) == 2
        assert len(capped['available_day_indices']['hybrid']) == 2
        assert len(capped['available_day_indices']['ecmwf']) == 7
        CACHE = previous
        print('Alpen-tabellen: afzonderlijke modelwaarden en horizons getest (1/2/7 dagen)')
    with TemporaryDirectory() as td:
        root = Path(td)
        fake = root / 'site'; fake.mkdir()
        # Use the same main-tab structure and selectModel() lists as
        # run_forecast.py AFTER enable_alpine_capture.py disables ICON-CH2.
        original_page = ('''<!doctype html><html><head><title>Snow maps</title></head>
<body><main><nav class="tabs" role="tablist">
<button id="tab-hybrid" data-model="hybrid">Hybrid</button>
<button id="tab-ecmwf" data-model="ecmwf">ECMWF</button>
<button id="tab-icon_d2" data-model="icon_d2">ICON-D2</button>
</nav>
<div id="panel-hybrid" class="model-panel">Hybrid Maps</div>
<div id="panel-ecmwf" class="model-panel" hidden>ECMWF Maps</div>
<div id="panel-icon_d2" class="model-panel" hidden>ICON-D2 Maps</div>
</main><script>
function selectModel(model) {
  if (!['hybrid','ecmwf','icon_d2'].includes(model)) model='hybrid';
  for (const key of ['hybrid','ecmwf','icon_d2']) {
    document.getElementById('panel-'+key).hidden = key !== model;
    document.getElementById('tab-'+key).setAttribute('aria-selected',String(key===model));
  }
}
</script></body></html>''')
        (fake / 'index.html').write_text(original_page, encoding='utf-8')
        sample = summary  # Synthetic 1/2/7-day model data from the test above
        publish_site(sample, fake)
        page = (fake/'index.html').read_text(encoding='utf-8')
        assert page.count('id="alpine-snow-tables"') == 1
        assert page.count('id="tab-tables"') == 1
        assert page.count('id="as-choice-hybrid"') == 1
        assert page.count('id="as-choice-icon_d2"') == 1
        assert page.count('id="as-choice-ecmwf"') == 1
        assert page.count('data-content-model=') == 3
        assert 'as-choice-ecmwf:checked ~ .as-model-ecmwf' in page
        assert 'alpine_snow_tables.js' not in page
        assert '10–20' in page and '25–45' in page and '0–5' in page
        assert page.count('id="panel-tables"') == 1
        assert page.count("['hybrid','ecmwf','icon_d2','tables']") == 2
        assert page.count('data-model="tables"') == 1
        assert page.index('</nav>') < page.index('id="panel-tables"') < page.index('id="panel-hybrid"')
        assert 'Hybrid Maps' in page and 'ECMWF Maps' in page and 'ICON-D2 Maps' in page
        assert 'aria-labelledby="tab-tables" hidden' in page
        assert '#panel-tables[hidden]{display:none!important}' in page
        # A second publication should not duplicate the tab or the tables.
        publish_site(sample, fake)
        page = (fake/'index.html').read_text(encoding='utf-8')
        assert page.count('id="alpine-snow-tables"') == 1
        assert page.count('id="tab-tables"') == 1
        assert (fake/'regional_snowfall.json').is_file()
        assert (fake/'hide_icon_ch2.js').is_file()
        assert 'icon_ch2' not in json.loads((fake/'regional_snowfall.json').read_text()).get('models_found', {})
        print('Alpen-tabellen: vierde website-tab getest; geen duplicaten')
    print('Alpen-tabellen self-test OK')


if __name__ == "__main__":
    main()
