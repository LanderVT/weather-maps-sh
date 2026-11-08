"""Standalone ICON-CH2-EPS Alpine snowfall maps for GitHub Pages.

Produces exactly five 24-hour snowfall maps and four cumulative maps
up to +120 h (or 4 + 3 maps if ICON_CH2_DAYS=4). Keeps the ORIGINAL
SnowDays_ICON-CH2.py meteorological calculations intact.  No maps of
new snow remaining on the ground are made.

Uses MeteoSwiss's *deterministic control run* of ICON-CH2-EPS, not the
21-member ensemble mean.  Inputs are cumulative TOT_PREC, SNOWLMT,
T_2M and the static native-grid/surface terrain fields; snowfall is
computed by the already-used SnowDays_ICON-CH2.py approach.

Only the plotting function is swapped for the established dark/slate
map_style_reference.py via snow_dark_adapter.install_snowdays().

Run via run_forecast.py, not directly for ordinary scheduled publishing.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "SnowDays_ICON-CH2.py"


def read_settings():
    days = int(os.environ.get("ICON_CH2_DAYS", "5"))
    if days not in (4, 5):
        raise ValueError("ICON_CH2_DAYS must be 4 or 5")

    # Six-hourly weather input is cheaper, but the displayed snowfall
    # windows are 24h. Set 3 for more accurate transition timing.
    step_h = int(os.environ.get("ICON_CH2_STEP_HOURS", "6"))
    if step_h not in (3, 6):
        raise ValueError("ICON_CH2_STEP_HOURS must be 3 or 6")

    grid_step = float(os.environ.get("ICON_CH2_GRID_STEP_ALPS", "0.012"))
    if not 0.008 <= grid_step <= 0.03:
        raise ValueError("ICON_CH2_GRID_STEP_ALPS must be 0.008..0.03")

    dpi = int(os.environ.get("ICON_CH2_DPI", "200"))
    if not 200 <= dpi <= 400:
        raise ValueError("ICON_CH2_DPI must be 200..400")

    max_age_h = float(os.environ.get("ICON_CH2_MAX_RUN_AGE_H", "24"))
    if not 6 <= max_age_h <= 24:
        raise ValueError("ICON_CH2_MAX_RUN_AGE_H must be 6..24")

    return days, step_h, grid_step, dpi, max_age_h


def aggregate_daily(steps, days, iterated_fields):
    """Accumulate *original* model snowfall into 24-hour blocks.

    Yields (daily_snow, daily_rain, x, y, land), with each day being
    an independent 0-24, 24-48, ... model-lead window. We do not
    calculate any on-ground accumulation or change original physics.
    """
    daily_snow = [None] * days
    daily_rain = [None] * days
    geometry = None
    seen = 0
    previous_step = 0
    for item in iterated_fields:
        a, b = int(item["a"]), int(item["b"])
        if a != previous_step or b - a != steps or b > 24 * days:
            raise ValueError(f"Unexpected ICON-CH2 interval +{a} to +{b}")
        previous_step = b
        day_index = (b - 1) // 24
        if geometry is None:
            geometry = (item["xf"], item["yf"], np.asarray(item["land"], dtype=bool))
        snow = np.asarray(item["snow_cm"], dtype=float)
        rain = np.asarray(item["rain_mm"], dtype=float)
        if snow.shape != geometry[2].shape or rain.shape != snow.shape:
            raise ValueError("ICON-CH2 field grid size changed between steps")
        if daily_snow[day_index] is None:
            daily_snow[day_index] = np.zeros(snow.shape, dtype=float)
            daily_rain[day_index] = np.zeros(snow.shape, dtype=float)
        daily_snow[day_index] += snow
        daily_rain[day_index] += rain
        seen += 1

    if seen != 24 * days // steps or any(x is None for x in daily_snow):
        raise RuntimeError(f"Incomplete ICON-CH2 time series: {seen} intervals")

    return daily_snow, daily_rain, geometry


def main() -> None:
    if not SOURCE.is_file():
        raise FileNotFoundError(f"Missing model script: {SOURCE}")

    days, step_h, grid_step, dpi, max_age_h = read_settings()
    spec = importlib.util.spec_from_file_location("icon_ch2_daily_standalone", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {SOURCE}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    out = Path(os.environ["ICON_CH2_OUTPUT_DIR"])
    out.mkdir(parents=True, exist_ok=True)
    if any(out.glob("*.png")):
        raise RuntimeError("ICON-CH2 output folder is not empty at start")

    mod.OUTPUT_DIR = out
    mod.AREAS_TO_MAKE = ("alps",)
    mod.GRID_STEP = grid_step
    mod.STEP_HOURS = step_h
    mod.MAX_LEAD = 24 * days
    mod.DPI = dpi
    mod.MAX_RUN_AGE_H = max_age_h
    mod.USE_ESRI_RELIEF = os.environ.get("ICON_CH2_ESRI_RELIEF", "1") != "0"
    mod.SHOW_RAIN = False
    mod.LOGO_PATH = Path.home() / "logo-sneeuwhoogte.png"
    if not mod.LOGO_PATH.is_file():
        raise FileNotFoundError(f"ICON-CH2 logo missing: {mod.LOGO_PATH}")

    # One common temp/cache folder across model subprocesses, cleared by
    # run_forecast.py after publishing. No large GRIBs go into GitHub Pages.
    mod.DATA_DIR = Path.home() / "Documents" / "SneeuwhoogteNL" / "data"
    mod.TILE_CACHE_DIR = mod.DATA_DIR / "esri_tiles"
    mod.DEM_CACHE_DIR = mod.DATA_DIR / "dem_tiles"

    from snow_dark_adapter import install_snowdays

    set_style_terrain = install_snowdays(mod, webcam=False)
    original_load_dem = mod.load_dem

    def load_dem_for_style(*args, **kwargs):
        x, y, dem = original_load_dem(*args, **kwargs)
        set_style_terrain(x, y, dem)
        return x, y, dem

    mod.load_dem = load_dem_for_style
    windows = [(24 * k, 24 * (k + 1)) for k in range(days)]
    steps = tuple(range(0, mod.MAX_LEAD + 1, mod.STEP_HOURS))

    print(
        f"ICON-CH2 separate snow forecast: {days} days, {len(steps)} "
        f"time points, {step_h}-hour meteorological inputs, {dpi} DPI, "
        f"grid={grid_step} deg; domain=Alps", flush=True,
    )

    with requests.Session() as session:
        session.headers.update({"User-Agent": "SneeuwhoogteNL-ICON-CH2-standalone/1.0"})
        run = mod.newest_run(session)
        age = (datetime.now(timezone.utc) - run).total_seconds() / 3600
        if age > max_age_h:
            raise RuntimeError(f"Latest complete ICON-CH2 run {age:.1f}h old (limit {max_age_h:g}h)")

        daily_snow, daily_rain, geometry = aggregate_daily(
            step_h, days,
            mod.iterate_steps(session, run, steps, "alps"),
        )
        x, y, land = geometry
        extent = mod.AREA_LIMITS["alps"]
        total_snow = np.zeros_like(daily_snow[0])
        total_rain = np.zeros_like(daily_rain[0])

        for day, (start, end) in enumerate(windows):
            snow = np.where(land, daily_snow[day], np.nan)
            rain = np.where(land, daily_rain[day], np.nan)
            print(f"ICON-CH2 day {day + 1}: +{start}-{end}h, peak {np.nanmax(snow):.0f} cm", flush=True)
            # Non-None rain signals *new snowfall* rather than on-ground snow;
            # the adapter does not render rain when SHOW_RAIN=False.
            mod.plot_snowfall(snow, x, y, extent, "alps", run, start, end,
                              rain=rain, total=False)
            total_snow += np.nan_to_num(daily_snow[day])
            total_rain += np.nan_to_num(daily_rain[day])
            if day > 0:  # +0..+24 would duplicate the first daily card
                mod.plot_snowfall(
                    np.where(land, total_snow, np.nan), x, y, extent, "alps",
                    run, 0, end,
                    rain=np.where(land, total_rain, np.nan), total=True,
                )

    found = sorted(out.glob("*.png"))
    expected = 2 * days - 1
    if len(found) != expected:
        raise RuntimeError(f"ICON-CH2 generated {len(found)} instead of {expected} maps")

    meta = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "forecast_start_utc": run.isoformat(),
        "forecast_end_utc": (run + timedelta(hours=24 * days)).isoformat(),
        "model_runs_utc": {"ICON-CH2 (MeteoSwiss)": run.isoformat()},
        "model": "ICON-CH2-EPS deterministic control run",
        "map_count": expected,
        "areas": ["alps"],
        "days": days,
        "snow_intervals": "24-hour blocks from ICON-CH2 model initialization",
        "snow_input_step_hours": step_h,
        "grid_step_deg": grid_step,
        "dpi": dpi,
    }
    (out / "forecast_metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"ICON-CH2 OK: {expected} Alpine maps, through +{24 * days} h", flush=True)


if __name__ == "__main__":
    main()
