"""Produce standalone ICON-D2 snow, snowline and freezing-level maps for Pages.

Uses the provided SnowAll_ICON-D2.py physics and map styling with local
calendar-day snowfall intervals. This wrapper configures output paths,
regions and DPI, verifies complete output and writes website metadata.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import importlib.util
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "SnowAll_ICON-D2.py"


def parse_areas(value: str) -> tuple[str, ...]:
    areas = tuple(part.strip().lower() for part in value.split(",") if part.strip())
    if not areas or len(set(areas)) != len(areas) or any(a not in ("alps", "full") for a in areas):
        raise ValueError("ICON_D2_AREAS must be 'alps', 'full' or 'alps,full'")
    return areas


def rename_snow_maps(folder: Path, run: datetime, start: int,
                     areas: tuple[str, ...], windows: list[tuple[int, int]]) -> None:
    """Mark daily and cumulative snowfall explicitly for website grouping."""
    stamp = run.strftime("%Y%m%d_%H")
    for area in areas:
        old_prefix = f"icon_d2_verse_sneeuw_{area}_{stamp}z_"
        conversions = {
            f"dag{i:02d}": f"icon_d2_sneeuwval_{area}_{stamp}z_f{a:03d}-{b:03d}.png"
            for i, (a, b) in enumerate(windows, 1)
        }
        conversions["cumulatief"] = (
            f"icon_d2_sneeuwval_cumulatief_{area}_{stamp}z_f{start:03d}-048.png"
        )
        for suffix, target in conversions.items():
            original = folder / f"{old_prefix}{suffix}.png"
            if original.is_file():
                dest = folder / target
                if dest.exists():
                    raise FileExistsError(f"Duplicate ICON-D2 output: {dest.name}")
                original.rename(dest)


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(f"Missing original ICON-D2 source: {SOURCE}")
    spec = importlib.util.spec_from_file_location("icon_d2_standalone", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load ICON-D2 source: {SOURCE}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    areas = parse_areas(os.environ.get("ICON_D2_AREAS", "alps"))
    dpi = int(os.environ.get("ICON_D2_DPI", "200"))
    if not 200 <= dpi <= 400:
        raise ValueError("ICON_D2_DPI must be between 200 and 400")
    max_age = float(os.environ.get("ICON_D2_MAX_RUN_AGE_H", "24"))
    if not 1 <= max_age <= 96:
        raise ValueError("ICON_D2_MAX_RUN_AGE_H must be in 1..96")

    out = Path(os.environ["ICON_D2_OUTPUT_DIR"])
    out.mkdir(parents=True, exist_ok=True)
    if list(out.glob("*.png")):
        raise RuntimeError("ICON-D2 output folder is not empty at start")

    mod.OUTPUT_DIR = out
    mod.AREAS_TO_MAKE = areas
    mod.DPI = dpi
    mod.USE_ESRI_RELIEF = os.environ.get("ICON_D2_ESRI_RELIEF", "1") != "0"
    mod.LOGO_PATH = Path.home() / "logo-sneeuwhoogte.png"
    if not mod.LOGO_PATH.is_file():
        raise FileNotFoundError(f"ICON-D2 logo missing: {mod.LOGO_PATH}")

    # Keep shared temporary data and relief tiles out of the published website.
    mod.DATA_DIR = Path.home() / "Documents" / "SneeuwhoogteNL" / "data"
    mod.TILE_CACHE_DIR = mod.DATA_DIR / "esri_tiles"
    mod.SNOW_INTERP_STEP["alps"] = float(os.environ.get("ICON_D2_GRID_STEP_ALPS", "0.008"))
    mod.SNOW_INTERP_STEP["full"] = float(os.environ.get("ICON_D2_GRID_STEP_FULL", "0.012"))
    for area in areas:
        if not 0.004 <= mod.SNOW_INTERP_STEP[area] <= 0.04:
            raise ValueError(f"Invalid ICON-D2 snow grid resolution for {area}")

    # UI ONLY: adopt the uploaded dark/slate design. All DWD fields, time
    # windows and the existing grey snow-/freezing-line logic remain intact.
    from snow_dark_adapter import install_icon_d2
    install_icon_d2(mod)

    selected_run = None
    selected_start = None
    recorded_extents = {}
    original_newest = mod.newest_common_run
    original_current_lead = mod.current_lead
    original_determine_extent = mod.determine_extent

    def record_extent(x, y, valid, area):
        # Reuse the EXACT existing snow/altitude map extent on new maps.
        extent = original_determine_extent(x, y, valid, area)
        recorded_extents[area] = extent
        return extent

    mod.determine_extent = record_extent

    def record_run(session):
        nonlocal selected_run
        run = original_newest(session)
        age = (datetime.now(timezone.utc) - run).total_seconds() / 3600
        if age > max_age:
            raise RuntimeError(f"Latest complete ICON-D2 run {age:.1f} h old (limit {max_age:g} h)")
        selected_run = run
        return run

    def record_lead(run):
        nonlocal selected_start
        selected_start = original_current_lead(run)
        return selected_start

    mod.newest_common_run = record_run
    mod.current_lead = record_lead

    print(f"ICON-D2 standalone: {areas=}, {dpi=}, latest run within {max_age:g} h", flush=True)
    mod.main()
    if selected_run is None or selected_start is None:
        raise RuntimeError("ICON-D2 did not record a model run and start time")

    windows = mod.local_day_windows(selected_run, selected_start)
    rename_snow_maps(out, selected_run, selected_start, areas, windows)

    # EXTRA, OPTIONAL: 1500/2000/3000 m temperature every 6 hours. These maps
    # never affect any existing ICON-D2 calculations. If DWD temporarily
    # omits a pressure-level field, keep the complete original forecast.
    temperature_count = 0
    if os.environ.get("ICON_D2_TEMPERATURE", "1") != "0":
        try:
            from icon_d2_temperature import HEIGHTS_M, produce_temperature_maps
            created = produce_temperature_maps(
                mod, selected_run, areas, recorded_extents
            )
            temperature_count = len(created)
            required = len(areas) * len(HEIGHTS_M) * len(mod.FORECAST_HOURS)
            if temperature_count != required:
                raise RuntimeError(
                    f"Incomplete temperature series: {temperature_count}/{required}"
                )
            print(f"ICON-D2 temperature: {temperature_count} new maps", flush=True)
        except Exception as exc:
            # Keep the existing snow/snowline/freezing-line maps.
            for filename in out.glob("icon_d2_temperatuur_*.png"):
                filename.unlink(missing_ok=True)
            temperature_count = 0
            print(
                f"::warning::ICON-D2 temperature skipped: {type(exc).__name__}: {exc}. "
                "All existing ICON-D2 maps remain available.", flush=True
            )

    found = sorted(out.glob("*.png"))
    count_per_area = 2 * len(mod.FORECAST_HOURS) + len(windows) + 1
    expected = len(areas) * count_per_area + temperature_count
    if len(found) != expected:
        raise RuntimeError(f"ICON-D2 produced {len(found)}/{expected} maps; check the plot errors above")

    meta = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "forecast_start_utc": (selected_run + timedelta(hours=selected_start)).isoformat(),
        "forecast_end_utc": (selected_run + timedelta(hours=48)).isoformat(),
        "model_runs_utc": {"ICON-D2": selected_run.isoformat()},
        "map_count": len(found),
        "areas": list(areas),
        "days": len(windows),
        "altitude_step_hours": 6,
        "temperature_step_hours": 6,
        "temperature_heights_m": list(HEIGHTS_M) if temperature_count else [],
        "temperature_map_count": temperature_count,
        "temperature_method": "T850/T700 linearly interpolated in FI-derived height (approximate)",
        "snow_intervals": "Belgian calendar days, cumulative to run +48h",
        "dpi": dpi,
    }
    (out / "forecast_metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"ICON-D2 OK: {len(found)} maps; forecast until run +48 h", flush=True)


if __name__ == "__main__":
    main()
