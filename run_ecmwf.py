"""Run the original standalone SnowDays_ECMWF.py on the GitHub worker.

The model's calculation and plotting functions are left unchanged: this wrapper
only supplies isolated output folders, appropriate cloud resolution and metadata.
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
import importlib.util
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "SnowDays_ECMWF.py"


def main() -> None:
    spec = importlib.util.spec_from_file_location("ecmwf_standalone", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {SOURCE}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    out = Path(os.environ["ECMWF_OUTPUT_DIR"])
    out.mkdir(parents=True, exist_ok=True)
    if list(out.glob("*.png")):
        raise RuntimeError("ECMWF output directory is not empty at start")
    mod.OUTPUT_DIR = out

    areas = tuple(
        s.strip().lower() for s in os.environ.get("ECMWF_AREAS", "alps").split(",")
        if s.strip()
    )
    if not areas or len(set(areas)) != len(areas) or any(s not in ("alps", "full") for s in areas):
        raise ValueError("ECMWF_AREAS must be 'alps', 'full', or 'alps,full'")
    mod.AREAS_TO_MAKE = areas
    mod.N_DAYS = int(os.environ.get("ECMWF_N_DAYS", "7"))
    if not 1 <= mod.N_DAYS <= 7:
        raise ValueError("ECMWF_N_DAYS must be in 1..7")

    for area in areas:
        env_name = "ECMWF_GRID_STEP_" + area.upper()
        default = "0.008" if area == "alps" else "0.016"
        step = float(os.environ.get(env_name, default))
        if not 0.004 <= step <= 0.04:
            raise ValueError(f"{env_name} must be between 0.004 and 0.04")
        mod.GRID_STEP[area] = step

    mod.DPI = int(os.environ.get("ECMWF_DPI", "200"))
    if not 80 <= mod.DPI <= 400:
        raise ValueError("ECMWF_DPI must be in 80..400")
    mod.USE_ESRI_RELIEF = os.environ.get("ECMWF_ESRI_RELIEF", "1") != "0"
    mod.LOGO_PATH = Path.home() / "logo-sneeuwhoogte.png"
    if not mod.LOGO_PATH.exists():
        raise FileNotFoundError(f"ECMWF logo missing: {mod.LOGO_PATH}")

    print(
        f"Standalone ECMWF: areas={areas}, days={mod.N_DAYS}, DPI={mod.DPI}, "
        f"grid={{{', '.join(f'{a}: {mod.GRID_STEP[a]}' for a in areas)}}}",
        flush=True,
    )
    selected_run = None
    original_newest_run = mod.newest_run

    def record_newest_run(session, last_lead):
        nonlocal selected_run
        selected_run = original_newest_run(session, last_lead)
        return selected_run

    mod.newest_run = record_newest_run
    mod.main()

    found = list(out.glob("*.png"))
    expected = len(areas) * (3 * mod.N_DAYS - 1)
    # The standalone script catches per-map plotting errors internally;
    # require a *complete* set rather than publishing partial forecasts.
    if len(found) != expected:
        raise RuntimeError(f"ECMWF produced {len(found)}/{expected} maps")
    if selected_run is None:
        raise RuntimeError("ECMWF did not find an available recent model run")

    windows = mod.day_windows(selected_run, datetime.now(timezone.utc))
    start = selected_run + timedelta(hours=windows[0][0])
    end = selected_run + timedelta(hours=windows[-1][1])
    data = {
        "forecast_start_utc": start.isoformat(),
        "forecast_end_utc": end.isoformat(),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "model_runs_utc": {"ECMWF": selected_run.isoformat()},
        "map_count": len(found),
        "areas": list(areas),
        "days": mod.N_DAYS,
        "grid_steps_deg": {a: mod.GRID_STEP[a] for a in areas},
    }
    (out / "forecast_metadata.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"ECMWF OK: {len(found)} standalone forecast maps", flush=True)


if __name__ == "__main__":
    main()
