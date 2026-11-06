"""

Sneeuwhoogte.nl Hybrid | Hybride sneeuwverwachting komende 48 uur

uit AROME (Météo-France), AROME-AT (GeoSphere), ICON-CH1, ICON-CH2,

ICON-D2 en ECMWF, gedownscaled naar ongeveer 400 m

Werkwijze (per blok van HYBRID_STEP uur):

1. Elk model rekent zelf, consistent, sneeuw (mm en cm) en regen (mm) uit

   met zijn EIGEN neerslag en sneeuwgrens, via iterate_steps() uit de

   SnowDays_*.py-scripts (zelfde DEM, gradient en dichtheid).

2. Per pixel een gewogen gemiddelde over de modellen die daar data

   hebben (AROME stopt rond 13.3 O; gewichten worden daar herverdeeld).

   Gewichten hangen af van de tijd vanaf nu (WEIGHTS).

3. Probability-matched mean (USE_PM_MEAN) op de DAGSOM van de neerslag:

   het patroon komt van het gewogen gemiddelde, de verdeling van de

   hoeveelheden van de modellen zelf. Zo vlakt middelen de pieken niet

   kunstmatig af. Alle blokken van die dag krijgen dezelfde schaalfactor

   (per blok matchen zou pieken op dezelfde plek opstapelen). De

   sneeuwfractie en sneeuwdichtheid per pixel blijven die van het

   gewogen gemiddelde.

4. Een keer het mini-sneeuwdekmodel (inklinken en smelten) op het hybride resultaat,

   met de gewogen gemiddelde temperatuur.

Kaarten (map outputmaps/HYBRID): sneeuwval per dag, cumulatief,

nieuwe sneeuw op de grond.

Python-pakketten: zie de SnowDays_*.py-scripts.

Gebruik: python SnowHybrid.py

"""

import importlib.util
import os

from datetime import datetime, timedelta, timezone

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np

import requests

# ============================================================

# 1. INSTELLINGEN

# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

OUTPUT_DIR = Path(os.environ.get("HYBRID_OUTPUT_DIR", str(Path.home() / "Documents" / "SneeuwhoogteNL" / "outputmaps" / "HYBRID")))

MODEL_FILES = {

    "AROME": "SnowDays_AROME.py",

    "AROME-AT": "SnowDays_AROME-AT.py",

    "ICON-CH1": "SnowDays_ICON-CH1.py",

    "ICON-CH2": "SnowDays_ICON-CH2.py",

    "ICON-D2": "SnowDays_ICON-D2.py",

    "ECMWF": "SnowDays_ECMWF.py",

}

# Gewicht per model voor (0-24 uur, 24-48 uur) vanaf nu. Startwaarden

# (expertschatting); later bij te stellen met verificatie. Ze hoeven niet

# op te tellen tot 1: per pixel wordt genormaliseerd over de modellen

# die daar data hebben.

WEIGHTS = {

    "AROME": (0.20, 0.15),

    "AROME-AT": (0.20, 0.15),

    "ICON-CH1": (0.20, 0.20),   # loopt maar tot +33 uur

    "ICON-CH2": (0.15, 0.20),

    "ICON-D2": (0.15, 0.15),

    "ECMWF": (0.10, 0.15),

}

HYBRID_HOURS = 48

HYBRID_STEP = 3          # uur; sommige modellen (ECMWF) leveren 6-uurlijks
# De 6-uursvelden worden in ModelStream.interval() naar overlappende
# 3-uursblokken verdeeld op basis van hun tijdsduur.

AREA = "alps"

USE_PM_MEAN = True

PM_SAMPLE = 5           # elke n-de pixel voor de verdeling (sneller)

LABEL = "Hybride sneeuwverwachting"

FILE_PREFIX = "hybrid"
MIN_MODELS_PER_BLOCK = int(os.environ.get("HYBRID_MIN_MODELS", "2"))

# ============================================================

# 2. MODELSCRIPTS LADEN

# ============================================================

def load_module(name, filename):

    spec = importlib.util.spec_from_file_location(

        name, SCRIPT_DIR / filename

    )

    module = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(module)

    return module

def enable_webcam_georeferencing(lib):
    """Save geo calibration next to Hybrid PNGs, without changing the images.

    Matplotlib tight-crops figures when saving. The overlay must use the
    *actual rendered axes position*, not an estimated rectangle, to avoid
    placing camera links over the wrong mountain or outside the map.
    """
    import json

    original_finish = lib.finish_map

    def finish_with_georef(fig, ax, cf, title, label, levels,
                           filepath, extent, area, *args, **kwargs):
        result = original_finish(
            fig, ax, cf, title, label, levels,
            filepath, extent, area, *args, **kwargs
        )
        try:
            # The figure object remains available even after plt.close(fig).
            # savefig has already rendered this figure; avoid a second
            # expensive Cartopy/relief redraw on the hourly GitHub runner.
            renderer = fig.canvas.get_renderer()
            tight = fig.get_tightbbox(renderer)  # inches
            axes = ax.get_window_extent(renderer).transformed(
                fig.dpi_scale_trans.inverted()
            )
            # Original SnowDays_ICON-CH2.py finish_map uses pad_inches=0.06.
            pad = 0.06
            w = float(tight.width + 2 * pad)
            h = float(tight.height + 2 * pad)
            box = [
                100.0 * (axes.x0 - tight.x0 + pad) / w,
                100.0 * (tight.y1 - axes.y1 + pad) / h,
                100.0 * axes.width / w,
                100.0 * axes.height / h,
            ]
            left, top, width, height = box
            if not (0 <= left <= 100 and 0 <= top <= 100
                    and 0 < width <= 100 and 0 < height <= 100
                    and left + width <= 100.5 and top + height <= 100.5):
                raise ValueError(f"Invalid Hybrid geo bounds: {box}")
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            metadata = {
                "plot_box_pct": [round(z, 6) for z in box],
                "extent_lonlat": [xmin, xmax, ymin, ymax],
            }
            target = Path(str(filepath) + '.geo.json')
            target.write_text(json.dumps(metadata), encoding='utf-8')
        except Exception as exc:
            # A map without hotspots is much better than a failed forecast.
            print(f"WAARSCHUWING: webcam-overlay overgeslagen ({exc})")
        return result

    lib.finish_map = finish_with_georef


def embed(field, xf_model, yf_model, xf, yf, fill):

    """Plaats een modelveld (deelrooster, bv. AROME) in het Hybrid-rooster."""

    ix = int(np.argmin(abs(xf - xf_model[0])))

    iy = int(np.argmin(abs(yf - yf_model[0])))

    if not (

        np.isclose(xf[ix], xf_model[0]) and np.isclose(yf[iy], yf_model[0])

    ):

        raise ValueError("Modelrooster valt niet op het Hybrid-rooster")

    out = np.full((len(yf), len(xf)), fill, dtype=field.dtype)

    out[iy:iy + len(yf_model), ix:ix + len(xf_model)] = field

    return out

class ModelStream:
    """Voeg modeltijdstappen (3u of 6u) samen tot 3-uurs Hybrid-blokken.

    Voor native 6-uursvelden wordt de totale neerslag en sneeuw binnen
    het overlappende deel van het tijdvak naar rato van de duur verdeeld.
    Temperatuur is een tijdgewogen gemiddelde. Dit behoudt totale neerslag
    maar veronderstelt bij ECMWF uniforme verdeling binnen zijn 6u-vak.
    """

    def __init__(self, name, module, session, run, t0, t_end, xf, yf):
        self.name = name
        self.run = run
        self.xf, self.yf = xf, yf
        self.pending = None

        step = module.STEP_HOURS
        max_lead = getattr(module, "MAX_LEAD", 10 ** 6)

        # Negeer de eerste overlappende native stap niet bij een
        # 3-uurs start die midden in een 6-uurs tijdvak ligt.
        start = max(0, int((t0 - run).total_seconds() // 3600))
        start -= start % step

        # Ook het laatste tijdvak moet tot t_end reiken; anders valt de
        # laatste halve ECMWF-stap weg wanneer t_end tussen twee 6u-tijden ligt.
        end = min(int((t_end - run).total_seconds() // 3600), max_lead)
        end += -end % step
        end = min(end, max_lead)

        self.gen = (
            module.iterate_steps(session, run, list(range(start, end + 1, step)), AREA)
            if end > start else iter(())
        )

    def interval(self, t_start, t_stop):
        acc = None
        covered = 0.0

        while True:
            try:
                item = self.pending if self.pending is not None else next(self.gen, None)
            except Exception as exc:
                print(f"WAARSCHUWING: {self.name} valt weg ({exc})")
                self.gen = iter(())
                item = None
            self.pending = None

            if item is None:
                break

            s = self.run + timedelta(hours=item["a"])
            e = self.run + timedelta(hours=item["b"])
            if e <= t_start:
                continue
            if s >= t_stop:
                self.pending = item
                break

            overlap_start = max(s, t_start)
            overlap_end = min(e, t_stop)
            hours = (overlap_end - overlap_start).total_seconds() / 3600
            source_hours = (e - s).total_seconds() / 3600
            if hours <= 0 or source_hours <= 0:
                continue

            fields = {
                k: embed(item[k], item["xf"], item["yf"], self.xf, self.yf, 0.0)
                for k in ("snow_mm", "rain_mm", "snow_cm", "temp")
            }
            valid = embed(item["land"], item["xf"], item["yf"],
                          self.xf, self.yf, False)
            if acc is None:
                acc = {k: np.zeros_like(v) for k, v in fields.items()}
                acc["valid"] = valid.copy()
            else:
                acc["valid"] &= valid

            fraction = hours / source_hours
            for k in ("snow_mm", "rain_mm", "snow_cm"):
                acc[k] += fields[k] * fraction
            acc["temp"] += fields["temp"] * hours
            covered += hours

            if e >= t_stop:
                # Bewaar de ongebruikte rest van een native 6-uursveld
                # voor het volgende Hybrid-blok van 3 uur.
                if e > t_stop:
                    self.pending = item
                break

        block_hours = (t_stop - t_start).total_seconds() / 3600
        if acc is None or not np.isclose(covered, block_hours):
            return None
        acc["temp"] /= covered
        return acc

# ============================================================

# 3. COMBINEREN

# ============================================================

def pm_mean(mean, samples, weights, mask):

    """Probability-matched mean: rangorde van 'mean', waarden uit de

    gewogen gepoolde verdeling van de modellen."""

    values = np.concatenate(samples)

    w = np.concatenate(weights)

    keep = w > 0

    values, w = values[keep], w[keep]

    if values.size == 0:

        return mean

    order = np.argsort(values)

    values = values[order]

    cdf = np.cumsum(w[order])

    cdf /= cdf[-1]

    flat = mean[mask]

    rank = np.empty(flat.size)

    rank[np.argsort(flat, kind="stable")] = (

        (np.arange(flat.size) + 0.5) / flat.size

    )

    out = mean.copy()

    out[mask] = np.interp(rank, cdf, values)

    return out

def hybrid_block(parts, weights_now, shape):

    """Gewogen gemiddelde van een blok: neerslag, sneeuwfractie,

    cm sneeuw per mm en temperatuur (nog zonder probability matching)."""

    wsum = np.zeros(shape)

    sums = {k: np.zeros(shape) for k in ("precip", "snow_mm", "snow_cm", "temp")}

    for name, acc in parts.items():

        W = weights_now[name] * acc["valid"]

        wsum += W

        sums["precip"] += W * (acc["snow_mm"] + acc["rain_mm"])

        sums["snow_mm"] += W * acc["snow_mm"]

        sums["snow_cm"] += W * acc["snow_cm"]

        sums["temp"] += W * acc["temp"]

    mask = wsum > 0

    safe = np.where(mask, wsum, 1.0)

    precip = sums["precip"] / safe

    snow = sums["snow_mm"] / safe

    frac = np.where(precip > 1e-6, snow / np.maximum(precip, 1e-6), 0.0)

    cm_per_mm = np.where(

        snow > 1e-6, sums["snow_cm"] / safe / np.maximum(snow, 1e-6), 1.0

    )

    return {

        "mask": mask,

        "precip": precip.astype(np.float32),

        "frac": frac.astype(np.float32),

        "cm_per_mm": cm_per_mm.astype(np.float32),

        "temp": (sums["temp"] / safe).astype(np.float32),

    }

def day_windows(t0, t_end):

    """(start, eind) in uren na t0: rest van vandaag + volle UTC-dagen,

    tot t_end. De laatste dag stopt bij t_end (eindtijd in de titel)."""

    total = int((t_end - t0).total_seconds() // 3600)

    windows = []

    start = 0

    while start < total:

        end = min(start + 24 - (t0 + timedelta(hours=start)).hour, total)

        windows.append((start, end))

        start = end

    return windows

# ============================================================

# 4. HOOFDPROGRAMMA

# ============================================================

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)

    t0 = now.replace(minute=0, second=0, microsecond=0)

    t0 -= timedelta(hours=t0.hour % HYBRID_STEP)

    windows = day_windows(t0, t0 + timedelta(hours=HYBRID_HOURS))

    t_end = t0 + timedelta(hours=windows[-1][1])

    print(f"\nHYBRID vanaf {t0:%Y-%m-%d %H} UTC; dagvensters (uur): {windows}")

    models = {

        name: load_module(f"snow_{name.lower().replace('-', '_')}", fn)

        for name, fn in MODEL_FILES.items()

    }

    # Aparte kopie voor kaarten en sneeuwdekmodel, met Hybrid-instellingen.

    lib = load_module("snow_hybrid_lib", MODEL_FILES["ICON-CH2"])

    lib.OUTPUT_DIR = OUTPUT_DIR

    lib.MODEL_LABEL = LABEL

    lib.FILE_PREFIX = FILE_PREFIX

    lib.STEP_HOURS = HYBRID_STEP
    # Calibrate interactive camera positions without touching the PNGs.
    enable_webcam_georeferencing(lib)
    # Optional downscaling of the entire common grid for the CI runner.
    # 0.004 is the original scientific setting; 0.008 is the economical pilot.
    # All six models MUST share the same target GRID_STEP.
    requested_step = os.environ.get("HYBRID_GRID_STEP", "").strip()
    if requested_step:
        grid_step = float(requested_step)
        if not 0.002 <= grid_step <= 0.04:
            raise ValueError("HYBRID_GRID_STEP must be between 0.002 and 0.04 degrees")
        for module in [*models.values(), lib]:
            if isinstance(module.GRID_STEP, dict):
                module.GRID_STEP[AREA] = grid_step
            else:
                module.GRID_STEP = grid_step
        print(f"Cloud grid resolution: {grid_step} degrees")

    cloud_dpi = os.environ.get("HYBRID_DPI", "").strip()
    if cloud_dpi:
        lib.DPI = int(cloud_dpi)
        print(f"Cloud image DPI: {lib.DPI}")

    # The six original scripts require a logo image for every map.
    # For a GitHub runner, the wrapper prepares it in the temporary HOME.
    fallback_logo = Path.home() / "logo-sneeuwhoogte.png"
    if fallback_logo.is_file():
        lib.LOGO_PATH = fallback_logo
    else:
        raise FileNotFoundError(f"Required map logo not found: {fallback_logo}")

    # Relief can be disabled when the upstream Esri tile endpoint is unavailable.
    if os.environ.get("HYBRID_ESRI_RELIEF", "1") == "0":
        lib.USE_ESRI_RELIEF = False


    with requests.Session() as session:

        session.headers.update({"User-Agent": "SneeuwhoogteNL-hybrid/1.0"})

        extent = lib.AREA_LIMITS[AREA]

        xf, yf, dem = lib.load_dem(session, extent)

        shape = dem.shape

        streams = {}

        for name, module in models.items():

            try:

                run = (

                    module.newest_run(session, 96)

                    if name == "ECMWF" else module.newest_run(session)

                )
                max_age = getattr(module, "MAX_RUN_AGE_H", 12)
                age_h = (now - run).total_seconds() / 3600
                if age_h > max_age:
                    print(f"WAARSCHUWING: {name} run {run:%Y-%m-%d %H} UTC is {age_h:.1f} h oud; overgeslagen")
                    continue

                streams[name] = ModelStream(

                    name, module, session, run, t0, t_end, xf, yf

                )

            except Exception as exc:

                print(f"WAARSCHUWING: {name} niet beschikbaar ({exc})")

        if not streams:

            raise RuntimeError("Geen enkel model beschikbaar")
        lib.MODEL_LABEL = f"Hybride sneeuwverwachting ({len(streams)} modelruns)"
        print("Gebruikte modelruns:")
        for model_name, stream in streams.items():
            print(f"  {model_name}: {stream.run:%Y-%m-%d %H:%M UTC}")

        # Doorgang 1: per blok het gewogen gemiddelde, plus per dag en

        # per model de dagsom op steekproefpixels voor de PM-verdeling.

        sample = np.arange(0, dem.size, PM_SAMPLE)

        blocks = {}

        day_samples = {w: {} for w in windows}

        day_blocks = {w: {} for w in windows}

        covered = np.zeros(shape, dtype=bool)

        for k in range(windows[-1][1] // HYBRID_STEP):

            a, b = k * HYBRID_STEP, (k + 1) * HYBRID_STEP

            t_a, t_b = t0 + timedelta(hours=a), t0 + timedelta(hours=b)

            bucket = 0 if (t_a + (t_b - t_a) / 2 - now) < timedelta(hours=24) else 1

            parts = {}

            for name, stream in streams.items():

                acc = stream.interval(t_a, t_b)

                if acc is not None:

                    parts[name] = acc

            if len(parts) < MIN_MODELS_PER_BLOCK:

                raise RuntimeError(f"Slechts {len(parts)} modellen beschikbaar in +{a}-{b} uur (minimaal {MIN_MODELS_PER_BLOCK})")

            weights_now = {n: WEIGHTS[n][bucket] for n in parts}

            print(

                f"\n=== Hybrid +{a:02d}-{b:02d} uur: "

                + ", ".join(f"{n} {w:.2f}" for n, w in weights_now.items())

            )

            blocks[(a, b)] = hybrid_block(parts, weights_now, shape)

            covered |= blocks[(a, b)]["mask"]

            window = next(w for w in windows if w[0] <= a < w[1])

            for name, acc in parts.items():

                tot = (acc["snow_mm"] + acc["rain_mm"]).ravel()[sample]

                valid = acc["valid"].ravel()[sample]

                if name in day_samples[window]:

                    day_samples[window][name][0] += tot

                    day_samples[window][name][1] += weights_now[name]

                else:

                    day_samples[window][name] = [tot, weights_now[name], valid]

                day_blocks[window][name] = day_blocks[window].get(name, 0) + 1

        # Doorgang 2: per dag probability matching op de dagsom; elk blok

        # van die dag krijgt dezelfde schaalfactor. Daarna sneeuw, regen

        # en het sneeuwdekmodel per blok.

        daily_snow = {w: np.zeros(shape) for w in windows}

        daily_rain = {w: np.zeros(shape) for w in windows}

        on_ground = {}

        swe = np.zeros(shape)

        depth = np.zeros(shape)

        for window in windows:

            in_day = [k for k in blocks if window[0] <= k[0] < window[1]]

            mean_day = sum(blocks[k]["precip"].astype(float) for k in in_day)

            mask = np.logical_or.reduce([blocks[k]["mask"] for k in in_day])

            factor = np.ones(shape)

            if USE_PM_MEAN:

                # Alleen modellen die de hele dag dekken tellen mee.

                full = {

                    n: v for n, v in day_samples[window].items()

                    if day_blocks[window][n] == len(in_day)

                }

                if not full:
                    raise RuntimeError(f"No complete model for probability matching in window {window}")

                wsum = sum(v[1] / len(in_day) * v[2] for v in full.values())

                wsum = np.where(wsum > 0, wsum, np.inf)

                pm = pm_mean(

                    mean_day,

                    [v[0] for v in full.values()],

                    [v[1] / len(in_day) * v[2] / wsum for v in full.values()],

                    mask

                )

                factor = np.where(

                    mean_day > 1e-6, pm / np.maximum(mean_day, 1e-6), 0.0

                )

            for key in sorted(in_day):

                blk = blocks.pop(key)

                precip = blk["precip"] * factor

                snow_mm = precip * blk["frac"]

                rain_mm = precip - snow_mm

                snow_cm = snow_mm * blk["cm_per_mm"]

                daily_snow[window] += snow_cm

                daily_rain[window] += rain_mm

                swe, depth = lib.snowpack_step(

                    swe, depth, snow_mm, snow_cm, rain_mm, blk["temp"]

                )

            on_ground[window] = depth.copy()

    land = covered & np.isfinite(dem)

    maps = [

        (start, end, daily_snow[(start, end)], daily_rain[(start, end)], False)

        for start, end in windows

    ]

    snow_sum = np.zeros(shape)

    rain_sum = np.zeros(shape)

    for k, window in enumerate(windows):

        snow_sum = snow_sum + daily_snow[window]

        rain_sum = rain_sum + daily_rain[window]

        if k > 0:

            maps.append((0, window[1], snow_sum, rain_sum, True))

    maps += [

        (0, end, on_ground[(start, end)], None, False)

        for start, end in windows

    ]

    print("**\n**Gebruikte runs: " + ", ".join(

        f"{n} {s.run:%d/%m %H}z" for n, s in streams.items()

    ))

    count = 0
    map_errors = []

    for start, end, snow, rain, total in maps:

        snow = np.where(land, snow, np.nan)

        if rain is not None:

            rain = np.where(land, rain, np.nan)

        kind = (

            "op de grond" if rain is None

            else "cumulatief" if total else "sneeuwval"

        )

        print(f"+{start:02d}-{end:02d} {kind}: max {np.nanmax(snow):.0f} cm")

        try:

            lib.plot_snowfall(

                snow, xf, yf, extent, AREA, t0, start, end, rain, total

            )

            count += 1

        except Exception as exc:

            map_errors.append(f"{kind} +{start}-{end}: {exc}")
            print(f"FOUT {kind} +{start}-{end}: {exc}")

    print(f"\nKlaar: {count} kaarten. Opslag: {OUTPUT_DIR}")

    if map_errors:
        raise RuntimeError(f"{len(map_errors)} requested map(s) failed: " + "; ".join(map_errors[:3]))

    if count < len(maps):

        raise RuntimeError(f"Expected {len(maps)} primary snowfall maps, but only {count} were created")
    import json
    metadata = {
        "forecast_start_utc": t0.isoformat(),
        "forecast_end_utc": t_end.isoformat(),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "model_runs_utc": {n: stream.run.isoformat() for n, stream in streams.items()},
        "map_count": count,
        "grid_step_deg": requested_step or "0.004",
    }
    (OUTPUT_DIR / "forecast_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

if __name__ == "__main__":

    main()
