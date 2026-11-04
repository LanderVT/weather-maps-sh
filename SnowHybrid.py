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

5. Onzekerheid uit de modellen samen (een 'poor man's ensemble'):

   - kans op >= PROB_THRESHOLDS_CM cumulatieve sneeuw (vanaf nu tot het

     einde van elke cumulatieve periode): gewogen aandeel van de modellen

     dat die hoeveelheid haalt;

   - modelonenigheid: gewogen standaardafwijking / gemiddelde van de

     dagsneeuw, alleen waar gemiddeld >= SPREAD_MIN_CM valt.

Kaarten (map outputmaps/HYBRID): sneeuwval per dag, cumulatief, nieuwe

sneeuw op de grond, kans per drempel (cumulatief) en modelonenigheid per dag.

Python-pakketten: zie de SnowDays_*.py-scripts.

Gebruik: python SnowHybrid.py

"""

import importlib.util
import os

from datetime import datetime, timedelta, timezone

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib.colors import BoundaryNorm, ListedColormap

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

HYBRID_STEP = 6          # uur; ECMWF levert 6-uurlijks

AREA = "alps"

USE_PM_MEAN = True

PM_SAMPLE = 5           # elke n-de pixel voor de verdeling (sneller)

# Kanskaarten (cumulatief) en modelonenigheid (per dag).

PROB_THRESHOLDS_CM = (5, 20)

PROB_LEVELS = [10, 30, 50, 70, 90, 100]

PROB_COLORS = ["#c6dbef", "#6baed6", "#2171b5", "#6a51a3", "#3f007d"]

SPREAD_MIN_CM = 2.0

SPREAD_LEVELS = [0, 0.25, 0.5, 1.0, 10]

SPREAD_COLORS = ["#4daf4a", "#ffd92f", "#fd8d3c", "#e31a1c"]

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

    """Levert per Hybrid-blok de opgetelde velden van een model (of None)."""

    def __init__(self, name, module, session, run, t0, t_end, xf, yf):

        self.name = name

        self.run = run

        self.xf, self.yf = xf, yf

        self.pending = None

        step = module.STEP_HOURS

        max_lead = getattr(module, "MAX_LEAD", 10 ** 6)

        start = max(0, int((t0 - run).total_seconds() // 3600))

        start += -start % step

        end = min(int((t_end - run).total_seconds() // 3600), max_lead)

        end -= end % step

        self.gen = (

            module.iterate_steps(

                session, run, list(range(start, end + 1, step)), AREA

            )

            if end > start else iter(())

        )

    def interval(self, t_start, t_stop):

        acc = None

        covered = 0.0

        while True:

            try:

                item = self.pending or next(self.gen, None)

            except Exception as exc:

                # Een haperend model mag Hybrid niet stoppen.

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

            fields = {

                k: embed(item[k], item["xf"], item["yf"],

                         self.xf, self.yf, 0.0)

                for k in ("snow_mm", "rain_mm", "snow_cm", "temp")

            }

            hours = (e - s).total_seconds() / 3600

            if acc is None:

                acc = {k: np.zeros_like(v) for k, v in fields.items()}

                acc["valid"] = embed(

                    item["land"], item["xf"], item["yf"],

                    self.xf, self.yf, False

                )

            for k in ("snow_mm", "rain_mm", "snow_cm"):

                acc[k] += fields[k]

            acc["temp"] += fields["temp"] * hours

            covered += hours

            if e >= t_stop:

                break

        block = (t_stop - t_start).total_seconds() / 3600

        if acc is None or not np.isclose(covered, block):

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

def plot_field(lib, data, xf, yf, extent, levels, colors, title, label, name):

    """Kaart met een eigen kleurenschaal (kans, onenigheid)."""

    fig, ax, proj = lib.common_axes(extent)

    cmap = ListedColormap(colors)

    cmap.set_bad((0, 0, 0, 0))

    cf = ax.contourf(

        xf, yf, np.ma.masked_invalid(data),

        levels=levels,

        cmap=cmap,

        norm=BoundaryNorm(levels, cmap.N),

        transform=proj,

        alpha=0.85,

        zorder=3

    )

    lib.finish_map(

        fig, ax, cf, title, label, levels, OUTPUT_DIR / name, extent, AREA

    )

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

        model_day = {w: {} for w in windows}  # dagsneeuw (cm) per model

        model_valid = {w: {} for w in windows}  # waar dat model data heeft

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

                snow = acc["snow_cm"].astype(np.float32)

                if name in model_day[window]:

                    model_day[window][name] += snow

                else:

                    model_day[window][name] = snow

                    model_valid[window][name] = acc["valid"]

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

    # Modelonenigheid per dag en kans cumulatief, uit de modellen die de

    # hele dag (resp. alle dagen tot dan) dekken (ruwe sneeuw per model,

    # dus zonder probability matching).

    n_extra = 0

    # Per model: cumulatieve sneeuw, som van de daggewichten, geldigheid.

    cum_snow, cum_w, cum_valid = {}, {}, {}

    for k, window in enumerate(windows):

        n_blocks = (window[1] - window[0]) // HYBRID_STEP

        full = [

            n for n in model_day[window]

            if day_blocks[window][n] == n_blocks

        ]

        # Een model blijft alleen in de cumulatieve kans als het elke dag

        # tot nu toe volledig dekt.

        for n in list(cum_snow):

            if n not in full:

                del cum_snow[n], cum_w[n], cum_valid[n]

        for n in full:

            if k == 0:

                cum_snow[n] = model_day[window][n].astype(float)

                cum_w[n] = day_samples[window][n][1] / n_blocks

                cum_valid[n] = model_valid[window][n].copy()

            elif n in cum_snow:

                cum_snow[n] = cum_snow[n] + model_day[window][n]

                cum_w[n] += day_samples[window][n][1] / n_blocks

                cum_valid[n] = cum_valid[n] & model_valid[window][n]

        if k > 0 and len(cum_snow) >= 2:

            Wc = {n: cum_w[n] / (k + 1) * cum_valid[n] for n in cum_snow}

            wcsum = sum(Wc.values())

            okc = land & (wcsum > 0)

            safec = np.where(okc, wcsum, 1.0)

            end_dt = (t0 + timedelta(hours=window[1])).astimezone(lib.LOCAL_TZ)

            cperiod = f"{lib.fmt_date(t0.astimezone(lib.LOCAL_TZ))} – {lib.fmt_date(end_dt)}"

            ctag = f"{t0:%Y%m%d_%H}z_f000-{window[1]:03d}"

            cmodels_txt = f"{len(cum_snow)} modellen"

            for thr in PROB_THRESHOLDS_CM:

                prob = 100 * sum(

                    Wc[n] * (cum_snow[n] >= thr) for n in cum_snow

                ) / safec

                prob = np.where(okc & (prob >= PROB_LEVELS[0]), prob, np.nan)

                try:

                    plot_field(

                        lib, prob, xf, yf, extent, PROB_LEVELS, PROB_COLORS,

                        f"Kans op ≥ {thr} cm cumulatief: {cperiod}",

                        f"Kans (%) volgens {cmodels_txt}, gewogen — gedownscaled",

                        f"{FILE_PREFIX}_kans_cumulatief_{thr}cm_{AREA}_{ctag}.png"

                    )

                    n_extra += 1

                except Exception as exc:

                    map_errors.append(f"kans cumulatief {thr} cm {window}: {exc}")
                    print(f"FOUT kans cumulatief {thr} cm {window}: {exc}")

        elif k > 0:

            print(f"Te weinig modellen voor cumulatieve kans tot +{window[1]}")

        if len(full) < 2:

            print(f"Te weinig modellen voor onzekerheid {window}")

            continue

        W = {

            n: day_samples[window][n][1] / n_blocks * model_valid[window][n]

            for n in full

        }

        wsum = sum(W.values())

        ok = land & (wsum > 0)

        safe = np.where(ok, wsum, 1.0)

        mean = sum(W[n] * model_day[window][n] for n in full) / safe

        var = sum(

            W[n] * (model_day[window][n] - mean) ** 2 for n in full

        ) / safe

        start_dt = (t0 + timedelta(hours=window[0])).astimezone(lib.LOCAL_TZ)

        end_dt = (t0 + timedelta(hours=window[1])).astimezone(lib.LOCAL_TZ)

        period = f"{lib.fmt_date(start_dt)} – {lib.fmt_date(end_dt)}"

        tag = f"{t0:%Y%m%d_%H}z_f{window[0]:03d}-{window[1]:03d}"

        models_txt = f"{len(full)} modellen"

        spread = np.where(

            ok & (mean >= SPREAD_MIN_CM),

            np.sqrt(var) / np.maximum(mean, 1e-6),

            np.nan

        )

        try:

            plot_field(

                lib, spread, xf, yf, extent, SPREAD_LEVELS, SPREAD_COLORS,

                f"Modelonenigheid: {period}",

                "Spreiding / gemiddelde dagsneeuw (< 0,25 eens, "

                f"> 1 sterk oneens) — {models_txt}",

                f"{FILE_PREFIX}_onenigheid_{AREA}_{tag}.png"

            )

            n_extra += 1

        except Exception as exc:

            map_errors.append(f"onenigheid {window}: {exc}")
            print(f"FOUT onenigheid {window}: {exc}")

        print(

            f"{period}: {len(full)} modellen ({', '.join(full)}); "

            f"mediane onenigheid {np.nanmedian(spread):.2f}"

        )

    count += n_extra

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
