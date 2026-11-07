
"""
ICON-D2 | Alle Sneeuwhoogte.nl-kaarten in een script

1. Vorstgrens (HZEROCL): +0, +6, ..., +48 uur
2. Sneeuwgrens (SNOWLMT): +0, +6, ..., +48 uur; grijs boven sneeuwgrens
3. Verse sneeuw (SNOW_GSP + SNOW_CON): lokale dagvensters en cumulatief
   tot +48 uur, met aftrek van reeds gevallen sneeuw sinds de modelrun

Maakt automatisch voor ELKE modelrun twee gebieden:
  FULL: de grootste geldige uitsnede inclusief Benelux
  ALPS: een afzonderlijke, grotere Alpenkaart.

Stadsnamen krijgen automatische plaatsing zonder label-overlap.
Steden zonder geschikte labelpositie worden niet getekend.

Python-pakketten: pip install numpy matplotlib xarray cfgrib eccodes requests cartopy scipy

Gebruik: python ICON_D2_All_Maps.py
"""

import bz2
import re
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.transforms import Bbox
import numpy as np
import requests
import xarray as xr
from scipy.interpolate import RectBivariateSpline, RegularGridInterpolator
from scipy.ndimage import binary_erosion

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.img_tiles as cimgt


# ============================================================
# 1. INSTELLINGEN
# ============================================================

HOME = Path.home()
DOCUMENTS_DIR = HOME / "Documents" / "SneeuwhoogteNL"
DATA_DIR = DOCUMENTS_DIR / "data"
OUTPUT_DIR = DOCUMENTS_DIR / "outputmaps"
TILE_CACHE_DIR = DATA_DIR / "esri_tiles"
LOGO_PATH = HOME / "logo-sneeuwhoogte.png"

DATA_URL = "https://opendata.dwd.de/weather/nwp/icon-d2/grib"
RUN_HOURS = tuple(range(0, 24, 3))
FORECAST_HOURS = range(0, 49, 6)

# Standalone default: beide gebieden. Cloud runner may select Alps only.
AREAS_TO_MAKE = ("full", "alps")

# Bovengrens van het gewenste venster. De werkelijke extent wordt
# automatisch beperkt tot een rechthoek met geldige ICON-D2-punten.
AREA_LIMITS = {
    "full": [-2.0, 21.0, 42.0, 56.0],
    "alps": [5.0, 16.8, 43.6, 49.0],
}

USE_ESRI_RELIEF = True
ESRI_ZOOM = 9
DPI = 400
LOCAL_TZ = ZoneInfo("Europe/Brussels")

# Grotere, beter leesbare labels en lagere dichtheid dan voorheen.
CITY_FONT = {"full": 7.5, "alps": 8.0}
MAX_CITY_LABELS = {"full": 45, "alps": 38}
MIN_CITY_DISTANCE_PX = {"full": 18, "alps": 16}
LABEL_GAP_PX = 4
SNOW_INTERP_STEP = {"full": 0.012, "alps": 0.008}

# 1 kg/m2 sneeuwwaterequivalent = 1 mm w.e.; bij 10:1 ongeveer
# 1 cm verse sneeuw. Dit houdt de numerieke schaal van SnowFresh aan.
SNOW_CM_PER_KGM2 = 1.0
SNOW_MIN_CM = 0.5

ALT_LEVELS = np.arange(0, 4501, 250)
ALT_COLORS = [
    "#3a063e", "#570d5d", "#9c1fa4", "#751d7c", "#570d3e",
    "#313363", "#123293", "#0000c4", "#0000f5", "#2254f5",
    "#4397f7", "#67c9fa", "#92fcfd", "#92fca3", "#92fc7a",
    "#92fc4d", "#cbf851", "#ffff56", "#ffff96", "#f9e96c",
    "#f7ce46", "#f19e38", "#ed702d", "#eb483f", "#eb4826",
    "#ea3323", "#d22d1f", "#a32015", "#8c1a11", "#621008",
    "#75144e", "#921c74", "#bb27c5", "#ea33f7", "#eb51f7",
    "#ef87f9",
]

SNOW_LEVELS = np.array([
    0, 1, 2, 3, 5, 7, 10, 15, 20, 25, 30, 40, 50,
    60, 70, 80, 90, 100, 125, 150, 175, 200, 250, 300, 400, 500
])

SNOW_COLORS = [
    "#e4e4f2", "#bbd6fb", "#85b8f9", "#5398f8", "#3980f7",
    "#2d69cc", "#13357b", "#428d31", "#61cc3d", "#63ec0b",
    "#fdf55b", "#e6dd49", "#df692a", "#ef8640", "#f2aa75",
    "#e65b7a", "#e33b58", "#af2418", "#7d170e", "#5b0e7a",
    "#b224f2", "#ce6df7", "#e1a9fa", "#f6e7fd", "#d4d4d4",
]

DAYS = [
    "maandag", "dinsdag", "woensdag", "donderdag",
    "vrijdag", "zaterdag", "zondag"
]

MONTHS = [
    "januari", "februari", "maart", "april", "mei", "juni",
    "juli", "augustus", "september", "oktober", "november", "december"
]

# Complete stedenlijst, met extra dekking rond Parijs en Noord-Duitsland.
# De labels worden later automatisch verdeeld; niet alle steden hoeven zichtbaar te zijn.
CITIES = [
    # BELGIE
    ("Brussel", 4.35, 50.85), ("Antwerpen", 4.40, 51.22),
    ("Gent", 3.72, 51.05), ("Brugge", 3.22, 51.21),
    ("Luik", 5.58, 50.63), ("Namen", 4.87, 50.47),
    ("Hasselt", 5.34, 50.93), ("Charleroi", 4.44, 50.41),
    ("Leuven", 4.70, 50.88), ("Kortrijk", 3.26, 50.83),

    # NEDERLAND
    ("Amsterdam", 4.90, 52.37), ("Rotterdam", 4.48, 51.92),
    ("Den Haag", 4.30, 52.08), ("Utrecht", 5.12, 52.09),
    ("Eindhoven", 5.48, 51.44), ("Maastricht", 5.69, 50.85),
    ("Groningen", 6.57, 53.22), ("Arnhem", 5.92, 51.98),
    ("Nijmegen", 5.87, 51.84), ("Zwolle", 6.09, 52.52),

    # LUXEMBURG
    ("Luxemburg", 6.13, 49.61),

    # FRANKRIJK: ook Parijs en Noord-/Midden-Frankrijk
    ("Parijs", 2.35, 48.86), ("Rouen", 1.10, 49.44),
    ("Orléans", 1.91, 47.90), ("Chartres", 1.49, 48.45),
    ("Beauvais", 2.08, 49.43), ("Compiègne", 2.82, 49.42),
    ("Auxerre", 3.57, 47.80), ("Bourges", 2.40, 47.08),
    ("Tours", 0.69, 47.39), ("Le Mans", 0.20, 48.01),
    ("Caen", -0.37, 49.18),
    ("Grenoble", 5.72, 45.17), ("Lyon", 4.84, 45.76),
    ("Annecy", 6.13, 45.90), ("Chamonix", 6.87, 45.92),
    ("Straatsburg", 7.75, 48.58), ("Metz", 6.18, 49.12),
    ("Nancy", 6.18, 48.69), ("Reims", 4.03, 49.26),
    ("Troyes", 4.07, 48.30), ("Dijon", 5.04, 47.32),
    ("Besançon", 6.03, 47.24), ("Lille", 3.06, 50.63),
    ("Amiens", 2.30, 49.90),

    # MIDDEN- EN ZUIDWEST-FRANKRIJK
    ("Poitiers", 0.34, 46.58), ("Limoges", 1.26, 45.83),
    ("Clermont-Ferrand", 3.08, 45.78), ("Nevers", 3.16, 46.99),
    ("Périgueux", 0.72, 45.18), ("Brive-la-Gaillarde", 1.53, 45.16),
    ("Angoulême", 0.16, 45.65), ("Cahors", 1.44, 44.45),
    ("Moulins", 3.33, 46.57),

    # LONDEN EN ZUIDOOST-ENGELAND
    ("Londen", -0.13, 51.51), ("Cambridge", 0.12, 52.21),

    # WEST-POLEN (VOOR ZOVER BINNEN DE GELDIGE ICON-D2-EXTENT)
    ("Szczecin", 14.55, 53.43), ("Poznań", 16.93, 52.41),
    ("Wrocław", 17.04, 51.11), ("Zielona Góra", 15.51, 51.94),
    ("Gorzów", 15.23, 52.73), ("Legnica", 16.16, 51.21),

    # ZWITSERLAND
    ("Genève", 6.14, 46.20), ("Lausanne", 6.63, 46.52),
    ("Bern", 7.45, 46.95), ("Zürich", 8.54, 47.38),
    ("Basel", 7.59, 47.56), ("Chur", 9.53, 46.85),

    # NOORD-DUITSLAND en MIDDEN-DUITSLAND
    ("Hamburg", 9.99, 53.55), ("Bremen", 8.80, 53.08),
    ("Hannover", 9.73, 52.38), ("Berlijn", 13.40, 52.52),
    ("Kiel", 10.12, 54.32), ("Lübeck", 10.69, 53.87),
    ("Rostock", 12.14, 54.09), ("Schwerin", 11.41, 53.63),
    ("Magdeburg", 11.63, 52.13), ("Braunschweig", 10.53, 52.27),
    ("Leipzig", 12.37, 51.34), ("Halle", 11.97, 51.48),
    ("Potsdam", 13.06, 52.40), ("Münster", 7.63, 51.96),
    ("Osnabrück", 8.05, 52.28), ("Bielefeld", 8.53, 52.02),
    ("Göttingen", 9.93, 51.53), ("Erfurt", 11.03, 50.98),
    ("Chemnitz", 12.93, 50.83), ("Cottbus", 14.33, 51.76),
    ("Würzburg", 9.94, 49.79), ("Mainz", 8.25, 50.00),
    ("Koblenz", 7.59, 50.36),

    # DUITSLAND: bestaande steden
    ("Freiburg", 7.85, 47.99), ("Stuttgart", 9.18, 48.78),
    ("Karlsruhe", 8.40, 49.01), ("München", 11.58, 48.14),
    ("Augsburg", 10.90, 48.37), ("Nürnberg", 11.08, 49.45),
    ("Regensburg", 12.10, 49.02), ("Passau", 13.43, 48.57),
    ("Frankfurt", 8.68, 50.11), ("Keulen", 6.96, 50.94),
    ("Dresden", 13.74, 51.05), ("Düsseldorf", 6.78, 51.23),
    ("Dortmund", 7.47, 51.51), ("Aken", 6.08, 50.78),
    ("Kassel", 9.48, 51.31),

    # OOSTENRIJK
    ("Innsbruck", 11.40, 47.27), ("Salzburg", 13.05, 47.81),
    ("Linz", 14.29, 48.31), ("Wenen", 16.37, 48.21),
    ("Graz", 15.44, 47.07),

    # ITALIE
    ("Turijn", 7.69, 45.07), ("Aosta", 7.32, 45.74),
    ("Milaan", 9.19, 45.46), ("Genua", 8.93, 44.41),
    ("Bologna", 11.34, 44.50), ("Verona", 10.99, 45.44),
    ("Bolzano", 11.35, 46.50), ("Trento", 11.12, 46.07),
    ("Venetië", 12.33, 45.44), ("Trieste", 13.77, 45.65),

    # TSJECHIE
    ("Praag", 14.42, 50.08), ("Plzeň", 13.38, 49.75),
    ("Brno", 16.61, 49.20),

    # SLOVENIE / KROATIE
    ("Ljubljana", 14.51, 46.06), ("Zagreb", 15.98, 45.81),
    ("Rijeka", 14.44, 45.33),
]


# ============================================================
# 3. DWD DATA, HERGEBRUIKT VOOR ALLE DRIE DE PRODUCTEN
# ============================================================

class EsriWorldShadedRelief(cimgt.GoogleTiles):

    def _image_url(self, tile):
        x, y, z = tile
        return (
            "https://server.arcgisonline.com/ArcGIS/rest/services/"
            f"World_Shaded_Relief/MapServer/tile/{z}/{y}/{x}.jpg"
        )


def dwd_url(run, lead, parameter):

    name = (
        "icon-d2_germany_regular-lat-lon_single-level_"
        f"{run:%Y%m%d%H}_{lead:03d}_2d_{parameter}.grib2.bz2"
    )

    return f"{DATA_URL}/{run:%H}/{parameter}/{name}", name


def remote_exists(session, url):

    try:
        r = session.head(url, timeout=12, allow_redirects=True)

        if r.status_code == 200:
            return True

        if r.status_code in (403, 405):
            with session.get(url, timeout=20, stream=True) as rr:
                return rr.status_code == 200

    except requests.RequestException:
        pass

    return False


def newest_common_run(session):
    """Kies een run met +48 voor vorstgrens, sneeuwgrens en sneeuwval."""

    now = datetime.now(timezone.utc)

    required = [
        ("hzerocl", 0), ("hzerocl", 48),
        ("snowlmt", 0), ("snowlmt", 48),
        ("snow_gsp", 0), ("snow_gsp", 24), ("snow_gsp", 48),
    ]

    for back in range(4):

        d = (now - timedelta(days=back)).date()

        for h in sorted(RUN_HOURS, reverse=True):

            run = datetime(
                d.year, d.month, d.day, h,
                tzinfo=timezone.utc
            )

            if run > now:
                continue

            if all(
                remote_exists(session, dwd_url(run, step, par)[0])
                for par, step in required
            ):

                print(
                    f"Gemeenschappelijke ICON-D2-run: "
                    f"{run:%Y-%m-%d %H:%M} UTC"
                )

                return run

    raise RuntimeError(
        "Geen run gevonden met +48 uur voor alle drie producten."
    )


def download_archive(
    session, url, name, parameter=None, run=None, lead=None
):

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = DATA_DIR / name

    if out.is_file() and out.stat().st_size > 1000:
        print("In cache:", name)
        return out

    # De oude SnowFresh-cache kan direct opnieuw gebruikt worden.
    if parameter in ("snow_gsp", "snow_con") and run is not None:

        legacy = DATA_DIR / (
            f"ICOND2_{parameter}_{run:%Y%m%d}_{lead:03d}_{run:%H}.bz2"
        )

        if legacy.is_file() and legacy.stat().st_size > 1000:
            print("In oude SnowFresh-cache:", legacy.name)
            return legacy

    print("Downloading:", name)
    partial = out.with_suffix(out.suffix + ".part")

    try:

        with session.get(
            url, stream=True, timeout=(15, 120)
        ) as r:

            r.raise_for_status()

            with partial.open("wb") as f:

                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)

        if partial.stat().st_size < 1000:
            raise RuntimeError(f"Leeg bestand: {name}")

        partial.replace(out)

    finally:
        partial.unlink(missing_ok=True)

    return out


def read_grib(archive, target_names, lead=None):
    """Lees 2D veld en selecteer de juiste forecaststap (indien aanwezig)."""

    extracted = archive.with_suffix("")

    try:

        with bz2.open(archive, "rb") as src:
            with extracted.open("wb") as dst:
                shutil.copyfileobj(src, dst)

        with xr.open_dataset(
            extracted,
            engine="cfgrib",
            backend_kwargs={"indexpath": ""}
        ) as ds:

            names = {name.lower() for name in target_names}

            vars_found = [
                k for k in ds.data_vars
                if k.lower() in names
            ]

            if not vars_found:
                raise KeyError(
                    f"{target_names} niet gevonden; "
                    f"beschikbaar: {list(ds.data_vars)}"
                )

            f = ds[vars_found[0]]

            if "step" in f.dims:

                step_vals = np.asarray(f["step"].values)

                if np.issubdtype(
                    step_vals.dtype, np.timedelta64
                ):
                    hrs = (
                        step_vals.astype("timedelta64[s]").astype(float)
                        / 3600.0
                    )
                else:
                    hrs = step_vals.astype(float)

                matches = (
                    np.where(
                        np.isclose(
                            np.atleast_1d(hrs),
                            lead,
                            atol=0.01
                        )
                    )[0]
                    if lead is not None else []
                )

                if len(matches):
                    index = int(matches[-1])

                elif f.sizes["step"] == 1:
                    index = 0

                elif lead is None:
                    index = -1

                else:
                    raise ValueError(
                        f"Forecasttijd +{lead} ontbreekt: {hrs}"
                    )

                f = f.isel(step=index, drop=True)

            for dim in tuple(f.dims):

                if dim not in ("latitude", "longitude"):
                    f = f.isel({dim: -1}, drop=True)

            f = f.transpose("latitude", "longitude")

            x = np.asarray(f.longitude.values, dtype=float)
            y = np.asarray(f.latitude.values, dtype=float)
            z = np.asarray(f.values, dtype=float)
            unit = str(f.attrs.get("units", ""))

        if x.ndim != 1 or y.ndim != 1 or z.ndim != 2:
            raise ValueError(
                "Dit script verwacht de regular-lat-lon "
                "ICON-D2-bestanden met 1D lon/lat"
            )

        ix, iy = np.argsort(x), np.argsort(y)
        x, y, z = x[ix], y[iy], z[np.ix_(iy, ix)]

        z[~np.isfinite(z)] = np.nan

        return x, y, z, unit

    finally:
        extracted.unlink(missing_ok=True)


def load_field(session, run, lead, parameter, optional=False):

    url, name = dwd_url(run, lead, parameter)

    legacy = DATA_DIR / (
        f"ICOND2_{parameter}_{run:%Y%m%d}_{lead:03d}_{run:%H}.bz2"
    )

    cached = (DATA_DIR / name).is_file() or legacy.is_file()

    if optional and not cached and not remote_exists(session, url):

        print(
            f"WAARSCHUWING: {parameter} +{lead:03d} ontbreekt."
        )

        return None

    archive = download_archive(
        session, url, name, parameter, run, lead
    )

    aliases = {
        "hzerocl": ["hzerocl"],
        "snowlmt": ["snowlmt"],
        "snow_gsp": ["snow_gsp", "lsfwe"],
        "snow_con": ["snow_con", "csfwe"],
    }

    return read_grib(
        archive, aliases[parameter], lead
    )


def load_hsurf(session, run):

    pattern = (
        r'icon-d2_germany_regular-lat-lon_time-invariant_'
        r'[^"<> ]+\.grib2\.bz2'
    )

    hours = [run.hour] + [
        h for h in (0, 6, 12, 18)
        if h != run.hour
    ]

    for h in hours:

        listing = f"{DATA_URL}/{h:02d}/hsurf/"

        try:

            r = session.get(listing, timeout=25)

            if r.status_code != 200:
                continue

            matches = sorted(
                set(re.findall(pattern, r.text))
            )

            if not matches:
                continue

            archive = download_archive(
                session,
                listing + matches[-1],
                matches[-1]
            )

            return read_grib(archive, ["hsurf", "orog"])

        except (
            requests.RequestException,
            OSError,
            ValueError,
            KeyError
        ) as exc:

            print(f"HSURF {h:02d} UTC: {exc}")

    raise RuntimeError(
        "HSURF niet gevonden in de officiële DWD-mappen."
    )


def assert_same_grid(ref, other, label):

    if not (
        np.array_equal(ref[0], other[0])
        and np.array_equal(ref[1], other[1])
    ):

        raise ValueError(
            f"Rooster van {label} komt niet overeen met HZEROCL"
        )


# ============================================================
# 4. MAXIMALE EXTENT: GROOTSTE RECHTHOEK ZONDER MODEL-NaNs
# ============================================================

def largest_valid_rectangle(mask, x, y, anchors):
    """Grootste rechthoek van geldige native cellen, met opgegeven steden."""

    height = np.zeros(len(x), dtype=np.int32)
    best = None
    best_area = -1

    anchor_ix = [
        (
            int(np.argmin(abs(x - ax))),
            int(np.argmin(abs(y - ay)))
        )
        for ax, ay in anchors
    ]

    for r in range(len(y)):

        height = np.where(
            mask[r], height + 1, 0
        )

        stack = []

        for j in range(len(x) + 1):

            hj = int(height[j]) if j < len(x) else 0

            while stack and int(height[stack[-1]]) > hj:

                h = int(height[stack.pop()])
                left = stack[-1] + 1 if stack else 0
                right = j - 1
                top = r - h + 1

                if h < 5 or right - left < 5:
                    continue

                if any(
                    not (left <= ix <= right and top <= iy <= r)
                    for ix, iy in anchor_ix
                ):
                    continue

                area = (
                    (x[right] - x[left])
                    * (y[r] - y[top])
                )

                if area > best_area:

                    best_area = area
                    best = (left, right, top, r)

            stack.append(j)

    return best


def determine_extent(x, y, valid, area):

    w, e, s, n = AREA_LIMITS[area]
    xx, yy = np.meshgrid(x, y)

    within = (
        (xx >= w) & (xx <= e)
        & (yy >= s) & (yy <= n)
    )

    # Een kleine interne buffer voorkomt gekleurde randartefacten.
    mask = binary_erosion(
        valid & within,
        iterations=2
    )

    if area == "full":

        anchor_sets = [
            # Grootste prioriteit: Benelux EN een flink deel van de Alpen.
            [
                (3.22, 51.21),
                (6.57, 53.22),
                (5.72, 45.17),
                (11.40, 47.27)
            ],
            [
                (4.35, 50.85),
                (4.90, 52.37),
                (5.72, 45.17),
                (11.40, 47.27)
            ],
            [
                (4.35, 50.85),
                (4.90, 52.37),
                (7.45, 46.95)
            ],
            [
                (4.35, 50.85),
                (4.90, 52.37)
            ],
            [
                (4.35, 50.85)
            ],
        ]

    else:

        anchor_sets = [
            [
                (7.45, 46.95),
                (11.40, 47.27),
                (9.19, 45.46)
            ],
            [
                (7.45, 46.95),
                (11.40, 47.27)
            ],
        ]

    rect = None

    for anchors in anchor_sets:

        if all(
            w < lon < e and s < lat < n
            for lon, lat in anchors
        ):

            rect = largest_valid_rectangle(
                mask, x, y, anchors
            )

        if rect is not None:
            break

    if rect is None:

        print(
            "WAARSCHUWING: geen rechthoek met alle voorkeurssteden. "
            "Gebruik grootste geldige rechthoek."
        )

        rect = largest_valid_rectangle(
            mask, x, y, []
        )

    if rect is None:

        raise RuntimeError(
            "Kon geen ononderbroken geldig ICON-D2-gebied vinden"
        )

    left, right, top, bottom = rect

    extent = [
        float(x[left]),
        float(x[right]),
        float(y[top]),
        float(y[bottom])
    ]

    print(
        f"Gebied {area}: maximale geldige rechthoek "
        f"{np.round(extent, 3).tolist()}"
    )

    for city, cx, cy in [
        ("Brussel", 4.35, 50.85),
        ("Amsterdam", 4.90, 52.37)
    ]:

        if area == "full" and not (
            extent[0] <= cx <= extent[1]
            and extent[2] <= cy <= extent[3]
        ):

            print(
                f"LET OP: {city} ligt buiten deze "
                "overal geldige rechthoek!"
            )

    return extent


# ============================================================
# 5. ALGEMENE KAARTOPMAAK
# ============================================================

def fmt_date(dt):

    return (
        f"{DAYS[dt.weekday()].capitalize()} "
        f"{dt.day} {MONTHS[dt.month-1]} {dt:%H}u"
    )


def region_label(area):

    return (
        "Alpen"
        if area == "alps"
        else "Benelux en Alpen"
    )


def common_axes(extent):

    proj = ccrs.PlateCarree()

    fig = plt.figure(
        figsize=(13.5, 8.1),
        facecolor="white"
    )

    ax = fig.add_axes(
        [0.035, 0.16, 0.93, 0.75],
        projection=proj
    )

    ax.set_extent(extent, crs=proj)

    if USE_ESRI_RELIEF:

        TILE_CACHE_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        ax.add_image(
            EsriWorldShadedRelief(
                cache=str(TILE_CACHE_DIR)
            ),
            ESRI_ZOOM,
            zorder=0
        )

    else:

        ax.add_feature(
            cfeature.LAND,
            facecolor="#e7e4dd",
            zorder=0
        )

        ax.add_feature(
            cfeature.OCEAN,
            facecolor="#d4e6ef",
            zorder=0
        )

    return fig, ax, proj


# Eerst een evenwichtige spreiding over Frankrijk, de Benelux,
# Noord-/Midden-Duitsland en de Alpen. Niet langer 15 Benelux-steden
# vóór de eerste stad in Noord-Duitsland.
PRIORITY_CITIES = {
    "full": [
        "Parijs", "Hamburg", "Berlijn", "Brussel", "Amsterdam",
        "Londen", "Szczecin", "Limoges", "Wrocław", "Hannover",
        "Leipzig", "Bremen", "Clermont-Ferrand", "Poitiers", "Lyon",
        "Zürich", "Rouen", "Orléans", "Poznań", "Frankfurt",
        "Keulen", "Groningen", "Dresden", "München", "Praag",
        "Milaan", "Genève", "Wenen", "Innsbruck", "Grenoble",
        "Zagreb", "Stuttgart", "Luxemburg", "Rostock", "Nürnberg",
        "Salzburg", "Genua", "Graz", "Venetië", "Dijon",
        "Brno", "Kassel", "Périgueux", "Nevers",
    ],

    "alps": [
        "Genève", "Bern", "Zürich", "Basel", "Grenoble", "Lyon",
        "Innsbruck", "Salzburg", "München", "Milaan", "Turijn",
        "Bolzano", "Chur", "Trento", "Venetië", "Genua", "Bologna",
        "Stuttgart", "Freiburg", "Wenen", "Graz", "Ljubljana",
        "Zagreb", "Linz", "Chamonix",
    ],
}

BENELUX_NAMES = {
    "Brussel", "Antwerpen", "Gent", "Brugge", "Luik", "Namen",
    "Hasselt", "Charleroi", "Leuven", "Kortrijk", "Amsterdam",
    "Rotterdam", "Den Haag", "Utrecht", "Eindhoven", "Maastricht",
    "Groningen", "Arnhem", "Nijmegen", "Zwolle", "Luxemburg",
}

# Maximaal 8 Benelux-labels op een FULL-kaart; zo blijft plaats
# over voor het lege westen van Frankrijk en Noord-Duitsland.
BENELUX_LABEL_CAP = 8

GRID_COLS = {"full": 6, "alps": 5}
GRID_ROWS = {"full": 4, "alps": 3}
GRID_CELL_CAP = {"full": 3, "alps": 4}


def add_cities(fig, ax, extent, area, logo_artist):
    """Geografisch gespreide steden met gecontroleerde tekstvakken.

    De stip blijft op de juiste coördinaat. Alleen het label verschuift.
    Prioriteiten worden afgewisseld tussen de landsdelen en per
    kaartsegment geldt een maximum, zodat geen gebied alle labels opslokt.
    """

    fig.canvas.draw()

    renderer = fig.canvas.get_renderer()
    bounds = ax.get_window_extent(renderer)
    occupied = []

    if logo_artist is not None:

        logo_bbox = logo_artist.get_window_extent(renderer)

        if logo_bbox.width > 0 and logo_bbox.height > 0:
            occupied.append(
                logo_bbox.padded(7)
            )

    w, e, s, n = extent
    visible = []

    for name, lon, lat in CITIES:

        if not (w < lon < e and s < lat < n):
            continue

        px, py = ax.transData.transform((lon, lat))

        if (
            bounds.x0 + 10 < px < bounds.x1 - 10
            and bounds.y0 + 10 < py < bounds.y1 - 10
        ):

            visible.append(
                (name, lon, lat, px, py)
            )

    city_by_name = {
        city[0]: city
        for city in visible
    }

    priority = [
        city_by_name[name]
        for name in PRIORITY_CITIES[area]
        if name in city_by_name
    ]

    remaining = [
        city for city in visible
        if city[0] not in PRIORITY_CITIES[area]
    ]

    chosen = []
    markers = []
    cell_counts = {}
    benelux_count = 0

    # Bescherm de geografische spreiding, ook wanneer veel meer steden
    # in de Benelux beschikbaar zijn dan in andere regio's.
    def cell_for(city):

        px, py = city[3], city[4]

        col = min(
            GRID_COLS[area] - 1,
            max(
                0,
                int(
                    GRID_COLS[area] *
                    (px - bounds.x0) / bounds.width
                )
            )
        )

        row = min(
            GRID_ROWS[area] - 1,
            max(
                0,
                int(
                    GRID_ROWS[area] *
                    (py - bounds.y0) / bounds.height
                )
            )
        )

        return col, row

    label_positions = [
        (5, 4, "left", "bottom"),
        (-5, 4, "right", "bottom"),
        (5, -4, "left", "top"),
        (-5, -4, "right", "top"),
        (0, 8, "center", "bottom"),
        (0, -8, "center", "top"),
        (9, 0, "left", "center"),
        (-9, 0, "right", "center"),
        (11, 7, "left", "bottom"),
        (-11, 7, "right", "bottom"),
        (11, -7, "left", "top"),
        (-11, -7, "right", "top"),
    ]

    hints = {
        "Brussel": (-5, -4, "right", "top"),
        "Amsterdam": (5, 4, "left", "bottom"),
        "Parijs": (5, 4, "left", "bottom"),
        "Londen": (5, 4, "left", "bottom"),
        "Wrocław": (-5, 4, "right", "bottom"),
        "Poznań": (-5, 4, "right", "bottom"),
        "Hamburg": (5, 4, "left", "bottom"),
        "Berlijn": (5, 4, "left", "bottom"),
    }

    def try_city(city):

        nonlocal benelux_count

        name, lon, lat, px, py = city
        cell = cell_for(city)

        if cell_counts.get(
            cell, 0
        ) >= GRID_CELL_CAP[area]:
            return False

        if (
            area == "full"
            and name in BENELUX_NAMES
            and benelux_count >= BENELUX_LABEL_CAP
        ):
            return False

        if any(
            np.hypot(px - mx, py - my) <
            MIN_CITY_DISTANCE_PX[area]
            for mx, my in markers
        ):
            return False

        marker_box = Bbox.from_extents(
            px - 4,
            py - 4,
            px + 4,
            py + 4
        )

        if any(
            marker_box.overlaps(b)
            for b in occupied
        ):
            return False

        options = list(label_positions)

        if name in hints:
            options = [
                hints[name]
            ] + [
                p for p in options
                if p != hints[name]
            ]

        for dx, dy, halign, valign in options:

            txt = ax.annotate(
                name,
                (lon, lat),
                xycoords="data",
                xytext=(dx, dy),
                textcoords="offset points",
                ha=halign,
                va=valign,
                fontsize=CITY_FONT[area],
                color="#101010",
                zorder=41,
                annotation_clip=False,
                path_effects=[
                    pe.withStroke(
                        linewidth=2.2,
                        foreground="white"
                    )
                ]
            )

            box = txt.get_window_extent(renderer)

            padded = box.padded(
                LABEL_GAP_PX
            )

            inside = (
                padded.x0 > bounds.x0 + 5
                and padded.y0 > bounds.y0 + 5
                and padded.x1 < bounds.x1 - 5
                and padded.y1 < bounds.y1 - 5
            )

            if not inside or any(
                padded.overlaps(b)
                for b in occupied
            ):
                txt.remove()
                continue

            ax.plot(
                lon,
                lat,
                "o",
                markersize=2.8,
                markerfacecolor="black",
                markeredgecolor="white",
                markeredgewidth=0.65,
                transform=ccrs.PlateCarree(),
                zorder=42
            )

            occupied.append(padded)
            occupied.append(marker_box)

            markers.append((px, py))
            chosen.append(name)

            cell_counts[cell] = (
                cell_counts.get(cell, 0) + 1
            )

            if name in BENELUX_NAMES:
                benelux_count += 1

            return True

        return False

    # Eerste fase: belangrijke steden, verspreid over het volledige domein.
    for city in priority:

        if len(chosen) >= MAX_CITY_LABELS[area]:
            break

        try_city(city)

    # Tweede fase: zoek telkens de stad in een minst gevuld kaartvak;
    # daarbinnen heeft grotere afstand tot bestaande steden voorrang.
    while remaining and len(chosen) < MAX_CITY_LABELS[area]:

        candidates = [
            city for city in remaining
            if cell_counts.get(
                cell_for(city), 0
            ) < GRID_CELL_CAP[area]
            and not (
                area == "full"
                and city[0] in BENELUX_NAMES
                and benelux_count >= BENELUX_LABEL_CAP
            )
        ]

        if not candidates:
            break

        def score(city):

            cell_occupancy = cell_counts.get(
                cell_for(city), 0
            )

            dist = (
                min(
                    np.hypot(
                        city[3] - mx,
                        city[4] - my
                    )
                    for mx, my in markers
                )
                if markers else 1e6
            )

            return -cell_occupancy, dist

        city = max(
            candidates,
            key=score
        )

        remaining.remove(city)
        try_city(city)

    print(
        f"Steden {area}: {len(chosen)} labels, "
        f"Benelux: {benelux_count}"
    )

    print(
        "  Getoond:", ", ".join(chosen)
    )

    return chosen


def finish_map(
    fig, ax, cf, title, label, levels,
    filepath, extent, area, tick_step=None
):

    proj = ccrs.PlateCarree()

    ax.add_feature(
        cfeature.BORDERS,
        linewidth=0.65,
        edgecolor="#383838",
        zorder=25
    )

    ax.add_feature(
        cfeature.COASTLINE,
        linewidth=0.65,
        edgecolor="#383838",
        zorder=25
    )

    if not LOGO_PATH.is_file():
        raise FileNotFoundError(
            f"Logo niet gevonden: {LOGO_PATH}"
        )

    img = plt.imread(LOGO_PATH)

    h, w = img.shape[:2]

    zoom = min(
        140.0 / w,
        45.0 / h
    )

    logo = AnnotationBbox(
        OffsetImage(
            img,
            zoom=zoom,
            resample=True
        ),
        (0.012, 0.985),
        xycoords=ax.transAxes,
        box_alignment=(0, 1),
        frameon=False,
        pad=0,
        zorder=200
    )

    ax.add_artist(logo)

    ax.set_title(
        title,
        fontsize=19,
        fontweight="bold",
        pad=10
    )

    add_cities(
        fig, ax, extent, area, logo
    )

    pos = ax.get_position()

    cax = fig.add_axes([
        pos.x0,
        pos.y0 - 0.015 - 0.035,
        pos.width,
        0.035
    ])

    cb = fig.colorbar(
        cf,
        cax=cax,
        orientation="horizontal"
    )

    cb.set_label(
        label,
        fontsize=14,
        labelpad=5
    )

    cb.set_ticks(
        levels
        if tick_step is None
        else np.arange(
            levels[0],
            levels[-1] + 1,
            tick_step
        )
    )

    cb.ax.tick_params(
        labelsize=9 if tick_step is None else 10
    )

    cb.outline.set_linewidth(0.8)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    fig.savefig(
        filepath,
        dpi=DPI,
        bbox_inches="tight",
        pad_inches=0.06,
        facecolor="white"
    )

    plt.close(fig)

    print(
        "Opgeslagen:", filepath.name
    )


def altitude_cmap(name):

    cmap = LinearSegmentedColormap.from_list(
        name,
        ALT_COLORS,
        N=len(ALT_LEVELS) - 1
    )

    cmap.set_bad((0, 0, 0, 0))
    cmap.set_under(ALT_COLORS[0])
    cmap.set_over(ALT_COLORS[-1])

    return cmap, BoundaryNorm(
        ALT_LEVELS, cmap.N
    )


# ============================================================
# 6. VORSTGRENS & SNEEUWGRENS
# ============================================================

def no_freezing_level_above_terrain(freezing_level, terrain, tolerance_m=10.0):
    """Markeer ICON HZEROCL-vulwaarden als er geen 0 C-isotherm is.

    DWD zet HZEROCL gelijk aan HSURF wanneer de VOLLEDIGE luchtkolom
    onder nul is. De hoge waarden op bergtoppen betekenen dan niet
    dat de vorstgrens hoog ligt. Een kleine tolerantie vangt afronding
    van GRIB-waarden af. Dit gebruikt expliciet het ICON-modelterrein,
    niet de gedetailleerde achtergrond-DEM van de kaart.
    """
    if terrain is None:
        return np.zeros(np.shape(freezing_level), dtype=bool)
    freezing_level = np.asarray(freezing_level)
    terrain = np.asarray(terrain)
    if freezing_level.shape != terrain.shape:
        raise ValueError("HZEROCL en HSURF hebben verschillende roosters")
    return (
        np.isfinite(freezing_level)
        & np.isfinite(terrain)
        & (freezing_level <= terrain + tolerance_m)
    )


def plot_altitude(
    field, x, y, extent, area,
    run, step, kind, hsurf=None
):

    valid_time = (
        run + timedelta(hours=step)
    ).astimezone(LOCAL_TZ)

    fig, ax, proj = common_axes(extent)

    cmap, norm = altitude_cmap(kind)

    # Bij volledig negatieve temperaturen rapporteert ICON HZEROCL=HSURF
    # (DWD-fillwaarde). Kleur deze pixels grijs i.p.v. met een ogenschijnlijk
    # hoge vorstgrens op bergtoppen. De oorspronkelijke data blijven intact.
    no_level = (
        no_freezing_level_above_terrain(field, hsurf)
        if kind == "hzerocl" else np.zeros(np.shape(field), dtype=bool)
    )
    z = np.ma.masked_where(no_level, np.ma.masked_invalid(field))

    cf = ax.contourf(
        x, y, z,
        levels=ALT_LEVELS,
        cmap=cmap,
        norm=norm,
        extend="max",
        transform=proj,
        zorder=3,
        antialiased=True
    )

    if kind == "hzerocl":
        if hsurf is None:
            print("WAARSCHUWING: HSURF ontbreekt; vorstgrens-vulwaarden niet grijs gemaskeerd")
        elif np.any(no_level):
            # Grijze, dekkende zones: overal onder nul, dus geen vrij
            # 0 C-niveau boven het modelterrein. Niet in de kleurenschaal.
            ax.contourf(
                x, y, no_level.astype(float),
                levels=[0.5, 1.5],
                colors=["#c8cdd3"],
                transform=proj,
                zorder=6,
                antialiased=True,
            )
            print(
                f"Vorstgrens: {np.count_nonzero(no_level)} modelpixels "
                "zonder 0 C-isotherm grijs weergegeven"
            )

    if kind == "snowlmt":
        if hsurf is None:
            print("WAARSCHUWING: HSURF ontbreekt; sneeuwgrensgebieden niet grijs weergegeven")
        else:
            # Zoals bij de vorstgrens: lichtgrijze, volledig dekkende zones.
            # Hier ligt het ICON-modelterrein op of boven de sneeuwgrens.
            # Behoud de oorspronkelijke SNOWLMT-waarden en de kleurenschaal.
            # Ongeldige punten buiten het modeldomein blijven transparant.
            if np.shape(field) != np.shape(hsurf):
                raise ValueError("SNOWLMT en HSURF hebben verschillende roosters")

            above_snowline = (
                np.isfinite(field)
                & np.isfinite(hsurf)
                & (hsurf >= field)
            )

            if np.any(above_snowline):
                ax.contourf(
                    x, y, above_snowline.astype(float),
                    levels=[0.5, 1.5],
                    colors=["#c8cdd3"],
                    transform=proj,
                    zorder=6,
                    antialiased=True,
                )
                print(
                    f"Sneeuwgrens: {np.count_nonzero(above_snowline)} "
                    "modelpixels boven de sneeuwgrens grijs weergegeven"
                )

    noun = (
        "Vorstgrens"
        if kind == "hzerocl"
        else "Sneeuwgrens"
    )

    stem = (
        "icon_d2_vorstgrens"
        if kind == "hzerocl"
        else "icon_d2_sneeuwgrens"
    )

    outfile = OUTPUT_DIR / (
        f"{stem}_{area}_{run:%Y%m%d_%H}z_"
        f"f{step:03d}_"
        f"{(run + timedelta(hours=step)):%Y%m%d_%H}z.png"
    )

    label = f"{noun} — {region_label(area)} (m)"
    if kind == "hzerocl" and np.any(no_level):
        label += " | grijs: geen 0 °C-niveau boven modelterrein"
    elif kind == "snowlmt" and hsurf is not None and np.any(above_snowline):
        label += " | grijs: terrein op/boven sneeuwgrens"

    finish_map(
        fig, ax, cf,
        fmt_date(valid_time),
        label,
        ALT_LEVELS,
        outfile,
        extent,
        area,
        tick_step=500
    )


# ============================================================
# 7. VERSE SNEEUW: 0-24 / 24-48 / 0-48
# ============================================================

def current_lead(run):
    """Aantal volle uren tussen de modelrun en nu (0..47)."""

    now = datetime.now(timezone.utc)
    hours = int((now - run).total_seconds() // 3600)

    return min(max(hours, 0), 47)


def make_snowfall_cm(
    total_start, total24, total48, start
):

    # DWD SNOW_GSP en SNOW_CON zijn cumulatief vanaf de start van de run.
    # Door het veld op het huidige uur (start) af te trekken, telt
    # sneeuw die al in het verleden gevallen is niet meer mee.

    maps = []

    if start < 24:

        day1 = (
            np.maximum(total24 - total_start, 0.0)
            * SNOW_CM_PER_KGM2
        )

        maps.append(("0-24u", start, 24, day1))

    day2_start = max(start, 24)
    total_day2 = total24 if start < 24 else total_start

    day2 = (
        np.maximum(total48 - total_day2, 0.0)
        * SNOW_CM_PER_KGM2
    )

    maps.append(("24-48u", day2_start, 48, day2))

    both = (
        np.maximum(total48 - total_start, 0.0)
        * SNOW_CM_PER_KGM2
    )

    maps.append(("0-48u", start, 48, both))

    return maps


def interpolate_snow(
    x, y, data, extent, area
):

    w, e, s, n = extent
    pad = 0.1

    xx = np.where(
        (x >= w - pad)
        & (x <= e + pad)
    )[0]

    yy = np.where(
        (y >= s - pad)
        & (y <= n + pad)
    )[0]

    x0, y0 = x[xx], y[yy]

    z0 = np.asarray(
        data[np.ix_(yy, xx)],
        dtype=float
    )

    known = np.isfinite(z0)

    if known.sum() < 100:

        raise ValueError(
            "Te weinig geldige sneeuwvalpunten voor interpolatie"
        )

    # Cubic interpolation zoals in SnowFresh, zonder het invullen van
    # gemaskeerde droge gebieden door de dichtstbijzijnde sneeuwval.
    spline = RectBivariateSpline(
        y0,
        x0,
        np.where(known, z0, 0),
        kx=3,
        ky=3,
        s=0
    )

    spacing = SNOW_INTERP_STEP[area]

    xf = np.arange(
        w,
        e + spacing / 2,
        spacing
    )

    yf = np.arange(
        s,
        n + spacing / 2,
        spacing
    )

    zf = np.maximum(
        spline(yf, xf),
        0.0
    )

    # Strikte begrenzing tot het oorspronkelijke geldige modelgebied.
    valid_nn = RegularGridInterpolator(
        (y0, x0),
        known.astype(float),
        method="nearest",
        bounds_error=False,
        fill_value=0.0
    )

    X, Y = np.meshgrid(xf, yf)

    covered = valid_nn(
        np.column_stack([
            Y.ravel(),
            X.ravel()
        ])
    ).reshape(Y.shape) > 0.5

    zf[~covered] = np.nan
    zf[zf < SNOW_MIN_CM] = np.nan

    return xf, yf, np.ma.masked_invalid(zf)


def plot_snowfall(
    data, x, y, extent, area,
    run, name, start, end
):

    xf, yf, zf = interpolate_snow(
        x, y, data, extent, area
    )

    fig, ax, proj = common_axes(extent)

    cmap = ListedColormap(SNOW_COLORS)
    cmap.set_bad((0, 0, 0, 0))
    cmap.set_over("#969696")

    norm = BoundaryNorm(
        SNOW_LEVELS,
        ncolors=cmap.N + 1,
        extend="max"
    )

    cf = ax.contourf(
        xf, yf, zf,
        levels=SNOW_LEVELS,
        cmap=cmap,
        norm=norm,
        transform=proj,
        extend="max",
        alpha=1,
        zorder=3
    )

    start_dt = (
        run + timedelta(hours=start)
    ).astimezone(LOCAL_TZ)

    end_dt = (
        run + timedelta(hours=end)
    ).astimezone(LOCAL_TZ)

    title = (
        f"{fmt_date(start_dt)} – "
        f"{fmt_date(end_dt)}"
    )

    outfile = OUTPUT_DIR / (
        f"icon_d2_verse_sneeuw_{area}_"
        f"{run:%Y%m%d_%H}z_{name}.png"
    )

    finish_map(
        fig, ax, cf,
        title,
        f"Sneeuwval — {region_label(area)} (cm, 10:1)",
        SNOW_LEVELS,
        outfile,
        extent,
        area
    )


# ============================================================
# 8. LOKALE DAGPERIODES: MIDDERNACHT EUROPE/BRUSSELS
# ============================================================

def local_day_windows(run, start_lead):
    """Return UTC lead-hour windows separated at Belgian local midnight.

    Midnight is converted with Europe/Brussels, so CET/CEST transitions
    are accounted for. Native model timestamps remain in UTC.
    """
    end_utc = run + timedelta(hours=48)
    first_local = (run + timedelta(hours=start_lead)).astimezone(LOCAL_TZ)
    day = first_local.date()
    boundaries = [start_lead]

    while True:
        next_day = day + timedelta(days=1)
        midnight_local = datetime(next_day.year, next_day.month, next_day.day, tzinfo=LOCAL_TZ)
        midnight_utc = midnight_local.astimezone(timezone.utc)
        if midnight_utc >= end_utc:
            break
        lead = int(round((midnight_utc - run).total_seconds() / 3600))
        if boundaries[-1] < lead < 48:
            boundaries.append(lead)
        day = next_day

    boundaries.append(48)
    return list(zip(boundaries[:-1], boundaries[1:]))


def cumulative_snow_at(session, run, step, parameter):
    """Fetch the exact hourly accumulation, with a 6-hour fallback.

    The fallback is only needed when DWD has not published a particular
    intermediate lead. It linearly splits a cumulative 6-hour amount,
    keeping the full total consistent with the native data.
    """
    field = load_field(session, run, step, parameter, optional=True)
    if field is not None:
        return field

    lower = (step // 6) * 6
    upper = min(48, lower + 6)
    if lower == upper:
        return None
    a = load_field(session, run, lower, parameter, optional=True)
    b = load_field(session, run, upper, parameter, optional=True)
    if a is None or b is None:
        return None
    assert_same_grid((a[0], a[1]), b, parameter)
    weight = (step - lower) / (upper - lower)
    print(f"WARNING: interpolated {parameter} +{step} from +{lower} and +{upper}")
    return a[0], a[1], a[2] * (1.0 - weight) + b[2] * weight, a[3]


# ============================================================
# 9. HOOFDPROGRAMMA
# ============================================================

def main():

    print(f"\nICON-D2: gebieden {AREAS_TO_MAKE}")
    print("9 vorstgrens + 9 sneeuwgrens per gebied; sneeuwval per lokale dag en cumulatief.")

    with requests.Session() as session:

        session.headers.update({
            "User-Agent": "SneeuwhoogteNL-ICON-D2-three-products/1.1"
        })

        run = newest_common_run(
            session
        )

        # Dezelfde DWD-data eenmalig binnenhalen voor beide gebieden.
        print("\n--- Vorstgrens laden ---")

        frost = {}
        x = y = None
        valid_domain = None

        for step in FORECAST_HOURS:

            value = load_field(
                session, run, step, "hzerocl"
            )

            if x is None:

                x, y = value[0], value[1]
                valid_domain = np.isfinite(value[2])

            else:

                assert_same_grid(
                    (x, y), value, "HZEROCL"
                )

                valid_domain &= np.isfinite(
                    value[2]
                )

            frost[step] = np.where(
                (value[2] >= -100)
                & (value[2] <= 8000),
                np.maximum(value[2], 0),
                np.nan
            )

            valid_domain &= np.isfinite(
                frost[step]
            )

        print("\n--- Sneeuwgrens laden ---")

        snowline = {}

        for step in FORECAST_HOURS:

            value = load_field(
                session, run, step, "snowlmt"
            )

            assert_same_grid(
                (x, y), value, "SNOWLMT"
            )

            snowline[step] = np.where(
                (value[2] >= -100)
                & (value[2] <= 8000),
                np.maximum(value[2], 0),
                np.nan
            )

        try:

            hfield = load_hsurf(
                session, run
            )

            assert_same_grid(
                (x, y), hfield, "HSURF"
            )

            terrain = np.where(
                (hfield[2] >= -500)
                & (hfield[2] <= 9000),
                np.maximum(hfield[2], 0),
                np.nan
            )

        except Exception as exc:

            print(
                f"WAARSCHUWING: HSURF niet beschikbaar "
                f"({exc}); geen arcering."
            )

            terrain = None

        snow_start = current_lead(run)
        windows = local_day_windows(run, snow_start)
        snow_steps = sorted({edge for window in windows for edge in window})

        print(f"\n--- Verse sneeuw laden: lokale dagvensters {windows} vanaf +{snow_start} h ---")
        gsp_fields = {}
        con_fields = {}

        for step in snow_steps:
            gsp = cumulative_snow_at(session, run, step, "snow_gsp")
            if gsp is None:
                raise RuntimeError(f"SNOW_GSP cumulative data not available at +{step} h")
            assert_same_grid((x, y), gsp, "SNOW_GSP")
            gsp_fields[step] = gsp[2]

            con = cumulative_snow_at(session, run, step, "snow_con")
            if con is not None:
                assert_same_grid((x, y), con, "SNOW_CON")
                con_fields[step] = con[2]

        if len(con_fields) != len(snow_steps):
            print("SNOW_CON not complete: using SNOW_GSP for every period")
            snowfall_cumulative = gsp_fields
        else:
            snowfall_cumulative = {
                step: gsp_fields[step] + con_fields[step]
                for step in snow_steps
            }

        snow_maps = []
        for i, (start, end) in enumerate(windows, 1):
            snow = np.maximum(snowfall_cumulative[end] - snowfall_cumulative[start], 0.0)
            snow_maps.append((f"dag{i:02d}", start, end, snow * SNOW_CM_PER_KGM2))

        snow_total = np.maximum(
            snowfall_cumulative[48] - snowfall_cumulative[snow_start], 0.0
        ) * SNOW_CM_PER_KGM2
        snow_maps.append(("cumulatief", snow_start, 48, snow_total))

        print(
            "\n--- Kaartuitsnedes bepalen ---"
        )

        extents = {
            area: determine_extent(
                x, y, valid_domain, area
            )
            for area in AREAS_TO_MAKE
        }

        count = 0

        for area in AREAS_TO_MAKE:

            extent = extents[area]

            print(
                f"\n--- Gebied: {area.upper()} | {extent} ---"
            )

            for step in FORECAST_HOURS:

                try:

                    plot_altitude(
                        frost[step],
                        x, y, extent, area,
                        run, step, "hzerocl",
                        terrain
                    )

                    count += 1

                except Exception as exc:

                    print(
                        f"FOUT vorstgrens {area} "
                        f"+{step:03d}: {exc}"
                    )

                try:

                    plot_altitude(
                        snowline[step],
                        x, y, extent, area,
                        run, step, "snowlmt",
                        terrain
                    )

                    count += 1

                except Exception as exc:

                    print(
                        f"FOUT sneeuwgrens {area} "
                        f"+{step:03d}: {exc}"
                    )

            for name, start, end, snowfall in snow_maps:

                try:

                    plot_snowfall(
                        snowfall,
                        x, y, extent, area,
                        run, name, start, end
                    )

                    count += 1

                except Exception as exc:

                    print(
                        f"FOUT verse sneeuw "
                        f"{area} {name}: {exc}"
                    )

    print(
        f"\nKlaar: {count}/"
        f"{len(AREAS_TO_MAKE) * (2 * len(FORECAST_HOURS) + len(snow_maps))} "
        f"kaarten. "
        f"Opslag: {OUTPUT_DIR}"
    )

    if count == 0:

        raise RuntimeError(
            "Geen kaarten gemaakt; "
            "controleer de fouten hierboven"
        )


if __name__ == "__main__":
    main()
