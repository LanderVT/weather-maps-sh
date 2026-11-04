"""
ICON-CH2 (MeteoSwiss) | Sneeuwval, regen en nieuwe sneeuw op de grond
per dag voor de Alpen, gedownscaled naar ~400 m

Zelfde aanpak als SnowDays_ECMWF.py, maar met ICON-CH2 (~2,1 km) als
invoer, tot +120 uur (5 dagen):

1. Neerslag (TOT_PREC) per STEP_HOURS, lineair van het native
   ICON-driehoeksrooster rechtstreeks naar het DEM-rooster.
2. Neerslag-hoogtegradient t.o.v. glad terrein (massabehoudend).
3. Sneeuwvalgrens: SNOWLMT van ICON-CH2 zelf (natteboltemperatuur
   +1.3 graden); lege cellen (grens op/onder het terrein) worden
   ingevuld met de modelterreinhoogte.
4. Op elke DEM-pixel sneeuw boven de grens, regen eronder
   (overgang van +/- SNOW_BAND_M meter).
5. Temperatuur per pixel uit T_2M en het modelterrein (HSURF) met een
   standaard verticale gradient; gebruikt voor de sneeuwdichtheid en
   het mini-sneeuwdekmodel (inklinken en smelten).

Kaarten (map outputmaps/ICON-CH2):
  - sneeuwval (+ regen) per dag
  - cumulatieve sneeuwval (+ regen) vanaf nu
  - nieuwe sneeuw op de grond (na inklinken en smelten)

Python-pakketten: pip install numpy matplotlib eccodes requests cartopy scipy

Gebruik: python SnowDays_ICON-CH2.py
"""

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
import eccodes
import numpy as np
import requests
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.spatial import Delaunay

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.img_tiles as cimgt


# ============================================================
# 1. INSTELLINGEN
# ============================================================

HOME = Path.home()
DOCUMENTS_DIR = HOME / "Documents" / "SneeuwhoogteNL"
DATA_DIR = DOCUMENTS_DIR / "data"
OUTPUT_DIR = DOCUMENTS_DIR / "outputmaps" / "ICON-CH2"
TILE_CACHE_DIR = DATA_DIR / "esri_tiles"
DEM_CACHE_DIR = DATA_DIR / "dem_tiles"
LOGO_PATH = HOME / "logo-sneeuwhoogte.png"

STAC_URL = "https://data.geo.admin.ch/api/stac/v1"
DEM_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium"

MODEL_LABEL = "ICON-CH2"
FILE_PREFIX = "icon_ch2"  # begin van de bestandsnamen
COLLECTION = "ch.meteoschweiz.ogd-forecasting-icon-ch2"
RUN_HOURS = (0, 6, 12, 18)
MAX_LEAD = 120
STEP_HOURS = 3

T_LAPSE = 0.0065       # graden C per meter, T_2M -> DEM-hoogte
SNOW_TW = 1.3          # natteboltemperatuur op de sneeuwvalgrens (SNOWLMT)
TW_LAPSE = 0.005       # graden C per meter boven de sneeuwvalgrens
SNOW_BAND_M = 200      # halve breedte van de regen/sneeuw-overgang

# Neerslag-hoogtegradient (MicroMet-vorm, Liston & Elder 2006):
#   P = P0 * (1 + chi*dz) / (1 - chi*dz), dz in km t.o.v. glad terrein.
# ICON-CH2 ziet de orografie al grotendeels; daarom een kleine
# gladmaakschaal (~effectieve resolutie van het model). 0 = uit.
PRECIP_CHI_PER_KM = 0.20
PRECIP_SMOOTH_DEG = 0.05
PRECIP_FACTOR_RANGE = (0.5, 2.0)

# GRIB2-sleutels (discipline, categorie, nummer). MeteoSwiss gebruikt
# deels lokale tabellen, waardoor eccodes 'unknown' als shortName geeft.
GRIB_KEYS = {
    "TOT_PREC": (0, 1, 52),
    "SNOWLMT": (0, 1, 204),
    "T_2M": (0, 0, 0),
    "HSURF": (0, 3, 6),
    "TLON": (0, 191, 2),
    "TLAT": (0, 191, 1),
}

AREAS_TO_MAKE = ("alps",)
AREA_LIMITS = {
    "alps": [5.0, 16.8, 43.6, 49.0],
}

DEM_ZOOM = 9
GRID_STEP = 0.004      # ~300-450 m; fijn doelrooster in graden

USE_ESRI_RELIEF = True
ESRI_ZOOM = 9
DPI = 400
LOCAL_TZ = ZoneInfo("Europe/Brussels")

# Is de nieuwste beschikbare run ouder dan dit (uur), dan maakt het
# script niets en blijven de vorige kaarten staan.
MAX_RUN_AGE_H = 12

CITY_FONT = {"alps": 8.0}
MAX_CITY_LABELS = {"alps": 38}
MIN_CITY_DISTANCE_PX = {"alps": 16}
LABEL_GAP_PX = 4

# Verse-sneeuwdichtheid naar temperatuur (Anderson 1976, SNOW-17):
#   rho = 50 + 1.7 * (T + 15)^1.5 kg/m3, T = natteboltemperatuur op de pixel.
# ~6.5:1 rond 0 graden en ~10:1 bij -5; nooit lichter dan SNOW_RHO_MIN.
# False = vast 10:1 zoals de ICON-scripts.
VARIABLE_DENSITY = True
SNOW_RHO_MIN = 100.0  # kg/m3, = 10:1

# Mini-sneeuwdekmodel voor de cumulatieve kaarten ('nieuwe sneeuw op de
# grond'). Alleen de NIEUWE sneeuw wordt gevolgd; de bestaande sneeuwlaag
# is onbekend. Eenvoudige, instelbare aannames:
#   inklinken: dichtheid nadert RHO_*_MAX met tijdconstante TAU_* (uur);
#              'nat' = T > 0 of regen in die 6 uur.
#   smelten:   graaddagen DDF * max(T, 0); in november weinig zon.
#              Bij regen op sneeuw extra (ROS_FACTOR, vuistregel voor
#              de warme, vochtige en winderige lucht) plus de warmte
#              van het regenwater zelf (regen * T / 80).
RHO_DRY_MAX = 250.0
RHO_WET_MAX = 350.0
TAU_DRY_H = 96.0
TAU_WET_H = 24.0
DDF_MM_PER_DEG_DAY = 2.5
ROS_FACTOR = 1.5
RAIN_ON_SNOW_MM = 0.5  # minimale regen per tijdstap voor 'regen op sneeuw'

# Regenlaag op de sneeuwvalkaarten (per dag en cumulatief), mm = l/m2.
# False = alleen sneeuw, zonder tweede kleurbalk.
SHOW_RAIN = False
RAIN_LEVELS = np.array([1, 4, 10, 20, 30, 40, 60, 80, 120])
RAIN_COLORS = ("#e3e8f0", "#3c4f85")
SNOW_CM_PER_MM = 1.0  # alleen bij VARIABLE_DENSITY = False
SNOW_MIN_CM = 0.5

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

    # WEST-POLEN (BUITEN DE ALPENKAART; WORDT AUTOMATISCH OVERGESLAGEN)
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
# 3. METEOSWISS DATA (STAC-API, NATIVE ICON-ROOSTER)
# ============================================================

class EsriWorldShadedRelief(cimgt.GoogleTiles):

    def _image_url(self, tile):
        x, y, z = tile
        return (
            "https://server.arcgisonline.com/ArcGIS/rest/services/"
            f"World_Shaded_Relief/MapServer/tile/{z}/{y}/{x}.jpg"
        )


def stac_asset(session, variable, run=None, lead=None):
    """Zoek de control-run-GRIB van een variabele (en optioneel run/lead)."""

    body = {
        "collections": [COLLECTION],
        "forecast:variable": variable,
        "forecast:perturbed": False,
        "limit": 100,
    }

    if run is not None:
        body["forecast:reference_datetime"] = f"{run:%Y-%m-%dT%H:%M:%SZ}"

    if lead is not None:
        body["forecast:horizon"] = (
            f"P{lead // 24}DT{lead % 24:02d}H00M00S"
        )

    r = session.post(f"{STAC_URL}/search", json=body, timeout=30)
    r.raise_for_status()

    return [
        (name, asset["href"])
        for feature in r.json().get("features", [])
        for name, asset in feature["assets"].items()
    ]


def newest_run(session):
    """Nieuwste run met +MAX_LEAD voor neerslag, sneeuwgrens en T_2M."""

    now = datetime.now(timezone.utc)

    for back in range(2):

        d = (now - timedelta(days=back)).date()

        for h in sorted(RUN_HOURS, reverse=True):

            run = datetime(
                d.year, d.month, d.day, h,
                tzinfo=timezone.utc
            )

            if run > now:
                continue

            if all(
                stac_asset(session, var, run, MAX_LEAD)
                for var in ("TOT_PREC", "SNOWLMT", "T_2M")
            ):

                print(f"{MODEL_LABEL}-run: {run:%Y-%m-%d %H:%M} UTC")

                return run

    raise RuntimeError(f"Geen {MODEL_LABEL}-run gevonden met +{MAX_LEAD} uur.")


def download_archive(session, url, name):

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = DATA_DIR / name

    # Velden die overal 0 zijn (bv. TOT_PREC op +0) zijn maar een paar
    # honderd bytes; controleer daarom de GRIB-header i.p.v. de grootte.
    if out.is_file() and out.read_bytes()[:4] == b"GRIB":
        print("In cache:", name)
        return out

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

        with partial.open("rb") as f:
            if f.read(4) != b"GRIB":
                raise RuntimeError(f"Geen GRIB-bestand: {name}")

        partial.replace(out)

    finally:
        partial.unlink(missing_ok=True)

    return out


def read_grib(path, wanted):
    """Lees de gevraagde velden (op GRIB_KEYS-naam) als 1D native arrays."""

    keys = {GRIB_KEYS[name]: name for name in wanted}
    fields = {}

    with path.open("rb") as f:

        while (gid := eccodes.codes_grib_new_from_file(f)) is not None:

            try:

                key = (
                    eccodes.codes_get(gid, "discipline"),
                    eccodes.codes_get(gid, "parameterCategory"),
                    eccodes.codes_get(gid, "parameterNumber"),
                )

                if key in keys and keys[key] not in fields:

                    values = eccodes.codes_get_values(gid).astype(float)

                    if eccodes.codes_get(gid, "bitmapPresent"):
                        values[
                            values == eccodes.codes_get(gid, "missingValue")
                        ] = np.nan

                    fields[keys[key]] = values

            finally:
                eccodes.codes_release(gid)

    missing = set(wanted) - set(fields)

    if missing:
        raise KeyError(f"{sorted(missing)} niet gevonden in {path.name}")

    return fields


class NativeRegridder:
    """Lineaire interpolatie van het ICON-driehoeksrooster naar (xf, yf).

    De Delaunay-triangulatie en barycentrische gewichten worden eenmaal
    berekend en daarna voor alle velden hergebruikt.
    """

    def __init__(self, lon, lat, xf, yf, max_edge_deg=0.06):

        scale = np.cos(np.radians(np.mean(lat)))

        X, Y = np.meshgrid(xf, yf)
        points = np.column_stack([X.ravel() * scale, Y.ravel()])

        tri = Delaunay(np.column_stack([lon * scale, lat]))
        simplex = tri.find_simplex(points)
        inside = simplex >= 0

        # Lange driehoeken aan de bolle rand vallen buiten het modeldomein.
        corners = tri.points[tri.simplices[simplex[inside]]]
        edges = np.max(
            np.linalg.norm(
                corners - np.roll(corners, 1, axis=1), axis=2
            ),
            axis=1,
        )
        inside[np.flatnonzero(inside)[edges >= max_edge_deg]] = False

        T = tri.transform[simplex[inside]]
        b = np.einsum(
            "nij,nj->ni", T[:, :2], points[inside] - T[:, 2]
        )

        self.inside = inside
        self.verts = tri.simplices[simplex[inside]]
        self.weights = np.column_stack([b, 1 - b.sum(axis=1)])
        self.shape = X.shape

    def __call__(self, values):

        out = np.full(self.inside.size, np.nan)

        # Een NaN-hoekpunt (bv. ontbrekende SNOWLMT) maakt de cel NaN.
        out[self.inside] = np.sum(
            values[self.verts] * self.weights, axis=1
        )

        return out.reshape(self.shape)


def load_grid(session, xf, yf):
    """Regridder naar het DEM-rooster en de modelterreinhoogte (HSURF)."""

    r = session.get(
        f"{STAC_URL}/collections/{COLLECTION}/assets", timeout=30
    )
    r.raise_for_status()

    asset = next(
        a for a in r.json()["assets"]
        if a["id"].startswith("horizontal_constants")
    )

    path = download_archive(session, asset["href"], asset["id"])
    fields = read_grib(path, ["TLON", "TLAT", "HSURF"])

    print(f"{MODEL_LABEL}: regridding naar het DEM-rooster ({GRID_STEP} graden) ...")

    regrid = NativeRegridder(fields["TLON"], fields["TLAT"], xf, yf)

    return regrid, regrid(fields["HSURF"])


def load_field(session, regrid, run, lead, variable):

    assets = stac_asset(session, variable, run, lead)

    if not assets:
        raise FileNotFoundError(
            f"{variable} +{lead:03d} ontbreekt voor run {run:%Y%m%d%H}"
        )

    name, href = assets[0]
    path = download_archive(session, href, name)

    return regrid(read_grib(path, [variable])[variable])


# ============================================================
# 4. HOOGTEMODEL (AWS TERRAIN TILES, TERRARIUM-PNG)
# ============================================================

def tile_xy(lon, lat, zoom):
    """Fractionele globale pixelcoordinaten (Web Mercator, 256 px tegels)."""

    size = 256 * 2 ** zoom
    lat = np.radians(lat)

    px = (np.asarray(lon) + 180.0) / 360.0 * size
    py = (
        1 - np.log(np.tan(lat) + 1 / np.cos(lat)) / np.pi
    ) / 2 * size

    return px, py


def load_dem(session, extent):
    """DEM op het regelmatige doelrooster; zee/meer < -10 m wordt NaN."""

    w, e, s, n = extent

    xf = np.round(np.arange(w, e + GRID_STEP / 2, GRID_STEP), 5)
    yf = np.round(np.arange(s, n + GRID_STEP / 2, GRID_STEP), 5)

    cache = DATA_DIR / (
        f"dem_z{DEM_ZOOM}_{GRID_STEP}_{w}_{e}_{s}_{n}.npy"
    )

    if cache.is_file():
        print("DEM in cache:", cache.name)
        return xf, yf, np.load(cache)

    DEM_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    px0, py1 = tile_xy(w, s, DEM_ZOOM)
    px1, py0 = tile_xy(e, n, DEM_ZOOM)

    tx = range(int(px0 // 256), int(px1 // 256) + 1)
    ty = range(int(py0 // 256), int(py1 // 256) + 1)

    mosaic = np.zeros((256 * len(ty), 256 * len(tx)), dtype=np.float32)

    print(f"DEM: {len(tx) * len(ty)} tegels (zoom {DEM_ZOOM})")

    for j, y in enumerate(ty):
        for i, x in enumerate(tx):

            path = DEM_CACHE_DIR / f"{DEM_ZOOM}_{x}_{y}.png"

            if not path.is_file():

                r = session.get(
                    f"{DEM_URL}/{DEM_ZOOM}/{x}/{y}.png", timeout=30
                )
                r.raise_for_status()
                path.write_bytes(r.content)

            rgb = plt.imread(path)[..., :3] * 255.0
            elev = (
                rgb[..., 0] * 256 + rgb[..., 1] + rgb[..., 2] / 256
            ) - 32768

            mosaic[j * 256:(j + 1) * 256, i * 256:(i + 1) * 256] = elev

    X, Y = np.meshgrid(xf, yf)
    px, py = tile_xy(X, Y, DEM_ZOOM)

    dem = map_coordinates(
        mosaic,
        [py - ty[0] * 256 - 0.5, px - tx[0] * 256 - 0.5],
        order=1,
        mode="nearest"
    )

    dem = np.where(dem < -10, np.nan, np.maximum(dem, 0.0))

    np.save(cache, dem)

    return xf, yf, dem


def precip_factor(dem):
    """Hoogteafhankelijke neerslagfactor t.o.v. het gladde modelterrein."""

    if PRECIP_CHI_PER_KM <= 0:
        return np.ones_like(dem)

    sigma = PRECIP_SMOOTH_DEG / GRID_STEP
    land = np.isfinite(dem)
    terrain = np.where(land, dem, 0.0)

    smooth = gaussian_filter(terrain, sigma, mode="nearest")
    dz = np.clip((terrain - smooth) / 1000.0, -2.0, 2.0)

    a = PRECIP_CHI_PER_KM * dz
    factor = np.clip(
        (1 + a) / np.maximum(1 - a, 1e-3), *PRECIP_FACTOR_RANGE
    )

    # Massabehoud: gemiddeld 1 over ~25 km.
    factor = factor / np.maximum(
        gaussian_filter(factor, sigma, mode="nearest"), 1e-3
    )

    print(
        f"Neerslagfactor: p5 {np.nanpercentile(factor[land], 5):.2f}, "
        f"p95 {np.nanpercentile(factor[land], 95):.2f}"
    )

    return factor


def snow_cm_per_mm(dem, line):
    """cm verse sneeuw per mm water, uit de natteboltemperatuur op de pixel.

    Boven de sneeuwvalgrens (Tw = SNOW_TW) daalt Tw met TW_LAPSE per meter.
    """

    tw = SNOW_TW - TW_LAPSE * (dem - line)
    tw = np.clip(tw, -15.0, SNOW_TW)

    # Nooit luchtiger dan 10:1: koude toppen hebben in de praktijk wind
    # en verdichting, en de formule geldt voor windstille vlakke grond.
    rho = np.maximum(50.0 + 1.7 * (tw + 15.0) ** 1.5, SNOW_RHO_MIN)

    return 100.0 / rho


def snowpack_step(swe, depth, snow_mm, snow_cm, rain_mm, temp):
    """Een tijdstap van STEP_HOURS: bijschrijven, inklinken, smelten."""

    dt = float(STEP_HOURS)

    swe = swe + snow_mm
    depth = depth + snow_cm

    has_snow = (swe > 1e-3) & (depth > 1e-3)

    # Inklinken naar een maximale dichtheid, sneller als de sneeuw nat is.
    wet = (temp > 0) | (rain_mm > RAIN_ON_SNOW_MM)
    rho = np.where(has_snow, 100.0 * swe / np.maximum(depth, 1e-3), 0.0)
    rho_max = np.where(wet, RHO_WET_MAX, RHO_DRY_MAX)
    tau = np.where(wet, TAU_WET_H, TAU_DRY_H)

    rho = np.where(
        rho < rho_max, rho_max - (rho_max - rho) * np.exp(-dt / tau), rho
    )

    # Smelten: graaddagen, extra bij regen op sneeuw, plus regenwarmte.
    warm = np.maximum(temp, 0.0)
    ros = np.where(rain_mm > RAIN_ON_SNOW_MM, ROS_FACTOR, 1.0)

    melt = DDF_MM_PER_DEG_DAY * warm * dt / 24.0 * ros + rain_mm * warm / 80.0
    melt = np.minimum(melt, swe)

    swe = swe - melt
    depth = np.where(
        has_snow & (swe > 1e-3), 100.0 * swe / np.maximum(rho, 1.0), 0.0
    )

    return np.where(depth > 0, swe, 0.0), depth


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
    filepath, extent, area, tick_step=None, rain=None
):
    """rain: optioneel (contourset, label, levels) voor een tweede kleurbalk."""

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

    if rain is not None:

        rain_cf, rain_label, rain_levels = rain

        rax = fig.add_axes([
            pos.x0,
            pos.y0 - 0.05 - 0.075 - 0.035,
            pos.width,
            0.035
        ])

        rcb = fig.colorbar(rain_cf, cax=rax, orientation="horizontal")
        rcb.set_label(rain_label, fontsize=14, labelpad=5)
        rcb.set_ticks(rain_levels)
        rcb.ax.tick_params(labelsize=9)
        rcb.outline.set_linewidth(0.8)

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


# ============================================================
# 6. VERSE SNEEUW PER DAG
# ============================================================

def day_windows(run, now):
    """(start, eind) in uren na de run: rest van vandaag + volle UTC-dagen,
    tot +MAX_LEAD. De laatste dag stopt bij +MAX_LEAD als het model niet
    verder rekent (de eindtijd staat in de titel)."""

    first = int((now - run).total_seconds() // 3600)
    first = max(0, first - first % STEP_HOURS)

    windows = []
    start = first

    while start < MAX_LEAD:

        end = min(start + 24 - (run + timedelta(hours=start)).hour, MAX_LEAD)

        windows.append((start, end))
        start = end

    return windows


def plot_snowfall(
    data, x, y, extent, area, run, start, end, rain=None, total=False
):
    """Sneeuwval + regen (per dag of cumulatief met total=True) of,
    zonder rain, nieuwe sneeuw op de grond."""

    fig, ax, proj = common_axes(extent)

    cmap = ListedColormap(SNOW_COLORS)
    cmap.set_bad((0, 0, 0, 0))
    cmap.set_over("#969696")

    norm = BoundaryNorm(
        SNOW_LEVELS,
        ncolors=cmap.N + 1,
        extend="max"
    )

    z = np.ma.masked_where(
        ~np.isfinite(data) | (data < SNOW_MIN_CM), data
    )

    cf = ax.contourf(
        x, y, z,
        levels=SNOW_LEVELS,
        cmap=cmap,
        norm=norm,
        transform=proj,
        extend="max",
        zorder=3
    )

    rain_bar = None

    if rain is not None and SHOW_RAIN:

        rcmap = LinearSegmentedColormap.from_list(
            "rain", RAIN_COLORS, N=len(RAIN_LEVELS) - 1
        )
        rcmap.set_bad((0, 0, 0, 0))
        rcmap.set_over(RAIN_COLORS[-1])

        rz = np.ma.masked_where(
            ~np.isfinite(rain) | (rain < RAIN_LEVELS[0]), rain
        )

        # Onder de sneeuw, licht doorzichtig zodat het reliëf zichtbaar blijft.
        rcf = ax.contourf(
            x, y, rz,
            levels=RAIN_LEVELS,
            cmap=rcmap,
            norm=BoundaryNorm(RAIN_LEVELS, rcmap.N),
            transform=proj,
            extend="max",
            alpha=0.8,
            zorder=2
        )

        rain_bar = (rcf, "Regen (mm)", RAIN_LEVELS)

    start_dt = (run + timedelta(hours=start)).astimezone(LOCAL_TZ)
    end_dt = (run + timedelta(hours=end)).astimezone(LOCAL_TZ)

    if rain is not None:

        outfile = OUTPUT_DIR / (
            f"{FILE_PREFIX}_sneeuwval{'_regen' if SHOW_RAIN else ''}"
            f"{'_cumulatief' if total else ''}_"
            f"{area}_{run:%Y%m%d_%H}z_f{start:03d}-{end:03d}.png"
        )
        title = (
            f"{'Totaal ' if total else ''}"
            f"{fmt_date(start_dt)} – {fmt_date(end_dt)}"
        )
        label = (
            f"Sneeuwval — {region_label(area)} "
            f"(cm, {'dichtheid naar T' if VARIABLE_DENSITY else '10:1'}) — "
            f"{MODEL_LABEL}, gedownscaled"
        )

    else:

        outfile = OUTPUT_DIR / (
            f"{FILE_PREFIX}_sneeuw_op_grond_{area}_{run:%Y%m%d_%H}z_"
            f"f{start:03d}-{end:03d}.png"
        )
        title = f"Nieuwe sneeuw op de grond — {fmt_date(end_dt)}"
        label = (
            f"Sinds {fmt_date(start_dt).lower()}, na inklinken en smelten "
            "(cm) — "
            f"{MODEL_LABEL}, gedownscaled"
        )

    finish_map(
        fig, ax, cf,
        title,
        label,
        SNOW_LEVELS,
        outfile,
        extent,
        area,
        rain=rain_bar
    )


def iterate_steps(session, run, steps, area):
    """Per tijdstap (a, b) sneeuw (mm en cm), regen (mm) en temperatuur
    op het DEM-rooster van 'area'. Gebruikt door main() en SnowHybrid.py."""

    extent = AREA_LIMITS[area]

    xf, yf, dem = load_dem(session, extent)
    regrid, hsurf = load_grid(session, xf, yf)

    land = np.isfinite(dem) & np.isfinite(hsurf)
    dem0 = np.where(land, dem, 0.0)
    hsurf = np.where(land, hsurf, 0.0)
    factor = precip_factor(np.where(land, dem, np.nan))

    print(f"\n--- {MODEL_LABEL}-velden laden ---")

    prev = None

    for step in steps:

        try:
            tp = load_field(session, regrid, run, step, "TOT_PREC")
        except FileNotFoundError:
            if step != 0:
                raise
            tp = np.zeros_like(dem0)  # +0: nog geen neerslag

        line = load_field(session, regrid, run, step, "SNOWLMT")
        t2m = load_field(session, regrid, run, step, "T_2M")

        # Lege SNOWLMT = grens op/onder het modelterrein.
        line = np.where(
            np.isfinite(line) & (line >= -100) & (line <= 8000),
            line, hsurf
        )

        cur = {
            "tp": np.nan_to_num(tp),
            "limit": line,
            "temp": t2m - 273.15 - T_LAPSE * (dem0 - hsurf),
        }

        if prev is None:
            prev = cur
            continue

        a, b = step - STEP_HOURS, step

        precip = np.maximum(cur["tp"] - prev["tp"], 0.0) * factor
        line = 0.5 * (prev["limit"] + cur["limit"])
        temp = 0.5 * (prev["temp"] + cur["temp"])

        snow_frac = np.clip(
            (dem0 - (line - SNOW_BAND_M)) / (2 * SNOW_BAND_M),
            0.0, 1.0
        )

        cm_per_mm = (
            snow_cm_per_mm(dem0, line)
            if VARIABLE_DENSITY else SNOW_CM_PER_MM
        )

        snow_mm = precip * snow_frac
        rain_mm = precip - snow_mm
        snow_cm = snow_mm * cm_per_mm

        yield {
            "a": a, "b": b, "xf": xf, "yf": yf, "land": land,
            "snow_mm": np.nan_to_num(snow_mm),
            "rain_mm": np.nan_to_num(rain_mm),
            "snow_cm": np.nan_to_num(snow_cm),
            "temp": np.nan_to_num(temp),
        }

        prev = cur


def main():

    print(
        f"\n{MODEL_LABEL}: sneeuwval + regen per dag en nieuwe sneeuw op de "
        f"grond, gedownscaled"
    )

    now = datetime.now(timezone.utc)

    with requests.Session() as session:

        session.headers.update({
            "User-Agent": "SneeuwhoogteNL-ICON-CH2-snowdays/1.0"
        })

        run = newest_run(session)

        age_h = (now - run).total_seconds() / 3600

        if age_h > MAX_RUN_AGE_H:
            print(
                f"Nieuwste {MODEL_LABEL}-run ({run:%Y-%m-%d %H} UTC) is "
                f"{age_h:.0f} uur oud (> {MAX_RUN_AGE_H} uur); niets gedaan."
            )
            return

        windows = day_windows(run, now)
        steps = list(range(windows[0][0], windows[-1][1] + 1, STEP_HOURS))

        print(
            f"Dagvensters (uur na run): {windows}; "
            f"{len(steps)} tijdstappen"
        )

        count = 0

        for area in AREAS_TO_MAKE:

            extent = AREA_LIMITS[area]

            daily_snow = daily_rain = None
            on_ground = {}

            for item in iterate_steps(session, run, steps, area):

                if daily_snow is None:

                    xf, yf, land = item["xf"], item["yf"], item["land"]
                    dem0 = np.zeros(land.shape)

                    daily_snow = {window: np.zeros_like(dem0) for window in windows}
                    daily_rain = {window: np.zeros_like(dem0) for window in windows}
                    swe = np.zeros_like(dem0)
                    depth = np.zeros_like(dem0)

                a, b = item["a"], item["b"]

                for window in windows:
                    if window[0] <= a < window[1]:
                        daily_snow[window] += item["snow_cm"]
                        daily_rain[window] += item["rain_mm"]

                swe, depth = snowpack_step(
                    swe, depth, item["snow_mm"], item["snow_cm"],
                    item["rain_mm"], item["temp"]
                )

                for window in windows:
                    if b == window[1]:
                        on_ground[window] = depth.copy()

            maps = [
                (start, end, daily_snow[(start, end)],
                 daily_rain[(start, end)], False)
                for start, end in windows
            ]

            # Cumulatieve sneeuwval + regen (zonder inklinken/smelten); de
            # eerste dag is identiek aan zijn dagkaart en wordt overgeslagen.
            snow_sum = np.zeros_like(dem0)
            rain_sum = np.zeros_like(dem0)

            for k, window in enumerate(windows):

                snow_sum = snow_sum + daily_snow[window]
                rain_sum = rain_sum + daily_rain[window]

                if k > 0:
                    maps.append(
                        (windows[0][0], window[1], snow_sum, rain_sum, True)
                    )

            maps += [
                (windows[0][0], end, on_ground[(start, end)], None, False)
                for start, end in windows
            ]

            for start, end, snow, rain, total in maps:

                snow = np.where(land, snow, np.nan)

                if rain is not None:
                    rain = np.where(land, rain, np.nan)

                kind = (
                    "op de grond" if rain is None
                    else "cumulatief" if total else "sneeuwval"
                )

                print(
                    f"+{start:03d}-{end:03d} {kind}: "
                    f"max {np.nanmax(snow):.0f} cm"
                    + (
                        f", regen max {np.nanmax(rain):.0f} mm"
                        if rain is not None else ""
                    )
                )

                try:

                    plot_snowfall(
                        snow, xf, yf, extent, area, run, start, end, rain,
                        total
                    )

                    count += 1

                except Exception as exc:

                    print(f"FOUT {kind} {area} +{start}-{end}: {exc}")

    print(
        f"\nKlaar: {count}/{len(AREAS_TO_MAKE) * (3 * len(windows) - 1)} "
        f"kaarten. Opslag: {OUTPUT_DIR}"
    )

    if count == 0:

        raise RuntimeError(
            "Geen kaarten gemaakt; "
            "controleer de fouten hierboven"
        )


if __name__ == "__main__":
    main()
