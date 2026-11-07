"""Shared Sneeuwhoogte dark/slate *cartography only*.

Adapted from the user's SnowAll_ICON-D2(4).py design. The forecast inputs,
scientific model routines, accumulation, downscaling and day windows are NOT
in this module and are not changed by it. The plotting functions here are
used by snow_dark_adapter.py for all 3 forecast tabs.
"""
from __future__ import annotations
from pathlib import Path
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm, LightSource, LinearSegmentedColormap, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.transforms import Bbox
from scipy.interpolate import RectBivariateSpline, RegularGridInterpolator
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.img_tiles as cimgt

HOME = Path.home()
LOGO_PATH = HOME / "logo-sneeuwhoogte.png"
OUTPUT_DIR = HOME / "Documents" / "SneeuwhoogteNL" / "outputmaps"
TILE_CACHE_DIR = HOME / "Documents" / "SneeuwhoogteNL" / "data" / "esri_tiles"
USE_ESRI_RELIEF = True
ESRI_ZOOM = 9
DPI = 200
LOCAL_TZ = ZoneInfo("Europe/Brussels")

MAP_STYLE = "dark"


CITY_FONT = {"full": 7.5, "alps": 8.0}


MAX_CITY_LABELS = {"full": 45, "alps": 38}


MIN_CITY_DISTANCE_PX = {"full": 18, "alps": 16}


LABEL_GAP_PX = 4


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


DARK = {
    "bg": "#0a111c",
    "text": "#ffffff",
    "subtext": "#c6d0de",
    "line": "#2c3747",
    "pill_edge": "#8592a6",
    "relief_tint": "#0a1626",
    "relief_alpha": 0.60,
    "sea": "#050b16",
    "border": "#ffffff",
}


RELIEF_THEME = "slate"


THEMES = {
    "graphite": {
        "stops": [(0.00, "#0a0b0d"), (0.15, "#0f1113"), (0.40, "#181b1f"),
                  (0.70, "#25292e"), (1.00, "#373c42")],
        "ice": (0.86, 0.88, 0.91), "glint": (0.08, 0.30),
        "light_gain": 1.0, "shade": 0.75,
        "ui": {"bg": "#0c0d10", "sea": "#07080a", "line": "#2a2d33",
               "pill_edge": "#8a8f98", "subtext": "#c9ccd2"},
        "glow": (0.85, 0.88, 0.95), "vignette": (0.01, 0.01, 0.012),
    },
    "slate": {
        "stops": [(0.00, "#090c10"), (0.15, "#0e131a"), (0.40, "#171e27"),
                  (0.70, "#242d38"), (1.00, "#36414e")],
        "ice": (0.83, 0.88, 0.94), "glint": (0.10, 0.38),
        "light_gain": 1.05, "shade": 0.75,
        "ui": {"bg": "#0a0d12", "sea": "#06080c", "line": "#2a313b",
               "pill_edge": "#8791a0", "subtext": "#c6cdd7"},
        "glow": (0.70, 0.80, 0.95), "vignette": (0.01, 0.015, 0.02),
    },
    "night": {
        "stops": [(0.00, "#070708"), (0.15, "#0b0b0d"), (0.40, "#131315"),
                  (0.70, "#1e1e21"), (1.00, "#2e2d30")],
        "ice": (0.92, 0.90, 0.88), "glint": (0.05, 0.22),
        "light_gain": 0.9, "shade": 0.8,
        "ui": {"bg": "#09090b", "sea": "#040405", "line": "#26262a",
               "pill_edge": "#8c8c92", "subtext": "#cfcfd4"},
        "glow": (0.95, 0.95, 0.95), "vignette": (0.0, 0.0, 0.0),
    },
    "ice": {
        "stops": [(0.00, "#070e1a"), (0.15, "#0b1628"), (0.40, "#16273f"),
                  (0.70, "#253d5c"), (1.00, "#3c5878")],
        "ice": (0.80, 0.90, 1.00), "glint": (0.18, 0.55),
        "light_gain": 1.1, "shade": 0.75,
        "ui": {"bg": "#0a111c", "sea": "#050b16", "line": "#2c3747",
               "pill_edge": "#8592a6", "subtext": "#c6d0de"},
        "glow": (0.55, 0.82, 1.0), "vignette": (0.012, 0.025, 0.05),
    },
}


ALT_ALPHA_DARK = 0.72


DARK.update(THEMES[RELIEF_THEME]["ui"])


FONT_FAMILY = ["Arial", "DejaVu Sans"]


FIG_WIDTH_IN = 16.9


HEADER_IN = 0.95


FOOTER_IN = 1.25


CITY_FONT_DARK = {"full": 10.0, "alps": 11.0}


SNOW_INTERP_STEP_DARK = {"full": 0.008, "alps": 0.005}


DAYS_SHORT = ["ma", "di", "wo", "do", "vr", "za", "zo"]


DAYS = [
    "maandag", "dinsdag", "woensdag", "donderdag",
    "vrijdag", "zaterdag", "zondag"
]


MONTHS_SHORT = [
    "jan", "feb", "mrt", "apr", "mei", "jun",
    "jul", "aug", "sep", "okt", "nov", "dec"
]


MONTHS = [
    "januari", "februari", "maart", "april", "mei", "juni",
    "juli", "augustus", "september", "oktober", "november", "december"
]


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


class EsriWorldShadedRelief(cimgt.GoogleTiles):

    def _image_url(self, tile):
        x, y, z = tile
        return (
            "https://server.arcgisonline.com/ArcGIS/rest/services/"
            f"World_Shaded_Relief/MapServer/tile/{z}/{y}/{x}.jpg"
        )


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

    dark = MAP_STYLE == "dark"

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
                fontsize=(
                    CITY_FONT_DARK[area] if dark else CITY_FONT[area]
                ),
                fontweight="bold" if dark else "normal",
                color="white" if dark else "#101010",
                zorder=41,
                annotation_clip=False,
                path_effects=[
                    pe.withStroke(
                        linewidth=2.6 if dark else 2.2,
                        foreground=DARK["bg"] if dark else "white"
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
                markersize=3.8 if dark else 2.8,
                markerfacecolor="white" if dark else "black",
                markeredgecolor=DARK["bg"] if dark else "white",
                markeredgewidth=0.8 if dark else 0.65,
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


def apply_theme(name):
    """Kies een achtergrondthema (zie THEMES)."""

    global RELIEF_THEME
    RELIEF_THEME = name
    DARK.update(THEMES[name]["ui"])
    _RELIEF_CACHE.clear()


def fmt_pill(dt):
    return (
        f"{DAYS_SHORT[dt.weekday()].capitalize()} "
        f"{dt.day} {MONTHS_SHORT[dt.month-1]} {dt:%H:%M}"
    )


def fmt_long(dt):
    return (
        f"{DAYS[dt.weekday()].capitalize()} "
        f"{dt.day} {MONTHS[dt.month-1]} {dt:%H:%M}"
    )


def fmt_short(dt):
    return (
        f"{DAYS_SHORT[dt.weekday()]} "
        f"{dt.day} {MONTHS[dt.month-1]} {dt:%H:%M}"
    )


def region_title(area):
    return "Alpen" if area == "alps" else "Benelux & Alpen"


def snow_dark_cmap():
    """De vertrouwde Sneeuwhoogte-kleurschaal (SNOW_LEVELS/SNOW_COLORS)."""

    cmap = ListedColormap(SNOW_COLORS, name="snow_classic")
    cmap.set_over("#969696")
    cmap.set_under((0, 0, 0, 0))
    cmap.set_bad((0, 0, 0, 0))
    return cmap, BoundaryNorm(SNOW_LEVELS, cmap.N)


_DARK_LOGO = None


def dark_logo():
    """Logo met witte tekst; het icoon blijft in de originele kleuren."""

    global _DARK_LOGO

    if _DARK_LOGO is None:

        if not LOGO_PATH.is_file():
            raise FileNotFoundError(f"Logo niet gevonden: {LOGO_PATH}")

        img = plt.imread(LOGO_PATH).astype(float)

        if img.max() > 1.0:
            img /= 255.0

        if img.shape[2] == 3:
            img = np.dstack([img, np.ones(img.shape[:2])])

        from scipy.ndimage import label

        rgb = img[..., :3]
        darkness = 1.0 - rgb.min(axis=2)          # 0 = wit
        ink = np.clip(darkness / max(darkness.max(), 1e-6), 0, 1)

        # Witte achtergrond die aan de rand grenst -> transparant.
        near_white = darkness < 0.12
        lab, _ = label(near_white)
        edge = np.unique(np.concatenate([
            lab[0], lab[-1], lab[:, 0], lab[:, -1]
        ]))
        background = np.isin(lab, edge[edge > 0])

        alpha = img[..., 3].copy()
        alpha[background] = 0.0

        # Tekst (rechts van het icoon) wit, met anti-aliasing via 'ink'.
        cut = int(img.shape[1] * 0.22)
        img[:, cut:, :3] = 1.0
        alpha[:, cut:] = np.minimum(img[:, cut:, 3], ink[:, cut:])

        # Zachte rand rond het icoon.
        icon_edge = ~background[:, :cut]
        alpha[:, :cut] = np.where(
            icon_edge, img[:, :cut, 3],
            0.0
        )

        img[..., 3] = alpha
        _DARK_LOGO = img

    return _DARK_LOGO


_RELIEF_CACHE = {}


_DARK_TERRAIN = None


def set_dark_terrain(x, y, terrain):
    """HSURF doorgeven aan de achtergrond (hoogtetint)."""

    global _DARK_TERRAIN
    _DARK_TERRAIN = None if terrain is None else (x, y, terrain)


def _lerp_stops(v, stops):
    """Kleurovergang over (positie, hexkleur)-stops; v in 0..1."""

    pos = np.array([p for p, _ in stops])
    cols = np.array([
        [int(c[i:i + 2], 16) / 255.0 for i in (1, 3, 5)]
        for _, c in stops
    ])
    return np.stack(
        [np.interp(v, pos, cols[:, k]) for k in range(3)],
        axis=-1
    )


def dark_relief(ax, tiler):
    """Nachtelijk 'ijs'-reliëf: ESRI-schaduw x HSURF-hoogtetint,
    ijzige highlights op de zonhellingen en een vignet."""

    key = tuple(np.round(ax.get_extent(ccrs.PlateCarree()), 4))

    if key not in _RELIEF_CACHE:

        domain = ax._get_extent_geom(tiler.crs)
        img, extent, origin = tiler.image_for_domain(domain, ESRI_ZOOM)

        g = np.asarray(img, dtype=float)[..., :3].mean(axis=2) / 255.0
        lo, hi = np.percentile(g, [0.5, 99.8])
        g = np.clip((g - lo) / max(hi - lo, 1e-6), 0, 1)

        # Vlak terrein in ESRI is ~middengrijs; schaduw<->licht rond 0.
        flat = np.median(g)
        light = np.clip((g - flat) / max(1 - flat, 1e-6), 0, 1)
        shade = np.clip((flat - g) / max(flat, 1e-6), 0, 1)

        # Hoogte per pixel (Mercator -> lon/lat is scheidbaar).
        elev = np.zeros_like(g)
        if _DARK_TERRAIN is not None:
            tx, ty, tz = _DARK_TERRAIN
            ny, nx = g.shape
            x0, x1, y0, y1 = extent
            R = 6378137.0
            px = np.linspace(x0, x1, nx)
            py = np.linspace(y1, y0, ny) if origin == "upper" \
                else np.linspace(y0, y1, ny)
            lon = np.degrees(px / R)
            lat = np.degrees(np.arctan(np.sinh(py / R)))
            spline = RectBivariateSpline(
                ty, tx, np.nan_to_num(tz, nan=0.0), kx=1, ky=1
            )
            elev = np.clip(spline(lat, lon), 0, 4500)

        e = np.clip(elev / 3200.0, 0, 1) ** 0.7

        th = THEMES[RELIEF_THEME]
        base = _lerp_stops(e, th["stops"])

        L = light[..., None]
        S = shade[..., None]
        rgb = base * (1.0 - th["shade"] * S) * (1.0 + th["light_gain"] * L)

        # Glans op de zonhellingen, sterker in het hooggebergte.
        g0, g1 = th["glint"]
        rgb += np.array(th["ice"]) * (L ** 2.2) * (g0 + g1 * e[..., None])

        _RELIEF_CACHE[key] = (np.clip(rgb, 0, 1), extent, origin)

    rgb, extent, origin = _RELIEF_CACHE[key]

    ax.imshow(
        rgb, extent=extent, origin=origin,
        transform=tiler.crs, interpolation="bilinear",
        regrid_shape=3000, zorder=0
    )

    return True


def add_vignette(ax):
    """Donkere randen voor diepte (onder de datalaag)."""

    yy, xx = np.mgrid[0:1:300j, 0:1:600j]
    r = np.hypot((xx - 0.5) / 0.62, (yy - 0.5) / 0.66)
    a = np.clip((r - 0.55) / 0.6, 0, 1) ** 1.6 * 0.55

    rgba = np.zeros(r.shape + (4,))
    rgba[..., :3] = THEMES[RELIEF_THEME]["vignette"]
    rgba[..., 3] = a

    ax.imshow(
        rgba, extent=list(ax.get_extent(ccrs.PlateCarree())),
        transform=ccrs.PlateCarree(),
        origin="lower", interpolation="bilinear", zorder=2.5
    )


def dark_axes(extent):

    proj = ccrs.PlateCarree()
    w, e, s, n = extent

    map_h = FIG_WIDTH_IN / ((e - w) / (n - s))
    H = HEADER_IN + map_h + FOOTER_IN

    fig = plt.figure(
        figsize=(FIG_WIDTH_IN, H),
        facecolor=DARK["bg"]
    )

    ax = fig.add_axes(
        [0, FOOTER_IN / H, 1, map_h / H],
        projection=proj
    )

    ax.set_extent(extent, crs=proj)
    ax.set_facecolor(DARK["sea"])
    ax.spines["geo"].set_visible(False)

    relief_done = False

    if USE_ESRI_RELIEF:
        TILE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tiler = EsriWorldShadedRelief(cache=str(TILE_CACHE_DIR))

        try:
            relief_done = dark_relief(ax, tiler)
        except Exception as exc:
            print(f"Donker reliëf mislukt ({exc}); eenvoudige tint.")

        if not relief_done:
            ax.add_image(tiler, ESRI_ZOOM, zorder=0)
            ax.add_patch(Rectangle(
                (0, 0), 1, 1,
                transform=ax.transAxes,
                facecolor=DARK["relief_tint"],
                edgecolor="none",
                alpha=DARK["relief_alpha"],
                zorder=1
            ))
    else:
        ax.add_feature(cfeature.LAND, facecolor="#1c2635", zorder=0)

    for feature in (cfeature.OCEAN, cfeature.LAKES):
        ax.add_feature(
            feature,
            facecolor=DARK["sea"],
            edgecolor="none",
            alpha=0.92,
            zorder=1.5
        )

    add_vignette(ax)

    for yy in (FOOTER_IN / H, (FOOTER_IN + map_h) / H):
        fig.add_artist(Line2D(
            [0, 1], [yy, yy],
            transform=fig.transFigure,
            color=DARK["line"],
            linewidth=1.0,
            zorder=300
        ))

    return fig, ax, proj


def _calendar_icon(fig, cx, cy, size):
    """Klein kalendericoon, coördinaten in inch."""

    t = fig.dpi_scale_trans
    col = DARK["text"]
    w = size
    h = size * 0.9
    x0, y0 = cx - w / 2, cy - h / 2

    fig.add_artist(FancyBboxPatch(
        (x0, y0), w, h,
        boxstyle=f"round,pad=0,rounding_size={size * 0.12}",
        transform=t, facecolor="none", edgecolor=col,
        linewidth=1.6, zorder=310
    ))

    fig.add_artist(Line2D(
        [x0, x0 + w], [y0 + h * 0.72] * 2,
        transform=t, color=col, linewidth=1.6, zorder=310
    ))

    for fx in (0.28, 0.72):
        fig.add_artist(Line2D(
            [x0 + w * fx] * 2, [y0 + h * 0.86, y0 + h * 1.12],
            transform=t, color=col, linewidth=1.8,
            solid_capstyle="round", zorder=310
        ))

    for row in (0.48, 0.24):
        for fx in (0.25, 0.5, 0.75):
            fig.add_artist(Rectangle(
                (x0 + w * fx - size * 0.05, y0 + h * row - size * 0.04),
                size * 0.10, size * 0.08,
                transform=t, facecolor=col, edgecolor="none", zorder=310
            ))


def dark_header(fig, title, subtitle, date_text):

    W, H = fig.get_size_inches()
    yc_in = H - HEADER_IN / 2
    yc = yc_in / H
    renderer = fig.canvas.get_renderer()

    # Logo links
    img = dark_logo()
    logo_h_pt = 38.0
    zoom = logo_h_pt / img.shape[0]
    logo_w_in = img.shape[1] * zoom / 72.0

    fig.add_artist(AnnotationBbox(
        OffsetImage(img, zoom=zoom, resample=True),
        (0.28 / W, yc),
        xycoords="figure fraction",
        box_alignment=(0, 0.5),
        frameon=False, pad=0, zorder=310
    ))

    # Scheidingslijn
    x_div = 0.28 + logo_w_in + 0.38
    fig.add_artist(Line2D(
        [x_div / W] * 2,
        [(yc_in - 0.27) / H, (yc_in + 0.27) / H],
        transform=fig.transFigure,
        color="#9aa6b8", linewidth=1.2, zorder=310
    ))

    # Titel + ondertitel
    x_txt = (x_div + 0.38) / W
    if subtitle:
        fig.text(
            x_txt, (yc_in + 0.01) / H, title,
            color=DARK["text"], fontsize=25, fontweight="bold",
            ha="left", va="bottom", zorder=310
        )
        fig.text(
            x_txt, (yc_in - 0.05) / H, subtitle,
            color=DARK["subtext"], fontsize=14,
            ha="left", va="top", zorder=310
        )
    else:
        fig.text(
            x_txt, yc, title,
            color=DARK["text"], fontsize=27, fontweight="bold",
            ha="left", va="center_baseline", zorder=310
        )

    # Datum-pil rechts
    x_right = W - 0.48
    date = fig.text(
        x_right / W, yc, date_text,
        color=DARK["text"], fontsize=17, fontweight="bold",
        ha="right", va="center", zorder=310
    )

    bb = date.get_window_extent(renderer)
    text_w_in = bb.width / fig.dpi
    icon = 0.30
    pill_h = 0.58
    pill_x0 = x_right - text_w_in - icon - 0.52
    pill_w = W - 0.22 - pill_x0

    fig.add_artist(FancyBboxPatch(
        (pill_x0, yc_in - pill_h / 2), pill_w, pill_h,
        boxstyle=f"round,pad=0,rounding_size={pill_h / 2}",
        transform=fig.dpi_scale_trans,
        facecolor="#0f1826", edgecolor=DARK["pill_edge"],
        linewidth=1.2, zorder=305
    ))

    _calendar_icon(fig, pill_x0 + 0.28 + icon / 2, yc_in, icon)


def footer_source_text(filepath, run):
    """Display the correct model credit for each rendered map.

    Hybrid's ``run`` is the aligned forecast start, not a single model run.
    Use the output filename generated by each existing model renderer to
    identify the source without changing model calculations or downloads.
    """
    filename = Path(filepath).name.lower() if filepath is not None else ""
    if filename.startswith("ecmwf_"):
        return f"ECMWF - run {run:%d-%m-%Y %H} UTC"
    if filename.startswith("icon_d2_"):
        return f"ICON-D2 (DWD) - run {run:%d-%m-%Y %H} UTC"
    if filename.startswith("hybrid_"):
        return f"Hybrid (meerdere modellen) - vanaf {run:%d-%m-%Y %H} UTC"
    return f"Modelbron onbekend - {run:%d-%m-%Y %H} UTC"


def dark_footer(fig, cmap, norm, label, ticks, ticklabels, run, filepath=None):

    W, H = fig.get_size_inches()
    renderer = fig.canvas.get_renderer()
    yc_in = FOOTER_IN * 0.62

    fig.text(
        0.36 / W, yc_in / H, "❄",
        color=DARK["text"], fontsize=27,
        fontfamily="DejaVu Sans",
        ha="left", va="center", zorder=310
    )

    lab = fig.text(
        0.86 / W, yc_in / H, label,
        color=DARK["text"], fontsize=17,
        ha="left", va="center", zorder=310
    )

    lab_x1 = lab.get_window_extent(renderer).x1 / fig.dpi

    bar_x0 = lab_x1 + 0.38
    bar_x1 = W - 0.55
    bar_h = 0.27

    cax = fig.add_axes([
        bar_x0 / W, (yc_in - bar_h / 2) / H,
        (bar_x1 - bar_x0) / W, bar_h / H
    ])

    cb = fig.colorbar(
        ScalarMappable(norm=norm, cmap=cmap),
        cax=cax,
        orientation="horizontal",
        spacing="uniform",
        extend="neither"
    )

    cb.outline.set_visible(False)
    cax.patch.set_visible(False)
    cb.set_ticks(ticks)
    cb.set_ticklabels(ticklabels)
    cb.ax.tick_params(
        length=0, pad=9, labelsize=14 if len(ticks) <= 12 else 12,
        labelcolor=DARK["text"]
    )

    for lbl in cb.ax.get_xticklabels():
        lbl.set_fontweight("bold")

    # Afgeronde uiteinden
    clip = FancyBboxPatch(
        (bar_x0, yc_in - bar_h / 2), bar_x1 - bar_x0, bar_h,
        boxstyle=f"round,pad=0,rounding_size={bar_h / 2}",
        transform=fig.dpi_scale_trans
    )
    cb.solids.set_clip_path(clip)

    fig.text(
        (W - 0.22) / W, 0.10 / H,
        footer_source_text(filepath, run),
        color="#7d8899", fontsize=9,
        ha="right", va="bottom", zorder=310
    )


def finish_map_dark(
    fig, ax, title, subtitle, date_text,
    cbar_label, cmap, norm, ticks, ticklabels,
    filepath, extent, area, run
):

    gc = THEMES[RELIEF_THEME]["glow"]
    glow = [
        pe.Stroke(linewidth=4.5, foreground=(*gc, 0.08)),
        pe.Stroke(linewidth=2.4, foreground=(*gc, 0.18)),
        pe.Normal(),
    ]

    for feature, lw in ((cfeature.COASTLINE, 0.9), (cfeature.BORDERS, 0.9)):
        art = ax.add_feature(
            feature,
            linewidth=lw,
            edgecolor="#eef6ff",
            alpha=0.95,
            zorder=25
        )
        art.set_path_effects(glow)

    dark_header(fig, title, "", date_text)  # Geen subtitel op de kaarten
    dark_footer(fig, cmap, norm, cbar_label, ticks, ticklabels, run, filepath=filepath)
    add_cities(fig, ax, extent, area, None)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        filepath,
        dpi=DPI,
        facecolor=DARK["bg"]
    )

    plt.close(fig)
    print("Opgeslagen:", filepath.name)


def shade_snow_rgba(xf, yf, zf, x, y, terrain):
    """Sneeuwkleuren met reliëfschaduw uit HSURF (zoals de voorbeeldkaart)."""

    cmap, norm = snow_dark_cmap()
    vals = np.ma.filled(zf.astype(float), np.nan)
    finite = np.isfinite(vals)

    rgba = cmap(norm(np.where(finite, vals, 0.0)))

    if terrain is not None:

        interp = RegularGridInterpolator(
            (y, x),
            np.nan_to_num(terrain, nan=0.0),
            method="linear",
            bounds_error=False,
            fill_value=0.0
        )

        X, Y = np.meshgrid(xf, yf)
        elev = interp(
            np.column_stack([Y.ravel(), X.ravel()])
        ).reshape(Y.shape)

        step = float(xf[1] - xf[0])
        dx = step * 111320.0 * np.cos(np.radians(yf.mean()))
        dy = step * 110540.0

        ls = LightSource(azdeg=315, altdeg=38)

        # Rijen lopen zuid->noord; hillshade verwacht noord bovenaan.
        inten = ls.hillshade(
            elev[::-1], vert_exag=2.5, dx=dx, dy=dy
        )[::-1]

        rgb = ls.blend_soft_light(rgba[..., :3], inten[..., None])
        rgb = rgb * (0.80 + 0.30 * inten[..., None])
        rgba[..., :3] = np.clip(rgb, 0, 1)

    alpha = np.clip(
        0.62 + 0.38 * (vals - SNOW_MIN_CM) / 2.5,
        0.62, 1.0
    ) * 0.96

    alpha[~finite] = 0.0
    rgba[..., 3] = alpha

    return rgba

