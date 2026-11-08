
"""ICON-D2 temperature maps at 1500, 2000 and 3000 m.

Standalone add-on to run_icon_d2.py.

Uses T and FI at 850 and 700 hPa.
All three heights share the same four GRIB downloads per forecast step.

Forecast hours: 0, 6, 12, ..., 48.

Map style:
- Original blue/yellow temperature colours
- 25 equal discrete classes of 2 degrees Celsius
- Regularly spaced colourbar labels
- Subtle hatched grey mountain mask
- Dark/slate Sneeuwhoogte.nl layout
"""

from __future__ import annotations

from datetime import datetime, timedelta

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import requests

import map_style_reference as style


# ============================================================
# 1. SETTINGS
# ============================================================

GRAVITY = 9.80665

HEIGHTS_M = (1500, 2000, 3000)
PRESSURE_LEVELS = (850, 700)

# IMPORTANT: equal 2-degree classes.
# 26 boundaries produce 25 equally sized colour blocks.
TEMP_LEVELS_C = np.arange(-30, 22, 2, dtype=float)

# Labels every 5 degrees, as in the preferred legend.
TEMP_TICKS_C = list(range(-30, 21, 5))

# Preserve approved transparency.
TEMP_ALPHA = 0.91

# Mountain mask.
MASK_GRAY = "#c8cdd3"

# Subtle diagonal hatch above the selected elevation.
HATCH_PATTERN = "///"
HATCH_COLOR = (0.23, 0.29, 0.37, 0.52)
HATCH_WIDTH = 0.65


# ============================================================
# 2. DWD PRESSURE-LEVEL DATA
# ============================================================

def pressure_level_url(
    mod,
    run: datetime,
    lead: int,
    level: int,
    parameter: str,
):
    """Construct DWD pressure-level GRIB URL."""

    if level not in PRESSURE_LEVELS:
        raise ValueError(
            f"Unsupported pressure level: {level}"
        )

    if parameter not in ("t", "fi"):
        raise ValueError(
            f"Unsupported parameter: {parameter}"
        )

    filename = (
        "icon-d2_germany_regular-lat-lon_pressure-level_"
        f"{run:%Y%m%d%H}_{lead:03d}_"
        f"{level}_{parameter}.grib2.bz2"
    )

    url = (
        f"{mod.DATA_URL}/{run:%H}/"
        f"{parameter}/{filename}"
    )

    return url, filename


# ============================================================
# 3. UNIT CONVERSIONS
# ============================================================

def kelvin_to_celsius(raw, unit: str):
    """Convert temperature to Celsius."""

    data = np.asarray(raw, dtype=float)
    unit = unit.strip().lower()

    if unit in (
        "k",
        "kelvin",
        "degrees kelvin",
        "degree kelvin",
    ):
        data = data - 273.15

    elif unit in (
        "c",
        "degc",
        "degree celsius",
        "degrees celsius",
        "celsius",
        "°c",
    ):
        pass

    elif not unit:
        if np.nanmedian(data) > 100:
            data = data - 273.15

    else:
        raise ValueError(
            f"Unexpected temperature unit: {unit!r}"
        )

    return data


def geopotential_to_metres(raw, unit: str):
    """Convert geopotential into height in metres."""

    data = np.asarray(raw, dtype=float)

    unit = unit.strip().lower().replace(" ", "")
    unit = unit.replace("²", "2").replace("−", "-")

    if unit in (
        "m",
        "gpm",
        "geopotentialmetre",
        "geopotentialmeter",
    ):
        return data

    if unit in (
        "m**2s**-2",
        "m2s-2",
        "m2/s2",
        "m^2s^-2",
        "m**2s-2",
    ):
        return data / GRAVITY

    finite = data[np.isfinite(data)]

    if finite.size == 0:
        raise ValueError(
            "All geopotential values are missing"
        )

    typical = float(np.median(finite))

    if 6000 < typical < 90000:
        return data / GRAVITY

    if 500 < typical < 5000:
        return data

    raise ValueError(
        f"Cannot interpret FI units {unit!r}; "
        f"median={typical:g}"
    )


# ============================================================
# 4. TEMPERATURE INTERPOLATION
# ============================================================

def calculate_fixed_height_temperature(
    t850_c,
    t700_c,
    z850_m,
    z700_m,
    target_m: int,
    hsurf_m,
):
    """Interpolate temperature between 850 and 700 hPa.

    The same vertical profile is used for 1500, 2000
    and 3000 metres.

    Small extrapolations are allowed.

    Temperatures below model terrain are masked.
    """

    t850_c = np.asarray(t850_c, dtype=float)
    t700_c = np.asarray(t700_c, dtype=float)

    z850_m = np.asarray(z850_m, dtype=float)
    z700_m = np.asarray(z700_m, dtype=float)

    hsurf_m = np.asarray(hsurf_m, dtype=float)

    if not all(
        a.shape == t850_c.shape
        for a in (
            t700_c,
            z850_m,
            z700_m,
            hsurf_m,
        )
    ):
        raise ValueError(
            "DWD pressure/surface fields do not share a grid"
        )

    dz = z700_m - z850_m

    valid = (
        np.isfinite(t850_c)
        & np.isfinite(t700_c)
        & np.isfinite(z850_m)
        & np.isfinite(z700_m)
        & (dz > 500)
        & (dz < 2500)
    )

    frac = np.divide(
        target_m - z850_m,
        dz,
        out=np.full_like(dz, np.nan),
        where=valid,
    )

    result = t850_c + frac * (t700_c - t850_c)

    nearest_distance = np.minimum(
        np.abs(target_m - z850_m),
        np.abs(target_m - z700_m),
    )

    within_pressure_profile = (
        (target_m >= z850_m)
        & (target_m <= z700_m)
    )

    valid &= (
        within_pressure_profile
        | (nearest_distance <= 600.0)
    )

    terrain_mask = (
        np.isfinite(hsurf_m)
        & (hsurf_m >= target_m)
    )

    result[~valid | terrain_mask] = np.nan

    return result, terrain_mask


# ============================================================
# 5. DISCRETE TEMPERATURE COLOURS
# ============================================================

def temperature_colormap():
    """Original blue/yellow palette with equal 2 C classes.

    Exactly 25 colour blocks.

    The two classes around freezing:
      -2 to 0 C: light blue
       0 to 2 C: warm yellow

    No extra narrow class around zero.
    """

    # Original first-version colour anchors.
    anchors = [
        (0.00, "#311d62"),   # Deep violet
        (0.15, "#2a4caa"),   # Royal blue
        (0.30, "#237fc3"),   # Blue
        (0.45, "#74c5ed"),   # Light blue
        (0.53, "#9dd8f4"),   # Icy blue
        (0.58, "#76bde8"),   # Cool blue
        (0.625, "#f0f1eb"),  # Neutral transition
        (0.66, "#ffe28b"),   # Yellow
        (0.73, "#ffd17b"),   # Golden yellow
        (0.86, "#f7a04f"),   # Orange
        (1.00, "#d73b43"),   # Red
    ]

    # Continuous original palette used only to obtain
    # colours at the CENTRE of each discrete class.
    base = mcolors.LinearSegmentedColormap.from_list(
        "original_temperature_palette",
        anchors,
        N=1024,
    )

    # Equal-width intervals:
    # -30..-28, -28..-26, ... 18..20.
    centres = (
        TEMP_LEVELS_C[:-1] + TEMP_LEVELS_C[1:]
    ) / 2.0

    colors = [
        mcolors.to_hex(
            base((t + 30.0) / 50.0)
        )
        for t in centres
    ]

    # Slightly clearer colours either side of freezing.
    # One blue class, immediately followed by one yellow class.
    index_cold = int(
        np.flatnonzero(
            np.isclose(centres, -1.0)
        )[0]
    )

    index_warm = int(
        np.flatnonzero(
            np.isclose(centres, 1.0)
        )[0]
    )

    colors[index_cold] = "#b8ddf3"
    colors[index_warm] = "#ffe9b0"

    # Discrete colormap: exactly 25 colours.
    cmap = mcolors.ListedColormap(
        colors,
        name="icon_d2_temperature_discrete_2c",
    )

    cmap.set_bad((0, 0, 0, 0))
    cmap.set_under(colors[0])
    cmap.set_over(colors[-1])

    norm = mcolors.BoundaryNorm(
        TEMP_LEVELS_C,
        ncolors=cmap.N,
    )

    return cmap, norm


# ============================================================
# 6. TEMPERATURE MAP PLOTTING
# ============================================================

def render_temperature_map(
    mod,
    x,
    y,
    field_c,
    terrain_mask,
    extent,
    area,
    run,
    lead,
    target_m,
):
    """Render discrete temperature classes and hatched terrain."""

    fig, ax, proj = style.dark_axes(extent)

    cmap, norm = temperature_colormap()

    values = np.ma.masked_invalid(field_c)

    # --------------------------------------------------------
    # DISCRETE TEMPERATURE CLASSES
    # --------------------------------------------------------

    ax.contourf(
        x,
        y,
        values,
        levels=TEMP_LEVELS_C,
        cmap=cmap,
        norm=norm,
        extend="both",
        transform=proj,
        alpha=TEMP_ALPHA,
        zorder=3,
        antialiased=False,
    )

    # --------------------------------------------------------
    # GREY MOUNTAIN MASK WITH DIAGONAL HATCHING
    # --------------------------------------------------------

    # These are places where model terrain exceeds
    # the selected temperature altitude.
    #
    # Example:
    # On a 1500 m temperature map, terrain above 1500 m
    # is shown grey with diagonal stripes.
    #
    # This makes grey areas distinguishable from
    # the light-blue temperature classes.

    above_terrain = np.asarray(
        terrain_mask,
        dtype=bool,
    )

    if np.any(above_terrain):

        mask_field = above_terrain.astype(float)

        # Base grey mountain fill.
        ax.contourf(
            x,
            y,
            mask_field,
            levels=[0.5, 1.5],
            colors=[MASK_GRAY],
            transform=proj,
            zorder=6,
            antialiased=False,
        )

        # Subtle diagonal hatch overlay.
        with plt.rc_context({
            "hatch.linewidth": HATCH_WIDTH,
        }):

            hatch = ax.contourf(
                x,
                y,
                mask_field,
                levels=[0.5, 1.5],
                colors="none",
                hatches=[HATCH_PATTERN],
                transform=proj,
                zorder=7,
                antialiased=False,
            )

        # Support different Matplotlib versions.
        if hasattr(hatch, "set_edgecolor"):

            hatch.set_edgecolor(HATCH_COLOR)
            hatch.set_linewidth(0.0)

        else:

            for collection in hatch.collections:

                collection.set_edgecolor(
                    HATCH_COLOR
                )

                collection.set_linewidth(0.0)

    # --------------------------------------------------------
    # FORECAST TIME
    # --------------------------------------------------------

    valid_utc = run + timedelta(hours=lead)

    valid_local = valid_utc.astimezone(
        mod.LOCAL_TZ
    )

    # --------------------------------------------------------
    # OUTPUT FILENAME
    # --------------------------------------------------------

    outfile = mod.OUTPUT_DIR / (
        f"icon_d2_temperatuur_{target_m}m_{area}_"
        f"{run:%Y%m%d_%H}z_"
        f"f{lead:03d}_"
        f"{valid_utc:%Y%m%d_%H}z.png"
    )

    # --------------------------------------------------------
    # DARK HEADER, FOOTER AND REGULAR COLOURBAR
    # --------------------------------------------------------

    style.finish_map_dark(
        fig,
        ax,
        f"Temperatuur {target_m} m – "
        f"{style.region_title(area)}",
        "",
        style.fmt_long(valid_local),
        "Temperatuur (°C)",
        cmap,
        norm,
        TEMP_TICKS_C,
        [str(t) for t in TEMP_TICKS_C],
        outfile,
        extent,
        area,
        run,
    )

    return outfile


# ============================================================
# 7. DOWNLOAD DATA AND GENERATE ALL TEMPERATURE MAPS
# ============================================================

def produce_temperature_maps(mod, run, areas, extents):
    """Generate 1500, 2000 and 3000 m temperature maps.

    All three heights use the same four fields per step:
      T850
      T700
      FI850
      FI700

    Forecast steps: 0, 6, ..., 48 h.
    """

    steps = tuple(mod.FORECAST_HOURS)

    if steps != tuple(range(0, 49, 6)):
        raise ValueError(
            "ICON-D2 temperature expects "
            "steps 0..48 every 6 hours"
        )

    if any(a not in extents for a in areas):
        raise ValueError(
            "No recorded ICON-D2 map extent for: "
            f"{set(areas) - set(extents)}"
        )

    output = []

    with requests.Session() as session:

        session.headers.update({
            "User-Agent":
                "SneeuwhoogteNL-ICON-D2-temperature/1.0",
        })

        # ----------------------------------------------------
        # SURFACE ELEVATION
        # ----------------------------------------------------

        hfield = mod.load_hsurf(
            session,
            run,
        )

        xsurf, ysurf, terrain_raw, terrain_unit = hfield

        terrain = np.asarray(
            terrain_raw,
            dtype=float,
        )

        terrain = np.where(
            (terrain >= -500)
            & (terrain <= 9000),
            terrain,
            np.nan,
        )

        style.set_dark_terrain(
            xsurf,
            ysurf,
            terrain,
        )

        # ----------------------------------------------------
        # FORECAST STEPS
        # ----------------------------------------------------

        for lead in steps:

            data = {}
            grid = None

            # -----------------------------------------------
            # DOWNLOAD FOUR PRESSURE-LEVEL FIELDS
            # -----------------------------------------------

            for level in PRESSURE_LEVELS:

                for param in ("t", "fi"):

                    url, filename = pressure_level_url(
                        mod,
                        run,
                        lead,
                        level,
                        param,
                    )

                    archive = mod.download_archive(
                        session,
                        url,
                        filename,
                    )

                    names = (
                        ["t", "temperature"]
                        if param == "t"
                        else [
                            "fi",
                            "z",
                            "gh",
                            "geopotential",
                        ]
                    )

                    result = mod.read_grib(
                        archive,
                        names,
                        lead,
                    )

                    if grid is None:

                        grid = result[:2]

                        if not (
                            np.array_equal(grid[0], xsurf)
                            and np.array_equal(grid[1], ysurf)
                        ):
                            raise ValueError(
                                "Pressure-level grid differs from HSURF"
                            )

                    else:

                        mod.assert_same_grid(
                            grid,
                            result,
                            f"{param.upper()} {level} hPa",
                        )

                    data[(param, level)] = result

            # -----------------------------------------------
            # CONVERT TEMPERATURE AND GEOPOTENTIAL
            # -----------------------------------------------

            x, y = grid

            t850 = kelvin_to_celsius(
                data[("t", 850)][2],
                data[("t", 850)][3],
            )

            t700 = kelvin_to_celsius(
                data[("t", 700)][2],
                data[("t", 700)][3],
            )

            z850 = geopotential_to_metres(
                data[("fi", 850)][2],
                data[("fi", 850)][3],
            )

            z700 = geopotential_to_metres(
                data[("fi", 700)][2],
                data[("fi", 700)][3],
            )

            # -----------------------------------------------
            # GENERATE ALL THREE TEMPERATURE HEIGHTS
            # -----------------------------------------------

            for target_m in HEIGHTS_M:

                field_c, mask = (
                    calculate_fixed_height_temperature(
                        t850,
                        t700,
                        z850,
                        z700,
                        target_m,
                        terrain,
                    )
                )

                coverage = np.isfinite(
                    field_c
                ).sum()

                if coverage < 500:
                    raise RuntimeError(
                        f"T {target_m} m +{lead}: "
                        "insufficient valid cells "
                        f"({coverage})"
                    )

                print(
                    f"ICON-D2 T {target_m} m "
                    f"+{lead:03d}: "
                    f"{coverage} valid cells",
                    flush=True,
                )

                for area in areas:

                    output.append(
                        render_temperature_map(
                            mod,
                            x,
                            y,
                            field_c,
                            mask,
                            extents[area],
                            area,
                            run,
                            lead,
                            target_m,
                        )
                    )

    return output
