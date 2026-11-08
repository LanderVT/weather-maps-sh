
"""ICON-D2 temperature at 1500, 2000 and 3000 m a.s.l. for Sneeuwhoogte.nl.

Standalone add-on to run_icon_d2.py, not a replacement for SnowAll_ICON-D2.py.
Uses T and FI at 850/700 hPa (4 GRIB files per forecast step),
interpolated/extrapolated to the requested heights using pressure-level
geopotential heights.

For each grid cell, T(h) is approximated by a straight line between the
two pressure levels. This is not a full model-level interpolation.
Areas below the model surface are masked.

Forecast hours: +0, +6, ..., +48.
Map style: dark/slate, discrete temperature classes.
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

# 25 discrete classes, each exactly 2 degrees wide.
TEMP_LEVELS_C = np.arange(-30, 22, 2, dtype=float)

# Keep the preferred legend with labels every 5 degrees.
TEMP_TICKS_C = list(range(-30, 21, 5))

# Approved transparency.
TEMP_ALPHA = 0.91

# Mountain masking and hatching.
MASK_GRAY = "#c8cdd3"
HATCH_PATTERN = "///"
HATCH_COLOR = (0.23, 0.29, 0.37, 0.52)
HATCH_WIDTH = 0.65


# ============================================================
# 2. DWD PRESSURE-LEVEL DATA
# ============================================================

def pressure_level_url(
    mod, run: datetime, lead: int, level: int, parameter: str
):
    """Construct the DWD ICON-D2 pressure-level GRIB URL."""

    if level not in PRESSURE_LEVELS or parameter not in ("t", "fi"):
        raise ValueError(
            f"Unsupported pressure-level field {parameter}/{level}"
        )

    filename = (
        "icon-d2_germany_regular-lat-lon_pressure-level_"
        f"{run:%Y%m%d%H}_{lead:03d}_{level}_{parameter}.grib2.bz2"
    )

    url = f"{mod.DATA_URL}/{run:%H}/{parameter}/{filename}"

    return url, filename


# ============================================================
# 3. UNIT CONVERSIONS
# ============================================================

def kelvin_to_celsius(raw, unit: str):
    """Convert temperature to Celsius."""

    data = np.asarray(raw, dtype=float)
    unit = unit.strip().lower()

    if unit in (
        "k", "kelvin", "degrees kelvin", "degree kelvin"
    ):
        data = data - 273.15

    elif unit in (
        "c", "degc", "degree celsius",
        "degrees celsius", "celsius", "°c"
    ):
        pass

    elif not unit:
        if np.nanmedian(data) > 100:
            data = data - 273.15

    else:
        raise ValueError(
            f"Unexpected DWD temperature unit: {unit!r}"
        )

    return data


def geopotential_to_metres(raw, unit: str):
    """Convert geopotential to geopotential height in metres."""

    data = np.asarray(raw, dtype=float)

    unit = unit.strip().lower().replace(" ", "")
    unit = unit.replace("²", "2").replace("−", "-")

    if unit in (
        "m", "gpm", "geopotentialmetre", "geopotentialmeter"
    ):
        return data

    if unit in (
        "m**2s**-2", "m2s-2", "m2/s2",
        "m^2s^-2", "m**2s-2"
    ):
        return data / GRAVITY

    finite = data[np.isfinite(data)]

    if finite.size == 0:
        raise ValueError("All geopotential values are missing")

    typical = float(np.median(finite))

    if 6000 < typical < 90000:
        return data / GRAVITY

    if 500 < typical < 5000:
        return data

    raise ValueError(
        f"Cannot interpret FI units {unit!r}; median={typical:g}"
    )


# ============================================================
# 4. VERTICAL TEMPERATURE INTERPOLATION
# ============================================================

def calculate_fixed_height_temperature(
    t850_c,
    t700_c,
    z850_m,
    z700_m,
    target_m: int,
    hsurf_m,
):
    """Interpolate temperatures between 850 and 700 hPa.

    The same pressure-level data provide temperatures at
    1500, 2000 and 3000 metres.

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
            t700_c, z850_m, z700_m, hsurf_m
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
# 5. TEMPERATURE COLOURS
# ============================================================

def temperature_colormap():
    """25 discrete classes with stronger cold and warm contrasts.

    Cold side:
      Deep violet -> purple -> blue -> light blue

    Near freezing:
      Clearly separated blue and yellow classes

    Warm side:
      Yellow -> orange -> red -> dark red

    All class widths remain exactly 2 degrees Celsius.
    """

    # Exactly 25 colours for 25 temperature intervals.
    colors = [
        "#37105d",  # -30 to -28: dark violet
        "#501574",  # -28 to -26
        "#63198b",  # -26 to -24
        "#7022a2",  # -24 to -22
        "#6c2fb4",  # -22 to -20
        "#5f3fc4",  # -20 to -18
        "#514dcc",  # -18 to -16
        "#405dd1",  # -16 to -14: purple-blue
        "#346cd6",  # -14 to -12
        "#2a78d8",  # -12 to -10
        "#2585d9",  # -10 to -8
        "#258fdc",  #  -8 to -6

        # More contrast between the blue classes near 0 C.
        "#3b9fdd",  #  -6 to -4: medium blue
        "#75bce9",  #  -4 to -2: sky blue
        "#b2dcf3",  #  -2 to  0: light icy blue

        # Faster transition from yellow towards red.
        "#ffe39a",  #   0 to  2: yellow
        "#ffda70",  #   2 to  4: golden yellow
        "#ffc653",  #   4 to  6
        "#f9ab42",  #   6 to  8: orange
        "#f28b37",  #   8 to 10
        "#e96c32",  #  10 to 12: orange-red
        "#dc4f33",  #  12 to 14: red
        "#c93833",  #  14 to 16
        "#a82437",  #  16 to 18: dark red
        "#80163b",  #  18 to 20: deepest red
    ]

    if len(colors) != len(TEMP_LEVELS_C) - 1:
        raise ValueError(
            "Number of temperature colours does not match classes"
        )

    cmap = mcolors.ListedColormap(
        colors,
        name="icon_d2_temperature_bold_discrete_2c",
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
# 6. MAP PLOTTING
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
    """Render discrete temperature classes on dark terrain."""

    fig, ax, proj = style.dark_axes(extent)

    cmap, norm = temperature_colormap()

    values = np.ma.masked_invalid(field_c)

    # Temperature classes.
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
    # GREY MOUNTAIN MASK AND DIAGONAL HATCHING
    # --------------------------------------------------------

    above_terrain = np.asarray(
        terrain_mask,
        dtype=bool,
    )

    if np.any(above_terrain):

        mask_field = above_terrain.astype(float)

        # Light-grey base.
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

        # Diagonal hatching.
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

        # Compatible with different Matplotlib versions.
        if hasattr(hatch, "set_edgecolor"):

            hatch.set_edgecolor(HATCH_COLOR)
            hatch.set_linewidth(0.0)

        else:

            for collection in hatch.collections:

                collection.set_edgecolor(HATCH_COLOR)
                collection.set_linewidth(0.0)

    # --------------------------------------------------------
    # FORECAST TIME
    # --------------------------------------------------------

    valid_utc = run + timedelta(hours=lead)
    valid_local = valid_utc.astimezone(mod.LOCAL_TZ)

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
    # DARK HEADER, FOOTER AND REGULAR LEGEND
    # --------------------------------------------------------

    style.finish_map_dark(
        fig,
        ax,
        f"Temperatuur {target_m} m – {style.region_title(area)}",
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
# 7. DOWNLOAD DATA AND GENERATE TEMPERATURE MAPS
# ============================================================

def produce_temperature_maps(mod, run, areas, extents):
    """Generate 1500, 2000 and 3000 m maps from shared DWD data."""

    steps = tuple(mod.FORECAST_HOURS)

    if steps != tuple(range(0, 49, 6)):
        raise ValueError(
            "ICON-D2 temperature expects steps 0..48 every 6 hours"
        )

    if any(a not in extents for a in areas):
        raise ValueError(
            f"No recorded ICON-D2 map extent for: "
            f"{set(areas) - set(extents)}"
        )

    output = []

    with requests.Session() as session:

        session.headers.update({
            "User-Agent": "SneeuwhoogteNL-ICON-D2-temperature/1.0",
        })

        # Reuse surface height from existing model cache.
        hfield = mod.load_hsurf(session, run)

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

        # Loop over forecast times.
        for lead in steps:

            data = {}
            grid = None

            # Four pressure-level GRIB fields per forecast time.
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
                        else ["fi", "z", "gh", "geopotential"]
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

            x, y = grid

            # Convert DWD temperature and geopotential fields.
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

            # Generate all three heights from the same fields.
            for target_m in HEIGHTS_M:

                field_c, mask = calculate_fixed_height_temperature(
                    t850,
                    t700,
                    z850,
                    z700,
                    target_m,
                    terrain,
                )

                coverage = np.isfinite(field_c).sum()

                if coverage < 500:
                    raise RuntimeError(
                        f"T {target_m} m +{lead}: "
                        f"insufficient valid cells ({coverage})"
                    )

                print(
                    f"ICON-D2 T {target_m} m "
                    f"+{lead:03d}: {coverage} valid cells",
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
