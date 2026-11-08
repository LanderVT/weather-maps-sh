
"""ICON-D2 temperature at 1500, 2000 and 3000 m a.s.l.

Standalone add-on to run_icon_d2.py.

Uses T and FI at 850 and 700 hPa to calculate temperatures
at fixed elevations. All three heights share the same downloads.

Forecast hours: +0, +6, ..., +48.

Map style:
- Original blue / light blue / cream / yellow / orange palette
- DISCRETE temperature classes
- Same transparency as previous version (0.91)
- Grey masking where terrain is above the selected elevation
- Existing dark/slate map layout

Do not run directly. Called by run_icon_d2.py.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import matplotlib.colors as mcolors
import numpy as np
import requests

import map_style_reference as style


# ============================================================
# 1. SETTINGS
# ============================================================

GRAVITY = 9.80665

HEIGHTS_M = (1500, 2000, 3000)

PRESSURE_LEVELS = (850, 700)

MASK_GRAY = "#c8cdd3"

# Keep approved transparency
TEMP_ALPHA = 0.91

# Discrete temperature-class boundaries in degrees Celsius
TEMP_BOUNDS_C = np.array([
    -30.0,
    -25.0,
    -20.0,
    -15.0,
    -10.0,
    -7.5,
    -5.0,
    -2.5,
     0.0,
     2.5,
     5.0,
     7.5,
    10.0,
    12.5,
    15.0,
    20.0,
], dtype=float)

# Colorbar labels every 5 degrees
TEMP_TICKS_C = list(range(-30, 21, 5))


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
    """Construct the DWD ICON-D2 pressure-level GRIB URL."""

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
    """Convert geopotential into geopotential height in metres."""

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

    The same vertical temperature profile is used for:
      - 1500 m
      - 2000 m
      - 3000 m

    Small extrapolations are allowed.

    Temperatures below the model terrain are masked.
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
# 5. DISCRETE TEMPERATURE COLORMAP
# ============================================================

def temperature_colormap():
    """Discrete version of the approved blue-yellow palette.

    Colours:
      Cold: violet -> dark blue -> blue -> light blue
      Near 0 C: light blue -> cream/yellow
      Warm: yellow -> orange -> red

    No narrow extra class at 0 C.
    """

    # Exactly one colour for each temperature interval.
    # 16 boundaries = 15 colour classes.

    colors = [
        "#38268e",  # -30.0 to -25.0
        "#3049a9",  # -25.0 to -20.0
        "#2c69bf",  # -20.0 to -15.0
        "#3587ca",  # -15.0 to -10.0

        "#59a4d8",  # -10.0 to -7.5
        "#78b9e2",  #  -7.5 to -5.0
        "#98ccec",  #  -5.0 to -2.5
        "#bdddf0",  #  -2.5 to  0.0

        "#f3e8bd",  #   0.0 to  2.5
        "#f9e2ad",  #   2.5 to  5.0
        "#f8d391",  #   5.0 to  7.5
        "#f5c079",  #   7.5 to 10.0

        "#f2a05f",  #  10.0 to 12.5
        "#ea7954",  #  12.5 to 15.0
        "#dd5150",  #  15.0 to 20.0
    ]

    cmap = mcolors.ListedColormap(
        colors,
        name="icon_d2_temperature_discrete",
    )

    cmap.set_bad((0, 0, 0, 0))

    cmap.set_under(colors[0])

    cmap.set_over(colors[-1])

    norm = mcolors.BoundaryNorm(
        TEMP_BOUNDS_C,
        ncolors=cmap.N,
        clip=False,
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
    """Render discrete temperature classes on dark relief."""

    fig, ax, proj = style.dark_axes(extent)

    cmap, norm = temperature_colormap()

    values = np.ma.masked_invalid(field_c)

    x = np.asarray(x, dtype=float)

    y = np.asarray(y, dtype=float)

    if values.shape != (len(y), len(x)):
        raise ValueError(
            "Temperature field dimensions do not match lon/lat"
        )

    # --------------------------------------------------------
    # DISCRETE TEMPERATURE CLASSES
    # --------------------------------------------------------
    #
    # Use contourf rather than bilinear imshow.
    #
    # contourf draws discrete colour areas while preserving
    # reasonably smooth spatial class boundaries.
    #
    # antialiased=False prevents thin seams between classes.
    # --------------------------------------------------------

    ax.contourf(
        x,
        y,
        values,
        levels=TEMP_BOUNDS_C,
        cmap=cmap,
        norm=norm,
        extend="both",
        transform=proj,
        alpha=TEMP_ALPHA,
        antialiased=False,
        zorder=3,
    )

    # --------------------------------------------------------
    # GREY MOUNTAIN MASK
    # --------------------------------------------------------

    if bool(np.any(terrain_mask)):

        ax.contourf(
            x,
            y,
            terrain_mask.astype(float),
            levels=[0.5, 1.5],
            colors=[MASK_GRAY],
            transform=proj,
            zorder=6,
            antialiased=False,
        )

    # No extra 0 C contour.
    # The blue/yellow class boundary marks 0 C clearly.

    # --------------------------------------------------------
    # VALID TIME
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
    # DARK HEADER / FOOTER / CITY LABELS
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
# 7. DOWNLOAD DATA AND GENERATE MAPS
# ============================================================

def produce_temperature_maps(mod, run, areas, extents):
    """Generate temperature maps at 1500, 2000 and 3000 m.

    All heights use the same:
      - T850
      - T700
      - FI850
      - FI700

    No additional downloads for 2000 m.

    Expected output: 27 maps for one geographic area.
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
            "User-Agent": (
                "SneeuwhoogteNL-ICON-D2-temperature/1.0"
            ),
        })

        # ----------------------------------------------------
        # SURFACE ELEVATION
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # LOOP OVER FORECAST TIMES
        # ----------------------------------------------------

        for lead in steps:

            data = {}

            grid = None

            # ------------------------------------------------
            # DOWNLOAD FOUR PRESSURE-LEVEL FIELDS
            # ------------------------------------------------

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
                                "Pressure-level grid "
                                "differs from HSURF"
                            )

                    else:

                        mod.assert_same_grid(
                            grid,
                            result,
                            f"{param.upper()} {level} hPa",
                        )

                    data[(param, level)] = result

            # ------------------------------------------------
            # CONVERT TEMPERATURE AND GEOPOTENTIAL
            # ------------------------------------------------

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

            # ------------------------------------------------
            # GENERATE ALL THREE ELEVATIONS
            # ------------------------------------------------

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

                coverage = np.isfinite(field_c).sum()

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
