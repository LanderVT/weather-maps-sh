
"""ICON-D2 temperature at 1500, 2000 and 3000 m a.s.l. for Sneeuwhoogte.nl.

Standalone ADD-ON to run_icon_d2.py, not a replacement for SnowAll_ICON-D2.py.
Uses T and FI at 850/700 hPa (4 small GRIB files per forecast step),
interpolated/extrapolated to the requested geometric heights using pressure-
level geopotential heights. For each grid cell, T(h) is approximated by the
straight line between its two pressure levels; the result is NOT a full
model-level interpolation. Areas below the model's surface are greyed out.

New maps follow the existing dark/slate render style and are generated for
+0, +6, ..., +48 h, using the same run, map extent, relief and timestamps as
the existing ICON-D2 maps. Do not execute this module directly; it is called
by run_icon_d2.py after the existing products have been written.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import matplotlib.colors as mcolors
import numpy as np
import requests

import map_style_reference as style


# ============================================================
# SETTINGS
# ============================================================

GRAVITY = 9.80665

HEIGHTS_M = (1500, 2000, 3000)
PRESSURE_LEVELS = (850, 700)

TEMP_TICKS_C = list(range(-30, 21, 5))

MASK_GRAY = "#c8cdd3"

# Transparency already approved in previous version.
TEMP_ALPHA = 0.91


# ============================================================
# DWD PRESSURE-LEVEL DATA
# ============================================================

def pressure_level_url(
    mod, run: datetime, lead: int, level: int, parameter: str
):
    """DWD pressure-level file naming."""

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
    """Convert geopotential (m2/s2) into geopotential height (m)."""

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
        raise ValueError(
            "All geopotential values are missing"
        )

    typical = float(np.median(finite))

    if 6000 < typical < 90000:
        return data / GRAVITY

    if 500 < typical < 5000:
        return data

    raise ValueError(
        f"Cannot interpret FI units {unit!r}; median={typical:g}"
    )


# ============================================================
# TEMPERATURE INTERPOLATION
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

    Small extrapolations from the closest pressure level are allowed.
    Temperatures inside mountains are masked.
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
# TEMPERATURE COLORS: ORIGINAL STYLE, SMOOTHED
# ============================================================

def temperature_colormap():
    """Original blue-to-yellow palette, smoothly continuous.

    Improvements:
    - more distinguishable shades of blue;
    - smooth transition around zero;
    - no hard blue/yellow discontinuity;
    - no extra narrow color band;
    - original yellow/orange/red tones retained.
    """

    vmin = -30.0
    vmax = 20.0

    def pos(t_c):
        return (t_c - vmin) / (vmax - vmin)

    # Anchor temperatures in Celsius.
    # Colors interpolate smoothly between these points.

    points = [
        (-30.0, "#311d62"),  # Violet
        (-22.5, "#2a4caa"),  # Deep blue
        (-15.0, "#237fc3"),  # Blue
        (-10.0, "#4396d2"),  # Medium blue
        (-7.5,  "#68b0df"),  # Cool blue
        (-5.0,  "#8cc6e9"),  # Lighter blue
        (-2.5,  "#b1d9f0"),  # Pale blue

        # Smooth transition near freezing
        ( 0.0,  "#edf1ed"),  # Soft neutral, not a hard white band
        ( 2.5,  "#f9e6b1"),  # Soft yellow

        # Original warm tones
        ( 6.5,  "#ffe3a0"),  # Yellow
        (13.0,  "#f7a04f"),  # Orange
        (20.0,  "#d73b43"),  # Red
    ]

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "icon_d2_temperature_smooth_original",
        [(pos(temp), color) for temp, color in points],
        N=512,
    )

    cmap.set_bad((0, 0, 0, 0))
    cmap.set_under("#311d62")
    cmap.set_over("#d73b43")

    # IMPORTANT:
    # Continuous normalization instead of BoundaryNorm.
    # This removes the old 2-degree color bands.
    #
    # A linear Normalize also preserves the original
    # temperature-to-color positions on the -30..20 C scale.

    norm = mcolors.Normalize(
        vmin=vmin,
        vmax=vmax,
        clip=False,
    )

    return cmap, norm


# ============================================================
# MAP PLOTTING
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
    """Render a temperature map with a smooth color gradient."""

    fig, ax, proj = style.dark_axes(extent)

    cmap, norm = temperature_colormap()

    values = np.ma.masked_invalid(field_c)

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # The source is the regular ICON-D2 longitude/latitude grid.
    if (
        values.shape != (len(y), len(x))
        or len(x) < 2
        or len(y) < 2
        or np.any(np.diff(x) <= 0)
        or np.any(np.diff(y) <= 0)
    ):
        raise ValueError(
            "Expected a regular, increasing lon/lat temperature grid"
        )

    dx = float(np.median(np.diff(x)))
    dy = float(np.median(np.diff(y)))

    image_extent = [
        x[0] - dx / 2,
        x[-1] + dx / 2,
        y[0] - dy / 2,
        y[-1] + dy / 2,
    ]

    # Smooth display interpolation instead of discrete contourf bands.
    # This changes only the visual rendering, not the temperatures.

    ax.imshow(
        values,
        origin="lower",
        extent=image_extent,
        transform=proj,
        cmap=cmap,
        norm=norm,
        interpolation="bilinear",
        alpha=TEMP_ALPHA,
        zorder=3,
    )

    # Light-grey mask above the selected altitude.

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

    # No additional 0 C contour:
    # the smooth blue-to-yellow gradient marks the transition.

    valid_utc = run + timedelta(hours=lead)
    valid_local = valid_utc.astimezone(mod.LOCAL_TZ)

    outfile = mod.OUTPUT_DIR / (
        f"icon_d2_temperatuur_{target_m}m_{area}_"
        f"{run:%Y%m%d_%H}z_"
        f"f{lead:03d}_"
        f"{valid_utc:%Y%m%d_%H}z.png"
    )

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
# DOWNLOAD DATA + GENERATE ALL TEMPERATURE MAPS
# ============================================================

def produce_temperature_maps(mod, run, areas, extents):
    """Generate temperature maps for 1500, 2000 and 3000 m.

    All three elevations use the same four pressure-level
    fields per forecast step: T850, T700, FI850, FI700.
    """

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

        # Reuse surface-height field from source cache.
        hfield = mod.load_hsurf(session, run)

        xsurf, ysurf, terrain_raw, terrain_unit = hfield

        terrain = np.asarray(terrain_raw, dtype=float)

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

        for lead in steps:

            data = {}
            grid = None

            # Four pressure-level fields, shared by all heights.
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

            # Generate all three altitudes from the same
            # temperature profile, with no extra downloads.

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
