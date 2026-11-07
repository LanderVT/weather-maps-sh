
"""ICON-D2 temperature at 1500 m and 3000 m a.s.l. for Sneeuwhoogte.nl.

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

DWD file form:
icon-d2_germany_regular-lat-lon_pressure-level_YYYYMMDDHH_000_850_t.grib2.bz2
and the equivalent 700, fi combinations.
"""

from __future__ import annotations

from datetime import datetime, timedelta
import os
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import numpy as np
import requests

import map_style_reference as style


GRAVITY = 9.80665
HEIGHTS_M = (1500, 3000)
PRESSURE_LEVELS = (850, 700)

TEMP_LEVELS_C = np.arange(-30, 22, 2, dtype=float)
TEMP_TICKS_C = list(range(-30, 21, 5))

MASK_GRAY = "#c8cdd3"


def pressure_level_url(
    mod, run: datetime, lead: int, level: int, parameter: str
):
    """DWD open-data naming: lowercase 't' and 'fi'."""

    if level not in PRESSURE_LEVELS or parameter not in ("t", "fi"):
        raise ValueError(
            f"Unsupported pressure-level field {parameter}/{level}"
        )

    filename = (
        f"icon-d2_germany_regular-lat-lon_pressure-level_"
        f"{run:%Y%m%d%H}_{lead:03d}_{level}_{parameter}.grib2.bz2"
    )

    url = f"{mod.DATA_URL}/{run:%H}/{parameter}/{filename}"

    return url, filename


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
            f"Unexpected DWD temperature unit: {unit!r}"
        )

    return data


def geopotential_to_metres(raw, unit: str):
    """Convert geopotential (m2/s2) into geopotential height (m)."""

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
        f"Cannot interpret FI units {unit!r}; median={typical:g}"
    )


def calculate_fixed_height_temperature(
    t850_c,
    t700_c,
    z850_m,
    z700_m,
    target_m: int,
    hsurf_m,
):
    """Interpolate temperature vertically between 850 and 700 hPa.

    Small extrapolations from the closest pressure level are allowed.

    Temperatures inside the mountains are masked.
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


def temperature_colormap():
    """Dark temperature colors for the slate Alpine relief.

    Avoid near-white and pale beige around 0 C.
    Keep the same temperature scale for both altitudes.
    """

    anchors = [
        (0.00, "#251451"),  # -30 C: violet
        (0.20, "#213b91"),  # -20 C: deep blue
        (0.30, "#245bac"),  # -15 C: blue
        (0.40, "#2a7daa"),  # -10 C: dark cyan-blue
        (0.50, "#478caa"),  #  -5 C: muted blue
        (0.60, "#637285"),  #   0 C: slate blue-gray
        (0.70, "#ac805f"),  #  +5 C: muted copper
        (0.80, "#d16a4c"),  # +10 C: terracotta
        (0.90, "#ce484c"),  # +15 C: red
        (1.00, "#a32d50"),  # +20 C: red-magenta
    ]

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "icon_d2_temperature_dark_slate",
        anchors,
        N=256,
    )

    cmap.set_bad((0, 0, 0, 0))
    cmap.set_under(anchors[0][1])
    cmap.set_over(anchors[-1][1])

    norm = mcolors.BoundaryNorm(
        TEMP_LEVELS_C,
        ncolors=cmap.N,
    )

    return cmap, norm


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
    """Render a temperature map using the shared dark layout."""

    import cartopy.crs as ccrs

    fig, ax, proj = style.dark_axes(extent)

    cmap, norm = temperature_colormap()

    values = np.ma.masked_invalid(field_c)

    ax.contourf(
        x,
        y,
        values,
        levels=TEMP_LEVELS_C,
        cmap=cmap,
        norm=norm,
        extend="both",
        transform=proj,
        alpha=0.76,
        zorder=3,
        antialiased=True,
    )

    # Gray mountain areas above the selected elevation.
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

    # Thin 0 C contour.
    finite = np.asarray(field_c, dtype=float)

    if (
        np.isfinite(finite).any()
        and np.nanmin(finite) < 0 < np.nanmax(finite)
    ):
        zero = ax.contour(
            x,
            y,
            values,
            levels=[0],
            colors=["#f7f9fd"],
            linewidths=1.0,
            transform=proj,
            zorder=9,
        )

        zero.set_path_effects([
            pe.Stroke(
                linewidth=1.7,
                foreground="#182738",
            ),
            pe.Normal(),
        ])

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


def produce_temperature_maps(mod, run, areas, extents):
    """Generate temperature maps for 1500 m and 3000 m.

    Called by run_icon_d2.py after the snow maps have been produced.
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
            "User-Agent": (
                "SneeuwhoogteNL-ICON-D2-temperature/1.0"
            ),
        })

        # Reuse the surface-height field.
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

        for lead in steps:

            data = {}
            grid = None

            # Four pressure-level fields per timestep.
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
