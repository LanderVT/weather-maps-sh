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

DWD file form, verified against the DWD pressure-level directory:
icon-d2_germany_regular-lat-lon_pressure-level_YYYYMMDDHH_000_850_t.grib2.bz2
and the equivalent 700, fi combinations. DWD distributes T in kelvin and FI
as geopotential (m^2 s^-2), which is converted using standard gravity.
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

GRAVITY = 9.80665  # standard gravity (m s^-2)
HEIGHTS_M = (1500, 2000, 3000)
PRESSURE_LEVELS = (850, 700)
TEMP_LEVELS_C = np.arange(-30, 22, 2, dtype=float)
TEMP_TICKS_C = list(range(-30, 21, 5))
MASK_GRAY = "#c8cdd3"   # same light gray as snow/freezing-line masks


def pressure_level_url(mod, run: datetime, lead: int, level: int, parameter: str):
    """DWD legacy open-data naming: lowercase 't' and 'fi'."""
    if level not in PRESSURE_LEVELS or parameter not in ("t", "fi"):
        raise ValueError(f"Unsupported pressure-level field {parameter}/{level}")
    filename = (
        f"icon-d2_germany_regular-lat-lon_pressure-level_"
        f"{run:%Y%m%d%H}_{lead:03d}_{level}_{parameter}.grib2.bz2"
    )
    url = f"{mod.DATA_URL}/{run:%H}/{parameter}/{filename}"
    return url, filename


def kelvin_to_celsius(raw, unit: str):
    """Temperature in Celsius, with a sanity guard for missing GRIB unit attrs."""
    data = np.asarray(raw, dtype=float)
    unit = unit.strip().lower()
    if unit in ("k", "kelvin", "degrees kelvin", "degree kelvin"):
        data = data - 273.15
    elif unit in ("c", "degc", "degree celsius", "degrees celsius", "celsius", "°c"):
        pass
    elif not unit:
        if np.nanmedian(data) > 100:
            data = data - 273.15
    else:
        raise ValueError(f"Unexpected DWD temperature unit: {unit!r}")
    return data


def geopotential_to_metres(raw, unit: str):
    """DWD FI is geopotential in m2/s2, not a geopotential height in m."""
    data = np.asarray(raw, dtype=float)
    unit = unit.strip().lower().replace(" ", "")
    unit = unit.replace("²", "2").replace("−", "-")
    if unit in ("m", "gpm", "geopotentialmetre", "geopotentialmeter"):
        return data
    if unit in ("m**2s**-2", "m2s-2", "m2/s2", "m^2s^-2", "m**2s-2"):
        return data / GRAVITY
    # Cfgrib/ecCodes may report an empty/alternate unit string; recognise
    # the physical magnitude for these two levels (FI ~ 15,000/30,000 m2s-2).
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        raise ValueError("All geopotential values are missing")
    typical = float(np.median(finite))
    if 6000 < typical < 90000:
        return data / GRAVITY
    if 500 < typical < 5000:
        return data
    raise ValueError(f"Cannot interpret FI units {unit!r}; median={typical:g}")


def calculate_fixed_height_temperature(t850_c, t700_c, z850_m, z700_m,
                                       target_m: int, hsurf_m):
    """Linear interpolation of T(z) between 850 and 700 hPa.

    Small extrapolations from the closest pressure level are allowed, since
    Z850/Z700 vary in space/time and fixed heights are not exact pressure levels.
    Values inside mountains are NaN and drawn using a grey terrain mask.
    """
    t850_c = np.asarray(t850_c, dtype=float)
    t700_c = np.asarray(t700_c, dtype=float)
    z850_m = np.asarray(z850_m, dtype=float)
    z700_m = np.asarray(z700_m, dtype=float)
    hsurf_m = np.asarray(hsurf_m, dtype=float)
    if not all(a.shape == t850_c.shape for a in (t700_c, z850_m, z700_m, hsurf_m)):
        raise ValueError("DWD pressure/surface fields do not share a grid")
    dz = z700_m - z850_m
    valid = (np.isfinite(t850_c) & np.isfinite(t700_c)
             & np.isfinite(z850_m) & np.isfinite(z700_m)
             & (dz > 500) & (dz < 2500))
    frac = np.divide(target_m - z850_m, dz,
                     out=np.full_like(dz, np.nan), where=valid)
    result = t850_c + frac * (t700_c - t850_c)
    # Avoid long extrapolations from two levels and below-terrain estimates.
    nearest_distance = np.minimum(np.abs(target_m-z850_m), np.abs(target_m-z700_m))
    within_pressure_profile = ((target_m >= z850_m) & (target_m <= z700_m))
    valid &= (within_pressure_profile | (nearest_distance <= 600.0))
    terrain_mask = np.isfinite(hsurf_m) & (hsurf_m >= target_m)
    result[~valid | terrain_mask] = np.nan
    return result, terrain_mask


def temperature_colormap():
    """Original blue-to-yellow temperature scale, with more contrast near 0 C."""

    anchors = [
        (0.00,  "#311d62"),  # -30 C: violet
        (0.15,  "#2a4caa"),  # deep blue
        (0.30,  "#237fc3"),  # blue
        (0.45,  "#74c5ed"),  # light blue
        (0.53,  "#9dd8f4"),  # icy blue
        (0.58,  "#76bde8"),  # around -2 C: clearer blue
        (0.625, "#f0f1eb"),  # near 0 C: neutral transition
        (0.66,  "#ffe28b"),  # around +2 C: warm yellow
        (0.73,  "#ffd17b"),  # yellow-gold
        (0.86,  "#f7a04f"),  # orange
        (1.00,  "#d73b43"),  # +20 C: red
    ]

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "icon_d2_temperature", anchors, N=256
    )
    cmap.set_bad((0, 0, 0, 0))
    cmap.set_under("#311d62")
    cmap.set_over("#d73b43")
    norm = mcolors.BoundaryNorm(TEMP_LEVELS_C, ncolors=cmap.N)
    return cmap, norm


def render_temperature_map(mod, x, y, field_c, terrain_mask, extent, area,
                           run, lead, target_m):
    """Purely graphical: new map with precisely the shared dark layout."""
    import cartopy.crs as ccrs

    fig, ax, proj = style.dark_axes(extent)
    cmap, norm = temperature_colormap()
    values = np.ma.masked_invalid(field_c)
    ax.contourf(x, y, values, levels=TEMP_LEVELS_C,
                cmap=cmap, norm=norm, extend="both", transform=proj,
                # Slightly stronger temperature colours with relief still visible.
                alpha=0.94, zorder=3, antialiased=True)

    if bool(np.any(terrain_mask)):
        ax.contourf(x, y, terrain_mask.astype(float), levels=[0.5, 1.5],
                    colors=[MASK_GRAY], transform=proj, zorder=6,
                    antialiased=False)

    # Thin, clearly visible 0-degree line wherever the requested height
    # lies above the terrain; labels would clutter the map, so no labels.
    finite = np.asarray(field_c, dtype=float)
    if np.isfinite(finite).any() and np.nanmin(finite) < 0 < np.nanmax(finite):
        zero = ax.contour(x, y, values, levels=[0], colors=["#f7f9fd"],
                          linewidths=1.0, transform=proj, zorder=9)
        zero.set_path_effects([
            pe.Stroke(linewidth=1.7, foreground="#182738"), pe.Normal()
        ])

    valid_utc = run + timedelta(hours=lead)
    valid_local = valid_utc.astimezone(mod.LOCAL_TZ)
    outfile = mod.OUTPUT_DIR / (
        f"icon_d2_temperatuur_{target_m}m_{area}_"
        f"{run:%Y%m%d_%H}z_f{lead:03d}_{valid_utc:%Y%m%d_%H}z.png"
    )
    style.finish_map_dark(fig, ax,
                          f"Temperatuur {target_m} m – {style.region_title(area)}",
                          "", style.fmt_long(valid_local),
                          "Temperatuur (°C)", cmap, norm,
                          TEMP_TICKS_C, [str(t) for t in TEMP_TICKS_C],
                          outfile, extent, area, run)
    return outfile


def produce_temperature_maps(mod, run, areas, extents):
    """Create all three height series using the same ICON-D2 run and extents.

    This is an optional add-on; the caller catches errors and rolls back any
    partially generated temperature maps without touching snow products.
    """
    steps = tuple(mod.FORECAST_HOURS)
    if steps != tuple(range(0, 49, 6)):
        raise ValueError("ICON-D2 temperature expects steps 0..48 every 6 hours")
    if any(a not in extents for a in areas):
        raise ValueError(f"No recorded ICON-D2 map extent for: {set(areas)-set(extents)}")

    output = []
    with requests.Session() as session:
        session.headers.update({"User-Agent": "SneeuwhoogteNL-ICON-D2-temperature/1.0"})
        hfield = mod.load_hsurf(session, run)  # reused from source cache, no new HSURF download
        xsurf, ysurf, terrain_raw, terrain_unit = hfield
        terrain = np.asarray(terrain_raw, dtype=float)
        terrain = np.where((terrain >= -500) & (terrain <= 9000), terrain, np.nan)
        style.set_dark_terrain(xsurf, ysurf, terrain)

        for lead in steps:
            data = {}
            grid = None
            for level in PRESSURE_LEVELS:
                for param in ("t", "fi"):
                    url, filename = pressure_level_url(mod, run, lead, level, param)
                    archive = mod.download_archive(session, url, filename)
                    names = ["t", "temperature"] if param == "t" else ["fi", "z", "gh", "geopotential"]
                    result = mod.read_grib(archive, names, lead)
                    if grid is None:
                        grid = result[:2]
                        if not (np.array_equal(grid[0], xsurf) and np.array_equal(grid[1], ysurf)):
                            raise ValueError("Pressure-level grid differs from HSURF")
                    else:
                        mod.assert_same_grid(grid, result, f"{param.upper()} {level} hPa")
                    data[(param, level)] = result

            x, y = grid
            t850 = kelvin_to_celsius(data[("t", 850)][2], data[("t", 850)][3])
            t700 = kelvin_to_celsius(data[("t", 700)][2], data[("t", 700)][3])
            z850 = geopotential_to_metres(data[("fi", 850)][2], data[("fi", 850)][3])
            z700 = geopotential_to_metres(data[("fi", 700)][2], data[("fi", 700)][3])

            for target_m in HEIGHTS_M:
                field_c, mask = calculate_fixed_height_temperature(
                    t850, t700, z850, z700, target_m, terrain
                )
                coverage = np.isfinite(field_c).sum()
                if coverage < 500:
                    raise RuntimeError(f"T {target_m} m +{lead}: insufficient valid cells ({coverage})")
                print(f"ICON-D2 T {target_m} m +{lead:03d}: {coverage} valid cells", flush=True)
                for area in areas:
                    output.append(render_temperature_map(
                        mod, x, y, field_c, mask, extents[area], area,
                        run, lead, target_m
                    ))
    return output
