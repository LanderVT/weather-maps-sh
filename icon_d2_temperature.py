
"""ICON-D2 temperature at 1500, 2000 and 3000 m a.s.l.

Standalone add-on to run_icon_d2.py.
Uses T and FI at 850/700 hPa (4 GRIB files per forecast step).
Temperatures at all three heights are interpolated from the same
pressure-level fields.

Forecast steps: +0, +6, ..., +48 hours.
Map style: original light blue / ivory / yellow temperature scale,
with slightly stronger blue/yellow contrast around zero.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import numpy as np
import requests

import map_style_reference as style


# ============================================================
# SETTINGS
# ============================================================

GRAVITY = 9.80665

HEIGHTS_M = (1500, 2000, 3000)
PRESSURE_LEVELS = (850, 700)

TEMP_LEVELS_C = np.arange(-30, 22, 2, dtype=float)
TEMP_TICKS_C = list(range(-30, 21, 5))

MASK_GRAY = "#c8cdd3"

# ORIGINAL FIRST VERSION: alpha=0.88
# New version: a tiny bit less transparent
TEMP_ALPHA = 0.91


# ============================================================
# DWD PRESSURE-LEVEL DOWNLOADS
# ============================================================

def pressure_level_url(
    mod,
    run: datetime,
    lead: int,
    level: int,
    parameter: str,
):
    """Construct DWD ICON-D2 pressure-level GRIB URL."""

    if level not in PRESSURE_LEVELS:
        raise ValueError(f"Unsupported pressure level: {level}")

    if parameter not in ("t", "fi"):
        raise ValueError(f"Unsupported parameter: {parameter}")

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
# UNIT CONVERSIONS
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
    """Convert DWD FI geopotential to metres."""

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
# VERTICAL INTERPOLATION
# ============================================================

def calculate_fixed_height_temperature(
    t850_c,
    t700_c,
    z850_m,
    z700_m,
    target_m: int,
    hsurf_m,
):
    """Interpolate T between 850 and 700 hPa.

    Uses the same vertical profile for 1500, 2000 and 3000 m.
    Small extrapolations are allowed.

    Values below the model terrain are masked.
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
# ORIGINAL TEMPERATURE COLORMAP
# ============================================================

def temperature_colormap():
    """FIRST version: blue / icy white / yellow / orange / red.

    Original colours retained outside the immediate freezing range.

    Modified:
      -2.5 to 0 C: slightly stronger blue
       0 to +2.5 C: slightly stronger yellow

    No brown, copper or dark slate temperature colours.
    """

    # Position 0.60 corresponds to 0 C on the -30..20 scale.

    anchors = [
        # ORIGINAL cold colours
        (0.00, "#311d62"),
        (0.15, "#2a4caa"),
        (0.30, "#237fc3"),
        (0.45, "#74c5ed"),

        # ONLY CHANGE: slightly stronger blue near zero
        (0.55, "#93ccf2"),   # -2.5 C
        (0.59, "#b2dcf8"),   # -0.5 C

        # ORIGINAL near-white 0 C transition
        (0.60, "#f5f8fa"),   #  0.0 C

        # ONLY CHANGE: slightly stronger yellow near zero
        (0.61, "#fff0af"),   # +0.5 C
        (0.65, "#ffe28b"),   # +2.5 C

        # ORIGINAL warm colours
        (0.73, "#ffe3a0"),
        (0.86, "#f7a04f"),
        (1.00, "#d73b43"),
    ]

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "icon_d2_temperature_original",
        anchors,
        N=256,
    )

    cmap.set_bad((0, 0, 0, 0))
    cmap.set_under("#311d62")
    cmap.set_over("#d73b43")

    norm = mcolors.BoundaryNorm(
        TEMP_LEVELS_C,
        ncolors=cmap.N,
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
    """Render temperature map with original temperature colours."""

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
        alpha=TEMP_ALPHA,
        zorder=3,
        antialiased=True,
    )

    # Light-grey mask where the selected elevation
    # is below the actual model terrain.

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

    # 0 C contour
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

    # Forecast validity time
    valid_utc = run + timedelta(hours=lead)
    valid_local = valid_utc.astimezone(mod.LOCAL_TZ)

    # Output filename (same pattern as before)
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
# DOWNLOAD AND GENERATE ALL TEMPERATURE MAPS
# ============================================================

def produce_temperature_maps(mod, run, areas, extents):
    """Generate temperature maps at 1500, 2000 and 3000 m.

    All heights reuse the same T850, T700, FI850 and FI700 fields.
    No extra downloads are needed for the additional altitude.
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

        # Reuse the model surface-height field
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

            # Four GRIB fields per timestep
            # Shared by all three elevations

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

            # Generate 1500, 2000 and 3000 m from
            # the same pressure-level data

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
