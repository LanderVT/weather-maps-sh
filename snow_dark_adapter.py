"""Display-only adapter for the Sneeuwhoogte dark/slate map layout.

The reference module contains the user's graphical design.  This adapter is
installed *after* loading each existing science script and replaces only its
plot functions.  No model downloads, numerical fields, window definitions,
densities, snowpack equations, or forecast data are altered.
"""

from __future__ import annotations

from datetime import timedelta
import json
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap, LinearSegmentedColormap

import map_style_reference as style


def configure(mod, *, theme: str | None = None) -> None:
    """Copy the model's existing palettes and output settings to the renderer."""
    choice = theme or os.environ.get("SNOW_MAP_THEME", "slate").strip().lower()
    if choice not in style.THEMES:
        raise ValueError("SNOW_MAP_THEME must be graphite, slate, night, or ice")
    style.apply_theme(choice)
    style.USE_ESRI_RELIEF = bool(mod.USE_ESRI_RELIEF)
    style.ESRI_ZOOM = mod.ESRI_ZOOM
    style.DPI = mod.DPI
    style.LOGO_PATH = mod.LOGO_PATH
    style.TILE_CACHE_DIR = mod.TILE_CACHE_DIR
    style.OUTPUT_DIR = mod.OUTPUT_DIR
    style.LOCAL_TZ = mod.LOCAL_TZ
    style._DARK_LOGO = None
    style.SNOW_LEVELS = np.asarray(mod.SNOW_LEVELS)
    style.SNOW_COLORS = list(mod.SNOW_COLORS)
    style.SNOW_MIN_CM = float(mod.SNOW_MIN_CM)
    style.CITIES = mod.CITIES
    style.PRIORITY_CITIES = mod.PRIORITY_CITIES
    style.MAX_CITY_LABELS = mod.MAX_CITY_LABELS
    style.MIN_CITY_DISTANCE_PX = mod.MIN_CITY_DISTANCE_PX
    print(f"KAARTSTIJL: dark/{choice}; {style.DPI} DPI; berekeningen ongewijzigd", flush=True)


def terrain_for_display(x, y, terrain) -> None:
    """Give the background renderer a terrain field already used by the model."""
    style.set_dark_terrain(x, y, terrain)


def _rgba_snow(data, x, y, terrain=None):
    """Apply the reference script's palette/shading to existing snow values."""
    data = np.asarray(data, dtype=float)
    z = np.ma.masked_where(~np.isfinite(data) | (data < style.SNOW_MIN_CM), data)
    return style.shade_snow_rgba(x, y, z, x, y, terrain)


def _plot_rgba(ax, proj, rgba, x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    if len(x) < 2 or len(y) < 2:
        raise ValueError("Snow-map grid must contain at least two x/y coordinates")
    hx = (x[1] - x[0]) / 2
    hy = (y[1] - y[0]) / 2
    ax.imshow(
        rgba, origin="lower",
        extent=(x[0] - hx, x[-1] + hx, y[0] - hy, y[-1] + hy),
        transform=proj, interpolation="bilinear",
        interpolation_stage="rgba", zorder=3,
    )


def _store_webcam_georef(fig, ax, outfile, extent):
    """Georeference the actual uncropped, full-bleed dark map for clickable webcams."""
    # Unlike the old maps the dark PNG has no bbox_inches='tight' cropping.
    left, bottom, width, height = ax.get_position().bounds
    west, east, south, north = map(float, extent)
    meta = {
        "plot_box_pct": [round(100*left, 6), round(100*(1-bottom-height), 6),
                         round(100*width, 6), round(100*height, 6)],
        "extent_lonlat": [west, east, south, north],
    }
    Path(str(outfile) + ".geo.json").write_text(json.dumps(meta), encoding="utf-8")
    print(f"WEB-CAM GEO: dark-map position saved for {outfile.name}", flush=True)


def render_snowdays(lib, data, x, y, extent, area, run, start, end,
                    rain=None, total=False, *, webcam=False, terrain=None):
    """Replacement for *plot_snowfall* only, matching original filenames."""
    fig, ax, proj = style.dark_axes(extent)
    rgba = _rgba_snow(data, x, y, terrain)
    _plot_rgba(ax, proj, rgba, x, y)
    start_dt = (run + timedelta(hours=start)).astimezone(lib.LOCAL_TZ)
    end_dt = (run + timedelta(hours=end)).astimezone(lib.LOCAL_TZ)

    if rain is None:
        product = "Nieuwe sneeuw op de grond"
        subtitle = "Alleen nieuw gevallen sneeuw, na inklinken en smelten (cm)"
        outfile = lib.OUTPUT_DIR / (
            f"{lib.FILE_PREFIX}_sneeuw_op_grond_{area}_{run:%Y%m%d_%H}z_"
            f"f{start:03d}-{end:03d}.png"
        )
        date_text = style.fmt_pill(end_dt)
    else:
        product = "Cumulatieve sneeuwval" if total else "Sneeuwval"
        subtitle = "Verwachte sneeuwval (cm), niet de bestaande sneeuwhoogte"
        outfile = lib.OUTPUT_DIR / (
            f"{lib.FILE_PREFIX}_sneeuwval{'_regen' if lib.SHOW_RAIN else ''}"
            f"{'_cumulatief' if total else ''}_{area}_"
            f"{run:%Y%m%d_%H}z_f{start:03d}-{end:03d}.png"
        )
        date_text = f"{style.fmt_pill(start_dt)}  →  {style.fmt_pill(end_dt)}"

    cmap, norm = style.snow_dark_cmap()
    fig_title = f"{product} – {style.region_title(area)}"
    style.finish_map_dark(
        fig, ax, fig_title, subtitle, date_text,
        f"{product} (cm)", cmap, norm,
        list(style.SNOW_LEVELS),
        [str(t) for t in style.SNOW_LEVELS[:-1]] + [f"{style.SNOW_LEVELS[-1]}+"],
        outfile, extent, area, run,
    )
    if webcam:
        _store_webcam_georef(fig, ax, outfile, extent)


def install_snowdays(mod, *, webcam=False):
    """Only replace map generation; all other module functions are untouched."""
    configure(mod)
    terrain_state = {"x": None, "y": None, "dem": None}

    def set_terrain(x, y, dem):
        terrain_state.update(x=x, y=y, dem=dem)
        terrain_for_display(x, y, dem)

    def plot(data, x, y, extent, area, run, start, end, rain=None, total=False):
        return render_snowdays(
            mod, data, x, y, extent, area, run, start, end, rain, total,
            webcam=webcam,
            terrain=terrain_state["dem"] if terrain_state["dem"] is not None
                    and np.shape(terrain_state["dem"]) == np.shape(data) else None,
        )

    mod.plot_snowfall = plot
    return set_terrain


def install_icon_d2(mod):
    """Change only plot_altitude/plot_snowfall of the existing ICON-D2 code."""
    configure(mod)
    terrain_state = {"terrain": None, "x": None, "y": None}

    def altitude(field, x, y, extent, area, run, step, kind, hsurf=None):
        if hsurf is not None:
            terrain_state.update(terrain=hsurf, x=x, y=y)
            terrain_for_display(x, y, hsurf)
        fig, ax, proj = style.dark_axes(extent)
        cmap = LinearSegmentedColormap.from_list(
            kind, mod.ALT_COLORS, N=len(mod.ALT_LEVELS)-1,
        )
        cmap.set_bad((0, 0, 0, 0))
        cmap.set_over(mod.ALT_COLORS[-1])
        norm = BoundaryNorm(mod.ALT_LEVELS, cmap.N)
        field = np.asarray(field, dtype=float)
        mask = np.zeros(field.shape, dtype=bool)
        if hsurf is not None:
            if np.shape(hsurf) != field.shape:
                raise ValueError("ICON-D2 terrain grid differs from snow/freezing limit")
            if kind == "hzerocl":
                # Keep the earlier gray correction: DWD HZEROCL=HSURF when
                # the entire atmospheric column is below freezing.
                mask = (np.isfinite(field) & np.isfinite(hsurf)
                        & (field <= hsurf + 10.0))
            else:
                # Keep the earlier gray area above the ICON snow limit.
                mask = np.isfinite(field) & np.isfinite(hsurf) & (hsurf >= field)
        z = np.ma.masked_where(mask | ~np.isfinite(field), field)
        ax.contourf(
            x, y, z, levels=mod.ALT_LEVELS, cmap=cmap, norm=norm,
            transform=proj, extend="max", zorder=3,
            alpha=style.ALT_ALPHA_DARK, antialiased=True,
        )
        if mask.any():
            ax.contourf(
                x, y, mask.astype(float), levels=[0.5, 1.5],
                colors=["#c8cdd3"], transform=proj, zorder=6,
                antialiased=True,
            )
            print(f"{kind}: {int(mask.sum())} grey terrain cells", flush=True)
        noun = "Vorstgrens" if kind == "hzerocl" else "Sneeuwgrens"
        stem = "icon_d2_vorstgrens" if kind == "hzerocl" else "icon_d2_sneeuwgrens"
        outfile = mod.OUTPUT_DIR / (
            f"{stem}_{area}_{run:%Y%m%d_%H}z_f{step:03d}_"
            f"{(run + timedelta(hours=step)):%Y%m%d_%H}z.png"
        )
        valid_time = (run + timedelta(hours=step)).astimezone(mod.LOCAL_TZ)
        ticks = list(range(0, int(mod.ALT_LEVELS[-1])+1, 500))
        style.finish_map_dark(
            fig, ax, f"{noun} – {style.region_title(area)}", "",
            style.fmt_long(valid_time), f"{noun} (m)",
            cmap, norm, ticks,
            [str(t) for t in ticks[:-1]] + [f"{ticks[-1]}+"],
            outfile, extent, area, run,
        )

    def snowfall(data, x, y, extent, area, run, name, start, end):
        # Preserve the original map interpolation and DWD precipitation.
        xf, yf, zf = mod.interpolate_snow(x, y, data, extent, area)
        fig, ax, proj = style.dark_axes(extent)
        terr = terrain_state["terrain"]
        if terr is not None and np.shape(terr) != np.shape(data):
            terr = None
        rgba = style.shade_snow_rgba(xf, yf, zf, x, y, terr)
        _plot_rgba(ax, proj, rgba, xf, yf)
        outfile = mod.OUTPUT_DIR / (
            f"icon_d2_verse_sneeuw_{area}_{run:%Y%m%d_%H}z_{name}.png"
        )
        start_dt = (run + timedelta(hours=start)).astimezone(mod.LOCAL_TZ)
        end_dt = (run + timedelta(hours=end)).astimezone(mod.LOCAL_TZ)
        product = "Cumulatieve sneeuwval" if name == "cumulatief" else "Sneeuwval"
        cmap, norm = style.snow_dark_cmap()
        style.finish_map_dark(
            fig, ax, f"{product} – {style.region_title(area)}",
            "Verwachte sneeuwval (cm), niet de bestaande sneeuwhoogte",
            f"{style.fmt_pill(start_dt)}  →  {style.fmt_pill(end_dt)}",
            f"{product} (cm)", cmap, norm, list(style.SNOW_LEVELS),
            [str(t) for t in style.SNOW_LEVELS[:-1]] + [f"{style.SNOW_LEVELS[-1]}+"],
            outfile, extent, area, run,
        )

    mod.plot_altitude = altitude
    mod.plot_snowfall = snowfall
