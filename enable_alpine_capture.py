"""Prepare an ephemeral GitHub Actions checkout for Alpine snow tables.

This file edits scripts ONLY in the checked-out Actions runner, not in GitHub.
A) Exclude ICON-CH2 as a Hybrid input while keeping its shared plotting/
   snowpack library (SnowHybrid.main requires MODEL_FILES['ICON-CH2']).
B) Remove standalone ICON-CH2 from run_forecast's jobs and website tabs.
C) Add numerical snowfall/DEM capture hooks to the existing science modules.
D) Force a direct extraction from the final snow arrays of Hybrid and ICON-D2,
   matched with their own terrain, before plotting (reliable even if adapters
   replace plotting functions).
"""
from __future__ import annotations

import ast
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
MARKER = '# ALPINE_SNOW_TABLES_CAPTURE_V3'
BLOCK = [
    '\n' + MARKER + '\n',
    'try:\n',
    '    from alpine_snow_tables import install_module_capture as _as_install_module\n',
    '    _as_install_module(globals())\n',
    'except Exception as _as_exc:\n',
    '    print(f"::warning::Alpen tabel hook niet geladen: {_as_exc}", flush=True)\n',
    '\n',
]


def _compile_write(path: Path, source: str) -> None:
    compile(source, str(path), 'exec')
    path.write_text(source, encoding='utf-8')


def disable_hybrid_ch2_input() -> None:
    """Keep the CH2 library, but do not download/run CH2 as a Hybrid source."""
    path = ROOT / 'SnowHybrid.py'
    if not path.exists():
        raise SystemExit('Missing SnowHybrid.py')
    source = path.read_text(encoding='utf-8')
    needle = 'for name, fn in MODEL_FILES.items()'
    replacement = "for name, fn in MODEL_FILES.items() if name != 'ICON-CH2'"
    if replacement in source:
        print('Hybrid: ICON-CH2 input already excluded')
        return
    if source.count(needle) != 1:
        raise SystemExit('ERROR: Hybrid model input loop changed: cannot disable ICON-CH2 safely')
    if 'MODEL_FILES["ICON-CH2"]' not in source and "MODEL_FILES['ICON-CH2']" not in source:
        raise SystemExit('ERROR: Hybrid plotting library changed: cannot verify safe ICON-CH2 exclusion')
    source = source.replace(needle, replacement, 1)
    # The shared SnowDays_ICON-CH2 library is re-labelled 'hybrid' here.
    # Capture that library's actual plot_snowfall data and DEM directly,
    # rather than trying to reverse-engineer the finished PNG.
    lib_needle = 'lib.FILE_PREFIX = FILE_PREFIX'
    if source.count(lib_needle) != 1:
        raise SystemExit('ERROR: Hybrid library setup changed; capture hook not safely insertable')
    source = source.replace(lib_needle,
        lib_needle + '\n    from alpine_snow_tables import install_module_capture as _capture_alpine\n'
                     '    _capture_alpine(lib.__dict__)', 1)
    _compile_write(path, source)
    print('Hybrid: ICON-CH2 input excluded, shared plotting library retained and captured', flush=True)


def disable_standalone_ch2_and_tab() -> None:
    """Edit known run_forecast.py control flow; fail if it drifts."""
    path = ROOT / 'run_forecast.py'
    if not path.exists():
        raise SystemExit('Missing run_forecast.py')
    source = path.read_text(encoding='utf-8')
    if 'SNOW_TABLES_SKIP_CH2_V3' in source:
        print('run_forecast.py already prepared')
        return

    # The exact model tuple is used by collect_models(), make_site(), status
    # and the HTML tabs. Removing the key avoids all dead CH2 panels.
    old_tuple = 'MODELS = ("hybrid", "ecmwf", "icon_d2", "icon_ch2")'
    new_tuple = 'MODELS = ("hybrid", "ecmwf", "icon_d2")'
    if source.count(old_tuple) != 1:
        raise SystemExit('ERROR: run_forecast.MODELS changed; cannot disable CH2 safely')
    source = source.replace(old_tuple, new_tuple, 1)

    # Remove the CH2 subprocess launch from the model loop. No stub is needed.
    source, n = re.subn(r',\s*\("icon_ch2",\s*"run_icon_ch2\.py"\)', '', source)
    if n != 1:
        raise SystemExit('ERROR: run_forecast ICON-CH2 subprocess command not found')

    # Remove CH2 button from generated HTML and JS tab selector to prevent
    # JS errors (missing panel/tab), and leave the three working tabs intact.
    source, n = re.subn(r'^<button[^\n]*id="tab-icon_ch2"[^\n]*>\n', '', source, flags=re.M)
    if n != 1:
        raise SystemExit('ERROR: run_forecast ICON-CH2 website button not found')
    old_js = "['hybrid','ecmwf','icon_d2','icon_ch2']"
    if source.count(old_js) != 2:
        raise SystemExit('ERROR: run_forecast JS tab model list changed')
    source = source.replace(old_js, "['hybrid','ecmwf','icon_d2']")

    # Clean up source checks, optional output folder, header, and metadata.
    source = source.replace('    "run_icon_ch2.py",\n', '')
    source = source.replace(', ROOT / "run_icon_ch2.py"', '')
    source = source.replace(', MAPS / "icon_ch2"', '')
    source = re.sub(r'^[ \t]*"ICON_CH2_OUTPUT_DIR": str\(MAPS / "icon_ch2"\),\n', '', source, flags=re.M)
    source = source.replace(', "icon_ch2": "ICON-CH2 Model"', '')
    source = source.replace('<title>Sneeuwhoogte · Hybrid, ECMWF, ICON-D2 & ICON-CH2</title>',
                            '<title>Sneeuwhoogte · Hybrid, ECMWF & ICON-D2</title>')
    source = source.replace('Hybride voorspelling, ECMWF, ICON-D2 en ICON-CH2 als afzonderlijke modellen',
                            'Hybride voorspelling, ECMWF en ICON-D2 als afzonderlijke modellen')
    # Old CH2-specific panel-intro elif branch stays unreachable and harmless.
    source = '# SNOW_TABLES_SKIP_CH2_V3\n' + source
    _compile_write(path, source)
    print('Forecast runner: ICON-CH2 disabled (downloads, status, map tabs, JS)', flush=True)


def eligible(path: Path) -> bool:
    name = path.name.lower()
    if name in ('alpine_snow_tables.py','enable_alpine_capture.py',
                'run_forecast.py','sitecustomize.py') or 'ch2' in name:
        return False
    return (name.startswith('snow') or name.startswith('run_')) and any(
        word in name for word in ('snow','hybrid','ecmwf','icon','d2')
    )


def hook(path: Path) -> str:
    source = path.read_text(encoding='utf-8')
    if MARKER in source:
        return 'already instrumented'
    tree = ast.parse(source, filename=str(path))
    entrypoint = None
    for node in tree.body:
        if isinstance(node, ast.If):
            if '__name__' in ast.unparse(node.test) and '__main__' in ast.unparse(node.test):
                entrypoint = node
                break
    if entrypoint is None:
        return 'no main entrypoint (skipped)'
    lines = source.splitlines(keepends=True)
    lines[entrypoint.lineno-1:entrypoint.lineno-1] = BLOCK
    _compile_write(path, ''.join(lines))
    return f'capture hook installed at line {entrypoint.lineno}'


def instrument_final_module_wrappers():
    """Capture actual styled plotting functions after wrappers replace them.

    run_icon_d2.py and run_ecmwf.py import Snow* modules then install the
    dark/slate adapters. The adapter may replace plot_snowfall/load_dem,
    so inject a final hook just before mod.main(), after the replacements.
    """
    for name in ('run_icon_d2.py', 'run_ecmwf.py'):
        path = ROOT / name
        if not path.exists():
            print(f'::warning::Wrapper {name} missing; module source hook only')
            continue
        source = path.read_text(encoding='utf-8')
        mark = '# ALPINE_SNOW_TABLES_FINAL_MODULE_HOOK_V3'
        if mark in source:
            continue
        match = re.search(r'^([ \t]*)mod\.main\(\)\s*$', source, flags=re.M)
        if match is None:
            print(f'::warning::{name}: no mod.main(); using existing module hooks only')
            continue
        indent = match.group(1)
        block = (
            indent + mark + '\n' +
            indent + 'from alpine_snow_tables import install_module_capture as _as_capture_final\n' +
            indent + '_as_capture_final(mod.__dict__)\n'
        )
        source = source[:match.start()] + block + source[match.start():]
        _compile_write(path, source)
        print(f'{name}: final styled model hook installed before mod.main()', flush=True)




def _cached_alpine_dem(x, y):
    """Interpolate Hybrid's existing cached terrain onto ICON-D2's cropped grid.

    Only used if the optional DWD HSURF field is unavailable. The cached DEM
    was produced earlier by the ordinary Hybrid model run, using the same HOME
    set by run_forecast.py. No new network request or scientific assumption.
    """
    import numpy as np
    from scipy.interpolate import RegularGridInterpolator

    folder = Path.home() / 'Documents' / 'SneeuwhoogteNL' / 'data'
    expression = re.compile(
        r'^dem_z\d+_([-\d.]+)_([-\d.]+)_([-\d.]+)_([-\d.]+)_([-\d.]+)\.npy$'
    )
    options = []
    if not folder.is_dir():
        return None
    for file in folder.glob('dem_z*.npy'):
        match = expression.match(file.name)
        if not match:
            continue
        step, west, east, south, north = map(float, match.groups())
        # The whole requested cropped grid must fit inside the cached DEM.
        if not (0.001 <= step <= 0.03 and
                west <= float(x.min()) + step and east >= float(x.max()) - step and
                south <= float(y.min()) + step and north >= float(y.max()) - step):
            continue
        options.append((step, file, west, east, south, north))
    for step, file, west, east, south, north in sorted(options):
        try:
            elevation = np.load(file, mmap_mode='r', allow_pickle=False)
            axis_x = np.round(np.arange(west, east + step / 2, step), 5)
            axis_y = np.round(np.arange(south, north + step / 2, step), 5)
            if elevation.shape != (len(axis_y), len(axis_x)):
                continue
            interp = RegularGridInterpolator(
                (axis_y, axis_x), elevation, method='linear',
                bounds_error=False, fill_value=np.nan,
            )
            mesh_x, mesh_y = np.meshgrid(x, y)
            target = interp(np.column_stack((mesh_y.ravel(), mesh_x.ravel())))
            target = target.reshape(mesh_x.shape).astype(np.float32)
            if np.count_nonzero(np.isfinite(target)) < 100:
                continue
            print(f'Alpen tabel: ICON-D2 gebruikt Hybrid DEM cache {file.name}', flush=True)
            return target
        except (OSError, ValueError) as exc:
            print(f'::warning::DEM-cache {file.name} onleesbaar: {exc}', flush=True)
    return None


def capture_icon_d2_region(data, x, y, terrain, area, run, start, end):
    """Capture the *original* ICON-D2 snow field at native grid resolution.

    DWD uses regular 1-D longitude/latitude arrays over a larger European
    domain. Crop to the same Alps domain as the map first, so the DEM and snow
    grid are indisputably aligned and the generic Alpine coordinate checks
    cannot mistake a Central-European native grid for the Alps. The map is
    not altered, and the 0-48h cumulative input is excluded by the caller.
    """
    import numpy as np
    from alpine_snow_tables import register_dem, capture_numeric_field

    if area != 'alps':
        return False
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    snow = np.ma.asarray(data)
    if x.ndim != 1 or y.ndim != 1 or snow.shape != (y.size, x.size):
        print('::warning::ICON-D2 grid expected regular 1-D lon/lat', flush=True)
        return False
    xi = np.flatnonzero((x >= 5.0) & (x <= 16.8))
    yi = np.flatnonzero((y >= 43.6) & (y <= 49.0))
    if xi.size < 80 or yi.size < 80:
        print('::warning::ICON-D2 Alpine coverage too small for table', flush=True)
        return False
    xx, yy = x[xi], y[yi]
    snow_alps = snow[np.ix_(yi, xi)]
    terrain_alps = None
    if terrain is not None:
        elev = np.asanyarray(terrain)
        if elev.shape == snow.shape:
            terrain_alps = elev[np.ix_(yi, xi)]
    if terrain_alps is None or np.count_nonzero(np.isfinite(terrain_alps)) < 100:
        terrain_alps = _cached_alpine_dem(xx, yy)
    if terrain_alps is None:
        print('::warning::ICON-D2: HSURF en Hybrid DEM-cache ontbreken; geen hoogteverdeling', flush=True)
        return False
    if not register_dem(xx, yy, terrain_alps):
        return False
    return capture_numeric_field(snow_alps, xx, yy, area, run, start, end, 'icon_d2')


def install_direct_numeric_captures() -> None:
    """Capture the two models that lack ECMWF's dependable DEM hook.

    Hybrid and ICON-D2 both produced maps but no regional data when only the
    generic Matplotlib/module wrappers were used. Work instead with the arrays
    already in their actual main loops. Nothing in the meteorological methods,
    snowfall arrays, plotting calls, or image format is changed.

    These patches apply only to the short-lived GitHub Actions checkout.
    """
    hybrid_file = ROOT / 'SnowHybrid.py'
    icon_file = ROOT / 'SnowAll_ICON-D2.py'
    if not hybrid_file.is_file() or not icon_file.is_file():
        raise SystemExit('ERROR: Hybrid or ICON-D2 original script missing')

    h_source = hybrid_file.read_text(encoding='utf-8')
    h_marker = '# ALPINE_SNOW_TABLES_HYBRID_DIRECT_V6'
    if h_marker not in h_source:
        # Exactly one live call in SnowHybrid.main(), after array masking.
        h_pat = re.compile(r'^(?P<indent>[ \t]*)lib\.plot_snowfall\(\s*$', re.M)
        h_hits = list(h_pat.finditer(h_source))
        if len(h_hits) != 1:
            raise SystemExit(
                f'ERROR: expected exactly one Hybrid map call; found {len(h_hits)}'
            )
        hit = h_hits[0]
        ind = hit.group('indent')
        block = (
            f'{ind}{h_marker}\n'
            f'{ind}if rain is not None and not total:\n'
            f'{ind}    try:\n'
            f'{ind}        from alpine_snow_tables import register_dem, capture_numeric_field\n'
            f'{ind}        _as_terrain_ok = register_dem(xf, yf, dem)\n'
            f'{ind}        _as_ok = (_as_terrain_ok and capture_numeric_field(\n'
            f'{ind}            snow, xf, yf, AREA, t0, start, end, "hybrid"))\n'
            f'{ind}        print("ALPINE CAPTURE Hybrid +{{}}-{{}}: {{}}".format(\n'
            f'{ind}            start, end, "OK" if _as_ok else "NO DATA"), flush=True)\n'
            f'{ind}    except Exception as _as_err:\n'
            f'{ind}        print(f"::warning::ALPINE CAPTURE Hybrid: {{_as_err}}", flush=True)\n'
        )
        h_source = h_source[:hit.start()] + block + h_source[hit.start():]
        _compile_write(hybrid_file, h_source)
        print('Hybrid: final daily numerical snowfall and DEM capture installed', flush=True)
    else:
        print('Hybrid: direct capture already installed', flush=True)

    d_source = icon_file.read_text(encoding='utf-8')
    d_marker = '# ALPINE_SNOW_TABLES_ICON_D2_DIRECT_V6'
    if d_marker not in d_source:
        # Exactly one call in the ICON-D2 daily snow-map loop; leave
        # 0-48h cumulative map out so snowfall is not double counted.
        d_pat = re.compile(r'^(?P<indent>[ \t]*)plot_snowfall\(\s*$', re.M)
        d_hits = list(d_pat.finditer(d_source))
        if len(d_hits) != 1:
            raise SystemExit(
                f'ERROR: expected exactly one ICON-D2 map call; found {len(d_hits)}'
            )
        hit = d_hits[0]
        ind = hit.group('indent')
        block = (
            f'{ind}{d_marker}\n'
            f'{ind}if area == "alps" and not str(name).startswith("0-48"):\n'
            f'{ind}    try:\n'
            f'{ind}        from enable_alpine_capture import capture_icon_d2_region\n'
            f'{ind}        _as_ok = capture_icon_d2_region(\n'
            f'{ind}            snowfall, x, y, terrain, area, run, start, end)\n'
            f'{ind}        print("ALPINE CAPTURE ICON-D2 +{{}}-{{}}: {{}}".format(\n'
            f'{ind}            start, end, "OK" if _as_ok else "NO DATA"), flush=True)\n'
            f'{ind}    except Exception as _as_err:\n'
            f'{ind}        print(f"::warning::ALPINE CAPTURE ICON-D2: {{_as_err}}", flush=True)\n'
        )
        d_source = d_source[:hit.start()] + block + d_source[hit.start():]
        _compile_write(icon_file, d_source)
        print('ICON-D2: final snow-map numerical snowfall and HSURF capture installed', flush=True)
    else:
        print('ICON-D2: direct capture already installed', flush=True)


def main():
    # Do not instrument the shared CH2 science library because its functions
    # are loaded by the Hybrid module; they are not standalone CH2 forecasts.
    disable_hybrid_ch2_input()
    disable_standalone_ch2_and_tab()
    instrument_final_module_wrappers()
    install_direct_numeric_captures()
    patched = 0
    for path in sorted(ROOT.glob('*.py')):
        if eligible(path):
            status = hook(path)
            print(f'Capture: {path.name}: {status}', flush=True)
            if 'installed' in status or 'instrumented' in status:
                patched += 1
    if patched == 0:
        raise SystemExit('ERROR: no numerical model hook was installed')
    print(f'Capture hooks active in {patched} model runner(s)', flush=True)


if __name__ == '__main__':
    main()
