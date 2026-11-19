"""Prepare an ephemeral GitHub Actions checkout for Alpine snow tables.

This file edits scripts ONLY in the checked-out Actions runner, not in GitHub.
A) Exclude ICON-CH2 as a Hybrid input while keeping its shared plotting/
   snowpack library (SnowHybrid.main requires MODEL_FILES['ICON-CH2']).
B) Remove standalone ICON-CH2 from run_forecast's jobs and website tabs.
C) Add numerical snowfall/DEM capture hooks to the existing science modules.
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


def main():
    # Do not instrument the shared CH2 science library because its functions
    # are loaded by the Hybrid module; they are not standalone CH2 forecasts.
    disable_hybrid_ch2_input()
    disable_standalone_ch2_and_tab()
    instrument_final_module_wrappers()
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
