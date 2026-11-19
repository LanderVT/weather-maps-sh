"""Temporarily instrument the existing scientific model scripts for GitHub Actions.

Run this on the CI checkout (before run_forecast.py), NOT as a permanent rewrite
of the scientific equations. The only inserted block installs a read-only hook
that records the numeric snowfall map arrays before PNG rendering.
"""
from pathlib import Path
import ast
import sys

ROOT = Path(__file__).resolve().parent
TARGETS = (
    'SnowHybrid.py',
    'SnowDays_ECMWF.py',
    'SnowDays_ICON-D2.py',
    'SnowDays_ICON-CH2.py',
)
MARKER = '# ALPINE_SNOW_TABLES_CAPTURE_INSTALL'
BLOCK = [
    f'{MARKER}\n',
    'try:\n',
    '    from alpine_snow_tables import install_matplotlib_capture as _install_alpine_snow_capture\n',
    '    _install_alpine_snow_capture()\n',
    'except Exception as _snow_tables_exc:\n',
    '    print(f"::warning::Alpine snowfall table capture unavailable: {_snow_tables_exc}", flush=True)\n',
    '\n',
]


def instrument(path: Path):
    original = path.read_text(encoding='utf-8')
    if MARKER in original:
        return 'already patched'
    tree = ast.parse(original, filename=str(path))
    entry = next((node for node in tree.body if isinstance(node, (ast.FunctionDef,ast.AsyncFunctionDef)) and node.name == 'main'), None)
    if entry is None:
        # A model without a main() is left intact (data may still be covered by other models).
        return 'no main() found (skipped)'
    pos = min([entry.lineno] + [d.lineno for d in entry.decorator_list]) - 1
    lines = original.splitlines(keepends=True)
    lines[pos:pos] = BLOCK
    new = ''.join(lines)
    compile(new, str(path), 'exec')
    path.write_text(new, encoding='utf-8')
    return f'instrumented before main() at line {pos+1}'


def main():
    ok = 0
    for filename in TARGETS:
        path = ROOT / filename
        if not path.is_file():
            print(f'::warning::{filename}: not found, skipped')
            continue
        result = instrument(path)
        if result.startswith('instrumented') or result == 'already patched':
            ok += 1
        print(f'Alpen-tabellen {filename}: {result}')
    if not ok:
        raise SystemExit('No scientific model scripts could be instrumented')


if __name__ == '__main__':
    main()
