"""Build distribution packages for PyPI release (package name: dynopy).

Temporarily sets ``name = "dynopy"`` in ``pyproject.toml`` during build
so that PyPI receives distributions named ``dynopy`` (importable as ``dyno``),
while preserving ``name = "dyno"`` in the repository for local/conda development.
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build PyPI distributions named dynopy"
    )
    parser.add_argument(
        "--out-dir", default="dist", help="Output directory for build artifacts"
    )
    args, unknown = parser.parse_known_args()

    content = PYPROJECT.read_text()
    if 'name = "dyno"' not in content:
        print('error: expected name = "dyno" in pyproject.toml', file=sys.stderr)
        return 1

    pypi_content = content.replace('name = "dyno"', 'name = "dynopy"', 1)

    try:
        PYPROJECT.write_text(pypi_content)
        cmd = ["uv", "build", "--out-dir", args.out_dir] + unknown
        print(f"Building PyPI distributions with package name 'dynopy': {' '.join(cmd)}")
        res = subprocess.run(cmd, cwd=ROOT)
        return res.returncode
    finally:
        PYPROJECT.write_text(content)
        print('Restored name = "dyno" in pyproject.toml')


if __name__ == "__main__":
    sys.exit(main())
