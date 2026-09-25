"""Keep the editable install importable when the virtualenv is flagged "hidden".

Python 3.12 and later deliberately skip any ``.pth`` file whose macOS ``hidden`` flag
(``UF_HIDDEN``) is set. Some tools on macOS recursively flag dot-directories such as
``.venv`` as hidden, after which ``uv run bankcanary`` fails with
``ModuleNotFoundError: No module named 'bankcanary'`` even though the ``.pth`` file is
correct. A ``sitecustomize`` module goes through the normal import system, which ignores
the flag, so it is a stable place to put the ``src/`` path. This script writes that module
into the active environment and clears the hidden flags. Run it via ``make fix-venv``.
"""

from __future__ import annotations

import subprocess
import sys
import sysconfig
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

TEMPLATE = '''"""Keeps src/ importable when .pth files are hidden (scripts/fix_venv.py)."""

import sys

_src = {src!r}
if _src not in sys.path:
    sys.path.insert(0, _src)
'''


def main() -> int:
    site_packages = Path(sysconfig.get_paths()["purelib"])
    target = site_packages / "sitecustomize.py"
    content = TEMPLATE.format(src=str(SRC))
    if not target.exists() or target.read_text(encoding="utf-8") != content:
        target.write_text(content, encoding="utf-8")
        print(f"wrote {target}")
    else:
        print(f"up to date: {target}")
    if sys.platform == "darwin":
        subprocess.run(["chflags", "-R", "nohidden", sys.prefix], check=False)
        print(f"cleared hidden flags under {sys.prefix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
