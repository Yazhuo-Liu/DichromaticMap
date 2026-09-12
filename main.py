"""Run the viewer directly from a source checkout: python main.py."""

from pathlib import Path
import sys

# Only this checkout launcher adjusts the path. Installed users import the
# package normally; no GUI is imported until main() parses the arguments.
_source = str(Path(__file__).resolve().parent / "src")
if _source not in sys.path:
    sys.path.insert(0, _source)

from dichromatic_map.__main__ import main


if __name__ == "__main__":
    main()
