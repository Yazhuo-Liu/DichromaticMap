"""Command-line entry point; GUI imports happen only when starting a viewer."""

from __future__ import annotations
import argparse
import multiprocessing
import os
from pathlib import Path
from .crystal import SUPPORTED_LATTICES
from .state import PatternParameters


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone FCC/BCC/SC integer-axis tilt-GB viewer; coordinates in a0"
    )
    parser.add_argument(
        "--angle",
        type=float,
        default=None,
        help="degrees; default is Sigma9 for 110, Sigma5 for 100, lowest-Sigma preset otherwise",
    )
    parser.add_argument("--lattice", choices=SUPPORTED_LATTICES, default="FCC")
    parser.add_argument(
        "--axis",
        default="110",
        help='Tilt axis: 100, 110, 111, 112, or an integer triple, e.g. "1 -1 3"',
    )
    parser.add_argument(
        "--lattice-constant",
        type=float,
        default=3.52,
        help="reference a0 in Angstrom; display is normalized by a0",
    )
    parser.add_argument(
        "--width", type=float, default=12.0, help="base view width in a0 (not Angstrom)"
    )
    parser.add_argument(
        "--height",
        type=float,
        default=9.0,
        help="base view height in a0 (not Angstrom)",
    )
    parser.add_argument("--marker-size", type=float, default=32.0)
    parser.add_argument(
        "--view-scale",
        type=float,
        default=1.0,
        help="initial field-size multiplier, 0.1 to 5 (default: %(default)s)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(4, max(1, os.cpu_count() or 1)),
        help="calculation processes; use 1 to disable multiprocessing (default: %(default)s)",
    )
    parser.add_argument("--save", type=Path, metavar="PNG")
    return parser.parse_args()


def main() -> None:
    multiprocessing.freeze_support()
    arguments = parse_arguments()
    from .ui.controls import create_application
    from .ui.window import DichromaticPatternWindow

    parameters = PatternParameters(
        angle_deg=arguments.angle,
        lattice_constant=arguments.lattice_constant,
        width=arguments.width,
        height=arguments.height,
        marker_size=arguments.marker_size,
        view_scale=arguments.view_scale,
        lattice=arguments.lattice,
        axis=arguments.axis,
    )
    application = create_application()
    window = DichromaticPatternWindow(parameters, worker_count=arguments.workers)
    window.show()
    application.processEvents()
    if arguments.save is not None:
        window.save(arguments.save)
        print(f"Saved dichromatic pattern to {arguments.save}")
        window.close()
        return
    raise SystemExit(application.exec())


if __name__ == "__main__":
    main()
