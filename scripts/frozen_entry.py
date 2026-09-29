"""Executable entry point: dispatch spawned workers before importing NumPy/Qt."""

import multiprocessing
import os
from pathlib import Path
import sys


if __name__ == "__main__":
    # Windowed Windows/macOS bootloaders may have no attached standard streams.
    for stream in ("stdout", "stderr"):
        if getattr(sys, stream) is None:
            setattr(sys, stream, open(os.devnull, "w", encoding="utf-8"))
    multiprocessing.freeze_support()
    if len(sys.argv) == 3 and sys.argv[1] == "--self-test":
        import traceback
        from frozen_smoke import run

        output = Path(sys.argv[2])
        output.mkdir(parents=True, exist_ok=True)
        try:
            run(output)
        except Exception:
            (output / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
            raise SystemExit(1)
    else:
        from dichromatic_map.__main__ import main

        main()
