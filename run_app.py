"""Entry point for running from source and for the PyInstaller build."""
import os
import sys

# A windowed (no console) exe has no stdout/stderr; give libraries somewhere harmless to write.
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w", encoding="utf-8")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w", encoding="utf-8")

from ytmp3.gui import main

if __name__ == "__main__":
    main()
