# PyInstaller spec, with FFmpeg and a JS runtime (Deno or Node) bundled under bin/.
#   Windows: one-file, windowed dist\VaniScribe-<arch>.exe   (build.ps1 sets VANISCRIBE_ARCH to x64 or x86)
#   macOS:   dist/VaniScribe.app                             (build_macos.sh sets VANISCRIBE_ARCH to macos-arm64 or macos-x64)
# The build scripts run it with the matching Python, e.g.:
#   $env:VANISCRIBE_ARCH = "x86"; .venv-x86\Scripts\python -m PyInstaller --noconfirm --clean ytmp3.spec
import glob
import importlib.util
import os
import sys

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

sys.path.insert(0, os.path.abspath("."))
from ytmp3 import __version__

arch = os.environ.get("VANISCRIBE_ARCH", "x64")
vendored = [
    (path, "bin")
    for path in glob.glob(f"vendor/{arch}/bin/*")
    if os.path.isfile(path) and not path.endswith(".txt")
]
# CTranslate2 (64-bit builds only) loads ctranslate2.dll and its OpenMP runtime from its own
# folder at run time; PyInstaller has no hook for it, so collect them explicitly.
if importlib.util.find_spec("ctranslate2"):
    vendored += collect_dynamic_libs("ctranslate2")

a = Analysis(
    ["run_app.py"],
    pathex=[],
    binaries=vendored,
    # yt-dlp loads extractors and the YouTube challenge solver scripts dynamically;
    # python-docx builds every document from its bundled default.docx template.
    datas=collect_data_files("yt_dlp_ejs") + collect_data_files("docx"),
    hiddenimports=collect_submodules("yt_dlp") + collect_submodules("yt_dlp_ejs"),
    excludes=["tkinter"],
    noarchive=False,
)
pyz = PYZ(a.pure)

if sys.platform == "darwin":
    # A one-folder build wrapped in an .app bundle: the standard, fast-starting form on macOS.
    exe = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name="VaniScribe",
        debug=False,
        strip=False,
        upx=False,
        console=False,
    )
    coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="VaniScribe")
    app = BUNDLE(
        coll,
        name="VaniScribe.app",
        bundle_identifier="com.vaniscribe.app",
        version=__version__,
        info_plist={
            "CFBundleDisplayName": "VaniScribe",
            "CFBundleShortVersionString": __version__,
            "NSHighResolutionCapable": True,
            "LSMinimumSystemVersion": "11.0",
        },
    )
else:
    exe = EXE(
        pyz,
        a.scripts,
        a.binaries,
        a.datas,
        [],
        name=f"VaniScribe-{arch}",
        debug=False,
        strip=False,
        upx=False,
        console=False,
        version="version_info.txt",
    )
