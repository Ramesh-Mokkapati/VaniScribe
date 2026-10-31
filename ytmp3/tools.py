"""Locating the external tools and model files the app bundles (FFmpeg, Deno/Node, whisper.cpp)."""
from __future__ import annotations

import os
import platform
import shutil
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _vendor_key() -> str:
    """Name of this platform's vendor\\<key>\\bin folder, matching the build scripts."""
    if sys.platform == "win32":
        return "x64" if sys.maxsize > 2**32 else "x86"
    if sys.platform == "darwin":
        return "macos-arm64" if platform.machine() == "arm64" else "macos-x64"
    return f"{sys.platform}-{platform.machine()}"


VENDOR_KEY = _vendor_key()


def app_dir() -> str:
    """Folder the user sees the app in: next to the exe when frozen (outside a macOS .app bundle)."""
    if not getattr(sys, "frozen", False):
        return PROJECT_ROOT
    exe_dir = os.path.dirname(sys.executable)
    if sys.platform == "darwin" and exe_dir.endswith(os.path.join("Contents", "MacOS")):
        return os.path.dirname(os.path.dirname(os.path.dirname(exe_dir)))
    return exe_dir


def user_data_dir() -> str:
    """Per-user writable folder, e.g. %LOCALAPPDATA%\\VaniScribe or ~/Library/Application Support/VaniScribe."""
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    elif sys.platform == "darwin":
        base = os.path.expanduser("~/Library/Application Support")
    else:
        base = os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")
    return os.path.join(base, "VaniScribe")


def bundle_dirs() -> list[str]:
    """Folders that may hold bundled tools, most specific first."""
    dirs = []
    if getattr(sys, "frozen", False):
        dirs.append(os.path.join(getattr(sys, "_MEIPASS", ""), "bin"))
        dirs.append(os.path.dirname(sys.executable))
    dirs.append(os.path.join(PROJECT_ROOT, "vendor", VENDOR_KEY, "bin"))
    return dirs


def find_tool(name: str) -> str | None:
    exe = name + ".exe" if sys.platform == "win32" else name
    for folder in bundle_dirs():
        candidate = os.path.join(folder, exe)
        if os.path.isfile(candidate):
            return candidate
    return shutil.which(name)


def model_dirs() -> list[str]:
    """Where models are looked for: inside the macOS app bundle, next to the app, then the user's folder.

    On macOS the models ship inside VaniScribe.app (Contents/Resources/models) so that dragging the
    app to Applications keeps them; on Windows they sit in a models folder next to the exe.
    """
    dirs = []
    if getattr(sys, "frozen", False) and sys.platform == "darwin":
        contents = os.path.dirname(os.path.dirname(sys.executable))  # .../VaniScribe.app/Contents
        dirs.append(os.path.join(contents, "Resources", "models"))
    dirs += [os.path.join(app_dir(), "models"), os.path.join(user_data_dir(), "models")]
    return dirs


def find_model(filename: str) -> str | None:
    for folder in model_dirs():
        candidate = os.path.join(folder, filename)
        if os.path.isfile(candidate):
            return candidate
    return None


def short_path(path: str) -> str:
    """8.3 short form of an existing Windows path.

    whisper.cpp reads its command line in the ANSI code page on Windows, so a path containing
    Telugu or other non-ASCII characters would be garbled; the short form is plain ASCII.
    """
    if sys.platform != "win32":
        return path
    import ctypes

    buf = ctypes.create_unicode_buffer(1024)
    n = ctypes.windll.kernel32.GetShortPathNameW(path, buf, len(buf))
    return buf.value if 0 < n < len(buf) else path
