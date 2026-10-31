"""YouTube -> MP3 download engine built on yt-dlp.

A DownloadJob runs in a background thread and reports what it is doing
through a DownloadListener. Listener methods are called from the worker
thread, so GUI code must marshal them onto the UI thread itself.
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass

import yt_dlp
from yt_dlp.utils import DownloadCancelled

from .tools import find_tool

BITRATES = ("320", "256", "192", "128")


@dataclass
class DownloadRequest:
    url: str
    output_dir: str
    bitrate: str = "192"
    whole_playlist: bool = True
    embed_tags: bool = True


class DownloadListener:
    """Receives job events. Override the methods you care about."""

    def log(self, message: str, level: str = "info") -> None:
        pass

    def playlist_found(self, title: str, count: int) -> None:
        pass

    def item_started(self, index: int, total: int, title: str) -> None:
        pass

    def progress(self, item_fraction: float, overall_fraction: float, detail: str) -> None:
        pass

    def file_saved(self, path: str) -> None:
        """An MP3 is complete on disk (after conversion, tagging and cover art)."""

    def finished(self, saved: int, failed: int, cancelled: bool, error: str | None) -> None:
        pass


def find_ffmpeg() -> str | None:
    """Return the folder containing ffmpeg (yt-dlp also looks for ffprobe there)."""
    path = find_tool("ffmpeg")
    return os.path.dirname(path) if path else None


def find_js_runtime() -> tuple[str, str] | None:
    """yt-dlp needs a JavaScript runtime to solve YouTube's signature challenges.

    The x64 build bundles Deno; the x86 build bundles Node.js because Deno has no 32-bit build.
    """
    for name in ("deno", "node"):
        path = find_tool(name)
        if path:
            return name, path
    return None


def _format_bytes(n: float | None) -> str:
    if not n:
        return "?"
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024 or unit == "GiB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return "?"


class _Logger:
    """Routes yt-dlp's console output to the listener."""

    def __init__(self, listener: DownloadListener):
        self._listener = listener

    def debug(self, msg: str) -> None:
        # yt-dlp sends ordinary info lines to debug() too; only real debug lines are prefixed.
        if not msg.startswith("[debug] "):
            self.info(msg)

    def info(self, msg: str) -> None:
        # yt-dlp probes many thumbnail sizes before finding one; those lines are just noise.
        if "Video Thumbnail" in msg or "Downloading video thumbnail" in msg:
            return
        self._listener.log(msg, "info")

    def warning(self, msg: str) -> None:
        self._listener.log(msg, "warning")

    def error(self, msg: str) -> None:
        self._listener.log(msg, "error")


class DownloadJob(threading.Thread):
    def __init__(self, request: DownloadRequest, listener: DownloadListener):
        super().__init__(daemon=True)
        self.request = request
        self.listener = listener
        self._cancel = threading.Event()
        self._total = 1
        self._index = 0
        self._current_id = None
        self._saved_ids: set[str] = set()

    def cancel(self) -> None:
        self._cancel.set()

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def run(self) -> None:
        error = None
        try:
            self._run()
        except DownloadCancelled:
            pass
        except Exception as exc:  # yt-dlp raises many unrelated types; surface them all
            if not self.cancelled:
                error = str(exc)
        failed = 0 if self.cancelled else max(0, self._total - len(self._saved_ids))
        self.listener.finished(len(self._saved_ids), failed, self.cancelled, error)

    def _check_cancel(self) -> None:
        if self.cancelled:
            raise DownloadCancelled("Cancelled by user")

    def _base_options(self) -> dict:
        ffmpeg_dir = find_ffmpeg()
        if not ffmpeg_dir:
            raise RuntimeError(
                "FFmpeg was not found. Install it and add it to PATH, "
                "or run build.ps1 which downloads it into vendor\\bin."
            )
        opts = {
            "logger": _Logger(self.listener),
            "quiet": True,
            "noprogress": True,
            "noplaylist": not self.request.whole_playlist,
            "ffmpeg_location": ffmpeg_dir,
            "windowsfilenames": True,
        }
        runtime = find_js_runtime()
        if runtime:
            name, path = runtime
            opts["js_runtimes"] = {name: {"path": path}}
        return opts

    def _run(self) -> None:
        req = self.request
        os.makedirs(req.output_dir, exist_ok=True)
        base = self._base_options()

        # Quick flat lookup so we know whether this is a playlist and how many items it has.
        self.listener.log(f"Looking up {req.url}")
        with yt_dlp.YoutubeDL({**base, "extract_flat": "in_playlist"}) as ydl:
            info = ydl.extract_info(req.url, download=False)
        self._check_cancel()
        if not info:
            raise RuntimeError("Could not read that link.")

        out_dir = req.output_dir.replace("%", "%%")  # outtmpl treats % as a field marker
        if info.get("_type") == "playlist":
            entries = [e for e in (info.get("entries") or []) if e]
            self._total = max(1, len(entries))
            self.listener.playlist_found(info.get("title") or "Playlist", len(entries))
            outtmpl = os.path.join(
                out_dir, "%(playlist_title,playlist_id)s", "%(playlist_index)03d - %(title)s.%(ext)s"
            )
        else:
            self._total = 1
            outtmpl = os.path.join(out_dir, "%(title)s.%(ext)s")

        postprocessors = [
            {"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": req.bitrate},
        ]
        if req.embed_tags:
            postprocessors += [
                {"key": "FFmpegMetadata", "add_metadata": True},
                {"key": "EmbedThumbnail"},
            ]

        opts = {
            **base,
            "format": "bestaudio/best",
            "outtmpl": outtmpl,
            "ignoreerrors": True,  # one unavailable video must not stop a whole playlist
            "writethumbnail": req.embed_tags,
            "postprocessors": postprocessors,
            "progress_hooks": [self._on_download],
            "postprocessor_hooks": [self._on_postprocess],
            # Called once per video with the final file path, after every post-processor has run.
            "post_hooks": [self._on_saved],
        }
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([req.url])

    def _note_item(self, info: dict) -> None:
        video_id = info.get("id")
        if video_id and video_id != self._current_id:
            self._current_id = video_id
            self._index = info.get("playlist_autonumber") or self._index + 1
            self.listener.item_started(self._index, self._total, info.get("title") or video_id)

    def _report(self, item_fraction: float, detail: str) -> None:
        overall = (max(self._index, 1) - 1 + item_fraction) / self._total
        self.listener.progress(item_fraction, min(overall, 1.0), detail)

    def _on_download(self, d: dict) -> None:
        self._check_cancel()
        self._note_item(d.get("info_dict") or {})
        if d["status"] == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate")
            done = d.get("downloaded_bytes") or 0
            # Downloading is 90% of an item; the MP3 conversion is the rest.
            fraction = min(done / total, 1.0) * 0.9 if total else 0.0
            detail = f"{_format_bytes(done)} of {_format_bytes(total)}"
            if d.get("speed"):
                detail += f" at {_format_bytes(d['speed'])}/s"
            if d.get("eta") is not None:
                detail += f", {int(d['eta'])}s left"
            self._report(fraction, detail)
        elif d["status"] == "finished":
            self._report(0.9, "Converting to MP3...")

    def _on_saved(self, path: str) -> None:
        self.listener.file_saved(os.path.abspath(path))

    def _on_postprocess(self, d: dict) -> None:
        self._check_cancel()
        if d.get("postprocessor") == "ExtractAudio" and d["status"] == "finished":
            # yt-dlp can report "finished" more than once per file, so count videos, not events.
            info = d.get("info_dict") or {}
            self._saved_ids.add(info.get("id") or info.get("filepath") or str(self._index))
            self._report(1.0, "Saved")
