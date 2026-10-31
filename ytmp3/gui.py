"""wxPython front end: a "Download MP3" tab and a "Transcribe to Word" tab."""
from __future__ import annotations

import os
import subprocess
import sys

import wx

from . import __version__
from .downloader import BITRATES, DownloadJob, DownloadListener, DownloadRequest
from .transcriber import (
    TranscribeJob, TranscribeListener, TranscribeRequest, available_languages, missing_components,
    translation_available,
)

APP_NAME = "VaniScribe"
DEFAULT_BITRATE = "192"
AUDIO_EXTENSIONS = (".mp3", ".m4a", ".aac", ".wav", ".flac", ".ogg", ".opus", ".webm", ".mp4", ".wma")

LOG_COLOURS = {
    "info": wx.Colour(60, 60, 60),
    "warning": wx.Colour(180, 110, 0),
    "error": wx.Colour(200, 30, 30),
    "success": wx.Colour(0, 130, 60),
}


def _post(target: wx.Window, method_name: str, *args) -> None:
    """Call target.method_name(*args) on the UI thread, unless the window has been destroyed."""
    def call():
        if target:
            getattr(target, method_name)(*args)
    wx.CallAfter(call)


def _open_in_file_manager(path: str) -> None:
    if sys.platform == "win32":
        os.startfile(path)
    else:
        subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", path])


_caffeinate: subprocess.Popen | None = None


def _keep_awake(on: bool) -> None:
    """Stop the computer from sleeping while a long job runs (the screen may still turn off)."""
    global _caffeinate
    if sys.platform == "win32":
        import ctypes

        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        flags = ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0)
        ctypes.windll.kernel32.SetThreadExecutionState(flags)
    elif sys.platform == "darwin":
        # caffeinate -i holds off idle sleep; -w ends it automatically if the app quits or crashes.
        if on and _caffeinate is None:
            try:
                _caffeinate = subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())])
            except OSError:
                _caffeinate = None
        elif not on and _caffeinate is not None:
            _caffeinate.terminate()
            _caffeinate = None


class _DownloadEvents(DownloadListener):
    def __init__(self, panel: "DownloadPanel"):
        self._panel = panel

    def log(self, message, level="info"):
        _post(self._panel, "append_log", message, level)

    def playlist_found(self, title, count):
        _post(self._panel, "append_log", f"Playlist '{title}' has {count} item(s).", "success")

    def item_started(self, index, total, title):
        _post(self._panel, "on_item_started", index, total, title)

    def progress(self, item_fraction, overall_fraction, detail):
        _post(self._panel, "on_progress", item_fraction, overall_fraction, detail)

    def file_saved(self, path):
        _post(self._panel, "on_file_saved", path)

    def finished(self, saved, failed, cancelled, error):
        _post(self._panel, "on_finished", saved, failed, cancelled, error)


class _TranscribeEvents(TranscribeListener):
    def __init__(self, panel: "TranscribePanel"):
        self._panel = panel

    def log(self, message, level="info"):
        _post(self._panel, "append_log", message, level)

    def file_started(self, index, total, path):
        _post(self._panel, "on_item_started", index, total, os.path.basename(path))

    def progress(self, file_fraction, overall_fraction, detail):
        _post(self._panel, "on_progress", file_fraction, overall_fraction, detail)

    def file_finished(self, path, document):
        _post(self._panel, "on_file_finished", path, document)

    def finished(self, done, failed, cancelled, error):
        _post(self._panel, "on_finished", done, failed, cancelled, error)


class _JobPanel(wx.Panel):
    """Shared layout for a tab that runs a background job: form, buttons, progress and log."""

    action_label = "Start"
    activity = "working"
    item_word = "file"

    def __init__(self, parent: wx.Window, config: wx.ConfigBase):
        super().__init__(parent)
        self.config = config
        self.job = None

        self.root = wx.BoxSizer(wx.VERTICAL)
        self.form = wx.FlexGridSizer(cols=2, vgap=8, hgap=8)
        self.form.AddGrowableCol(1)
        self.build_form()
        self.root.Add(self.form, 0, wx.EXPAND | wx.ALL, 12)

        buttons = wx.BoxSizer(wx.HORIZONTAL)
        self.start_btn = wx.Button(self, label=self.action_label)
        self.cancel_btn = wx.Button(self, label="Cancel")
        self.cancel_btn.Disable()
        open_btn = wx.Button(self, label="Open folder")
        buttons.Add(self.start_btn, 0, wx.RIGHT, 6)
        buttons.Add(self.cancel_btn, 0, wx.RIGHT, 6)
        self.add_buttons(buttons)
        buttons.AddStretchSpacer()
        buttons.Add(open_btn)
        self.root.Add(buttons, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 12)

        self.status = wx.StaticText(self, label="Ready.")
        self.detail = wx.StaticText(self, label="")
        self.item_gauge = wx.Gauge(self, range=1000)
        self.overall_gauge = wx.Gauge(self, range=1000)
        progress = wx.FlexGridSizer(cols=2, vgap=4, hgap=8)
        progress.AddGrowableCol(1)
        progress.Add(wx.StaticText(self, label="Current:"), 0, wx.ALIGN_CENTER_VERTICAL)
        progress.Add(self.item_gauge, 1, wx.EXPAND)
        progress.Add(wx.StaticText(self, label="Overall:"), 0, wx.ALIGN_CENTER_VERTICAL)
        progress.Add(self.overall_gauge, 1, wx.EXPAND)
        self.root.Add(self.status, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP, 12)
        self.root.Add(self.detail, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP, 6)
        self.root.Add(progress, 0, wx.EXPAND | wx.ALL, 12)

        self.log = wx.TextCtrl(self, style=wx.TE_MULTILINE | wx.TE_READONLY | wx.TE_RICH2 | wx.HSCROLL)
        self.root.Add(self.log, 1, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 12)
        self.SetSizer(self.root)

        self.start_btn.Bind(wx.EVT_BUTTON, self.on_start)
        self.cancel_btn.Bind(wx.EVT_BUTTON, self.on_cancel)
        open_btn.Bind(wx.EVT_BUTTON, self.on_open_folder)

    # Subclasses fill these in.
    def build_form(self) -> None: ...
    def add_buttons(self, row: wx.BoxSizer) -> None: ...
    def make_job(self): ...
    def output_folder(self) -> str: ...
    def form_controls(self) -> list[wx.Window]: ...
    def save_settings(self) -> None: ...
    def summary(self, done: int, failed: int) -> str: ...

    @property
    def busy(self) -> bool:
        return bool(self.job and self.job.is_alive())

    def add_row(self, label: str, item, proportion: int = 1) -> None:
        self.form.Add(wx.StaticText(self, label=label), 0, wx.ALIGN_CENTER_VERTICAL)
        self.form.Add(item, proportion, wx.EXPAND)

    def on_start(self, _event):
        if self.busy:
            return
        job = self.make_job()
        if job is None:
            return
        self.save_settings()
        self.log.Clear()
        self.item_gauge.SetValue(0)
        self.overall_gauge.SetValue(0)
        self.status.SetLabel("Starting...")
        self.detail.SetLabel("")
        self.set_busy(True)
        self.job = job
        self.job.start()

    def on_cancel(self, _event):
        if self.busy:
            self.job.cancel()
            self.cancel_btn.Disable()
            self.status.SetLabel("Cancelling...")

    def on_open_folder(self, _event):
        path = self.output_folder()
        if not path or not os.path.isdir(path):
            wx.MessageBox("That folder does not exist yet.", APP_NAME, wx.ICON_INFORMATION, self)
            return
        _open_in_file_manager(path)

    def set_busy(self, busy: bool):
        self.start_btn.Enable(not busy)
        self.cancel_btn.Enable(busy)
        for ctrl in self.form_controls():
            ctrl.Enable(not busy)

    # ----- job events (always on the UI thread) -----

    def append_log(self, message: str, level: str = "info"):
        self.log.SetDefaultStyle(wx.TextAttr(LOG_COLOURS.get(level, LOG_COLOURS["info"])))
        self.log.AppendText(message.rstrip() + "\n")

    def on_item_started(self, index: int, total: int, title: str):
        prefix = f"[{index}/{total}] " if total > 1 else ""
        self.status.SetLabel(f"{prefix}{title}")

    def on_progress(self, item_fraction: float, overall_fraction: float, detail: str):
        self.item_gauge.SetValue(int(item_fraction * 1000))
        self.overall_gauge.SetValue(int(overall_fraction * 1000))
        self.detail.SetLabel(detail)

    def on_finished(self, done: int, failed: int, cancelled: bool, error: str | None):
        self.set_busy(False)
        self.detail.SetLabel("")
        if error:
            self.status.SetLabel("Failed.")
            self.append_log(error, "error")
            wx.MessageBox(error, APP_NAME, wx.ICON_ERROR, self)
            return
        if cancelled:
            self.status.SetLabel(f"Cancelled. {done} {self.item_word}(s) saved.")
            self.append_log("Cancelled.", "warning")
            return
        self.overall_gauge.SetValue(1000)
        summary = self.summary(done, failed)
        self.status.SetLabel(summary)
        self.append_log(summary, "warning" if failed else "success")


class DownloadPanel(_JobPanel):
    action_label = "Download MP3"
    activity = "downloading"
    item_word = "file"

    def build_form(self):
        url_row = wx.BoxSizer(wx.HORIZONTAL)
        self.url = wx.TextCtrl(self, style=wx.TE_PROCESS_ENTER)
        self.url.SetHint("https://www.youtube.com/watch?v=...  or  https://www.youtube.com/playlist?list=...")
        paste = wx.Button(self, label="Paste")
        url_row.Add(self.url, 1, wx.EXPAND | wx.RIGHT, 6)
        url_row.Add(paste)
        self.add_row("Video or playlist URL:", url_row)

        default_dir = os.path.join(os.path.expanduser("~"), "Music", "VaniScribe")
        self.folder = wx.DirPickerCtrl(
            self,
            path=self.config.Read("output_dir", default_dir),
            message="Choose where to save MP3 files",
            style=wx.DIRP_USE_TEXTCTRL | wx.DIRP_SMALL,
        )
        self.add_row("Save to:", self.folder)

        opts_row = wx.BoxSizer(wx.HORIZONTAL)
        self.bitrate = wx.Choice(self, choices=[f"{b} kbps" for b in BITRATES])
        saved_bitrate = self.config.Read("bitrate", DEFAULT_BITRATE)
        self.bitrate.SetSelection(BITRATES.index(saved_bitrate) if saved_bitrate in BITRATES else 2)
        self.whole_playlist = wx.CheckBox(self, label="Download whole playlist if the link has one")
        self.whole_playlist.SetValue(self.config.ReadBool("whole_playlist", True))
        self.embed_tags = wx.CheckBox(self, label="Add title tags and cover art")
        self.embed_tags.SetValue(self.config.ReadBool("embed_tags", True))
        opts_row.Add(self.bitrate, 0, wx.RIGHT, 16)
        opts_row.Add(self.whole_playlist, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 16)
        opts_row.Add(self.embed_tags, 0, wx.ALIGN_CENTER_VERTICAL)
        self.add_row("MP3 quality:", opts_row)

        # Hand finished downloads straight to the Transcribe tab. Left enabled during a download
        # so it can still be ticked halfway through a long playlist.
        self.auto_transcribe = wx.CheckBox(
            self, label="Then transcribe the MP3s to Word (uses the Transcribe tab's settings, "
                        "including English translation if ticked there)")
        self.auto_transcribe.SetValue(self.config.ReadBool("auto_transcribe", False))
        if missing_components():
            self.auto_transcribe.SetValue(False)
            self.auto_transcribe.Disable()
            self.auto_transcribe.SetToolTip("Transcription isn't installed in this copy of VaniScribe.")
        self.add_row("After download:", self.auto_transcribe)

        self.send_to_transcribe = None  # set by MainFrame: callable(paths, start)
        self.saved_files: list[str] = []
        paste.Bind(wx.EVT_BUTTON, self.on_paste)
        self.url.Bind(wx.EVT_TEXT_ENTER, self.on_start)

    def add_buttons(self, row):
        self.transcribe_btn = wx.Button(self, label="Transcribe these")
        self.transcribe_btn.SetToolTip("Add the MP3s from the last download to the Transcribe tab's list")
        self.transcribe_btn.Disable()
        self.transcribe_btn.Bind(wx.EVT_BUTTON, self.on_transcribe_these)
        row.Add(self.transcribe_btn, 0, wx.RIGHT, 6)

    def on_start(self, event):
        if self.busy:
            return
        self.saved_files = []
        self.transcribe_btn.Disable()
        super().on_start(event)

    def on_file_saved(self, path: str):
        if path not in self.saved_files:
            self.saved_files.append(path)

    def on_finished(self, done, failed, cancelled, error):
        super().on_finished(done, failed, cancelled, error)
        files = [f for f in self.saved_files if os.path.isfile(f)]
        self.transcribe_btn.Enable(bool(files) and not missing_components())
        if files and not cancelled and not error and self.auto_transcribe.IsEnabled() \
                and self.auto_transcribe.GetValue() and self.send_to_transcribe:
            self.append_log(f"Sending {len(files)} MP3 file(s) to the Transcribe tab.", "success")
            self.send_to_transcribe(files, True)

    def on_transcribe_these(self, _event):
        files = [f for f in self.saved_files if os.path.isfile(f)]
        if files and self.send_to_transcribe:
            self.send_to_transcribe(files, False)

    def on_paste(self, _event):
        if wx.TheClipboard.Open():
            try:
                data = wx.TextDataObject()
                if wx.TheClipboard.GetData(data):
                    self.url.SetValue(data.GetText().strip())
            finally:
                wx.TheClipboard.Close()

    def make_job(self):
        url = self.url.GetValue().strip()
        if not url.lower().startswith(("http://", "https://")):
            wx.MessageBox("Please enter a YouTube video or playlist link.", APP_NAME, wx.ICON_WARNING, self)
            return None
        output_dir = self.folder.GetPath().strip()
        if not output_dir:
            wx.MessageBox("Please choose a folder to save into.", APP_NAME, wx.ICON_WARNING, self)
            return None
        request = DownloadRequest(
            url=url,
            output_dir=output_dir,
            bitrate=BITRATES[self.bitrate.GetSelection()],
            whole_playlist=self.whole_playlist.GetValue(),
            embed_tags=self.embed_tags.GetValue(),
        )
        return DownloadJob(request, _DownloadEvents(self))

    def output_folder(self):
        return self.folder.GetPath()

    def form_controls(self):
        return [self.url, self.folder, self.bitrate, self.whole_playlist, self.embed_tags]

    def save_settings(self):
        self.config.Write("output_dir", self.folder.GetPath())
        self.config.Write("bitrate", BITRATES[self.bitrate.GetSelection()])
        self.config.WriteBool("whole_playlist", self.whole_playlist.GetValue())
        self.config.WriteBool("embed_tags", self.embed_tags.GetValue())
        self.config.WriteBool("auto_transcribe", self.auto_transcribe.GetValue())
        self.config.Flush()

    def summary(self, done, failed):
        text = f"Done. {done} MP3 file(s) saved"
        return text + (f", {failed} failed (see log)." if failed else ".")


class _AudioDropTarget(wx.FileDropTarget):
    def __init__(self, panel: "TranscribePanel"):
        super().__init__()
        self._panel = panel

    def OnDropFiles(self, x, y, filenames):
        if self._panel.busy:
            return False
        self._panel.add_paths(filenames)
        return True


class TranscribePanel(_JobPanel):
    action_label = "Start transcribing"
    activity = "transcribing"
    item_word = "document"
    SPEEDS = ("Accurate (slower)", "Faster (slightly less accurate)")

    def build_form(self):
        # File list with its buttons beside it.
        files_row = wx.BoxSizer(wx.HORIZONTAL)
        self.files = wx.ListBox(self, style=wx.LB_EXTENDED | wx.LB_HSCROLL, size=(-1, 110))
        self.files.SetDropTarget(_AudioDropTarget(self))
        file_buttons = wx.BoxSizer(wx.VERTICAL)
        add_files = wx.Button(self, label="Add files...")
        add_folder = wx.Button(self, label="Add folder...")
        remove = wx.Button(self, label="Remove")
        clear = wx.Button(self, label="Clear")
        for b in (add_files, add_folder, remove, clear):
            file_buttons.Add(b, 0, wx.EXPAND | wx.BOTTOM, 4)
        files_row.Add(self.files, 1, wx.EXPAND | wx.RIGHT, 6)
        files_row.Add(file_buttons, 0)
        self.file_buttons = [add_files, add_folder, remove, clear]
        self.add_row("Audio files:", files_row)

        # Where to save.
        save_row = wx.BoxSizer(wx.HORIZONTAL)
        self.next_to_audio = wx.CheckBox(self, label="Next to each audio file")
        self.next_to_audio.SetValue(self.config.ReadBool("docx_next_to_audio", True))
        default_dir = os.path.join(os.path.expanduser("~"), "Documents", "VaniScribe")
        self.docx_folder = wx.DirPickerCtrl(
            self, path=self.config.Read("docx_dir", default_dir),
            message="Choose where to save Word documents", style=wx.DIRP_USE_TEXTCTRL | wx.DIRP_SMALL,
        )
        save_row.Add(self.next_to_audio, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 12)
        save_row.Add(self.docx_folder, 1, wx.EXPAND)
        self.add_row("Save Word files:", save_row)

        # Language and speed.
        opts_row = wx.BoxSizer(wx.HORIZONTAL)
        self.languages = available_languages()
        self.language = wx.Choice(self, choices=list(self.languages.values()) or ["(no model installed)"])
        self.language.SetSelection(0)
        self.speed = wx.Choice(self, choices=list(self.SPEEDS))
        self.speed.SetSelection(1 if self.config.ReadBool("transcribe_fast", False) else 0)
        opts_row.Add(self.language, 0, wx.RIGHT, 12)
        opts_row.Add(self.speed, 0)
        self.add_row("Language / speed:", opts_row)

        # English translation: off by default.
        self.translate = wx.CheckBox(self, label="Also translate to English (rough machine translation)")
        self.translate.SetValue(self.config.ReadBool("translate_en", False))
        if translation_available():
            self.translate.SetToolTip(
                "Adds an English version under each Telugu paragraph. It conveys the gist, not an exact "
                "translation. Adds a little time on 64-bit Windows and a lot on 32-bit."
            )
        else:
            self.translate.SetValue(False)
            self.translate.Disable()
            self.translate.SetToolTip("The translation model isn't installed in the models folder.")
        self.add_row("Translation:", self.translate)

        note = wx.StaticText(self, label=(
            "Runs offline on this computer. Expect roughly 1 to 2 hours per hour of audio; the computer "
            "is kept awake while it works. If you stop, the next run continues where it left off."
        ))
        note.SetForegroundColour(LOG_COLOURS["info"])
        self.form.Add((0, 0))
        self.form.Add(note, 0, wx.EXPAND)
        note.Wrap(560)

        add_files.Bind(wx.EVT_BUTTON, self.on_add_files)
        add_folder.Bind(wx.EVT_BUTTON, self.on_add_folder)
        remove.Bind(wx.EVT_BUTTON, self.on_remove)
        clear.Bind(wx.EVT_BUTTON, lambda _e: self.files.Clear())
        self.next_to_audio.Bind(wx.EVT_CHECKBOX, lambda _e: self._sync_folder_state())
        self._sync_folder_state()

        missing = missing_components()
        if missing:
            wx.CallAfter(self._report_missing, missing)

    def _report_missing(self, missing):
        self.start_btn.Disable()
        self.status.SetLabel("Transcription is not available in this copy of VaniScribe.")
        self.append_log("Missing: " + ", ".join(missing) + ".", "error")
        self.append_log("Place the model file in a 'models' folder next to VaniScribe, then restart.", "info")

    def _sync_folder_state(self):
        self.docx_folder.Enable(not self.next_to_audio.GetValue() and not self.busy)

    # ----- file list -----

    def add_paths(self, paths):
        existing = set(self.files.GetStrings())
        added = 0
        for path in paths:
            candidates = []
            if os.path.isdir(path):
                for name in sorted(os.listdir(path)):
                    candidates.append(os.path.join(path, name))
            else:
                candidates.append(path)
            for c in candidates:
                if os.path.isfile(c) and c.lower().endswith(AUDIO_EXTENSIONS) and c not in existing:
                    self.files.Append(c)
                    existing.add(c)
                    added += 1
        if not added:
            self.append_log("No new audio files found there.", "warning")
        return added

    def receive_downloads(self, paths: list[str], start: bool) -> None:
        """MP3s from the Download tab. With start, transcribe exactly these files right away."""
        if start and not self.busy:
            self.files.Clear()
        added = self.add_paths(paths)
        if added:
            self.append_log(f"Added {added} downloaded MP3 file(s).", "success")
        if not start:
            return
        if self.busy:
            self.append_log("A transcription is already running; press Start transcribing for the new "
                            "files when it finishes.", "warning")
        else:
            self.on_start(None)

    def on_add_files(self, _event):
        wildcard = "Audio files|" + ";".join("*" + e for e in AUDIO_EXTENSIONS) + "|All files|*.*"
        with wx.FileDialog(self, "Choose audio files", wildcard=wildcard,
                           style=wx.FD_OPEN | wx.FD_MULTIPLE | wx.FD_FILE_MUST_EXIST) as dlg:
            if dlg.ShowModal() == wx.ID_OK:
                self.add_paths(dlg.GetPaths())

    def on_add_folder(self, _event):
        with wx.DirDialog(self, "Choose a folder of audio files", style=wx.DD_DIR_MUST_EXIST) as dlg:
            if dlg.ShowModal() == wx.ID_OK:
                self.add_paths([dlg.GetPath()])

    def on_remove(self, _event):
        for index in sorted(self.files.GetSelections(), reverse=True):
            self.files.Delete(index)

    # ----- job -----

    def make_job(self):
        files = list(self.files.GetStrings())
        if not files:
            wx.MessageBox("Add one or more audio files first.", APP_NAME, wx.ICON_WARNING, self)
            return None
        if not self.languages:
            return None
        output_dir = None
        if not self.next_to_audio.GetValue():
            output_dir = self.docx_folder.GetPath().strip()
            if not output_dir:
                wx.MessageBox("Please choose a folder for the Word files.", APP_NAME, wx.ICON_WARNING, self)
                return None
        language = list(self.languages)[self.language.GetSelection()]
        request = TranscribeRequest(files, output_dir, language, fast=self.speed.GetSelection() == 1,
                                    translate=self.translate.IsEnabled() and self.translate.GetValue())
        return TranscribeJob(request, _TranscribeEvents(self))

    def on_start(self, event):
        super().on_start(event)
        if self.busy:
            _keep_awake(True)

    def on_finished(self, done, failed, cancelled, error):
        _keep_awake(False)
        super().on_finished(done, failed, cancelled, error)

    def on_file_finished(self, path: str, document: str):
        self.append_log(f"Saved {document}", "success")
        self.last_document = document

    def output_folder(self):
        if self.next_to_audio.GetValue():
            last = getattr(self, "last_document", None)
            if last:
                return os.path.dirname(last)
            first = self.files.GetString(0) if self.files.GetCount() else ""
            return os.path.dirname(first)
        return self.docx_folder.GetPath()

    def form_controls(self):
        return [self.files, *self.file_buttons, self.next_to_audio, self.docx_folder, self.language, self.speed]

    def set_busy(self, busy):
        super().set_busy(busy)
        self._sync_folder_state()
        self.translate.Enable(not busy and translation_available())

    def save_settings(self):
        self.config.WriteBool("docx_next_to_audio", self.next_to_audio.GetValue())
        self.config.Write("docx_dir", self.docx_folder.GetPath())
        self.config.WriteBool("transcribe_fast", self.speed.GetSelection() == 1)
        self.config.WriteBool("translate_en", self.translate.GetValue())
        self.config.Flush()

    def summary(self, done, failed):
        text = f"Done. {done} Word document(s) saved"
        return text + (f", {failed} failed (see log)." if failed else ".")


OLD_APP_NAME = "ForVani"  # the app's name before 1.1; its saved settings are carried over once
_SETTINGS = {
    "output_dir": "str", "bitrate": "str", "docx_dir": "str", "tab": "int",
    "whole_playlist": "bool", "embed_tags": "bool", "docx_next_to_audio": "bool",
    "transcribe_fast": "bool", "translate_en": "bool",
}


def _migrate_settings(config: wx.ConfigBase) -> None:
    """Copy settings saved under the old app name the first time the renamed app starts."""
    if config.GetNumberOfEntries() or config.ReadBool("migrated", False):
        return
    old = wx.Config(OLD_APP_NAME)
    for key, kind in _SETTINGS.items():
        if not old.HasEntry(key):
            continue
        if kind == "bool":
            config.WriteBool(key, old.ReadBool(key, False))
        elif kind == "int":
            config.WriteInt(key, old.ReadInt(key, 0))
        else:
            config.Write(key, old.Read(key, ""))
    config.WriteBool("migrated", True)
    config.Flush()


class MainFrame(wx.Frame):
    def __init__(self):
        super().__init__(None, title=f"{APP_NAME} {__version__}", size=(820, 640))
        self.SetMinSize((620, 520))
        self.config = wx.Config(APP_NAME)
        _migrate_settings(self.config)

        self.notebook = wx.Notebook(self)
        self.download = DownloadPanel(self.notebook, self.config)
        self.transcribe = TranscribePanel(self.notebook, self.config)
        self.notebook.AddPage(self.download, "Download MP3")
        self.notebook.AddPage(self.transcribe, "Transcribe to Word")
        self.notebook.SetSelection(min(self.config.ReadInt("tab", 0), 1))
        self.download.send_to_transcribe = self.send_to_transcribe

        self.Bind(wx.EVT_CLOSE, self.on_close)
        self.Centre()

    def send_to_transcribe(self, paths: list[str], start: bool) -> None:
        self.notebook.SetSelection(1)
        self.transcribe.receive_downloads(paths, start)

    def on_close(self, event):
        running = [p for p in (self.download, self.transcribe) if p.busy]
        if running:
            what = " and ".join(p.activity for p in running)
            answer = wx.MessageBox(
                f"Still working ({what}). Stop and quit?\n\nA stopped transcription continues where it "
                "left off next time.", APP_NAME, wx.YES_NO | wx.ICON_QUESTION, self,
            )
            if answer != wx.YES:
                event.Veto()
                return
            for panel in running:
                panel.job.cancel()
            _keep_awake(False)
        for panel in (self.download, self.transcribe):
            panel.save_settings()
        self.config.WriteInt("tab", self.notebook.GetSelection())
        self.config.Flush()
        event.Skip()


def main() -> None:
    app = wx.App()
    MainFrame().Show()
    app.MainLoop()
