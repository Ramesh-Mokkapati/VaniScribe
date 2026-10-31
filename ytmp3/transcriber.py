"""Offline speech-to-text for audio files, using whisper.cpp (whisper-cli) and a bundled model.

How a file is processed:
  1. FFmpeg decodes it to 16 kHz mono PCM, streamed in blocks so a two-hour file never sits in
     memory (this also runs in the 32-bit build).
  2. The audio is cut at pauses into chunks of at most CHUNK_MAX_S seconds. Telugu script costs
     about 15 tokens per word and Whisper stops after ~224 tokens per window, so a normal
     30-second window silently loses most of the speech; short chunks avoid that.
  3. Chunks are written as WAV files and transcribed by whisper-cli a batch at a time (one model
     load per batch), which gives regular progress updates and a quick cancel.
  4. Results are saved after every batch, so a cancelled or interrupted file resumes where it
     stopped.
  5. Optionally, each paragraph is translated to English with NLLB-200 (see nllb.py), after
     replacing known names and Sanskrit terms with their English spelling (glossary.py).

A TranscribeJob runs in a background thread and reports through a TranscribeListener; listener
methods are called from the worker thread, so GUI code must marshal them onto the UI thread.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import wave
from dataclasses import asdict, dataclass

import numpy as np

from .tools import find_model, find_tool, short_path

SAMPLE_RATE = 16000
FRAME = 320                 # 20 ms analysis frames for pause detection
CHUNK_MAX_S = 6.0
CHUNK_MIN_S = 2.0
BLOCK_S = 60                # audio decoded per block
BATCH_AUDIO_S = 60          # audio per whisper-cli run: progress/resume granularity (model reload ~1 s)
FAST_AUDIO_CTX = 384        # encoder context for "Faster" mode: covers ~7.7 s instead of 30 s

# Language code -> (display name, model file). Add entries here as models for more languages ship.
MODELS = {
    "te": ("Telugu", "ggml-telugu-small-q8_0.bin"),
}
TRANSLATION_MODEL_DIR = "nllb-200-600m-int8"   # folder in models/ (CTranslate2 format)
TRANSLATE_CHUNKS = 3                           # speech chunks (~6 s each) translated together

_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0  # CREATE_NO_WINDOW: no console flashes


@dataclass
class Segment:
    start: float
    end: float
    text: str


@dataclass
class TranscribeRequest:
    files: list[str]
    output_dir: str | None = None   # None: save each document next to its audio file
    language: str = "te"
    fast: bool = False
    translate: bool = False         # add a rough English translation under each paragraph


@dataclass
class Transcript:
    source: str
    language: str
    model: str
    duration: float
    segments: list[Segment]


class TranscribeListener:
    """Receives job events. Override the methods you care about."""

    def log(self, message: str, level: str = "info") -> None:
        pass

    def file_started(self, index: int, total: int, path: str) -> None:
        pass

    def progress(self, file_fraction: float, overall_fraction: float, detail: str) -> None:
        pass

    def file_finished(self, path: str, document: str) -> None:
        pass

    def finished(self, done: int, failed: int, cancelled: bool, error: str | None) -> None:
        pass


class Cancelled(Exception):
    pass


def translation_model_dir() -> str | None:
    path = find_model(os.path.join(TRANSLATION_MODEL_DIR, "model.bin"))
    return os.path.dirname(path) if path else None


def translation_available() -> bool:
    return translation_model_dir() is not None


def available_languages() -> dict[str, str]:
    """Languages whose model is installed: code -> display name."""
    return {code: name for code, (name, model) in MODELS.items() if find_model(model)}


def missing_components(language: str = "te") -> list[str]:
    """Human-readable list of what is needed but not found (empty when transcription can run)."""
    missing = []
    if not find_tool("ffmpeg"):
        missing.append("FFmpeg")
    if not find_tool("whisper-cli"):
        missing.append("the whisper.cpp engine (whisper-cli)")
    name, model = MODELS.get(language, (language, None))
    if not model or not find_model(model):
        missing.append(f"the {name} speech model ({model})")
    return missing


def probe_duration(path: str) -> float:
    ffprobe = find_tool("ffprobe")
    if not ffprobe:
        return 0.0
    out = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", path],
        capture_output=True, text=True, creationflags=_NO_WINDOW,
    )
    try:
        return float(out.stdout.strip())
    except ValueError:
        return 0.0


def _cut_chunks(samples: np.ndarray, final: bool) -> tuple[list[tuple[int, int]], int]:
    """Split int16 audio at pauses into chunks of at most CHUNK_MAX_S seconds.

    Returns (speech_chunks, consumed): chunk (start, end) sample offsets that contain speech, and how
    many samples were fully handled. Unless final, the unfinished tail is left for the next block.
    """
    n = len(samples) // FRAME
    if n == 0:
        return [], len(samples) if final else 0
    frames = samples[: n * FRAME].astype(np.float32).reshape(n, FRAME)
    db = 20 * np.log10(np.sqrt((frames ** 2).mean(axis=1)) + 1e-3)
    silence = np.percentile(db, 10) + 8  # a little above this stretch's background noise
    per_s = SAMPLE_RATE // FRAME
    max_f, min_f = int(CHUNK_MAX_S * per_s), int(CHUNK_MIN_S * per_s)

    chunks, start = [], 0
    while start < n:
        hard_end = start + max_f
        if hard_end >= n:
            if not final:
                break
            cut = n
        else:
            window = db[start + min_f:hard_end]
            quiet = np.nonzero(window < silence)[0]
            # Prefer the last quiet frame before the limit; otherwise the quietest moment.
            cut = start + min_f + (int(quiet[-1]) if len(quiet) else int(np.argmin(window)))
        if db[start:cut].max() > silence:
            chunks.append((start * FRAME, cut * FRAME))
        start = cut
    consumed = len(samples) if final else start * FRAME
    return chunks, consumed


class TranscribeJob(threading.Thread):
    def __init__(self, request: TranscribeRequest, listener: TranscribeListener):
        super().__init__(daemon=True)
        self.request = request
        self.listener = listener
        self._cancel = threading.Event()
        self._proc: subprocess.Popen | None = None
        self._translator = None
        # Share of each file's progress bar given to transcription; the rest is translation, which
        # is quick with CTranslate2 (64-bit) and comparatively slow with the numpy engine (32-bit).
        if not request.translate:
            self._transcribe_share = 1.0
        else:
            self._transcribe_share = 0.95 if importlib.util.find_spec("ctranslate2") else 0.75

    def cancel(self) -> None:
        self._cancel.set()
        proc = self._proc
        if proc and proc.poll() is None:
            proc.kill()

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def _check_cancel(self) -> None:
        if self.cancelled:
            raise Cancelled()

    def run(self) -> None:
        from .docx_writer import write_document  # python-docx is only needed once a job runs

        done = failed = 0
        error = None
        files = self.request.files
        try:
            missing = missing_components(self.request.language)
            if missing:
                raise RuntimeError("Transcription is not available: missing " + ", ".join(missing) + ".")
            if self.request.translate and not translation_available():
                raise RuntimeError("English translation is not available: the translation model "
                                   f"({TRANSLATION_MODEL_DIR}) is missing from the models folder.")
            for index, path in enumerate(files, 1):
                self._check_cancel()
                self.listener.file_started(index, len(files), path)
                try:
                    transcript = self._transcribe_file(path, index - 1, len(files))
                    english = None
                    if self.request.translate and transcript.segments:
                        english = self._translate(transcript, index - 1, len(files))
                    document = self._output_path(path)
                    write_document(transcript, document, english)
                    _remove(self._partial_path(path))
                    done += 1
                    self.listener.file_finished(path, document)
                except Cancelled:
                    raise
                except Exception as exc:  # keep going with the remaining files
                    failed += 1
                    self.listener.log(f"{os.path.basename(path)}: {exc}", "error")
        except Cancelled:
            pass
        except Exception as exc:
            error = str(exc)
        self.listener.finished(done, failed, self.cancelled, error)

    # ----- paths -----

    def _output_path(self, audio_path: str) -> str:
        folder = self.request.output_dir or os.path.dirname(os.path.abspath(audio_path))
        os.makedirs(folder, exist_ok=True)
        base = os.path.splitext(os.path.basename(audio_path))[0]
        return os.path.join(folder, base + ".docx")

    def _partial_path(self, audio_path: str) -> str:
        return os.path.splitext(self._output_path(audio_path))[0] + ".partial.json"

    # ----- one file -----

    def _transcribe_file(self, path: str, file_index: int, file_count: int) -> Transcript:
        req = self.request
        name, model_file = MODELS[req.language]
        model = find_model(model_file)
        duration = probe_duration(path)

        # Resume a previous, interrupted run with the same settings.
        settings = {"language": req.language, "model": model_file, "fast": req.fast}
        partial = self._partial_path(path)
        segments: list[Segment] = []
        resume_at = 0.0
        saved = _load_json(partial)
        if saved and saved.get("settings") == settings:
            segments = [Segment(**s) for s in saved["segments"]]
            resume_at = float(saved["processed_until"])
            self.listener.log(f"Resuming {os.path.basename(path)} at {_clock(resume_at)}")
        else:
            self.listener.log(f"Transcribing {os.path.basename(path)} ({_clock(duration)}, {name})")

        started = time.time()
        processed_at_start = resume_at

        def report(processed: float) -> None:
            frac = min(processed / duration, 1.0) if duration else 0.0
            elapsed = time.time() - started
            done_now = processed - processed_at_start
            detail = f"{_clock(processed)} of {_clock(duration)}"
            if done_now > 30 and duration and processed < duration:
                remaining = elapsed / done_now * (duration - processed)
                detail += f", about {_eta(remaining)} left"
            frac *= self._transcribe_share
            self.listener.progress(frac, (file_index + frac) / file_count, detail)

        report(resume_at)
        with tempfile.TemporaryDirectory(prefix="vaniscribe-") as tmp:
            batch: list[tuple[float, float, np.ndarray]] = []
            batch_audio = 0.0
            for chunk_start, chunk in self._chunks(path, resume_at):
                self._check_cancel()
                seconds = len(chunk) / SAMPLE_RATE
                batch.append((chunk_start, chunk_start + seconds, chunk))
                batch_audio += seconds
                if batch_audio >= BATCH_AUDIO_S:
                    segments += self._run_batch(batch, model, tmp)
                    resume_at = batch[-1][1]
                    _save_json(partial, {"settings": settings, "processed_until": resume_at,
                                         "segments": [asdict(s) for s in segments]})
                    report(resume_at)
                    batch, batch_audio = [], 0.0
            if batch:
                segments += self._run_batch(batch, model, tmp)
            # Saved even when complete, so a cancel during translation doesn't redo the transcription.
            _save_json(partial, {"settings": settings, "processed_until": duration or resume_at,
                                 "segments": [asdict(s) for s in segments]})
            report(duration)

        return Transcript(path, req.language, model_file, duration, segments)

    def _translate(self, transcript: Transcript, file_index: int, file_count: int) -> list[str]:
        """English for each paragraph of the document (same grouping as the Word writer)."""
        from .docx_writer import group_segments
        from .glossary import Glossary
        from .nllb import NllbTranslator

        if self._translator is None:
            self.listener.log("Loading the English translation model...")
            self._translator = NllbTranslator(translation_model_dir())
        share = self._transcribe_share
        glossary = Glossary(transcript.language)

        # Translate a few speech chunks (~25 words) at a time: NLLB works sentence by sentence and
        # loses track of long passages. Each paragraph's pieces are joined back together.
        pieces, owner = [], []
        for p, paragraph in enumerate(group_segments(transcript.segments)):
            for i in range(0, len(paragraph), TRANSLATE_CHUNKS):
                pieces.append(glossary.protect(" ".join(s.text for s in paragraph[i:i + TRANSLATE_CHUNKS])))
                owner.append(p)
        self.listener.log(f"Translating to English ({len(pieces)} passages, {self._translator.engine_name})")
        started = time.time()

        def progress(frac: float) -> None:
            self._check_cancel()
            detail = f"Translating to English: {int(frac * 100)}%"
            elapsed = time.time() - started
            if 0.05 < frac < 1 and elapsed > 20:
                detail += f", about {_eta(elapsed / frac * (1 - frac))} left"
            file_frac = share + (1 - share) * frac
            self.listener.progress(file_frac, (file_index + file_frac) / file_count, detail)

        progress(0.0)
        english = self._translator.translate(pieces, transcript.language, progress)
        paragraphs: list[list[str]] = [[] for _ in range(owner[-1] + 1)] if owner else []
        for p, text in zip(owner, english):
            if text:
                paragraphs[p].append(text)
        return [" ".join(parts) for parts in paragraphs]

    def _chunks(self, path: str, start_at: float):
        """Yield (start_seconds, int16 samples) speech chunks, decoding the file block by block."""
        ffmpeg = find_tool("ffmpeg")
        cmd = [ffmpeg, "-nostdin", "-v", "error"]
        if start_at > 0:
            cmd += ["-ss", f"{start_at:.3f}"]
        cmd += ["-i", path, "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "s16le", "-"]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=_NO_WINDOW)
        try:
            pending = np.zeros(0, dtype=np.int16)
            offset = 0  # samples (relative to start_at) before `pending`
            block_bytes = BLOCK_S * SAMPLE_RATE * 2
            while True:
                self._check_cancel()
                data = proc.stdout.read(block_bytes)
                final = len(data) < block_bytes
                if data:
                    pending = np.concatenate([pending, np.frombuffer(data[: len(data) // 2 * 2], dtype=np.int16)])
                chunks, consumed = _cut_chunks(pending, final)
                for a, b in chunks:
                    yield start_at + (offset + a) / SAMPLE_RATE, pending[a:b].copy()
                pending = pending[consumed:]
                offset += consumed
                if final:
                    break
            proc.wait()
            if proc.returncode not in (0, None) and not self.cancelled:
                err = proc.stderr.read().decode("utf-8", "replace").strip()
                raise RuntimeError(f"could not read the audio: {err or 'FFmpeg failed'}")
        finally:
            if proc.poll() is None:
                proc.kill()

    def _run_batch(self, batch, model: str, tmp: str) -> list[Segment]:
        files = []
        for i, (_, _, samples) in enumerate(batch):
            wav_path = os.path.join(tmp, f"c{i:04d}.wav")
            with wave.open(wav_path, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(SAMPLE_RATE)
                w.writeframes(samples.tobytes())
            files.append(wav_path)

        threads = str(max(1, min(8, (os.cpu_count() or 2))))
        cmd = [find_tool("whisper-cli"), "-m", short_path(model), "-l", self.request.language,
               "-t", threads, "-bs", "1", "-bo", "1", "-nt", "-np", "-otxt"]
        if self.request.fast:
            cmd += ["-ac", str(FAST_AUDIO_CTX)]
        for f in files:
            cmd += ["-f", short_path(f)]

        self._proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                      creationflags=_NO_WINDOW)
        _, err = self._proc.communicate()
        code, self._proc = self._proc.returncode, None
        self._check_cancel()
        if code:
            tail = err.decode("utf-8", "replace").strip().splitlines()[-3:]
            raise RuntimeError("speech engine failed: " + " | ".join(tail))

        segments = []
        for (start, end, _), wav_path in zip(batch, files):
            # A multi-byte Telugu character can be cut off at the token limit; drop the broken bytes.
            try:
                with open(wav_path + ".txt", encoding="utf-8", errors="replace") as fh:
                    text = " ".join(fh.read().replace("�", "").split())
            except FileNotFoundError:
                text = ""
            if text:
                segments.append(Segment(round(start, 2), round(end, 2), text))
            _remove(wav_path)
            _remove(wav_path + ".txt")
        return segments


def _clock(seconds: float) -> str:
    seconds = int(seconds)
    h, rest = divmod(seconds, 3600)
    m, s = divmod(rest, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _eta(seconds: float) -> str:
    minutes = int(seconds // 60)
    if minutes >= 60:
        return f"{minutes // 60} h {minutes % 60} min"
    return f"{max(minutes, 1)} min"


def _load_json(path: str):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _save_json(path: str, data) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False)
    os.replace(tmp, path)


def _remove(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass

