"""Build the models VaniScribe ships with, in models/.

1. ggml-telugu-small-q8_0.bin (speech to text). Source: vasista22/whisper-telugu-small
   (Apache-2.0), a Whisper-small fine-tuned on Telugu by the SPRING Lab, IIT Madras. Converted to
   whisper.cpp's GGML format and quantised to 8 bits (q8_0), which tested as accurate as the
   original while 4x smaller than full precision.
2. nllb-200-600m-int8/ (Telugu -> English translation). Source: facebook/nllb-200-distilled-600M
   (CC BY-NC 4.0: non-commercial use). Converted to CTranslate2 with 8-bit weights; the 64-bit app
   runs it with CTranslate2, the 32-bit app with VaniScribe's own numpy engine (ytmp3/nllb.py).

Needs a 64-bit Python with requirements-model.txt installed (PyTorch, transformers, CTranslate2);
run it through prepare_model.ps1. Downloads go to build/model and can be deleted after.

usage: python scripts/prepare_model.py <whisper-quantize executable>
"""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK = os.path.join(ROOT, "build", "model")
OUT_DIR = os.path.join(ROOT, "models")
OUT_NAME = "ggml-telugu-small-q8_0.bin"

HF_REPO = "vasista22/whisper-telugu-small"
HF_FILES = [
    "config.json", "generation_config.json", "preprocessor_config.json", "tokenizer_config.json",
    "special_tokens_map.json", "added_tokens.json", "normalizer.json", "vocab.json", "merges.txt",
    "pytorch_model.bin",
]
NLLB_REPO = "facebook/nllb-200-distilled-600M"
NLLB_FILES = [
    "config.json", "generation_config.json", "special_tokens_map.json", "tokenizer_config.json",
    "tokenizer.json", "sentencepiece.bpe.model", "pytorch_model.bin",
]
NLLB_OUT = "nllb-200-600m-int8"
WHISPER_CPP_TAG = "v1.9.5"
CONVERTER_URL = f"https://raw.githubusercontent.com/ggml-org/whisper.cpp/{WHISPER_CPP_TAG}/models/convert-h5-to-ggml.py"
MEL_FILTERS_URL = "https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/mel_filters.npz"


def download(url: str, dest: str) -> None:
    """Resumable download with retries; slow or flaky connections are common."""
    if os.path.isfile(dest):
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    part = dest + ".part"
    for attempt in range(1, 11):
        have = os.path.getsize(part) if os.path.exists(part) else 0
        req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
        try:
            with urllib.request.urlopen(req, timeout=60) as resp, open(part, "ab" if have else "wb") as fh:
                if have and resp.status != 206:  # server ignored the range: start over
                    fh.truncate(0)
                shutil.copyfileobj(resp, fh, 1024 * 1024)
            os.replace(part, dest)
            print(f"  downloaded {os.path.basename(dest)} ({os.path.getsize(dest) / 1e6:.0f} MB)", flush=True)
            return
        except Exception as exc:
            print(f"  retry {attempt} for {os.path.basename(dest)}: {exc}", flush=True)
            time.sleep(min(5 * attempt, 30))
    raise SystemExit(f"could not download {url}")


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def build_speech_model(quantize: str) -> None:
    out = os.path.join(OUT_DIR, OUT_NAME)
    if os.path.isfile(out):
        print(f"{out} already exists; delete it to rebuild.")
        return

    print(f"Downloading {HF_REPO}...")
    src = os.path.join(WORK, "hf")
    for name in HF_FILES:
        download(f"https://huggingface.co/{HF_REPO}/resolve/main/{name}", os.path.join(src, name))
    download(CONVERTER_URL, os.path.join(WORK, "convert-h5-to-ggml.py"))
    download(MEL_FILTERS_URL, os.path.join(WORK, "openai", "whisper", "assets", "mel_filters.npz"))

    print("Converting to GGML...")
    f16 = os.path.join(WORK, "ggml-model.bin")
    if not os.path.isfile(f16):
        subprocess.run(
            [sys.executable, os.path.join(WORK, "convert-h5-to-ggml.py"), src, os.path.join(WORK, "openai"), WORK],
            check=True, stdout=subprocess.DEVNULL,
        )

    print("Quantising to q8_0...")
    os.makedirs(OUT_DIR, exist_ok=True)
    subprocess.run([quantize, f16, out + ".tmp", "q8_0"], check=True, stdout=subprocess.DEVNULL)
    os.replace(out + ".tmp", out)

    with open(out + ".sha256", "w", encoding="ascii") as fh:
        fh.write(f"{sha256(out)}  {OUT_NAME}\n")
    with open(os.path.join(OUT_DIR, "MODEL-LICENSE.txt"), "w", encoding="utf-8") as fh:
        fh.write(
            f"{OUT_NAME} is converted from {HF_REPO} (https://huggingface.co/{HF_REPO}),\n"
            "a Whisper-small model fine-tuned on Telugu by the SPRING Lab, IIT Madras,\n"
            "which is based on OpenAI Whisper (MIT). Licensed under the Apache License 2.0:\n"
            "https://www.apache.org/licenses/LICENSE-2.0\n"
            "Converted to GGML and quantised to 8 bits (q8_0) for whisper.cpp.\n"
        )
    print(f"Built {out} ({os.path.getsize(out) / 1e6:.0f} MB)")


def build_translation_model() -> None:
    out = os.path.join(OUT_DIR, NLLB_OUT)
    if os.path.isfile(os.path.join(out, "model.bin")):
        print(f"{out} already exists; delete it to rebuild.")
        return

    print(f"Downloading {NLLB_REPO}...")
    src = os.path.join(WORK, "nllb")
    for name in NLLB_FILES:
        download(f"https://huggingface.co/{NLLB_REPO}/resolve/main/{name}", os.path.join(src, name))

    print("Converting to CTranslate2 (int8)...")
    from ctranslate2.converters import TransformersConverter

    tmp = out + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    # tokenizer.json is what VaniScribe's own (pure Python) tokenizer reads.
    TransformersConverter(src, copy_files=["tokenizer.json"]).convert(tmp, quantization="int8", force=True)
    with open(os.path.join(tmp, "LICENSE.txt"), "w", encoding="utf-8") as fh:
        fh.write(
            f"Converted from {NLLB_REPO} (https://huggingface.co/{NLLB_REPO}), Meta AI's\n"
            "No Language Left Behind model, to CTranslate2 format with 8-bit weights.\n"
            "Licensed under Creative Commons Attribution-NonCommercial 4.0 (CC BY-NC 4.0):\n"
            "https://creativecommons.org/licenses/by-nc/4.0/ -- personal, non-commercial use only.\n"
        )
    shutil.rmtree(out, ignore_errors=True)
    os.replace(tmp, out)
    size = sum(os.path.getsize(os.path.join(out, f)) for f in os.listdir(out))
    print(f"Built {out} ({size / 1e6:.0f} MB)")


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    build_speech_model(sys.argv[1])
    build_translation_model()


if __name__ == "__main__":
    main()
