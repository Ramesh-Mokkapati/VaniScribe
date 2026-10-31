#!/usr/bin/env bash
# Builds dist/VaniScribe.app and dist/VaniScribe-macos-<arch>.zip on macOS 11+ (Apple Silicon or Intel).
#
#   bash build_macos.sh                    # uses python3 from PATH
#   PYTHON=python3.13 bash build_macos.sh
#   bash build_macos.sh --prepare-models   # also build models/ here (~3.5 GB download, needs PyTorch)
#   bash build_macos.sh --clean            # delete this architecture's build output and re-download tools
#
# 1. Creates/updates .venv-macos-<arch> and installs requirements (CTranslate2 included).
# 2. Puts the tools into vendor/macos-<arch>/bin:
#    - FFmpeg/FFprobe (pinned release, SHA-256 checked) and Deno, downloaded;
#    - whisper.cpp's whisper-cli, compiled from source (no macOS binaries are published). On Apple
#      Silicon it uses the GPU through Metal.
# 3. Runs PyInstaller with ytmp3.spec, copies models/ (speech + translation, from prepare_model)
#    into VaniScribe.app/Contents/Resources/models so dragging the app to Applications keeps them,
#    re-signs the app and zips it.
# Needs Xcode Command Line Tools and CMake (brew install cmake). The same script is used by the
# GitHub Actions pipeline. It builds for this Mac's own chip.
set -euo pipefail
cd "$(dirname "$0")"

CLEAN=0
PREPARE_MODELS=0
for arg in "$@"; do
  case "$arg" in
    --clean) CLEAN=1 ;;
    --prepare-models) PREPARE_MODELS=1 ;;
    *) echo "Unknown option: $arg" >&2; exit 2 ;;
  esac
done

case "$(uname -m)" in
  arm64)  ARCH=arm64; FFMPEG_BUILD="arm64/1789931890_9.0.2"; DENO_TARGET="aarch64-apple-darwin" ;;
  x86_64) ARCH=x64;   FFMPEG_BUILD="amd64/1789931006_9.0.2"; DENO_TARGET="x86_64-apple-darwin" ;;
  *) echo "Unsupported CPU: $(uname -m)" >&2; exit 1 ;;
esac
KEY="macos-$ARCH"
PYTHON="${PYTHON:-python3}"
VENV=".venv-$KEY"
VENDOR="vendor/$KEY/bin"
FFMPEG_BASE="https://ffmpeg.martin-riedl.de/download/macos/$FFMPEG_BUILD"
DENO_URL="https://github.com/denoland/deno/releases/latest/download/deno-$DENO_TARGET.zip"
WHISPER_CPP_TAG="v1.9.5"   # same version as the Windows builds
WHISPER_SRC="build/whisper.cpp-$WHISPER_CPP_TAG"
WHISPER_BUILD="build/whisper.cpp-$KEY"
SPEECH_MODEL="models/ggml-telugu-small-q8_0.bin"
TRANSLATION_MODEL="models/nllb-200-600m-int8"

if [[ $CLEAN == 1 ]]; then
  rm -rf "build/$KEY" "$WHISPER_BUILD" "dist/VaniScribe" "dist/VaniScribe.app" "dist/VaniScribe-$KEY.zip" "vendor/$KEY"
fi

# --- 1. Python environment -------------------------------------------------
if [[ ! -x "$VENV/bin/python" ]]; then
  echo "Creating $VENV with: $PYTHON"
  "$PYTHON" -m venv "$VENV"
fi
"$VENV/bin/python" -m pip install --upgrade pip
"$VENV/bin/python" -m pip install -r requirements.txt -r requirements-build.txt -r requirements-ctranslate2.txt

# --- 2. Vendored tools -----------------------------------------------------
fetch() {  # url dest: resumable, retries on a flaky connection; a transfer stalled under 10 KB/s for 2 minutes counts as failed
  curl -fsSL --retry 10 --retry-delay 5 --retry-all-errors --connect-timeout 30 \
    --speed-limit 10240 --speed-time 120 -C - -o "$2" "$1"
}
mkdir -p "$VENDOR"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

for tool in ffmpeg ffprobe; do
  if [[ ! -x "$VENDOR/$tool" ]]; then
    echo "Downloading $tool ($ARCH)..."
    fetch "$FFMPEG_BASE/$tool.zip" "$tmp/$tool.zip"
    fetch "$FFMPEG_BASE/$tool.zip.sha256" "$tmp/$tool.zip.sha256"
    (cd "$tmp" && shasum -a 256 -c "$tool.zip.sha256")
    unzip -o -q "$tmp/$tool.zip" -d "$VENDOR"
    chmod +x "$VENDOR/$tool"
  fi
done
if [[ ! -x "$VENDOR/deno" ]]; then
  echo "Downloading Deno ($ARCH)..."
  fetch "$DENO_URL" "$tmp/deno.zip"
  unzip -o -q "$tmp/deno.zip" -d "$VENDOR"
  chmod +x "$VENDOR/deno"
fi

if [[ ! -x "$VENDOR/whisper-cli" ]]; then
  if [[ ! -d "$WHISPER_SRC" ]]; then
    echo "Downloading whisper.cpp $WHISPER_CPP_TAG source..."
    fetch "https://github.com/ggml-org/whisper.cpp/archive/refs/tags/$WHISPER_CPP_TAG.tar.gz" "$tmp/whisper.tar.gz"
    mkdir -p "$WHISPER_SRC"
    tar -xzf "$tmp/whisper.tar.gz" -C "$WHISPER_SRC" --strip-components 1
  fi
  echo "Compiling whisper.cpp ($ARCH)..."
  cpu_flags=(-DGGML_NATIVE=OFF)  # build for any Mac of this architecture, not just this one
  if [[ $ARCH == x64 ]]; then
    # Every Intel Mac that runs macOS 11+ has AVX2/FMA, so enable them explicitly.
    cpu_flags+=(-DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON -DGGML_METAL=OFF)
  fi
  cmake -S "$WHISPER_SRC" -B "$WHISPER_BUILD" -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=OFF -DWHISPER_BUILD_TESTS=OFF -DWHISPER_SDL2=OFF \
    -DCMAKE_OSX_DEPLOYMENT_TARGET=11.0 "${cpu_flags[@]}"
  cmake --build "$WHISPER_BUILD" --config Release -j "$(sysctl -n hw.ncpu)" --target whisper-cli whisper-quantize
  cp "$WHISPER_BUILD/bin/whisper-cli" "$VENDOR/"
  chmod +x "$VENDOR/whisper-cli"
fi
for tool in ffmpeg ffprobe deno whisper-cli; do
  [[ -x "$VENDOR/$tool" ]] || { echo "Missing $VENDOR/$tool" >&2; exit 1; }
done

# --- 2b. Models (optional here; the pipeline downloads them from its model job) ----------
if [[ $PREPARE_MODELS == 1 ]]; then
  if [[ ! -x ".venv-model/bin/python" ]]; then
    "$PYTHON" -m venv .venv-model
  fi
  .venv-model/bin/python -m pip install --upgrade pip
  .venv-model/bin/python -m pip install -r requirements-model.txt
  .venv-model/bin/python scripts/prepare_model.py "$WHISPER_BUILD/bin/whisper-quantize"
fi

# --- 3. Build the app ------------------------------------------------------
VANISCRIBE_ARCH="$KEY" "$VENV/bin/python" -m PyInstaller --noconfirm --clean --workpath "build/$KEY" ytmp3.spec

APP="dist/VaniScribe.app"
[[ -d "$APP" ]] || { echo "Build finished but $APP was not produced" >&2; exit 1; }

if [[ -f "$SPEECH_MODEL" ]]; then
  echo "Adding models to the app bundle..."
  mkdir -p "$APP/Contents/Resources/models"
  cp "$SPEECH_MODEL" "$APP/Contents/Resources/models/"
  [[ -f models/MODEL-LICENSE.txt ]] && cp models/MODEL-LICENSE.txt "$APP/Contents/Resources/models/"
  if [[ -f "$TRANSLATION_MODEL/model.bin" ]]; then
    cp -R "$TRANSLATION_MODEL" "$APP/Contents/Resources/models/"
  else
    echo "  $TRANSLATION_MODEL not found: packaging without English translation." >&2
  fi
  # Adding files invalidates PyInstaller's ad-hoc signature; Apple Silicon refuses to run unsigned code.
  codesign --force --deep --sign - "$APP"
else
  echo "$SPEECH_MODEL not found: packaging without transcription (run with --prepare-models)." >&2
fi
codesign --verify --deep "$APP"
# ditto keeps the bundle's symlinks and permissions, which a plain zip can break.
ditto -c -k --keepParent "$APP" "dist/VaniScribe-$KEY.zip"
echo "Built $APP and dist/VaniScribe-$KEY.zip ($(du -sh "dist/VaniScribe-$KEY.zip" | cut -f1))"
