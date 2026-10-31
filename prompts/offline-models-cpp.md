# Offline AI models in a C++ application: prompt

Two versions of the same request:
- **Short version:** paste it as-is after replacing the `<...>` placeholders. The model works out the details by inspecting your codebase.
- **Detailed version:** use it when you want every requirement spelled out.

## Short version

> Add offline AI capabilities to this <C++ / wxWidgets / Qt> application as a reusable module that is free and fully offline at runtime, targeting <CPU-only Windows laptops with 8 GB RAM>.
>
> 1. Perform <TASK, e.g. speech-to-text> on <INPUT, e.g. MP3 files> to produce <OUTPUT, e.g. a Word document in the spoken language>. Run local quantised models behind a swappable engine interface, on a worker thread with progress and cancel.
> 2. Ship two models next to the exe: a small general model that detects the input's language or type, and the best default model for <PRIMARY LANGUAGE / DOMAIN>. Let users add models at runtime through a wizard that:
>    - detects the input with the bundled model;
>    - recommends models from a versioned online catalog that fit this machine;
>    - downloads them with resume and checksum verification;
>    - tests the model on the sample and sets it as the default for that language.
> 3. Optionally, diagnose problems from the application's log files using a version-matched knowledge base built in CI from the source code, commit history and a curated known-issues list:
>    - match log lines to source code with plain text search first;
>    - use a local LLM only to summarise the evidence it found;
>    - cite every claim and label unverified workarounds;
>    - ask me before shipping anything derived from source code to end users.
>
> Before writing code, inspect the codebase and propose:
> - the architecture;
> - the engines and models, with sizes, licences and hardware needs;
> - an evaluation plan using my samples in <PATH>.
>
> Wait for my approval. Then build in small, tested steps, with a CI pipeline that packages the app and its bundled models. Ask before downloading anything over 500 MB.

---

## Detailed version

I want to add offline AI model support to a C++ desktop application, built as a reusable module I can carry into future apps.

### Context (fill in)
- Application: <name and what it does>
- Task the model performs: <e.g. speech-to-text, translation, OCR, summarisation>
- Input -> output: <e.g. MP3/WAV audio -> timestamped text -> Word document>
- Languages: <e.g. Telugu first, more later>
- Platforms: <e.g. Windows 10/11 x64>
- Target hardware: <e.g. CPU-only laptops, 8 GB RAM; use an NVIDIA GPU if present>
- UI framework: <wxWidgets / Qt / MFC / none>
- Build: <e.g. CMake + vcpkg, MSVC 2022>
- Sample inputs for testing: <folder path>, reference outputs if any: <folder path>
- Constraints: free and fully offline at runtime, no paid or cloud APIs; model licences must allow <personal / commercial> use; my connection is slow, so ask before downloading anything over 500 MB.

### What to build

1. **Inference backend**
   - Recommend the engine for my task and justify it (e.g. whisper.cpp for speech, llama.cpp for text LLMs, ONNX Runtime or CTranslate2 for others).
   - Put it behind an interface (load, run with progress callback and cooperative cancel, unload) so engines can be swapped.
   - Use quantised models (int8 / q5) on CPU, use the GPU when available, and always run on a worker thread so the UI never blocks.

2. **Bundled models**
   - Ship two models with the app:
     - a small general (multilingual) model used to detect the input's language or type, and as a fallback;
     - the best default model for my main task and language.
   - Do not rely on a single-language fine-tuned model for detection; fine-tuning usually destroys that ability.
   - Ship models as files next to the exe (zip or installer), never embedded inside the exe. Bundled models are read-only.

3. **Model catalog**
   - A versioned JSON catalog. Each entry has: id, task, display name, languages, engine, format, quantisation, files (url, size, sha256), RAM needed, expected speed, licence, minimum app version, notes.
   - A copy ships with the app. The app refreshes it from <catalog URL> at startup (using ETag) and caches it so it works offline. Unknown fields are ignored.
   - List only models that run without conversion. Document how to convert fine-tuned models (e.g. Hugging Face Whisper -> GGML) and host them on Hugging Face or GitHub Releases (each file under 2 GB).

4. **Runtime model downloads**
   - HTTPS only (WinHTTP or libcurl): resumable via Range requests, progress, cancel, retries with backoff, proxy support, and a disk-space check.
   - Download to a `.part` file, verify the SHA-256, then rename it into place atomically.
   - Install into the per-user data folder (e.g. `%LOCALAPPDATA%\<App>\models`). Keep a registry of installed models with versions, uninstall, and a default model per task and language.

5. **Model wizard and manager**
   - Wizard steps:
     1. Pick a sample input.
     2. Run detection with the bundled model and show the top 3 results with confidence; let me override.
     3. Recommend catalog models filtered by task, language and this machine's RAM/GPU, showing size, estimated time on this PC, licence and installed status.
     4. Download the chosen model.
     5. Run a quick test on the sample and set it as the default.
   - A Manage models screen: list, delete, change defaults.
   - At runtime: detect -> use the default model for that language -> if none, use the bundled fallback and suggest the wizard.

6. **Evaluation tool**
   - A command-line tool that runs candidate models on my samples and reports:
     - speed (processing time vs. audio length);
     - peak RAM;
     - accuracy (WER/CER) when reference outputs exist.
   - Output a markdown table. Use it to choose the bundled default model before it gets shipped.

7. **Build, packaging, CI**
   - CMake with a vcpkg manifest and a GitHub Actions Windows build.
   - Cache models between CI runs; package a zip and an installer (Inno Setup or WiX); keep each artifact under 2 GB.
   - Add a smoke test that loads the bundled models and processes a 5-second sample.

8. **Quality**
   - Unit tests for catalog parsing, hash checks, download resume, and model routing.
   - Clear user-facing errors and a log file.

9. **Optional module: log diagnosis assistant (offline)**

   Goal: the user picks the application's log files and the app reports likely causes and known workarounds, with citations, all offline.

   - **Knowledge base sources.** Workarounds are rarely written in source code, so the code alone is not enough. Use:
     - source code, for log/error message strings and their file and function, error codes and enums, config flags and their meaning;
     - a curated known-issues file (`known_issues.yaml`: symptom, matching log patterns, cause, workaround, affected versions, fixed-in version);
     - git commit messages and release notes;
     - optionally, exported bug-tracker tickets.
   - **KB build pipeline (runs in CI, never on the user's PC).**
     - Parse C++ with tree-sitter or libclang. Chunk by function; extract every logging call's format string -> source location, error codes, and config keys.
     - Embed chunks with a small offline embedding model (e.g. bge-small / nomic-embed, GGUF).
     - Store everything in one versioned SQLite file: FTS5 for keyword search, plus sqlite-vec (or hnswlib) for vectors.
     - Tag the KB with the app version, and publish it as a catalog item so it downloads and updates like a model. Diagnosis must use the KB that matches the version that wrote the logs.
   - **Log analysis (deterministic first, AI last).**
     1. Parse the log format (timestamp, level, thread, component, message).
     2. Group repeated lines into templates (e.g. the Drain algorithm). Extract error codes, stack traces, and the first error before a failure.
     3. Match message templates exactly against the KB's format-string index to find the source location. This needs no AI.
     4. Look up `known_issues.yaml` patterns.
     5. Only then use hybrid retrieval (keyword + vector) to find related code, commits, and issues.
   - **Local LLM (llama.cpp, GGUF, run through the same backend interface).**
     - Recommend models that fit CPU-only laptops (e.g. a 3–4B model at Q4 for speed, a 7B coder model at Q4 for quality) with RAM and tokens/second estimates; downloadable via the wizard.
     - The LLM only summarises retrieved evidence. Every claim must cite a KB entry (known-issue id, file:line, commit).
     - Label suggestions as "Known workaround" (from `known_issues.yaml`) or "Inferred from code, unverified".
     - Say "no known workaround found" rather than guessing.
   - **Privacy / IP.**
     - Ask me whether the assistant runs on end-user machines or only for my support team.
     - For end users, ship a KB that contains extracted messages, explanations and workarounds only, not raw source code.
     - Never send logs anywhere; redact usernames, paths, and IPs before display/export.
   - **Evaluation.**
     - Build a test set from past incidents (log file -> correct cause/workaround).
     - Report retrieval hit rate (correct entry in the top 5), answer accuracy, and time per diagnosis on target hardware.
     - Feed confirmed diagnoses back into `known_issues.yaml`.

### How to work
- Start by proposing the architecture: components, interfaces, folder layout, catalog schema, and the recommended engine and models for my task and languages, with sizes and licences. Wait for my confirmation before writing code.
- Then implement in small steps, building and testing each one.
- Run the evaluation tool on my samples and show me the results before choosing the bundled model.
