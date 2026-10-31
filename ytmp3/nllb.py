"""Offline Telugu -> English translation with Meta's NLLB-200 (distilled 600M), 8-bit.

The model is a CTranslate2 model directory (model.bin, shared_vocabulary.txt, tokenizer.json).
Two engines run it:
  * CTranslate2, when installed (64-bit builds): fast, beam search.
  * A small numpy implementation (everywhere else, including 32-bit Windows, where neither
    CTranslate2, PyTorch, ONNX Runtime nor sentencepiece exist). It memory-maps the same 8-bit
    model.bin, dequantises one layer at a time so the whole model never sits in memory as floats,
    and decodes many sentences together so each layer's dequantisation is shared.
The tokenizer is pure Python for the same reason.
"""
from __future__ import annotations

import json
import mmap
import os
import struct
import sys
import unicodedata
from concurrent.futures import ThreadPoolExecutor

import numpy as np

SOURCE_LANG = {"te": "tel_Telu"}
TARGET_LANG = "eng_Latn"
METASPACE = "▁"


class Tokenizer:
    """NLLB's BPE tokenizer, read from tokenizer.json (pure Python)."""

    def __init__(self, model_dir: str):
        with open(os.path.join(model_dir, "tokenizer.json"), encoding="utf-8") as fh:
            spec = json.load(fh)
        model = spec["model"]
        self.vocab: dict[str, int] = model["vocab"]
        for tok in spec["added_tokens"]:
            self.vocab[tok["content"]] = tok["id"]
        self.ranks = {tuple(m.split(" ", 1)) if isinstance(m, str) else tuple(m): i
                      for i, m in enumerate(model["merges"])}
        self.unk = self.vocab[model.get("unk_token") or "<unk>"]
        self._cache: dict[str, list[str]] = {}

    @staticmethod
    def normalize(text: str) -> str:
        return " ".join(unicodedata.normalize("NFKC", text).split())

    def _bpe(self, word: str) -> list[str]:
        cached = self._cache.get(word)
        if cached is not None:
            return cached
        parts = list(word)
        while len(parts) > 1:
            best, best_rank = None, None
            for i in range(len(parts) - 1):
                rank = self.ranks.get((parts[i], parts[i + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best, best_rank = i, rank
            if best is None:
                break
            parts[best:best + 2] = [parts[best] + parts[best + 1]]
        if len(self._cache) < 200_000:
            self._cache[word] = parts
        return parts

    def pieces(self, text: str) -> list[str]:
        out = []
        for word in self.normalize(text).split(" "):
            if word:
                out += self._bpe(METASPACE + word)
        return out

    def encode(self, text: str) -> list[int]:
        return [self.vocab.get(p, self.unk) for p in self.pieces(text)]

    @staticmethod
    def detokenize(pieces: list[str]) -> str:
        return "".join(pieces).replace(METASPACE, " ").strip()


# ----- CTranslate2 model file -----

_DTYPES = (np.float32, np.int8, np.int16, np.int32, np.float16)


class _ModelFile:
    """Read-only, memory-mapped view of a CTranslate2 model.bin."""

    def __init__(self, path: str):
        self._file = open(path, "rb")
        self._map = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        buf, pos = self._map, 0

        def read(fmt):
            nonlocal pos
            value = struct.unpack_from(fmt, buf, pos)
            pos += struct.calcsize(fmt)
            return value[0]

        def string():
            nonlocal pos
            n = read("H")
            s = bytes(buf[pos:pos + n - 1]).decode("utf-8")
            pos += n
            return s

        version = read("I")
        if version < 5:
            raise ValueError(f"unsupported CTranslate2 model version {version}")
        string()           # spec name
        read("I")          # revision
        self.vars = {}
        for _ in range(read("I")):
            name = string()
            shape = tuple(read("I") for _ in range(read("B")))
            dtype = _DTYPES[read("B")]
            nbytes = read("I")
            count = int(np.prod(shape)) if shape else 1
            arr = np.frombuffer(buf, dtype=dtype, count=count, offset=pos)
            self.vars[name] = arr.reshape(shape) if shape else arr[0]
            pos += nbytes
        for _ in range(read("I")):
            alias, target = string(), string()
            self.vars[alias] = self.vars[target]

    def __getitem__(self, name):
        return self.vars[name]

    def __contains__(self, name):
        return name in self.vars


def _layer_norm(x, gamma, beta, eps=1e-5):
    mean = x.mean(-1, keepdims=True)
    var = ((x - mean) ** 2).mean(-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * gamma + beta


def _softmax(x):
    x = x - x.max(-1, keepdims=True)
    np.exp(x, out=x)
    x /= x.sum(-1, keepdims=True)
    return x


class NumpyNllb:
    """Greedy NLLB decoding in numpy over an 8-bit CTranslate2 model."""

    def __init__(self, model_dir: str, vocabulary: list[str], target_ids: np.ndarray, threads: int = 0):
        self.m = _ModelFile(os.path.join(model_dir, "model.bin"))
        self._buffers: dict[tuple, np.ndarray] = {}
        self._threads = threads or max(1, min(8, os.cpu_count() or 1))
        self._pool = ThreadPoolExecutor(self._threads) if self._threads > 1 else None
        self.heads = int(self.m["encoder/num_heads"])
        self.d = self.m["decoder/layer_norm/gamma"].shape[0]
        self.scale_emb = float(self.m["decoder/scale_embeddings"])
        self.n_enc = sum(1 for k in self.m.vars if k.startswith("encoder/layer_") and k.endswith("ffn/linear_0/weight"))
        self.n_dec = sum(1 for k in self.m.vars if k.startswith("decoder/layer_") and k.endswith("ffn/linear_0/weight"))
        self.pos = self.m["decoder/position_encodings/encodings"]
        self.vocab = vocabulary
        # Output projection restricted to tokens that can appear in English text. It stays 8-bit
        # (~110k x 1024) and is dequantised in slices at each step: as floats it would need ~450 MB.
        self.target_ids = target_ids
        self.out_w = self.m["decoder/embeddings/weight"][target_ids]
        self.out_inv_scale = (1.0 / self.m["decoder/embeddings/weight_scale"][target_ids]).astype(np.float32)

    def _logits(self, h):
        out = np.empty((h.shape[0], len(self.target_ids)), dtype=np.float32)
        for a in range(0, len(self.target_ids), self.PROJ_SLICE):
            b = a + self.PROJ_SLICE
            out[:, a:b] = self._matmul_t(h, self._dequant(self.out_w[a:b])) * self.out_inv_scale[a:b]
        return out

    PROJ_SLICE = 16384

    # --- building blocks ---

    def _dequant(self, w):
        """Copy an int8 weight into a reused float32 buffer of its shape.

        Reusing buffers matters: allocating a fresh 12-48 MB array for every layer at every step
        makes the OS zero new pages each time, which costs more than the conversion itself.
        """
        buf = self._buffers.get(w.shape)
        if buf is None:
            buf = self._buffers[w.shape] = np.empty(w.shape, dtype=np.float32)
        np.copyto(buf, w, casting="unsafe")
        return buf

    def _matmul_t(self, x, w):
        """x @ w.T, with w's rows split across threads.

        32-bit numpy's BLAS runs on one core; numpy releases the GIL inside matmul, so splitting
        the work by hand scales almost linearly with cores.
        """
        if self._pool is None or w.shape[0] < 1024:
            return x @ w.T
        out = np.empty(x.shape[:-1] + (w.shape[0],), dtype=np.float32)
        bounds = np.linspace(0, w.shape[0], self._threads + 1).astype(int)

        def part(i):
            a, b = bounds[i], bounds[i + 1]
            out[..., a:b] = x @ w[a:b].T

        list(self._pool.map(part, range(self._threads)))
        return out

    def _linear(self, x, name):
        y = self._matmul_t(x, self._dequant(self.m[name + "/weight"]))
        y *= 1.0 / self.m[name + "/weight_scale"]
        if name + "/bias" in self.m:
            y += self.m[name + "/bias"]
        return y

    def _ln(self, x, name):
        return _layer_norm(x, self.m[name + "/gamma"], self.m[name + "/beta"])

    def _split(self, x):  # (B, T, D) -> (B, H, T, D/H)
        b, t, _ = x.shape
        return x.reshape(b, t, self.heads, -1).transpose(0, 2, 1, 3)

    def _attend(self, q, k, v, mask):
        scores = (q * (q.shape[-1] ** -0.5)) @ k.transpose(0, 1, 3, 2)
        if mask is not None:
            scores = np.where(mask, scores, np.float32(-1e9))
        out = _softmax(scores) @ v
        b, h, t, dh = out.shape
        return out.transpose(0, 2, 1, 3).reshape(b, t, h * dh)

    def _embed(self, ids, offset=0):
        emb, scale = self.m["decoder/embeddings/weight"], self.m["decoder/embeddings/weight_scale"]
        x = emb[ids].astype(np.float32) / scale[ids][..., None] * self.scale_emb
        return x + self.pos[offset:offset + ids.shape[1]]

    # --- encoder / decoder ---

    def encode(self, ids, mask):
        x = self._embed(ids)
        attn_mask = mask[:, None, None, :]
        for i in range(self.n_enc):
            p = f"encoder/layer_{i}"
            h = self._ln(x, p + "/self_attention/layer_norm")
            q, k, v = np.split(self._linear(h, p + "/self_attention/linear_0"), 3, axis=-1)
            x = x + self._linear(self._attend(self._split(q), self._split(k), self._split(v), attn_mask),
                                 p + "/self_attention/linear_1")
            h = self._ln(x, p + "/ffn/layer_norm")
            x = x + self._linear(np.maximum(self._linear(h, p + "/ffn/linear_0"), 0), p + "/ffn/linear_1")
        return self._ln(x, "encoder/layer_norm")

    def translate(self, batch: list[list[int]], start_ids: list[int], eos_id: int, max_len: int,
                  repetition_penalty: float = 1.2, no_repeat_ngram: int = 4) -> list[list[int]]:
        n = len(batch)
        src_len = max(len(s) for s in batch)
        ids = np.full((n, src_len), 1, dtype=np.int64)  # <pad>
        mask = np.zeros((n, src_len), dtype=bool)
        for i, s in enumerate(batch):
            ids[i, :len(s)] = s
            mask[i, :len(s)] = True
        memory = self.encode(ids, mask)
        cross_mask = mask[:, None, None, :]

        cross_kv = []
        for i in range(self.n_dec):
            k, v = np.split(self._linear(memory, f"decoder/layer_{i}/attention/linear_1"), 2, axis=-1)
            cross_kv.append((self._split(k), self._split(v)))
        self_k = [None] * self.n_dec
        self_v = [None] * self.n_dec

        done = np.zeros(n, dtype=bool)
        step_input = np.full((n, 1), start_ids[0], dtype=np.int64)
        generated = [[] for _ in range(n)]
        pos = 0
        while True:
            x = self._embed(step_input, pos)
            for i in range(self.n_dec):
                p = f"decoder/layer_{i}"
                h = self._ln(x, p + "/self_attention/layer_norm")
                q, k, v = np.split(self._linear(h, p + "/self_attention/linear_0"), 3, axis=-1)
                k, v = self._split(k), self._split(v)
                self_k[i] = k if self_k[i] is None else np.concatenate([self_k[i], k], axis=2)
                self_v[i] = v if self_v[i] is None else np.concatenate([self_v[i], v], axis=2)
                x = x + self._linear(self._attend(self._split(q), self_k[i], self_v[i], None),
                                     p + "/self_attention/linear_1")
                h = self._ln(x, p + "/attention/layer_norm")
                q = self._split(self._linear(h, p + "/attention/linear_0"))
                x = x + self._linear(self._attend(q, *cross_kv[i], cross_mask), p + "/attention/linear_2")
                h = self._ln(x, p + "/ffn/layer_norm")
                x = x + self._linear(np.maximum(self._linear(h, p + "/ffn/linear_0"), 0), p + "/ffn/linear_1")
            pos += 1
            if pos < len(start_ids):  # still feeding the forced prefix (the target language token)
                step_input = np.array([[start_ids[pos]]] * n, dtype=np.int64)
                continue

            logits = self._logits(self._ln(x[:, -1], "decoder/layer_norm"))  # (n, |target vocab|)
            next_ids = np.empty(n, dtype=np.int64)
            for b in range(n):
                if done[b]:
                    next_ids[b] = eos_id
                    continue
                row = logits[b]
                seen = generated[b]
                if seen:
                    cols = np.searchsorted(self.target_ids, seen)
                    cols = cols[(cols < len(self.target_ids)) & (self.target_ids[np.minimum(cols, len(self.target_ids) - 1)] == seen)]
                    vals = row[cols]
                    row[cols] = np.where(vals > 0, vals / repetition_penalty, vals * repetition_penalty)
                    if len(seen) >= no_repeat_ngram - 1:
                        prefix = tuple(seen[-(no_repeat_ngram - 1):])
                        for j in range(len(seen) - no_repeat_ngram + 1):
                            if tuple(seen[j:j + no_repeat_ngram - 1]) == prefix:
                                banned = seen[j + no_repeat_ngram - 1]
                                c = np.searchsorted(self.target_ids, banned)
                                if c < len(self.target_ids) and self.target_ids[c] == banned:
                                    row[c] = -np.inf
                next_ids[b] = self.target_ids[int(np.argmax(row))]
                if next_ids[b] == eos_id or len(seen) + 1 >= max_len:
                    done[b] = True
                if next_ids[b] != eos_id:
                    generated[b].append(int(next_ids[b]))
            if done.all():
                break
            step_input = next_ids[:, None]
        return generated


class _Ct2Engine:
    def __init__(self, model_dir: str):
        import ctranslate2
        threads = max(1, min(8, os.cpu_count() or 2))
        self.t = ctranslate2.Translator(model_dir, device="cpu", compute_type="int8", intra_threads=threads)

    def translate(self, pieces: list[list[str]], src_lang: str) -> list[list[str]]:
        source = [[src_lang] + p + ["</s>"] for p in pieces]
        results = self.t.translate_batch(
            source, target_prefix=[[TARGET_LANG]] * len(source), beam_size=4, max_batch_size=16,
            max_decoding_length=256, repetition_penalty=1.2, no_repeat_ngram_size=4,
        )
        return [r.hypotheses[0][1:] for r in results]


def _read_vocabulary(model_dir: str) -> list[str]:
    """CTranslate2 writes shared_vocabulary.json (newer) or .txt (older): one token per entry."""
    path = os.path.join(model_dir, "shared_vocabulary.json")
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    with open(os.path.join(model_dir, "shared_vocabulary.txt"), encoding="utf-8") as fh:
        vocab = fh.read().split("\n")
    if vocab and vocab[-1] == "":
        vocab.pop()
    return vocab


def _english_token(piece: str) -> bool:
    core = piece.replace(METASPACE, "")
    return (bool(core) and all(c.isascii() for c in core)) or piece == METASPACE


class NllbTranslator:
    """Telugu -> English sentence translation; picks CTranslate2 or the numpy engine."""

    # Sentences decoded together: shares each step's weight dequantisation. Attention caches grow
    # with it, so a 32-bit process (2 GB address space, 0.6 GB of it the mapped model) uses fewer.
    BATCH = 64 if sys.maxsize > 2**32 else 32

    def __init__(self, model_dir: str, prefer_numpy: bool = False, threads: int = 0):
        self.tok = Tokenizer(model_dir)
        self.vocab = _read_vocabulary(model_dir)
        self.ids = {p: i for i, p in enumerate(self.vocab)}
        self.engine_name = "numpy"
        self.ct2 = None
        if not prefer_numpy:
            try:
                self.ct2 = _Ct2Engine(model_dir)
                self.engine_name = "ctranslate2"
            except Exception:  # not installed (32-bit) or failed to load: fall back to numpy
                self.ct2 = None
        if self.ct2 is None:
            # The numpy engine only scores tokens an English sentence can use (ASCII pieces).
            target = [i for i, p in enumerate(self.vocab) if _english_token(p)]
            target.append(self.ids["</s>"])
            self.np = NumpyNllb(model_dir, self.vocab, np.array(sorted(set(target)), dtype=np.int64), threads)

    def translate(self, texts: list[str], language: str = "te", progress=None) -> list[str]:
        src_lang = SOURCE_LANG[language]
        pieces = [self.tok.pieces(t)[:200] for t in texts]
        out: list[str] = [""] * len(texts)
        # Sort by length so each batch pads little; report progress per batch.
        order = sorted(range(len(texts)), key=lambda i: len(pieces[i]))
        for start in range(0, len(order), self.BATCH):
            idx = [i for i in order[start:start + self.BATCH] if pieces[i]]
            if idx:
                if self.ct2:
                    results = self.ct2.translate([pieces[i] for i in idx], src_lang)
                    for i, r in zip(idx, results):
                        out[i] = Tokenizer.detokenize(r)
                else:
                    src = [[self.ids[src_lang]] + [self.ids.get(p, self.tok.unk) for p in pieces[i]] + [self.ids["</s>"]]
                           for i in idx]
                    longest = max(len(s) for s in src)
                    results = self.np.translate(src, [self.ids["</s>"], self.ids[TARGET_LANG]], self.ids["</s>"],
                                                max_len=min(256, 2 * longest + 10))
                    for i, r in zip(idx, results):
                        out[i] = Tokenizer.detokenize([self.vocab[t] for t in r])
            if progress:
                progress(min(start + self.BATCH, len(order)) / len(order))
        return out
