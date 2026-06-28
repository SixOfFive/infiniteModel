"""Standalone GGUF -> HF safetensors converter.

GGUF (llama.cpp's format) packs weights in k-quant / i-quant block layouts that the GGML
tensor library runs — not transformers/PyTorch. Rather than port those kernels (huge, and
pointless: we already have a fast int4 path), InfiniteModel NORMALIZES a GGUF model to a
standard HuggingFace safetensors checkpoint ONCE at add/download time. After that it is an
ordinary model in the system: chunk-streamed to workers, int4/int8 shard-cached, and run on the
distributed pipeline — no GGUF awareness anywhere downstream. This mirrors how fp8/nvfp4 source
checkpoints are handled (dequantize to bf16, then re-quantize to our int4 for serving).

Run as a SUBPROCESS by the controller (model_store.convert_gguf_to_model_dir) so a big
`from_pretrained` (which fully materializes the model in RAM) can OOM the SUBPROCESS without
taking down the controller box it co-hosts. Usage:

    python gguf_convert.py <repo_id> <gguf_file> <dst_dir>

The HF token (if any) is read from the HF_TOKEN env var (never a CLI arg — process listings leak
args). Prints a final ``GGUF_CONVERT_OK <dst_dir>`` line on success; exits non-zero on failure.
"""
import os
import sys
import subprocess


def _ensure_deps() -> None:
    """transformers' GGUF loader needs `gguf` (to parse the file) AND `accelerate` (it loads via the
    low-memory/device_map path). Auto-install whatever's missing on demand (the m4c84 worker pattern)
    so a controller env with torch+transformers but not these optional extras can still convert
    without a manual pip step on the (SSH-less) box."""
    need = []
    try:
        import gguf  # noqa: F401
    except Exception:
        need.append("gguf")
    try:
        import accelerate  # noqa: F401
    except Exception:
        need.append("accelerate")
    # Building the model's tokenizer from a GGUF (then saving it as a FAST tokenizer.json so the
    # controller loads it without a slow->fast conversion at serve time) needs sentencepiece/tiktoken,
    # and sentencepiece's converter needs protobuf. Without these the save leaves only a slow tokenizer
    # and the LATER load fails ("need sentencepiece or tiktoken to convert a slow tokenizer to a fast one").
    try:
        import sentencepiece  # noqa: F401
    except Exception:
        need.append("sentencepiece")
    try:
        import tiktoken  # noqa: F401
    except Exception:
        need.append("tiktoken")
    try:
        import google.protobuf  # noqa: F401
    except Exception:
        need.append("protobuf")
    if need:
        print(f"[gguf-convert] installing missing deps: {', '.join(need)}", flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", *need], check=False)


def main() -> int:
    if len(sys.argv) < 4:
        print("usage: gguf_convert.py <repo_id> <gguf_file> <dst_dir>", file=sys.stderr)
        return 2
    repo_id, gguf_file, dst = sys.argv[1], sys.argv[2], sys.argv[3]
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None

    # A split GGUF (model-00001-of-00003.gguf) can't be loaded from a single part by transformers'
    # reader — reject early with guidance rather than producing a half-converted model.
    base = os.path.basename(gguf_file).lower()
    import re as _re
    if _re.search(r"-\d{5}-of-\d{5}\.gguf$", base):
        print("[gguf-convert] split/sharded GGUF is not supported — pick a single-file quant "
              "(e.g. a *-Q4_K_M.gguf that isn't split into NNNNN-of-NNNNN parts)", file=sys.stderr)
        return 3

    _ensure_deps()
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    kw = {"token": token} if token else {}
    print(f"[gguf-convert] downloading {gguf_file} from {repo_id}", flush=True)
    path = hf_hub_download(repo_id, gguf_file, **kw)
    src_dir, fn = os.path.dirname(path), os.path.basename(path)

    print(f"[gguf-convert] dequantizing {fn} -> bf16 (transformers GGUF loader)", flush=True)
    model = AutoModelForCausalLM.from_pretrained(src_dir, gguf_file=fn, dtype=torch.bfloat16)
    # transformers may dequantize to fp32 regardless of dtype on some versions; force bf16 so the
    # saved checkpoint is the size our planner/streamer expects (we re-quantize to int4 anyway).
    model = model.to(torch.bfloat16)

    os.makedirs(dst, exist_ok=True)
    print(f"[gguf-convert] saving safetensors -> {dst}", flush=True)
    model.save_pretrained(dst, safe_serialization=True)

    if not _save_tokenizer(src_dir, fn, repo_id, dst, token):
        print("[gguf-convert] ERROR: could not produce a serve-loadable tokenizer "
              "(GGUF-embedded slow tokenizer needs sentencepiece/tiktoken to convert, and the base "
              "repo had none) — model saved but unusable; aborting", file=sys.stderr)
        return 4

    print(f"GGUF_CONVERT_OK {dst}", flush=True)
    return 0


def _save_tokenizer(src_dir: str, gguf_file: str, repo_id: str, dst: str, token) -> bool:
    """Produce a tokenizer in `dst` that the controller can load at SERVE time WITHOUT a slow->fast
    conversion (which needs sentencepiece/tiktoken — C/Rust extensions that may have no wheel on a
    bleeding-edge Python). Strategy, each VERIFIED by reloading from `dst`:
      1) the GGUF-embedded tokenizer (works when the slow->fast deps are installable), then
      2) the base model's native (already-fast) tokenizer — most GGUF repos are '<base>-GGUF', and the
         base repo ships a fast tokenizer.json that loads with no extra deps.
    Returns True if a reload-verified tokenizer was saved."""
    import re
    import shutil
    from transformers import AutoTokenizer

    def _try(make, why) -> bool:
        try:
            tok = make()
            tok.save_pretrained(dst)
            # CRITICAL: the controller is a long-running process that cached "sentencepiece/tiktoken
            # unavailable" at startup (this subprocess pip-installed them AFTER), so it can NOT convert
            # a slow tokenizer to fast at serve time. Require a fast `tokenizer.json` (loads purely via
            # the `tokenizers` Rust lib, which transformers always has) so the serve-time load needs no
            # conversion deps. A slow-only save would "verify" HERE (this subprocess has the deps) yet
            # fail on the controller — reject it so we fall through to the base repo's native fast one.
            if not os.path.exists(os.path.join(dst, "tokenizer.json")):
                raise RuntimeError("save produced no fast tokenizer.json (slow-only tokenizer)")
            AutoTokenizer.from_pretrained(dst)   # sanity: reloads
            print(f"[gguf-convert] tokenizer: {why} (fast tokenizer.json, verified)", flush=True)
            return True
        except Exception as exc:
            print(f"[gguf-convert] tokenizer via {why} failed: {exc!r}", flush=True)
            # wipe a partial/slow-only tokenizer so the next attempt (or the load) isn't fooled
            for f in os.listdir(dst):
                if "token" in f.lower() or f in ("vocab.json", "merges.txt", "special_tokens_map.json"):
                    with __import__("contextlib").suppress(Exception):
                        os.remove(os.path.join(dst, f))
            return False

    kw = {"token": token} if token else {}
    if _try(lambda: AutoTokenizer.from_pretrained(src_dir, gguf_file=gguf_file), "GGUF-embedded"):
        return True
    base = re.sub(r"[-_.]?gguf$", "", repo_id, flags=re.I)
    if base and base != repo_id:
        if _try(lambda: AutoTokenizer.from_pretrained(base, **kw), f"base repo {base}"):
            return True
    return False


if __name__ == "__main__":
    sys.exit(main())
