# GGUF ingestion

InfiniteModel can run models that are published **only** as a llama.cpp **`.gguf`** file — the
large pool of community quants on Hugging Face that never shipped a safetensors checkpoint. It does
this by **normalizing the GGUF to a standard HF safetensors checkpoint once**, at add/download time.
After that one-time conversion the model is *ordinary* in the system: chunk-streamed to workers,
int4/int8 shard-cached, and run on the distributed pipeline with **no GGUF awareness anywhere
downstream** — exactly the same path a native safetensors model takes. (Same idea as the fp8 / nvfp4
source checkpoints: dequantize to bf16 once, then re-quantize to our own int4 for serving.)

Why not run GGUF directly? GGUF packs weights in GGML's k-quant / i-quant block layouts that the
GGML tensor library executes — not PyTorch. Porting those kernels would be large and pointless: the
engine already has a fast int4 decode path (`torch tinygemm` on NVIDIA/CPU, a Triton w4a16 kernel on
ROCm — see [ACCELERATION.md](ACCELERATION.md)). So the GGUF quantization is discarded; the value
GGUF ingestion unlocks is **access to the weights**, which are then served at the engine's own
quantization.

---

## Adding a GGUF model

You give the engine the **HF repo id** and the **single `.gguf` filename** within it.

**Dashboard:** **+ Add model** → put the repo id in the model field, and the `.gguf` filename in the
**"GGUF file (optional)"** box → Add + download.

**API:**

```bash
# repo id + the ONE quant file you want
curl -X POST "http://<controller>:21434/add_model?model=<hf-repo>&gguf_file=<file>.gguf"

# optionally give it a friendly name
curl -X POST "http://<controller>:21434/add_model?model=<hf-repo>&gguf_file=<file>.gguf&name=my-model"
```

The `gguf_file` must be a single `.gguf` filename that exists in the repo (validated — a value not
ending in `.gguf` is rejected). The repo is recorded as **GGUF-sourced** in `custom_gguf.json`; from
then on it's an ordinary registered model. Download (and thus conversion) starts like any other
model; the dashboard shows conversion progress on the model's row.

After it's converted, use it exactly like any model — `/load`, the chat/completions APIs, shard
compile, etc. Nothing else in the workflow is GGUF-specific.

---

## What the conversion does (under the hood)

The heavy step runs in a **subprocess** (`gguf_convert.py`, driven by
`model_store.convert_gguf_to_model_dir`) so a large `from_pretrained` — which fully materializes the
model in RAM — can OOM the *subprocess* without taking down the controller box it co-hosts. The
subprocess:

1. **Rejects a split GGUF early.** A `model-00001-of-00003.gguf`-style sharded quant can't be loaded
   from a single part by the transformers reader, so it's rejected up front with guidance (pick a
   single-file quant) rather than producing a half-converted model.
2. **Auto-installs its optional deps on demand** (the controller env has torch+transformers but may
   not have these extras, and the box may be SSH-less): `gguf` (parse the file), `accelerate`
   (the low-memory load path), and `sentencepiece` / `tiktoken` / `protobuf` (to build a tokenizer).
3. **Downloads the chosen `.gguf`** from the repo (`hf_hub_download`; HF token read from the
   `HF_TOKEN` env var, never a CLI arg — process listings leak args).
4. **Dequantizes to bf16** via the transformers GGUF loader and **saves a safetensors checkpoint**
   into `models/<name>/`.
5. **Produces a fast tokenizer**, verified by reloading it (this is the fiddly part — see below).
6. Prints `GGUF_CONVERT_OK <dir>` on success; any non-zero exit fails the add with the captured
   error.

### The tokenizer step

The controller is a long-running process that caches "is sentencepiece/tiktoken available?" **at
startup** — so even though the subprocess just pip-installed them, the controller can't convert a
*slow* tokenizer to *fast* at serve time. The conversion therefore insists on saving a **fast
`tokenizer.json`** (which loads purely via the `tokenizers` Rust lib that transformers always has),
trying two sources in order, each verified by reloading from the saved dir:

1. the **GGUF-embedded** tokenizer (works when the slow→fast deps convert cleanly), then
2. the **base repo's native** tokenizer — most GGUF repos are named `<base>-GGUF`, and the base repo
   usually ships a ready `tokenizer.json` that loads with no extra deps.

If neither yields a reload-verified fast tokenizer, the conversion **aborts** (the model would save
but be unusable at serve time) rather than leaving a broken model registered.

---

## Coverage & limitations

- **Architectures:** whatever the transformers GGUF loader supports — Llama, Qwen2, Mistral, Gemma,
  and the other mainstream families. An unsupported arch fails the conversion with the loader's error.
- **Single-file quants only.** Split `NNNNN-of-NNNNN.gguf` sets are rejected — choose a single-file
  quant (e.g. a `*-Q4_K_M.gguf` that isn't split).
- **One quant per repo.** A repo is mapped to one chosen `.gguf`. Re-adding the same repo with a
  different `gguf_file` updates the choice (and re-converts on next download).
- **The GGUF quantization is not preserved.** The weights are dequantized to bf16 and then served at
  the engine's own int4/int8 (or bf16). So a `Q4_K_M` GGUF doesn't stay `Q4_K_M` — it becomes bf16
  on disk and is re-quantized to our int4 for serving. Pick a GGUF quant that's high enough quality
  to survive that round-trip (a very low-bit GGUF has already lost information the engine can't
  recover).
- **Conversion is a one-time heavy step.** It materializes the full model in RAM in the subprocess;
  size the controller box accordingly (it's the same memory profile as loading the bf16 model once).

---

## Lifecycle & troubleshooting

- The GGUF mapping persists in `custom_gguf.json` (kept in lockstep with the model registry). It's
  gitignored / per-controller, like other custom-model state.
- **`/delete`** (purge files) and **`/forget`** drop the repo's GGUF mark along with the model.
- **Conversion failed with a tokenizer error** → the GGUF-embedded tokenizer needed sentencepiece/
  tiktoken and the base repo had no fast tokenizer to fall back to; try a repo whose base model
  ships a `tokenizer.json`, or a different GGUF of the same model.
- **"split/sharded GGUF is not supported"** → pick a single-file quant, not an `NNNNN-of-NNNNN` part.
- **OOM during conversion** → the box lacks RAM to materialize the full model; convert on a
  higher-RAM controller (the conversion is subprocess-isolated, so the controller itself survives
  the OOM and reports the failure).

See also: [OPERATIONS.md](OPERATIONS.md) (model lifecycle, shard-cache compile), and
[ACCELERATION.md](ACCELERATION.md) (the int4 path the converted model runs on).
