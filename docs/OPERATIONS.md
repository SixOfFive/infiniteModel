# Operations guide

Running a fleet day-to-day: model lifecycle, overload behavior, self-healing, the config
reference, and deploy/self-update semantics. Everything here is reachable from the dashboard's
settings page and per-model detail modal, or over plain HTTP (`curl`-able).

---

## Model lifecycle

### Multiple resident models

N models can be resident at once, sharing nodes; requests queue per model (`queue_depth`) and
generations for one model are serialized. The dashboard shows placement previews before a load,
per-load progress, fleet memory/throughput, and per-model live tok/s.

### Idle unload

`/config?idle_unload_m=N` (settings page): unload any model that served no requests for N
minutes. Default `0` = every model stays loaded forever (`-1`, the Ollama-style spelling, is
accepted with the same meaning). Models with an active or queued request — and models holding
either lifecycle pin below — are never idle-unloaded. The reaper is group-wise: a model's
replicas are considered together.

### Per-model lifecycle pins

Both live on the model-detail modal:

- **Autoload on restart** (`/config?persist=<model>` / `unpersist=`): re-streams the model to its
  workers on controller startup, so it survives a restart or redeploy. Startup reload waits for
  the worker fleet to settle **plus** `autostart_delay_s` (default 60 s) so API clients reconnect
  before the controller gets busy streaming weights.
- **Do not auto-unload** (`/config?no_unload=<model>` / `no_unload_off=`): an **absolute veto** on
  automatic removal — never reclaimed by idle-unload *or* by LRU eviction. A new load that can't
  fit without evicting a pinned model **fails** instead. (The juggler may still *re-place* a
  pinned model into a better layout — a reload-in-a-better-way, never a removal.)

The two are independent: persist survives restarts but stays evictable; no-unload blocks removal
but doesn't survive a restart unless also persisted.

### The juggler — hitless VRAM promotion

`/config?juggler=true` (settings page, opt-in, default off). Models often load *hybrid* (part
GPU, part CPU RAM) under memory pressure; decode speed tracks the GPU-resident fraction, so a
hybrid model left that way is permanently slow. The juggler fixes that automatically:

- On a **~60 s periodic sweep** and immediately after an idle-unload frees VRAM, it looks for
  the **hottest** resident hybrid model that would now fit **entirely on GPU**.
- If that model is **momentarily idle** (no active or queued request), it is *promoted* by a
  **hitless re-place**: new requests briefly pause on their still-open connections (no reconnect,
  no error) while the model re-places VRAM-first, then resume on the faster copy. The pause spans
  only the re-place itself (~10–20 s for a small model).
- A **busy** model is skipped, never stalled — a later sweep catches it between requests.
  Embedding models and models too big to ever fit GPU are skipped. An **anti-churn guard**
  remembers a promotion that could only reach a partial fit and won't retry until the fleet
  actually frees more VRAM.

### Placement is static after load

A model's GPU/RAM split never drifts while loaded. If a model "moved to RAM at some point," that
was a **re-load** (restart, redeploy, eviction, or an auto-load that raced a busy fleet) — check
`autoload_mode` (below) and the juggler.

### Shard-cache ops

- **`POST /compile_shards?model=<name>&quant=int4|int8|int2`** — build the pre-quantized
  `_shards/<quant>/` cache explicitly. Compiles run in a background **subprocess** (they never
  starve live generations), concurrently with loads and other compiles; a duplicate model+quant
  is refused with `409`; the dashboard shows a live progress card (`ready/total · elapsed · eta`).
- **int4/int8** caches also build automatically on the first cache-less load (`precompile=0`
  opts out). **int2 is explicit-compile only**: its GPTQ calibration is sequential and can run
  hours for a big model, so it is never built on the fly — and an int2 *load* without a valid
  calibrated cache **fails loud** with the compile instruction instead of serving a degraded
  model.
- **`GET /shard_status`** — which quants are pre-compiled per model; **`POST /verify_shards`** —
  full sha256 integrity check of a cache.

---

## Overload & failure behavior

### Honest backpressure

Under GPU contention the endpoints degrade into *retryable* backpressure, not failures:

- Slow-but-**advancing** prefills are never reclaimed as wedged — workers report per-layer
  forward progress over their heartbeat, and the controller extends its wait while progress
  advances.
- Contention-class failures return `503 + Retry-After` (Ollama/OpenAI) or `529 overloaded_error`
  (Anthropic) instead of bare 500s or dropped sockets.
- **Terminal capacity is honest too:** a request for a cold model when the fleet is at
  `max_loaded` with **no automatic path to room** (`auto_unload` off, or every resident
  no-unload-pinned) returns a `503` with code `at_capacity` and **no** `Retry-After` — a signal
  to stop retrying — on all API surfaces. With `auto_unload` on (or an eviction candidate
  available), the same situation stays the *retryable* `503 + Retry-After` shape.

### Worker stage errors surface instantly

A worker-side compute exception during a forward is reported two ways at once: an error frame
down the data-plane chain **and** a mirror over the heartbeat-kept control link (`stage_error`).
Whichever arrives first fails the request immediately with the worker's real error — a fast,
causal 500 instead of a silent multi-minute stall. Every arrival is logged in the activity feed,
matched to a live request or not.

### Node drop, reap, and re-register

- A worker that misses heartbeats is **reaped** (idle nodes on a short timeout; a node actively
  serving a shard — or the target of an in-flight load — gets a much longer grace, so a busy box
  is never falsely reaped). Models with a stage on the reaped node are invalidated and recover on
  the next request.
- Reaping also **closes the node's control connection**. This matters after a network blip: a
  worker whose TCP connection *survived* the blip would otherwise keep heartbeating into a socket
  the controller no longer recognizes — invisible zombie, forever — since workers only register on
  a fresh connect. The close forces the worker's reconnect loop, and it re-registers within
  seconds. (A heartbeat arriving for an unregistered node id drops the link the same way.)
- A worker that **restarts** and re-registers under a new node id is auto-recovered: the stale
  registration is dropped and any model the old entry held is failed fast + re-placed, instead of
  hanging the next generation on a dead stage.

### Gen-stall watchdog + wedge quarantine

- **Watchdog** (`gen_stall_s`, default 240 s; `gen_stall_decode_s`, default 60 s once a token has
  been produced): a generation that stops producing tokens (and reports no forward progress) is
  reclaimed — its slot, queue, and per-model lock are reset so the next request re-flows the
  pipeline.
- **Quarantine** (`wedge_reload_n`, default 3; `0` disables): if the *same model* is reclaimed
  N times inside 15 minutes, it is systematically broken (stale worker state, poisoned pipeline),
  and the controller forces a **fresh re-place** automatically — new shards, new data connections,
  rollback-safe — instead of letting client retries re-wedge it forever.

### Frame-level guards (distributed data plane)

- Stage 0 refuses a floating-point input frame where token ids are expected (a mispaired or
  misrouted frame) with a self-describing error instead of a cryptic kernel crash.
- A vision prefill's multimodal companion frame is *declared* on the ids frame; declared-but-
  missing fails loud (the prefill is never silently run unspliced), and undeclared frames never
  consume a stale companion. Staged companions expire after 10 minutes.
- Next-hop wiring translates loopback addresses for remote receivers, so a worker co-located with
  the controller can never cause a remote stage to dial itself.

---

## Configuration reference

All runtime knobs persist across restarts and are settable from the dashboard settings page or
`POST /config?<knob>=<value>`:

| Knob | Default | Meaning |
|---|---|---|
| `max_loaded` | 8 | Max resident models before LRU eviction (pins win). |
| `auto_load` | true | First request for a non-resident model loads it. |
| `auto_unload` | false | Let a load that doesn't fit LRU-evict **idle** residents to make room (pins win). Off = the load fails instead (and cold requests at capacity get the terminal `at_capacity` 503). |
| `auto_tp` / `auto_tp_ratio` | false / — | Auto-route a CPU-bound model to CPU tensor-parallel when its weights exceed `ratio ×` the GPU pool. Off by default — measured: CPU-TP never beats pipelining onto a GPU on this fleet. |
| `autoload_quant` | `int4` | Quant tier for auto-loads (one-shot bf16 fallback on failure). |
| `autoload_ctx` | 8192 | Default ctx for auto/click loads. |
| `autoload_mode` | `auto` | Placement mode for auto-loads. **`auto` = GPU-first**; `single` is RAM-first — if models "randomly" land in RAM, check this. |
| `queue_depth` | 8 | Per-model request queue before `429/503`. |
| `idle_unload_m` | 0 | Idle minutes before unload (0/-1 = never). |
| `persist` / `unpersist` | — | Add/remove a model from autoload-on-restart. |
| `no_unload` / `no_unload_off` | — | Add/remove the absolute do-not-auto-unload veto. |
| `juggler` | false | Hitless VRAM promotion of hybrid models. |
| `autostart_delay_s` | 60 | Client-reconnect grace before the startup persisted-model reload. |
| `gen_stall_s` / `gen_stall_decode_s` | 240 / 60 | Watchdog thresholds (prefill / mid-decode). |
| `wedge_reload_n` | 3 | Auto re-place after N watchdog reclaims of one model in 15 min (0 = off). |
| `vram_weights_first` | true | Budget new weights against live-free VRAM. |

Per-load knobs (query params on `/load`): `quant` (`none|int8|int4|int2` — **omitted ⇒ the
`autoload_quant` default, normally int4**, NOT bf16; pass `quant=none` explicitly for bf16; int2
requires its pre-compiled calibrated cache, see Shard-cache ops), `ctx`, `mode`
(`auto|single|gpu-spread|all-gpu|distribute|spread|proportional`), `node` (pin to one node),
`tp` (tensor-parallel width), `replicas`, `kv_offload=1` (KV cache in system RAM — frees the VRAM
KV reserve for layers; CUDA only, force-disabled on ROCm), `kv_quant` (`turbo2|turbo3|turbo4`),
`temperature` / `min_p` (per-model defaults used when a request sends none), `draft_gpu=1` +
`draft_margin_gb` (reserve VRAM for a speculative-decode draft), `precompile=0` (skip
cache-on-first-load), `force=1` (cancel + restart a wedged load), `t2i_offload=1` (image models
only — rest the bf16 diffusion pipeline in system RAM and stream blocks to the co-located GPU
per step; ~4 GB free VRAM + RAM for the weights, never evicts residents; see
[T2I.md](T2I.md)).

Per-model **runtime sampling defaults** — `POST /model_config?model=...` with any of `top_p`,
`top_k`, `min_p`, `temperature`, `repeat_penalty`, `repeat_last_n`, `presence_penalty`,
`frequency_penalty`, `seed`, `num_predict` — apply instantly to a loaded model (empty string
clears; explicit request values always win). The dashboard's model-detail **Runtime settings**
panel edits all of them.

The same sampling family works **per-request** on all three APIs (Ollama `options.*`; top-level
on OpenAI/Anthropic; `repetition_penalty` accepted as an alias). **Min-p** drops tokens below
`min_p ×` the top token's probability — a confidence-adaptive floor that pairs well with high
temperature.

---

## Observability

- `GET /status` — full fleet + per-model state (JSON). Judge deploy freshness by `code_date`,
  not the version string.
- `GET /logs` — the controller's live log over HTTP; `GET /logs?node=<host>` — any worker's log,
  relayed over its heartbeat (no shell access needed).
- `GET /code_manifest?grep=<marker>` — per-file mtime/sha1/grep-hit of the deployed sources, for
  verifying exactly what a deploy landed.
- `GET /history?model=<name>` — recent prompts/outputs for a loaded model.
- Dashboard: per-client connections panel (bytes/tokens, `POST /terminate`), bandwidth page,
  activity feed.

---

## Restart semantics

Three restart shapes, all on the Config page (or `POST /restart`):

| Button | Call | What happens |
|---|---|---|
| **Restart controller** | `/restart?workers=0` | Controller only — **hitless for loaded models**: workers keep their shards across the restart and report them on reconnect; the controller **re-adopts** the resident models from those reports instead of re-streaming them (`ADOPTED <model> from the running fleet` in the activity log). |
| **Restart fleet** | `/restart?workers=1&controller=0` | Every **worker** process relaunches; the controller stays up. Resident models drop (a worker restart wipes its shards) and re-load on demand or via persist pins. The reset for stale worker state / wedged loads / allocator-held VRAM when the controller is healthy. |
| **Restart all** | `/restart?workers=1` | Controller + workers — the full reset. Persisted models re-stream after startup + `autostart_delay_s`; everything else re-auto-loads on demand. |
| **Restart one node** (↻ on each node row) | `/restart_node?node=<hostname\|id>` | Relaunches **just that one worker** process — the per-node fresh start that clears whatever VRAM/RAM it's holding, without touching the controller or the rest of the fleet. Models with a stage on the node drop, then split by usage (below). |

Adoption details (controller-only restart): models re-adopt within seconds of the workers
reconnecting. **Not adopted** — tensor-parallel models, and any model whose stage workers don't
all come back within the ~90 s grace (their kept shards are then freed and the model re-loads
normally). Spec-decode **drafts** are controller-local and are not re-attached — reload the model
to restore speculative decode. A request that arrives mid-adoption waits briefly (~10 s) for the
adoption instead of racing a duplicate load. Mixed code versions degrade safely: an old worker
(or old controller) simply drops shards on link loss like before.

Per-node restart handles the models it drops by usage, so a fresh-start on a busy node isn't
disruptive:

- **In use** (serving/queued right now, or used within the last 10 minutes) → **auto-recovered**:
  once the node's death invalidates the model, a background task re-loads it with its original
  ctx / quant / KV settings and the planner re-places it onto whatever capacity is up — other
  nodes' GPU/CPU, or the restarted node itself once it rejoins (usually within seconds). A failed
  recovery falls back to on-demand auto-load, logged loudly. (Replicas aren't auto-recovered —
  re-add them manually.)
- **Idle** → the surviving stages on the *other* nodes are freed too, so the model costs nothing
  anywhere and re-auto-loads on the next request.

The response and the dashboard toast list both sets (`recovering: [...]` vs the dropped ones).

## Deploy & self-update

- **Idle self-update** polls the repo every **15 minutes** (was 2), applies fetched module files
  as they change, but only *restarts* on a VERSION bump. The immediate path is the forced update:
- **Forced `POST /update` ("Update + deploy") is fleet-wide immediate**: it unloads all models,
  pushes an **immediate update command to every worker** (files stage now; `/update?workers=1`
  has each worker stage the new files *before* its relaunch so it comes back on fresh code), then
  the controller swaps its own code and restarts. Deploys never wait on the 15-minute poll.
- **Hitless deploy — `POST /update?hitless=1`** (added 2026-07-18): pull the latest code and
  adopt-restart the controller **without unloading any model**. Unlike a plain `/update` (which
  full-unloads every model + frees worker RAM), `hitless=1` leaves the workers entirely untouched —
  they keep their shards and the relaunched controller **re-adopts** them on the new code (see
  Restart semantics above). It runs the same `_self_update_check(force=True)` fetch-apply of
  `server.py` + all `EXTRA_UPDATE_FILES`, just without the destructive preamble. This is the
  preferred path for **controller-side** changes (dashboard / routes / status / placement /
  serving / graphs). It is **not** for worker code (`client.py` / `worker_*.py`): a worker only
  runs new code after a restart that wipes its shards, so use a plain `/update` or **Restart all**
  for those. Guarded like `/restart` (refuses mid load / compile / render unless `force=1`).
  Bootstrapping the endpoint on a controller that predates it: let the idle poll stage the file
  (same-VERSION → applied to disk, no auto-restart), then `/restart?workers=0` picks it up.
- **Worker processes still do NOT restart on a plain `/update?workers=0`** — files land
  immediately, but the running processes keep executing the old code until a VERSION bump or a
  restart. After a deploy that changes worker-side modules, use `/update?workers=1` or
  **Restart all**. Long-lived worker processes accumulating stale state is a real failure class —
  a periodic fleet-wide worker restart is cheap hygiene.
- **Both `/update` and `/restart` refuse (`409`) while work is in flight** — `/restart` while a
  load, compile, or text-to-image render is running; `/update` while a render is running (a
  forced update once orphaned a finished 12-minute render into a broken pipe). Renders are
  minutes-bounded: wait, or pass `force=1` to abort the work and proceed.
- Raw-CDN propagation after a push is **per-file** and can lag minutes unevenly. Before a forced
  `/update`, verify **every** changed file's marker (e.g. `curl raw.../<file> | grep <marker>` or
  `GET /code_manifest?grep=` after), not just one.
- A **controller-only** restart keeps workers' shards (see Restart semantics above); a forced
  `/update` deliberately does NOT — a deploy may change model/worker code, so it starts clean.
  Persisted models re-stream on startup (after `autostart_delay_s`), everything else
  re-auto-loads on demand.
