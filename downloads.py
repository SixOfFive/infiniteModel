"""downloads: model download / registry lifecycle, relocated VERBATIM from server.py
(code-split Inc 5): _pull_repo_interruptible (module level) and, inside register(app):
_start_download, _do_delete, _resolve_or_404, /download, /download/pause|stop|resume|clear,
/add_model, /delete, /forget, /api/pull, /api/delete.

WHERE THE STATE LIVES (do not "fix" this): every DOWNLOAD_* global (DOWNLOADING,
DOWNLOAD_PROGRESS/_ERROR/_CONTROL/_STATE/_EPOCH, DOWNLOAD_STATE_PATH) is DEFINED in
server.py and only MUTATED IN PLACE here -- the self-updater's idle lambda in server.py
reads DOWNLOADING as a live server global, so moving those definitions (or rebinding them
anywhere) would silently decouple the self-update idle gate: the ENCODING hazard documented
in state.py (ENCODING itself moved to media_encode.py WITH its mutators in Inc 11; the
lambda reads media_encode.ENCODING live). load/save_download_state also stay in server.py
beside their data. The persistence loaders are in-place as of Inc 4, so this module's bound
snapshot of CUSTOM_MODELS/GGUF_FILES/DELETED_MODELS/MODELS stays live across a reload.
Bodies BYTE-IDENTICAL; module globals injected by state.bind() -- see state.py.
Controller-only leaf; never imports server; in EXTRA_UPDATE_FILES.
"""
from __future__ import annotations


def _pull_repo_interruptible(friendly: str, repo_id: str):
    """Download a repo's *.safetensors/*.json into the HF cache ONE FILE AT A TIME so a
    pause/stop (DOWNLOAD_CONTROL[friendly]) can interrupt it between files. Returns
    'done' (all files present), 'paused', or 'stopped'.

    Why per-file and not snapshot_download: the heavy pull runs in a thread, and a
    Python thread can't be force-killed — so the only clean interrupt point is between
    files. The control flag is checked BEFORE each file and the current file is allowed
    to finish, so resume granularity is WHOLE-FILE: every completed shard stays in the
    cache and is skipped on resume, and nothing in flight is abandoned (we never kill a
    file mid-write). The cost is that pause/stop take effect only after the current shard
    finishes — up to a couple minutes for a big one. (A hard crash mid-shard is a
    different matter: huggingface_hub 1.x discards that one partial shard's bytes and
    re-pulls it, but the already-completed shards are kept.) hf_transfer (env) still
    accelerates each individual file. If the repo listing keeps failing, fall back to a
    single (non-interruptible) snapshot_download so a download is never blocked by a flaky
    list call — pause/stop won't bite until it finishes in that rare mode."""
    from huggingface_hub import HfApi, hf_hub_download
    tok = HF_TOKEN or None
    files = None
    for attempt in range(3):                          # transient list failures are common
        try:
            files = HfApi().list_repo_files(repo_id, token=tok)
            break
        except Exception:
            if attempt == 2:
                from huggingface_hub import snapshot_download
                print(f"[model] {friendly}: repo listing failed -> non-interruptible "
                      f"snapshot_download fallback (pause/stop won't apply this run)")
                snapshot_download(repo_id, allow_patterns=["*.safetensors", "*.json", "*.py",
                                                           "*.jinja", "*.txt", "*.model",
                                                           "*.pth", "*.pt"], token=tok)
                return "done"
    # include *.py: trust_remote_code models (auto_map) ship their modeling/configuration code as
    # .py — without them a worker builds the native class for the model_type (wrong arch -> meta
    # tensors, e.g. MiniMax-M2 'minimax' -> lightning Text-01). #78.
    # include *.jinja/*.txt/*.model: chat templates + tokenizer sidecars (merges.txt,
    # sentencepiece .model) — diffusers repos (#t2i) ship the tokenizer under tokenizer/ with
    # these, and Mistral3-style LLMs ship chat_template.jinja. Extension set mirrors
    # _hf_total_bytes so the progress denominator matches what is pulled.
    _ext = [".safetensors", ".json", ".py", ".jinja", ".txt", ".model"]
    # #tts: a repo with NO safetensors ships its weights as .pth/.pt (Kokoro = kokoro-v1_0.pth
    # + voices/*.pt). Pull those too so "+ Add model" fetches a COMPLETE non-safetensors model
    # instead of just config.json. Gated on "no safetensors" so ordinary checkpoints never pull
    # redundant/stray .pt (training snapshots, EMA copies) alongside their real safetensors.
    if not any(f.endswith(".safetensors") for f in files):
        _ext += [".pth", ".pt"]
    wanted = [f for f in files if f.endswith(tuple(_ext))]
    for f in wanted:
        ctrl = DOWNLOAD_CONTROL.get(friendly)        # checked BETWEEN files (cheap dict read)
        if ctrl in ("pause", "stop"):
            return "paused" if ctrl == "pause" else "stopped"
        # Windows WinError 32: an AV / a cache scanner can momentarily lock the freshly-pulled
        # blob right as huggingface_hub renames its `.incomplete` -> final, killing the download
        # at ~finalize with "used by another process". The bytes ARE on disk (hf_hub keeps the
        # `.incomplete`), so the finalize just needs to be retried once the handle releases —
        # retry with backoff instead of failing the whole pull. Non-WinError-32 errors re-raise.
        for _att in range(6):
            try:
                hf_hub_download(repo_id, f, token=tok)   # instant if the shard is already cached
                break
            except OSError as exc:                   # PermissionError (WinError 32) is an OSError
                if getattr(exc, "winerror", None) != 32 or _att == 5:
                    raise
                print(f"[model] {friendly}: {f} finalize locked (WinError 32) — "
                      f"retry {_att + 1}/5 after {2 * (_att + 1)}s", flush=True)
                time.sleep(2 * (_att + 1))
    return "done"


def register(app):

    async def _start_download(friendly: str) -> dict:
        """Kick off a background download of a configured model to the controller
        cache (fire-and-forget). Idempotent: no-op if ready. If a pull is already in
        flight, a pending pause/stop is CANCELLED (so Resume-during-pausing un-pauses
        the live thread instead of being a silent no-op)."""
        target = MODELS[friendly][0] if friendly in MODELS else friendly
        if model_ready(target):
            return {"ok": True, "status": "ready"}
        if friendly in DOWNLOADING:
            # Already pulling — if a pause/stop was pending, drop it so the running thread
            # keeps going (the between-files check now sees no signal). No new _dl needed.
            if DOWNLOAD_CONTROL.pop(friendly, None) is not None:
                log_activity(f"download {friendly}: resume (cancelled pending halt)")
                return {"ok": True, "status": "resuming"}
            return {"ok": True, "status": "downloading"}
        DOWNLOADING.add(friendly)
        DOWNLOAD_ERROR.pop(friendly, None)   # clear any prior failure on a fresh attempt
        DOWNLOAD_CONTROL.pop(friendly, None)  # drop any stale pause/stop signal from a prior run
        if DOWNLOAD_STATE.pop(friendly, None) is not None:  # starting/resuming -> no longer halted
            save_download_state()
        epoch = DOWNLOAD_EPOCH[friendly] = DOWNLOAD_EPOCH.get(friendly, 0) + 1
        log_activity(f"download {friendly}: starting")

        async def _dl():
            total = await asyncio.to_thread(_hf_total_bytes, target)
            DOWNLOAD_PROGRESS[friendly] = {
                "downloaded": await asyncio.to_thread(_hf_cache_bytes, target),
                "total": total}

            async def _poll():   # update bytes-so-far (+ rolling rate/ETA) while the download runs
                samples: list[tuple[float, int]] = []   # (monotonic ts, bytes) over a ~30s window
                try:
                    while friendly in DOWNLOADING and DOWNLOAD_EPOCH.get(friendly) == epoch:
                        db = await asyncio.to_thread(_hf_cache_bytes, target)
                        pr = DOWNLOAD_PROGRESS.get(friendly)
                        if pr is not None:
                            pr["downloaded"] = db
                            # Rolling average rate over the trailing ~30s window (smooths the
                            # per-file steps), then ETA = bytes-remaining / rate. Both live in pr
                            # so /status can surface them; cleared with pr when the download ends.
                            now = time.monotonic()
                            samples.append((now, db))
                            cutoff = now - 30.0
                            while len(samples) > 2 and samples[0][0] < cutoff:
                                samples.pop(0)
                            dt = samples[-1][0] - samples[0][0]
                            dbytes = samples[-1][1] - samples[0][1]
                            if dt >= 1.0 and dbytes > 0:
                                rate = dbytes / dt          # bytes/sec
                                pr["rate"] = rate
                                tot = pr.get("total") or 0
                                pr["eta_s"] = (tot - db) / rate if tot > db else 0.0
                        await asyncio.sleep(2)
                except asyncio.CancelledError:
                    pass

            poller = asyncio.create_task(_poll())
            halted = None
            try:
                # GGUF source: no safetensors to pull. Normalize the .gguf to a safetensors checkpoint
                # in a SUBPROCESS (download + dequant + save), then it's an ordinary model. Pause/stop
                # don't apply (it's a one-shot subprocess), so skip the interruptible pull entirely.
                if target in GGUF_FILES:
                    log_activity(f"download {friendly}: GGUF -> safetensors conversion (subprocess)")
                    await asyncio.to_thread(_controller_model_dir, target)   # triggers convert_gguf_to_model_dir
                    if DOWNLOAD_EPOCH.get(friendly) != epoch:
                        return
                    _invalidate_ready_cache(target)
                    print(f"[model] converted GGUF {friendly} ({target} :: {GGUF_FILES[target]})")
                    log_activity(f"download {friendly}: complete (GGUF normalized)")
                    return
                # Interruptible per-file pull -> 'done' | 'paused' | 'stopped'.
                result = await asyncio.to_thread(_pull_repo_interruptible, friendly, target)
                # If a clear (or a fresh start) bumped the epoch while we were pulling, THIS
                # run is stale — that handler already cleaned up; don't write/resurrect state.
                if DOWNLOAD_EPOCH.get(friendly) != epoch:
                    return
                if result in ("paused", "stopped"):
                    halted = result
                    DOWNLOAD_STATE[friendly] = result    # persist so a restart keeps it halted
                    await asyncio.to_thread(save_download_state)
                    done = (DOWNLOAD_PROGRESS.get(friendly) or {}).get("downloaded", 0)
                    print(f"[model] download {friendly} {result} at {done / GB:.1f} GB")
                    log_activity(f"download {friendly}: {result}")
                else:
                    # every file now in the HF cache -> migrate to models/ + purge the dup
                    await asyncio.to_thread(_controller_model_dir, target)
                    _invalidate_ready_cache(target)
                    print(f"[model] downloaded {friendly} ({target})")
                    log_activity(f"download {friendly}: complete")
            except Exception as exc:
                if DOWNLOAD_EPOCH.get(friendly) != epoch:
                    return                               # superseded -> swallow
                msg = f"{type(exc).__name__}: {exc}"
                low = msg.lower()
                if any(k in low for k in ("gated", "403", "401", "awaiting", "access to model",
                                          "restricted", "you must")):
                    msg += "  (gated repo — accept the license for this model on huggingface.co "
                    msg += "with the account whose token the controller uses)"
                DOWNLOAD_ERROR[friendly] = msg[:400]
                print(f"[model] download failed for {friendly}: {exc!r}")
                log_activity(f"download {friendly}: FAILED ({type(exc).__name__})")
            finally:
                poller.cancel()
                if DOWNLOAD_EPOCH.get(friendly) == epoch:   # only OUR run owns this state
                    DOWNLOADING.discard(friendly)
                    DOWNLOAD_CONTROL.pop(friendly, None)
                    # done/error -> drop the progress bar; pause/stop -> KEEP it frozen so the
                    # dashboard shows where it halted (and offers Resume from there).
                    if halted is None:
                        DOWNLOAD_PROGRESS.pop(friendly, None)

        asyncio.create_task(_dl())
        return {"ok": True, "status": "downloading"}

    async def _do_delete(friendly: str) -> dict:
        """Delete a model COMPLETELY from the controller: its weight/quant CACHE
        (models/<name>/ incl. the _shards/<quant>/ pre-quant caches AND the HF-cache
        copy) AND its registry footprint — EVERY registered name that resolves to the
        same repo (the model + any alias names re-registered against the same HF id),
        its GGUF mark, and any built-in alias pointing at it. Full removal: delete ==
        forget + purge files. Refuses if any of those names is loaded or downloading."""
        target = MODELS[friendly][0] if friendly in MODELS else friendly
        # Every name that resolves to the SAME repo shares the on-disk files we're about
        # to remove, so all of them must be unregistered too — otherwise they'd dangle on
        # a now-missing model. Custom 'aliases' = multiple CUSTOM_MODELS keys -> one HF id;
        # plus any built-in MODEL_ALIASES entry whose key or canonical target is in the set.
        names = {friendly} | {k for k, hf in CUSTOM_MODELS.items() if hf == target}
        names |= {a for a, c in MODEL_ALIASES.items() if a in names or c in names}
        loaded = sorted(n for n in names if n in engine.models)
        if loaded:
            return {"ok": False,
                    "error": f"model is currently loaded ({', '.join(loaded)}) — unload it first"}
        busy = sorted(n for n in names if n in DOWNLOADING)
        if busy:
            return {"ok": False,
                    "error": f"model is downloading ({', '.join(busy)}) — wait for it to finish"}
        deleted = await asyncio.to_thread(delete_model_cache, target)
        # Purge the registry footprint regardless of whether files were present, so a
        # half-registered model (registered, no files) is still fully removed by a delete.
        forgot, hidden = [], []
        for n in list(names):
            if CUSTOM_MODELS.pop(n, None) is not None:
                forgot.append(n)               # custom: persistence via custom_models.json
            elif n in MODELS:
                hidden.append(n)               # built-in: persistence via the deleted hide-set
            MODELS.pop(n, None)                # drop from the live list (no stale download button)
            MODEL_ALIASES.pop(n, None)         # drop any alias keyed by this name
        if forgot:
            GGUF_FILES.pop(target, None)       # all registrations for this repo are gone
            save_custom_models()
        if hidden:
            DELETED_MODELS.update(hidden)
            save_deleted_models()
        removed = bool(deleted or forgot or hidden)
        if removed:
            print(f"[model] deleted {friendly} ({target}) — cache_removed={deleted} "
                  f"unregistered={sorted(forgot) or '[]'} hidden={sorted(hidden) or '[]'}", flush=True)
        return {"ok": removed,
                "error": None if removed else "model not present in cache or registry"}

    @app.post("/download")           # dashboard: pull a configured model to controller
    async def download(model: str) -> JSONResponse:
        try:
            friendly = resolve_model_name(model)
        except ValueError as exc:
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=404)
        return JSONResponse(await _start_download(friendly))

    def _resolve_or_404(model: str):
        try:
            return resolve_model_name(model), None
        except ValueError as exc:
            return None, JSONResponse({"ok": False, "error": str(exc)}, status_code=404)

    @app.post("/download/pause")     # dashboard: pause an in-flight download (cache kept, resumable)
    async def download_pause(model: str) -> JSONResponse:
        friendly, err = _resolve_or_404(model)
        if err:
            return err
        if friendly not in DOWNLOADING:
            return JSONResponse({"ok": False, "error": "not currently downloading"}, status_code=409)
        DOWNLOAD_CONTROL[friendly] = "pause"   # the per-file pull stops after the current shard
        log_activity(f"download {friendly}: pause requested")
        return JSONResponse({"ok": True, "status": "pausing"})

    @app.post("/download/stop")      # dashboard: stop an in-flight download (cache kept, resumable)
    async def download_stop(model: str) -> JSONResponse:
        friendly, err = _resolve_or_404(model)
        if err:
            return err
        if friendly not in DOWNLOADING:
            return JSONResponse({"ok": False, "error": "not currently downloading"}, status_code=409)
        DOWNLOAD_CONTROL[friendly] = "stop"
        log_activity(f"download {friendly}: stop requested")
        return JSONResponse({"ok": True, "status": "stopping"})

    @app.post("/download/resume")    # dashboard: resume a paused/stopped download from the cache
    async def download_resume(model: str) -> JSONResponse:
        friendly, err = _resolve_or_404(model)
        if err:
            return err
        # _start_download clears the persisted halt + stale control signal, then re-runs the
        # per-file pull — cached files are skipped instantly and the partial file resumes.
        return JSONResponse(await _start_download(friendly))

    @app.post("/download/clear")     # dashboard: wipe a model's cached + partial files ("reset")
    async def download_clear(model: str) -> JSONResponse:
        friendly, err = _resolve_or_404(model)
        if err:
            return err
        if friendly in engine.models:
            return JSONResponse({"ok": False, "error": "model is loaded — unload it first"},
                                status_code=409)
        target = MODELS[friendly][0] if friendly in MODELS else friendly
        if friendly in DOWNLOADING:
            DOWNLOAD_CONTROL[friendly] = "stop"   # ask the in-flight pull to stop between files
        else:
            DOWNLOAD_CONTROL.pop(friendly, None)  # nothing running -> don't leave a stale signal
        # Bump the epoch so the (possibly still-running) _dl for this model sees its result is
        # stale and won't re-persist DOWNLOAD_STATE after we clear it (the clear-resurrect race).
        DOWNLOAD_EPOCH[friendly] = DOWNLOAD_EPOCH.get(friendly, 0) + 1
        DOWNLOADING.discard(friendly)
        DOWNLOAD_PROGRESS.pop(friendly, None)
        DOWNLOAD_ERROR.pop(friendly, None)
        if DOWNLOAD_STATE.pop(friendly, None) is not None:
            save_download_state()
        # Measure both locations first (cache copy + any models/<name>), then delete both.
        # rmtree uses ignore_errors, so a file still locked by an in-flight pull thread is
        # skipped — a second Clear (after the thread exits between files) mops up any residue.
        def _measure() -> int:
            n = _hf_cache_bytes(target)
            mdir = os.path.join(MODELS_DIR, _safe_name(target))
            if os.path.isdir(mdir):
                for root, _dirs, files in os.walk(mdir):
                    for f in files:
                        with contextlib.suppress(OSError):
                            n += os.path.getsize(os.path.join(root, f))
            return n
        freed = await asyncio.to_thread(_measure)
        removed = await asyncio.to_thread(delete_model_cache, target)
        log_activity(f"download {friendly}: cache cleared (~{freed / GB:.1f} GB)")
        print(f"[model] cleared cache for {friendly} ({target})")
        return JSONResponse({"ok": True, "removed": removed, "freed_gb": round(freed / GB, 2)})

    @app.post("/add_model")          # dashboard: register + download ANY Hugging Face id
    async def add_model(model: str, name: str = "", gguf_file: str = "") -> JSONResponse:
        # `name` (optional): override the client-facing model name instead of deriving it from
        # the HF id. Lets a precision-suffixed repo (e.g. ModelCloud/MiniMax-M2-BF16) be served
        # under a clean, quant-agnostic name (e.g. minimax-m2) — quant is a load-time choice, so
        # it shouldn't live in the name. Re-registering an already-cached HF id under a new name
        # is instant (no re-download — the cache is keyed by HF id, not the friendly name).
        hf = (model or "").strip()
        # HF repo ids are colon-free (dash form, e.g. 'Qwen/Qwen3-4B'). A user may paste the
        # Ollama 'family:size' form into the org/name field ('qwen/qwen3:4b'), which 404s on the
        # Hub. Normalize ':' -> '-' in the TARGET id so both forms resolve to the real repo — the
        # friendly registry KEY derived below already collapses ':' via _friendly_from_hf, but the
        # download target came straight from this string. (No HF id legitimately contains ':'.)
        hf = hf.replace(":", "-")
        if "/" not in hf or " " in hf or hf.count("/") > 1:
            return JSONResponse({"ok": False,
                                 "error": "enter a Hugging Face id like org/name"},
                                status_code=400)
        # A user-supplied override may be typed in the Ollama 'family:size' form ('qwen3:4b');
        # collapse it to the canonical colon-free dash key ('qwen3-4b') so the registry key,
        # the URL query param, and the on-disk filename stay simple — the colon display is
        # rendered on demand by _ollama_name(). Validate AFTER normalizing (so ':' is allowed
        # as input but never stored as a key).
        friendly = _normalize_model_request(name) if (name or "").strip() else _friendly_from_hf(hf)
        if not re.fullmatch(r"[a-z0-9._-]+", friendly):
            return JSONResponse({"ok": False,
                                 "error": "name must be lowercase [a-z0-9._-] (':' allowed as the size separator)"},
                                status_code=400)
        # GGUF source (optional): the repo ships weights only as a llama.cpp .gguf — record which
        # single-file quant to use, so acquisition normalizes it to safetensors (subprocess) instead
        # of pulling safetensors that don't exist. Keyed by the HF repo (== the target id).
        gf = (gguf_file or "").strip()
        if gf and not gf.lower().endswith(".gguf"):
            return JSONResponse({"ok": False,
                                 "error": "gguf_file must be a single .gguf filename in the repo"},
                                status_code=400)
        if friendly not in MODELS:
            MODELS[friendly] = (hf, hf)          # draft = target (no speculative)
            CUSTOM_MODELS[friendly] = hf
            if gf:
                GGUF_FILES[hf] = gf              # mark this target as GGUF-sourced
            save_custom_models()
            log_activity(f"added model {friendly} ({hf})" + (f" [GGUF {gf}]" if gf else ""))
        elif gf and GGUF_FILES.get(hf) != gf:
            GGUF_FILES[hf] = gf                  # update the chosen quant for an already-registered repo
            save_custom_models()
        if friendly in DELETED_MODELS:           # re-adding a previously deleted model un-hides it
            DELETED_MODELS.discard(friendly)
            save_deleted_models()
        r = await _start_download(friendly)
        return JSONResponse({"ok": True, "friendly": friendly, "target": hf,
                             "gguf_file": gf or None, "status": r.get("status")})

    @app.post("/delete")             # dashboard: delete a model from controller
    async def delete_model(model: str) -> JSONResponse:
        try:
            friendly = resolve_model_name(model)
        except ValueError as exc:
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=404)
        r = await _do_delete(friendly)
        return JSONResponse(r, status_code=200 if r["ok"] else 409)

    @app.post("/forget")             # remove a custom-model REGISTRY entry but KEEP its weight files
    async def forget_model(model: str) -> JSONResponse:
        """Unregister a custom (added) model: drop its friendly->HF mapping from CUSTOM_MODELS +
        MODELS + custom_models.json. UNLIKE /delete, this does NOT delete the cached weight files
        — the model stays on disk, just no longer registered. Refuses if currently loaded."""
        # Prefer the LITERAL registered entry over an alias redirect: a custom model whose name
        # is shadowed by a built-in MODEL_ALIASES entry (e.g. 'qwen2.5:14b', shadowed by
        # qwen2.5-14b -> qwen2.5-14b-instruct) is otherwise UNFORGETTABLE — resolve_model_name
        # would redirect to the alias target and report it "loaded". (#forget-shadow)
        literal = _normalize_model_request(model)
        if literal in CUSTOM_MODELS:
            friendly = literal
        else:
            try:
                friendly = resolve_model_name(model)
            except ValueError as exc:
                return JSONResponse({"ok": False, "error": str(exc)}, status_code=404)
        if friendly in engine.models:
            return JSONResponse({"ok": False, "error": "model is loaded — unload it first"},
                                status_code=409)
        if friendly not in CUSTOM_MODELS:
            # Not a custom entry. A BUILT-IN can still be removed from the list: "forget" HIDES it
            # (persisted in the deleted hide-set) while KEEPING any downloaded weights — unlike
            # /delete, which also purges files. Re-adding via /add_model un-hides it. Without this a
            # built-in (e.g. mixtral:8x7b) flashed "built-ins can't be forgotten" and never left the
            # list. (#forget-builtin)
            if friendly in MODELS:
                MODELS.pop(friendly, None)
                MODEL_ALIASES.pop(friendly, None)   # drop any alias keyed by this name
                DELETED_MODELS.add(friendly)
                save_deleted_models()
                print(f"[model] forgot built-in {friendly} (hidden from list; weight files KEPT)",
                      flush=True)
                return JSONResponse({"ok": True, "forgot": friendly, "hf": None,
                                     "files_kept": True, "builtin": True})
            return JSONResponse({"ok": False, "error": f"'{friendly}' is not a registered model"},
                                status_code=404)
        hf = CUSTOM_MODELS.pop(friendly, None)
        MODELS.pop(friendly, None)
        if hf and not any(v == hf for v in CUSTOM_MODELS.values()):
            GGUF_FILES.pop(hf, None)   # last registry entry for this repo gone -> drop its GGUF mark
        save_custom_models()
        print(f"[model] forgot registry entry {friendly} ({hf}) — weight files KEPT", flush=True)
        return JSONResponse({"ok": True, "forgot": friendly, "hf": hf, "files_kept": True})

    @app.post("/api/pull")           # Ollama-compat pull -> background download
    async def api_pull(req: Request) -> JSONResponse:
        body = await req.json()
        name = body.get("model") or body.get("name") or ""
        try:
            friendly = resolve_model_name(name)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=404)
        r = await _start_download(friendly)
        return JSONResponse({"status": "success" if r.get("status") == "ready"
                             else "pulling manifest"})

    @app.delete("/api/delete")       # Ollama-compat delete
    async def api_delete(req: Request) -> JSONResponse:
        body = await req.json()
        name = body.get("model") or body.get("name") or ""
        try:
            friendly = resolve_model_name(name)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=404)
        r = await _do_delete(friendly)
        return (JSONResponse({"status": "success"}) if r["ok"]
                else JSONResponse({"error": r["error"]}, status_code=409))
