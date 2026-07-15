"""WorkerLoadMixin: relocated Worker methods (m4c153 code-split). BODIES BYTE-IDENTICAL to the
originals in client.py; module globals injected at startup by state.bind() — see state.py.
Composed via ``class Worker(WorkerLoadMixin, …)`` so self.* resolves across mixins by MRO. Worker-side
leaf module; in client.py EXTRA_UPDATE_FILES.
"""
from __future__ import annotations


class WorkerLoadMixin:

    def _build_shard(self, base: str, model_id: str, a: dict) -> Shard:
        import tempfile
        cfg = json.loads(_http_get(f"{base}/modelmeta?model={urllib.parse.quote(model_id)}"))
        if cfg.get("auto_map"):                  # trust_remote_code model: also fetch its modeling .py
            with contextlib.suppress(Exception): # so the shard builds the CORRECT architecture instead
                rc = json.loads(_http_get(      # of transformers' native fallback class (which can
                    f"{base}/modelcode?model={urllib.parse.quote(model_id)}"))   # mismatch the ckpt)
                if isinstance(rc, dict) and rc:
                    cfg["__im_remote_code__"] = rc
        tp_size = int(a.get("tp_size", 1))
        tp_rank = int(a.get("tp_rank", 0))
        device = a.get("device") or self.device   # controller's per-node tier choice wins
        quant = a.get("quant") or self.quant      # load-time quant (controller) wins over launch flag
        plan_ram_bytes = int(a.get("plan_ram_bytes", 0) or 0)   # #63: planned resident RAM to reserve
        # #95: controller's committed-aware GPU budget for this stage (free VRAM after co-resident
        # models, minus the plan floor). -1 when the controller didn't send one (old controller) ->
        # the worker placement stays uncapped (legacy behavior). >=0 caps GPU placement in _place_modules.
        gpu_budget_gb = float(a.get("gpu_budget_gb", -1.0))
        # #moe-offload: controller opt-in to keep a MoE layer's attention+norms on GPU and leave the
        # routed-expert block in CPU RAM (llama.cpp --override-tensor experts=CPU, intra-layer).
        # Pipeline-only (the TP path ignores it); the worker further gates to int4 experts.
        moe_offload = bool(a.get("moe_offload", False))
        # #shard-cache Inc 2 (serve-from-cache): controller flags '' | 'int4' | 'int2'. When set,
        # fetch PRE-PACKED layer units (cache=<quant> on /weights) and install them directly — no
        # bf16 stream, no per-layer re-quant. Pipeline only (the controller never sets it for TP);
        # gated to the matching quant so a stale flag can never install a cross-tier cache.
        cache = (a.get("cache", "") or "") if quant in ("int4", "int2") else ""
        if tp_size <= 1:
            # DEFAULT PATH: stream each slice ONE LAYER AT A TIME straight into RAM bytes, then
            # st_load -> HEAP tensors (m4c25). NO temp files anywhere. The old path staged each slice
            # in a /dev/shm tmpfs file and mmap-loaded it; but tmpfs IS RAM, so it double-charged
            # memory, capped a node at its (smaller) shm size INDEPENDENT of free RAM, and ENOSPC'd
            # mid-load (the /dev/shm FileNotFound failure). And for CPU-resident layers it cloned to
            # heap anyway (_drop_slice_mmap), so there was no 1x win to keep. Pure bytes uses the
            # node's FULL RAM, frees the buffer per layer (~2x ONE layer transient — same as before),
            # and deletes the temp-file / ENOSPC / cleanup-race failure mode entirely.
            def fetch(start: int, end: int, embed: int, head: int, skip_experts: bool = False):
                qd = {"model": model_id, "start": start, "end": end,
                      "embed": int(bool(embed)), "head": int(bool(head)),
                      "skip_experts": int(bool(skip_experts))}
                if cache:
                    qd["cache"] = cache   # #shard-cache Inc 2: fetch the pre-packed int4 unit
                q = urllib.parse.urlencode(qd)
                url = f"{base}/weights?{q}"
                last = None
                for attempt in range(3):       # L+2 small fetches -> bounded retry vs LAN hiccups
                    try:
                        return _http_get(url)               # straight into RAM bytes -> heap tensors
                    except Exception as exc:
                        last = exc
                        time.sleep(0.5 * (attempt + 1))
                raise last

            def fetch_experts(layer: int, e0: int, k: int):
                # Stream a CHUNK of per-expert source tensors [e0:e0+k] of one MoE layer (#62) ->
                # dict {'{local_e}.{proj}': bf16}. Small blob (~chunk experts); in-RAM bytes is fine.
                from safetensors.torch import load as st_load
                q = urllib.parse.urlencode({"model": model_id, "layer": layer, "e0": e0, "k": k})
                url = f"{base}/experts?{q}"
                last = None
                for attempt in range(3):
                    try:
                        return st_load(_http_get(url))
                    except Exception as exc:
                        last = exc
                        time.sleep(0.5 * (attempt + 1))
                raise last

            shard = Shard.from_stream(cfg, fetch, a["layer_start"], a["layer_end"],
                                      a["has_embed"], a["has_head"], a.get("dtype", "bfloat16"),
                                      device=device, gpu_mem_gb=self.gpu_mem_gb,
                                      attn=self.attn, quant=quant, fetch_experts=fetch_experts,
                                      plan_ram_bytes=plan_ram_bytes, ctx=int(a.get("ctx", 0) or 0),
                                      gpu_budget_gb=gpu_budget_gb,   # #95 coexistence cap
                                      moe_offload=moe_offload,       # #moe-offload (pipeline only)
                                      cache=cache,                   # #shard-cache Inc 2 serve-from-cache
                                      kv_quant=a.get("kv_quant", "none"),   # #172 TurboQuant KV preset
                                      kv_offload=bool(a.get("kv_offload", False)))  # #kv-offload: KV in RAM
        else:
            # TENSOR-PARALLEL PATH (tp>1) — TP-v2 PER-RANK STREAMING: this rank fetches ONLY its
            # 1/tp tensor slice from /weights_tp and builds reduced-dim modules directly, so a node
            # holds ~1/tp of each layer (NOT the v1 load-full-then-shard footprint). Stand up the
            # all-reduce mesh BEFORE fetching so ranks rendezvous early (rank 0 binds; peers connect).
            tp = _TPAllReduce(tp_rank, tp_size, a.get("tp_root_host"), int(a.get("tp_root_port", 0)))
            self._tp = tp
            self._tp_model_id = model_id
            tpw = a.get("tp_weights")   # #68: per-rank capacity weights -> heterogeneous split (else uniform)
            wstr = ",".join(str(x) for x in tpw) if tpw else ""
            def fetch(start: int, end: int, embed: int, head: int, skip_experts: bool = False):
                # PER-RANK slice serve: /weights_tp returns this stage's tensors already sliced for
                # (tp_rank, tp_size). One blob per layer-slice (+ embed/head); small -> retry-bounded.
                # Straight into RAM bytes -> heap tensors; no /dev/shm temp file (m4c25, see pipeline path).
                qd2 = {"model": model_id, "start": start, "end": end,
                       "embed": int(bool(embed)), "head": int(bool(head)),
                       "tp_rank": tp_rank, "tp_size": tp_size}
                if wstr:
                    qd2["weights"] = wstr   # must match what _tp_make_structure_ built from
                q = urllib.parse.urlencode(qd2)
                url = f"{base}/weights_tp?{q}"
                last = None
                for attempt in range(3):
                    try:
                        return _http_get(url)
                    except Exception as exc:
                        last = exc
                        time.sleep(0.5 * (attempt + 1))
                raise last

            shard = Shard.from_stream(cfg, fetch, a["layer_start"], a["layer_end"],
                                      a["has_embed"], a["has_head"], a.get("dtype", "bfloat16"),
                                      device=device, gpu_mem_gb=self.gpu_mem_gb,
                                      attn=self.attn, quant=quant,
                                      tp_rank=tp_rank, tp_size=tp_size, tp_allreduce=tp,
                                      plan_ram_bytes=plan_ram_bytes, tp_weights=tpw,
                                      ctx=int(a.get("ctx", 0) or 0),
                                      gpu_budget_gb=gpu_budget_gb,   # #95 coexistence cap
                                      kv_quant=a.get("kv_quant", "none"),   # #172 TurboQuant KV preset
                                      kv_offload=bool(a.get("kv_offload", False)))  # #kv-offload: KV in RAM
        print(f"[load] stage L{a['layer_start']}-{a['layer_end']} placement: {shard.placement}"
              f" device={device} attn={self.attn} quant={quant} ({shard.loaded_bytes/GB:.2f} GB)")
        # #2 pre-alloc: reserve the full-ctx KV now so a node that can't hold it fails the LOAD
        # (clean, replannable) instead of OOMing mid-generation. ctx comes from the load msg.
        ctx = int(a.get("ctx", 0) or 0)
        if ctx > 0:
            shard.kv_reserve_probe(ctx)
        return shard

    def _cleanup_weight_tmp(self, model_id: str) -> None:
        tmp = self._weight_tmps.pop(model_id, None)
        if tmp:
            with contextlib.suppress(Exception):
                os.remove(tmp)

    def _cleanup_all_weight_tmps(self) -> None:
        for mid in list(self._weight_tmps):
            self._cleanup_weight_tmp(mid)

    async def handle_load(self, msg: dict) -> dict:
        model_id = msg["model_id"]
        a = msg
        # Reload of the SAME model: drop just its old shard first; keep other models resident.
        # (The Inc 1/2 controller still unloads every node before a load, so usually nothing
        # else is resident yet — this matters once Inc 3 enables fit-as-many.)
        await self._unload_model(model_id)
        self.assignments[model_id] = msg
        # A tensor-parallel PEER (tp_rank>0) is NOT in the pipeline: it has no data port and
        # no 'next' — it's driven entirely by rank 0's broadcasts over the all-reduce mesh.
        is_peer = int(a.get("tp_size", 1)) > 1 and int(a.get("tp_rank", 0)) > 0
        try:
            if not is_peer and self.data_server is None:
                # Bind the shared data port ONCE; every model's pipeline reuses it (frames
                # carry model_id so _data_inbound routes each frame to the right shard).
                try:
                    self.data_server = await asyncio.start_server(
                        self._data_inbound, "0.0.0.0", self.args.data_port)
                except OSError as exc:
                    raise RuntimeError(
                        f"data port {self.args.data_port} unavailable on "
                        f"{socket.gethostname()} ({exc}); give each worker on a shared "
                        f"host a distinct --data-port") from exc
            base = f"http://{self.args.controller}:{msg['controller_http_port']}"
            # EMBEDDING load (encoder, BERT-family): build the whole model on THIS one node — no
            # pipeline, no Shard, no KV. Acquire the weights via snapshot_download (mirrors from_hf's
            # pattern) but ALSO pull *.py so nomic's custom modeling/tokenizer code comes down; the
            # repo is public, so no token (matches from_hf). The encoder holder lives in self.shards
            # like a Shard but only serves kind:"embed" frames.
            if a.get("kind") == "embedding":
                import torch
                from huggingface_hub import snapshot_download
                device = a.get("device") or self.device
                if device == "":
                    device = self.device
                dtype = torch.float32   # encoder runs fp32 (CPU default; tiny model)
                self._building += 1
                try:
                    def _build_embed():
                        local_dir = snapshot_download(
                            model_id,
                            allow_patterns=["*.json", "*.py", "*.safetensors", "*.txt",
                                            "*.model", "tokenizer*"])
                        # auto-install any pip dep the model's trust_remote_code needs (#84)
                        return _build_with_autodeps(
                            lambda: EmbeddingModel(local_dir, device, dtype), label=model_id)
                    em = await asyncio.to_thread(_build_embed)
                    self.shards[model_id] = em
                finally:
                    self._building -= 1
                # No next hop: the encoder replies straight to the controller over its inbound conn.
                self.next_writers.pop(model_id, None)
                self.next_peer[model_id] = "controller"
                print(f"[load] embedding {model_id} on {em.device} "
                      f"({em.loaded_bytes / GB:.2f} GB, {em.loaded_params/1e6:.0f}M params)",
                      flush=True)
                return {"loaded_params": em.loaded_params,
                        "loaded_bytes": em.loaded_bytes,
                        "gpu_bytes": getattr(em, "gpu_bytes", 0)}
            # T2I load (#t2i-serve): a diffusers image-generation pipeline (Qwen-Image class)
            # builds WHOLE on THIS node — no pipeline stages, no KV. v1 constraint: the worker
            # must be CO-LOCATED with the controller (shared filesystem) — the controller sends
            # its LOCAL model dir and this branch reads it directly (no weight streaming), and
            # generated PNGs are handed back as local paths. worker_t2i is imported lazily with
            # a fetch-if-missing bridge so pre-#t2i workers converge without a restart loop.
            if a.get("kind") == "t2i":
                mdir = a.get("model_dir") or ""
                if not os.path.isdir(mdir):
                    raise RuntimeError(
                        f"t2i model dir not visible on this worker: {mdir!r} — v1 serves "
                        "image models only on a GPU worker co-located with the controller")
                try:
                    import worker_t2i
                except Exception:
                    from worker_update import _fetch_repo_file
                    _src = _fetch_repo_file("worker_t2i.py")
                    if not _src:
                        raise RuntimeError("worker_t2i.py missing and unfetchable (CDN lag?) — retry")
                    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                           "worker_t2i.py"), "wb") as _tf:
                        _tf.write(_src)
                    import worker_t2i
                device = a.get("device") or self.device
                if device == "":
                    device = self.device
                self._building += 1
                try:
                    eng = await asyncio.to_thread(
                        worker_t2i.T2IPipeline, mdir, device,
                        str(a.get("quant") or "int4"), int(a.get("t2i_edge", 2)),
                        bool(a.get("t2i_offload", False)))
                    self.shards[model_id] = eng
                finally:
                    self._building -= 1
                self.next_writers.pop(model_id, None)
                self.next_peer[model_id] = "controller"
                return {"loaded_params": eng.loaded_params,
                        "loaded_bytes": eng.loaded_bytes,
                        "gpu_bytes": eng.gpu_bytes}
            # TTS load (#tts-serve): a Kokoro speech model builds WHOLE on THIS node (co-located
            # with the controller — reads the model dir directly, hands back WAV paths). worker_tts
            # is imported lazily with the same fetch-if-missing bridge as worker_t2i. The leaf falls
            # back to CPU on its own if the GPU's MIOpen kernels fail to JIT-compile (gfx1151).
            if a.get("kind") == "tts":
                mdir = a.get("model_dir") or ""
                if not os.path.isdir(mdir):
                    raise RuntimeError(
                        f"tts model dir not visible on this worker: {mdir!r} — v1 serves "
                        "speech models only on a worker co-located with the controller")
                try:
                    import worker_tts
                except Exception:
                    from worker_update import _fetch_repo_file
                    _src = _fetch_repo_file("worker_tts.py")
                    if not _src:
                        raise RuntimeError("worker_tts.py missing and unfetchable (CDN lag?) — retry")
                    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                           "worker_tts.py"), "wb") as _tf:
                        _tf.write(_src)
                    import worker_tts
                device = a.get("device") or self.device
                if device == "":
                    device = self.device
                self._building += 1
                try:
                    eng = await asyncio.to_thread(
                        worker_tts.KokoroPipeline, mdir, device,
                        str(a.get("quant") or "none"), False)
                    self.shards[model_id] = eng
                finally:
                    self._building -= 1
                self.next_writers.pop(model_id, None)
                self.next_peer[model_id] = "controller"
                return {"loaded_params": eng.loaded_params,
                        "loaded_bytes": eng.loaded_bytes,
                        "gpu_bytes": eng.gpu_bytes}
            self._building += 1   # mark BUSY across the build so reclaim/self-update can't kill it
            try:
                shard = await asyncio.to_thread(self._build_shard, base, model_id, a)
                self.shards[model_id] = shard
            finally:
                self._building -= 1
            if is_peer:
                self._tp_stop = False
                self._tp_thread = threading.Thread(target=self._tp_follow, daemon=True)
                self._tp_thread.start()
            else:
                next_host = a.get("next_host") or self.args.controller
                # Do NOT pre-connect the next hop at LOAD. A connection opened here then left idle
                # until the first generate can silently go half-open — reliably so on the FIRST
                # generate after a CONTROLLER RESTART, where the worker's logits write SUCCEEDS (no
                # exception) but the bytes never reach the controller, so m4bz's reconnect-on-error
                # never fires and the controller just waits out GEN_TIMEOUT (~600s). Confirmed by
                # tracing both ends: the worker's stage ran and its logits write returned success
                # (bytes "sent"), yet the controller's _on_data never received a frame for that req.
                # Leaving next_writers UNSET makes _send_next lazy-connect it
                # FRESH on the first send (zero idle gap) — exactly what a manual unload+reload did to
                # self-heal. Decode-step sends reuse that hot connection (rapid, no idle gap).
                self.next_writers.pop(model_id, None)
                # label the next hop for the bandwidth page: "controller" (last stage) or the
                # next worker's IP. a.get("next_host") is None on the last stage -> controller.
                self.next_peer[model_id] = "controller" if not a.get("next_host") else str(next_host)
                # #tp-mesh-keepalive: if THIS load made us TP rank 0, start the idle-ping keepalive
                # thread so the lockstep mesh sockets never sit idle long enough to half-open.
                if self._tp is not None and getattr(self._tp, "rank", 1) == 0:
                    self._tp_stop = False
                    self._tp_last_fwd = time.time()
                    self._tp_ka_thread = threading.Thread(target=self._tp_keepalive_loop, daemon=True)
                    self._tp_ka_thread.start()
            return {"loaded_params": shard.loaded_params,
                    "loaded_bytes": shard.loaded_bytes,
                    "gpu_bytes": getattr(shard, "gpu_bytes", 0),
                    "gpu_kv_bytes": getattr(shard, "gpu_kv_bytes", 0),
                    "placement": getattr(shard, "placement", None),   # observability + #moe-offload diag
                    "moe": getattr(shard, "_moe_dbg", None)}
        except Exception as exc:
            import traceback
            print(f"[load] {model_id} build FAILED: {exc!r}\n{traceback.format_exc()}", flush=True)
            await self._unload_model(model_id)  # drop the half-built shard; stay connected + idle
            # Do NOT _maybe_self_restart_if_stuck() here: a FAILED build that exit(42)s turns one
            # recoverable load failure into a restart -> reconnect -> retry -> fail loop that
            # desyncs the TP mesh (the observed churn). Reclaim belongs to the explicit-unload path;
            # if a failed partial build leaked RAM, the periodic mem flush / a controller-sent
            # unload reclaims it without an uncontrolled process exit.
            raise

    def _tp_follow(self) -> None:
        """Peer-rank loop: block for rank 0's broadcast input, run the sharded forward
        (all-reducing with the group via the mesh), discard the result, repeat. Exits on a
        stop broadcast (b'') or when the mesh drops (unload)."""
        import pickle
        while not self._tp_stop:
            try:
                payload = self._tp.recv_broadcast()
            except Exception:
                break
            if not payload or self._tp_stop:
                break
            if payload == _TP_PING:   # #tp-mesh-keepalive: liveness ping from rank 0 -> ack + wait
                with contextlib.suppress(Exception):
                    self._tp.ack_ping()
                continue
            try:
                data = pickle.loads(payload)
                # tuple grew over versions: 4 (base) -> 5 (+inject) -> 6 (+position_ids) ->
                # 7 (+bidir_spans); tolerate all during a rolling self-update.
                inject = position_ids = bidir_spans = None
                if len(data) >= 7:
                    xt, cache_start, reset, all_logits, inject, position_ids, bidir_spans = data[:7]
                elif len(data) == 6:
                    xt, cache_start, reset, all_logits, inject, position_ids = data
                elif len(data) == 5:
                    xt, cache_start, reset, all_logits, inject = data
                else:
                    xt, cache_start, reset, all_logits = data
                self.shards[self._tp_model_id].forward(xt, cache_start, reset, all_logits,
                                                       inject, position_ids, bidir_spans=bidir_spans)
            except Exception as exc:
                print(f"[tp-follow] forward error: {exc!r}")
                break

    def _tp_keepalive_loop(self) -> None:
        """#tp-mesh-keepalive (rank 0): while the TP model is resident, ping the peers whenever the
        mesh has been idle for TP_KEEPALIVE_S so the lockstep sockets never sit idle long enough to
        go silently half-open (the 'peer rank stalled' break after an idle gap between requests). A
        real forward stamps _tp_last_fwd, so a busy model skips the ping. Runs under _tp_lock so it
        can't interleave with a forward's broadcast/all-reduce. On a failed ping the mesh is dead;
        flag it (served on the heartbeat) and stop — the model needs a reload either way."""
        while not self._tp_stop:
            time.sleep(min(TP_KEEPALIVE_S, 2.0))
            tp = self._tp
            if self._tp_stop or tp is None or getattr(tp, "rank", 1) != 0:
                return
            if time.time() - self._tp_last_fwd < TP_KEEPALIVE_S:
                continue   # a recent forward already kept the mesh warm
            ok = True
            if self._tp_lock.acquire(timeout=TP_KEEPALIVE_S):
                try:
                    if self._tp is not None and not self._tp_stop:
                        ok = self._tp.keepalive()
                        if ok:
                            self._tp_last_fwd = time.time()
                finally:
                    self._tp_lock.release()
            if not ok:
                self._tp_broken = True
                print("[tp-keepalive] mesh ping FAILED — a peer is unreachable; "
                      "TP mesh is down, model needs a reload", flush=True)
                return

    def _teardown_tp(self) -> None:
        """Tear down the all-reduce mesh (TP is a single-model mode). Caller joins the thread."""
        if self._tp is not None:
            self._tp_stop = True
            if getattr(self._tp, "rank", 1) == 0:
                with contextlib.suppress(Exception):
                    self._tp.broadcast(b"")   # tell peers to leave their follow loop
            with contextlib.suppress(Exception):
                self._tp.close()
        self._tp = None
        self._tp_model_id = None
        self._tp_ka_thread = None
        self._tp_broken = False

    def _maybe_self_restart_if_stuck(self) -> None:
        """After going fully idle (no shards), if this process still holds far more RAM than its
        fresh baseline, the OS didn't reclaim the dropped shard — Windows keeps committed private
        bytes until the process exits; glibc can retain freed arenas. A restart is then the ONLY
        reliable reclaim, so exit(42) and let the supervisor relaunch clean. Guards: only when
        fully idle (no live shard is dropped) and only well above baseline. A fresh process sits
        near baseline, so it never restart-loops. Call ONLY from explicit-unload paths, never from
        the unload at the START of handle_load (that would kill an incoming load)."""
        if self.shards or self._building:   # a resident shard OR an in-flight build -> not idle
            return
        try:
            import psutil
            rss_gb = psutil.Process().memory_info().rss / GB
        except Exception:
            return
        if rss_gb > self._rss_baseline_gb + 8.0:
            print(f"[reclaim] idle but still holding {rss_gb:.1f} GB (fresh baseline "
                  f"{self._rss_baseline_gb:.1f} GB) — OS won't reclaim it; restarting to free it",
                  flush=True)
            os._exit(42)   # supervisor relaunches on the same code -> RAM returns to the OS

    def _release_shard_vram(self, shard) -> None:
        """#vram-release-rocm: free a shard's GPU tensor STORAGES IN PLACE before dropping it. On
        gfx1151 a dropped int4 shard stayed pinned at full weight size (~13.5 GB) after unload —
        empty_cache works there (verified) but only frees tensors with NO remaining refs, and some
        lingering ref (triton-kernel closure / fused tuple / autograd) still pointed at the shard's
        weights, so they never freed until a process restart. Resetting each CUDA param/buffer to an
        EMPTY storage releases the bytes regardless of who else holds the module (the held object's
        own tensor is emptied in place); the gc + empty_cache that follow then reclaim. No-op for a
        CPU-resident shard; harmless on CUDA (frees a touch earlier). Safe at unload (model idle)."""
        import torch
        import contextlib as _cl
        if getattr(shard, "kind", "") in ("t2i", "t2a", "tts"):
            # #t2i/#t2a/#tts-vram-release: a media pipeline (T2IPipeline / T2APipeline /
            # KokoroPipeline) has NONE of the shard attrs walked below (model/embed/norm/head/
            # encoder), so this generic pass freed NOTHING and the weights stayed pinned on
            # ROCm/CUDA after unload (observed live for t2i's ~12 GB DiT). Their own release is
            # RENDER-SAFE: under a live generate it defers the free to the render's end.
            with _cl.suppress(Exception):
                shard.release_vram()
            return
        m = getattr(shard, "model", None)
        mods = [m] if m is not None else []
        for _attr in ("embed", "norm", "head", "encoder"):
            x = getattr(shard, _attr, None)
            if x is not None and x is not m:
                mods.append(x)
        seen: set = set()
        for mod in mods:
            if not hasattr(mod, "parameters"):
                continue
            with _cl.suppress(Exception):
                for t in list(mod.parameters(recurse=True)) + list(mod.buffers(recurse=True)):
                    if t is None or getattr(t, "device", None) is None or t.device.type != "cuda":
                        continue
                    if id(t) in seen:
                        continue
                    seen.add(id(t))
                    # #39 ROOT CAUSE: the #dram-dealias pad registers a VIEW (buf[:, :rs]) as the
                    # buffer; emptying only the view leaves the padded BASE's full storage alive
                    # through the C-level ._base reference (invisible to gc referrers) — measured
                    # ~10 GB surviving a qwen3-30b unload (48 padded expert stacks + dense pads).
                    # Empty the BASE first, then the view.
                    with _cl.suppress(Exception):
                        b = getattr(t, "_base", None)
                        if b is not None and getattr(b, "device", None) is not None \
                                and b.device.type == "cuda" and id(b) not in seen:
                            seen.add(id(b))
                            b.data = torch.empty(0, dtype=b.dtype, device=b.device)
                    with _cl.suppress(Exception):
                        t.data = torch.empty(0, dtype=t.dtype, device=t.device)
            with _cl.suppress(Exception):   # drop each int4 layer's cached fused tuple (qweight + op)
                for sub in mod.modules():
                    if getattr(sub, "_fused", None) is not None:
                        sub._fused = None

    async def _unload_model(self, model_id: str) -> None:
        """Drop ONE model's shard + next-hop conn + temp file, keeping other models resident.
        If it was the TP model, tear the mesh + follower thread down too. Closes the shared
        data server only once no shards remain."""
        if model_id == self._tp_model_id:
            self._teardown_tp()
            if self._tp_thread is not None:
                with contextlib.suppress(Exception):
                    await asyncio.to_thread(self._tp_thread.join, 5)
                self._tp_thread = None
        w = self.next_writers.pop(model_id, None)
        if w is not None:
            with contextlib.suppress(Exception):
                w.close()
        _sh = self.shards.pop(model_id, None)
        if _sh is not None:
            self._release_shard_vram(_sh)   # #vram-release-rocm: free GPU storages in place first
        self.assignments.pop(model_id, None)
        # Drop any staged multimodal embeds for THIS model so they can't be mis-consumed by a
        # later request after a controller restart (req_id resets to 0 -> key reuse).
        for k in [k for k in self.pending_mm if k[0] == model_id]:
            self.pending_mm.pop(k, None)
        import gc
        gc.collect()
        _release_vram()   # return this shard's VRAM to the pool so the next load can use the GPU
        self._cleanup_weight_tmp(model_id)   # release mmap (gc above) then delete the temp file
        _release_ram(trim_working_set=not self.shards)   # return freed CPU RAM (trim only if now idle)
        if not self.shards and self.data_server is not None:
            with contextlib.suppress(Exception):
                self.data_server.close()
            self.data_server = None

    def _pack_skeleton(self, base: str, model_id: str):
        """Build (once, CACHED per model_id) the meta skeleton used to fuse a per-expert MoE checkpoint
        into the model's fused-3D layout at distributed-pack time (#distributed-packing Inc 3b). Fetches
        the model config (+ any trust_remote_code .py) the SAME way a cold load does (/modelmeta +
        /modelcode), then shards.build_skeleton_from_config -> the IDENTICAL skeleton the controller's
        _quant_scope builds, so the fused+packed unit is bit-identical to a controller-local compile by
        construction. Cheap to cache (meta-only model, no real tensors); built once for all N layers."""
        m = self._pack_skel.get(model_id)
        if m is not None:
            return m
        import shards
        cfg = json.loads(_http_get(f"{base}/modelmeta?model={urllib.parse.quote(model_id)}"))
        if cfg.get("auto_map"):                  # trust_remote_code model: also fetch its modeling .py
            with contextlib.suppress(Exception):
                rc = json.loads(_http_get(f"{base}/modelcode?model={urllib.parse.quote(model_id)}"))
                if isinstance(rc, dict) and rc:
                    cfg["__im_remote_code__"] = rc
        m = shards.build_skeleton_from_config(cfg)
        self._pack_skel[model_id] = m
        return m

    async def handle_t2i_gen(self, msg: dict, reply) -> None:
        """#t2i-serve: run ONE image generation on this worker's resident T2IPipeline and mirror
        the result over the control link. Dispatched as a TASK by command_loop (a render takes
        minutes — awaiting it inline would block unload/ping handling), so replies are keyed by
        req_id. Per-step progress mirrors as `t2i_step` (scheduled threadsafe from the render
        thread); the finished PNG's LOCAL path returns in `t2i_done` (controller is co-located,
        v1). _building marks the worker busy so self-update/reclaim won't kill a live render."""
        rid = msg.get("req_id")
        mid = msg.get("model_id")
        eng = self.shards.get(mid)
        if eng is None or getattr(eng, "kind", "") != "t2i":
            await reply({"type": "t2i_err", "req_id": rid, "model_id": mid,
                         "error": "t2i model not resident on this worker"})
            return
        loop = asyncio.get_running_loop()

        def _on_step(i: int, n: int) -> None:
            with contextlib.suppress(Exception):
                asyncio.run_coroutine_threadsafe(
                    reply({"type": "t2i_step", "req_id": rid, "model_id": mid,
                           "step": i, "total": n}), loop)

        try:
            self._building += 1
            try:
                path, secs = await asyncio.to_thread(
                    eng.generate, str(msg.get("prompt") or ""),
                    str(msg.get("negative_prompt") or " "),
                    int(msg.get("width", 1024)), int(msg.get("height", 1024)),
                    int(msg.get("steps", 20)), float(msg.get("cfg", 4.0)),
                    msg.get("seed"), _on_step)
            finally:
                self._building -= 1
            await reply({"type": "t2i_done", "req_id": rid, "model_id": mid,
                         "path": path, "seconds": round(secs, 1)})
        except Exception as exc:
            with contextlib.suppress(Exception):
                await reply({"type": "t2i_err", "req_id": rid, "model_id": mid,
                             "error": repr(exc)})
            print(f"[t2i] generate FAILED: {exc!r}", flush=True)

    async def handle_tts_gen(self, msg: dict, reply) -> None:
        """#tts-serve: run ONE speech synthesis on this worker's resident KokoroPipeline and
        mirror the result over the control link. Dispatched as a TASK by command_loop (a long
        text takes many seconds — awaiting inline would block unload/ping handling), so replies
        are keyed by req_id. Per-chunk progress mirrors as `tts_step` (scheduled threadsafe from
        the synth thread); the finished WAV's LOCAL path returns in `tts_done` (controller is
        co-located, v1). _building marks the worker busy so self-update/reclaim won't kill it."""
        rid = msg.get("req_id")
        mid = msg.get("model_id")
        eng = self.shards.get(mid)
        if eng is None or getattr(eng, "kind", "") != "tts":
            await reply({"type": "tts_err", "req_id": rid, "model_id": mid,
                         "error": "tts model not resident on this worker"})
            return
        loop = asyncio.get_running_loop()

        def _on_step(i: int, n: int) -> None:
            with contextlib.suppress(Exception):
                asyncio.run_coroutine_threadsafe(
                    reply({"type": "tts_step", "req_id": rid, "model_id": mid,
                           "step": i, "total": n}), loop)

        try:
            self._building += 1
            try:
                path, secs = await asyncio.to_thread(
                    eng.generate, str(msg.get("text") or ""),
                    str(msg.get("voice") or ""), float(msg.get("speed", 1.0)),
                    str(msg.get("fmt") or "wav"), _on_step)
            finally:
                self._building -= 1
            await reply({"type": "tts_done", "req_id": rid, "model_id": mid,
                         "path": path, "seconds": round(secs, 1)})
        except Exception as exc:
            with contextlib.suppress(Exception):
                await reply({"type": "tts_err", "req_id": rid, "model_id": mid,
                             "error": repr(exc)})
            print(f"[tts] generate FAILED: {exc!r}", flush=True)

    async def handle_pack(self, msg: dict) -> dict:
        """#distributed-packing Inc 1b/3b: pack ONE shard-cache unit FOR the controller (offloads the
        slow per-layer pack off the controller + uses the fleet's idle CPUs). Fetch the unit's bf16
        from /weights (the SAME stream a load uses -> renamed 'model.*' dict), pack via the SHARED
        shards.pack_unit_tensors (so the result is BIT-IDENTICAL to a controller-local compile by
        construction), serialize, and POST it back to /pack_result. Supports DENSE + FUSED-MoE + (Inc
        3b) PER-EXPERT MoE: when the controller sets `fuse`, build the meta skeleton so
        pack_unit_tensors fuses per-expert experts.N.* -> fused 3D gate_up_proj/down_proj exactly as
        the cold load / local compile does. The controller sends the EXACT quant scope (lin2d/exp3d)."""
        import base64
        import shards
        import shard_compile   # code-split Inc 9: the shared packer moved (INT4_GROUP stays in shards)
        from safetensors.torch import load as st_load, save as st_save
        base = f"http://{self.args.controller}:{msg['controller_http_port']}"
        quant = msg.get("quant", "int4")
        gs = int(msg.get("group_size", shards.INT4_GROUP))
        _l, _e = msg.get("lin2d"), msg.get("exp3d")
        lin2d = set(_l) if _l is not None else None    # None -> pack_unit_tensors name-heuristic
        exp3d = set(_e) if _e is not None else None
        fuse = bool(msg.get("fuse"))                    # Inc 3b: per-expert MoE -> fuse against skeleton
        qd = {"model": msg["model_id"], "start": int(msg.get("start", 0)),
              "end": int(msg.get("end", 0)), "embed": int(bool(msg.get("embed", 0))),
              "head": int(bool(msg.get("head", 0))), "skip_experts": 0}
        url = f"{base}/weights?{urllib.parse.urlencode(qd)}"

        def _work() -> tuple[bytes, dict]:
            skel = self._pack_skeleton(base, msg["model_id"]) if fuse else None
            raw = st_load(_http_get(url))                       # {model.* : bf16}, same as compile's raw
            out_sd, mtensors = shard_compile.pack_unit_tensors(raw, lin2d, exp3d, skel, quant, gs)
            return st_save(out_sd), mtensors

        blob, mtensors = await asyncio.to_thread(_work)
        hdr = base64.b64encode(json.dumps(mtensors).encode()).decode()
        purl = (f"{base}/pack_result?req_id={urllib.parse.quote(str(msg['req_id']))}"
                f"&unit={urllib.parse.quote(str(msg['unit']))}"
                f"&model_id={urllib.parse.quote(str(msg['model_id']))}&quant={quant}")
        await asyncio.to_thread(_http_post, purl, blob, {"X-Manifest": hdr})
        return {"req_id": msg.get("req_id"), "unit": msg.get("unit"),
                "bytes": len(blob), "tensors": len(mtensors)}

    async def handle_unload(self, model_id: str | None = None) -> None:
        """Per-model unload when model_id is given; otherwise a FULL teardown of every model
        (what the controller sends today, and what the session does on disconnect)."""
        if model_id is not None:
            await self._unload_model(model_id)
            self._maybe_self_restart_if_stuck()
            return
        self._teardown_tp()
        if self._tp_thread is not None:
            with contextlib.suppress(Exception):
                await asyncio.to_thread(self._tp_thread.join, 5)
            self._tp_thread = None
        for w in self.next_writers.values():
            with contextlib.suppress(Exception):
                w.close()
        self.next_writers.clear()
        if self.data_server is not None:
            with contextlib.suppress(Exception):
                self.data_server.close()
            self.data_server = None
        for _sh in list(self.shards.values()):   # #vram-release-rocm: free GPU storages in place first
            self._release_shard_vram(_sh)
        self.shards.clear()
        self.assignments.clear()
        # Full teardown (incl. on controller disconnect) -> flush ALL staged multimodal embeds
        # so a fresh controller epoch (req_id from 0) can't pop a stale entry into a new prefill.
        self.pending_mm.clear()
        import gc
        gc.collect()
        _release_vram()   # return all freed VRAM to the pool (see _unload_model)
        self._cleanup_all_weight_tmps()
        _release_ram(trim_working_set=True)   # full teardown -> idle: trim heap back to the OS
        self._maybe_self_restart_if_stuck()   # if the OS still won't reclaim, restart for a clean slate


# ---- code-split Inc 8: model-build helpers relocated from client.py (VERBATIM) ----
# EmbeddingModel + _build_with_autodeps/_missing_pkgs_from_err land beside their only
# call site (WorkerLoadMixin), and the HF-local weight helpers beside shard_build's use.

def _missing_pkgs_from_err(exc: Exception) -> list[str]:
    """Best-effort extract pip package name(s) from an ImportError raised while BUILDING a model
    (esp. trust_remote_code modeling code, e.g. nomic-embed-text needing einops). Handles
    transformers' "Run `pip install X Y`" and "...not found in your environment: A, B" forms, plus
    plain "No module named 'X'". Returns ONLY safe package tokens so we never feed pip junk."""
    import re
    msg = str(exc)
    pkgs: list[str] = []
    m = re.search(r"pip install ([^\n`'\"]+)", msg)
    if m:
        pkgs = m.group(1).split()
    if not pkgs:
        m = re.search(r"not found in your environment:\s*([^\n.]+)", msg)
        if m:
            pkgs = [p.strip() for p in m.group(1).split(",")]
    if not pkgs:
        m = re.search(r"No module named ['\"]([A-Za-z0-9_][A-Za-z0-9_.\-]*)", msg)
        if m:
            pkgs = [m.group(1).split(".")[0]]
    return [p for p in pkgs if re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.\-]*", p or "")]


def _build_with_autodeps(build_fn, label: str = ""):
    """Run build_fn(); if it raises ImportError naming missing pip package(s), install them into
    THIS worker's env (sys.executable -m pip) and RETRY — so a model whose trust_remote_code needs
    a package the worker lacks (e.g. einops) self-heals ON LOAD instead of failing the whole load
    (#84). Bounded: each package is tried once and it gives up after a few rounds, so a genuinely
    broken import can't loop forever; a pip failure surfaces as a clear ImportError."""
    import subprocess
    tried: set = set()
    while True:
        try:
            return build_fn()
        except ImportError as exc:
            pkgs = [p for p in _missing_pkgs_from_err(exc) if p not in tried]
            if not pkgs or len(tried) >= 8:
                raise
            tried.update(pkgs)
            print(f"[deps] {label}: missing {pkgs} — pip-installing into worker env", flush=True)
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *pkgs])
                import importlib
                importlib.invalidate_caches()   # make the freshly-installed package importable now
            except Exception as pe:
                raise ImportError(f"auto-install of {pkgs} failed ({pe}); install it on this "
                                  f"worker manually") from exc


class EmbeddingModel:
    """Single-node sentence encoder (BERT-family). One forward -> masked mean-pool -> L2 norm.
    No KV cache, no pipeline, no lm_head. Stored in Worker.shards like a Shard but only needs
    loaded_params/loaded_bytes + unloadability — it never enters the decoder data path (it only
    receives kind:"embed" frames), so it can omit the Shard-only attrs (next_writers/layer
    ranges/has_head). torch is imported lazily (module-scope torch isn't guaranteed), so encode
    uses `with torch.inference_mode()` rather than the decorator form."""
    def __init__(self, model_dir, device, dtype):
        from transformers import AutoModel
        # nomic's custom config may reject _attn_implementation="eager"; retry without it.
        try:
            self.model = AutoModel.from_pretrained(
                model_dir, trust_remote_code=True, dtype=dtype,
                _attn_implementation="eager").eval()
        except ImportError:
            raise   # a MISSING dep (e.g. einops) -> let _build_with_autodeps install + retry the
            #         whole build (don't waste a 2nd no-eager attempt that hits the same ImportError)
        except Exception:
            self.model = AutoModel.from_pretrained(
                model_dir, trust_remote_code=True, dtype=dtype).eval()
        try:
            self.model.to(device)
        except Exception:
            device = "cpu"
            self.model.to("cpu")
        self.device = device
        self.loaded_params = sum(p.numel() for p in self.model.parameters())
        self.loaded_bytes = sum(p.numel() * p.element_size() for p in self.model.parameters())
        self.gpu_bytes = self.loaded_bytes if "cuda" in str(device) else 0

    def encode(self, input_ids, attention_mask):
        import torch
        with torch.inference_mode():
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            out = self.model(input_ids=input_ids, attention_mask=attention_mask)
            h = out.last_hidden_state                                   # [B,T,H]
            m = attention_mask.unsqueeze(-1).to(h.dtype)
            pooled = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)          # masked mean
            # L2-normalize, return float32 on CPU -> [B,H]
            return torch.nn.functional.normalize(pooled, p=2, dim=1).to(torch.float32).cpu()

def _weight_map(model_dir: str) -> dict[str, str]:
    index = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index):
        with open(index, encoding="utf-8") as fh:
            wm = json.load(fh)["weight_map"]
        return {name: os.path.join(model_dir, fn) for name, fn in wm.items()}
    single = os.path.join(model_dir, "model.safetensors")
    if os.path.exists(single):
        from safetensors import safe_open
        with safe_open(single, framework="pt") as fh:
            return {name: single for name in fh.keys()}
    raise FileNotFoundError(f"no safetensors found in {model_dir}")


def _load_tensors(names: list[str], weight_map: dict[str, str]) -> dict:
    from safetensors import safe_open
    by_file: dict[str, list[str]] = {}
    for n in names:
        by_file.setdefault(weight_map[n], []).append(n)
    out = {}
    for fn, ns in by_file.items():
        with safe_open(fn, framework="pt") as fh:
            for n in ns:
                out[n] = fh.get_tensor(n)
    return out


def _assemble_sd(tensors: dict, start: int, end: int, has_embed: bool,
                 has_head: bool, tied: bool) -> dict:
    """Map raw tensors to the state-dict keys load_state_dict expects, resolving
    the tied head to a (cloned) copy of the embedding matrix."""
    sd: dict = {}
    if has_embed:
        sd["model.embed_tokens.weight"] = tensors["model.embed_tokens.weight"]
    for i in range(start, end):
        for n in (x for x in tensors if x.startswith(f"model.layers.{i}.")):
            sd[n] = tensors[n]
    if has_head:
        sd["model.norm.weight"] = tensors["model.norm.weight"]
        if tied:
            sd["lm_head.weight"] = tensors["model.embed_tokens.weight"].clone()
        else:
            sd["lm_head.weight"] = tensors["lm_head.weight"]
    return sd


# ---------------------------------------------------------------------------
# Worker — owns the current stage's shard + data-plane wiring
# ---------------------------------------------------------------------------
