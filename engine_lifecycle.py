"""EngineLifecycleMixin: relocated Engine methods (m4c152 code-split). BODIES ARE BYTE-IDENTICAL
to the originals in server.py; their module globals (registry, log_activity, ModelSpec,
ENGINE_CONFIG …) are injected at startup by state.bind() — see state.py. Composed back
into the live class via ``class Engine(EngineLifecycleMixin, …)`` in server.py, so ``self.*`` resolves
across all mixins by MRO. Controller-only leaf module; in EXTRA_UPDATE_FILES.
"""
from __future__ import annotations


class EngineLifecycleMixin:

    # -- data listener: receives logits frames from the last stage --
    async def ensure_data_listener(self) -> None:
        if self.data_server is None:
            # Resilient accept loop: a transient per-accept OSError (WinError 64
            # during a reconnect storm) is logged and skipped, never killing the
            # data-plane listener. _on_data's (reader, writer) signature unchanged.
            self.data_server = await _resilient_serve(
                ARGS.host, ARGS.data_port, self._on_data, "data")
            print(f"[*] data plane listening on {ARGS.host}:{ARGS.data_port}")

    async def _on_data(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        served: set = set()   # model_ids whose logits came back on THIS connection
        _set_keepalive(writer.get_extra_info("socket"))   # survive the load->generate idle gap
        try:
            while True:
                try:
                    hdr, raw, wire = await _read_frame(reader)
                except (asyncio.IncompleteReadError, ConnectionError):
                    break  # connection dropped; pending futures failed in finally
                # Frames carry model_id (= HF target_id); attribute net to THAT model's head
                # node (correct with several models streaming back concurrently).
                mid = hdr.get("model_id")
                if mid is not None:
                    served.add(mid)
                hm = next((mm for mm in self.models.values()
                           if mm.target_id == mid), None)
                net_account(self._head_id(hm), from_node=wire)  # head -> controller
                rid = hdr.get("req_id")
                fut = self.pending.pop(rid, None) if rid is not None else None
                if rid is not None:
                    self.pending_model.pop(rid, None)
                    self.pending_friendly.pop(rid, None)   # #5 keep replica map in lockstep
                if hdr.get("kind") == "error":
                    if fut and not fut.done():
                        fut.set_exception(RuntimeError(hdr.get("error", "stage error")))
                    continue
                if fut and not fut.done():
                    try:
                        if hdr.get("kind") == "ntensor":
                            # #ntensor-manifest: N-tensor manifest return frame — arrives ONLY
                            # when this controller requested it (request-header 'ntensor' flag,
                            # cap-gated in engine_gen), so _unpack_ntensor is present by
                            # construction (a freak mismatch raises -> the except below fails
                            # the future, same as any malformed frame). raw = [count:u8]
                            # [count x (kind:u8, nbytes:u32 BE)][payloads]; dtype/shape metas
                            # ride hdr['tensors'] positionally. Reconstruct the LEGACY result
                            # shape by kind so _send's callers stay format-agnostic: logits
                            # alone -> tensor; logits+hidden -> (logits, hidden) tuple; any
                            # other combination -> the raw [(kind, tensor), ...] list for
                            # kind-aware consumers — LIVE since #logits-diet: NT_TOKEN_IDS
                            # (greedy/spec argmax ids) and NT_TOPK_VALS+NT_TOPK_IDX (sampled
                            # top-K candidates) arrive here and are consumed in engine_gen's
                            # _decode_plain/_decode_spec, which detect the list reply type.
                            parts = _unpack_ntensor(hdr.get("tensors") or [], raw)
                            _by = dict(parts)
                            if set(_by) == {NT_LOGITS}:
                                fut.set_result(_by[NT_LOGITS])
                            elif set(_by) == {NT_LOGITS, NT_HIDDEN}:
                                fut.set_result((_by[NT_LOGITS], _by[NT_HIDDEN]))
                            else:
                                fut.set_result(parts)
                        elif hdr.get("hid_meta") is not None:
                            # #P6 speech: two-tensor result frame = logits ++ post-norm hidden.
                            ln = int(hdr["logits_nbytes"])
                            logits = _unpack_tensor(hdr, raw[:ln])
                            hidden = _unpack_tensor(hdr["hid_meta"], raw[ln:])
                            fut.set_result((logits, hidden))
                        else:
                            fut.set_result(_unpack_tensor(hdr, raw))
                    except Exception as exc:  # malformed/short frame
                        fut.set_exception(exc)
        except asyncio.CancelledError:
            pass
        except Exception as exc:  # pragma: no cover
            print(f"[data] listener error: {exc!r}")
        finally:
            # A dead data connection dooms only the requests for the model(s) THIS connection was
            # serving — fail just those (not every model's), so an unrelated model's concurrent
            # generations survive one head's hiccup.
            # #stage0-stale-reconnect: a head node now FRESHENS its return conn at each prefill (it
            # drops the possibly-idle-half-open socket + reconnects), so a return-conn close is
            # usually a RECONNECT, not a death — and the request's logits are about to arrive on the
            # NEW conn. Dooming synchronously here killed exactly that request ("data connection
            # closed" right after an idle gap). So GRACE it: snapshot the in-flight reqs this conn
            # served, then doom only those STILL pending after REQUEUE_GRACE_S. A reconnect delivers
            # within the head stage's compute (well under the grace) -> future resolves -> skipped;
            # a genuinely dead head still fails fast (grace << gen-stall watchdog / GEN_TIMEOUT).
            # (#pipefill: a chunked prefill has SEVERAL pending rids for one prompt — they all
            # land in this snapshot and ride the same progress-aware grace below; dooming any
            # one of them fails the burst's fail-fast gather in _send_prefill immediately.)
            at_close = [rid for rid, f in self.pending.items()
                        if not f.done() and self.pending_model.get(rid) in served]
            with contextlib.suppress(Exception):
                writer.close()
            if at_close:
                async def _grace_doom(rids):
                    # #70b-long-prefill: the freshened return conn only re-delivers AFTER the
                    # in-flight forward completes — on a big model that is a multi-minute prefill
                    # (one 2048-token chunk through 70B int4 on an APU outlasts any flat grace),
                    # so a flat sleep doomed every long prefill right after the head's reconnect.
                    # Instead, after each grace slice keep granting grace while the model that
                    # owns a still-pending req reports ADVANCING per-layer forward progress
                    # (fwd_progress_ts — heartbeat-fed, credited only while the rid is pending —
                    # plus last_send_ts for the first-stamp gap; the same liveness basis as
                    # _send's adaptive prefill wait). A genuinely dead head goes progress-quiet
                    # and still fails in grace+quiet; the hard ceiling backstops a stuck feed.
                    _quiet_s = 120.0   # matches _send's _PROG_QUIET_S (same signal, same tolerance)
                    _hard_s = max(3600.0, GEN_TIMEOUT_S)
                    _t0 = time.time()
                    while True:
                        await asyncio.sleep(REQUEUE_GRACE_S)
                        live = [rid for rid in rids
                                if rid in self.pending and not self.pending[rid].done()]
                        if not live:
                            return   # everything re-delivered (or already failed elsewhere)
                        if (time.time() - _t0) >= _hard_s:
                            break
                        mids = {self.pending_model.get(rid) for rid in live}
                        _fp = max((max(getattr(mm, "fwd_progress_ts", 0.0) or 0.0,
                                       getattr(mm, "last_send_ts", 0.0) or 0.0)
                                   for mm in self.models.values() if mm.target_id in mids),
                                  default=0.0)
                        if (time.time() - _fp) > _quiet_s:
                            break
                    for rid in rids:
                        fut = self.pending.get(rid)
                        if fut is not None and not fut.done():
                            self.pending.pop(rid, None)
                            self.pending_model.pop(rid, None)
                            self.pending_friendly.pop(rid, None)   # #5 keep replica map in lockstep
                            fut.set_exception(ConnectionError(
                                "data connection closed (head did not re-deliver within grace)"))
                with contextlib.suppress(Exception):
                    asyncio.create_task(_grace_doom(at_close))

    async def _connect_retry(self, host: str, port: int, timeout: float = 30) -> asyncio.StreamWriter:
        deadline = time.time() + timeout
        while True:
            try:
                _r, w = await asyncio.open_connection(host, port)
                _set_keepalive(w.get_extra_info("socket"))   # stage0 conn idles load->generate; keep alive
                return w
            except OSError:
                if time.time() > deadline:
                    raise
                await asyncio.sleep(0.4)

    def next_req(self) -> int:
        self.req_counter += 1
        return self.req_counter

    def _stage0_id(self, model: Optional[LoadedModel]) -> Optional[str]:
        """Node the controller pushes frames TO (first stage of `model`); None if absent."""
        return model.stage_node_ids[0] if model and model.stage_node_ids else None

    def _head_id(self, model: Optional[LoadedModel]) -> Optional[str]:
        """Node the controller receives logits FROM (last stage of `model`); None if absent."""
        return model.stage_node_ids[-1] if model and model.stage_node_ids else None

    def invalidate(self, reason: str) -> None:
        """A node in the active pipeline dropped — tear loaded model(s) down. (Inc 1: all;
        Inc 3 will scope this to only the models that used the dropped node.)"""
        if self.models:
            print(f"[!] pipeline invalidated: {reason} (reload required)")
        for m in self.models.values():
            if m.stage0_writer is not None:
                with contextlib.suppress(Exception):
                    m.stage0_writer.close()
            self._unload_draft(m)   # free this model's controller-local draft
        if getattr(self, "_mtp_heads", None):
            self._mtp_heads.clear()   # #91 free all controller-resident MTP heads
            _free_mtp_cuda()          # the GPU head's VRAM isn't released until empty_cache()
        self.models.clear()
        REQUEST_HISTORY.clear()   # #ctx-history: history is only meaningful while a model is resident
        for fut in self.pending.values():
            if not fut.done():
                fut.set_exception(RuntimeError("pipeline invalidated"))
        self.pending.clear()
        self.pending_model.clear()
        self.pending_friendly.clear()   # #5 keep replica map in lockstep
        _release_ram()              # actually hand the freed draft RAM back (gc cycles + OS)

    def invalidate_model(self, friendly: str, reason: str) -> None:
        """Tear down ONE resident model (a node it used dropped). Other models keep running.
        With disjoint placement (Inc 3a) a dropped node serves exactly one model; that model's
        in-flight generation (if any) fails via the broken data connection, so this just drops
        the model's resident state and frees its controller-local draft."""
        m = self.models.pop(friendly, None)
        REQUEST_HISTORY.pop(friendly, None)   # #ctx-history
        if m is None:
            return
        print(f"[!] {friendly} invalidated: {reason} (reload required)")
        # #5 replica-precise recovery: a node that LEFT/was reaped serves exactly THIS replica (the
        # caller filtered by node_id in m.stage_node_ids). Its in-flight generate() is blocked in
        # _send's wait_for(fut, GEN_TIMEOUT ~600s) with no upstream error frame, so fail this
        # replica's leaked pending futures NOW. Match on pending_friendly (the routed replica's
        # UNIQUE key), NOT pending_model (target_id is SHARED across replicas), so a healthy sibling
        # replica's in-flight request is left untouched.
        for _rid in [r for r, fr in list(self.pending_friendly.items()) if fr == friendly]:
            _f = self.pending.pop(_rid, None)
            self.pending_model.pop(_rid, None)
            self.pending_friendly.pop(_rid, None)
            if _f is not None and not _f.done():
                with contextlib.suppress(Exception):
                    _f.set_exception(ConnectionError(f"{friendly} replica invalidated: {reason}"))
        hosts = [s.hostname for s in m.plan.stages]
        # Non-graceful eviction (a node holding a shard died) — surface WHY on the dashboard.
        # This path used to be console-only, so a model that vanished on a node OOM/drop just
        # disappeared from the UI with no explanation.
        log_activity(f"DROPPED {friendly}: {reason} — lost shard host(s) " + ", ".join(hosts))
        record_unload(friendly, reason, hosts=hosts, kind="node-loss")
        if m.stage0_writer is not None:
            with contextlib.suppress(Exception):
                m.stage0_writer.close()
        for nid in m.stage_node_ids:
            # Tell the SURVIVING workers to drop this model's shard so a reaped-then-reconnected
            # node (or the other stages) don't leak the old shard into the next reload — which then
            # mis-plans against stale free memory. The dropped node has no link; the others free it. (#47)
            link = self.links.get(nid)
            if link is not None:
                with contextlib.suppress(Exception):
                    asyncio.create_task(link.send({"type": "unload", "model_id": m.target_id}))
            nd = registry._nodes.get(nid)
            if nd:
                nd.clear_assignment()
        self._unload_draft(m)
        _release_ram()

    async def recover_node_restart(self, stale: "Node", new_node_id: str) -> int:
        """A worker RE-REGISTERED (it restarted): `stale` is its OLD node entry, still listed
        with whatever model stage(s) it held, while `new_node_id` is its fresh registration. The
        restart wiped that worker's shard + KV and dropped its data connection, but the old link
        may not have errored yet (idle Windows sockets go half-open silently), so the controller
        still thinks the model is loaded — the next generate would route to the dead stage and hang
        out the full GEN_TIMEOUT (~600s) instead of failing.

        Recovery (deliberately the CONSERVATIVE minimum, not an auto-reload): mark every resident
        model that had a stage on the stale node BROKEN and drop it, so a generate fails FAST with
        a clear "worker restarted — reload" message. invalidate_model already (a) closes the
        pipeline conn, (b) fails this model's in-flight requests, (c) tells the SURVIVING stages to
        unload their slice (so they don't leak the old shard into a reload), (d) logs a DROPPED line
        + records the unload for the dashboard, and (e) clears node assignments. We do NOT auto-
        re-stream the shards here: a flapping worker would thrash the whole pipeline, and a wrong
        recovery that tears down a healthy model is worse than the status quo. Returns the count of
        models recovered."""
        affected = [fr for fr, m in self.models.items()
                    if stale.node_id in m.stage_node_ids]
        for fr in affected:
            self.invalidate_model(
                fr, f"worker {stale.hostname} restarted (re-registered as {new_node_id}) — reload")
        # Drop the now-superseded control link to the stale node id so a later unload/heartbeat
        # can't fire on a dead writer (the fresh registration installed its own link under the new
        # id). The stale Node entry itself is removed by the caller.
        old_link = self.links.pop(stale.node_id, None)
        if old_link is not None:
            with contextlib.suppress(Exception):
                old_link.writer.close()
        return len(affected)

    def _lru_friendly(self) -> Optional[str]:
        """Least-recently-used resident model (by last_used), or None if none loaded."""
        if not self.models:
            return None
        return min(self.models.items(), key=lambda kv: kv[1].last_used)[0]

    def _lru_evictable(self) -> Optional[str]:
        """LRU resident model with NO active/queued requests (safe to auto-unload), or None
        if every resident model is busy serving. Never picks an actively-serving model, the
        model currently being loaded (_no_evict_base), or a #no-unload-pinned model — the
        absolute do-not-auto-unload veto: such a model is never evicted, so a new load that
        can't otherwise fit FAILS rather than displacing it (that's the flag 'winning')."""
        no_unload = set(ENGINE_CONFIG.get("no_unload_models") or {})
        idle = [(fr, m) for fr, m in self.models.items()
                if m.active == 0 and m.queued == 0
                and (m.base or m.friendly) != self._no_evict_base
                and (m.base or m.friendly) not in no_unload]
        if not idle:
            return None
        return min(idle, key=lambda kv: kv[1].last_used)[0]

    # ---- data-parallel replication (#39) -------------------------------------------------
    def replicas_of(self, base: str) -> list["LoadedModel"]:
        """All resident replicas of a user-facing model `base`, ordered by replica_idx.
        A non-replicated model (base == friendly) returns its single LoadedModel."""
        rs = [m for m in self.models.values() if (m.base or m.friendly) == base]
        rs.sort(key=lambda m: m.replica_idx)
        return rs

    def replica_count(self, base: str) -> int:
        """Number of concurrent decode slots for `base` (one per resident replica; >=1)."""
        return max(1, len(self.replicas_of(base)))

    def _pick_replica(self, base: str) -> Optional["LoadedModel"]:
        """Route a request for `base` to the least-loaded resident replica (active+queued),
        round-robin on ties — the data-parallel dispatch. Skips replicas whose pipeline
        connection is down; falls back to a direct key hit for non-replicated models."""
        rs = [r for r in self.replicas_of(base) if r.stage0_writer is not None]
        if not rs:
            return self.models.get(base)
        best = min(r.active + r.queued for r in rs)
        cands = [r for r in rs if r.active + r.queued == best]
        if len(cands) == 1:
            return cands[0]
        i = self._rr.get(base, 0) % len(cands)
        self._rr[base] = i + 1
        return cands[i]

    async def unload(self, friendly: Optional[str] = None, force: bool = False) -> None:
        async with self.lock:
            if friendly is None:
                # Blanket teardown sends EVERY worker a model-less unload (drops ALL shards) — including
                # any in-flight load's half-built ones. The decision is made HERE, under self.lock, so
                # it's atomic with a load registering its card at its first line: if a load is in flight
                # we refuse rather than nuke it mid-stream (the unload-all TOCTOU — a check at the HTTP
                # layer races the load's released-lock gather window). force= overrides (shutdown).
                if self.loadings and not force:
                    raise LoadInProgressError(sorted(self.loadings))
                await self._unload_locked("manual unload (all)")
            else:
                # Drop ALL replicas of this user-facing model (friendly, friendly#1, ...). Per-model
                # unload is always allowed — even during another model's load (it only frees that model;
                # it never touches the loading model's shards), which is the whole point of #parallel-load.
                keys = [m.friendly for m in self.replicas_of(friendly)] or [friendly]
                for k in keys:
                    await self._unload_model_locked(k, "manual unload")

    async def _unload_model_locked(self, friendly: str, reason: str) -> None:
        """Evict ONE resident model: tell the nodes holding its stages to drop just that
        shard (worker handles {type:unload,model_id}), clear their assignment, and remove it
        from the registry. Other resident models keep running. Caller holds self.lock."""
        m = self.models.get(friendly)
        if m is None:
            return
        log_activity(f"unload {friendly} ({reason}) — reclaiming "
                     + ", ".join(f"{s.hostname} ~{s.est_gb:.1f}GB" for s in m.plan.stages))
        record_unload(friendly, reason, hosts=[s.hostname for s in m.plan.stages])
        loop = asyncio.get_event_loop()
        for nid in m.stage_node_ids:
            link = self.links.get(nid)
            if link:
                fut = loop.create_future()
                link.pending_unloads[m.target_id] = fut   # #1: key by model (worker echoes model_id)
                with contextlib.suppress(Exception):
                    await link.send({"type": "unload", "model_id": m.target_id})
                    await asyncio.wait_for(fut, timeout=10)  # confirm teardown
            nd = registry._nodes.get(nid)
            if nd:
                nd.clear_assignment()
        if m.stage0_writer is not None:
            with contextlib.suppress(Exception):
                m.stage0_writer.close()
        self._unload_draft(m)        # free this model's controller-local draft
        if getattr(self, "_mtp_heads", None) and self._mtp_heads.pop(friendly, None) is not None:
            _free_mtp_cuda()          # #91 release the GPU head's VRAM (empty_cache); else it leaks
        self.models.pop(friendly, None)
        REQUEST_HISTORY.pop(friendly, None)   # #ctx-history
        _release_ram()
        print(f"[unload] evicted {friendly} ({reason})")

    async def _unload_locked(self, reason: str) -> None:
        """Tear down the current model on every worker and confirm teardown. Caller holds
        self.lock. Unloads ALL connected workers (not just the loaded stages) so any
        orphaned/stale shard is dropped too."""
        # Capture residents first (invalidate clears them) so we can report the RAM reclaimed.
        freed = [(m.friendly, [(s.hostname, s.est_gb) for s in m.plan.stages])
                 for m in self.models.values()]
        loop = asyncio.get_event_loop()
        for nid in list(self.links):
            link = self.links.get(nid)
            if link:
                fut = loop.create_future()
                link.pending_unloads[None] = fut   # #1: unload-all sends no model_id -> worker echoes None
                with contextlib.suppress(Exception):
                    await link.send({"type": "unload"})
                    await asyncio.wait_for(fut, timeout=10)  # confirm teardown
        for n in registry._nodes.values():
            n.clear_assignment()
        self.invalidate(reason)
        # Workers have acked (shard dropped + gc'd + mmap released) -> report what was reclaimed.
        for friendly, hosts in freed:
            log_activity(f"unloaded {friendly} ({reason}) — reclaimed "
                         + ", ".join(f"{h} ~{gb:.1f}GB" for h, gb in hosts))
            record_unload(friendly, reason, hosts=[h for h, _gb in hosts])
        if not freed:
            log_activity(f"unload-all ({reason}) — no resident models; cleared any orphan shards")
