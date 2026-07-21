"""WorkerNetMixin: relocated Worker methods (m4c153 code-split). BODIES BYTE-IDENTICAL to the
originals in client.py; module globals injected at startup by state.bind() — see state.py.
Composed via ``class Worker(WorkerNetMixin, …)`` so self.* resolves across mixins by MRO. Worker-side
leaf module; in client.py EXTRA_UPDATE_FILES.
"""
from __future__ import annotations


class WorkerNetMixin:

    async def _connect_next(self, host: str, port: int) -> asyncio.StreamWriter:
        deadline = time.time() + 30
        while True:
            try:
                # Bind the chosen LAN source so outbound activations to the next
                # pipeline stage ride the fast NIC too (not just inbound traffic).
                _r, w = await asyncio.open_connection(host, port, local_addr=_local_addr())
                _set_keepalive(w.get_extra_info("socket"))   # survive the load->generate idle gap
                return w
            except OSError:
                if time.time() > deadline:
                    raise
                await asyncio.sleep(0.4)

    async def _reconnect_next(self, model_id: str) -> asyncio.StreamWriter:
        """Re-dial this model's NEXT-hop data connection from its stored load assignment.
        The pipeline next-hop conn is opened at LOAD and then sits IDLE until the first
        generate — often many minutes later. Windows aborts an idle socket (ConnectionReset
        / WinError 10053 'software caused connection abort'), so the first send after the gap
        finds it dead. Left unhandled the stage's write fails, the error is swallowed (it was
        being reported down the SAME dead hop), and the controller just waits out GEN_TIMEOUT
        — the observed 'loads then breaks' hang. Reconnecting on a write failure self-heals
        the pipeline. host/port come from the saved load message (assignments[model_id])."""
        a = self.assignments.get(model_id)
        if a is None:
            raise RuntimeError(f"no load assignment for model_id={model_id!r}; can't reconnect next hop")
        old = self.next_writers.pop(model_id, None)
        if old is not None:
            with contextlib.suppress(Exception):
                old.close()
        next_host = a.get("next_host") or self.args.controller   # last stage -> controller
        next_port = a["next_port"]
        w = await self._connect_next(next_host, next_port)
        self.next_writers[model_id] = w
        self.next_peer[model_id] = "controller" if not a.get("next_host") else str(next_host)
        print(f"[data] reconnected next hop for {model_id} -> "
              f"{self.next_peer[model_id]} ({next_host}:{next_port})", flush=True)
        return w

    def _freshen_next(self, model_id: str) -> None:
        """#stage0-stale-reconnect: at a PREFILL, drop a next-hop conn that's been idle past
        STAGE_STALE_S so the upcoming _send_next lazy-reconnects it FRESH. A reconnect-on-FAILURE
        (in _send_next) can't catch a SILENTLY half-open idle socket — the write succeeds but the
        bytes never arrive, so the downstream stage / controller never sees the frame and just
        waits out GEN_TIMEOUT. Proactively dropping the stale socket here is the cure. Caller gates
        this on reset=True (prefill) only, so a slow decode's inter-token gaps never trigger it."""
        last = self._next_last_send.get(model_id, 0.0)
        if model_id in self.next_writers and (time.time() - last) > STAGE_STALE_S:
            w = self.next_writers.pop(model_id, None)
            if w is not None:
                with contextlib.suppress(Exception):
                    w.close()
            print(f"[data] dropping idle next-hop for {model_id} "
                  f"(stale {time.time() - last:.0f}s) -> will reconnect fresh", flush=True)

    async def _send_next(self, model_id: str, hdr: dict, raw: bytes) -> int:
        """Send one frame to this model's next hop, RECONNECTING once if the (possibly
        idle-dead) connection fails. This is what makes a distributed generation survive the
        load->first-generate idle gap (and a transient next-hop blip). Returns bytes sent;
        raises only if the next hop is genuinely unreachable after a fresh reconnect."""
        nxt = self.next_writers.get(model_id)
        if nxt is not None:
            try:
                _nb = await _write_frame(nxt, hdr, raw)
                self._next_last_send[model_id] = time.time()   # #stage0-stale-reconnect freshness clock
                return _nb
            except (ConnectionError, OSError, asyncio.IncompleteReadError) as exc:
                print(f"[data] next-hop send for {model_id} failed ({exc!r}); "
                      f"reconnecting + retrying once", flush=True)
        nxt = await self._reconnect_next(model_id)   # no writer, or the send just died
        _nb = await _write_frame(nxt, hdr, raw)
        self._next_last_send[model_id] = time.time()
        return _nb

    def _run_stage(self, model_id, x, cache_start, reset, all_logits, inject=None,
                   position_ids=None, capture_hidden=False, capture_pre_norm=False,
                   bidir_spans=None):
        # TP rank 0 drives the group: broadcast this forward's input to the peers (who run
        # their sharded forward in lockstep), then run ours, all-reducing via the mesh hooks.
        if self._tp is not None and self._tp.rank == 0 and model_id == self._tp_model_id:
            import pickle
            # #tp-mesh-keepalive: hold the mesh lock across the WHOLE forward (broadcast + every
            # hook all-reduce) so the idle keepalive ping can never interleave + corrupt the byte
            # stream. Stamp the warmth clock so the keepalive thread skips a ping right after a real
            # forward (a busy model keeps its own mesh warm).
            with self._tp_lock:
                self._tp_last_fwd = time.time()
                # include inject + position_ids + bidir_spans so peers (replicated embeddings +
                # rotary + per-stage masks) build an identical mask -> matching all-reduce
                self._tp.broadcast(pickle.dumps(
                    (x.detach().to("cpu"), int(cache_start), bool(reset), bool(all_logits),
                     inject, position_ids, bidir_spans)))
                return self.shards[model_id].forward(x, cache_start, reset, all_logits, inject,
                                                     position_ids, capture_hidden, capture_pre_norm,
                                                     bidir_spans)
        return self.shards[model_id].forward(x, cache_start, reset, all_logits, inject,
                                             position_ids, capture_hidden, capture_pre_norm,
                                             bidir_spans)

    async def _data_inbound(self, reader: asyncio.StreamReader,
                            writer: asyncio.StreamWriter) -> None:
        # Identify who's on the OTHER end of this inbound data conn for per-peer accounting:
        # the controller (-> stage 0) or the previous-stage worker's IP (-> mid/last stage).
        _pn = writer.get_extra_info("peername")
        peer_in = "controller" if (_pn and _pn[0] == self.args.controller) else (
            str(_pn[0]) if _pn else "?")
        _set_keepalive(writer.get_extra_info("socket"))   # survive the load->generate idle gap

        try:
            while True:
                hdr, raw, _nb = await _read_frame(reader)
                _net_peer(peer_in, rx=_nb)
                # The data port is shared across models; route every frame by model_id.
                model_id = hdr.get("model_id")
                shard = self.shards.get(model_id)
                nxt = self.next_writers.get(model_id)
                if hdr.get("kind") == "error":  # propagate upstream error downstream
                    with contextlib.suppress(Exception):
                        await self._send_next(model_id, hdr, b"")   # reconnect-safe (reach the controller)
                    continue
                if hdr.get("kind") == "crop":  # speculative-decode KV rollback
                    if shard is not None:
                        shard.crop(int(hdr.get("cache_position", 0)))
                    with contextlib.suppress(Exception):  # propagate down the chain
                        if nxt is not None:
                            await _write_frame(nxt, hdr, b"")
                    continue
                if hdr.get("kind") == "mm":   # #22 inc 3: stage multimodal embeds for the
                    # next prefill of this (model_id, req_id). Stage-0 only; NOT forwarded down.
                    if shard is not None:
                        emb = _unpack_tensor(hdr, raw)
                        # #mm-pairing: stamp a staged-at time (3rd slot, stripped at claim) and sweep
                        # companions never claimed — a reclaimed/cancelled gen leaks its entry, and a
                        # controller RESTART resets req_id numbering, so a leaked entry could otherwise
                        # be claimed by an UNRELATED future request (stale image embeds spliced into
                        # the wrong prompt). 10 min >> any legit mm->ids gap (same two-frame flush).
                        self.pending_mm[(model_id, hdr.get("req_id"))] = (
                            hdr.get("positions") or [], emb, time.time())
                        _mm_cut = time.time() - 600.0
                        for _mk, _mv in list(self.pending_mm.items()):
                            if len(_mv) > 2 and _mv[2] < _mm_cut:
                                self.pending_mm.pop(_mk, None)
                    continue
                if hdr.get("kind") == "embed":   # encoder: ONE two-tensor frame (ids ++ mask) ->
                    # masked mean-pool + L2-norm vecs, replied straight to the controller. Mirrors
                    # the server's two-tensor hid_meta packing (primary meta = ids, mask_meta = mask).
                    try:
                        em = shard
                        if em is None:
                            raise RuntimeError(f"no embedding model for model_id={model_id!r} on this node")
                        ids_nbytes = int(hdr["ids_nbytes"])
                        ids = _unpack_tensor(hdr, raw[:ids_nbytes])
                        mask = _unpack_tensor(hdr["mask_meta"], raw[ids_nbytes:])
                        vecs = await asyncio.to_thread(em.encode, ids, mask)
                        vmeta, vraw = _pack_tensor(vecs)
                        ohdr = {"req_id": hdr.get("req_id"), "model_id": model_id,
                                "kind": "embedding", **vmeta}
                        _tx = await self._send_next(model_id, ohdr, vraw)
                        _net_peer(self.next_peer.get(model_id, "?"), tx=_tx)
                    except Exception as exc:   # mirror the stage-error path: tell the controller
                        import traceback
                        tb = traceback.format_exc()
                        print(f"[data] embed error: {exc!r}\n{tb}")
                        frames = [ln.strip() for ln in tb.splitlines() if ln.strip().startswith("File ")]
                        with contextlib.suppress(Exception):
                            await self._send_next(model_id, {
                                "req_id": hdr.get("req_id"), "model_id": model_id, "kind": "error",
                                "error": f"{exc!r} | " + " <- ".join(frames[-3:])}, b"")
                    continue
                cache_start = int(hdr.get("cache_position", 0))
                reset = bool(hdr.get("reset", True))
                all_logits = bool(hdr.get("all_logits", False))
                if reset:   # #stage0-stale-reconnect: new generation -> drop a stale (idle) next hop
                    self._freshen_next(model_id)   # so this prefill's forward rides a fresh socket
                try:
                    if shard is None:
                        raise RuntimeError(f"no shard for model_id={model_id!r} on this node")
                    x = _unpack_tensor(hdr, raw)
                    # #mm-pairing: the ids frame now DECLARES its mm companion (hdr['mm'], m4c190+
                    # controller). Declared -> the companion MUST be staged (a lost/mispaired frame
                    # would silently run the vision prefill UNSPLICED — the model then hallucinates
                    # on raw placeholder embeddings). Undeclared -> NEVER claim (protects against a
                    # leaked companion + controller-restart req_id collision splicing stale image
                    # embeds into an unrelated prompt). Pre-m4c190 controller (no 'mm' key): legacy
                    # unconditional claim, unchanged behavior.
                    if "mm" in hdr:
                        _mm3 = (self.pending_mm.pop((model_id, hdr.get("req_id")), None)
                                if hdr.get("mm") else None)
                        if hdr.get("mm") and _mm3 is None:
                            raise RuntimeError(
                                f"mm companion frame missing for req {hdr.get('req_id')} "
                                f"({model_id}) — refusing to run the vision prefill unspliced")
                    else:
                        _mm3 = self.pending_mm.pop((model_id, hdr.get("req_id")), None)
                    inject = _mm3[:2] if _mm3 is not None else None
                    # #stage0-dtype-guard: a first-stage (has_embed) frame must carry token IDS; a
                    # floating-point x here is a mispaired mm/ids or a misrouted hidden frame (the
                    # 2026-07-09/10 bf16-into-F.embedding wedge class on beast — each occurrence
                    # burned a 240s gen-stall reclaim). Classify it AT THE DOOR with a self-evident
                    # error instead of letting F.embedding throw its cryptic dtype message.
                    if (shard.has_embed and bool(hdr.get("reset", True))
                            and getattr(x, "is_floating_point", None) is not None
                            and x.is_floating_point()):
                        raise RuntimeError(
                            f"stage0 for {model_id} received a floating-point "
                            f"'{hdr.get('kind')}' frame (dtype={hdr.get('dtype')} "
                            f"shape={hdr.get('shape')}) where token ids were expected — "
                            f"mispaired mm/ids or misrouted hidden frame")
                    # #22 inc 4: 3D mRoPE positions ride the frame header (small list); every
                    # stage uses them for its rotary, so propagate them to the next stage too.
                    position_ids = hdr.get("position_ids")
                    # #gemma4-bidir: image-span runs for bidirectional vision attention ride the
                    # header (small JSON list of [start,end]); every stage rebuilds its own mask,
                    # so propagate them downstream exactly like position_ids.
                    bidir_spans = hdr.get("bidir_spans")
                    # #P6 speech: capture thinker hidden states for the talker. The flag rides
                    # the header down the chain so the LAST stage (has_head) returns the
                    # post-norm hidden alongside the logits in a two-tensor result frame.
                    capture_hidden = bool(hdr.get("capture_hidden", False))
                    # #91 MTP: capture_pre_norm rides the same chain as capture_hidden but the head
                    # returns the PRE-final-norm trunk hidden (what the MTP head consumes).
                    capture_pre_norm = bool(hdr.get("capture_pre_norm", False))
                    # #ntensor-manifest: the controller OPTED IN to the N-tensor manifest return
                    # frame for this request (it only sets the flag when every node in the chain
                    # advertised the 'ntensor' cap — #wire-caps). Old workers ignore this unknown
                    # header key by construction (this parser reads keys via hdr.get) and reply in
                    # the legacy format, which the controller always still accepts — so a stale
                    # gate is harmless in both directions.
                    ntensor_req = bool(hdr.get("ntensor", False))
                    # #prefill-progress: hand the frame's req_id to the shard so its per-layer
                    # progress heartbeat is ATTRIBUTED to this request (shard_forward copies it to
                    # _fwd_cur_rid once it OWNS _fwd_lock, so an orphaned forward keeps its own
                    # rid and can never shield a newer gen from the controller's watchdog).
                    shard._fwd_next_rid = hdr.get("req_id")
                    out = await asyncio.to_thread(self._run_stage, model_id, x, cache_start,
                                                  reset, all_logits, inject, position_ids,
                                                  capture_hidden, capture_pre_norm, bidir_spans)
                    kind = "logits" if shard.has_head else "hidden"
                    if ntensor_req and shard.has_head and _pack_ntensor is not None:
                        # #ntensor-manifest: head stage, controller requested the manifest frame.
                        # raw = [count:u8][count x (kind:u8, nbytes:u32 BE)][payloads...]; the
                        # per-tensor dtype/shape metas ride the JSON header ('tensors', positional
                        # — the same self-describing meta as every legacy frame). Kinds live in
                        # wire.py: 0=logits, 1=hidden; 2=token_ids / 3=topk_vals / 4=topk_idx are
                        # RESERVED for the logits-diet (next stage). _pack_ntensor is None only
                        # when wire.py is stale (self-update convergence) — then this worker never
                        # advertised 'ntensor' and the flag can't legitimately be set; falling
                        # through keeps the legacy format either way.
                        if (capture_hidden or capture_pre_norm) and isinstance(out, tuple):
                            _parts = [(NT_LOGITS, out[0]), (NT_HIDDEN, out[1])]
                        else:
                            _parts = [(NT_LOGITS, out)]
                        tmeta, oraw = _pack_ntensor(_parts)
                        ohdr = {"req_id": hdr.get("req_id"), "model_id": model_id,
                                "kind": "ntensor", "cache_position": cache_start,
                                "reset": reset, "all_logits": all_logits, "tensors": tmeta}
                        if position_ids is not None:
                            ohdr["position_ids"] = position_ids
                        if bidir_spans is not None:
                            ohdr["bidir_spans"] = bidir_spans
                        _tx = await self._send_next(model_id, ohdr, oraw)
                        _net_peer(self.next_peer.get(model_id, "?"), tx=_tx)
                    elif (capture_hidden or capture_pre_norm) and shard.has_head and isinstance(out, tuple):
                        logits_t, hidden_t = out
                        lmeta, lraw = _pack_tensor(logits_t)
                        hmeta, hraw = _pack_tensor(hidden_t)
                        ohdr = {"req_id": hdr.get("req_id"), "model_id": model_id, "kind": kind,
                                "cache_position": cache_start, "reset": reset,
                                "all_logits": all_logits, **lmeta,
                                "logits_nbytes": len(lraw), "hid_meta": hmeta}
                        if position_ids is not None:
                            ohdr["position_ids"] = position_ids
                        if bidir_spans is not None:
                            ohdr["bidir_spans"] = bidir_spans
                        _tx = await self._send_next(model_id, ohdr, lraw + hraw)
                        _net_peer(self.next_peer.get(model_id, "?"), tx=_tx)
                    else:
                        meta, oraw = _pack_tensor(out)
                        ohdr = {"req_id": hdr.get("req_id"), "model_id": model_id, "kind": kind,
                                "cache_position": cache_start, "reset": reset,
                                "all_logits": all_logits, **meta}
                        if position_ids is not None:
                            ohdr["position_ids"] = position_ids
                        if bidir_spans is not None:   # #gemma4-bidir: reach every stage's mask
                            ohdr["bidir_spans"] = bidir_spans
                        # propagate the capture flag to the next stage so it reaches the head stage
                        if capture_hidden and not shard.has_head:
                            ohdr["capture_hidden"] = True
                        if capture_pre_norm and not shard.has_head:   # #91 MTP
                            ohdr["capture_pre_norm"] = True
                        if ntensor_req and not shard.has_head:   # #ntensor-manifest: reach the head
                            ohdr["ntensor"] = True
                        _tx = await self._send_next(model_id, ohdr, oraw)
                        _net_peer(self.next_peer.get(model_id, "?"), tx=_tx)   # to next stage / controller
                except Exception as exc:  # stage failed -> tell the controller, fast
                    import traceback
                    tb = traceback.format_exc()
                    print(f"[data] stage error: {exc!r}\n{tb}")
                    # #hop-recovery: a CONNECTION-type exc here means _send_next re-raised AFTER its own
                    # reconnect-once already failed (client _send_next) — i.e. the next-hop worker is
                    # genuinely DEAD, not a transient the idle-socket freshen heals. A real stage COMPUTE
                    # failure raises a model exception (RuntimeError/etc.), never one of these. The data
                    # chain is one-way, so an error frame can't reach the controller over the dead hop;
                    # push an UNSOLICITED hop_error up the (separate) control link so the controller fails
                    # THIS rid's pending future at once instead of waiting out GEN_TIMEOUT (~600s) / the
                    # gen-stall watchdog (~240s). Best-effort: skipped if the control link is mid-reconnect
                    # (the watchdog still backstops). Sent via session's `reply` (wlock+_enc) so it never
                    # interleaves a heartbeat.
                    if isinstance(exc, (ConnectionError, OSError, asyncio.IncompleteReadError,
                                        asyncio.TimeoutError)) and self._ctrl_send is not None:
                        with contextlib.suppress(Exception):
                            await self._ctrl_send({
                                "type": "hop_error", "node_id": self._node_id,
                                "model_id": model_id, "req_id": hdr.get("req_id"),
                                "stage": self.assignments.get(model_id, {}).get("stage"),
                                "next_host": self.next_peer.get(model_id),
                                "error": repr(exc)})
                        print(f"[data] next-hop for {model_id} died -> signalled controller hop_error "
                              f"(req {hdr.get('req_id')}, next={self.next_peer.get(model_id)})", flush=True)
                    elif self._ctrl_send is not None:
                        # #stage-error-ctrl: a COMPUTE exception's error frame (sent below) rides the
                        # one-way DATA chain (this stage -> downstream -> controller); a stale or dead
                        # hop anywhere on that chain silently eats it, and the controller then blind-
                        # waits out the gen-stall watchdog (~240s) — the qwen2.5-vl wedge-storm class
                        # (2026-07-09: 37 wedges in 5.5h on beast, each request re-wedging on retry,
                        # feeding the kernel panic). Mirror the error over the CONTROL link too — it is
                        # heartbeat-kept and reconnect-managed, so delivery is near-guaranteed; the
                        # controller fails this rid's pending future at once (a fast, causal 500
                        # instead of a silent 240s stall). Idempotent with the data-plane error frame:
                        # whichever lands first wins, the other finds the future already gone.
                        with contextlib.suppress(Exception):
                            await self._ctrl_send({
                                "type": "stage_error", "node_id": self._node_id,
                                "model_id": model_id, "req_id": hdr.get("req_id"),
                                "stage": self.assignments.get(model_id, {}).get("stage"),
                                "error": f"{exc!r}"})
                    # surface the deepest FILE:line frames (skip caret-only lines) to the controller.
                    # Route the error through _send_next so it RECONNECTS the next hop first — else a
                    # downstream-conn failure would report the error over the SAME dead socket and be
                    # swallowed (the controller then only sees a GEN_TIMEOUT, never the real cause).
                    frames = [ln.strip() for ln in tb.splitlines() if ln.strip().startswith("File ")]
                    with contextlib.suppress(Exception):
                        await self._send_next(model_id, {
                            "req_id": hdr.get("req_id"), "model_id": model_id,
                            "kind": "error",
                            "error": f"{exc!r} | " + " <- ".join(frames[-3:])}, b"")
        except (asyncio.IncompleteReadError, ConnectionError, asyncio.CancelledError):
            pass
        except Exception as exc:  # pragma: no cover
            print(f"[data] inbound error: {exc!r}")
        finally:
            with contextlib.suppress(Exception):
                writer.close()
