"""control_plane: the controller's worker-facing control plane, relocated VERBATIM from
server.py (code-split Inc 2): control-frame IO (_read_frame/_write_frame/_enc), ControlLink,
the resilient TCP listener (_ResilientServer/_resilient_serve), the register/heartbeat/reply
handler (handle_control + _resolve_pending), reaper_loop, and gen_stall_watchdog.

Bodies are BYTE-IDENTICAL to the originals; their former server.py module globals (engine,
registry, net_account, ARGS, VERSION, Node, NODE_LOGS*, REAPER_INTERVAL_S, ENGINE_CONFIG,
GEN_STALL*, INFLIGHT, resolve_model_name, _inflight_release, _ollama_name, log_activity, ...)
are injected at startup by state.bind() -- see state.py. The stdlib imports below are the
module's OWN because @dataclass/field(default_factory=asyncio.Lock) execute at IMPORT time,
before state.bind() runs. Controller-only leaf; never imports server; in EXTRA_UPDATE_FILES.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import socket
import time
from dataclasses import dataclass, field
from typing import Optional

# #media-anywhere: the control link is line-framed (_enc = JSON + newline), read via
# reader.readline() whose asyncio default cap is 64 KB. A t2a_done now carries the rendered WAV
# as base64 (a REMOTE media worker has no shared filesystem to hand back a local path), which is
# multi-MB — so the accept bridge below creates every reader with a generous line limit.
# (_bridge is shared by the control AND data listeners; the data plane reads via readexactly
# and doesn't care about the line limit, so the bump is harmless there — it only lifts the
# readline cap the control plane needs.) Controller->worker control messages stay tiny, so the
# worker-side reader (client.py open_connection) keeps the asyncio default.
_CTRL_READER_LIMIT = 128 * 1024 * 1024

async def _read_frame(reader: asyncio.StreamReader) -> tuple[dict, bytes, int]:
    """Return (header, payload, wire_bytes). wire_bytes is the exact number of
    bytes pulled off the socket, so the controller can meter its own data-in."""
    hdr_len = int.from_bytes(await reader.readexactly(4), "big")
    hdr = json.loads((await reader.readexactly(hdr_len)).decode())
    raw = await reader.readexactly(hdr["nbytes"]) if hdr["nbytes"] else b""
    return hdr, raw, 4 + hdr_len + len(raw)


async def _write_frame(writer: asyncio.StreamWriter, hdr: dict, raw: bytes) -> int:
    """Write a frame and return the exact wire byte count (for controller metering)."""
    hdr = {**hdr, "nbytes": len(raw)}
    hb = json.dumps(hdr).encode()
    buf = len(hb).to_bytes(4, "big") + hb + raw
    writer.write(buf)
    await writer.drain()
    return len(buf)


def _enc(obj: dict) -> bytes:
    return (json.dumps(obj) + "\n").encode()


@dataclass
class ControlLink:
    node_id: str
    writer: asyncio.StreamWriter
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    # #1: in-flight loads/unloads keyed by model_id — a single Future cross-resolved when
    # two models load onto the SAME node concurrently (B's reply popped A's future). Keyed by
    # model_id (target_id); unload-all (no model_id in the frame) keys under None.
    pending_loads: dict = field(default_factory=dict)    # model_id -> asyncio.Future
    pending_unloads: dict = field(default_factory=dict)  # model_id (or None) -> asyncio.Future

    async def send(self, obj: dict) -> None:
        async with self.lock:
            payload = _enc(obj)
            self.writer.write(payload)
            await self.writer.drain()
            net_account(self.node_id, to_node=len(payload))  # controller -> node


# ---------------------------------------------------------------------------
# Resilient TCP listener — manual accept loop that survives per-accept OSError
# ---------------------------------------------------------------------------
#
# asyncio.start_server() runs an internal accept loop inside the event loop's
# transport layer. On Windows (Proactor/IOCP) a single failed accept — e.g.
# WinError 64 "The specified network name is no longer available" during a
# reconnect storm — raises out of the IocpProactor.accept() task and SILENTLY
# kills that accept loop. serve_forever() keeps awaiting but no further
# connections are ever accepted (observed: 0 nodes for 7+ minutes after a
# self-update restart, only fixed by a manual controller restart).
#
# Fix: drive accept() ourselves with loop.sock_accept() so we own a per-accept
# try/except. A transient OSError on one accept is logged and skipped; the loop
# keeps running. Accepted raw sockets are bridged to StreamReader/StreamWriter
# exactly the way start_server would, so the existing (reader, writer) handlers
# (handle_control / EngineState._on_data) are unchanged.

class _ResilientServer:
    """Drop-in-ish replacement for asyncio.start_server() whose accept loop does
    NOT die on a transient per-accept OSError. Exposes .close() / .wait_closed()
    so existing shutdown code (ctrl.close(); await ctrl.wait_closed()) keeps
    working, plus .task for cancellation alongside the other lifespan tasks."""

    def __init__(self, sock: socket.socket, name: str) -> None:
        self._sock = sock
        self._name = name
        self.task: Optional[asyncio.Task] = None   # the accept loop task

    def close(self) -> None:
        if self.task is not None:
            self.task.cancel()
        with contextlib.suppress(Exception):
            self._sock.close()

    async def wait_closed(self) -> None:
        if self.task is not None:
            with contextlib.suppress(Exception, asyncio.CancelledError):
                await self.task


async def _resilient_serve(host: Optional[str], port: int, handler, name: str) -> _ResilientServer:
    """Bind a listening socket on (host, port) and start a fault-tolerant accept
    loop that dispatches each connection to `handler(reader, writer)`. Returns a
    _ResilientServer once the socket is listening (the accept loop runs as a
    background task). A per-accept OSError is logged and the loop continues — a
    single bad/aborted connection can never take the listener down."""
    loop = asyncio.get_running_loop()
    # Strong refs to in-flight per-connection bridge tasks. asyncio keeps only a WEAK ref to a bare
    # create_task(), so an un-referenced _bridge task can be garbage-collected mid-await -> the
    # "Task was destroyed but it is pending!" warning (and a prematurely dropped connection). Hold a
    # ref until each finishes (discard on done) so the GC can't reap a live connection's handler.
    _bridge_tasks: set = set()

    # Resolve the bind address the same way start_server does (None -> all
    # interfaces). Use getaddrinfo so an explicit host (e.g. "0.0.0.0" or a
    # hostname) and IPv4/IPv6 both work; bind the first usable result.
    bind_host = host if host else None
    infos = socket.getaddrinfo(bind_host, port, type=socket.SOCK_STREAM,
                               flags=socket.AI_PASSIVE)
    lsock: Optional[socket.socket] = None
    last_err: Optional[Exception] = None
    for af, socktype, proto, _canon, sockaddr in infos:
        try:
            s = socket.socket(af, socktype, proto)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(sockaddr)
            s.listen(128)
            s.setblocking(False)
            lsock = s
            break
        except OSError as e:
            last_err = e
            with contextlib.suppress(Exception):
                s.close()
            continue
    if lsock is None:
        raise OSError(f"_resilient_serve({name}): could not bind {bind_host}:{port}: {last_err!r}")

    async def _bridge(conn: socket.socket) -> None:
        # Wrap a raw accepted socket in the canonical StreamReader/StreamWriter
        # pair, exactly as asyncio.start_server does internally, then hand off to
        # the existing handler. Guarded so one bad socket can't kill the loop.
        try:
            conn.setblocking(False)
            reader = asyncio.StreamReader(limit=_CTRL_READER_LIMIT)   # #media-anywhere: multi-MB t2a_done
            proto = asyncio.StreamReaderProtocol(reader)
            transport, _ = await loop.connect_accepted_socket(lambda: proto, conn)
            writer = asyncio.StreamWriter(transport, proto, reader, loop)
            await handler(reader, writer)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            print(f"[{name}] connection bridge error: {exc!r}")
            with contextlib.suppress(Exception):
                conn.close()

    async def _accept_loop() -> None:
        print(f"[*] {name} resilient accept loop on {host}:{port}")
        while True:
            try:
                conn, _addr = await loop.sock_accept(lsock)
            except asyncio.CancelledError:
                raise
            except OSError as e:
                # THE core fix: a transient accept failure (WinError 64, ECONNRESET,
                # EMFILE, etc.) must NOT kill the listener. Log it and keep going.
                print(f"[{name}] accept failed (transient, listener survives): {e!r}")
                await asyncio.sleep(0.05)
                continue
            except Exception as e:  # pragma: no cover — be paranoid for infra
                print(f"[{name}] accept unexpected error (listener survives): {e!r}")
                await asyncio.sleep(0.05)
                continue
            _bt = asyncio.create_task(_bridge(conn))   # keep a strong ref until done (no GC mid-flight)
            _bridge_tasks.add(_bt)
            _bt.add_done_callback(_bridge_tasks.discard)

    server = _ResilientServer(lsock, name)
    server.task = asyncio.create_task(_accept_loop())
    return server


# ---------------------------------------------------------------------------
# Control plane (bidirectional: heartbeats/responses in, commands out)
# ---------------------------------------------------------------------------

def _resolve_pending(d: dict, msg: dict, peer_host: str = "?") -> None:
    """#1: resolve an in-flight load/unload future by model_id (pops the matching future).
    Old worker builds that don't echo model_id fall back to the sole pending future (the
    common case during a rolling deploy); ambiguous (>1 pending, no model_id) -> log + ignore."""
    if msg.get("req_id") is not None:
        return   # #1: a pack/compile frame (keyed by req_id in _pack_futures) — never a load/unload reply
    mid = msg.get("model_id")
    fut = d.pop(mid, None)
    if fut is None:                      # missing/None model_id (old build) -> sole-pending fallback
        if len(d) == 1:
            fut = d.pop(next(iter(d)))
        elif len(d) > 1:
            print(f"[!] {peer_host}: {msg.get('type')} reply with no model_id and "
                  f"{len(d)} pending — cannot disambiguate; ignored")
            return
    if fut is not None and not fut.done():
        fut.set_result(msg)


async def handle_control(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    peer = writer.get_extra_info("peername")
    peer_host = peer[0] if peer else "?"
    node: Optional[Node] = None
    try:
        line = await reader.readline()
        if not line:
            return
        msg = json.loads(line.decode())
        if msg.get("type") != "register":
            writer.write(_enc({"type": "error", "error": "expected register"}))
            await writer.drain()
            return
        node = await registry.add(msg, peer_host=peer_host)
        net_account(node.node_id, from_node=len(line))  # register msg, node -> controller
        engine.links[node.node_id] = ControlLink(node.node_id, writer)
        # #adopt: advertise that THIS controller re-adopts kept shards, so the worker keeps
        # its loaded models across a controller-only restart instead of dropping them on
        # link loss (old workers ignore the flag; old controllers don't send it -> workers
        # drop shards exactly as before — mixed versions degrade to the old behavior).
        reg_reply = _enc({"type": "registered", "node_id": node.node_id,
                          "os_reserve_gb": ARGS.os_reserve_gb, "server_version": VERSION,
                          "adopt": True})
        writer.write(reg_reply)
        await writer.drain()
        net_account(node.node_id, to_node=len(reg_reply))  # controller -> node
        print(f"[+] joined  {node.node_id}  {node.hostname}  {node.device}  "
              f"{node.usable_mem_gb:.1f} GB usable  ({peer_host})")

        # Worker-restart auto-recovery (#77): if this same physical worker (same data endpoint +
        # hostname) was ALREADY registered under an older node id, it RESTARTED and re-registered
        # before its old link was noticed dropped. That restart wiped the worker's shard + KV, so
        # any model the old entry held is dead — recover it now (fail fast) instead of letting the
        # next generate hang out GEN_TIMEOUT on the stale stage, then drop the stale entry. The
        # fork's lazy-connect fix (client m4c10) heals the worker's OWN sockets on first send; this
        # is the controller side that frees the registry/model state the restart invalidated.
        for stale in await registry.find_stale_dupes(node):
            n_rec = await engine.recover_node_restart(stale, node.node_id)
            await registry.remove(stale.node_id)
            print(f"[+] {node.hostname} re-registered as {node.node_id} (was {stale.node_id}); "
                  f"recovered {n_rec} model(s) the restart invalidated")

        # #restart-stale backstop: this worker just (re)registered — if the controller still
        # holds resident models with a stage on this BOX that the worker's kept-model
        # inventory does NOT carry, those shards are GONE (a worker restart wipes them; the
        # inventory is refreshed on EVERY reconnect — client.py session()). The id-keyed
        # cleanups can all miss that: every re-register mints a fresh node id (registry.add),
        # the old socket can die silently half-open (no link-death event), and #77's
        # find_stale_dupes needs the exact data endpoint to match — a dual-homed box that
        # re-registered via another route slips it (the 2026-07-21 om3nbox class:
        # /restart_node saw models_affected=[], then every request 500'd "no shard for
        # model_id=..." until a manual /unload). Match by HOSTNAME (physical identity, the
        # same assumption find_stale_dupes makes) and invalidate LOUDLY. invalidate_model
        # pops the row from engine.models, so this fires exactly once per stale model — a
        # reconnect storm can't turn it into an invalidation storm.
        _held = {e.get("model_id") or (e.get("assign") or {}).get("model_id")
                 for e in (msg.get("loaded") or []) if isinstance(e, dict)}
        _host_l = (node.hostname or "").strip().lower()
        for _fr, _lm in list(engine.models.items()):
            _stgs = getattr(getattr(_lm, "plan", None), "stages", None) or []
            if not any((getattr(_s, "hostname", "") or "").strip().lower() == _host_l
                       for _s in _stgs):
                continue
            if _lm.target_id in _held:
                continue   # worker still holds it (kept for adoption / healthy reconnect flap)
            log_activity(f"#restart-stale: {node.hostname} re-registered ({node.node_id}) "
                         f"WITHOUT its shard of {_ollama_name(_fr)} — controller state was "
                         "stale; invalidating it now")
            engine.invalidate_model(_fr, f"{node.hostname} re-registered without its shard "
                                         "(worker restarted; stale controller state)")

        # #adopt: the register carried a kept-loaded-model inventory (this worker held its
        # shards across a controller restart) — rebuild those models' controller state from
        # the reported recipes instead of re-streaming them. Fire-and-forget: registration
        # must not block on adoption, and a failed adoption falls back to normal reloads
        # (the adopt sweep frees any shard that never assembles).
        _inv = msg.get("loaded") or []
        if _inv:
            asyncio.create_task(engine.adopt_worker_models(node, _inv))

        link = engine.links[node.node_id]
        _fwd_seen: dict = {}   # #prefill-progress: last fwd ts seen per model on THIS connection
        while True:
            line = await reader.readline()
            if not line:
                break
            net_account(node.node_id, from_node=len(line))  # node -> controller
            msg = json.loads(line.decode())
            mtype = msg.get("type")
            if mtype == "heartbeat":
                # Net rates are measured by the controller itself (NODE_NET +
                # metrics_sampler); we deliberately ignore any client-reported
                # net_in_bps/net_out_bps here — the server watches its own wire.
                known = await registry.heartbeat(node.node_id, float(msg.get("free_mem_gb", 0.0)),
                                                 float(msg.get("cpu_percent", 0.0)),
                                                 float(msg.get("free_disk_gb", 0.0)))
                if not known:
                    # #reap-close-link belt: the registry dropped this node while its socket
                    # stayed up (reap race, stale #77 dupe). Heartbeating into a dead entry
                    # can never revive it — close the link so the worker re-registers fresh.
                    print(f"[!] heartbeat from unregistered {node.node_id} ({node.hostname}) "
                          f"— closing link to force re-register")
                    break
                if "proc_rss_gb" in msg:    # worker python RSS (engine-memory split)
                    node.proc_rss_gb = float(msg.get("proc_rss_gb", 0.0))
                if "vram_used_gb" in msg:   # worker-reported GPU memory
                    node.vram_used_gb = float(msg.get("vram_used_gb", 0.0))
                if "vram_reusable_gb" in msg:   # #vram-reusable: worker's vacant allocator pool
                    node.vram_reusable_gb = float(msg.get("vram_reusable_gb", 0.0))
                    node.vram_total_gb = float(msg.get("vram_total_gb", node.vram_total_gb))
                if "gpu_util" in msg:       # worker-reported GPU compute utilization %
                    node.gpu_util = float(msg.get("gpu_util", 0.0))
                if "net_peers" in msg:      # worker per-peer data-plane bytes (bandwidth page)
                    node.peer_bytes = msg.get("net_peers") or {}
                if msg.get("fwd_progress"):
                    # #prefill-progress: {model_id (HF target): worker-clock ts of the last
                    # completed layer}. Only an ADVANCING ts counts — compared against the SAME
                    # worker's previous report (skew-safe, and a wedged forward's frozen ts can't
                    # keep its gen alive). Stamp the matching ACTIVE model(s) that have a stage on
                    # THIS node with the CONTROLLER clock; the gen-stall watchdog's prefill branch
                    # reads it as liveness.
                    _hb_now = time.time()
                    for _mid, _v in dict(msg["fwd_progress"]).items():
                        # [rid, ts] (m4c181+) or bare ts (older worker). With a rid, credit the
                        # progress ONLY while that request is still pending — an orphaned
                        # forward (its rid already failed/removed) can't shield a newer gen.
                        _rid = None
                        try:
                            if isinstance(_v, (list, tuple)) and len(_v) == 2:
                                _rid, _ts = _v[0], float(_v[1])
                            else:
                                _ts = float(_v)
                        except (TypeError, ValueError):
                            continue
                        if _ts <= _fwd_seen.get(_mid, 0.0):
                            continue
                        _fwd_seen[_mid] = _ts
                        if _rid not in (None, ""):
                            # engine.pending is keyed by next_req()'s INT rids (JSON preserves the
                            # int end-to-end) — compare RAW, with an int-coercion fallback so a
                            # str-serialized rid from any future transport still matches. A rid
                            # that matches nothing is an orphaned/finished forward: real progress,
                            # wrong owner — never let it shield a live gen.
                            _live = _rid in engine.pending
                            if not _live:
                                try:
                                    _live = int(_rid) in engine.pending
                                except (TypeError, ValueError):
                                    _live = False
                            if not _live:
                                continue
                        for _lm in list(engine.models.values()):
                            if (getattr(_lm, "target_id", None) == _mid
                                    and getattr(_lm, "active", 0) > 0
                                    and any(getattr(_s, "node_id", None) == node.node_id
                                            for _s in getattr(getattr(_lm, "plan", None),
                                                              "stages", None) or [])):
                                _lm.fwd_progress_ts = _hb_now
                if msg.get("logs"):         # #logs: worker relayed its new stdout/stderr lines
                    _lb = NODE_LOGS.setdefault(node.node_id, [])
                    _lb.extend(str(x) for x in msg["logs"])
                    if len(_lb) > NODE_LOGS_MAX:
                        del _lb[:len(_lb) - NODE_LOGS_MAX]
            elif mtype == "error" and msg.get("req_id") in engine._pack_futures:
                # #distributed-packing: a worker PACK failed. Resolve its pack future with the error
                # NOW so the caller (pack_probe / compile_dist _dispatch_layer) fails fast and falls
                # back to a local pack — instead of blocking on it for the full per-unit timeout.
                f = engine._pack_futures.get(msg.get("req_id"))
                if f is not None and not f.done():
                    f.set_exception(RuntimeError(f"{node.hostname} pack failed: {msg.get('error')}"))
            elif mtype == "packed":
                # success ack; the packed unit itself arrives via POST /pack_result (which resolves
                # the future). Nothing to do here — kept so it isn't mistaken for a load reply.
                pass
            elif mtype == "hop_error":
                # #hop-recovery: a worker's forward to its NEXT pipeline hop died mid-generation (the
                # data chain is one-way, so no error frame can reach us over the dead hop). Fail ONLY
                # this rid's in-flight pending future NOW so the blocked _send wait_for(fut,
                # GEN_TIMEOUT ~600s) returns immediately, instead of waiting it (or the gen-stall
                # watchdog ~240s) out. FULLY IDEMPOTENT: the rid may already be resolved by a logits
                # frame or the watchdog -> pop + not-done check makes that a no-op. Keyed STRICTLY by
                # the failed frame's rid, so a healthy concurrent/sibling-replica request is untouched.
                # We do NOT decrement model.active here: failing the future synchronously resumes the
                # live generate(), whose finally does model.active = max(0, active-1); decrementing here
                # too would double-count. We only reset last_token_ts so the watchdog doesn't double-act.
                _rid = msg.get("req_id")
                _fr = engine.pending_friendly.get(_rid)
                _f = engine.pending.pop(_rid, None)
                engine.pending_model.pop(_rid, None)
                engine.pending_friendly.pop(_rid, None)
                getattr(engine, "pending_slot", {}).pop(_rid, None)   # #kv-slots lockstep
                if _f is not None and not _f.done():
                    _nh = msg.get("next_host") or "?"
                    _st = msg.get("stage")
                    with contextlib.suppress(Exception):
                        _f.set_exception(ConnectionError(
                            f"mid-pipeline hop {_nh} died (stage {_st}) — reload/retry"))
                    _m = engine.models.get(_fr) if _fr else None
                    if _m is not None:
                        _m.last_token_ts = time.time()   # don't let the watchdog double-act on this gen
                    log_activity(f"hop_error: {_ollama_name(_fr or msg.get('model_id') or '?')} "
                                 f"stage {_st} next-hop {_nh} died ({node.hostname}) — failed req "
                                 f"{_rid} fast (slot reclaimed)")
            elif mtype == "stage_error":
                # #stage-error-ctrl: a worker stage's COMPUTE exception (model forward raised),
                # mirrored over the control link because the data-plane error frame can be eaten by
                # a stale/dead downstream hop (the 2026-07-09 qwen2.5-vl silent-wedge class: the
                # controller learned nothing and blind-waited out the ~240s gen-stall watchdog on
                # EVERY retry). Fail the rid's pending future NOW with the worker's real error so the
                # API client gets a fast, causal 500. Same idempotence/anti-double-count contract as
                # hop_error above: pop + not-done check; never touch model.active; stamp
                # last_token_ts so the watchdog doesn't double-act on the just-failed gen.
                _rid = msg.get("req_id")
                _fr = engine.pending_friendly.get(_rid)
                _f = engine.pending.pop(_rid, None)
                engine.pending_model.pop(_rid, None)
                engine.pending_friendly.pop(_rid, None)
                getattr(engine, "pending_slot", {}).pop(_rid, None)   # #kv-slots lockstep
                _err = str(msg.get("error") or "stage compute error")
                _live = _f is not None and not _f.done()
                if _live:
                    with contextlib.suppress(Exception):
                        _f.set_exception(RuntimeError(
                            f"stage {msg.get('stage')} on {node.hostname} failed: {_err}"))
                    _m = engine.models.get(_fr) if _fr else None
                    if _m is not None:
                        _m.last_token_ts = time.time()
                # Log EVERY arrival, matched or not — an unmatched stage_error (request already
                # resolved/reclaimed, or an ORPHANED forward failing after its gen was taken away)
                # is still the only controller-side trace that a worker stage blew up.
                log_activity(f"stage_error: {_ollama_name(_fr or msg.get('model_id') or '?')} "
                             f"stage {msg.get('stage')} on {node.hostname} — {_err[:200]} — "
                             + (f"failed req {_rid} fast" if _live
                                else f"(req {_rid} not pending — already resolved/reclaimed)"))
            elif mtype == "t2i_step":
                # #t2i-serve: per-step render progress. Stored on the engine for /status (the
                # dashboard's live "step i/n" on the model card); also stamps the progress time
                # so a wedged render is distinguishable from a slow one (t2i_generate's timeout).
                _tp = getattr(engine, "_t2i_progress", None)
                if _tp is None:
                    _tp = engine._t2i_progress = {}
                _tp[msg.get("req_id")] = (int(msg.get("step", 0)), int(msg.get("total", 0)),
                                          time.time())
            elif mtype in ("t2i_done", "t2i_err"):
                # #t2i-serve: final render result — resolve the waiting t2i_generate future.
                _pend = getattr(engine, "_t2i_pending", None) or {}
                _fut = _pend.pop(msg.get("req_id"), None)
                if _fut is not None and not _fut.done():
                    _fut.set_result(msg)
                getattr(engine, "_t2i_progress", {}).pop(msg.get("req_id"), None)
            elif mtype == "tts_step":
                # #tts-serve: per-chunk speech progress (dashboard "chunk i/n"); stamps the
                # progress time so a wedged synth is distinguishable from a slow one.
                _sp = getattr(engine, "_tts_progress", None)
                if _sp is None:
                    _sp = engine._tts_progress = {}
                _sp[msg.get("req_id")] = (int(msg.get("step", 0)), int(msg.get("total", 0)),
                                          time.time())
            elif mtype in ("tts_done", "tts_err"):
                # #tts-serve: final speech result — resolve the waiting tts_generate future.
                _pend = getattr(engine, "_tts_pending", None) or {}
                _fut = _pend.pop(msg.get("req_id"), None)
                if _fut is not None and not _fut.done():
                    _fut.set_result(msg)
                getattr(engine, "_tts_progress", {}).pop(msg.get("req_id"), None)
            elif mtype == "t2a_step":
                # #t2a-serve: per-step music render progress (dashboard "step i/n"); stamps the
                # progress time so a wedged render is distinguishable from a slow one.
                _ap = getattr(engine, "_t2a_progress", None)
                if _ap is None:
                    _ap = engine._t2a_progress = {}
                _ap[msg.get("req_id")] = (int(msg.get("step", 0)), int(msg.get("total", 0)),
                                          time.time())
            elif mtype in ("t2a_done", "t2a_err"):
                # #t2a-serve: final music result — resolve the waiting t2a_generate future.
                _pend = getattr(engine, "_t2a_pending", None) or {}
                _fut = _pend.pop(msg.get("req_id"), None)
                if _fut is not None and not _fut.done():
                    _fut.set_result(msg)
                getattr(engine, "_t2a_progress", {}).pop(msg.get("req_id"), None)
            elif mtype in ("ready", "error"):
                _resolve_pending(link.pending_loads, msg, peer_host)
            elif mtype == "unloaded":
                _resolve_pending(link.pending_unloads, msg, peer_host)
    except (asyncio.IncompleteReadError, ConnectionResetError):
        pass
    except json.JSONDecodeError as exc:
        print(f"[!] bad control message from {peer_host}: {exc}")
    except Exception as exc:  # pragma: no cover
        print(f"[!] control error from {peer_host}: {exc!r}")
    finally:
        if node is not None:
            link = engine.links.pop(node.node_id, None)
            # If this node dropped while a load/unload was awaiting its reply, fail
            # that future now — otherwise the orchestrator blocks on it for the full
            # 900s timeout (a worker dying mid-load would hang the whole load).
            if link is not None:
                # #1: fail EVERY in-flight load/unload on this link (dict keyed by model_id) so
                # none block on the full multi-minute timeout when a worker drops mid-operation.
                for _d in (link.pending_loads, link.pending_unloads):
                    for fut in list(_d.values()):
                        if fut is not None and not fut.done():
                            fut.set_exception(ConnectionError(
                                f"{node.hostname} disconnected mid-operation"))
                    _d.clear()
            for fr in [fr for fr, m in engine.models.items()
                       if node.node_id in m.stage_node_ids]:
                engine.invalidate_model(fr, f"node {node.node_id} ({node.hostname}) left")
            await registry.remove(node.node_id)
            print(f"[-] left    {node.node_id}  {node.hostname}")
        with contextlib.suppress(Exception):
            writer.close()


async def reaper_loop() -> None:
    while True:
        await asyncio.sleep(REAPER_INTERVAL_S)
        for n in await registry.reap_dead():
            print(f"[-] reaped {n.node_id} ({n.hostname}, heartbeat timeout)")
            for fr in [fr for fr, m in engine.models.items() if n.node_id in m.stage_node_ids]:
                engine.invalidate_model(fr, f"node {n.hostname} reaped (heartbeat timeout)")
            # #reap-close-link: a reap only deletes the registry entry. If the worker's TCP
            # connection SURVIVED the network blip that made it miss heartbeats (half-open or
            # fully healed), the worker keeps heartbeating into a socket whose node id no longer
            # exists — heartbeat() no-ops for unknown ids — and since registration only happens
            # on a fresh connect, it stays a zombie forever (2026-07-11: prodesk/steamdeck/work
            # orphaned for hours after a LAN blip). Close the link so handle_control's finally
            # tears it down and the worker's reconnect loop re-registers within seconds.
            link = engine.links.get(n.node_id)
            if link is not None:
                with contextlib.suppress(Exception):
                    link.writer.close()


async def gen_stall_watchdog() -> None:
    """#gen-stall-watchdog: reclaim a model WEDGED on a dead pipeline hop. When an inter-worker hop of
    a distributed generation dies (idle-socket death / a flaky node), the decode produces 0 tokens with
    an idle data plane, yet model.active stays >0 — and a disconnecting client can leak it entirely — so
    the model shows BUSY forever and new requests queue behind a ghost. If a loaded model has active>0
    but has emitted NO token for GEN_STALL_S, cancel its in-flight request(s), reset its slot/queue
    counters, and swap in a FRESH per-model lock (an orphaned wedged gen may still 'hold' the old one,
    which would block every queued/new request). The next request re-flows the pipeline and the worker
    reconnects the dead hop (#distributed-gen-idle-socket-death). Threshold > worst-case legit
    first-token wait so a slow big-model prefill is never false-killed."""
    while True:
        await asyncio.sleep(20.0)
        try:
            stall_s = float(ENGINE_CONFIG.get("gen_stall_s", GEN_STALL_S))
        except (TypeError, ValueError):
            stall_s = GEN_STALL_S
        if stall_s <= 0:
            continue   # watchdog disabled
        now = time.time()
        for key, m in list(engine.models.items()):
            if getattr(m, "active", 0) <= 0:
                continue
            # #kv-slots (C>1): reclaim per (model, slot) — one wedged slot must never reclaim
            # its healthy siblings, so the model-level path below (zero active, swap the lock)
            # is WRONG for a slotted model and is skipped entirely. Each slot carries its own
            # token stamps (slot_state, stamped by generate); a stalled slot has ONLY its own
            # request cancelled, ONLY its own pending futures failed (pending_slot match), its
            # own #prefix-kv record nulled, and its slot returned to the pool via the ownership
            # token swap (the orphaned lease's release then no-ops — no double-free). We do NOT
            # touch model.active here: failing the future / cancelling the task resumes the
            # generation, whose finally decrements it (the hop_error anti-double-count rule).
            if int(getattr(m, "kv_slots", 1) or 1) > 1:
                try:
                    _ds = float(ENGINE_CONFIG.get("gen_stall_decode_s", GEN_STALL_DECODE_S))
                except (TypeError, ValueError):
                    _ds = GEN_STALL_DECODE_S
                for _sl, _st in list((getattr(m, "slot_state", None) or {}).items()):
                    _sl_last = float(_st.get("last_token_ts") or 0.0)
                    _sl_started = float(_st.get("gen_started_ts") or 0.0)
                    _sl_decoding = _sl_last > _sl_started > 0
                    _sl_eff = min(stall_s, _ds) if (_sl_decoding and _ds > 0) else stall_s
                    # prefill liveness: fwd_progress is MODEL-level (worker heartbeats credit
                    # rids, not slots) — during any live prefill it shields every slot's
                    # PREFILL branch (conservative: never reclaims a slow prefill early; at
                    # most 1 prefill runs per replica via prefill_lock anyway). Decode stays
                    # tokens-only PER SLOT so a dead hop still reclaims that slot fast.
                    _sl_basis = (_sl_last if _sl_decoding
                                 else max(_sl_last, getattr(m, "fwd_progress_ts", 0.0) or 0.0))
                    if _sl_basis <= 0 or (now - _sl_basis) <= _sl_eff:
                        continue
                    _idle_s = int(now - _sl_basis)
                    _r = _st.get("rec")
                    if _r is not None:
                        _r["cancel"] = True
                        _r["reclaimed"] = True   # #endpoint-weather: retryable 503/529 marker
                        _t = _r.get("task")
                        if _t is not None and not _t.done():
                            with contextlib.suppress(Exception):
                                _t.cancel()
                        _inflight_release(_r)
                    _psl = getattr(engine, "pending_slot", {}) or {}
                    # default 0, NOT None: _send/_send_prefill record pending_slot only for
                    # slot>0 (``if slot:`` guard), so on this C>1 replica an UNRECORDED rid IS
                    # slot 0 — ``_psl.get(r)`` (None) would never match _sl==0 and a wedged
                    # slot 0 would fail no in-flight future (rec-less internal gens would hang
                    # until the generic hop/GEN timeout).
                    for _rid in [r for r, fr in
                                 list(getattr(engine, "pending_friendly", {}).items())
                                 if fr == key and _psl.get(r, 0) == _sl]:
                        _f = engine.pending.get(_rid)
                        engine.pending.pop(_rid, None)
                        engine.pending_model.pop(_rid, None)
                        engine.pending_friendly.pop(_rid, None)
                        _psl.pop(_rid, None)
                        if _f is not None and not _f.done():
                            with contextlib.suppress(Exception):
                                _f.set_exception(ConnectionError(
                                    "gen-stall watchdog reclaim (kv-slot)"))
                    with contextlib.suppress(Exception):   # half-appended KV possible -> record dead
                        (getattr(m, "kv_ids_slots", None) or {})[_sl] = None
                    if m.slot_owner.get(_sl) is _st.get("lease"):
                        m.slot_owner.pop(_sl, None)
                        m.slot_state.pop(_sl, None)
                        if _sl not in m.slot_free:
                            m.slot_free.append(_sl)
                        m.slots_active = max(0, int(getattr(m, "slots_active", 0) or 0) - 1)
                        with contextlib.suppress(Exception):
                            m.slot_sem.release()
                    else:
                        m.slot_state.pop(_sl, None)   # lease already gone — just drop the stamps
                    engine._last_load_failure = time.time()   # self-update cool-down (anti-churn)
                    log_activity(f"gen-stall watchdog: {_ollama_name(key)} slot {_sl} wedged — "
                                 f"no token for {_idle_s}s "
                                 f"({'decode' if _sl_decoding else 'prefill'}, "
                                 f"thresh {int(_sl_eff)}s) — reclaimed the slot; sibling slots "
                                 f"keep serving")
                    # #wedge-quarantine per (model,slot): repeated wedges of the SAME slot inside
                    # the window still mean a systematically broken replica — the re-place is
                    # inherently model-wide (fresh shards for every slot), same as C=1.
                    try:
                        _thr = int(ENGINE_CONFIG.get("wedge_reload_n", 3) or 0)
                    except (TypeError, ValueError):
                        _thr = 3
                    _wr = getattr(engine, "_wedge_recent", None)
                    if _wr is None:
                        _wr = engine._wedge_recent = {}
                    _wk = f"{key}::slot{_sl}"
                    _cnt, _ts = _wr.get(_wk, (0, 0.0))
                    _cnt = _cnt + 1 if (now - _ts) < 900.0 else 1
                    _wr[_wk] = (_cnt, now)
                    if (_thr > 0 and _cnt >= _thr and not engine._juggle_lock.locked()
                            and not getattr(getattr(m, "spec", None), "is_embedding", False)):
                        _wr[_wk] = (0, now)
                        log_activity(f"wedge-quarantine: {_ollama_name(key)} slot {_sl} wedged "
                                     f"{_cnt} times in {int((now - _ts) / 60) if _ts else 0}+ min "
                                     f"— forcing a fresh re-place (self-heal, keeps kv_slots)")

                        async def _selfheal_slot(fr=key, _tp=getattr(m, "tp_size", 1),
                                                 _ctx=m.ctx, _q=(m.quant or "none")):
                            async with engine._juggle_lock:   # one managed re-place at a time
                                try:
                                    if fr not in engine.models:
                                        return
                                    await engine.reconfigure(fr, tp=_tp, ctx=_ctx, quant=_q,
                                                             consolidate=True, prefer_vram=True,
                                                             cpu_only=False)   # kv_slots preserved
                                    log_activity(f"wedge-quarantine: {_ollama_name(fr)} "
                                                 f"re-placed OK")
                                except Exception as _exc:
                                    log_activity(f"wedge-quarantine: re-place of "
                                                 f"{_ollama_name(fr)} failed ({_exc!r}) — will "
                                                 f"retry after the next wedge")
                        asyncio.create_task(_selfheal_slot())
                continue
            last = getattr(m, "last_token_ts", 0.0) or 0.0
            # #active-decode-stall: once this gen produced its first token (last_token_ts advanced past
            # gen_started_ts) it's DECODING — apply the SHORTER gen_stall_decode_s so a wedged hop (the
            # buffered-write deadlock hop_error can't catch) is reclaimed fast (~60s not ~240s). Cold
            # prefill (no token yet) keeps the conservative stall_s so a slow big-model first-token is safe.
            _started = getattr(m, "gen_started_ts", 0.0) or 0.0
            _decoding = last > _started > 0
            _eff = stall_s
            if _decoding:
                try:
                    _ds = float(ENGINE_CONFIG.get("gen_stall_decode_s", GEN_STALL_DECODE_S))
                except (TypeError, ValueError):
                    _ds = GEN_STALL_DECODE_S
                if _ds > 0:
                    _eff = min(_eff, _ds)
            # #prefill-progress: in PREFILL (no token yet) count worker-reported per-layer forward
            # progress as liveness — a slow-but-advancing prefill under GPU contention is NOT a
            # wedge and must not be reclaimed (the endpoint-weather 21% run-abort class: every
            # client retry re-entered the same slow prefill and died at the threshold again).
            # DECODE stays tokens-only so a wedged mid-pipeline hop still reclaims fast.
            _basis = last if _decoding else max(last, getattr(m, "fwd_progress_ts", 0.0) or 0.0)
            if _basis <= 0 or (now - _basis) <= _eff:
                continue
            idle = int(now - _basis)
            cancelled = 0
            for r in list(INFLIGHT.values()):
                try:
                    rf = resolve_model_name(r.get("model", ""))
                except Exception:
                    rf = r.get("model")
                if rf in (key, getattr(m, "base", "") or key, getattr(m, "friendly", key)):
                    r["cancel"] = True
                    # #endpoint-weather: mark WHY — serving's CancelledError catch returns a
                    # RETRYABLE 503/529 only for a watchdog reclaim; a user /cancel or /terminate
                    # (which also set "cancel") keeps today's dropped-connection behavior so a
                    # deliberately-killed client isn't told to retry.
                    r["reclaimed"] = True
                    t = r.get("task")
                    if t is not None and not t.done():
                        with contextlib.suppress(Exception):
                            t.cancel()
                    _inflight_release(r)
                    cancelled += 1
            old_active = m.active
            m.active = 0
            m.queued = 0
            m.last_tok_s = 0.0
            m.last_token_ts = now
            m.fwd_progress_ts = 0.0   # #prefill-progress: stale stamps must not shield the NEXT gen
            m.kv_ids = None   # #prefix-kv: a reclaimed gen may have half-appended KV (orphaned
            #                   forwards can still land) — never let the next request resume it
            # #recovery: a mid-pipeline hop death never delivers an error frame upstream (the data
            # chain is one-way), so the orphaned generate() is blocked in _send's wait_for(fut,
            # GEN_TIMEOUT). Fail this model's leaked controller-side pending futures NOW so _send
            # returns immediately (ConnectionError) instead of hanging the coroutine for ~600s — the
            # cancel above only helps if the task handle is live; this frees it regardless.
            _failed = 0
            # #5: fail this WEDGED model's leaked controller-side pending futures by the routed
            # replica key (engine.pending_friendly[rid] == key, the unique per-replica registry
            # name). This supersedes the m4c146 replicated-SKIP: failing by friendly (not the
            # SHARED target_id) touches ONLY this stalled replica's requests, never a healthy
            # sibling replica's — so data-parallel models get the fast future-fail too instead of
            # waiting out GEN_TIMEOUT. Single-copy models: friendly == the only key, same effect as
            # before.
            for _rid in [r for r, fr in list(getattr(engine, "pending_friendly", {}).items())
                         if fr == key]:
                _f = engine.pending.get(_rid)
                engine.pending.pop(_rid, None)
                engine.pending_model.pop(_rid, None)
                engine.pending_friendly.pop(_rid, None)
                getattr(engine, "pending_slot", {}).pop(_rid, None)   # #kv-slots lockstep
                if _f is not None and not _f.done():
                    with contextlib.suppress(Exception):
                        _f.set_exception(ConnectionError("gen-stall watchdog reclaim"))
                    _failed += 1
            with contextlib.suppress(Exception):
                m.lock = asyncio.Lock()   # drop a lock an orphaned wedged gen may still hold -> unblock the queue
            engine._last_load_failure = time.time()   # arm the self-update cool-down (anti-churn after a fault)
            log_activity(f"gen-stall watchdog: {_ollama_name(key)} wedged — no token for {idle}s "
                         f"({'decode' if _decoding else 'prefill'}, thresh {int(_eff)}s) "
                         f"(active {old_active}, cancelled {cancelled} req) — reclaimed slot + reset pipeline lock")
            # #wedge-quarantine: repeated reclaims of the SAME model inside a short window mean the
            # replica is SYSTEMATICALLY broken (poisoned worker state / stale pipeline — the
            # 2026-07-09 beast wedge-storm: qwen2.5-vl re-wedged on every client retry, 37 times in
            # 5.5h, and the accumulated pathological load fed a kernel panic). A fresh re-place (new
            # shards + new data conns, via reconfigure's rollback-safe managed reload) is the
            # demonstrated cure — do it automatically after wedge_reload_n wedges in 15 min instead
            # of reclaim-retry-rewedge forever. 0 disables. Skips embeddings; serialized against the
            # juggler's own re-places via _juggle_lock (skip -> the next wedge tries again).
            try:
                _thr = int(ENGINE_CONFIG.get("wedge_reload_n", 3) or 0)
            except (TypeError, ValueError):
                _thr = 3
            _wr = getattr(engine, "_wedge_recent", None)
            if _wr is None:
                _wr = engine._wedge_recent = {}
            _cnt, _ts = _wr.get(key, (0, 0.0))
            _cnt = _cnt + 1 if (now - _ts) < 900.0 else 1
            _wr[key] = (_cnt, now)
            if (_thr > 0 and _cnt >= _thr and not engine._juggle_lock.locked()
                    and not getattr(getattr(m, "spec", None), "is_embedding", False)):
                _wr[key] = (0, now)
                log_activity(f"wedge-quarantine: {_ollama_name(key)} wedged {_cnt} times in "
                             f"{int((now - _ts) / 60) if _ts else 0}+ min — forcing a fresh "
                             f"re-place (self-heal)")

                async def _selfheal(fr=key, _tp=getattr(m, "tp_size", 1), _ctx=m.ctx,
                                    _q=(m.quant or "none")):
                    async with engine._juggle_lock:   # one managed re-place at a time, fleet-wide
                        try:
                            if fr not in engine.models:
                                return   # already gone (unloaded/re-placed elsewhere meanwhile)
                            await engine.reconfigure(fr, tp=_tp, ctx=_ctx, quant=_q,
                                                     consolidate=True, prefer_vram=True,
                                                     cpu_only=False)
                            log_activity(f"wedge-quarantine: {_ollama_name(fr)} re-placed OK")
                        except Exception as _exc:
                            log_activity(f"wedge-quarantine: re-place of {_ollama_name(fr)} "
                                         f"failed ({_exc!r}) — will retry after the next wedge")
                asyncio.create_task(_selfheal())
