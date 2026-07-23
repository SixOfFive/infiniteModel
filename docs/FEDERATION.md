# Multiple controllers (federation)

InfiniteModel can run **more than one controller** over a single fleet of workers, and the
controllers cooperate: they discover each other, show one unified view of the whole fleet, borrow
each other's loaded models, copy weights controller-to-controller, negotiate exclusive ownership of
each worker, hand nodes back and forth with no reload, and **fail over** when one goes away.

The physical invariant never changes and is worth stating first, because everything below follows
from it:

> **One controller drives any given shard at a time.** A shard has a single KV cache and
> per-controller request/slot counters; two controllers writing to it would interleave and corrupt
> generation. So "sharing" a model means **one physical copy with two front doors** — either
> controller can *serve* it (by federating the request to whoever drives it) — never two drivers.

Everything here is opt-in and additive. A single-controller fleet behaves exactly as before; nothing
in this document changes how one controller places or serves models on its own.

---

## Why run more than one controller

- **A second front door.** Point some clients at controller B; requests for models only controller A
  has loaded are transparently served from A. One set of weights, two API endpoints.
- **Pools that span administrative boundaries.** Two controllers can each own a disjoint set of
  workers and still present one fleet, without double-booking a node's memory.
- **Failover.** If the controller a fleet is attached to dies, the workers can re-home to a surviving
  controller and keep serving the models they already had resident — no reload.

A controller can be a full peer with its own workers, or a **weightless standby** that owns no nodes
and exists only to borrow models and take over on failure.

---

## Zero-config discovery

Controllers and workers find each other by **UDP broadcast** — no addresses to configure on a flat
LAN. A worker (or a peer controller) broadcasts a discovery query; a controller replies unicast with
the address the querier should use (computed from the interface the query arrived on, so a
multi-homed / VPN'd controller hands out the address that actually works from where the querier
sits).

- Workers default to `controller_host: "auto"` (in `config.json`) — broadcast-discover, retry
  forever until a controller answers. A static `controller_host` is still honored and is the path for
  anything broadcast can't cross (a different subnet, a VLAN, a VPN).
- Controllers announce themselves on the same channel, so **one datagram teaches both sides**: the
  responder records the querier as a peer, and every replier is recorded by the announcer.
- `discovery_port` (default `50099`) and an optional `cluster_id` (empty = join whoever answers; set
  it on both controller and workers to pin a worker to one fleet on a shared L2) round out the knobs.

Peers that broadcast can't reach are added by hand: `POST /peer_add?host=<h>&port=<http-port>`
(or the **Controllers** dashboard page).

---

## The unified fleet view

Once two controllers know each other they gossip every ~10 s over `GET /peer_info` (identity,
resident models, nodes, ownership claims), and each caches the other's full `/status`. The result:
**either controller renders the whole fleet** — its own nodes and models *plus* its peers' — on the
ordinary dashboard and in `/status`.

- Peer nodes appear in the Nodes grid (dimmed, with a `via <controller>` chip and no restart handle —
  the owning controller restarts its own hardware).
- Peer models appear in the Models list with the same chip; their card carries the *owner's* real
  measured footprint and throughput graph, not a reconstruction.
- The header tiles (nodes / GPU / RAM / throughput) sum the **whole** fleet, attributed
  (`13 · 4 GPU · 13 via <peer>`), while the underlying `pool` / `compute` fields stay
  strictly-this-controller for anything that plans against them.

`/status` gains `peer_nodes` and `peer_models` (the raw peer contribution) alongside the merged view.

---

## Request federation — borrow a peer's loaded model

A generation request for a model **not resident here** that a **healthy peer has resident** is
proxied to that peer and streamed straight back (chunk-by-chunk, so SSE/NDJSON token streams stay
live). Applies to every generation endpoint — `/api/chat`, `/api/generate`, `/api/embed`,
`/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/messages`.

- Only "ok" peers are eligible; the least-busy match wins so load spreads.
- An `X-InfiniteModel-Federated` header breaks A→B→A loops.
- Any failure falls through to normal local handling — federation can never turn a servable request
  into an error.
- Names are matched the way the resolver resolves them, so `qwen3:4b`, `qwen3-4b`, and
  `nomic-embed-text:latest` all match a peer's dash-form model.

Knob: `POST /config?federate=0` disables it.

A weightless standby controller with **zero models loaded** can therefore answer a request for any
model a peer has loaded, in roughly the time it takes to proxy one HTTP call.

---

## Copy a model between controllers (peer pull)

Instead of re-downloading a model from HuggingFace, a controller can copy it from a peer that already
has it on disk:

```
POST /peer_pull?host=<peer>&model=<name>[&port=<http>][&caches=0|1]
GET  /peer_pull_status
```

- Runs in the background (a pull is many GB); streams to `<file>.part` and renames on completion, so
  an interrupted pull resumes rather than leaving a truncated weight file, and re-running skips files
  already present at the advertised size.
- **Shard caches (`_shards/…`) are skipped by default** — they are regenerable and often larger than
  the weights themselves. Pass `caches=1` to include them.
- On completion the model is registered exactly as `/add_model` would, so it is immediately loadable.

The **Controllers** page lists a peer's entire on-disk catalogue (not just loaded models), each with
a one-click **Pull** button and its download size.

The controller serves its own files to peers via `GET /peer_model_manifest?model=` and
`GET /peer_model_file?model=&path=`; the `path` is validated against directory traversal and
absolute/drive-qualified escapes.

---

## Exclusive node ownership

Two controllers planning against the **same** node's memory would double-book it — each is blind to
the other's reserved-but-unfaulted KV and in-flight reservations, so both read the same "free" bytes
and both commit, OOM-ing the box. InfiniteModel avoids this by making ownership **exclusive**: a node
in use by one controller is simply invisible to the others' planners.

- `/peer_info` advertises `claims` (the hostnames a controller holds a shard on); a controller's
  planner **skips a healthy peer's claimed node**.
- Ours always wins — a controller never yields a node it currently holds a shard on — and a *stale*
  peer's claim is ignored, so a dead controller can't strand capacity.
- Knob: `POST /config?respect_peer_claims=0`.

**Lending** a node is just "stop using it here": `POST /peer_lend?node=<host>` disables that node's
tiers locally so a peer can take it; `POST /peer_reclaim?node=<host>` re-enables them.

---

## Move a node between controllers — handoff, no reload

`/peer_handoff` transfers a node **and its resident shards** to another controller with **no reload**:

```
POST /peer_handoff?node=<host>&to=<peer>[&port=<http>][&force=1]
POST /peer_handoff?node=*&to=<peer>            # the whole fleet at once
```

The worker keeps the weights in VRAM/RAM, reassigns their owner, drops its session, and reconnects to
the new controller — which **adopts** them through the ordinary adoption path (the same mechanism a
hitless controller restart uses, aimed at a different controller). Typical time: seconds.

- Refused (409) if a model spans this node **and** others — handing over one leg of a pipeline would
  split it across controllers. Move single-node models, or hand over every node they span (`node=*`).
- Handing over the controller's **own co-located worker** needs `force=1` (auto-placement often lands
  there; the guard stops you giving away this box's GPU by accident). `node=*` skips the co-located
  worker unless forced.
- The mirror, **`/reclaim_fleet?host=<peer>`**, *pulls* a peer's whole fleet here — useful because a
  handoff can only be issued by whoever currently owns the nodes.

---

## Failover — a controller can go away

If the controller a fleet is attached to becomes unreachable, its workers **keep their shards** (that
is adoption working as designed) and, after a grace window, re-home to a surviving controller.

The control plane is worker-initiated, so failover is worker-driven — effectively a self-issued
handoff:

1. A worker's control link drops. It retries **its own controller** with backoff.
2. Only after the controller has been unreachable for **`failover_after_s`** (default **180 s**) does
   the worker look elsewhere — it probes discovery.
3. If **its own controller answers** the probe, it stays put (a controller that is merely restarting
   is preferred over a handover). Only a **different** healthy controller triggers a re-home.
4. On re-home the worker re-owns its kept shards to the new controller, which adopts them — weights
   untouched, no reload.

The threshold is the key idea: it is a **grace window** that distinguishes "the controller rebooted"
from "the controller is gone." A hitless restart (~20–60 s) or a quick reboot lands well inside it, so
the workers simply wait and the original controller **re-adopts** them when it returns — a peer is
never involved. Raise `failover_after_s` if your controller's bare-metal reboots run longer than the
default; `0` disables failover (a worker then belongs to its controller for life).

### Standby recruitment

A controller that should only recruit workers **when its peer is down** runs with
`INFINITEMODEL_DISCOVERY_RESPOND=standby`: it answers worker discovery **only while no peer is
healthy**, and always answers peer announces (so federation keeps working while it stands by). This is
what makes a two-controller LAN both safe (workers don't drift between two live controllers) and
survivable (a stranded worker has somewhere to go). The other settings are `1` (always recruit — the
default) and `0` (never). It is an **environment variable**, not `config.json`, because `config.json`
is kept in sync across the fleet and a synced edit would clobber a per-controller choice.

### Failback

Failover does **not** auto-reverse — deliberately, to avoid flapping between two live controllers.
When the original controller returns, hand the fleet back with one call (or one dashboard click):

```
POST /peer_handoff?node=*&to=<original>     # from the controller that currently holds the fleet
POST /reclaim_fleet?host=<current-owner>    # from the controller you want to hold it
```

### What failover does and does not cover

- **Serving already-resident models continues** on the survivor once the workers re-home and it
  adopts them.
- **Loading/unloading continues** — the survivor is a full controller of the inherited workers — but
  it can only *load* a model whose **weights are on its own disk** (or that it can pull from a peer,
  which requires that peer to be up). Replicate the models you care about to the standby (peer pull)
  so it can run them standalone.
- If a worker node that held **part of a multi-node model** is the one that died, that model cannot
  be adopted (a stage is missing); the surviving stages are **freed** after an adoption grace so their
  VRAM/RAM is reclaimed, and the model re-loads fresh on demand if its weights are available.
- Failover is **recovery, not zero-downtime HA**: requests in the ~grace-window fail before the
  survivor takes over, and a controller dying mid-generation loses that in-flight request.

---

## Master controller + one-click failback

Any controller can be marked the **master** — the designated primary owner of the fleet — on its
**Config** page (or `POST /config?master=1`). It is a per-controller preference (stored in
`engine_config.json`, which is **not** synced), and it changes no placement or serving behaviour on
its own. Its job is failback ergonomics.

The master is advertised to peers and shown with a ★ badge. On the **Controllers** page the one-click
handoff is direction-aware, because a handoff can only be *issued* by whoever owns the nodes:

- You hold the fleet and a peer is master → **↩ Restore fleet to master**.
- You are the master but hold nothing (a peer took over) → **⇐ Take the fleet**.
- Plus per-peer **Give fleet** / **Take fleet** for manual control.

Set it on the controller that should normally hold the workers, and restoring after a failover is a
single click on either dashboard.

---

## Endpoint reference

| Endpoint | Purpose |
|---|---|
| `GET /peer_info` | What we advertise to peers: identity (+ `master`), resident models, nodes, on-disk catalogue, ownership claims. |
| `GET /peers` | Peers we know about + their cached inventory; `self` carries our node count. Dashboard source. |
| `POST /peer_refresh` | Announce ourselves and re-poll every peer now. |
| `POST /peer_add?host=&port=` | Add a peer by address (for peers broadcast can't reach). |
| `POST /peer_remove?host=&port=` | Forget a peer. |
| `GET /peer_nodes` | Per-node ownership view (ours / a peer's / lendable). |
| `GET /peer_model_manifest?model=` · `GET /peer_model_file?model=&path=` | Serve our model files to a peer. |
| `POST /peer_pull?host=&model=[&caches=]` · `GET /peer_pull_status` | Copy a peer's model here. |
| `POST /peer_lend?node=` · `POST /peer_reclaim?node=` | Lend / reclaim a node. |
| `POST /peer_handoff?node=<host or `*`>&to=[&force=1]` | Move a node (+ its shards) to a peer, no reload. |
| `POST /reclaim_fleet?host=` | Pull a peer's whole fleet here (the handoff mirror). |

## Configuration reference

| Setting | Where | Default | Effect |
|---|---|---|---|
| `controller_host` | `config.json` | `auto` | `auto` = broadcast-discover; or a static host for subnet/VLAN/VPN spans. |
| `discovery_port` | `config.json` | `50099` | UDP discovery channel. |
| `cluster_id` | `config.json` | `""` (unset) | Empty = join whoever answers; set on both sides to pin a worker to one fleet on a shared LAN. |
| `failover_after_s` | `config.json` | `180` | Seconds a controller must be unreachable before a worker looks elsewhere. `0` disables failover. |
| `federate` | `/config` (runtime) | on | Borrow a peer's resident model for a request. |
| `respect_peer_claims` | `/config` (runtime) | on | Skip a peer's claimed nodes when planning. |
| `master` | `/config` (runtime, per-controller) | off | Designate this controller the primary fleet owner (drives the one-click failback UI). |
| `INFINITEMODEL_DISCOVERY_RESPOND` | environment | `1` | `1` always recruit workers · `standby` only while no peer is healthy · `0` never (peering still works). |

## Limits & design notes

- **One controller drives a shard.** Sharing a model = federate the request or hand off the owner;
  two controllers never co-drive one shard.
- **A standby can only *reload* models whose weights are on its disk.** It can always *inherit*
  (adopt) whatever is already resident on the workers it takes over, and *borrow* (federate) anything
  a live peer has loaded — but to load a model fresh on its own it needs the weights locally
  (replicate them with peer pull) or a reachable peer to pull from.
- **Failover is recovery, not HA.** ~`failover_after_s` of downtime before takeover; in-flight
  requests at the moment of failure are lost.
- **A node's tiers/lending are per-controller.** Handoff coordinates them for you (the receiver
  re-enables what the sender disabled); an operator-disabled node stays disabled.
