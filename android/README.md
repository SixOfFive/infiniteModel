# InfiniteModel on Android (experimental)

> ⚠️ **Experimental.** This is a working proof-of-concept, not a polished
> product. It runs a real InfiniteModel **worker** on an Android phone/tablet
> over Wi-Fi, but a battery-powered ARM device on Wi-Fi is a flaky cluster node —
> expect reconnects, modest capacity, and some hand-tuning. Treat it as a fun
> edge node, not a dependable one. Nothing here roots, flashes, or modifies
> Android — it's all ordinary userspace inside Termux.

InfiniteModel is a from-scratch distributed LLM inference system: a controller
splits one transformer model's layers across a fleet of machines and streams
the weights to each worker, which holds only its slice. This folder lets an
Android device join that fleet as a **CPU worker**, and includes a standalone
**fleet bandwidth panel** you can run on the device as a always-on dashboard.

---

## What you can do with it

1. **Run a worker** — the device becomes a CPU node named `tablet` on your
   fleet, holding a small shard (a few pipeline layers / small models). It runs
   the *same* `client.py` the rest of the fleet runs.
2. **Run the bandwidth panel** — a tiny, read-only full-screen dashboard
   (`traffic_panel.py`) that polls the controller's `/status` and shows live
   per-node and total fleet traffic, with a peak-in-window MAX column and
   sparklines. Touches nothing in the fleet; great use for an idle tablet.

You can do either or both. The panel needs nothing but Python and network access
to the controller; the worker needs the proot setup below.

---

## Why proot (and not native Termux)

`client.py` is a glibc program that installs manylinux PyTorch wheels. Termux
itself is **Bionic** libc and has no PyTorch package, so torch won't install
there. A **proot Debian guest is glibc**, so `pip install torch==2.13.0`
(from the PyTorch **CPU** index) pulls the aarch64 CPU wheel straight from PyPI
and everything just works. The client auto-detects "no CUDA → CPU"; no code
changes are needed. proot is plain userspace — no root, no flashing.

> Heads-up on ARM CPUs without `FEAT_BF16` (many tablet SoCs, e.g. Unisoc): the
> client detects that bf16 matmul would fall back to a slow/noisy path and runs
> those ops in fp32 instead. Output is correct; it's just CPU-speed.

---

## Setup — worker

### 1. Install Termux from **F-Droid**, not the Play Store
<https://f-droid.org/packages/com.termux/> — the Play Store build is frozen and
its `pkg` is broken, and the `Termux:Widget`/`Termux:Boot` add-ons only install
against the F-Droid signature. Open Termux once so it finishes first-run setup.

### 2. Install a Debian guest
```bash
pkg update -y && pkg upgrade -y
pkg install -y proot-distro
proot-distro install debian
```

### 3. Get this folder into the guest
A lean sparse clone of just `android/` from the **public** repo (no auth, no
token):
```bash
proot-distro login debian          # you are now root inside Debian
apt update && apt install -y git python3 python3-venv python3-pip tmux

git clone --filter=blob:none --sparse https://github.com/SixOfFive/infiniteModel.git im
cd im && git sparse-checkout set android && cd android
```
*(No clone? Copy this `android/` folder in by any means — `adb push`, Termux
shared storage, `scp` — and `cd` into it.)*

You can also drive the whole install from the Termux host shell in one shot with
`termux-bootstrap.sh` (installs proot + Debian, deploys the files, runs setup,
and writes a tap-to-launch shortcut for `Termux:Widget`).

### 4. Build the env + install deps
```bash
bash setup.sh        # creates .venv and installs the worker deps (~200 MB torch wheel)
```

### 5. Point it at your controller
Nothing to do on a single LAN: `controller_host` defaults to `"auto"`, which
finds the controller by UDP-broadcast discovery and follows it if it moves. Set
a static IP in `config.json` (or pass `--controller <ip>`) only when broadcast
can't reach it — a different subnet, VLAN, or VPN. Ports rarely need changing.

### 6. Keep the device awake, then start the worker
In **Termux** (the host shell), keep the CPU from sleeping:
```bash
termux-wake-lock        # release later with: termux-wake-unlock
```
Also turn **off battery optimization** for Termux in Android settings, or the OS
freezes it in the background. Then, inside the guest, run under `tmux` so it
survives the terminal closing:
```bash
tmux new -s im
bash start-client.sh --name tablet
#   detach: Ctrl-b then d        reattach later: tmux attach -t im
```
Within a few seconds it should appear on the controller dashboard as a CPU node
named **tablet**.

#### Overriding defaults
`start-client.sh` passes extra flags straight through to `client.py`:
```bash
bash start-client.sh --name tablet --controller 192.168.1.50   # different controller
bash start-client.sh --name tablet --os-reserve-gb 3           # leave more RAM for Android
```

---

## Setup — bandwidth panel (optional, no worker needed)

A full-screen, read-only fleet bandwidth dashboard. It only reads the
controller's `/status`, so it's safe to run on any device on the LAN.

```bash
# in Termux (native — no proot/torch needed), with the panel file present:
bash termux-panel-setup.sh
```
This installs `tmux` + `python`, drops the panel in `~/.im/`, and wires a
`~/.bashrc` launcher so opening Termux shows the live panel full-screen. Type
`panel` to rebuild the layout. Or just run it directly:
```bash
python traffic_panel.py [controller_ip] [poll_seconds]
```
Columns: **DOWN/UP** (current rate), **MAX** (peak combined rate *within the
displayed window*, not all-time), **XFER** (bytes since the panel started; rows
sort busiest-first), plus a sparkline scaled to that window's max. A fleet
**TOTAL** and the **controller** row sit at the bottom.

---

## Capabilities & limits

- **It's a small worker.** Most of a device's RAM belongs to Android, so it
  holds only small shards — a few pipeline layers or a small model, not a 70B
  stage. A real node, modest capacity.
- **CPU only.** Android has no usable CUDA, so this is a CPU node. Fine for
  pipeline parallelism where the device owns a thin slice.
- **Same client as the fleet.** It's the unmodified `client.py`; it self-updates
  from the **public** GitHub repo's raw endpoint (no token), so it stays in sync
  with the fleet automatically. Harmless if you'd rather it didn't — there's no
  auth involved either way.

## Known rough edges

- **Wi-Fi reconnect churn.** On some devices the worker's control connection
  flaps (the OS reaps idle sockets, or the controller can't reach the device's
  data port back over Wi-Fi). Symptom: the node re-registers every few seconds.
  If that happens, prefer running it as a **leaf / single-node placement**, or
  just use the bandwidth panel instead.
- **Inbound data port.** The worker also *listens* on the data port (default
  `50200`) for inter-stage tensors. If other nodes can't reach the device over
  Wi-Fi, multi-stage placements that route *into* it won't work.
- **Stay awake.** Without `termux-wake-lock` + battery-optimization off, Android
  suspends the CPU and the worker stalls/drops.
- **Throughput.** Expect ARM-CPU speeds; on SoCs without `FEAT_BF16` the client
  runs in fp32 (correct, just slower).

## Status

First cut — works end-to-end on a real tablet, with the rough edges above. Most
likely iteration points: inbound data-port reachability over Wi-Fi and
RAM/`--os-reserve-gb` tuning. Contributions and bug reports welcome.

---

## What's in this folder

| File | Role |
|------|------|
| `client.py`, `wire.py` | the worker — vendored copies of the fleet client (run CPU-only on Android) |
| `config.json` | controller host/ports + the public self-update source (no secrets) |
| `requirements-android.txt` | pinned deps (torch 2.13.0 CPU **aarch64**, transformers 5.12.1, …) |
| `setup.sh` | run **inside** the proot guest — builds a venv + installs deps |
| `start-client.sh` | run **inside** the proot guest — starts the worker → controller |
| `termux-bootstrap.sh` | run in **Termux** — one-shot: install proot+Debian, deploy, setup, shortcut |
| `termux-panel-setup.sh` | run in **Termux** — installs the full-screen bandwidth panel layout |
| `traffic_panel.py` | the standalone read-only fleet bandwidth dashboard |

> `client.py` / `wire.py` / `config.json` are vendored copies of the fleet
> client; they run unmodified on aarch64 Linux and self-update from the public
> repo. No credentials are required or stored anywhere in this folder.
