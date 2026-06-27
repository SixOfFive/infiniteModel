"""Offline unit test for #39 data-parallel replication logic (no fleet needed).
Exercises Engine.replicas_of / replica_count / _pick_replica and _inflight_admit slot math
with stand-in replicas. Run: python test_replica_logic.py"""
import types
import server


def fake_replica(base, idx, active=0, queued=0, writer=True):
    key = base if idx == 0 else f"{base}#{idx}"
    return types.SimpleNamespace(friendly=key, base=base, replica_idx=idx,
                                 active=active, queued=queued,
                                 stage0_writer=(object() if writer else None))


def test_replicas_of_and_count():
    e = server.Engine()
    e.models = {}
    # one non-replicated model (base == friendly, base field "")
    solo = fake_replica("solo", 0); solo.base = ""        # legacy: base unset
    e.models["solo"] = solo
    # three replicas of qwen
    for i in range(3):
        e.models[fake_replica("qwen", i).friendly] = fake_replica("qwen", i)
    assert e.replica_count("solo") == 1, e.replica_count("solo")
    assert e.replica_count("qwen") == 3, e.replica_count("qwen")
    assert [m.replica_idx for m in e.replicas_of("qwen")] == [0, 1, 2]
    assert e.replica_count("missing") == 1   # >=1 floor
    print("OK replicas_of / replica_count")


def test_pick_least_loaded():
    e = server.Engine(); e.models = {}
    rs = [fake_replica("qwen", 0, active=1, queued=2),
          fake_replica("qwen", 1, active=0, queued=0),   # emptiest
          fake_replica("qwen", 2, active=1, queued=0)]
    for r in rs:
        e.models[r.friendly] = r
    pick = e._pick_replica("qwen")
    assert pick.replica_idx == 1, pick.replica_idx
    print("OK pick least-loaded")


def test_pick_round_robin_on_tie():
    e = server.Engine(); e.models = {}
    for i in range(3):
        r = fake_replica("qwen", i, active=0, queued=0)   # all tied
        e.models[r.friendly] = r
    picks = [e._pick_replica("qwen").replica_idx for _ in range(6)]
    assert picks == [0, 1, 2, 0, 1, 2], picks
    print("OK round-robin tie-break:", picks)


def test_pick_skips_dead_writer():
    e = server.Engine(); e.models = {}
    r0 = fake_replica("qwen", 0, active=5, queued=5, writer=False)  # least-loaded but DEAD
    r1 = fake_replica("qwen", 1, active=1, queued=0, writer=True)
    e.models[r0.friendly] = r0; e.models[r1.friendly] = r1
    assert e._pick_replica("qwen").replica_idx == 1
    print("OK skips dead-writer replica")


def test_pick_fallback_solo():
    e = server.Engine(); e.models = {}
    solo = fake_replica("solo", 0); solo.base = ""
    e.models["solo"] = solo
    assert e._pick_replica("solo") is solo
    assert e._pick_replica("nope") is None
    print("OK solo fallback")


def test_inflight_slots():
    server.INFLIGHT.clear()
    server.ENGINE_CONFIG["queue_depth"] = 2
    # 3 replicas => 3 running slots + 2 queued allowed = 5 in flight; 6th rejected.
    recs = []
    for _ in range(5):
        rec = server._inflight_admit("ip", "qwen", slots=3)
        assert rec is not None
        recs.append(rec)
        # simulate the first 3 acquiring a running slot
        if sum(1 for r in server.INFLIGHT.values() if r["state"] == "running") < 3:
            server._inflight_start(rec)
    assert server._inflight_admit("ip", "qwen", slots=3) is None, "6th should 503"
    running = sum(1 for r in server.INFLIGHT.values() if r["state"] == "running")
    queued = sum(1 for r in server.INFLIGHT.values() if r["state"] == "queued")
    assert running == 3 and queued == 2, (running, queued)
    print(f"OK inflight slots: running={running} queued={queued} (6th=503)")

    # single-slot model: 1 running + 2 queued = 3; 4th rejected.
    server.INFLIGHT.clear()
    for _ in range(3):
        rec = server._inflight_admit("ip", "solo", slots=1)
        assert rec is not None
        if sum(1 for r in server.INFLIGHT.values() if r["state"] == "running") < 1:
            server._inflight_start(rec)
    assert server._inflight_admit("ip", "solo", slots=1) is None, "4th should 503"
    print("OK inflight single-slot unchanged")


if __name__ == "__main__":
    test_replicas_of_and_count()
    test_pick_least_loaded()
    test_pick_round_robin_on_tie()
    test_pick_skips_dead_writer()
    test_pick_fallback_solo()
    test_inflight_slots()
    print("\nALL REPLICA-LOGIC TESTS PASSED")
