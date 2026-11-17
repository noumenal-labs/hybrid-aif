"""Tests for bounded snapshot retention and unhashable System cleanup in CompiledSystem."""

import dataclasses
import gc

import hybrid_aif.system.compiled as compiled_mod
from hybrid_aif.system import System


def test_snapshot_store_evicts_least_recently_used_entries():
    store = compiled_mod._SnapshotStore(limit=4)
    for i in range(10):
        store[("infer_states", i)] = {"arg": i}
    assert len(store) == 4
    assert list(store.keys()) == [("infer_states", i) for i in range(6, 10)]
    assert store.get(("infer_states", 6)) == {"arg": 6}
    store[("infer_states", 10)] = {"arg": 10}
    assert ("infer_states", 6) in store
    assert ("infer_states", 7) not in store


def test_unhashable_system_evicts_id_keyed_snapshots_on_gc():
    base = System(B_deps=((0,),), controlled=(False,), policy_len=1, A_deps=((0,),))
    unhashable = dataclasses.replace(base, scoring=(("custom", {"unhashable": 1}),))
    ident = id(unhashable)
    compiled = compiled_mod.CompiledSystem(unhashable)
    assert compiled._watched["infer_states"]._key == ("infer_states", ident)
    compiled_mod._SNAPSHOTS[("infer_states", ident)] = {"dummy": 1}
    assert ("infer_states", ident) in compiled_mod._SNAPSHOTS

    del compiled
    del unhashable
    gc.collect()

    assert ("infer_states", ident) not in compiled_mod._SNAPSHOTS
