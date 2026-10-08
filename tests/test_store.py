"""Unit tests for WordIdGenerator and ObjectStore."""

import pytest
from integrum.store import ObjectStore, WordIdGenerator


def test_word_id_generator():
    gen = WordIdGenerator(seed=42)
    id1 = gen.generate()
    id2 = gen.generate()
    assert "_" in id1
    assert "_" in id2
    assert id1 != id2
    # Verify word characters only
    parts = id1.split("_")
    assert len(parts) >= 2
    for p in parts:
        assert p.isalnum()


def test_object_store_register_and_get():
    store = ObjectStore()
    sample_data = {"key": "value", "numbers": [1, 2, 3]}

    obj_id = store.register(sample_data, label="sample")
    assert store.has(obj_id)
    retrieved = store.get(obj_id)
    assert retrieved == sample_data


def test_object_store_dot_notation():
    store = ObjectStore()
    
    class Inner:
        def __init__(self):
            self.value = 42

    class Outer:
        def __init__(self):
            self.inner = Inner()
            self.mapping = {"child": "found_me"}

    obj_id = store.register(Outer())
    assert store.get(f"{obj_id}.inner.value") == 42
    assert store.get(f"{obj_id}.mapping.child") == "found_me"
    assert store.resolve(f"{obj_id}.inner.value") == 42


def test_object_store_tuple_indexing():
    store = ObjectStore()
    data = (100, 200, 300)
    obj_id = store.register(data)

    assert store.get(f"{obj_id}[0]") == 100
    assert store.get(f"{obj_id}_1") == 200
    assert store.get(f"{obj_id}[2]") == 300


def test_object_store_session_isolation():
    store = ObjectStore()
    id1 = store.register("user_1_secret", session_id="session_alpha")
    id2 = store.register("user_2_secret", session_id="session_beta")

    # Alpha sees only id1
    assert store.has(id1, session_id="session_alpha")
    assert not store.has(id2, session_id="session_alpha")
    assert store.get(id1, session_id="session_alpha") == "user_1_secret"

    # Beta sees only id2
    assert store.has(id2, session_id="session_beta")
    assert not store.has(id1, session_id="session_beta")
    assert store.get(id2, session_id="session_beta") == "user_2_secret"


def test_object_store_delete_and_list():
    store = ObjectStore()
    id1 = store.register({"a": 1})
    id2 = store.register({"b": 2})

    listed = store.list_objects()
    assert id1 in listed
    assert id2 in listed

    assert store.delete(id1)
    assert not store.has(id1)
    assert store.has(id2)
