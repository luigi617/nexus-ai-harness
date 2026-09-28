from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from nexus_ai_harness.core.persistable import (
    PersistenceWarning,
    dump,
    persistable,
    persistable_class,
    persisted_name,
    restore,
)


@persistable("tests.persistable.counter")
@dataclass
class Counter:
    calls: int = 0
    tags: list[str] = field(default_factory=list)


@persistable("tests.persistable.custom")
class Custom:
    def __init__(self, value: str = "") -> None:
        self.value = value

    def to_dict(self) -> dict:
        return {"v": self.value}

    @classmethod
    def from_dict(cls, data: dict) -> Custom:
        return cls(data["v"])


@persistable("tests.persistable.raising")
class Raising:
    """Custom hooks that raise errors other than TypeError/ValueError."""

    def to_dict(self) -> dict:
        raise RuntimeError("to_dict exploded")

    @classmethod
    def from_dict(cls, data: dict) -> Raising:
        data["missing"]  # KeyError, as on a snapshot saved before a field existed
        return cls()


@persistable("tests.persistable.opaque")
@dataclass
class Opaque:
    handle: object = None


def test_registry_lookup_both_ways():
    assert persisted_name(Counter) == "tests.persistable.counter"
    assert persistable_class("tests.persistable.counter") is Counter
    assert persisted_name(int) is None
    assert persistable_class("tests.persistable.unknown") is None


def test_dataclass_roundtrip():
    data = dump(Counter(calls=3, tags=["a"]))
    assert data == {"calls": 3, "tags": ["a"]}
    assert restore("tests.persistable.counter", data) == Counter(3, ["a"])


def test_restore_ignores_fields_the_class_no_longer_declares():
    restored = restore("tests.persistable.counter", {"calls": 1, "removed": True})
    assert restored == Counter(calls=1)


def test_custom_to_dict_from_dict_roundtrip():
    data = dump(Custom("x"))
    assert data == {"v": "x"}
    restored = restore("tests.persistable.custom", data)
    assert isinstance(restored, Custom)
    assert restored.value == "x"


def test_dump_skips_non_json_state_with_warning():
    with pytest.warns(PersistenceWarning, match="could not be serialized"):
        assert dump(Opaque(handle=object())) is None


def test_dump_skips_unregistered_with_warning():
    @dataclass
    class Plain:
        x: int = 0

    with pytest.warns(PersistenceWarning, match="non-persistable"):
        assert dump(Plain()) is None


def test_restore_unknown_name_warns():
    with pytest.warns(PersistenceWarning, match="no class is registered"):
        assert restore("tests.persistable.missing", {}) is None


def test_restore_bad_data_warns():
    with pytest.warns(PersistenceWarning, match="cannot restore"):
        assert restore("tests.persistable.counter", ["not", "a", "dict"]) is None


def test_non_dataclass_without_hooks_rejected():
    with pytest.raises(TypeError, match="dataclass"):

        @persistable("tests.persistable.bad")
        class Bad:
            pass


def test_name_clash_rejected():
    with pytest.raises(ValueError, match="already used"):

        @persistable("tests.persistable.counter")
        @dataclass
        class Other:
            pass


def test_empty_name_rejected():
    with pytest.raises(ValueError, match="non-empty"):
        persistable(" ")


def test_dump_skips_state_whose_to_dict_raises_anything():
    with pytest.warns(PersistenceWarning, match="to_dict exploded"):
        assert dump(Raising()) is None


def test_restore_skips_state_whose_from_dict_raises_anything():
    with pytest.warns(PersistenceWarning, match="cannot restore.*missing"):
        assert restore("tests.persistable.raising", {}) is None
