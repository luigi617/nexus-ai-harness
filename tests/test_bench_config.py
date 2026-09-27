from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from benchmarks.core.config import BenchmarkConfig, build_config, load_config_file


@dataclass
class _Cfg(BenchmarkConfig):
    """Every coercion path in one dataclass."""

    name: str = "default"
    steps: int = 30
    ratio: float = 0.5
    on: bool = False
    split: str | None = None
    tags: list[str] = field(default_factory=list)
    nums: list[int] = field(default_factory=list)
    pair: tuple[str, ...] = ()


def test_defaults_when_no_overrides():
    cfg = build_config(_Cfg, {})
    assert (cfg.name, cfg.steps, cfg.ratio, cfg.on, cfg.split, cfg.tags) == (
        "default",
        30,
        0.5,
        False,
        None,
        [],
    )


def test_string_int_float_pass_through():
    cfg = build_config(_Cfg, {"name": "airline", "steps": "40", "ratio": "0.75"})
    assert cfg.name == "airline"
    assert cfg.steps == 40
    assert cfg.ratio == 0.75


@pytest.mark.parametrize("val", ["true", "TRUE", "1", "yes", "on"])
def test_bool_truthy_forms(val):
    assert build_config(_Cfg, {"on": val}).on is True


@pytest.mark.parametrize("val", ["false", "0", "no", "off"])
def test_bool_falsy_forms(val):
    assert build_config(_Cfg, {"on": val}).on is False


def test_bool_rejects_garbage():
    with pytest.raises(ValueError, match="expected a boolean"):
        build_config(_Cfg, {"on": "maybe"})


def test_optional_string_none_forms():
    assert build_config(_Cfg, {"split": "none"}).split is None
    assert build_config(_Cfg, {"split": "null"}).split is None
    assert build_config(_Cfg, {"split": ""}).split is None
    assert build_config(_Cfg, {"split": "validation"}).split == "validation"


def test_list_from_csv():
    cfg = build_config(_Cfg, {"tags": "simple,parallel,irrelevance"})
    assert cfg.tags == ["simple", "parallel", "irrelevance"]


def test_empty_list_from_empty_csv():
    assert build_config(_Cfg, {"tags": ""}).tags == []


def test_tuple_field_from_csv():
    # tuple[X, ...] reports args as (X, Ellipsis); coercion must not choke on it.
    cfg = build_config(_Cfg, {"pair": "a,b,c"})
    assert cfg.pair == ("a", "b", "c")


def test_typed_list_drops_blanks_before_coercing():
    # Blank entries are dropped before element coercion, so int([]) is never hit.
    cfg = build_config(_Cfg, {"nums": "1,,2"})
    assert cfg.nums == [1, 2]
    assert all(isinstance(n, int) for n in cfg.nums)


def test_unknown_key_raises():
    with pytest.raises(ValueError, match="unknown config key 'nope'"):
        build_config(_Cfg, {"nope": "x"})


def test_partial_override_keeps_other_defaults():
    cfg = build_config(_Cfg, {"steps": "5"})
    assert cfg.steps == 5
    assert cfg.name == "default"  # untouched
    assert cfg.tags == []


def test_get_benchmark_applies_overrides():
    # Full path through the registry: get_benchmark builds the config too.
    from benchmarks.core.registry import get_benchmark

    bench = get_benchmark("humaneval", {"timeout_s": "5"})
    assert bench.config.timeout_s == 5.0


def test_get_benchmark_rejects_bad_override():
    from benchmarks.core.registry import get_benchmark

    with pytest.raises(ValueError, match="unknown config key"):
        get_benchmark("humaneval", {"nope": "x"})


# --- native (YAML) values applied without string coercion -------------------


def test_native_values_pass_through_without_coercion():
    # YAML yields typed values (int, bool, list, None) — used as-is.
    cfg = build_config(
        _Cfg, {"steps": 40, "on": True, "tags": ["a", "b"], "split": None}
    )
    assert cfg.steps == 40
    assert cfg.on is True
    assert cfg.tags == ["a", "b"]
    assert cfg.split is None


def test_native_and_string_overrides_mix():
    # File gives native ints; a --set string coerces alongside them.
    cfg = build_config(_Cfg, {"steps": 40, "ratio": "0.25"})
    assert cfg.steps == 40
    assert cfg.ratio == 0.25


# --- YAML config file loading -----------------------------------------------


def test_load_config_file_returns_named_section(tmp_path):
    path = tmp_path / "bench.yaml"
    path.write_text(
        "tau-bench:\n  env: airline\n  max_steps: 40\nswe-bench:\n  max_steps: 10\n"
    )
    assert load_config_file(path, "tau-bench") == {"env": "airline", "max_steps": 40}


def test_load_config_file_missing_section_is_empty(tmp_path):
    path = tmp_path / "bench.yaml"
    path.write_text("tau-bench:\n  env: airline\n")
    assert load_config_file(path, "gpqa") == {}


def test_load_config_file_empty_file_is_empty(tmp_path):
    path = tmp_path / "bench.yaml"
    path.write_text("")
    assert load_config_file(path, "gpqa") == {}


def test_load_config_file_rejects_non_mapping_top_level(tmp_path):
    path = tmp_path / "bench.yaml"
    path.write_text("- just\n- a\n- list\n")
    with pytest.raises(ValueError, match="must map benchmark"):
        load_config_file(path, "gpqa")


def test_load_config_file_rejects_non_mapping_section(tmp_path):
    path = tmp_path / "bench.yaml"
    path.write_text("gpqa: not-a-mapping\n")
    with pytest.raises(ValueError, match="must be a mapping"):
        load_config_file(path, "gpqa")


def test_file_then_build_config_end_to_end(tmp_path):
    path = tmp_path / "bench.yaml"
    path.write_text("humaneval:\n  timeout_s: 45\n")
    section = load_config_file(path, "humaneval")
    from benchmarks.core.registry import get_benchmark

    bench = get_benchmark("humaneval", section)
    assert bench.config.timeout_s == 45.0
