"""Tests for config module."""

import pytest

from sova import config


@pytest.fixture
def isolated_config(monkeypatch, tmp_path):
    cfg_path = tmp_path / "config.json"
    monkeypatch.setattr(config, "_CONFIG_PATH", cfg_path)
    monkeypatch.setattr(config, "_METAL_PROBE_CACHE", {"value": None, "ts": 0.0})
    return cfg_path


def test_get_db_path_default():
    assert config.get_db_path() == config.DB_PATH


def test_get_memory_settings_runtime_snapshot(monkeypatch, isolated_config):
    monkeypatch.setattr(config, "_probe_total_ram_gib", lambda: 32.0)
    monkeypatch.setattr(config, "_probe_metal_ceiling_gib", lambda: 26.25)
    monkeypatch.setenv("SOVA_MEMORY_METAL_HEADROOM_GIB", "1.0")

    memory = config.get_memory_settings()

    assert memory["total_ram_gib"] == 32.0
    assert memory["metal_ceiling_gib"] == pytest.approx(25.25, abs=0.01)
    assert memory["reserve_index_gib"] == 4.0
    assert memory["reserve_search_gib"] == 10.0
    assert memory["swap_weight"] == 0.25
    assert memory["swap_boost_cap_gib"] == 2.0
    # Runtime path must not mutate config.json just to compute memory limits.
    assert not isolated_config.exists()


def test_get_memory_hard_cap_formula(monkeypatch, isolated_config):
    config._write_config(
        {
            "memory": {
                "reserve_index_gib": 4.0,
                "reserve_search_gib": 10.0,
                "swap_weight": 0.25,
                "swap_boost_cap_gib": 2.0,
            }
        }
    )
    monkeypatch.setenv("SOVA_MEMORY_METAL_CEILING_GIB", "26")
    monkeypatch.setenv("SOVA_MEMORY_AVAILABLE_GIB", "30")
    monkeypatch.setenv("SOVA_MEMORY_FREE_SWAP_GIB", "0")
    assert config.get_memory_hard_cap_gib("index") == 26.0

    monkeypatch.setenv("SOVA_MEMORY_AVAILABLE_GIB", "18")
    monkeypatch.setenv("SOVA_MEMORY_FREE_SWAP_GIB", "0")
    assert config.get_memory_hard_cap_gib("search") == 8.0


def test_get_memory_hard_cap_env_override(monkeypatch, isolated_config):
    monkeypatch.setenv("SOVA_MEMORY_HARD_CAP_GIB", "22.5")
    assert config.get_memory_hard_cap_gib("search") == 22.5


def test_get_memory_reserve_env_override(monkeypatch, isolated_config):
    config._write_config(
        {
            "memory": {
                "reserve_index_gib": 4.0,
                "reserve_search_gib": 10.0,
                "swap_weight": 0.25,
                "swap_boost_cap_gib": 2.0,
            }
        }
    )
    monkeypatch.setenv("SOVA_MEMORY_RESERVE_SEARCH_GIB", "7")
    assert config.get_memory_reserve_gib("search") == 7.0


def test_probe_metal_ceiling_parses_llama_output(monkeypatch):
    output = "ggml_metal_device_init: recommendedMaxWorkingSetSize  = 26800.60 MB"
    monkeypatch.setattr(config.sys, "platform", "darwin")
    monkeypatch.setattr(config.shutil, "which", lambda _cmd: "/usr/local/bin/llama-server")

    class Result:
        stdout = ""
        stderr = output

    monkeypatch.setattr(config.subprocess, "run", lambda *args, **kwargs: Result())
    value = config._probe_metal_ceiling_gib()
    assert value is not None
    assert value == pytest.approx(26.17, abs=0.01)


def test_runtime_metal_ceiling_uses_headroom(monkeypatch, isolated_config):
    monkeypatch.setattr(config, "_probe_metal_ceiling_gib", lambda: 26.2)
    monkeypatch.setenv("SOVA_MEMORY_METAL_HEADROOM_GIB", "1.5")
    assert config.get_metal_ceiling_gib() == pytest.approx(24.7, abs=0.01)


def test_runtime_metal_ceiling_no_probe_uses_ram_ratio(monkeypatch, isolated_config):
    monkeypatch.setattr(config, "_probe_metal_ceiling_gib", lambda: None)
    monkeypatch.setattr(config, "_probe_total_ram_gib", lambda: 32.0)
    assert config.get_metal_ceiling_gib() == 25.6


def test_runtime_metal_ceiling_probe_is_cached(monkeypatch, isolated_config):
    calls = {"count": 0}

    def probe():
        calls["count"] += 1
        return 26.0

    monkeypatch.setattr(config, "_probe_metal_ceiling_gib", probe)
    monkeypatch.setenv("SOVA_MEMORY_METAL_HEADROOM_GIB", "1.0")
    monkeypatch.setattr(config.time, "monotonic", lambda: 100.0)
    first = config.get_metal_ceiling_gib()
    second = config.get_metal_ceiling_gib()
    assert first == 25.0
    assert second == 25.0
    assert calls["count"] == 1


def test_effective_available_with_discounted_swap(monkeypatch, isolated_config):
    config._write_config(
        {
            "memory": {
                "metal_ceiling_gib": 30.0,
                "reserve_index_gib": 4.0,
                "reserve_search_gib": 10.0,
                "swap_weight": 0.25,
                "swap_boost_cap_gib": 2.0,
            }
        }
    )
    monkeypatch.setenv("SOVA_MEMORY_METAL_CEILING_GIB", "30")
    monkeypatch.setenv("SOVA_MEMORY_AVAILABLE_GIB", "20")
    monkeypatch.setenv("SOVA_MEMORY_FREE_SWAP_GIB", "8")
    assert config.get_effective_available_gib() == 22.0
    assert config.get_memory_hard_cap_gib("search") == 12.0


def test_swap_boost_respects_cap(monkeypatch, isolated_config):
    config._write_config(
        {
            "memory": {
                "metal_ceiling_gib": 30.0,
                "reserve_index_gib": 4.0,
                "reserve_search_gib": 10.0,
                "swap_weight": 1.0,
                "swap_boost_cap_gib": 1.5,
            }
        }
    )
    monkeypatch.setenv("SOVA_MEMORY_METAL_CEILING_GIB", "30")
    monkeypatch.setenv("SOVA_MEMORY_AVAILABLE_GIB", "20")
    monkeypatch.setenv("SOVA_MEMORY_FREE_SWAP_GIB", "10")
    assert config.get_effective_available_gib() == 21.5
