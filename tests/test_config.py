"""Tests for config module."""

from pathlib import Path

from sova import config


def test_get_db_path_default():
    config.set_db_path(None)
    assert config.get_db_path() == config.DB_PATH


def test_set_db_path_override():
    try:
        config.set_db_path("/tmp/custom-sova.db")
        assert config.get_db_path() == Path("/tmp/custom-sova.db")
    finally:
        config.set_db_path(None)
