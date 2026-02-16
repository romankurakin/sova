"""Tests for config module."""

from sova import config


def test_get_db_path_default():
    assert config.get_db_path() == config.DB_PATH
