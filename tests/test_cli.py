"""Tests for cli module."""

from sova.cli import fmt_duration, fmt_size


class TestFmtSize:
    def test_zero_bytes(self):
        assert fmt_size(0) == "-"

    def test_zero_bytes_dim(self):
        assert fmt_size(0, dim_zero=True) == "-"

    def test_bytes(self):
        assert fmt_size(500) == "500 B"

    def test_kilobytes(self):
        result = fmt_size(2048)
        assert "KB" in result
        assert "2.0" in result

    def test_megabytes(self):
        result = fmt_size(2 * 1024 * 1024)
        assert "MB" in result
        assert "2.0" in result

    def test_just_over_kb_boundary(self):
        result = fmt_size(1024)
        assert "KB" in result

    def test_just_over_mb_boundary(self):
        result = fmt_size(1024 * 1024)
        assert "MB" in result


class TestFmtDuration:
    def test_seconds(self):
        result = fmt_duration(30.0)
        assert "s" in result
        assert "30.0" in result

    def test_minutes(self):
        result = fmt_duration(120.0)
        assert "m" in result
        assert "2.0" in result

    def test_hours(self):
        result = fmt_duration(7200.0)
        assert "h" in result
        assert "2.0" in result

    def test_just_under_minute(self):
        result = fmt_duration(59.9)
        assert "s" in result

    def test_just_at_minute(self):
        result = fmt_duration(60.0)
        assert "m" in result

    def test_just_at_hour(self):
        result = fmt_duration(3600.0)
        assert "h" in result
