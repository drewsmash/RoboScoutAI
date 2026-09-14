"""Tests for chrome-less desktop window helpers."""

from __future__ import annotations

from desktop.app_window import _chrome_like_binaries, wait_for_server


def test_wait_for_server_timeout():
    assert wait_for_server("http://127.0.0.1:1/", timeout=0.3) is False


def test_chrome_like_binaries_are_strings():
    bins = _chrome_like_binaries()
    assert isinstance(bins, list)
    assert all(isinstance(b, str) and b for b in bins)
