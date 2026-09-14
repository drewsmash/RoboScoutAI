"""Tests for chrome-less desktop window helpers."""

from __future__ import annotations

import sys

from desktop.app_window import (
    _chrome_like_binaries,
    _should_try_pywebview,
    wait_for_server,
)


def test_wait_for_server_timeout():
    assert wait_for_server("http://127.0.0.1:1/", timeout=0.3) is False


def test_chrome_like_binaries_are_strings():
    bins = _chrome_like_binaries()
    assert isinstance(bins, list)
    assert all(isinstance(b, str) and b for b in bins)


def test_should_try_pywebview_skips_frozen_windows(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.delenv("RAMSCOUT_FORCE_WEBVIEW", raising=False)
    assert _should_try_pywebview() is False


def test_should_try_pywebview_force_env(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setenv("RAMSCOUT_FORCE_WEBVIEW", "1")
    assert _should_try_pywebview() is True


def test_should_try_pywebview_allows_linux(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delenv("RAMSCOUT_FORCE_WEBVIEW", raising=False)
    assert _should_try_pywebview() is True
