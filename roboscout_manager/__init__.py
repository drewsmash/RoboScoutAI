"""RoboScoutAI desktop manager: installer, launcher and updater.

Pure standard library so the frozen ``RoboScoutAI.exe`` manager stays tiny and
stable. The manager never runs the FastAPI app in-place: each app build lives in
``app\\<version>\\RoboScoutAI-app.exe`` and ``current.json`` selects the active one.
"""

from __future__ import annotations

__version__ = "0.6.0"
MANAGER_VERSION = __version__

APP_NAME = "RoboScoutAI"
PUBLISHER = "RoboScoutAI"
DEFAULT_GITHUB_REPO = "drewsmash/RoboScoutAI"
DEFAULT_GIT_REMOTE = f"https://github.com/{DEFAULT_GITHUB_REPO}.git"
DEFAULT_CHANNEL = "main"

# Installed file names inside %LOCALAPPDATA%\RoboScoutAI\.
MANAGER_EXE_NAME = "RoboScoutAI.exe"
INSTALLED_APP_EXE_NAME = "RoboScoutAI-app.exe"
CURRENT_JSON = "current.json"
MANAGER_JSON = "manager.json"
UPDATE_LOG = "update.log"
STATUS_JSON = "update-status.json"
IPC_JSON = "ipc.json"
APP_DIR_NAME = "app"
CACHE_DIR_NAME = "update-cache"

# Release / update-channel asset names (RoboScoutAI only).
APP_ASSET_NAME = "RoboScoutAI-app-windows-x64.exe"
MANAGER_ASSET_NAME = "RoboScoutAI.exe"
SETUP_ASSET_NAME = "RoboScoutAI-Setup.exe"
LEGACY_APP_ASSET_NAME = "RoboScoutAI-windows-x64.exe"
MANIFEST_NAME = "manifest.json"
RELEASE_ARTIFACTS_DIR = "release-artifacts"
UPDATE_CHANNEL_FILE = "update-channel.txt"

# How many side-by-side app versions to keep (current + previous for rollback).
KEEP_VERSIONS = 2

# Exit code the managed app uses to ask the manager to relaunch the current version.
RELAUNCH_EXIT_CODE = 75

__all__ = [
    "APP_ASSET_NAME",
    "APP_NAME",
    "DEFAULT_CHANNEL",
    "DEFAULT_GITHUB_REPO",
    "DEFAULT_GIT_REMOTE",
    "INSTALLED_APP_EXE_NAME",
    "KEEP_VERSIONS",
    "MANAGER_ASSET_NAME",
    "MANAGER_EXE_NAME",
    "MANAGER_VERSION",
    "MANIFEST_NAME",
    "RELAUNCH_EXIT_CODE",
    "SETUP_ASSET_NAME",
    "__version__",
]
