"""``python -m roboscout_manager`` / frozen ``RoboScoutAI.exe`` entry point."""

from __future__ import annotations

import sys

from roboscout_manager.cli import main

if __name__ == "__main__":
    sys.exit(main())
