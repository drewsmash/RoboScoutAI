"""Entry point for ``RoboScoutAI-Setup.exe``: the manager CLI forced into ``--install``.

The Setup binary is the manager code plus a ``payload/`` folder (app exe,
manager exe, manifest.json) bundled by ``packaging/setup.spec``. Running it
installs into %LOCALAPPDATA%\\RoboScoutAI, registers the uninstall entry, creates
shortcuts and launches the app. Any manager flag still works, e.g.
``RoboScoutAI-Setup.exe --silent --no-desktop-shortcut``.
"""

from __future__ import annotations

import sys

from roboscout_manager.cli import main as cli_main

_PASSTHROUGH = {"--version", "--uninstall", "--repair", "--check", "--update", "--rollback", "--list", "--status"}


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not any(a in _PASSTHROUGH for a in args) and "--install" not in args:
        args = ["--install", *args]
    return cli_main(args)


if __name__ == "__main__":
    sys.exit(main())
