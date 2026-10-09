#!/usr/bin/env python3
"""predict.fun click terminal.

    python run_terminal.py            # live (keys from .env)
    python run_terminal.py --demo     # offline simulator, fake money
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
import webbrowser
from logging.handlers import RotatingFileHandler
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from aiohttp import web  # noqa: E402

from terminal.config import load_settings  # noqa: E402
from terminal.server import Terminal, make_app  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="offline simulator, no keys, fake money")
    ap.add_argument("--env", default=str(ROOT / ".env"))
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    (ROOT / "logs").mkdir(exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s | %(message)s")
    fh = RotatingFileHandler(ROOT / "logs" / "terminal.log", maxBytes=5_000_000, backupCount=3, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logging.basicConfig(level=logging.INFO, handlers=[fh, sh])
    logging.getLogger("aiohttp.access").setLevel(logging.WARNING)

    s = load_settings(args.env, demo=args.demo, port=args.port)
    import os
    os.chdir(ROOT)
    term = Terminal(s)
    url = f"http://{s.host}:{s.port}/"
    print(f"\n  predict.fun terminal ({'DEMO' if s.demo else 'LIVE'}) -> {url}\n")
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    web.run_app(make_app(term), host=s.host, port=s.port, print=None, access_log=None)


if __name__ == "__main__":
    main()
