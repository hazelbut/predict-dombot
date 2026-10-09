"""Binance spot BTCUSDT market-order flow per second (aggressor buys vs sells, in USDT) for the history period.

    python btc_flow.py            # days covered by the markets table (optional input for dog_history.py)

This is the part of the order-book picture that HAS a history: every trade with its aggressor side
(data.binance.vision aggTrades archives). Resting orders (walls, pulls, book imbalance) have no public history;
those are tested with dom_recorder.py + dom_replay.py.

Only full days from the daily archives (published with a ~1 day delay); stored in table btc_flow_1s
(ts, buy_usd, sell_usd) of the same SQLite. Resumable. One day is a ~50-150 MB download.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import io
import sqlite3
import sys
import tempfile
import time
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import aiohttp

from btc1s import to_sec

HERE = Path(__file__).resolve().parent
VISION = "https://data.binance.vision/data/spot/daily/aggTrades/BTCUSDT/BTCUSDT-aggTrades-{d}.zip"


def connect(path: Path) -> sqlite3.Connection:
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE IF NOT EXISTS btc_flow_1s (ts INTEGER PRIMARY KEY, buy_usd REAL, sell_usd REAL)")
    db.execute("CREATE TABLE IF NOT EXISTS btc_flow_days (day TEXT PRIMARY KEY, n INTEGER)")
    return db


def aggregate(fh: io.TextIOBase) -> dict[int, list[float]]:
    """aggTrades CSV rows: id, price, qty, first id, last id, time, is_buyer_maker, best match.
    is_buyer_maker = True -> the aggressor SOLD."""
    out: dict[int, list[float]] = {}
    for row in csv.reader(fh):
        if not row or not row[0][:1].isdigit():
            continue
        usd = float(row[1]) * float(row[2])
        sec = to_sec(row[5])
        r = out.get(sec)
        if r is None:
            r = out[sec] = [0.0, 0.0]
        if row[6].strip().lower() in ("true", "1"):
            r[1] += usd
        else:
            r[0] += usd
    return out


def parse_zip(path: Path) -> dict[int, list[float]]:
    with zipfile.ZipFile(path) as z:
        with z.open(z.namelist()[0]) as fh:
            return aggregate(io.TextIOWrapper(fh, "utf-8"))


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=str(HERE / "data" / "pf_hist.sqlite3"))
    a = ap.parse_args()
    db = connect(Path(a.db))
    r = db.execute("SELECT MIN(start_s), MAX(end_s) FROM markets WHERE complete=1").fetchone()
    if not r or r[0] is None:
        print("в базе нет рынков — сначала collect.py")
        return 1
    d0 = datetime.fromtimestamp(r[0] - 3600, timezone.utc).date()
    d1 = datetime.fromtimestamp(min(r[1], time.time()), timezone.utc).date()
    have = {row[0] for row in db.execute("SELECT day FROM btc_flow_days WHERE n > 0")}
    days = [d0 + timedelta(days=i) for i in range((d1 - d0).days + 1)]
    print(f"поток BTC: {d0} … {d1}, дней {len(days)}, уже есть {len(have & {str(d) for d in days})}", flush=True)
    async with aiohttp.ClientSession() as s:
        for d in days:
            ds = str(d)
            if ds in have:
                continue
            t0 = time.time()
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "a.zip"
                try:
                    async with s.get(VISION.format(d=ds), timeout=aiohttp.ClientTimeout(total=900)) as resp:
                        if resp.status != 200:
                            print(f"  {ds}: архива ещё нет (HTTP {resp.status}) — пропуск", flush=True)
                            continue
                        with path.open("wb") as fh:
                            async for chunk in resp.content.iter_chunked(1 << 20):
                                fh.write(chunk)
                except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                    print(f"  {ds}: ошибка загрузки {exc!r} — пропуск", flush=True)
                    continue
                mb = path.stat().st_size / 1e6
                flow = await asyncio.to_thread(parse_zip, path)
            db.executemany("INSERT OR REPLACE INTO btc_flow_1s VALUES (?,?,?)",
                           [(sec, round(b, 2), round(x, 2)) for sec, (b, x) in flow.items()])
            db.execute("INSERT OR REPLACE INTO btc_flow_days VALUES (?,?)", (ds, len(flow)))
            db.commit()
            print(f"  {ds}: {len(flow)} сек, {mb:.0f} МБ, {time.time() - t0:.0f} с", flush=True)
    n = db.execute("SELECT COUNT(*) FROM btc_flow_1s").fetchone()[0]
    print(f"готово: {n} секунд потока")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
