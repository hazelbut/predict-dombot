"""Binance BTCUSDT 1-second closes for the period covered by the predict.fun history.

    python btc1s.py            # whatever the markets table covers (+1 h before for volatility)

Daily archives from data.binance.vision; days not published yet come from the REST API (1000 s per call).
Stored in the same SQLite (table btc_1s). Resumable.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import io
import sqlite3
import sys
import time
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import aiohttp

HERE = Path(__file__).resolve().parent
VISION = "https://data.binance.vision/data/spot/daily/klines/BTCUSDT/1s/BTCUSDT-1s-{d}.zip"
REST = "https://api.binance.com/api/v3/klines"


def to_sec(v: str | int | float) -> int:
    x = int(float(v))
    if x > 10 ** 14:      # microseconds (spot archives since 2025)
        return x // 1_000_000
    if x > 10 ** 11:      # milliseconds
        return x // 1000
    return x


def parse_zip(blob: bytes) -> list[tuple[int, float]]:
    out = []
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        name = z.namelist()[0]
        with z.open(name) as fh:
            for row in csv.reader(io.TextIOWrapper(fh, "utf-8")):
                if not row or not row[0][:1].isdigit():
                    continue
                out.append((to_sec(row[0]), float(row[4])))
    return out


def connect(path: Path) -> sqlite3.Connection:
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE IF NOT EXISTS btc_1s (ts INTEGER PRIMARY KEY, px REAL)")
    db.execute("CREATE TABLE IF NOT EXISTS btc_days (day TEXT PRIMARY KEY, n INTEGER)")
    return db


async def rest_range(session: aiohttp.ClientSession, t0: int, t1: int) -> list[tuple[int, float]]:
    out: list[tuple[int, float]] = []
    t = t0
    while t < t1:
        params = {"symbol": "BTCUSDT", "interval": "1s", "startTime": str(t * 1000), "limit": "1000"}
        for attempt in range(5):
            try:
                async with session.get(REST, params=params, timeout=aiohttp.ClientTimeout(total=20)) as r:
                    if r.status == 429 or r.status >= 500:
                        await asyncio.sleep(5 * (attempt + 1))
                        continue
                    rows = await r.json()
                    break
            except (aiohttp.ClientError, asyncio.TimeoutError):
                await asyncio.sleep(2 * (attempt + 1))
        else:
            raise RuntimeError("Binance REST недоступен")
        if not rows:
            t += 1000
            continue
        for k in rows:
            out.append((to_sec(k[0]), float(k[4])))
        t = out[-1][0] + 1
        await asyncio.sleep(0.15)
    return [x for x in out if t0 <= x[0] < t1]


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=str(HERE / "data" / "pf_hist.sqlite3"))
    a = ap.parse_args()
    db = connect(Path(a.db))
    r = db.execute("SELECT MIN(start_s), MAX(end_s) FROM markets WHERE complete=1").fetchone()
    if not r or r[0] is None:
        print("в базе нет рынков — сначала collect.py")
        return 1
    t0, t1 = r[0] - 3600, min(r[1] + 60, int(time.time()) - 5)
    d0 = datetime.fromtimestamp(t0, timezone.utc).date()
    d1 = datetime.fromtimestamp(t1, timezone.utc).date()
    have = {row[0] for row in db.execute("SELECT day FROM btc_days WHERE n >= 86000")}
    days = [d0 + timedelta(days=i) for i in range((d1 - d0).days + 1)]
    print(f"BTC 1s: {d0} … {d1}, дней {len(days)}, уже есть {len(have & {str(d) for d in days})}", flush=True)
    async with aiohttp.ClientSession() as s:
        for d in days:
            ds = str(d)
            if ds in have:
                continue
            day_t0 = int(datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp())
            rows: list[tuple[int, float]] = []
            try:
                async with s.get(VISION.format(d=ds), timeout=aiohttp.ClientTimeout(total=120)) as resp:
                    if resp.status == 200:
                        rows = parse_zip(await resp.read())
            except (aiohttp.ClientError, asyncio.TimeoutError):
                rows = []
            src = "архив"
            if len(rows) < 80_000:          # not published yet (today / yesterday) -> REST
                rows = await rest_range(s, max(day_t0, t0), min(day_t0 + 86400, t1))
                src = "REST"
            db.executemany("INSERT OR REPLACE INTO btc_1s VALUES (?,?)", rows)
            full = day_t0 + 86400 <= t1 and src == "архив"
            db.execute("INSERT OR REPLACE INTO btc_days VALUES (?,?)", (ds, len(rows) if full else 0))
            db.commit()
            print(f"  {ds}: {len(rows)} сек ({src})", flush=True)
    n, lo, hi = db.execute("SELECT COUNT(*), MIN(ts), MAX(ts) FROM btc_1s").fetchone()
    print(f"готово: {n} секунд, {datetime.fromtimestamp(lo, timezone.utc):%Y-%m-%d %H:%M} … "
          f"{datetime.fromtimestamp(hi, timezone.utc):%Y-%m-%d %H:%M} UTC")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
