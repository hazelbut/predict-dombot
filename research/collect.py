"""Download predict.fun trade history for finished BTC Up/Down markets into SQLite.

    python collect.py --days 14                 # 5m markets of the last 14 days
    python collect.py --days 14 --series 15m

Only the API key is used (PREDICT_API_KEY from the repo .env or the environment). Resumable:
markets already downloaded completely are skipped. Default pace 120 requests/min so the
terminal (same key, limit 240/min) keeps working.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiohttp

HERE = Path(__file__).resolve().parent
PERIOD = {"5m": 300, "15m": 900}
WEI = 10 ** 18

SCHEMA = [
    """CREATE TABLE IF NOT EXISTS markets (
        slug TEXT PRIMARY KEY, series TEXT, start_s INTEGER, end_s INTEGER, market_id INTEGER,
        start_price REAL, end_price REAL, winner TEXT, fee_bps INTEGER, status TEXT,
        n_trades INTEGER, complete INTEGER, missing INTEGER, fetched_s INTEGER, raw TEXT)""",
    """CREATE TABLE IF NOT EXISTS trades (
        market_id INTEGER, ts_ms INTEGER, outcome TEXT, side TEXT, price REAL, shares REAL,
        fee REAL, fee_type TEXT, tx TEXT,
        PRIMARY KEY (market_id, tx, ts_ms, outcome, side, price, shares))""",
    "CREATE INDEX IF NOT EXISTS ix_trades_m ON trades(market_id, ts_ms)",
]


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(path)
    for q in SCHEMA:
        c.execute(q)
    return c


def load_api_key() -> str:
    if os.environ.get("PREDICT_API_KEY"):
        return os.environ["PREDICT_API_KEY"].strip()
    for env in (HERE / ".env", HERE.parent / ".env", HERE.parent / "predict_terminal" / ".env",
                Path.home() / "predict_terminal" / ".env"):
        if env.exists():
            for line in env.read_text(encoding="utf-8", errors="ignore").splitlines():
                k, _, v = line.partition("=")
                if k.strip() == "PREDICT_API_KEY" and v.strip():
                    return v.strip().strip('"').strip("'")
    return ""


def num(v: Any, wei_if_above: float) -> float:
    """API amounts may come as plain decimals or wei strings; normalise to plain units."""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return 0.0
    return x / WEI if abs(x) > wei_if_above else x


def iso_ms(text: Any) -> int:
    try:
        return int(datetime.fromisoformat(str(text).replace("Z", "+00:00")).timestamp() * 1000)
    except ValueError:
        return 0


def side_name(o: dict[str, Any]) -> str:
    n = str((o or {}).get("name") or "").strip().lower()
    if n.startswith("up") or n == "yes":
        return "UP"
    if n.startswith("down") or n == "no":
        return "DOWN"
    return {1: "UP", 2: "DOWN"}.get(int((o or {}).get("indexSet") or 0), "?")


def parse_market(raw: dict[str, Any]) -> dict[str, Any] | None:
    ms = raw.get("markets") or []
    if not ms:
        return None
    m = ms[0]
    vd = m.get("variantData") or raw.get("variantData") or {}
    winner = None
    res = m.get("resolution")
    if isinstance(res, dict) and res.get("name"):
        winner = side_name(res)
    if winner is None:
        for o in m.get("outcomes") or []:
            if str(o.get("status") or "").upper() == "WON":
                winner = side_name(o)
    sp, ep = vd.get("startPrice"), vd.get("endPrice")
    return {"market_id": int(m.get("id") or 0),
            "start_price": float(sp) if sp not in (None, "") else None,
            "end_price": float(ep) if ep not in (None, "") else None,
            "winner": winner, "fee_bps": int(m.get("feeRateBps") or 0),
            "status": str(m.get("status") or "")}


def parse_match(x: dict[str, Any]) -> tuple | None:
    t = x.get("taker") or {}
    ts = iso_ms(x.get("executedAt"))
    price = num(x.get("priceExecuted", t.get("price")), 1.5)
    shares = num(x.get("amountFilled", t.get("amount")), 1e9)
    if not ts or not (0 < price < 1) or shares <= 0:
        return None
    side = "BUY" if str(t.get("quoteType") or "").lower() == "bid" else "SELL"
    fee = t.get("fee") or {}
    return (ts, side_name(t.get("outcome") or {}), side, price, shares,
            num(fee.get("amount"), 1e9), str(fee.get("type") or ""), str(x.get("transactionHash") or ""))


class Api:
    def __init__(self, base: str, key: str, rpm: int) -> None:
        self.base, self.key, self.gap = base.rstrip("/"), key, 60.0 / max(1, rpm)
        self.next_at = 0.0
        self.calls = 0
        self.session: aiohttp.ClientSession | None = None

    async def get(self, path: str, params: dict[str, str] | None = None) -> tuple[int, Any]:
        assert self.session is not None
        for attempt in range(6):
            wait = self.next_at - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)
            self.next_at = time.monotonic() + self.gap
            self.calls += 1
            try:
                async with self.session.get(self.base + path, params=params, headers={"x-api-key": self.key},
                                            timeout=aiohttp.ClientTimeout(total=20)) as r:
                    text = await r.text()
                    if r.status == 429 or r.status >= 500:
                        await asyncio.sleep(5 * (attempt + 1))
                        continue
                    try:
                        return r.status, json.loads(text) if text else None
                    except json.JSONDecodeError:
                        return r.status, text
            except (aiohttp.ClientError, asyncio.TimeoutError):
                await asyncio.sleep(3 * (attempt + 1))
        return 0, None


async def fetch_matches(api: Api, market_id: int, page: int) -> list[tuple] | None:
    out: list[tuple] = []
    after: str | None = None
    for _ in range(400):
        params = {"marketId": str(market_id), "first": str(page)}
        if after:
            params["after"] = after
        st, res = await api.get("/v1/orders/matches", params)
        if st != 200 or not isinstance(res, dict):
            return None
        rows = res.get("data") or []
        if rows and not after and all(int(((x.get("market") or {}).get("id")) or market_id) != market_id for x in rows):
            print("!!! API вернул сделки других рынков — фильтр marketId не работает, останавливаюсь")
            raise SystemExit(4)
        for x in rows:
            mid = int(((x.get("market") or {}).get("id")) or market_id)
            if mid != market_id:
                continue
            p = parse_match(x)
            if p:
                out.append(p)
        after = res.get("cursor")
        if not after or not rows:
            return out
    return out


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--series", default="5m", choices=sorted(PERIOD))
    ap.add_argument("--days", type=float, default=14)
    ap.add_argument("--rpm", type=int, default=120, help="requests per minute (key limit is 240, shared with the terminal)")
    ap.add_argument("--db", default=str(HERE / "data" / "pf_hist.sqlite3"))
    ap.add_argument("--api", default=os.environ.get("PREDICT_API_URL", "https://api.predict.fun"))
    ap.add_argument("--page", type=int, default=100)
    ap.add_argument("--refetch", action="store_true", help="download again even if complete")
    ap.add_argument("--probe", action="store_true", help="скачать один свежий рынок и показать, как разобраны данные")
    a = ap.parse_args()

    key = load_api_key()
    if not key:
        print("нет PREDICT_API_KEY: впиши его в .env в корне репозитория (см. .env.example)")
        return 2
    period = PERIOD[a.series]
    prefix = f"btc-updown-{a.series}"
    db = connect(Path(a.db))
    now = int(time.time())
    last_start = (now // period) * period - period          # newest window that has finished
    starts = list(range(last_start, int(now - a.days * 86400) - 1, -period))
    done = {r[0] for r in db.execute("SELECT slug FROM markets WHERE complete=1 OR missing=1")}
    todo = [s for s in starts if a.refetch or f"{prefix}-{s}" not in done]
    print(f"{prefix}: окон {len(starts)}, уже скачано {len(starts) - len(todo)}, качаю {len(todo)} "
          f"(~{len(todo) * 3 / a.rpm:.0f}+ мин при {a.rpm} запр/мин)", flush=True)

    api = Api(a.api, key, a.rpm)
    if a.probe:
        async with aiohttp.ClientSession() as session:
            api.session = session
            return await probe(api, prefix, last_start)
    t0 = time.time()
    n_tr = n_mk = n_miss = n_fail = 0
    async with aiohttp.ClientSession() as session:
        api.session = session
        for i, s in enumerate(todo, 1):
            slug = f"{prefix}-{s}"
            st, res = await api.get(f"/v1/categories/{slug}")
            if st == 404:
                db.execute("INSERT OR REPLACE INTO markets(slug, series, start_s, end_s, missing, complete, fetched_s) "
                           "VALUES (?,?,?,?,1,0,?)", (slug, a.series, s, s + period, now))
                n_miss += 1
            elif st == 401 or st == 403:
                print(f"API ответил {st}: ключ не принят", res)
                return 3
            elif st != 200 or not isinstance(res, dict) or not parse_market(res.get("data") or {}):
                n_fail += 1
            else:
                info = parse_market(res["data"])
                assert info is not None
                trades = await fetch_matches(api, info["market_id"], a.page)
                if trades is None and a.page > 50:
                    trades = await fetch_matches(api, info["market_id"], 50)
                complete = int(trades is not None and info["winner"] is not None)
                db.execute("INSERT OR REPLACE INTO markets VALUES (?,?,?,?,?,?,?,?,?,?,?,?,0,?,?)",
                           (slug, a.series, s, s + period, info["market_id"], info["start_price"], info["end_price"],
                            info["winner"], info["fee_bps"], info["status"], len(trades or []), complete, now,
                            json.dumps(res["data"])[:20000] if i == 1 else None))
                if trades:
                    db.executemany("INSERT OR IGNORE INTO trades VALUES (?,?,?,?,?,?,?,?,?)",
                                   [(info["market_id"], *t) for t in trades])
                n_mk += 1
                n_tr += len(trades or [])
                if trades is None:
                    n_fail += 1
            if i % 10 == 0 or i == len(todo):
                db.commit()
                el = time.time() - t0
                eta = el / i * (len(todo) - i)
                print(f"  {i}/{len(todo)}  рынков {n_mk}  сделок {n_tr}  нет рынка {n_miss}  ошибок {n_fail}  "
                      f"запросов {api.calls}  осталось ~{eta / 60:.0f} мин", flush=True)
    db.commit()
    summary(db, a.series)
    return 0


async def probe(api: Api, prefix: str, start: int) -> int:
    slug = f"{prefix}-{start}"
    st, res = await api.get(f"/v1/categories/{slug}")
    print(f"категория {slug}: HTTP {st}")
    info = parse_market((res or {}).get("data") or {}) if isinstance(res, dict) else None
    if not info:
        print("рынок не разобран, ответ:", json.dumps(res)[:1500] if res else res)
        return 1
    print("рынок:", info)
    st, res = await api.get("/v1/orders/matches", {"marketId": str(info["market_id"]), "first": "5"})
    print(f"сделки: HTTP {st}")
    rows = (res.get("data") or []) if isinstance(res, dict) else []
    if rows:
        x = dict(rows[0])
        x["market"] = {"id": (x.get("market") or {}).get("id")}
        print("сырой пример:", json.dumps(x, ensure_ascii=False)[:1500])
    for x in rows:
        print("  разобрано:", parse_match(x), " рынок", (x.get("market") or {}).get("id"))
    print("курсор следующей страницы:", (res or {}).get("cursor") if isinstance(res, dict) else None)
    return 0


def summary(db: sqlite3.Connection, series: str) -> None:
    r = db.execute("SELECT COUNT(*), SUM(n_trades), MIN(start_s), MAX(start_s), SUM(winner='UP'), SUM(winner='DOWN') "
                   "FROM markets WHERE series=? AND complete=1", (series,)).fetchone()
    if not r or not r[0]:
        print("готовых рынков нет")
        return
    d0 = datetime.fromtimestamp(r[2], timezone.utc).strftime("%Y-%m-%d %H:%M")
    d1 = datetime.fromtimestamp(r[3], timezone.utc).strftime("%Y-%m-%d %H:%M")
    print(f"\nв базе {series}: рынков {r[0]} ({d0} … {d1} UTC), сделок {r[1]}, UP выиграл {r[4]}, DOWN {r[5]}")
    # sanity: the tape just before expiry must agree with the winner (checks the price convention)
    agree = tot = 0
    for mid, win, end_s in db.execute("SELECT market_id, winner, end_s FROM markets WHERE series=? AND complete=1 "
                                      "AND n_trades>0", (series,)):
        row = db.execute("SELECT outcome, price FROM trades WHERE market_id=? AND ts_ms<=? ORDER BY ts_ms DESC LIMIT 1",
                         (mid, end_s * 1000)).fetchone()
        if not row:
            continue
        up_px = row[1] if row[0] == "UP" else 1 - row[1]
        if abs(up_px - 0.5) < 0.2:
            continue
        tot += 1
        agree += int((up_px > 0.5) == (win == "UP"))
    if tot:
        print(f"проверка цен: последняя сделка перед концом совпадает с исходом в {agree / tot:.0%} рынков ({tot})")
    fr = db.execute("SELECT AVG(CASE WHEN fee_type='SHARES' THEN fee * price ELSE fee END / (shares * MIN(price, 1 - price))), "
                    "COUNT(*) FROM trades WHERE fee > 0 AND shares > 0").fetchone()
    if fr and fr[1]:
        print(f"комиссия тейкера: {fr[0]:.2%} × min(p,1−p) за акцию — по {fr[1]} сделкам")

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
