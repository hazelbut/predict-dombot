"""Fair probability of UP from Binance BTCUSDT 1 s prices.

P(up) = Phi( ln(S_t / K) / (k * sigma * sqrt(tau)) )
  S_t   price at instant t (close of the 1 s kline that ended at t)
  K     start price of the window (Chainlink, from predict.fun)
  sigma per-second volatility: blend of realised vol over the last 15 and 60 minutes
  tau   seconds to expiry, k a scale fitted on the first half of the data only
"""

from __future__ import annotations

import sqlite3

import numpy as np

SQ2 = np.sqrt(2.0)


def ncdf(z: np.ndarray | float) -> np.ndarray:
    """Standard normal CDF (Abramowitz-Stegun 7.1.26 on erf, |err| < 1.5e-7)."""
    z = np.asarray(z, dtype=np.float64)
    x = np.abs(z) / SQ2
    t = 1.0 / (1.0 + 0.3275911 * x)
    poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))))
    erf = 1.0 - poly * np.exp(-x * x)
    return 0.5 * (1.0 + np.sign(z) * erf)


class Btc:
    def __init__(self, ts: np.ndarray, px: np.ndarray, floor: float = 1e-5) -> None:
        ts = np.asarray(ts, dtype=np.int64)
        px = np.asarray(px, dtype=np.float64)
        self.t0 = int(ts[0])
        n = int(ts[-1]) - self.t0 + 1
        arr = np.full(n, np.nan)
        arr[ts - self.t0] = px
        # forward fill gaps
        idx = np.where(np.isnan(arr), 0, np.arange(n))
        np.maximum.accumulate(idx, out=idx)
        self.arr = arr[idx]
        self.have = np.zeros(n, dtype=bool)
        self.have[ts - self.t0] = True
        lr = np.zeros(n)
        lr[1:] = np.diff(np.log(self.arr))
        self.cs2 = np.concatenate([[0.0], np.cumsum(lr * lr)])
        self.floor = floor

    @classmethod
    def from_db(cls, db: sqlite3.Connection) -> "Btc | None":
        rows = db.execute("SELECT ts, px FROM btc_1s ORDER BY ts").fetchall()
        if len(rows) < 1000:
            return None
        a = np.asarray(rows, dtype=np.float64)
        return cls(a[:, 0].astype(np.int64), a[:, 1])

    @property
    def t1(self) -> int:
        return self.t0 + len(self.arr) - 1

    def _i(self, t) -> np.ndarray:
        """index of the kline that closed at instant t (open time t-1)"""
        return np.clip(np.asarray(t, dtype=np.int64) - 1 - self.t0, 0, len(self.arr) - 1)

    def price_at(self, t) -> np.ndarray:
        return self.arr[self._i(t)]

    def coverage(self, t0: int, t1: int) -> float:
        i0, i1 = max(0, t0 - self.t0), min(len(self.have), t1 - self.t0)
        return float(self.have[i0:i1].mean()) if i1 > i0 else 0.0

    def _var(self, i: np.ndarray, win: int) -> np.ndarray:
        j = np.maximum(i - win, 0)
        n = np.maximum(i - j, 1)
        return (self.cs2[i + 1] - self.cs2[j + 1]) / n

    def sigma(self, t) -> np.ndarray:
        i = self._i(t)
        v = 0.5 * self._var(i, 900) + 0.5 * self._var(i, 3600)
        return np.sqrt(np.maximum(v, self.floor ** 2))

    def p_up(self, t, strike: float, end_s: int, k: float = 1.0) -> np.ndarray:
        t = np.asarray(t, dtype=np.float64)
        tau = np.maximum(end_s - t, 0.5)
        z = np.log(self.price_at(t) / strike) / (k * self.sigma(t) * np.sqrt(tau))
        return ncdf(z)
