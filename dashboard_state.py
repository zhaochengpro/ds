"""Thread-safe dashboard state store for trading dashboard."""

from __future__ import annotations

from threading import RLock
from typing import Any, Dict, List, Optional

from analytics_utils import build_equity_series, build_equity_timeframes, utc_now_iso
from db import database


class _DashboardState:
    __slots__ = (
        "_lock",
        "_account",
        "_positions",
        "_strategy_batches",
        "_signals",
        "_equity_history",
    )

    def __init__(self) -> None:
        self._lock = RLock()
        self._account: Dict[str, Any] = {
            "account_value": 0.0,
            "available_cash": 0.0,
            "return_pct": 0.0,
            "sharpe_ratio": 0.0,
            "timestamp": utc_now_iso(),
        }
        self._positions: Dict[str, Any] = {
            "items": [],
            "timestamp": utc_now_iso(),
        }
        self._strategy_batches: List[Dict[str, Any]] = []
        self._signals: Dict[str, List[Dict[str, Any]]] = {}
        self._equity_history: List[Dict[str, Any]] = []

    def update_account_snapshot(
        self,
        account_value: float,
        available_cash: float,
        return_pct: float,
        sharpe_ratio: float,
    ) -> None:
        with self._lock:
            self._account = {
                "account_value": float(account_value or 0.0),
                "available_cash": float(available_cash or 0.0),
                "return_pct": float(return_pct or 0.0),
                "sharpe_ratio": float(sharpe_ratio or 0.0),
                "timestamp": utc_now_iso(),
            }
            self._equity_history.append(
                {
                    "timestamp": self._account["timestamp"],
                    "value": self._account["account_value"],
                }
            )
            if len(self._equity_history) > 5000:
                self._equity_history = self._equity_history[-5000:]

    def update_positions_snapshot(self, positions: Optional[List[Dict[str, Any]]]) -> None:
        with self._lock:
            self._positions = {
                "items": list(positions or []),
                "timestamp": utc_now_iso(),
            }

    def record_strategy_batch(self, batch: Any) -> None:
        timestamp = utc_now_iso()
        records: List[Dict[str, Any]] = []

        if isinstance(batch, dict):
            source_iterable = batch.values()
        elif isinstance(batch, list):
            source_iterable = batch
        else:
            source_iterable = []

        for item in source_iterable:
            if isinstance(item, dict):
                record = dict(item)
                record.setdefault("timestamp", timestamp)
                if "coin" in record and isinstance(record["coin"], str):
                    record["coin"] = record["coin"].upper()
                records.append(record)

        if not records:
            return

        with self._lock:
            self._strategy_batches.append({"timestamp": timestamp, "signals": records})
            if len(self._strategy_batches) > 50:
                self._strategy_batches = self._strategy_batches[-50:]

            for record in records:
                coin = str(record.get("coin", "")).upper()
                if not coin:
                    continue
                signal_copy = dict(record)
                self._signals.setdefault(coin, []).append(signal_copy)
                if len(self._signals[coin]) > 50:
                    self._signals[coin] = self._signals[coin][-50:]

    def record_strategy_signal(self, coin: str, signal: Dict[str, Any]) -> None:
        if not coin or not isinstance(signal, dict):
            return

        coin_key = coin.upper()
        signal_copy = dict(signal)
        signal_copy.setdefault("timestamp", utc_now_iso())
        if "coin" not in signal_copy:
            signal_copy["coin"] = coin_key
        else:
            signal_copy["coin"] = str(signal_copy["coin"]).upper()

        with self._lock:
            self._signals.setdefault(coin_key, []).append(signal_copy)
            if len(self._signals[coin_key]) > 50:
                self._signals[coin_key] = self._signals[coin_key][-50:]

    def get_account_snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._account)

    def get_positions_snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "timestamp": self._positions.get("timestamp"),
                "items": [dict(item) for item in self._positions.get("items", [])],
            }

    def get_strategy_batches(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [
                {
                    "timestamp": batch.get("timestamp"),
                    "signals": [dict(signal) for signal in batch.get("signals", [])],
                }
                for batch in self._strategy_batches
            ]

    def get_strategy_signals(self) -> Dict[str, List[Dict[str, Any]]]:
        with self._lock:
            return {
                coin: [dict(signal) for signal in signals]
                for coin, signals in self._signals.items()
            }

    def get_equity_timeframes(self) -> Dict[str, List[Dict[str, Any]]]:
        db_timeframes = database.fetch_equity_timeframes()
        if db_timeframes:
            return db_timeframes
        with self._lock:
            history_copy = [dict(entry) for entry in self._equity_history]
        return build_equity_timeframes(history_copy)

    def get_equity_series(self, range_name: str) -> List[Dict[str, Any]]:
        normalized = (range_name or "").lower()
        with self._lock:
            history_copy = [dict(entry) for entry in self._equity_history]
        return build_equity_series(history_copy, normalized)


_STATE = _DashboardState()


def update_account_snapshot(
    account_value: float,
    available_cash: float,
    return_pct: float,
    sharpe_ratio: float,
) -> None:
    _STATE.update_account_snapshot(account_value, available_cash, return_pct, sharpe_ratio)


def update_positions_snapshot(positions: Optional[List[Dict[str, Any]]]) -> None:
    _STATE.update_positions_snapshot(positions)


def record_strategy_batch(batch: Any) -> None:
    _STATE.record_strategy_batch(batch)


def record_strategy_signal(coin: str, signal: Dict[str, Any]) -> None:
    _STATE.record_strategy_signal(coin, signal)


def get_account_snapshot() -> Dict[str, Any]:
    return _STATE.get_account_snapshot()


def get_positions_snapshot() -> Dict[str, Any]:
    return _STATE.get_positions_snapshot()


def get_strategy_batches() -> List[Dict[str, Any]]:
    return _STATE.get_strategy_batches()


def get_strategy_signals() -> Dict[str, List[Dict[str, Any]]]:
    return _STATE.get_strategy_signals()


def get_equity_timeframes() -> Dict[str, List[Dict[str, Any]]]:
    return _STATE.get_equity_timeframes()


def get_runtime_summary() -> Dict[str, Any]:
    summary = database.fetch_runtime_summary()
    if summary:
        return summary
    status = "disabled" if not database.enabled else "unavailable"
    return {
        "run_id": None,
        "status": status,
        "started_at": None,
        "last_heartbeat": None,
        "uptime_seconds": 0,
        "total_iterations": 0,
        "symbols": [],
        "timeframe": None,
        "recent_iterations": [],
    }


def get_equity_series(range_name: str) -> List[Dict[str, Any]]:
    normalized = (range_name or "").lower()
    db_series = database.fetch_equity_timeframes(range_name=normalized)
    if isinstance(db_series, list):
        return db_series
    if isinstance(db_series, dict):
        result = db_series.get(normalized)
        if isinstance(result, list):
            return result
    return _STATE.get_equity_series(normalized)


__all__ = [
    "update_account_snapshot",
    "update_positions_snapshot",
    "record_strategy_batch",
    "record_strategy_signal",
    "get_account_snapshot",
    "get_positions_snapshot",
    "get_strategy_batches",
    "get_strategy_signals",
    "get_equity_timeframes",
    "get_equity_series",
    "get_runtime_summary",
]
