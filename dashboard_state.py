"""Thread-safe dashboard state store for trading dashboard."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from threading import RLock
from typing import Any, Dict, List, Optional


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_timestamp(value: str) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        if value.endswith("Z"):
            try:
                return datetime.fromisoformat(value[:-1] + "+00:00")
            except ValueError:
                return None
        return None
    except TypeError:
        return None


def _sample_history(
    entries: List[Dict[str, Any]],
    *,
    cutoff: datetime,
    min_interval: timedelta,
) -> List[Dict[str, Any]]:
    if not entries:
        return []

    filtered = [item for item in entries if item["datetime"] >= cutoff]
    if not filtered:
        filtered = entries[-1:]

    result: List[Dict[str, Any]] = []
    last_dt: Optional[datetime] = None

    for item in filtered:
        dt = item["datetime"]
        if not result:
            result.append(item)
            last_dt = dt
            continue

        assert last_dt is not None
        if dt - last_dt >= min_interval:
            result.append(item)
            last_dt = dt
        else:
            result[-1] = item

    if result and result[-1] is not filtered[-1]:
        result.append(filtered[-1])

    return [
        {"timestamp": item["timestamp"], "value": item["value"]}
        for item in result
    ]


def _build_equity_timeframes(history: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    prepared: List[Dict[str, Any]] = []
    for entry in history:
        timestamp = entry.get("timestamp")
        value = entry.get("value")
        dt_obj = _parse_timestamp(timestamp)
        if dt_obj is None:
            continue
        prepared.append(
            {
                "timestamp": timestamp,
                "value": float(value or 0.0),
                "datetime": dt_obj,
            }
        )

    if not prepared:
        return {"day": [], "week": [], "month": [], "year": []}

    prepared.sort(key=lambda item: item["datetime"])
    now = datetime.now(timezone.utc)

    return {
        "day": _sample_history(
            prepared,
            cutoff=now - timedelta(days=1),
            min_interval=timedelta(minutes=15),
        ),
        "week": _sample_history(
            prepared,
            cutoff=now - timedelta(days=7),
            min_interval=timedelta(hours=1),
        ),
        "month": _sample_history(
            prepared,
            cutoff=now - timedelta(days=30),
            min_interval=timedelta(days=1),
        ),
        "year": _sample_history(
            prepared,
            cutoff=now - timedelta(days=365),
            min_interval=timedelta(days=7),
        ),
    }


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
            "timestamp": _utc_now(),
        }
        self._positions: Dict[str, Any] = {
            "items": [],
            "timestamp": _utc_now(),
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
                "timestamp": _utc_now(),
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
                "timestamp": _utc_now(),
            }

    def record_strategy_batch(self, batch: Any) -> None:
        timestamp = _utc_now()
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
        signal_copy.setdefault("timestamp", _utc_now())
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
        with self._lock:
            history_copy = [dict(entry) for entry in self._equity_history]
        return _build_equity_timeframes(history_copy)


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
]
