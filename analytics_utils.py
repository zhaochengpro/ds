from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_timestamp(value: str) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        if isinstance(value, str) and value.endswith("Z"):
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
    end: datetime,
    min_interval: timedelta,
) -> List[Dict[str, Any]]:
    if not entries:
        return []

    filtered = [
        item for item in entries if cutoff <= item["datetime"] < end
    ]
    if not filtered:
        fallback = [item for item in entries if item["datetime"] < end]
        if fallback:
            filtered = fallback[-1:]
        else:
            return []

    result: List[Dict[str, Any]] = []
    last_dt: Optional[datetime] = None

    for item in filtered:
        dt_val = item["datetime"]
        if not result:
            result.append(item)
            last_dt = dt_val
            continue

        assert last_dt is not None
        if dt_val - last_dt >= min_interval:
            result.append(item)
            last_dt = dt_val
        else:
            result[-1] = item

    if result and result[-1] is not filtered[-1]:
        result.append(filtered[-1])

    return [
        {"timestamp": entry["timestamp"], "value": entry["value"]}
        for entry in result
    ]


EQUITY_RANGE_CONFIG: Dict[str, Dict[str, Any]] = {
    "day": {"window": timedelta(days=1), "interval": timedelta(seconds=1)},
    "week": {"window": timedelta(days=7), "interval": timedelta(hours=1)},
    "month": {"window": timedelta(days=30), "interval": timedelta(days=1)},
    "year": {"window": timedelta(days=365), "interval": timedelta(days=7)},
    "all": {"window": None, "interval": timedelta(days=7)},
}


def _prepare_equity_history(history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    prepared: List[Dict[str, Any]] = []
    for entry in history:
        timestamp = entry.get("timestamp")
        value = entry.get("value")
        dt_obj = parse_timestamp(timestamp)
        if dt_obj is None:
            continue
        prepared.append(
            {
                "timestamp": timestamp,
                "value": float(value or 0.0),
                "datetime": dt_obj,
            }
        )
    prepared.sort(key=lambda item: item["datetime"])
    return prepared


HISTORY_END_OFFSET = timedelta(microseconds=1)


def build_equity_series(history: List[Dict[str, Any]], range_name: str) -> List[Dict[str, Any]]:
    range_key = (range_name or "").lower()
    config = EQUITY_RANGE_CONFIG.get(range_key)
    if not config:
        return []

    prepared = _prepare_equity_history(history)
    if not prepared:
        return []

    now = datetime.now(timezone.utc)
    end = now - HISTORY_END_OFFSET
    window = config.get("window")
    interval = config["interval"]
    if window is None:
        cutoff = prepared[0]["datetime"]
    else:
        cutoff = end - window
    if cutoff > end:
        cutoff = end
    return _sample_history(prepared, cutoff=cutoff, end=end, min_interval=interval)


def build_equity_timeframes(history: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    prepared = _prepare_equity_history(history)
    if not prepared:
        return {key: [] for key in EQUITY_RANGE_CONFIG}

    now = datetime.now(timezone.utc)
    end = now - HISTORY_END_OFFSET
    result: Dict[str, List[Dict[str, Any]]] = {}
    for key, config in EQUITY_RANGE_CONFIG.items():
        window = config.get("window")
        if window is None:
            cutoff = prepared[0]["datetime"]
        else:
            cutoff = end - window
        if cutoff > end:
            cutoff = end
        result[key] = _sample_history(
            prepared,
            cutoff=cutoff,
            end=end,
            min_interval=config["interval"],
        )
    return result


__all__ = [
    "utc_now_iso",
    "parse_timestamp",
    "build_equity_series",
    "build_equity_timeframes",
    "EQUITY_RANGE_CONFIG",
]
