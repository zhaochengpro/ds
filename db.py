import json
import logging
import os
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import pymysql
from pymysql.cursors import DictCursor

from analytics_utils import (
    EQUITY_RANGE_CONFIG,
    build_equity_series,
    build_equity_timeframes,
    parse_timestamp,
)


LOGGER = logging.getLogger("ai_trade_bot.database")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _to_db_timestamp(value: Optional[datetime]) -> datetime:
    if value is None:
        value = _utc_now()
    if value.tzinfo is not None:
        return value.astimezone(timezone.utc).replace(tzinfo=None)
    return value


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _timestamp_to_iso(value: Any) -> str:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc).isoformat()
        return value.astimezone(timezone.utc).isoformat()
    if isinstance(value, str):
        parsed = parse_timestamp(value)
        if parsed is not None:
            return parsed.astimezone(timezone.utc).isoformat()
        return value
    return _utc_now().isoformat()


class DatabaseClient:
    """Thin wrapper around PyMySQL for persisting trading telemetry."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._config = {
            "host": os.getenv("MYSQL_HOST"),
            "port": int(os.getenv("MYSQL_PORT", "3306") or "3306"),
            "user": os.getenv("MYSQL_USER"),
            "password": os.getenv("MYSQL_PASSWORD"),
            "database": os.getenv("MYSQL_DATABASE"),
        }
        self.enabled = all(
            self._config.get(key)
            for key in ("host", "user", "password", "database")
        )
        self._initialized = False
        self._run_starts: Dict[str, datetime] = {}
        self._current_run_id: Optional[str] = None
        if not self.enabled:
            LOGGER.info("MySQL persistence disabled – environment variables missing.")

    @contextmanager
    def _connection(self):
        if not self.enabled:
            raise RuntimeError("MySQL is not configured.")
        connection = None
        try:
            connection = pymysql.connect(
                host=self._config["host"],
                port=self._config["port"],
                user=self._config["user"],
                password=self._config["password"],
                database=self._config["database"],
                charset="utf8mb4",
                autocommit=True,
                cursorclass=DictCursor,
            )
            yield connection
        finally:
            if connection is not None:
                try:
                    connection.close()
                except Exception:  # pragma: no cover - best-effort close
                    pass

    def initialize(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            if self._initialized:
                return
            try:
                with self._connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        """
                        CREATE TABLE IF NOT EXISTS bot_runs (
                            run_id CHAR(36) PRIMARY KEY,
                            started_at DATETIME(6) NOT NULL,
                            last_heartbeat DATETIME(6) NOT NULL,
                            uptime_seconds BIGINT DEFAULT 0,
                            total_iterations INT DEFAULT 0,
                            symbols TEXT,
                            timeframe VARCHAR(32),
                            status VARCHAR(16) DEFAULT 'running'
                        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                        """
                    )
                    cursor.execute(
                        """
                        CREATE TABLE IF NOT EXISTS runtime_metrics (
                            id BIGINT AUTO_INCREMENT PRIMARY KEY,
                            run_id CHAR(36) NOT NULL,
                            iteration INT NOT NULL,
                            loop_started_at DATETIME(6) NOT NULL,
                            loop_finished_at DATETIME(6) NOT NULL,
                            duration_ms INT,
                            minutes_elapsed INT,
                            UNIQUE KEY run_iteration (run_id, iteration),
                            INDEX idx_runtime_run (run_id)
                        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                        """
                    )
                    cursor.execute(
                        """
                        CREATE TABLE IF NOT EXISTS equity_snapshots (
                            id BIGINT AUTO_INCREMENT PRIMARY KEY,
                            run_id CHAR(36) NOT NULL,
                            snapshot_ts DATETIME(6) NOT NULL,
                            account_value DOUBLE,
                            available_cash DOUBLE,
                            return_pct DOUBLE,
                            sharpe_ratio DOUBLE,
                            INDEX idx_equity_run (run_id),
                            INDEX idx_equity_ts (snapshot_ts)
                        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                        """
                    )
                    cursor.execute(
                        """
                        CREATE TABLE IF NOT EXISTS positions_history (
                            id BIGINT AUTO_INCREMENT PRIMARY KEY,
                            run_id CHAR(36) NOT NULL,
                            snapshot_ts DATETIME(6) NOT NULL,
                            coin VARCHAR(40),
                            symbol VARCHAR(80),
                            quantity DOUBLE,
                            entry_price DOUBLE,
                            current_price DOUBLE,
                            liquidation_price DOUBLE,
                            notional_usd DOUBLE,
                            leverage DOUBLE,
                            unrealized_pnl DOUBLE,
                            risk_usd DOUBLE,
                            confidence DOUBLE,
                            confidence_level VARCHAR(16),
                            take_profit DOUBLE,
                            stop_loss DOUBLE,
                            raw_payload JSON,
                            INDEX idx_position_run (run_id),
                            INDEX idx_position_ts (snapshot_ts),
                            INDEX idx_position_coin (coin)
                        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                        """
                    )
                    cursor.execute(
                        """
                        CREATE TABLE IF NOT EXISTS ai_signals (
                            id BIGINT AUTO_INCREMENT PRIMARY KEY,
                            run_id CHAR(36) NOT NULL,
                            signal_ts DATETIME(6) NOT NULL,
                            coin VARCHAR(40),
                            action VARCHAR(32),
                            confidence_label VARCHAR(16),
                            confidence_score DOUBLE,
                            leverage DOUBLE,
                            amount DOUBLE,
                            usdt_amount DOUBLE,
                            notional_usd DOUBLE,
                            price_snapshot DOUBLE,
                            risk_usd DOUBLE,
                            take_profit DOUBLE,
                            stop_loss DOUBLE,
                            justification TEXT,
                            invalidation TEXT,
                            is_fallback TINYINT(1) DEFAULT 0,
                            raw_payload JSON,
                            INDEX idx_signal_run (run_id),
                            INDEX idx_signal_ts (signal_ts),
                            INDEX idx_signal_coin (coin)
                        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                        """
                    )
                self._initialized = True
                LOGGER.info("MySQL persistence initialized successfully.")
            except Exception as exc:  # pragma: no cover - initialization errors
                LOGGER.exception("Failed to initialize MySQL persistence: %s", exc)
                self.enabled = False

    def register_run(self, *, symbols: Sequence[str], timeframe: Optional[str]) -> Optional[str]:
        if not self.enabled:
            return None
        run_id = str(uuid.uuid4())
        started_at = _utc_now()
        started_db = _to_db_timestamp(started_at)
        symbols_text = ",".join(sorted({(s or "").upper() for s in symbols or []}))
        try:
            with self._connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO bot_runs (
                        run_id, started_at, last_heartbeat, uptime_seconds,
                        total_iterations, symbols, timeframe, status
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, 'running')
                    """,
                    (
                        run_id,
                        started_db,
                        started_db,
                        0,
                        0,
                        symbols_text or None,
                        timeframe,
                    ),
                )
            with self._lock:
                self._run_starts[run_id] = started_at
                self._current_run_id = run_id
            LOGGER.info("Registered trading run %s for symbols [%s]", run_id, symbols_text)
            return run_id
        except Exception:
            LOGGER.exception("Failed to register trading run.")
            return None

    def record_runtime_iteration(
        self,
        run_id: Optional[str],
        iteration: int,
        loop_started_at: datetime,
        loop_finished_at: datetime,
        minutes_elapsed: int,
    ) -> None:
        if not self.enabled or not run_id:
            return
        duration_ms = max(
            0,
            int((loop_finished_at - loop_started_at).total_seconds() * 1000),
        )
        finished_db = _to_db_timestamp(loop_finished_at)
        started_db = _to_db_timestamp(loop_started_at)
        with self._lock:
            run_start = self._run_starts.get(run_id, loop_started_at)
        uptime_seconds = max(
            0,
            int((loop_finished_at - run_start).total_seconds()),
        )
        try:
            with self._connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO runtime_metrics (
                        run_id, iteration, loop_started_at, loop_finished_at,
                        duration_ms, minutes_elapsed
                    )
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                        loop_started_at = VALUES(loop_started_at),
                        loop_finished_at = VALUES(loop_finished_at),
                        duration_ms = VALUES(duration_ms),
                        minutes_elapsed = VALUES(minutes_elapsed)
                    """,
                    (
                        run_id,
                        iteration,
                        started_db,
                        finished_db,
                        duration_ms,
                        minutes_elapsed,
                    ),
                )
                cursor.execute(
                    """
                    UPDATE bot_runs
                    SET last_heartbeat = %s,
                        uptime_seconds = %s,
                        total_iterations = GREATEST(total_iterations, %s)
                    WHERE run_id = %s
                    """,
                    (finished_db, uptime_seconds, iteration + 1, run_id),
                )
        except Exception:
            LOGGER.exception("Failed to record runtime metrics for run %s", run_id)

    def record_equity_snapshot(
        self,
        run_id: Optional[str],
        snapshot_ts: datetime,
        account_value: Optional[float],
        available_cash: Optional[float],
        return_pct: Optional[float],
        sharpe_ratio: Optional[float],
    ) -> None:
        if not self.enabled or not run_id:
            return
        try:
            with self._connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO equity_snapshots (
                        run_id, snapshot_ts, account_value,
                        available_cash, return_pct, sharpe_ratio
                    )
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        run_id,
                        _to_db_timestamp(snapshot_ts),
                        _safe_float(account_value),
                        _safe_float(available_cash),
                        _safe_float(return_pct),
                        _safe_float(sharpe_ratio),
                    ),
                )
        except Exception:
            LOGGER.exception("Failed to persist equity snapshot for run %s", run_id)

    def record_positions_snapshot(
        self,
        run_id: Optional[str],
        snapshot_ts: datetime,
        positions: Optional[Iterable[Dict[str, Any]]],
    ) -> None:
        if not self.enabled or not run_id:
            return
        rows: List[tuple] = []
        for position in positions or []:
            if not isinstance(position, dict):
                continue
            coin_symbol = position.get("symbol") or ""
            coin = coin_symbol.split("/")[0] if "/" in coin_symbol else coin_symbol
            exit_plan = position.get("exit_plan") or {}
            confidence = _safe_float(position.get("confidence"))
            confidence_level = None
            if confidence is not None:
                if confidence >= 0.7:
                    confidence_level = "HIGH"
                elif confidence >= 0.4:
                    confidence_level = "MEDIUM"
                else:
                    confidence_level = "LOW"
            rows.append(
                (
                    run_id,
                    _to_db_timestamp(snapshot_ts),
                    coin.upper() or None,
                    coin_symbol or None,
                    _safe_float(position.get("quantity")),
                    _safe_float(position.get("entry_price")),
                    _safe_float(position.get("current_price")),
                    _safe_float(position.get("liquidation_price")),
                    _safe_float(position.get("notional_usd")),
                    _safe_float(position.get("leverage")),
                    _safe_float(position.get("unrealized_pnl")),
                    _safe_float(position.get("risk_usd")),
                    confidence,
                    confidence_level,
                    _safe_float(exit_plan.get("profit_target")),
                    _safe_float(exit_plan.get("stop_loss")),
                    json.dumps(position, ensure_ascii=False),
                )
            )
        if not rows:
            return
        try:
            with self._connection() as conn:
                cursor = conn.cursor()
                cursor.executemany(
                    """
                    INSERT INTO positions_history (
                        run_id, snapshot_ts, coin, symbol, quantity,
                        entry_price, current_price, liquidation_price,
                        notional_usd, leverage, unrealized_pnl,
                        risk_usd, confidence, confidence_level,
                        take_profit, stop_loss, raw_payload
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    rows,
                )
        except Exception:
            LOGGER.exception("Failed to persist positions snapshot for run %s", run_id)

    def record_ai_signals(
        self,
        run_id: Optional[str],
        signals: Optional[Iterable[Dict[str, Any]]],
    ) -> None:
        if not self.enabled or not run_id:
            return
        entries: List[tuple] = []
        for signal in signals or []:
            if not isinstance(signal, dict):
                continue
            timestamp = signal.get("timestamp")
            try:
                signal_ts = datetime.fromisoformat(timestamp)
            except Exception:
                signal_ts = _utc_now()
            entries.append(
                (
                    run_id,
                    _to_db_timestamp(signal_ts),
                    str(signal.get("coin") or "").upper() or None,
                    str(signal.get("signal") or "").upper() or None,
                    str(signal.get("confidence", "")).upper() or None,
                    _safe_float(signal.get("confidence_score")),
                    _safe_float(signal.get("leverage")),
                    _safe_float(signal.get("amount") or signal.get("quantity")),
                    _safe_float(signal.get("usdt_amount")),
                    _safe_float(signal.get("notional_usd")),
                    _safe_float(signal.get("price_snapshot")),
                    _safe_float(signal.get("risk_usd")),
                    _safe_float(signal.get("take_profit") or signal.get("profit_target")),
                    _safe_float(signal.get("stop_loss")),
                    signal.get("reason") or signal.get("justification"),
                    signal.get("invalidation_condition"),
                    1 if signal.get("is_fallback") else 0,
                    json.dumps(signal, ensure_ascii=False),
                )
            )
        if not entries:
            return
        try:
            with self._connection() as conn:
                cursor = conn.cursor()
                cursor.executemany(
                    """
                    INSERT INTO ai_signals (
                        run_id, signal_ts, coin, action, confidence_label,
                        confidence_score, leverage, amount, usdt_amount,
                        notional_usd, price_snapshot, risk_usd,
                        take_profit, stop_loss, justification,
                        invalidation, is_fallback, raw_payload
                    )
                    VALUES (
                        %s, %s, %s, %s, %s,
                        %s, %s, %s, %s,
                        %s, %s, %s,
                        %s, %s, %s,
                        %s, %s, %s
                    )
                    """,
                    entries,
                )
        except Exception:
            LOGGER.exception("Failed to persist AI signals for run %s", run_id)

    def fetch_equity_timeframes(
        self,
        run_id: Optional[str] = None,
        range_name: Optional[str] = None,
    ) -> Optional[Union[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]]:
        if not self.enabled:
            return None
        with self._lock:
            target_run = run_id or self._current_run_id
        range_key = (range_name or "").lower() or None
        if range_key and range_key not in EQUITY_RANGE_CONFIG:
            range_key = None
        window = max(
            (cfg["window"] for cfg in EQUITY_RANGE_CONFIG.values()),
            default=timedelta(days=365),
        )
        if range_key:
            window = EQUITY_RANGE_CONFIG[range_key]["window"]
        now = _utc_now()
        cutoff = now - window
        try:
            with self._connection() as conn:
                cursor = conn.cursor()
                if not target_run:
                    cursor.execute(
                        "SELECT run_id FROM bot_runs ORDER BY started_at DESC LIMIT 1"
                    )
                    latest = cursor.fetchone()
                    if not latest:
                        return None
                    target_run = latest.get("run_id")
                if not target_run:
                    return None
                cursor.execute(
                    """
                    SELECT snapshot_ts, account_value
                    FROM equity_snapshots
                    WHERE snapshot_ts >= %s AND snapshot_ts <= %s
                    ORDER BY snapshot_ts ASC
                    """,
                    (
                        _to_db_timestamp(cutoff),
                        _to_db_timestamp(now),
                    ),
                )
                rows = cursor.fetchall()
        except Exception:
            LOGGER.exception("Failed to fetch equity snapshots for run %s", target_run)
            return None
        if not rows:
            return None
        history: List[Dict[str, Any]] = []
        for row in rows:
            ts_value = row.get("snapshot_ts")
            if ts_value is None:
                continue
            account_value = _safe_float(row.get("account_value"))
            history.append(
                {
                    "timestamp": _timestamp_to_iso(ts_value),
                    "value": account_value or 0.0,
                }
            )
        if not history:
            return None
        if range_key:
            return build_equity_series(history, range_key)
        return build_equity_timeframes(history)

    def fetch_runtime_summary(
        self,
        run_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None
        with self._lock:
            target_run = run_id or self._current_run_id
        try:
            with self._connection() as conn:
                cursor = conn.cursor()
                if not target_run:
                    cursor.execute(
                        "SELECT run_id FROM bot_runs ORDER BY started_at DESC LIMIT 1"
                    )
                    latest = cursor.fetchone()
                    if not latest:
                        return None
                    target_run = latest.get("run_id")
                if not target_run:
                    return None
                cursor.execute(
                    """
                    SELECT run_id, started_at, last_heartbeat, uptime_seconds,
                           total_iterations, symbols, timeframe, status
                    FROM bot_runs
                    WHERE run_id = %s
                    """,
                    (target_run,),
                )
                run_row = cursor.fetchone()
                if not run_row:
                    return None
                cursor.execute(
                    """
                    SELECT iteration, loop_started_at, loop_finished_at,
                           duration_ms, minutes_elapsed
                    FROM runtime_metrics
                    WHERE run_id = %s
                    ORDER BY iteration DESC
                    LIMIT 10
                    """,
                    (target_run,),
                )
                iteration_rows = cursor.fetchall()
        except Exception:
            LOGGER.exception("Failed to fetch runtime summary for run %s", target_run)
            return None

        uptime_seconds = run_row.get("uptime_seconds")
        if uptime_seconds is None:
            started_at = run_row.get("started_at")
            last_heartbeat = run_row.get("last_heartbeat")
            if isinstance(started_at, datetime) and isinstance(last_heartbeat, datetime):
                uptime_seconds = int((last_heartbeat - started_at).total_seconds())
        iteration_items: List[Dict[str, Any]] = []
        for row in iteration_rows or []:
            iteration_items.append(
                {
                    "iteration": row.get("iteration"),
                    "started_at": _timestamp_to_iso(row.get("loop_started_at")),
                    "finished_at": _timestamp_to_iso(row.get("loop_finished_at")),
                    "duration_ms": row.get("duration_ms"),
                    "minutes_elapsed": row.get("minutes_elapsed"),
                }
            )
        symbols_value = run_row.get("symbols")
        if isinstance(symbols_value, str):
            symbols_list = [item for item in symbols_value.split(",") if item]
        elif isinstance(symbols_value, (list, tuple)):
            symbols_list = list(symbols_value)
        else:
            symbols_list = []
        return {
            "run_id": run_row.get("run_id"),
            "status": run_row.get("status") or "unknown",
            "started_at": _timestamp_to_iso(run_row.get("started_at")),
            "last_heartbeat": _timestamp_to_iso(run_row.get("last_heartbeat")),
            "uptime_seconds": uptime_seconds if uptime_seconds is not None else 0,
            "total_iterations": run_row.get("total_iterations") or 0,
            "symbols": symbols_list,
            "timeframe": run_row.get("timeframe"),
            "recent_iterations": iteration_items,
        }

    def mark_run_completed(self, run_id: Optional[str]) -> None:
        if not self.enabled or not run_id:
            return
        finished_at = _utc_now()
        finished_db = _to_db_timestamp(finished_at)
        with self._lock:
            run_start = self._run_starts.get(run_id)
        uptime_seconds = None
        if run_start is not None:
            uptime_seconds = int((finished_at - run_start).total_seconds())
        try:
            with self._connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    UPDATE bot_runs
                    SET status = 'completed',
                        last_heartbeat = %s,
                        uptime_seconds = COALESCE(%s, uptime_seconds)
                    WHERE run_id = %s
                    """,
                    (finished_db, uptime_seconds, run_id),
                )
        except Exception:
            LOGGER.exception("Failed to mark run %s as completed", run_id)


database = DatabaseClient()


__all__ = ["database"]
