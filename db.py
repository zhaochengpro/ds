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


def _as_utc_datetime(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    if isinstance(value, str):
        parsed = parse_timestamp(value)
        if parsed is not None:
            return parsed.astimezone(timezone.utc)
    return None


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
        print(self._config)
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
                print(exc)
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
                    (finished_db, uptime_seconds, iteration, run_id),
                )
        except Exception:
            LOGGER.exception("Failed to record runtime metrics for run %s", run_id)

    def _collect_iteration_stats(self, cursor) -> Dict[str, Any]:
        cursor.execute(
            """
            SELECT
                MIN(iteration) AS min_iteration,
                MAX(iteration) AS max_iteration,
                COUNT(*) AS iteration_count,
                MIN(loop_started_at) AS first_loop_ts,
                MAX(loop_finished_at) AS last_loop_ts
            FROM runtime_metrics
            """
        )
        row = cursor.fetchone() or {}
        return {
            "min_iteration": row.get("min_iteration"),
            "max_iteration": row.get("max_iteration"),
            "iteration_count": row.get("iteration_count"),
            "first_loop_ts": row.get("first_loop_ts"),
            "last_loop_ts": row.get("last_loop_ts"),
        }

    def _fetch_first_snapshot_ts(self, cursor) -> Optional[datetime]:
        cursor.execute(
            """
            SELECT snapshot_ts
            FROM equity_snapshots
            ORDER BY snapshot_ts ASC
            LIMIT 1
            """
        )
        row = cursor.fetchone()
        if not row:
            return None
        return row.get("snapshot_ts")

    def _fetch_last_snapshot_ts(self, cursor) -> Optional[datetime]:
        cursor.execute(
            """
            SELECT snapshot_ts
            FROM equity_snapshots
            ORDER BY snapshot_ts DESC
            LIMIT 1
            """
        )
        row = cursor.fetchone()
        if not row:
            return None
        return row.get("snapshot_ts")

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
        range_key = (range_name or "").lower() or None
        if range_key and range_key not in EQUITY_RANGE_CONFIG:
            range_key = None
        finite_windows = [
            cfg["window"]
            for cfg in EQUITY_RANGE_CONFIG.values()
            if cfg.get("window") is not None
        ]
        max_window = max(finite_windows) if finite_windows else timedelta(days=365)
        include_unbounded = any(
            cfg.get("window") is None for cfg in EQUITY_RANGE_CONFIG.values()
        )
        if range_key is not None:
            window = EQUITY_RANGE_CONFIG[range_key]["window"]
        else:
            window = None if include_unbounded else max_window
        now = _utc_now()
        cutoff = None if window is None else now - window
        try:
            with self._connection() as conn:
                cursor = conn.cursor()
                conditions = []
                params: List[Any] = []
                if cutoff is not None:
                    conditions.append("snapshot_ts >= %s")
                    params.append(_to_db_timestamp(cutoff))
                conditions.append("snapshot_ts <= %s")
                params.append(_to_db_timestamp(now))
                where_clause = " AND ".join(conditions)
                cursor.execute(
                    f"""
                    SELECT snapshot_ts, account_value
                    FROM equity_snapshots
                    WHERE {where_clause}
                    ORDER BY snapshot_ts ASC
                    """,
                    tuple(params),
                )
                rows = cursor.fetchall()
        except Exception:
            LOGGER.exception("Failed to fetch equity snapshots")
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

    def fetch_initial_equity_value(
        self,
        run_id: Optional[str] = None,
    ) -> Optional[float]:
        if not self.enabled:
            return None
        try:
            with self._connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT account_value
                    FROM equity_snapshots
                    ORDER BY snapshot_ts ASC
                    LIMIT 1
                    """
                )
                row = cursor.fetchone()
                if row and row.get("account_value") is not None:
                    value = _safe_float(row.get("account_value"))
                    if value is not None:
                        return value
        except Exception:
            LOGGER.exception("Failed to fetch initial equity value")
        return None

    def fetch_runtime_summary(
        self,
        run_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None
        try:
            with self._connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT
                        MIN(started_at) AS min_started,
                        MAX(last_heartbeat) AS max_heartbeat,
                        SUM(uptime_seconds) AS sum_uptime,
                        SUM(total_iterations) AS sum_iterations
                    FROM bot_runs
                    """
                )
                aggregate_row = cursor.fetchone() or {}
                if not any(aggregate_row.values()):
                    return None

                iteration_stats = self._collect_iteration_stats(cursor)
                first_equity_raw = self._fetch_first_snapshot_ts(cursor)
                last_equity_raw = self._fetch_last_snapshot_ts(cursor)

                cursor.execute(
                    """
                    SELECT iteration, loop_started_at, loop_finished_at,
                           duration_ms, minutes_elapsed
                    FROM runtime_metrics
                    ORDER BY loop_finished_at DESC
                    LIMIT 10
                    """
                )
                iteration_rows = cursor.fetchall()

                cursor.execute(
                    """
                    SELECT symbols
                    FROM bot_runs
                    WHERE symbols IS NOT NULL
                    """
                )
                symbol_rows = cursor.fetchall()
        except Exception:
            LOGGER.exception("Failed to fetch runtime summary (aggregate)")
            return None

        iteration_count = iteration_stats.get("iteration_count") if iteration_stats else None
        if (iteration_count is None or iteration_count == 0) and aggregate_row.get("sum_iterations") is not None:
            iteration_count = aggregate_row.get("sum_iterations")
        loops_total = int(iteration_count or 0)

        started_at_value = aggregate_row.get("min_started")
        last_heartbeat_value = aggregate_row.get("max_heartbeat")

        baseline_candidates: List[datetime] = []
        last_candidates: List[datetime] = []

        if first_equity_raw is not None:
            equity_dt = _as_utc_datetime(first_equity_raw)
            if equity_dt is not None:
                baseline_candidates.append(equity_dt)

        if iteration_stats:
            first_loop_dt = _as_utc_datetime(iteration_stats.get("first_loop_ts"))
            last_loop_dt = _as_utc_datetime(iteration_stats.get("last_loop_ts"))
            if first_loop_dt is not None:
                baseline_candidates.append(first_loop_dt)
            if last_loop_dt is not None:
                last_candidates.append(last_loop_dt)

        if last_equity_raw is not None:
            equity_last_dt = _as_utc_datetime(last_equity_raw)
            if equity_last_dt is not None:
                last_candidates.append(equity_last_dt)

        baseline_candidates.append(_as_utc_datetime(started_at_value))
        last_candidates.append(_as_utc_datetime(last_heartbeat_value))

        baseline_candidates = [dt for dt in baseline_candidates if dt is not None]
        last_candidates = [dt for dt in last_candidates if dt is not None]

        baseline_dt = min(baseline_candidates) if baseline_candidates else None
        last_dt = max(last_candidates) if last_candidates else None

        uptime_seconds = aggregate_row.get("sum_uptime")
        if baseline_dt is not None and last_dt is not None:
            uptime_seconds = max(int((last_dt - baseline_dt).total_seconds()), 0)
        elif uptime_seconds is None:
            uptime_seconds = 0

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

        symbols_set = set()
        for row in symbol_rows or []:
            value = row.get("symbols")
            if not value:
                continue
            if isinstance(value, str):
                for token in value.split(","):
                    token = token.strip()
                    if token:
                        symbols_set.add(token)
            elif isinstance(value, (list, tuple)):
                for token in value:
                    if not token:
                        continue
                    token_str = str(token).strip()
                    if token_str:
                        symbols_set.add(token_str)

        symbols_list = sorted(symbols_set)

        started_at_iso = _timestamp_to_iso(baseline_dt) if baseline_dt is not None else _timestamp_to_iso(started_at_value)
        last_heartbeat_iso = _timestamp_to_iso(last_heartbeat_value)
        return {
            "run_id": None,
            "status": "aggregate",
            "started_at": started_at_iso,
            "last_heartbeat": last_heartbeat_iso,
            "uptime_seconds": uptime_seconds if uptime_seconds is not None else 0,
            "total_iterations": loops_total,
            "symbols": symbols_list,
            "timeframe": None,
            "recent_iterations": iteration_items,
        }

    def fetch_first_snapshot_timestamp(
        self,
        run_id: Optional[str] = None,
    ) -> Optional[datetime]:
        if not self.enabled:
            return None
        try:
            with self._connection() as conn:
                cursor = conn.cursor()
                raw_ts = self._fetch_first_snapshot_ts(cursor)
        except Exception:
            LOGGER.exception("Failed to fetch first snapshot timestamp")
            return None
        if raw_ts is None:
            return None
        return _as_utc_datetime(raw_ts)

    def fetch_iteration_stats(
        self,
        run_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None
        try:
            with self._connection() as conn:
                cursor = conn.cursor()
                stats = self._collect_iteration_stats(cursor)
        except Exception:
            LOGGER.exception("Failed to fetch iteration stats")
            return None

        if not stats:
            return {
                "min_iteration": None,
                "max_iteration": None,
                "iteration_count": 0,
                "first_loop_ts": None,
                "last_loop_ts": None,
            }

        return {
            "min_iteration": stats.get("min_iteration"),
            "max_iteration": stats.get("max_iteration"),
            "iteration_count": int(stats.get("iteration_count") or 0),
            "first_loop_ts": _as_utc_datetime(stats.get("first_loop_ts")),
            "last_loop_ts": _as_utc_datetime(stats.get("last_loop_ts")),
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
