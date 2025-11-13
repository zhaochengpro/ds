"""FastAPI application serving the trading dashboard UI and APIs."""

from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from analytics_utils import EQUITY_RANGE_CONFIG, utc_now_iso
from dashboard_state import (
    get_account_snapshot,
    get_equity_series,
    get_equity_timeframes,
    get_positions_snapshot,
    get_strategy_batches,
    get_strategy_signals,
    get_runtime_summary,
)


LOGGER = logging.getLogger("ai_trade_bot.dashboard.server")


def _normalize_equity_range(range_name: Optional[str]) -> str:
    candidate = (range_name or "").lower()
    if candidate in EQUITY_RANGE_CONFIG:
        return candidate
    return "day"


def _build_full_state_snapshot() -> Dict[str, Any]:
    return {
        "account": get_account_snapshot(),
        "positions": get_positions_snapshot(),
        "strategy": {
            "batches": get_strategy_batches(),
            "signals": get_strategy_signals(),
        },
        "equity": get_equity_timeframes(),
    }


def _build_equity_payload(
    range_name: Optional[str],
    *,
    include_timeframes: bool = False,
) -> Dict[str, Any]:
    normalized = _normalize_equity_range(range_name)
    timeframes = get_equity_timeframes()
    points = list(timeframes.get(normalized, []))
    payload: Dict[str, Any] = {
        "range": normalized,
        "points": points,
    }
    if include_timeframes:
        payload["timeframes"] = timeframes
    if points:
        payload["latest_timestamp"] = points[-1].get("timestamp")
    return payload


class DashboardStreamSession:
    STATE_INTERVAL = 5.0
    ACCOUNT_INTERVAL = 2.0
    RUNTIME_INTERVAL = 5.0
    EQUITY_INTERVAL = 30.0
    HEARTBEAT_INTERVAL = 15.0

    def __init__(self, websocket: WebSocket) -> None:
        self.websocket = websocket
        self.active = True
        self.equity_range = "day"
        self._include_timeframes_next = True
        self._send_lock = asyncio.Lock()

    async def run(self) -> None:
        await self.websocket.accept()
        await self._send_snapshot(include_equity=True)

        sender = asyncio.create_task(self._sender_loop())
        receiver = asyncio.create_task(self._receiver_loop())

        try:
            await asyncio.wait({sender, receiver}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            self.active = False
            sender.cancel()
            receiver.cancel()
            with suppress(Exception):
                await sender
            with suppress(Exception):
                await receiver
            with suppress(Exception):
                await self.websocket.close()

    async def _send_json(self, message: Dict[str, Any]) -> None:
        if not self.active:
            return
        try:
            async with self._send_lock:
                await self.websocket.send_json(message)
        except WebSocketDisconnect:
            self.active = False
            raise
        except Exception:
            self.active = False
            raise

    async def _send_snapshot(self, *, include_equity: bool) -> None:
        await self._send_json({"type": "state", "payload": _build_full_state_snapshot()})
        await self._send_json({"type": "account", "payload": get_account_snapshot()})
        await self._send_json({"type": "runtime", "payload": get_runtime_summary()})
        if include_equity:
            await self._send_equity_snapshot(include_timeframes=True)

    async def _send_equity_snapshot(self, *, include_timeframes: bool = False) -> None:
        include_all = include_timeframes or self._include_timeframes_next
        payload = _build_equity_payload(self.equity_range, include_timeframes=include_all)
        await self._send_json({"type": "equity", **payload})
        if include_all:
            self._include_timeframes_next = False

    async def _sender_loop(self) -> None:
        loop = asyncio.get_running_loop()
        next_state = loop.time() + self.STATE_INTERVAL
        next_account = loop.time() + self.ACCOUNT_INTERVAL
        next_runtime = loop.time() + self.RUNTIME_INTERVAL
        next_equity = loop.time() + self.EQUITY_INTERVAL
        next_heartbeat = loop.time() + self.HEARTBEAT_INTERVAL

        while self.active:
            now = loop.time()
            try:
                if now >= next_state:
                    await self._send_json({"type": "state", "payload": _build_full_state_snapshot()})
                    next_state = now + self.STATE_INTERVAL

                if not self.active:
                    break

                if now >= next_account:
                    await self._send_json({"type": "account", "payload": get_account_snapshot()})
                    next_account = now + self.ACCOUNT_INTERVAL

                if not self.active:
                    break

                if now >= next_runtime:
                    await self._send_json({"type": "runtime", "payload": get_runtime_summary()})
                    next_runtime = now + self.RUNTIME_INTERVAL

                if not self.active:
                    break

                if now >= next_equity:
                    await self._send_equity_snapshot()
                    next_equity = now + self.EQUITY_INTERVAL

                if not self.active:
                    break

                if now >= next_heartbeat:
                    await self._send_json({"type": "heartbeat", "timestamp": utc_now_iso()})
                    next_heartbeat = now + self.HEARTBEAT_INTERVAL

            except asyncio.CancelledError:
                self.active = False
                break
            except WebSocketDisconnect:
                self.active = False
                break
            except Exception:
                LOGGER.exception("Dashboard websocket sender loop error")
                self.active = False
                break

            next_event = min(next_state, next_account, next_runtime, next_equity, next_heartbeat)
            sleep_for = max(next_event - loop.time(), 0.1)
            await asyncio.sleep(sleep_for)

    async def _receiver_loop(self) -> None:
        while self.active:
            try:
                data = await self.websocket.receive_json()
            except asyncio.CancelledError:
                self.active = False
                break
            except WebSocketDisconnect:
                self.active = False
                break
            except Exception:
                self.active = False
                break

            if not isinstance(data, dict):
                continue

            message_type = str(data.get("type") or "").lower()

            if message_type in {"set_equity_range", "request_equity"}:
                requested = _normalize_equity_range(data.get("range"))
                force_full = bool(data.get("include_timeframes")) or message_type == "set_equity_range"
                if requested != self.equity_range:
                    self.equity_range = requested
                    self._include_timeframes_next = True
                    force_full = True
                await self._send_equity_snapshot(include_timeframes=force_full)
            elif message_type == "request_snapshot":
                await self._send_snapshot(include_equity=True)
            elif message_type == "ping":
                await self._send_json({"type": "pong", "timestamp": utc_now_iso()})



def _resolve_static_dir() -> Path:
    repo_root = Path(__file__).parent
    static_dir = repo_root / "dashboard_ui"
    static_dir.mkdir(parents=True, exist_ok=True)
    (static_dir / "static").mkdir(parents=True, exist_ok=True)
    return static_dir


def create_dashboard_app() -> FastAPI:
    app = FastAPI(title="AI Trader Dashboard", docs_url=None, redoc_url=None)

    static_dir = _resolve_static_dir()
    static_assets = static_dir / "static"
    if static_assets.exists():
        app.mount("/static", StaticFiles(directory=str(static_assets)), name="static")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        index_file = static_dir / "index.html"
        if index_file.exists():
            return index_file.read_text(encoding="utf-8")
        return "<html><body><h1>Dashboard assets missing</h1></body></html>"

    @app.get("/api/account")
    def account_endpoint() -> Dict[str, Any]:
        return get_account_snapshot()

    @app.get("/api/positions")
    def positions_endpoint() -> Dict[str, Any]:
        return get_positions_snapshot()

    @app.get("/api/strategies/batches")
    def strategy_batches_endpoint() -> Dict[str, Any]:
        return {"items": get_strategy_batches()}

    @app.get("/api/strategies/signals")
    def strategy_signals_endpoint() -> Dict[str, Any]:
        return {"items": get_strategy_signals()}

    @app.get("/api/state")
    def full_state_endpoint() -> Dict[str, Any]:
        return _build_full_state_snapshot()

    @app.get("/api/analytics/equity")
    def equity_endpoint(range: Optional[str] = Query(None)) -> Dict[str, Any]:
        if range:
            normalized = _normalize_equity_range(range)
            points = get_equity_series(normalized)
            return {"range": normalized, "points": points}
        return get_equity_timeframes()

    @app.get("/api/analytics/runtime")
    def runtime_endpoint() -> Dict[str, Any]:
        return get_runtime_summary()

    @app.websocket("/ws/dashboard")
    async def dashboard_stream(websocket: WebSocket) -> None:
        session = DashboardStreamSession(websocket)
        try:
            await session.run()
        except WebSocketDisconnect:
            pass
        except Exception:
            LOGGER.exception("Dashboard websocket session terminated unexpectedly")

    return app


dashboard_app = create_dashboard_app()


__all__ = ["dashboard_app", "create_dashboard_app"]
