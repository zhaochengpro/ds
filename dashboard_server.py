"""FastAPI application serving the trading dashboard UI and APIs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from dashboard_state import (
    get_account_snapshot,
    get_equity_series,
    get_equity_timeframes,
    get_positions_snapshot,
    get_strategy_batches,
    get_strategy_signals,
    get_runtime_summary,
)


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
        return {
            "account": get_account_snapshot(),
            "positions": get_positions_snapshot(),
            "strategy": {
                "batches": get_strategy_batches(),
                "signals": get_strategy_signals(),
            },
            "equity": get_equity_timeframes(),
        }

    @app.get("/api/analytics/equity")
    def equity_endpoint(range: Optional[str] = Query(None)) -> Dict[str, Any]:
        if range:
            normalized = (range or "").lower()
            points = get_equity_series(normalized)
            return {"range": normalized, "points": points}
        return get_equity_timeframes()

    @app.get("/api/analytics/runtime")
    def runtime_endpoint() -> Dict[str, Any]:
        return get_runtime_summary()

    return app


dashboard_app = create_dashboard_app()


__all__ = ["dashboard_app", "create_dashboard_app"]
