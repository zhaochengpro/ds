from __future__ import annotations

import time
import logging
from typing import Dict, Iterable, List, Optional, Tuple, Any

import numpy as np
import ccxt

from db import database


performance_tracker: Dict[str, Any] = {
    "initial_equity": None,
    "initial_source": None,
    "equity_history": [],
    "last_seed_attempt": 0.0,
}


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def update_performance_metrics(account_value: Optional[float]) -> Tuple[float, float]:
    state = performance_tracker
    if account_value is None:
        return 0.0, 0.0

    now = time.time()
    needs_db_seed = (
        state.get("initial_equity") is None
        or state.get("initial_source") != "db"
    )
    last_attempt = state.get("last_seed_attempt", 0.0)
    if needs_db_seed and (now - last_attempt >= 5.0):
        state["last_seed_attempt"] = now
        try:
            db_initial = database.fetch_initial_equity_value()
        except Exception:
            db_initial = None
        if db_initial is not None:
            state["initial_equity"] = float(db_initial)
            state["initial_source"] = "db"

    if state["initial_equity"] is None and account_value > 0:
        state["initial_equity"] = account_value
        state["initial_source"] = state.get("initial_source") or "fallback"

    state["equity_history"].append({"timestamp": time.time(), "value": account_value})
    if len(state["equity_history"]) > 500:
        state["equity_history"] = state["equity_history"][-500:]

    initial_equity = state["initial_equity"] or 0.0
    return_pct = (
        ((account_value - initial_equity) / initial_equity) * 100.0
        if initial_equity
        else 0.0
    )

    returns: List[float] = []
    history = state["equity_history"]
    for idx in range(1, len(history)):
        prev_value = history[idx - 1]["value"]
        curr_value = history[idx]["value"]
        if prev_value:
            returns.append((curr_value - prev_value) / prev_value)

    sharpe_ratio = 0.0
    if len(returns) > 1:
        mean_return = float(np.mean(returns))
        std_return = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0
        if std_return > 0:
            sharpe_ratio = mean_return / std_return * (len(returns) ** 0.5)

    return return_pct, sharpe_ratio


def compute_account_metrics(balance: Dict[str, Any]) -> Tuple[float, float, float, float]:
    info = balance.get("info", {}) if isinstance(balance, dict) else {}
    account_value = safe_float(info.get("totalEq"), 0.0)
    if account_value == 0.0:
        totals = balance.get("total") if isinstance(balance, dict) else None
        if isinstance(totals, dict):
            account_value = safe_float(totals.get("USDT"), account_value)
    if account_value == 0.0:
        usdt_section = balance.get("USDT") if isinstance(balance, dict) else None
        if isinstance(usdt_section, dict):
            account_value = safe_float(usdt_section.get("total"), account_value)

    available_cash = safe_float(info.get("cashBal"), 0.0)
    if available_cash == 0.0:
        usdt_section = balance.get("USDT") if isinstance(balance, dict) else None
        if isinstance(usdt_section, dict):
            available_cash = safe_float(usdt_section.get("free"), 0.0)

    return_pct, sharpe_ratio = update_performance_metrics(account_value)
    return account_value, available_cash, return_pct, sharpe_ratio


def format_position(position_obj: Dict[str, Dict[str, Any]]) -> str:
    if not position_obj:
        return "    当前无持仓。\n"

    lines: List[str] = []
    for coin, position in position_obj.items():
        entry_price = safe_float(position.get("entry_price"))
        current_price = safe_float(position.get("current_price") or entry_price)
        pnl = safe_float(position.get("unrealized_pnl"))
        leverage = safe_float(position.get("leverage"))
        notional = safe_float(position.get("notional_usd"))
        
        # print('format postion:', position)

        lines.append(f"    **{coin}持仓情况：**")
        lines.append(f"    方向：{'多单' if position.get('side') == 'long' else '空单'}")
        lines.append(f"    合约张数：{position.get('size')}")
        lines.append(f"    合约名义：${notional:,.2f}")
        lines.append(f"    入场价：{entry_price:,.4f}")
        lines.append(f"    当前价：{current_price:,.4f}")
        lines.append(f"    未实现盈亏：${pnl:,.2f}")
        lines.append(f"    杠杆：{leverage}x")
        
        if position.get("tp") :
            tp = safe_float(position.get("tp"))
            lines.append(f"    止盈价：{tp:,.4f}")
        else:
            lines.append(f"    止盈价：未设置")
            
        if position.get("sl") :
            sl = safe_float(position.get("sl"))
            lines.append(f"    止损价：{sl:,.4f}")
        else:
            lines.append(f"    止损价：未设置")
        lines.append("")

    return "\n".join(lines)


def get_current_positions(
    exchange: ccxt.Exchange,
    logger: logging.Logger,
    coins: Iterable[str],
    retries: int = 50,
) -> Dict[str, Dict[str, Any]]:
    target_symbols = {coin: f"{coin}/USDT:USDT" for coin in coins}
    position_obj: Dict[str, Dict[str, Any]] = {}

    for attempt in range(retries):
        try:
            positions = exchange.fetch_positions(symbols=None, params={"instType": "SWAP"})
        except ccxt.RateLimitExceeded:
            wait_time = min(2**attempt, 5)
            logger.warning(f"获取持仓命中限频，等待{wait_time}秒后重试...")
            time.sleep(wait_time)
            continue
        except Exception as exc:
            logger.exception("获取持仓失败")
            if attempt == retries - 1:
                return position_obj
            time.sleep(0.5)
            continue

        positions_by_symbol: Dict[str, Dict[str, Any]] = {}
        for pos in positions:
            symbol = pos.get("symbol")
            if symbol in target_symbols.values():
                positions_by_symbol[symbol] = pos

        for coin, symbol in target_symbols.items():
            pos = positions_by_symbol.get(symbol)
            if not pos:
                continue

            contracts = safe_float(pos.get("contracts"))
            if contracts <= 0:
                continue

            try:
                orders = exchange.fetch_open_orders(symbol, params={"ordType": "conditional"})
            except ccxt.RateLimitExceeded:
                time.sleep(0.2)
                orders = []
            except Exception:
                orders = []

            open_order = orders[0] if orders else {}
            order_info = open_order.get("info", {}) if isinstance(open_order, dict) else {}
            sl = safe_float(order_info.get("slTriggerPx"))
            tp = safe_float(order_info.get("tpTriggerPx"))
            algo_id = open_order.get("id") if isinstance(open_order, dict) else None
            algo_amount = safe_float(open_order.get("amount")) if isinstance(open_order, dict) else 0.0

            contract_size = safe_float(
                pos.get("contractSize"),
                safe_float(pos.get("info", {}).get("mult"), 1.0),
            )
            coin_size = contracts * contract_size if contract_size else contracts
            entry_price = safe_float(pos.get("entryPrice"))
            mark_price = safe_float(pos.get("markPrice"), safe_float(pos.get("info", {}).get("markPx")))
            current_price = mark_price or entry_price
            liquidation_price = safe_float(
                pos.get("liquidationPrice"),
                safe_float(pos.get("info", {}).get("liqPx")),
            )
            unrealized_pnl = safe_float(pos.get("unrealizedPnl"), safe_float(pos.get("info", {}).get("upl")))
            leverage = safe_float(pos.get("leverage"), safe_float(pos.get("info", {}).get("lever")))

            notional_usd = coin_size * current_price if current_price else 0.0
            risk_usd = 0.0
            if sl and entry_price:
                risk_usd = abs(entry_price - sl) * coin_size

            position_obj[coin] = {
                "side": pos.get("side"),
                "size": contracts,
                "contract_size": contract_size,
                "coin_size": coin_size,
                "entry_price": entry_price,
                "current_price": current_price,
                "liquidation_price": liquidation_price,
                "unrealized_pnl": unrealized_pnl,
                "leverage": leverage,
                "notional_usd": notional_usd,
                "risk_usd": risk_usd,
                "symbol": pos.get("symbol"),
                "tp": None if tp == '' else tp,
                "sl": None if sl == '' else sl,
                "algoId": algo_id,
                "algoAmount": algo_amount,
            }

        return position_obj

    return position_obj


def build_position_payload(
    position_obj: Dict[str, Dict[str, Any]],
    signal_history: Optional[Dict[str, List[Dict[str, Any]]]] = None,
) -> List[Dict[str, Any]]:
    payload: List[Dict[str, Any]] = []
    signal_history = signal_history or {}

    for coin, position in (position_obj or {}).items():
        confidence_score = 0.0
        history_key = coin if coin in signal_history else coin.upper()
        coin_history = signal_history.get(history_key, [])
        if coin_history:
            last_signal = coin_history[-1]
            confidence_raw = last_signal.get("confidence_score", last_signal.get("confidence"))
            if isinstance(confidence_raw, (int, float)):
                confidence_score = float(confidence_raw)
            elif isinstance(confidence_raw, str):
                confidence_map = {"LOW": 0.3, "MEDIUM": 0.6, "HIGH": 0.9}
                confidence_score = confidence_map.get(confidence_raw.upper(), 0.0)

        quantity = safe_float(position.get("coin_size"), safe_float(position.get("size")))
        entry_price = safe_float(position.get("entry_price"))
        current_price = safe_float(position.get("current_price") or entry_price)
        liquidation_price = safe_float(position.get("liquidation_price"))
        unrealized_pnl = safe_float(position.get("unrealized_pnl"))
        leverage = safe_float(position.get("leverage"))
        profit_target = safe_float(position.get("tp"))
        stop_loss = safe_float(position.get("sl"))
        risk_usd = safe_float(position.get("risk_usd"))
        notional_usd = safe_float(position.get("notional_usd"))

        payload.append(
            {
                "symbol": position.get("symbol", f"{coin}/USDT:USDT"),
                "quantity": quantity,
                "entry_price": entry_price,
                "current_price": current_price,
                "liquidation_price": liquidation_price,
                "unrealized_pnl": unrealized_pnl,
                "leverage": leverage,
                "exit_plan": {
                    "profit_target": profit_target,
                    "stop_loss": stop_loss,
                },
                "confidence": confidence_score,
                "risk_usd": risk_usd,
                "notional_usd": notional_usd,
            }
        )

    return payload


__all__ = [
    "performance_tracker",
    "safe_float",
    "compute_account_metrics",
    "format_position",
    "get_current_positions",
    "build_position_payload",
]
