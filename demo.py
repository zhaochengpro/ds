"""Market data utilities converted from demo.go.

This module mirrors the original Go implementation while using ccxt to
collect Binance futures kline data and related derivatives.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import ccxt


@dataclass
class OIData:
    latest: float
    average: float


@dataclass
class IntradayData:
    mid_prices: List[float] = field(default_factory=list)
    ema20_values: List[float] = field(default_factory=list)
    macd_values: List[float] = field(default_factory=list)
    rsi7_values: List[float] = field(default_factory=list)
    rsi14_values: List[float] = field(default_factory=list)


@dataclass
class LongerTermData:
    ema20: float = 0.0
    ema50: float = 0.0
    atr3: float = 0.0
    atr14: float = 0.0
    current_volume: float = 0.0
    average_volume: float = 0.0
    macd_values: List[float] = field(default_factory=list)
    rsi14_values: List[float] = field(default_factory=list)


@dataclass
class Data:
    symbol: str
    current_price: float
    price_change_1h: float
    price_change_4h: float
    current_ema20: float
    current_macd: float
    current_rsi7: float
    open_interest: Optional[OIData]
    funding_rate: float
    intraday_series: Optional[IntradayData]
    longer_term_context: Optional[LongerTermData]


@dataclass
class Kline:
    open_time: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    close_time: int


_exchange: Optional[ccxt.binance] = None


def get_exchange() -> ccxt.binance:
    global _exchange
    if _exchange is None:
        exchange = ccxt.binance({
            "enableRateLimit": True,
            "options": {"defaultType": "future"},
        })
        try:
            exchange.load_markets()
        except Exception:
            # Loading markets is convenient but not mandatory for raw requests.
            pass
        _exchange = exchange
    return _exchange


def normalize_symbol(symbol: str) -> str:
    symbol = symbol.upper().replace("/", "")
    if not symbol.endswith("USDT"):
        symbol += "USDT"
    return symbol


def to_ccxt_symbol(symbol: str) -> str:
    normalized = normalize_symbol(symbol)
    base = normalized[:-4]
    quote = normalized[-4:]
    return f"{base}/{quote}:{quote}"


def get_klines(symbol: str, interval: str, limit: int) -> List[Kline]:
    exchange = get_exchange()
    ccxt_symbol = to_ccxt_symbol(symbol)
    ohlcvs = exchange.fetch_ohlcv(ccxt_symbol, timeframe=interval, limit=limit)
    klines: List[Kline] = []
    for open_time, open_, high, low, close, volume in ohlcvs:
        klines.append(
            Kline(
                open_time=int(open_time),
                open=float(open_),
                high=float(high),
                low=float(low),
                close=float(close),
                volume=float(volume),
                close_time=int(open_time),
            )
        )
    return klines


def calculate_ema(klines: List[Kline], period: int) -> float:
    if len(klines) < period:
        return 0.0

    closes = [k.close for k in klines]
    sma = sum(closes[:period]) / float(period)
    multiplier = 2.0 / float(period + 1)
    ema = sma
    for price in closes[period:]:
        ema = (price - ema) * multiplier + ema
    return ema


def calculate_macd(klines: List[Kline]) -> float:
    if len(klines) < 26:
        return 0.0
    ema12 = calculate_ema(klines, 12)
    ema26 = calculate_ema(klines, 26)
    return ema12 - ema26


def calculate_rsi(klines: List[Kline], period: int) -> float:
    if len(klines) <= period:
        return 0.0

    gains = 0.0
    losses = 0.0
    closes = [k.close for k in klines]

    for i in range(1, period + 1):
        change = closes[i] - closes[i - 1]
        if change > 0:
            gains += change
        else:
            losses += -change

    avg_gain = gains / float(period)
    avg_loss = losses / float(period)

    for i in range(period + 1, len(closes)):
        change = closes[i] - closes[i - 1]
        if change > 0:
            avg_gain = (avg_gain * float(period - 1) + change) / float(period)
            avg_loss = (avg_loss * float(period - 1)) / float(period)
        else:
            avg_gain = (avg_gain * float(period - 1)) / float(period)
            avg_loss = (avg_loss * float(period - 1) + (-change)) / float(period)

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def calculate_atr(klines: List[Kline], period: int) -> float:
    if len(klines) <= period:
        return 0.0

    trs = [0.0] * len(klines)
    for i in range(1, len(klines)):
        high = klines[i].high
        low = klines[i].low
        prev_close = klines[i - 1].close

        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        trs[i] = max(tr1, tr2, tr3)

    atr = sum(trs[1 : period + 1]) / float(period)

    for i in range(period + 1, len(trs)):
        atr = (atr * float(period - 1) + trs[i]) / float(period)

    return atr


def calculate_intraday_series(klines: List[Kline]) -> IntradayData:
    data = IntradayData()
    start = max(0, len(klines) - 10)

    for idx in range(start, len(klines)):
        subset = klines[: idx + 1]
        data.mid_prices.append(subset[-1].close)

        if idx >= 19:
            data.ema20_values.append(calculate_ema(subset, 20))
        if idx >= 25:
            data.macd_values.append(calculate_macd(subset))
        if idx >= 7:
            data.rsi7_values.append(calculate_rsi(subset, 7))
        if idx >= 14:
            data.rsi14_values.append(calculate_rsi(subset, 14))

    return data


def calculate_longer_term_data(klines: List[Kline]) -> LongerTermData:
    data = LongerTermData()

    data.ema20 = calculate_ema(klines, 20)
    data.ema50 = calculate_ema(klines, 50)
    data.atr3 = calculate_atr(klines, 3)
    data.atr14 = calculate_atr(klines, 14)

    if klines:
        data.current_volume = klines[-1].volume
        data.average_volume = sum(k.volume for k in klines) / float(len(klines))

    start = max(0, len(klines) - 10)
    for idx in range(start, len(klines)):
        subset = klines[: idx + 1]
        if idx >= 25:
            data.macd_values.append(calculate_macd(subset))
        if idx >= 14:
            data.rsi14_values.append(calculate_rsi(subset, 14))

    return data


def call_exchange_method(exchange: ccxt.binance, names: List[str], params: dict) -> Optional[dict]:
    for name in names:
        method = getattr(exchange, name, None)
        if method is None:
            continue
        response = method(params)
        if isinstance(response, list):
            if response:
                return response[0]
        elif isinstance(response, dict):
            return response
    return None


def get_open_interest(symbol: str) -> Optional[OIData]:
    exchange = get_exchange()
    binance_symbol = normalize_symbol(symbol)
    response = call_exchange_method(
        exchange,
        [
            "fapiPublic_get_openinterest",
            "fapiPublicGetOpenInterest",
            "public_get_futures_data",
        ],
        {"symbol": binance_symbol},
    )
    if not response or "openInterest" not in response:
        return None

    latest = float(response["openInterest"])
    return OIData(latest=latest, average=latest * 0.999)


def get_funding_rate(symbol: str) -> float:
    exchange = get_exchange()
    binance_symbol = normalize_symbol(symbol)
    response = call_exchange_method(
        exchange,
        ["fapiPublic_get_premiumindex", "fapiPublicGetPremiumIndex"],
        {"symbol": binance_symbol},
    )
    if response and "lastFundingRate" in response:
        return float(response["lastFundingRate"])
    return 0.0


def get_market_data(symbol: str) -> Data:
    symbol = normalize_symbol(symbol)

    klines_3m = get_klines(symbol, "3m", 40)
    klines_4h = get_klines(symbol, "4h", 60)

    current_price = klines_3m[-1].close if klines_3m else 0.0
    current_ema20 = calculate_ema(klines_3m, 20)
    current_macd = calculate_macd(klines_3m)
    current_rsi7 = calculate_rsi(klines_3m, 7)

    price_change_1h = 0.0
    if len(klines_3m) >= 21:
        price_1h_ago = klines_3m[-21].close
        if price_1h_ago:
            price_change_1h = ((current_price - price_1h_ago) / price_1h_ago) * 100.0

    price_change_4h = 0.0
    if len(klines_4h) >= 2:
        price_4h_ago = klines_4h[-2].close
        if price_4h_ago:
            price_change_4h = ((current_price - price_4h_ago) / price_4h_ago) * 100.0

    oi_data = get_open_interest(symbol)
    funding_rate = get_funding_rate(symbol)

    intraday_series = calculate_intraday_series(klines_3m) if klines_3m else None
    longer_term_context = calculate_longer_term_data(klines_4h) if klines_4h else None

    return Data(
        symbol=symbol,
        current_price=current_price,
        price_change_1h=price_change_1h,
        price_change_4h=price_change_4h,
        current_ema20=current_ema20,
        current_macd=current_macd,
        current_rsi7=current_rsi7,
        open_interest=oi_data,
        funding_rate=funding_rate,
        intraday_series=intraday_series,
        longer_term_context=longer_term_context,
    )


def format_market_data(data: Data) -> str:
    lines: List[str] = []
    lines.append(
        (
            f"current_price = {data.current_price:.2f}, current_ema20 = {data.current_ema20:.3f}, "
            f"current_macd = {data.current_macd:.3f}, current_rsi (7 period) = {data.current_rsi7:.3f}\n"
        )
    )

    lines.append(
        f"In addition, here is the latest {data.symbol} open interest and funding rate for perps:\n"
    )

    if data.open_interest:
        lines.append(
            (
                f"Open Interest: Latest: {data.open_interest.latest:.2f} "
                f"Average: {data.open_interest.average:.2f}\n"
            )
        )

    lines.append(f"Funding Rate: {data.funding_rate:.2e}\n")

    if data.intraday_series:
        series = data.intraday_series
        lines.append("Intraday series (3-minute intervals, oldest → latest):\n")

        if series.mid_prices:
            lines.append(f"Mid prices: {format_float_slice(series.mid_prices)}\n")
        if series.ema20_values:
            lines.append(
                f"EMA indicators (20-period): {format_float_slice(series.ema20_values)}\n"
            )
        if series.macd_values:
            lines.append(f"MACD indicators: {format_float_slice(series.macd_values)}\n")
        if series.rsi7_values:
            lines.append(
                f"RSI indicators (7-Period): {format_float_slice(series.rsi7_values)}\n"
            )
        if series.rsi14_values:
            lines.append(
                f"RSI indicators (14-Period): {format_float_slice(series.rsi14_values)}\n"
            )

    if data.longer_term_context:
        lt = data.longer_term_context
        lines.append("Longer-term context (4-hour timeframe):\n")

        lines.append(
            (
                f"20-Period EMA: {lt.ema20:.3f} vs. 50-Period EMA: {lt.ema50:.3f}\n"
            )
        )
        lines.append(
            (
                f"3-Period ATR: {lt.atr3:.3f} vs. 14-Period ATR: {lt.atr14:.3f}\n"
            )
        )
        lines.append(
            (
                f"Current Volume: {lt.current_volume:.3f} vs. Average Volume: {lt.average_volume:.3f}\n"
            )
        )

        if lt.macd_values:
            lines.append(f"MACD indicators: {format_float_slice(lt.macd_values)}\n")
        if lt.rsi14_values:
            lines.append(
                f"RSI indicators (14-Period): {format_float_slice(lt.rsi14_values)}\n"
            )

    return "\n".join(lines)


def format_float_slice(values: List[float]) -> str:
    formatted = ", ".join(f"{value:.3f}" for value in values)
    return f"[{formatted}]"


__all__ = [
    "Data",
    "OIData",
    "IntradayData",
    "LongerTermData",
    "Kline",
    "get_market_data",
    "format_market_data",
]

