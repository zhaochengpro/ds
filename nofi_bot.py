import os
import time
import schedule
from openai import OpenAI
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime
import json
import re
from dotenv import load_dotenv
import argparse
import logging
from logging.handlers import RotatingFileHandler
import inspect


load_dotenv()

# 初始化DeepSeek客户端
deepseek_client = OpenAI(
    api_key=os.getenv('OPENROUTER_API_KEY'),
    base_url="https://openrouter.ai/api/v1"
)
AI_MODEL = os.getenv('DEEPSEEK_MODEL', 'deepseek/deepseek-chat-v3.1')
# 初始化OKX交易所
exchange = ccxt.okx({
    'options': {
        'defaultType': 'swap',  # OKX使用swap表示永续合约
    },
    'apiKey': os.getenv('OKX_API_KEY'),
    'secret': os.getenv('OKX_SECRET'),
    'password': os.getenv('OKX_PASSWORD'),  # OKX需要交易密码
})

start_time = datetime.now()
minutes_elapsed = 0

def parse_args():
    parser = argparse.ArgumentParser(
        prog="ai_trade_bot_ok_plus.py",
        description="这是一个示例：如何用 argparse 获取命令行参数",
    )
    # 添加「位置参数」（必选），类型为 str
    parser.add_argument("--symbols", nargs="+", type=str, help="输入数字数组")
    parser.add_argument("--timeframe", type=str, help="输入时间周期")
    parser.add_argument("--klineNum", type=int, help="输入K线数量")
    args = parser.parse_args()
    return args


args = parse_args()

coin_list = args.symbols
logger = logging.getLogger("ai_trade_bot")
coin_loggers = {}


def get_coin_logger(coin: str):
    if not coin:
        return logger
    return coin_loggers.get(coin.upper(), logger)


def setup_log():
    global coin_loggers

    LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
    os.makedirs(LOG_DIR, exist_ok=True)

    formatter = logging.Formatter('%(asctime)s | %(levelname)-7s | %(message)s')

    # reset base logger handlers to avoid duplicates when重启
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    logger.setLevel(logging.INFO)
    logger.propagate = False

    general_log_file = os.path.join(LOG_DIR, "ai_trade_bot.log")
    general_handler = RotatingFileHandler(general_log_file, maxBytes=5_000_000, backupCount=5, encoding='utf-8')
    general_handler.setLevel(logging.INFO)
    general_handler.setFormatter(formatter)
    logger.addHandler(general_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    coin_loggers = {}
    for coin in coin_list:
        coin_key = coin.upper()
        coin_logger = logging.getLogger(f"ai_trade_bot.{coin_key}")
        coin_logger.setLevel(logging.INFO)
        coin_logger.propagate = True

        for handler in list(coin_logger.handlers):
            coin_logger.removeHandler(handler)
            handler.close()

        log_file = os.path.join(LOG_DIR, f"{coin.lower()}_trade_bot.log")
        file_handler = RotatingFileHandler(log_file, maxBytes=5_000_000, backupCount=5, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        coin_logger.addHandler(file_handler)

        coin_loggers[coin_key] = coin_logger

    return logger



# 交易参数配置 - 结合两个版本的优点
TRADE_CONFIG = {
    'amount': 0.013,  # 交易数量 (BTC)
    'timeframe': args.timeframe,
    'data_points': args.klineNum,
    'analysis_periods': {
        'short_term': 20,  # 短期均线
        'medium_term': 50,  # 中期均线
        'long_term': 96  # 长期趋势
    }
}

# 全局变量存储历史数据
price_history = []
signal_history = {}
position = None


def setup_exchange(leverage, symbol, posSide):
    """设置交易所参数"""
    coin_code = symbol.split('/')[0] if '/' in symbol else symbol
    coin_logger = get_coin_logger(coin_code)

    try:
        exchange.set_leverage(
            leverage,
            symbol,
            {'mgnMode': 'cross', 'posSide': posSide}
        )
        time.sleep(5)
        coin_logger.info(f"杠杆设置 | {leverage}x | 方向: {posSide}")

        usdt_balance = get_usdt_balance()
        coin_logger.info(f"账户余额 | 可用USDT: {usdt_balance:.2f}")

        return True
    except Exception as e:
        coin_logger.error(f"杠杆设置失败: {e}")
        return False


def calculate_technical_indicators(df):
    """计算技术指标 - 来自第一个策略"""
    try:
        df = df.copy()

        # 移动平均线
        df['sma_5'] = df['close'].rolling(window=5, min_periods=1).mean()
        df['sma_20'] = df['close'].rolling(window=20, min_periods=1).mean()
        df['sma_50'] = df['close'].rolling(window=50, min_periods=1).mean()

        # 指数移动平均线
        df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']

        # 相对强弱指数 (RSI)
        delta = df['close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1/14, min_periods=1, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/14, min_periods=1, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi_default = (100 - (100 / (1 + rs))).to_numpy()
        zero_loss = (avg_loss == 0).to_numpy()
        zero_gain = (avg_gain == 0).to_numpy()
        both_zero = zero_loss & zero_gain
        rsi_values = np.select(
            [both_zero, zero_loss, zero_gain],
            [50, 100, 0],
            default=rsi_default
        )
        df['rsi'] = pd.Series(rsi_values, index=df.index).ffill().clip(lower=0, upper=100)

        # 布林带
        df['bb_middle'] = df['close'].rolling(window=20, min_periods=1).mean()
        bb_std = df['close'].rolling(window=20, min_periods=1).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        bb_range = df['bb_upper'] - df['bb_lower']
        bb_ratio = (df['close'] - df['bb_lower']) / bb_range.replace(0, np.nan)
        df['bb_position'] = bb_ratio.clip(lower=0, upper=1)
        df.loc[bb_range.abs() < np.finfo(float).eps, 'bb_position'] = 0.5

        # 成交量均线
        df['volume_ma'] = df['volume'].rolling(window=20, min_periods=1).mean()
        volume_ma_safe = df['volume_ma'].replace(0, np.nan)
        df['volume_ratio'] = (df['volume'] / volume_ma_safe).fillna(0)

        # 支撑阻力位
        df['resistance'] = df['high'].rolling(20).max()
        df['support'] = df['low'].rolling(20).min()

        # 填充NaN值
        df = df.replace([np.inf, -np.inf], np.nan).ffill()

        return df
    except Exception as e:
        logger.error(f"技术指标计算失败: {e}")
        return df


def get_support_resistance_levels(df, lookback=20):
    """计算支撑阻力位"""
    try:
        recent_high = df['high'].tail(lookback).max()
        recent_low = df['low'].tail(lookback).min()
        current_price = df['close'].iloc[-1]

        resistance_level = recent_high
        support_level = recent_low

        # 动态支撑阻力（基于布林带）
        bb_upper = df['bb_upper'].iloc[-1]
        bb_lower = df['bb_lower'].iloc[-1]

        return {
            'static_resistance': resistance_level,
            'static_support': support_level,
            'dynamic_resistance': bb_upper,
            'dynamic_support': bb_lower,
            'price_vs_resistance': ((resistance_level - current_price) / current_price) * 100,
            'price_vs_support': ((current_price - support_level) / support_level) * 100
        }
    except Exception as e:
        logger.error(f"支撑阻力计算失败: {e}")
        return {}


def get_market_trend(df):
    """判断市场趋势"""
    try:
        current_price = df['close'].iloc[-1]

        # 多时间框架趋势分析
        trend_short = "上涨" if current_price > df['sma_20'].iloc[-1] else "下跌"
        trend_medium = "上涨" if current_price > df['sma_50'].iloc[-1] else "下跌"

        # MACD趋势
        macd_trend = "bullish" if df['macd'].iloc[-1] > df['macd_signal'].iloc[-1] else "bearish"

        # 综合趋势判断
        if trend_short == "上涨" and trend_medium == "上涨":
            overall_trend = "强势上涨"
        elif trend_short == "下跌" and trend_medium == "下跌":
            overall_trend = "强势下跌"
        else:
            overall_trend = "震荡整理"

        return {
            'short_term': trend_short,
            'medium_term': trend_medium,
            'macd': macd_trend,
            'overall': overall_trend,
            'rsi_level': df['rsi'].iloc[-1]
        }
    except Exception as e:
        logger.error(f"趋势分析失败: {e}")
        return {}


def get_coins_ohlcv_enhanced(retries = 50):
    """增强版：获取COIN K线数据并计算技术指标"""
    coins_ohlcv = {}
    
    for coin in coin_list:
        for attempt in range(retries):
            try:
                # 获取K线数据
                ohlcv = exchange.fetch_ohlcv(f"{coin}/USDT:USDT", TRADE_CONFIG['timeframe'],
                                            limit=TRADE_CONFIG['data_points'])

                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

                # 计算技术指标
                df = calculate_technical_indicators(df)

                current_data = df.iloc[-1]
                previous_data = df.iloc[-2]

                # 获取技术分析数据
                trend_analysis = get_market_trend(df)
                levels_analysis = get_support_resistance_levels(df)

                coins_ohlcv[coin] = {
                    'price': current_data['close'],
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'high': current_data['high'],
                    'low': current_data['low'],
                    'volume': current_data['volume'],
                    'timeframe': TRADE_CONFIG['timeframe'],
                    'price_change': ((current_data['close'] - previous_data['close']) / previous_data['close']) * 100,
                    'kline_data': df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'sma_5', 'sma_20','sma_50','ema_12','ema_26','macd', 'rsi', 'bb_middle', 'bb_upper', 'bb_lower', 'bb_position']].to_dict('records'),
                    'technical_data': {
                        'sma_5': current_data.get('sma_5', 0),
                        'sma_20': current_data.get('sma_20', 0),
                        'sma_50': current_data.get('sma_50', 0),
                        'rsi': current_data.get('rsi', 0),
                        'macd': current_data.get('macd', 0),
                        'macd_signal': current_data.get('macd_signal', 0),
                        'macd_histogram': current_data.get('macd_histogram', 0),
                        'bb_upper': current_data.get('bb_upper', 0),
                        'bb_lower': current_data.get('bb_lower', 0),
                        'bb_position': current_data.get('bb_position', 0),
                        'volume_ratio': current_data.get('volume_ratio', 0)
                    },
                    'trend_analysis': trend_analysis,
                    'levels_analysis': levels_analysis,
                    'full_data': df
                }

                coin_logger = get_coin_logger(coin)
                coin_logger.info(
                    f"行情更新 | {coin} 价格:${coins_ohlcv[coin]['price']:,.2f} | 涨跌:{coins_ohlcv[coin]['price_change']:+.2f}% | 周期:{TRADE_CONFIG['timeframe']}"
                )
                break
            except Exception as e:
                if attempt == retries - 1:
                    return None
                get_coin_logger(coin).error(f"获取增强K线数据失败: {e}")
                time.sleep(5)
                continue
        
    return coins_ohlcv


def generate_technical_analysis_text(price_data):
    """生成技术分析文本"""

    analysis_text = {}

    for coin, price_item in price_data.items():
        if 'technical_data' not in price_item:
            return "技术指标数据不可用"

        tech = price_item['technical_data']
        trend = price_item.get('trend_analysis', {})
        levels = price_item.get('levels_analysis', {})

        # 检查数据有效性
        def safe_float(value, default=0):
            return float(value) if value and pd.notna(value) else default

        analysis_text[coin] = f"""
        提示：以下所有数据都由96根15分钟 {coin}/USDT k线生成
        【{coin}技术指标分析】（该数据为程序计算生成，仅供参考）
        📈 移动平均线:
        - 5周期: {safe_float(tech['sma_5']):.2f} | 价格相对: {(price_item['price'] - safe_float(tech['sma_5'])) / safe_float(tech['sma_5']) * 100:+.2f}%
        - 20周期: {safe_float(tech['sma_20']):.2f} | 价格相对: {(price_item['price'] - safe_float(tech['sma_20'])) / safe_float(tech['sma_20']) * 100:+.2f}%
        - 50周期: {safe_float(tech['sma_50']):.2f} | 价格相对: {(price_item['price'] - safe_float(tech['sma_50'])) / safe_float(tech['sma_50']) * 100:+.2f}%

        🎯 趋势分析:
        - 短期趋势: {trend.get('short_term', 'N/A')}
        - 中期趋势: {trend.get('medium_term', 'N/A')}
        - 整体趋势: {trend.get('overall', 'N/A')}
        - MACD方向: {trend.get('macd', 'N/A')}

        📊 动量指标:
        - RSI: {safe_float(tech['rsi']):.2f} ({'超买' if safe_float(tech['rsi']) > 70 else '超卖' if safe_float(tech['rsi']) < 30 else '中性'})
        - MACD: {safe_float(tech['macd']):.4f}
        - 信号线: {safe_float(tech['macd_signal']):.4f}

        🎚️ 布林带位置: {safe_float(tech['bb_position']):.2%} ({'上部' if safe_float(tech['bb_position']) > 0.7 else '下部' if safe_float(tech['bb_position']) < 0.3 else '中部'})

        💰 关键水平:
        - 静态阻力: {safe_float(levels.get('static_resistance', 0)):.2f}
        - 静态支撑: {safe_float(levels.get('static_support', 0)):.2f}
        """

    return analysis_text


def get_current_position(data_price, retries=50):
    """获取当前持仓情况 - OKX版本"""
    position_obj = {}
    for attempt in range(retries):
        for coin, _ in data_price.items():
            try:
                positions = exchange.fetch_positions([f"{coin}/USDT:USDT"])
                # logger.info(f"positions: {positions}")
                for pos in positions:
                    if pos['symbol'] == f"{coin}/USDT:USDT":
                        contracts = float(pos['contracts']) if pos['contracts'] else 0

                        if contracts > 0:
                            orders = exchange.fetch_open_orders(f"{coin}/USDT:USDT", params={'ordType': 'oco'})
                            # logger.info(f"orders: {orders}")
                            sl = 0
                            tp = 0
                            if len(orders) > 0:
                                open_order = orders[0]
                                sl = open_order.get('info').get('slOrdPx')
                                tp = open_order.get('info').get('tpOrdPx')

                            position_obj[coin] = {
                                'side': pos['side'],  # 'long' or 'short'
                                'size': contracts,
                                'entry_price': float(pos['entryPrice']) if pos['entryPrice'] else 0,
                                'unrealized_pnl': float(pos['unrealizedPnl']) if pos['unrealizedPnl'] else 0,
                                'leverage': pos['leverage'],
                                'symbol': pos['symbol'],
                                'tp': tp,
                                'sl': sl,
                                'algoId': open_order.get('id', None),
                                'algoAmount': open_order.get('amount', 0)
                            }

            except Exception as e:
                logger.exception("获取持仓失败")
                if attempt == retries - 1:
                    return None
                continue
    return position_obj


def summarize_position_entry(coin, position):
    if not position:
        return f"{coin}: 无持仓"

    entry_price = position.get('entry_price')
    tp = position.get('tp') or '未设置'
    sl = position.get('sl') or '未设置'
    entry_display = f"{entry_price:.4f}" if entry_price else "0"
    return (
        f"{coin}: 方向 {position.get('side', 'N/A')} | 数量 {position.get('size', 0)} | 入场 {entry_display} | "
        f"盈亏 {position.get('unrealized_pnl', 0):.2f}USDT | 止盈 {tp} | 止损 {sl}"
    )


def summarize_positions(position_map):
    if not position_map:
        return "无持仓"
    parts = []
    for coin, pos in position_map.items():
        parts.append(summarize_position_entry(coin, pos))
    return " || ".join(parts)


def safe_json_parse(json_str):
    """安全解析JSON，处理格式不规范的情况"""
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        try:
            # 修复常见的JSON格式问题
            json_str = json_str.replace("'", '"')
            json_str = re.sub(r'(\w+):', r'"\1":', json_str)
            json_str = re.sub(r',\s*}', '}', json_str)
            json_str = re.sub(r',\s*]', ']', json_str)
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"JSON解析失败，原始内容: {json_str}")
            logger.error(f"错误详情: {e}")
            return None


def create_fallback_signal(price_data):
    """创建备用交易信号"""
    return {
        "signal": "HOLD",
        "reason": "因技术分析暂时不可用，采取保守策略",
        "stop_loss": 0,  # -2%
        "take_profit": 0,  # +2%
        "confidence": "LOW",
        "is_fallback": True
    }


def generate_kline_data(price_data):
    kline_data = {}
    for coin, data_item in price_data.items():
        kline_text = f"【{coin}最近24小时96根15分钟K线数据】\n"
        for i, kline in enumerate(data_item['kline_data']):
            trend = "阳线" if kline['close'] > kline['open'] else "阴线"
            change = ((kline['close'] - kline['open']) / kline['open']) * 100
            kline_text += f"K线{i + 1}: {trend} 开盘:{kline['open']:.2f} 收盘:{kline['close']:.2f} 涨跌:{change:+.2f}% 交易量:{kline['volume']} sma5:{kline['sma_5']} sma20:{kline['sma_20']} sma50:{kline['sma_50']} ema12:{kline['ema_12']} ema26:{kline['ema_26']} macd:{kline['macd']} rsi:{kline['rsi']} 20期布林线中线，上线，下线分别为:{kline['bb_middle']}, {kline['bb_upper']}, {kline['bb_lower']}\n"
        kline_data[coin] = kline_text
    return kline_data

def generate_last_singal(price_data):
    signal_text = {}
    for coin, date_item in price_data.items():
        if coin in signal_history:
            last_signal = signal_history[coin][-1]
            signal_text[coin] = f"\n【{coin}上次交易信号】\n信号: {last_signal.get('signal', 'N/A')}\n信心: {last_signal.get('confidence', 'N/A')}"
    return signal_text

def generate_position(price_data, positions):
    position_texts = {}
    for coin, _ in price_data.items():
        if coin in positions:
            pos = positions[coin]
            position_texts[coin] = f"{pos['side']}仓, 数量: {pos['size']}, 盈亏: {pos['unrealized_pnl']:.2f}USDT, 止盈价格: {pos['tp']} 止损价格: {pos['sl']}\n"
        else:
            position_texts[coin] = "无持仓"

    return position_texts

def generate_current_market(price_data, positions):
    markets = {}
    for coin, date_item in price_data.items():
        markets[coin] = f"""
        【{coin}当前行情】
            - 当前价格: ${date_item['price']:,.2f}
            - 时间: {date_item['timestamp']}
            - 本K线最高: ${date_item['high']:,.2f}
            - 本K线最低: ${date_item['low']:,.2f}
            - 本K线成交量: {date_item['volume']:.2f} {coin}
            - 价格变化: {date_item['price_change']:+.2f}%
            - 当前持仓: {positions[coin]}
            - 当前账户可用余额: {get_usdt_balance() * 0.99:.2f} USDT
        """
    return markets

def generate_full_text(price_data, technical_analysis, klines, signals, markets):
    full_text = ""
    for coin, _ in price_data.items():
        ta = technical_analysis[coin] if coin in technical_analysis else ""
        kline_text = klines[coin] if coin in klines else ""
        signal_text = signals[coin] if coin in signals else ""
        market_text = markets[coin] if coin in markets else ""
        full_text += f"""
        【{coin}数据】
        \t{ta}
        \t{kline_text}
        \t{signal_text}
        \t{market_text}

        """
    return full_text

def generate_coin_market_text(price_data):
    coin_market_text = ""
    for coin, _ in price_data.items():
        coin_market_text += f"""
        ### 所有BTC数据

        **当前快照：**
        - current_price = {price_data[coin]['price']}
        - 当前_EMA20 = {price_data[coin]['ema_20']}
        - 当前_macd = {price_data[coin]['macd']}
        - 当前_rsi（7周期） = {price_data[coin]['rsi_7']}

        **永续合约指标：**
        - 未平仓合约：最新值：{btc_oi_latest} | 平均值：{btc_oi_avg}
        - 资金费率：{btc_funding_rate}

        **日内系列（3分钟间隔，最旧→最新）：**

        中价：[{btc_prices_3m}]

        指数移动平均指标（20周期）：[{btc_ema20_3m}]

        MACD指标：[{btc_macd_3m}]

        RSI指标（7周期）：[{btc_rsi7_3m}]

        RSI指标（14周期）：[{btc_rsi14_3m}]

        **长期背景（4小时周期）：**

        20周期EMA：{btc_ema20_4h} vs. 50周期EMA：{btc_ema50_4h}

        3周期ATR：{btc_atr3_4h} vs. 14周期ATR：{btc_atr14_4h}

        当前成交量：{btc_volume_current} vs. 平均成交量：{btc_volume_avg}

        MACD指标（4小时）：[{btc_macd_4h}]

        RSI指标（14周期，4小时）：[{btc_rsi14_4h}]

        ---
        """
    return coin_market_text

def analyze_with_deepseek(price_data):
    """使用DeepSeek分析市场并生成交易信号（增强版）"""

    # 生成技术分析文本
    technical_analysis = generate_technical_analysis_text(price_data)

    # 构建K线数据文本
    klines= generate_kline_data(price_data)
    
    # 添加上次交易信号
    signals = generate_last_singal(price_data)

    # 添加当前持仓信息
    filter_positions = get_current_position(price_data)
    positions = generate_position(price_data, filter_positions)
    
    # 添加当前行情
    markets = generate_current_market(price_data, positions)
    
    full_text = generate_full_text(price_data, technical_analysis, klines, signals, markets)

    coin_market_text = generate_coin_market_text(price_data)


    balance = exchange.fetch_balance()
    usdt_balance = balance['USDT']['free']

    prompt = f"""
    自您开始交易以来已过去{minutes_elapsed}分钟。

    下方为您提供各类状态数据、价格数据及预测信号，助您发掘超额收益。其下为您的当前账户信息、资产价值、业绩表现、持仓情况等。

    ⚠️ **重要提示：以下所有价格或信号数据均按时间排序：最旧 → 最新**

    **时间周期说明：**除非章节标题另有说明，日内系列数据均以**3分钟间隔**提供。若某币种采用不同间隔，将在该币种专属章节中明确标注。

    ---

    ## 所有币种当前市场状态

    ### 所有BTC数据

    **当前快照：**
    - current_price = {btc_price}
    - 当前_EMA20 = {btc_ema20}
    - 当前_macd = {btc_macd}
    - 当前_rsi（7周期） = {btc_rsi7}

    **永续合约指标：**
    - 未平仓合约：最新值：{btc_oi_latest} | 平均值：{btc_oi_avg}
    - 资金费率：{btc_funding_rate}

    **日内系列（3分钟间隔，最旧→最新）：**

    中价：[{btc_prices_3m}]

    指数移动平均指标（20周期）：[{btc_ema20_3m}]

    MACD指标：[{btc_macd_3m}]

    RSI指标（7周期）：[{btc_rsi7_3m}]

    RSI指标（14周期）：[{btc_rsi14_3m}]

    **长期背景（4小时周期）：**

    20周期EMA：{btc_ema20_4h} vs. 50周期EMA：{btc_ema50_4h}

    3周期ATR：{btc_atr3_4h} vs. 14周期ATR：{btc_atr14_4h}

    当前成交量：{btc_volume_current} vs. 平均成交量：{btc_volume_avg}

    MACD指标（4小时）：[{btc_macd_4h}]

    RSI指标（14周期，4小时）：[{btc_rsi14_4h}]

    ---

    ### 所有ETH数据

    **当前快照：**
    - current_price = {eth_price}
    - 当前_20期指数移动平均线 = {eth_ema20}
    - 当前_macd = {eth_macd}
    - 当前_rsi（7周期） = {eth_rsi7}

    **永续合约指标：**
    - 未平仓合约：最新值：{eth_oi_latest} | 平均值：{eth_oi_avg}
    - 资金费率：{eth_funding_rate}

    **日内系列（3分钟间隔，按时间倒序排列）：**

    中价：[{eth_prices_3m}]

    指数移动平均指标（20周期）：[{eth_ema20_3m}]

    MACD指标：[{eth_macd_3m}]

    RSI指标（7周期）：[{eth_rsi7_3m}]

    RSI指标（14周期）：[{eth_rsi14_3m}]

    **长期背景（4小时周期）：**

    20周期EMA：{eth_ema20_4h} vs. 50周期EMA：{eth_ema50_4h}

    3周期ATR：{eth_atr3_4h} vs. 14周期ATR：{eth_atr14_4h}

    当前成交量：{eth_volume_current} vs. 平均成交量：{eth_volume_avg}

    MACD指标（4小时）：[{eth_macd_4h}]

    RSI指标（14周期，4小时）：[{eth_rsi14_4h}]

    ---

    ### 所有SOL数据

    [结构与BTC/ETH相同...]

    ---

    ### 所有BNB数据

    [与BTC/ETH相同结构...]

    ---

    ### 所有DOGE数据

    [与BTC/ETH相同结构...]

    ---

    ### 所有瑞波币数据

    [与BTC/ETH相同结构...]

    ---

    ## 您的账户信息与表现

    **绩效指标：**
    - 当前总回报率（百分比）：{return_pct}%
    - 夏普比率：{sharpe_ratio}

    **账户状态：**
    - 可用现金：${cash_available}
    - **当前账户价值：** ${account_value}

    **当前持仓与业绩：**

    ```python
    [
    {{
    'symbol': '{coin_symbol}',
    '数量': {持仓数量},
    '买入价': {买入价},
    '当前价格': {当前价格},
    '止损价': {止损价},
    '未实现盈亏': {unrealized_pnl},
    '杠杆': {杠杆},
    '退出计划': {
    '盈利目标': {盈利目标},
    '止损': {止损},
    '失效条件': '{失效条件}'
    },
    'confidence': {confidence},
    '风险美元': {风险美元},
    '名义本金美元': {名义本金美元}
    }},
    # ... 如有其他职位则在此处列出
    ]
    ```

    若无开放职位：
    ```python
    []
    ```

    根据上述数据，请以要求的JSON格式提供您的交易决策。
    """

    try:
        response = deepseek_client.chat.completions.create(
            model=AI_MODEL,
            messages=[
                {"role": "system",
                 'content': f"""
                # 角色与身份

                您是自主运行的加密货币交易代理，在OKX中心化交易所的实时市场中运作。

                您的身份：AI交易模型[{AI_MODEL}]
                你的使命：通过系统化、纪律化的交易策略实现风险调整后收益（PnL）最大化。

                ---

                # 交易环境规范

                ## 市场参数

                - **交易所**：OKX（中心化交易所）
                - **资产池**：BTC、ETH、SOL、BNB、DOGE、XRP（永续合约）
                - **初始资金**：{usdt_balance}美元
                - **交易时段**：全天候连续交易
                - **决策频率**：每2-3分钟一次（中低频交易）
                - **杠杆范围**：1倍至20倍（根据信心审慎使用）

                ## 交易机制

                - **合约类型**：永续合约（无到期日）
                - **资金结算机制**：
                - 正资金费率 = 多头支付空头（市场看涨情绪）
                - 负资金费率 = 做空方支付做多方（看跌市场情绪）
                - **交易手续费**：每笔交易约0.02-0.05%（按挂单/吃单费率执行）
                - **滑点**：市价单预计0.01-0.1%，具体取决于交易规模

                ---

                # 操作空间定义

                每个决策周期仅有四种可能操作：

                1. **买入建仓**：开立新多头头寸（押注价格上涨）
                - 适用场景：技术面看涨、动能积极、风险回报率利好上涨

                2. **卖入开仓**：建立新空头头寸（押注价格下跌）
                - 适用场景：技术面看跌、动能疲软、风险回报率利空

                3. **持仓**：维持现有仓位不变
                - 适用场景：现有头寸表现符合预期，或无明显优势

                4. **平仓**：完全退出现有头寸
                - 适用场景：盈利目标达成、止损触发或交易逻辑失效

                ## 持仓管理限制

                - **禁止金字塔式加仓**：不得追加现有仓位（每种币种最多持有一个仓位）
                - **禁止对冲**：不得同时持有同一资产的多空头寸
                - **禁止部分平仓**：必须一次性平掉全部仓位

                ---

                # 仓位规模框架

                按此公式计算仓位规模：

                仓位规模（美元）= 可用现金 × 杠杆倍数 × 分配比例
                持仓规模（币种）= 持仓规模（美元）÷ 当前价格

                ## 规模考量因素

                1. **可用资本**：仅使用可用现金（非账户总值）
                2. **杠杆选择**：
                - 低信心（0.3-0.5）：使用1-3倍杠杆
                - 中度信心（0.5-0.7）：使用3-8倍杠杆
                - 高信心（0.7-1.0）：采用8-20倍杠杆
                3. **分散投资**：避免单一仓位占比超过40%
                4. **费用影响**：持仓金额低于500美元时，手续费将显著侵蚀利润
                5. **强制平仓风险**：确保平仓价格距建仓价高出15%以上

                ---

                # 风险管理规程（强制执行）

                每次交易决策时，必须明确指定：

                1. **盈利目标** (浮动值)：精确止盈价格位
                - 需提供至少2:1的风险回报比
                - 依据技术阻力位、斐波那契扩展位或波动率区间设定

                2. **止损价**（浮点型）：精确止损价格位
                - 每笔交易亏损应控制在账户价值的1-3%内
                - 设置于近期支撑/阻力位之外以避免过早止损

                3. **止损失效条件** (字符串)：使交易策略失效的特定市场信号
                - 示例："BTC跌破10万美元"、"RSI跌破30"、"资金费率转负"
                - 必须客观可验证

                4. **confidence** (浮点数, 0-1): 对该交易的信心程度
                - 0.0-0.3：低信心（避免交易或采用最小仓位）
                - 0.3-0.6：中等信心（采用标准仓位）
                - 0.6-0.8：高信心（可扩大仓位规模）
                - 0.8-1.0：极高信心（谨慎操作，警惕过度自信）

                5. **risk_usd** (浮点型)：风险金额（入场价至止损位的距离）
                - 计算公式：|入场价 - 止损价| × 仓位规模 × 杠杆倍数

                ---

                # 输出格式规范

                请以**有效JSON对象**形式返回决策结果，必须包含以下字段：

                ```json
                {
                "signal": "买入入场" | "卖出入场" | "持有" | "平仓",
                "coin": "BTC" | "ETH" | "SOL" | "BNB" | "DOGE" | "XRP",
                "数量": <浮点数>,
                "杠杆": <1-20之间的整数>,
                "盈利目标": <浮点数>,
                "止损": <浮点数>,
                "失效条件": "<字符串>",
                "置信度": <浮点数 0-1>,
                "risk_usd": <float>,
                "justification": "<string>"
                }
                ```

                ## 输出验证规则

                - 所有数值字段必须为正数（信号为"持仓"时除外）
                - 止盈价：多单需高于开仓价，空单需低于开仓价
                - 止损价：多单必须低于入场价，空单必须高于入场价
                - 操作说明需简明扼要（最多500字符）
                - 当信号为"持仓"时：设置数量=0，杠杆=1，风险字段使用占位符值

                ---

                # 绩效指标与反馈

                每次调用时将获取夏普比率：

                夏普比率 = (平均收益率 - 无风险利率) / 收益率标准差

                解读：
                - < 0：平均处于亏损状态
                - 0-1：收益为正但波动性高
                - 1-2：风险调整后表现良好
                - > 2：风险调整后表现卓越

                运用夏普比率校准投资行为：
                - 夏普比率低 → 缩减仓位规模，收紧止损位，提高选择性
                - 高夏普比率 → 当前策略有效，保持纪律性

                ---

                # 数据解读指南

                ## 技术指标说明

                **EMA（指数移动平均线）**：趋势方向
                - 价格 > EMA = 上升趋势
                - 价格 < EMA = 下行趋势

                **MACD（移动平均线收敛/发散指标）**：动量指标
                - 正值MACD = 看涨动能
                - MACD为负值 = 空头动能

                **RSI（相对强弱指数）**：超买/超卖状态
                - RSI > 70 = 超买（潜在下跌反转）
                - RSI < 30 = 超卖（可能反转上涨）
                - RSI 40-60 = 中性区域

                **ATR（平均真实波动幅度）**：波动性衡量指标
                - ATR值越高 = 波动性越强（需设置更宽止损位）
                - ATR较低 = 波动性较小（可设置更窄止损）

                **未平仓合约**：总流通合约量
                - 未平仓量上升 + 价格上涨 = 强劲上升趋势
                - 未平仓量上升 + 价格下跌 = 强劲下跌趋势
                - 未平仓量下降 = 趋势减弱

                **资金费率**：市场情绪指标
                - 正值资金费率 = 看涨情绪（多头支付空头）
                - 负费率 = 看跌情绪（空头支付多头）
                - 极端资金费率（>0.01%）= 潜在反转信号

                ## 数据排序（关键）

                ⚠️ **所有价格与指标数据均按：最旧 → 最新排序**

                **数组中的最后一个元素即为最新数据点。**
                **数组首项即为最旧数据点。**

                切勿混淆排序顺序。此为常见错误，将导致决策失误。

                ---

                # 操作限制

                ## 您无法访问的内容

                - 无新闻推送或社交媒体情绪分析
                - 无对话历史（每次决策均为无状态）
                - 无法调用外部API
                - 无法获取中间价以外的订单簿深度
                - 无法下达限价单（仅支持市价单）

                ## 必须从数据中推断的内容

                - 市场叙事与情绪（通过价格走势+资金费率解读）
                - 机构持仓布局（通过未平仓合约变化判断）
                - 趋势强度与可持续性（通过技术指标判断）
                - 风险偏好与风险规避模式（通过跨币种相关性判断）

                ---

                # 交易哲学与最佳实践

                ## 核心原则

                1. **资金保全优先**：保护本金比追逐收益更重要
                2. **纪律胜于情绪**：严格执行止损计划，切勿随意调整止损位或目标位
                3. **质量重于数量**：少数高确信度交易胜过大量低确信度交易
                4. **顺应波动**：根据市场状况调整仓位规模
                5. **顺应趋势**：勿逆强劲方向性行情而为

                ## 常见陷阱需规避

                - ⚠️ **过度交易**：频繁交易将通过手续费蚕食本金
                - ⚠️ **报复性交易**：切勿在亏损后加仓试图"挽回损失"
                - ⚠️ **分析瘫痪**：勿等待完美交易机会，世上本无完美布局
                - ⚠️ **忽视相关性**：比特币常引领山寨币走势，请优先关注比特币
                - ⚠️ **过度杠杆**：高杠杆会放大收益与亏损

                ## 决策框架

                1. 优先分析当前持仓（表现是否符合预期？）
                2. 检查现有交易的失效条件
                3. 仅在资金充足时筛选新机会
                4. 风险管理优先于利润最大化
                5. 犹豫时选择"持仓"而非强行交易

                ---

                # 窗口管理背景

                上下文信息有限。提示包含：
                - 每个指标约10个近期数据点（3分钟间隔）
                - 4小时周期约10个近期数据点
                - 当前账户状态及持仓情况

                优化分析策略：
                - 关注最近3-5个数据点获取短期信号
                - 运用4小时数据把握趋势背景及支撑/阻力位
                - 无需死记硬背所有数字，重点识别模式规律

                ---

                # 最终说明

                1. 决策前务必仔细阅读完整用户提示
                2. 核对仓位计算（双重检查）
                3. 确保生成的JSON输出格式正确且内容完整
                4. 提供真实的信心评分（切勿夸大判断力度）
                5. 严格执行止损计划（切勿提前放弃止损位）

                谨记：您正在真实市场中使用真实资金交易。每个决策都将产生后果。请系统化交易、严格管控风险，让概率在时间长河中为您创造优势。

                现在，请分析下方提供的市场数据并作出交易决策。
                """},
                {"role": "user", "content": prompt}
            ],
            stream=False,
            temperature=0.1
        )

        # 安全解析JSON
        result = response.choices[0].message.content
        logger.info(f"DeepSeek回复片段: {result}")

        # 提取JSON部分
        start_idx = result.find('[')
        end_idx = result.rfind(']') + 1

        signal_data = []

        if start_idx != -1 and end_idx != 0:
            signal_data = safe_json_parse(result)

            if signal_data is None:
                raise TypeError('AI返回类型错误')
        else:
            raise TypeError('AI返回类型错误')
        
 
        
        # 验证必需字段
        for item in signal_data:
            required_fields = ['signal', 'reason', 'stop_loss', 'take_profit', 'confidence', 'amount', 'coin', 'usdt_amount']
            if not all(field in item for field in required_fields):
                raise ValueError('AI返回代币json中参数不存在')

            # 保存信号到历史记录
            item['timestamp'] = price_data[item['coin']]['timestamp']
            if not item['coin'] in signal_history:
                signal_history[item['coin']] = []
                signal_history[item['coin']].append(item)
            else:
                signal_history[item['coin']].append(item)
                if len(signal_history[item['coin']]) > 30:
                    signal_history[item['coin']].pop(0)

            # logger.info(signal_history)
            # 信号统计
            coin_logger = get_coin_logger(item['coin'])

            if item['coin'] in signal_history:
                signal_count = 0
                for s in signal_history[item['coin']]:
                    if s.get('signal') == item['signal']:
                        signal_count += 1
                total_signals = len(signal_history[item['coin']])
                coin_logger.info(
                    f"信号统计 | {item['signal']} | 最近{total_signals}次出现{signal_count}次"
                )

            # 信号连续性检查
            if len(signal_history[item['coin']]) >= 3:
                last_three = [s['signal'] for s in signal_history[item['coin']][-3:]]
                if len(set(last_three)) == 1:
                    coin_logger.warning(f"连续重复信号 | 最近3次均为{item['signal']}")

        return signal_data

    except Exception as e:
        logger.exception("DeepSeek分析失败")
        return create_fallback_signal(price_data)
    
def get_usdt_balance():
    # 获取账户余额
    balance = exchange.fetch_balance()
    usdt_balance = balance['USDT']['free']
    return usdt_balance


def execute_trade(signal_data, price_data_obj):
    """执行交易 - OKX版本（修复保证金检查）"""
    """成功能够执行的订单必须先设置倍数"""
    global position

    pos_obj = get_current_position(price_data_obj)

    for signal in signal_data:
        coin = signal['coin']
        coin_logger = get_coin_logger(coin)
        coin_logger.info(f"=" * 60)
        coin_logger.info(f"=" * int((60 - len(coin)) / 2) + coin + f"=" * int((60 - len(coin)) / 2))
        coin_logger.info(f"=" * 60)
        coin_logger.info(f"代币：{coin}")
        price_data = price_data_obj[coin]
        current_position = pos_obj.get(coin)
        posSide = 'long' if signal['signal'] == 'BUY' else 'short'
        leverage = int(signal['leverage'])

        if current_position and signal['signal'] != 'HOLD':
            current_side = current_position['side']
            if signal['signal'] == 'BUY':
                new_side = 'long'
            elif signal['signal'] == 'SELL':
                new_side = 'short'
            else:
                new_side = None

            if new_side != current_side:
                if signal['confidence'] != 'HIGH':
                    coin_logger.info(
                        f"信号忽略 | 低信心反转 | 当前:{current_side} -> 建议:{new_side}"
                    )
                    return

                if len(signal_history[coin]) >= 2:
                    last_signals = [s['signal'] for s in signal_history[coin][-2:]]
                    if signal['signal'] in last_signals:
                        coin_logger.info(
                            f"信号忽略 | 近期已出现{signal['signal']} | 避免频繁反转"
                        )
                        return

        coin_logger.info(
            f"信号摘要 | 动作:{signal['signal']} | 信心:{signal['confidence']} | 杠杆:{leverage}x | 数量:{signal['amount']:,.5f} | USDT:{signal['usdt_amount']:,.2f}"
        )
        coin_logger.info(f"理由: {signal['reason']}")
        coin_logger.info(
            f"止损/止盈 | {signal['stop_loss']:,.2f} / {signal['take_profit']:,.2f}"
        )

        usdt_amount = float(signal['usdt_amount'])

        amount_obj = get_fact_amount(
            f"{coin}/USDT:USDT", usdt_amount * 0.9, leverage, price_data['price']
        )

        op_amount = amount_obj.get('amount')
        margin_needed = amount_obj.get('margin_needed')

        if signal['confidence'] == 'LOW':
            coin_logger.warning("低信心信号，跳过执行")
            continue

        try:
            balance = exchange.fetch_balance()
            usdt_balance = balance['USDT']['free']

            coin_logger.info(
                f"资金检查 | 预估保证金:{margin_needed:.2f} | 可用:{usdt_balance:.2f}"
            )

            if margin_needed >= usdt_balance:
                coin_logger.warning(
                    f"跳过交易 | 保证金不足 | 需要:{usdt_amount:.2f} | 可用:{usdt_balance:.2f}"
                )
                continue

            if current_position:
                pos_tp = float(current_position.get('tp', 0))
                pos_sl = float(current_position.get('sl', 0))
                current_pos_side = current_position['side']
                algo_amount = float(current_position.get('algoAmount', 0))
            else:
                pos_tp = 0
                pos_sl = 0
                current_pos_side = None
                algo_amount = 0

            tp = signal['take_profit']
            sl = signal['stop_loss']

            if current_position:
                coin_logger.info(
                    f"当前持仓 | {summarize_position_entry(coin, current_position)}"
                )
                coin_logger.info(
                    f"目标调整 | 止盈 {pos_tp:.2f} -> {tp:.2f} | 止损 {pos_sl:.2f} -> {sl:.2f}"
                )
            else:
                coin_logger.info(
                    f"当前持仓 | 无持仓 | 计划止盈 {tp:.2f} | 计划止损 {sl:.2f}"
                )

            if signal['signal'] != 'HOLD':
                setup_exchange(leverage, f"{coin}/USDT:USDT", posSide)

            if signal['signal'] == 'BUY':
                if current_position and current_pos_side == 'short':
                    coin_logger.info("操作 | 平空仓并开多仓")
                    exchange.create_market_order(
                        f"{coin}/USDT:USDT",
                        'buy',
                        current_pos_side,
                        params={'reduceOnly': True, 'tag': '60bb4a8d3416BCDE', 'posSide': 'short'}
                    )
                    time.sleep(1)
                    exchange.create_market_order(
                        f"{coin}/USDT:USDT",
                        'buy',
                        op_amount,
                        params={'tag': 'f1ee03b510d5SUDE', 'posSide': posSide, 'attachAlgoOrds': [{
                            'tpTriggerPx': str(tp),
                            'tpOrdPx': str(tp),
                            'slTriggerPx': str(sl),
                            'slOrdPx': str(sl)
                        }]}
                    )
                elif current_position and current_pos_side == 'long':
                    if (tp != 0 and f"{pos_tp:.2f}" != f"{tp:.2f}") or (sl != 0 and f"{pos_sl:.2f}" != f"{sl:.2f}"):
                        exchange.private_post_trade_cancel_algos([{
                            "instId": f"{coin}-USDT-SWAP",
                            "algoId": current_position['algoId']
                        }])
                        params = {
                            "instId": f"{coin}-USDT-SWAP",
                            "tdMode": "cross",
                            "side": "sell",
                            "ordType": "oco",
                            "sz": algo_amount,
                            "tpTriggerPx": str(tp),
                            "tpOrdPx": str(tp),
                            "slTriggerPx": str(sl),
                            "slOrdPx": str(sl),
                            "posSide": current_pos_side,
                        }
                        exchange.private_post_trade_order_algo(params=params)
                        coin_logger.info(
                            f"调整止盈止损 | 止盈 {pos_tp:.2f} -> {tp:.2f} | 止损 {pos_sl:.2f} -> {sl:.2f}"
                        )

                    coin_logger.info("持仓保持不变 | 维持多头")
                else:
                    coin_logger.info("操作 | 开多仓")
                    exchange.create_market_order(
                        f"{coin}/USDT:USDT",
                        'buy',
                        op_amount,
                        params={'tag': 'f1ee03b510d5SUDE', 'posSide': posSide, 'attachAlgoOrds': [{
                            'tpTriggerPx': str(tp),
                            'tpOrdPx': str(tp),
                            'slTriggerPx': str(sl),
                            'slOrdPx': str(sl)
                        }]}
                    )

            elif signal['signal'] == 'SELL':
                if current_position and current_pos_side == 'long':
                    coin_logger.info("操作 | 平多仓并开空仓")
                    exchange.create_market_order(
                        f"{coin}/USDT:USDT",
                        'sell',
                        current_pos_side,
                        params={'reduceOnly': True, 'tag': 'f1ee03b510d5SUDE', 'posSide': 'long'}
                    )
                    time.sleep(1)
                    exchange.create_market_order(
                        f"{coin}/USDT:USDT",
                        'sell',
                        op_amount,
                        params={'tag': 'f1ee03b510d5SUDE', 'posSide': posSide, 'attachAlgoOrds': [{
                            'tpTriggerPx': str(tp),
                            'tpOrdPx': str(tp),
                            'slTriggerPx': str(sl),
                            'slOrdPx': str(sl)
                        }]}
                    )
                elif current_position and current_pos_side == 'short':
                    if (tp != 0 and f"{pos_tp:.2f}" != f"{tp:.2f}") or (sl != 0 and f"{pos_sl:.2f}" != f"{sl:.2f}"):
                        exchange.private_post_trade_cancel_algos([{
                            "instId": f"{coin}-USDT-SWAP",
                            "algoId": current_position['algoId']
                        }])
                        params = {
                            "instId": f"{coin}-USDT-SWAP",
                            "tdMode": "cross",
                            "side": 'buy',
                            "ordType": "oco",
                            "sz": algo_amount,
                            "tpTriggerPx": str(tp),
                            "tpOrdPx": str(tp),
                            "slTriggerPx": str(sl),
                            "slOrdPx": str(sl),
                            "posSide": current_pos_side,
                        }
                        exchange.private_post_trade_order_algo(params=params)
                        coin_logger.info(
                            f"调整止盈止损 | 止盈 {pos_tp:.2f} -> {tp:.2f} | 止损 {pos_sl:.2f} -> {sl:.2f}"
                        )
                    coin_logger.info("持仓保持不变 | 维持空头")
                else:
                    coin_logger.info("操作 | 开空仓")
                    exchange.create_market_order(
                        f"{coin}/USDT:USDT",
                        'sell',
                        op_amount,
                        params={'tag': 'f1ee03b510d5SUDE', 'posSide': posSide, 'attachAlgoOrds': [{
                            'tpTriggerPx': str(tp),
                            'tpOrdPx': str(tp),
                            'slTriggerPx': str(sl),
                            'slOrdPx': str(sl)
                        }]}
                    )
            elif signal['signal'] == 'HOLD':
                if current_position:
                    if (tp != 0 and f"{pos_tp:.2f}" != f"{tp:.2f}") or (sl != 0 and f"{pos_sl:.2f}" != f"{sl:.2f}"):
                        exchange.private_post_trade_cancel_algos([{
                            "instId": f"{coin}-USDT-SWAP",
                            "algoId": current_position['algoId']
                        }])
                        coin_logger.info(
                            f"取消历史止盈止损"
                        )
                        params = {
                            "instId": f"{coin}-USDT-SWAP",
                            "tdMode": "cross",
                            "side": "sell" if current_pos_side == 'long' else 'buy',
                            "ordType": "oco",
                            "sz": algo_amount,
                            "tpTriggerPx": str(tp),
                            "tpOrdPx": str(tp),
                            "slTriggerPx": str(sl),
                            "slOrdPx": str(sl),
                            "posSide": current_pos_side,
                        }
                        exchange.private_post_trade_order_algo(params=params)
                        coin_logger.info(
                            f"调整止盈止损 | 止盈 {pos_tp:.2f} -> {tp:.2f} | 止损 {pos_sl:.2f} -> {sl:.2f}"
                        )

            coin_logger.info("执行完成 | 已提交订单")
            time.sleep(2)
            position = get_current_position(price_data_obj)
            coin_logger.info(f"最新持仓 | {summarize_positions(position)}")
        except Exception as e:
            coin_logger.exception(f"订单执行失败: {e}")
            import traceback
            traceback.print_exc()

def analyze_with_deepseek_with_retry(price_data, max_retries=50):
    """带重试的DeepSeek分析"""
    for attempt in range(max_retries):
        try:
            signal_data = analyze_with_deepseek(price_data)
            if isinstance(signal_data, list):
                return signal_data
            else:
                logger.warning(f"第{attempt + 1}次尝试失败，进行重试...")
                time.sleep(1)

        except Exception as e:
            logger.warning(f"第{attempt + 1}次尝试异常: {e}")
            if attempt == max_retries - 1:
                return create_fallback_signal(price_data)
            time.sleep(1)

    return create_fallback_signal(price_data)


def trading_bot():
    """主交易机器人函数"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    minutes_elapsed = datetime.now().minute - start_time.minute
    logger.info("=" * 60)
    logger.info(f"执行时间: {timestamp}")
    logger.info(f"执行时长: {minutes_elapsed}分钟")
    logger.info("=" * 60)

    # 1. 获取增强版K线数据
    price_data = get_coins_ohlcv_enhanced()
    if not price_data:
        return

    # 2. 使用DeepSeek分析（带重试）
    signal_data = analyze_with_deepseek_with_retry(price_data)

    # 3. 执行交易
    execute_trade(signal_data, price_data)



def get_fact_amount(symbol, notional, leverage, price):
    mark = exchange.load_markets()
    contract_size = mark[symbol]['contractSize']
    # 计算张数
    position_value = notional * leverage            # 总名义价值
    contract_value = price * contract_size          # 每张合约价值
    amount = position_value / contract_value        # 张数
    margin_needed = 0 if leverage == 0 else (price * contract_size * amount) / leverage

    # print(f"amount: {amount}, margin_needed: {margin_needed}, leverage: {leverage}, price: {price}, contract_size: {contract_size}, position_value: {position_value}, contract_value: {contract_value}")

    return {
        'amount':amount,
        'margin_needed': margin_needed
    }

def main():
    """主函数"""
    exchange.httpsProxy = os.getenv('https_proxy')
    minutes_elapsed = datetime.now().minute

    setup_log()
    logger.info(f"OKX自动交易机器人启动成功！")
    logger.info("融合技术指标策略 + OKX实盘接口")
    logger.info(f"交易周期: {TRADE_CONFIG['timeframe']}")
    logger.info("已启用完整技术指标分析和持仓跟踪功能")

    for coin in coin_list:
        coin_logger = get_coin_logger(coin)
        coin_logger.info("=" * 60)
        coin_logger.info(f"代币：{coin}")
        coin_logger.info(f"交易周期: {TRADE_CONFIG['timeframe']}")
        coin_logger.info("已启用完整技术指标分析和持仓跟踪功能")
        coin_logger.info("=" * 60)

    # 根据时间周期设置执行频率
    frequency_msg = "每小时一次"
    schedule.every(15).minutes.do(trading_bot)

    logger.info(f"执行频率: {frequency_msg}")

    # params = {
    #     "instId": "BTC-USDT-SWAP",  # ✅ 正确
    #     "tdMode": "cross",
    #     "side": "buy",              # 空单平仓用 buy
    #     "ordType": "oco",
    #     "sz": "0.01",
    #     "tpTriggerPx": "200000",
    #     "tpOrdPx": "200000",
    #     "slTriggerPx": "67000",
    #     "slOrdPx": "67000",
    #     "posSide": "short",
    # }
    # resp = exchange.private_post_trade_order_algo(params)
    # open_algos = exchange.private_get_trade_orders_algo_pending({
    #     "ordType": "oco",   # 双向止盈止损
    #     "instId": "XRP-USDT-SWAP"  # 对应合约
    # })
    # print(open_algos)


    # print('done')


    # schedule.every(5).minutes.do(trading_bot)
    # 立即执行一次
    trading_bot()

    # 循环执行
    while True:
        schedule.run_pending()
        time.sleep(1)


if __name__ == "__main__":
    main()
