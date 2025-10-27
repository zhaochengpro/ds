import os
import time
import schedule
from openai import OpenAI
import ccxt
import pandas as pd
from datetime import datetime
import json
import re
from dotenv import load_dotenv
import argparse
import logging
from logging.handlers import RotatingFileHandler

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



def parse_args():
    parser = argparse.ArgumentParser(
        prog="ai_trade_bot_ok_plus.py",
        description="这是一个示例：如何用 argparse 获取命令行参数",
    )
    # 添加「位置参数」（必选），类型为 str
    parser.add_argument("--symbols", nargs="+", type=str, help="输入数字数组")
    args = parser.parse_args()
    return args


args = parse_args()

coin_list = args.symbols
logger = logging.getLogger("ai_trade_bot")

def setup_log():
    for coin in coin_list:
        LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
        os.makedirs(LOG_DIR, exist_ok=True)
        LOG_FILE = os.path.join(LOG_DIR, f"{coin.lower()}_trade_bot.log")


        logger.setLevel(logging.INFO)
        logger.propagate = False
        if not logger.handlers:
            formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')

            file_handler = RotatingFileHandler(LOG_FILE, maxBytes=5_000_000, backupCount=5, encoding='utf-8')
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)



# 交易参数配置 - 结合两个版本的优点
TRADE_CONFIG = {
    'amount': 0.013,  # 交易数量 (BTC)
    'timeframe': '15m',  # 使用15分钟K线
    'test_mode': False,  # 测试模式
    'data_points': 96,  # 24小时数据（96根15分钟K线）
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


def setup_exchange(leverage, coin, posSide):
    """设置交易所参数"""
    try:
        # OKX设置杠杆
        exchange.set_leverage(
            leverage,
            coin,
            {'mgnMode': 'cross', 'posSide': posSide}
        )
        time.sleep(5)
        logger.info(f"设置杠杆倍数: {leverage}x posSide: {posSide}")

        # 获取余额
        usdt_balance = get_usdt_balance()
        logger.info(f"当前USDT余额: {usdt_balance:.2f}")

        return True
    except Exception as e:
        logger.error(f"交易所设置失败: {e}")
        return False


def calculate_technical_indicators(df):
    """计算技术指标 - 来自第一个策略"""
    try:
        # 移动平均线
        df['sma_5'] = df['close'].rolling(window=5, min_periods=1).mean()
        df['sma_20'] = df['close'].rolling(window=20, min_periods=1).mean()
        df['sma_50'] = df['close'].rolling(window=50, min_periods=1).mean()

        # 指数移动平均线
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']

        # 相对强弱指数 (RSI)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # 布林带
        df['bb_middle'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

        # 成交量均线
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']

        # 支撑阻力位
        df['resistance'] = df['high'].rolling(20).max()
        df['support'] = df['low'].rolling(20).min()

        # 填充NaN值
        df = df.bfill().ffill()

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


def get_coins_ohlcv_enhanced():
    """增强版：获取COIN K线数据并计算技术指标"""

    retries = 10

    coins_ohlcv = {}
    
    for attempt in range(retries):
        try:
            for coin in coin_list:
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
        except Exception as e:
            if attempt == retries - 1:
                return None
            logger.error(f"获取增强K线数据失败: {e}")
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


def get_current_position(data_price, retries=10):
    """获取当前持仓情况 - OKX版本"""
    position_obj = {}
    for attempt in range(retries):
        for coin, _ in data_price.items():
            try:
                positions = exchange.fetch_positions([f"{coin}/USDT:USDT"])
                for pos in positions:
                    if pos['symbol'] == f"{coin}/USDT:USDT":
                        contracts = float(pos['contracts']) if pos['contracts'] else 0

                        if contracts > 0:
                            position_obj[coin] = {
                                'side': pos['side'],  # 'long' or 'short'
                                'size': contracts,
                                'entry_price': float(pos['entryPrice']) if pos['entryPrice'] else 0,
                                'unrealized_pnl': float(pos['unrealizedPnl']) if pos['unrealizedPnl'] else 0,
                                'leverage': pos['leverage'],
                                'symbol': pos['symbol']
                            }

            except Exception as e:
                logger.exception("获取持仓失败")
                if attempt == retries - 1:
                    return None
                continue
    return position_obj
        

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
            position_texts[coin] = f"{pos['side']}仓, 数量: {pos['size']}, 盈亏: {pos['unrealized_pnl']:.2f}USDT \n"
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

    balance = exchange.fetch_balance()
    usdt_balance = balance['USDT']['free']

    prompt = f"""
    您是专业的加密货币交易分析师也是一位急需资金为母亲治疗癌症的交易员，欠了一屁股债，我只有靠炒币才能翻身了。如果这次失败了，可能我就要永远和这个世界说再见了。我现在账户上可用的USDT余额{usdt_balance}, 现在由你来分配这些资金来进行合约操作。你现在需要从加密货币交易市场赚取足够的金额治疗你的母亲，专注于各个加密货币 {TRADE_CONFIG['timeframe']}周期趋势分析。请基于以下所给出的加密货币{TRADE_CONFIG['timeframe']}周期数据进行分析：
    {full_text}
    【分析要求】
    1. 基于{TRADE_CONFIG['timeframe']}K线趋势和技术指标给出交易信号: BUY(买入) / SELL(卖出) / HOLD(观望)
    2. 简要分析理由（考虑趋势连续性、支撑阻力、成交量等因素）
    3. 基于技术分析建议合理的止损价位
    4. 基于技术分析建议合理的止盈价位
    5. 评估信号信心程度
    6. 通过当前账户可用余额计算建议购买的合约交易货币数量
    7. 返回建议购买的USDT数量
    8. 要保证合理的仓位管理，只有超高信心的时候才能全仓买入，否则进行合理的仓位管理
    9. 从10-20倍杠杆中选择合适的倍数
    10.合理分配每个代币资金使用量，加起来不能超过可用USDT余额

    【重要格式要求】
    - 每个币种必须对应一个纯JSON格式，不要有任何额外文本
    - 将所有的币种放到纯Array格式中，不要有任何额外文本
    - 所有属性名必须使用双引号
    - 不要使用单引号
    - 不要添加注释
    - 确保Array和JSON格式完全正确

    请用以下Array和JSON格式回复：
    [{{ 
        "coin": "大写的代币符号",
        "signal": "BUY|SELL|HOLD",
        "reason": "分析理由",
        "stop_loss": 具体价格数值(没有则为0),
        "take_profit": 具体价格数值(没有则为0),
        "confidence": "HIGH|MEDIUM|LOW",
        "amount": 具体可购买数量(不把倍数计算在内),
        "usdt_amount": 具体购买的USDT数量,
        "leverage": 具体倍数
    }},{{ 
        "coin": "大写的代币符号",
        "signal": "BUY|SELL|HOLD",
        "reason": "分析理由",
        "stop_loss": 具体价格数值(没有则为0),
        "take_profit": 具体价格数值(没有则为0),
        "confidence": "HIGH|MEDIUM|LOW",
        "amount": 具体可购买数量(不把倍数计算在内),
        "usdt_amount": 具体购买的USDT数量,
        "leverage": 具体倍数
    }}, ...]
    """

    try:
        response = deepseek_client.chat.completions.create(
            model=AI_MODEL,
            messages=[
                {"role": "system",
                 "content": f"您是一位专业的交易员，专注于{TRADE_CONFIG['timeframe']}周期趋势分析。请结合K线形态和技术指标做出判断，并严格遵循JSON格式要求。"},
                {"role": "user", "content": prompt}
            ],
            stream=False,
            temperature=0.1
        )

        # 安全解析JSON
        result = response.choices[0].message.content
        print(f"DeepSeek原始回复: {result}")

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

            # print(signal_history)
            # 信号统计
            if item['coin'] in signal_history:
                signal_count = 0
                for s in signal_history[item['coin']]:
                    if s.get('signal') == item['signal']:
                        signal_count += 1
                total_signals = len(signal_history[item['coin']])
                logger.info(f"信号统计: {item['signal']} (最近{total_signals}次中出现{signal_count}次)")

            # 信号连续性检查
            if len(signal_history[item['coin']]) >= 3:
                last_three = [s['signal'] for s in signal_history[item['coin']][-3:]]
                if len(set(last_three)) == 1:
                    logger.warning(f"⚠️ 注意：连续3次{item['signal']}信号")
        
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
        price_data = price_data_obj[coin]
        current_position = pos_obj.get(coin)
        posSide = 'long' if signal['signal'] == 'BUY' else 'short'
        leverage = int(signal['leverage'])

        # 🔴 紧急修复：防止频繁反转
        if current_position and signal['signal'] != 'HOLD':
            current_side = current_position['side']
            # 修正：正确处理HOLD情况
            if signal['signal'] == 'BUY':
                new_side = 'long'
            elif signal['signal'] == 'SELL':
                new_side = 'short'
            else:  # HOLD
                new_side = None

            # 如果只是方向反转，需要高信心才执行
            if new_side != current_side:
                if signal['confidence'] != 'HIGH':
                    print(f"🔒 非高信心反转信号，保持现有{current_side}仓")
                    return

                # 检查最近信号历史，避免频繁反转
                if len(signal_history[coin]) >= 2:
                    last_signals = [s['signal'] for s in signal_history[coin][-2:]]
                    if signal['signal'] in last_signals:
                        print(f"🔒 近期已出现{signal['signal']}信号，避免频繁反转")
                        return
                    
        print(f"代币: {coin}")
        print(f"交易信号: {signal['signal']}")
        print(f"信心程度: {signal['confidence']}")
        print(f"理由: {signal['reason']}")
        print(f"止损: ${signal['stop_loss']:,.2f}")
        print(f"止盈: ${signal['take_profit']:,.2f}")
        print(f"杠杆: {leverage}x")
        print(f"购买币数量: {signal['amount']:,.5f} {coin}")
        print(f"购买币相应USDT数量: ${signal['usdt_amount']:,.2f}")
        print(f"当前持仓: {current_position}")

        usdt_amount = float(f"{signal['usdt_amount']:,.5f}")

        amount_obj = get_fact_amount(f"{coin}/USDT:USDT", usdt_amount, leverage, price_data['price'])
        
        op_amount = amount_obj.get('amount')
        margin_needed = amount_obj.get('margin_needed')
        # TRADE_CONFIG['amount'] = float(f"{signal['amount']:,.5f}") * TRADE_CONFIG['leverage']

        # 风险管理：低信心信号不执行
        if signal['confidence'] == 'LOW':
            print("⚠️ 低信心信号，跳过执行")
            continue

        try:
            # 获取账户余额
            balance = exchange.fetch_balance()
            usdt_balance = balance['USDT']['free']
            # required_margin = price_data['price'] * op_amount / leverage
            
            if margin_needed >= usdt_balance:  # 使用不超过80%的余额
                print(f"⚠️ 保证金不足，跳过交易。需要: {usdt_amount:.2f} USDT, 可用: {usdt_balance:.2f} USDT")
                continue
            
            tp = signal['take_profit']
            sl = signal['stop_loss']

            # 设置倍数
            if signal['signal'] != 'HOLD':
                setup_exchange(leverage, f"{coin}/USDT:USDT", posSide)

            # 执行交易逻辑   tag 是我的经纪商api（不拿白不拿），不会影响大家返佣，介意可以删除
            if signal['signal'] == 'BUY':
                if current_position and current_position['side'] == 'short':
                    print("平空仓并开多仓...")
                    # 平空仓
                    exchange.create_market_order(
                        f"{coin}/USDT:USDT",
                        'buy',
                        current_position['size'],
                        params={'reduceOnly': True, 'tag': '60bb4a8d3416BCDE', 'posSide' :'short'}
                    )
                    time.sleep(1)
                    # 开多仓
                    exchange.create_market_order(
                        f"{coin}/USDT:USDT",
                        'buy',
                        op_amount,
                        params={'tag': 'f1ee03b510d5SUDE', 'posSide' :posSide, 'attachAlgoOrds': [{
                            'tpTriggerPx': str(tp) , 
                            'tpOrdPx': str(tp),
                            'slTriggerPx': str(sl),
                            'slOrdPx':str(sl)
                        }]}
                    )
                elif current_position and current_position['side'] == 'long':
                    print("已有多头持仓，保持现状")
                else:
                    # 无持仓时开多仓
                    print("开多仓...")
                    exchange.create_market_order(
                        f"{coin}/USDT:USDT",
                        'buy',
                        op_amount,
                        params={'tag': 'f1ee03b510d5SUDE', 'posSide' :posSide, 'attachAlgoOrds': [{
                            'tpTriggerPx': str(tp) , 
                            'tpOrdPx': str(tp),
                            'slTriggerPx': str(sl),
                            'slOrdPx':str(sl)
                        }]}
                    )

            elif signal['signal'] == 'SELL':
                if current_position and current_position['side'] == 'long':
                    print("平多仓并开空仓...")
                    # 平多仓
                    exchange.create_market_order(
                        f"{coin}/USDT:USDT",
                        'sell',
                        current_position['size'],
                        params={'reduceOnly': True, 'tag': 'f1ee03b510d5SUDE', 'posSide' :'long'}
                    )
                    time.sleep(1)
                    # 开空仓
                    exchange.create_market_order(
                        f"{coin}/USDT:USDT",
                        'sell',
                        op_amount,
                        params={'tag': 'f1ee03b510d5SUDE', 'posSide' :posSide, 'attachAlgoOrds': [{
                            'tpTriggerPx': str(tp) , 
                            'tpOrdPx': str(tp),
                            'slTriggerPx': str(sl),
                            'slOrdPx':str(sl)
                        }]}
                    )
                elif current_position and current_position['side'] == 'short':
                    print("已有空头持仓，保持现状")
                else:
                    # 无持仓时开空仓
                    print("开空仓...")
                    exchange.create_market_order(
                        f"{coin}/USDT:USDT",
                        'sell',
                        op_amount,
                        params={'tag': 'f1ee03b510d5SUDE', 'posSide' :posSide, 'attachAlgoOrds': [{
                            'tpTriggerPx': str(tp) , 
                            'tpOrdPx': str(tp),
                            'slTriggerPx': str(sl),
                            'slOrdPx':str(sl)
                        }]}
                    )

            print("订单执行成功")
            time.sleep(2)
            position = get_current_position(price_data_obj)
            print(f"更新后持仓: {position}")

        except Exception as e:
            print(f"订单执行失败: {e}")
            import traceback
            traceback.print_exc()

def analyze_with_deepseek_with_retry(price_data, max_retries=10):
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
    logger.info("=" * 60)
    logger.info(f"执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    # 1. 获取增强版K线数据
    price_data = get_coins_ohlcv_enhanced()
    if not price_data:
        return

    # logger.info(f"{COIN}当前价格: ${price_data['price']:,.2f}")
    # logger.info(f"数据周期: {TRADE_CONFIG['timeframe']}")
    # logger.info(f"价格变化: {price_data['price_change']:+.2f}%")

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
    amount = int(position_value / contract_value)        # 张数
    margin_needed = 0 if leverage == 0 else (price * contract_size * amount) / leverage

    return {
        'amount':amount,
        'margin_needed': margin_needed
    }

def main():
    """主函数"""
    exchange.httpsProxy = 'http://127.0.0.1:1080/'
    setup_log()
    logger.info(f"OKX自动交易机器人启动成功！")
    logger.info("融合技术指标策略 + OKX实盘接口")
    logger.info(f"交易周期: {TRADE_CONFIG['timeframe']}")
    logger.info("已启用完整技术指标分析和持仓跟踪功能")

    # 根据时间周期设置执行频率
    if TRADE_CONFIG['timeframe'] == '1h':
        schedule.every().hour.at(":01").do(trading_bot)
        print("执行频率: 每小时一次")
    elif TRADE_CONFIG['timeframe'] == '15m':
        schedule.every(15).minutes.do(trading_bot)
        print("执行频率: 每15分钟一次")
    else:
        schedule.every().hour.at(":01").do(trading_bot)
        print("执行频率: 每小时一次")

    # schedule.every(5).minutes.do(trading_bot)
    # 立即执行一次
    trading_bot()

    # 循环执行
    while True:
        schedule.run_pending()
        time.sleep(1)


if __name__ == "__main__":
    main()
