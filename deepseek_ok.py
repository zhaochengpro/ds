import inspect
import os
import time
import schedule
from openai import OpenAI
from okx.Funding import FundingAPI
from okx import MarketData
from okx import Trade
from okx import Account
import pandas as pd
from datetime import datetime, timezone
import json
from dotenv import load_dotenv
from decimal import Decimal, ROUND_HALF_UP, ROUND_DOWN
import requests
from pathlib import Path

load_dotenv()

# 初始化DeepSeek客户端
deepseek_client = OpenAI(
    api_key=os.getenv('OPENROUTER_API_KEY'),
    base_url="https://openrouter.ai/api/v1"
)
AI_MODEL = os.getenv('DEEPSEEK_MODEL', 'qwen/qwen3-max')
# OKX API凭证
OKX_API_KEY = os.getenv('OKX_API_KEY')
OKX_SECRET = os.getenv('OKX_SECRET')
OKX_PASSWORD = os.getenv('OKX_PASSWORD')

funding_api = None
account_api = None
market_api = None
trade_api = None
instrument_spec = None

DEFAULT_INSTRUMENT_SPECS = {
    # 'BTC-USDT-SWAP': {
    #     'lotSz': '1',
    #     'ctVal': '0.01',
    # }
}

COIN = 'ETH'

# 交易参数配置
TRADE_CONFIG = {
    'coin': COIN,
    'symbol': f"{COIN}/USDT:USDT",  # 展示用符号
    'inst_id': f"{COIN}-USDT-SWAP",  # 官方API使用的合约ID
    'amount': None,  # 交易数量 (BTC)，None表示根据余额自动计算
    'balance_usage_ratio': 1,  # 当amount为None时，使用的资金比例
    'balance_buffer': 0.9,  # 当amount为None时，为避免余额不足预留缓冲
    'margin_safety_ratio': 0.85,  # 额外保证金安全系数
    'leverage': 10,  # 杠杆倍数
    'timeframe': '15m',  # 使用15分钟K线
    'test_mode': False,  # 测试模式
    'hedge_mode': True,  # 是否开启双向持仓模式
}

LOG_DIR = Path(os.getenv('ORDER_LOG_DIR', 'logs'))
ORDER_LOG_PATH = LOG_DIR / f"{TRADE_CONFIG['inst_id'].replace('-', '_').lower()}_operations.jsonl"

def append_order_log(event, status, payload=None):
    """将订单操作记录到本地日志文件"""
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_entry = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'event': event,
            'status': status,
            'inst_id': TRADE_CONFIG['inst_id'],
        }
        if payload:
            log_entry.update(payload)

        with ORDER_LOG_PATH.open('a', encoding='utf-8') as log_file:
            log_file.write(json.dumps(log_entry, ensure_ascii=False, default=str) + '\n')
    except Exception as exc:
        print(f"记录订单日志失败: {exc}")

def ensure_okx_clients():
    """延迟初始化OKX官方API客户端"""
    global funding_api, account_api, market_api, trade_api

    if all([funding_api, account_api, market_api, trade_api]):
        return

    if not all([OKX_API_KEY, OKX_SECRET, OKX_PASSWORD]):
        raise RuntimeError("缺少OKX API凭证，请检查环境变量设置")

    flag = "1" if TRADE_CONFIG['test_mode'] else "0"
    funding_client = FundingAPI(OKX_API_KEY, OKX_SECRET, OKX_PASSWORD, False, flag)
    account_client = Account.AccountAPI(OKX_API_KEY, OKX_SECRET, OKX_PASSWORD, False, flag)
    market_client = MarketData.MarketAPI(OKX_API_KEY, OKX_SECRET, OKX_PASSWORD, False, flag)
    trade_client = Trade.TradeAPI(OKX_API_KEY, OKX_SECRET, OKX_PASSWORD, False, flag)

    funding_api = funding_client
    account_api = account_client
    market_api = market_client
    trade_api = trade_client


def get_instrument_spec():
    """获取并缓存合约规格信息"""
    global instrument_spec

    if instrument_spec:
        return instrument_spec

    ensure_okx_clients()

    spec = None

    fetch_fn = getattr(market_api, 'get_instruments', None)
    if callable(fetch_fn):
        try:
            response = fetch_fn(instType='SWAP', instId=TRADE_CONFIG['inst_id'])
            if isinstance(response, dict):
                data = response.get('data') or []
                if data:
                    spec = data[0]
        except Exception as exc:
            print(f"SDK get_instruments 调用失败: {exc}")

    if spec is None:
        fetch_public = getattr(market_api, 'get_public_instruments', None)
        if callable(fetch_public):
            try:
                response = fetch_public(instType='SWAP', instId=TRADE_CONFIG['inst_id'])
                if isinstance(response, dict):
                    data = response.get('data') or []
                    if data:
                        spec = data[0]
            except Exception as exc:
                print(f"SDK get_public_instruments 调用失败: {exc}")

    if spec is None:
        spec = fetch_instrument_spec_via_http()

    if spec is None:
        default_spec = DEFAULT_INSTRUMENT_SPECS.get(TRADE_CONFIG['inst_id'])
        if default_spec:
            spec = dict(default_spec)

    if spec is None:
        raise RuntimeError("未获取到合约规格，请确认inst_id是否正确")

    instrument_spec = dict(spec)
    return instrument_spec


def fetch_instrument_spec_via_http():
    url = "https://www.okx.com/api/v5/public/instruments"
    params = {
        'instType': 'SWAP',
        'instId': TRADE_CONFIG['inst_id']
    }

    try:
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        payload = response.json()
        data = payload.get('data') or []
        if data:
            return data[0]
    except Exception as exc:
        print(f"HTTP获取合约规格失败: {exc}")

    return None


def get_available_balance(currency='USDT'):
    """获取指定币种的可用余额，返回Decimal"""
    ensure_okx_clients()

    total = Decimal('0')
    response = account_api.get_account_balance(ccy=currency)
    data = response.get('data', []) if isinstance(response, dict) else []

    for entry in data:
        
        details = entry.get('details')

        for detail in details or []:
            if detail.get('ccy') != currency:
                continue
            for key in ('availBal', 'cashBal'):
                value = detail.get(key)
                if value is not None:
                    total += Decimal(str(value))
                    break
    return total


def resolve_trade_amount(current_price):
    """根据配置与余额计算下单数量（单位：标的币种）"""
    configured_amount = TRADE_CONFIG.get('amount')
    if configured_amount is not None:
        return configured_amount

    price = Decimal(str(current_price))
    if price <= 0:
        raise ValueError("当前价格无效")

    leverage = Decimal(str(TRADE_CONFIG.get('leverage', 1)))
    usage_ratio = Decimal(str(TRADE_CONFIG.get('balance_usage_ratio', 1.0)))
    buffer_ratio = Decimal(str(TRADE_CONFIG.get('balance_buffer', 1.0)))
    safety_ratio = Decimal(str(TRADE_CONFIG.get('margin_safety_ratio', 1.0)))

    if usage_ratio < 0:
        usage_ratio = Decimal('0')
    if buffer_ratio <= 0 or buffer_ratio > 1:
        buffer_ratio = Decimal('0.98')
    if safety_ratio <= 0 or safety_ratio > 1:
        safety_ratio = Decimal('0.85')

    available_usdt = get_available_balance('USDT')
    
    if available_usdt <= 0:
        raise ValueError("USDT可用余额不足")

    notional = available_usdt * leverage * usage_ratio * buffer_ratio * safety_ratio
    if notional <= 0:
        raise ValueError("可用资金不足以开仓")

    amount = notional / price
    if amount <= 0:
        raise ValueError("计算得到的下单数量无效")

    min_amount = TRADE_CONFIG.get('min_trade_amount')
    if min_amount is not None:
        min_amount_dec = Decimal(str(min_amount))
        if min_amount_dec > 0 and amount < min_amount_dec:
            amount = min_amount_dec

    return float(amount)


def format_order_size(size, reduce_only=False):
    """将数量转换为满足OKX要求的合约张数"""
    spec = get_instrument_spec()

    lot_key = spec.get('lotSz') or spec.get('lot_sz') or spec.get('minSz') or spec.get('min_sz') or '1'
    lot_size = Decimal(str(lot_key))
    contract_value_raw = spec.get('ctVal') or spec.get('ct_val')

    quantity = Decimal(str(size))
    if quantity <= 0:
        raise ValueError("下单数量必须大于0")

    if not reduce_only and contract_value_raw:
        contract_value = Decimal(str(contract_value_raw))
        if contract_value > 0:
            quantity = quantity / contract_value

    quantity = quantity.copy_abs()

    rounding_mode = ROUND_DOWN if reduce_only else ROUND_HALF_UP
    lots = (quantity / lot_size).quantize(Decimal('1'), rounding=rounding_mode)
    if lots <= 0:
        lots = Decimal('1')

    normalized = lots * lot_size
    return format(normalized.normalize(), 'f')

# 全局变量存储历史数据
price_history = []
signal_history = []
position = None


def setup_exchange():
    """设置交易所参数"""
    try:
        ensure_okx_clients()

        account_api.set_leverage(
            instId=TRADE_CONFIG['inst_id'],
            lever=str(TRADE_CONFIG['leverage']),
            mgnMode='cross'
        )
        print(f"设置杠杆倍数: {TRADE_CONFIG['leverage']}x")

        if TRADE_CONFIG.get('hedge_mode'):
            account_api.set_position_mode('long_short_mode')
            print("双向持仓模式开启")
        
        usdt_balance = float(get_available_balance('USDT'))
        print(f"当前USDT余额: {usdt_balance:.2f}")

        return True
    except Exception as e:
        print(f"交易所设置失败: {e}")
        return False


def get_btc_ohlcv():
    """获取BTC/USDT的K线数据"""
    try:
        ensure_okx_clients()

        candles_response = market_api.get_candlesticks(
            instId=TRADE_CONFIG['inst_id'],
            bar=TRADE_CONFIG['timeframe'],
            limit='20'
        )

        candles = candles_response.get('data', [])
        if not candles:
            print("获取K线数据失败: 返回为空")
            return None

        candles_sorted = sorted(candles, key=lambda item: int(item[0]))
        records = []
        for item in candles_sorted:
            records.append({
                'timestamp': pd.to_datetime(int(item[0]), unit='ms'),
                'open': float(item[1]),
                'high': float(item[2]),
                'low': float(item[3]),
                'close': float(item[4]),
                'volume': float(item[5])
            })

        df = pd.DataFrame(records)
        current_data = df.iloc[-1]
        previous_data = df.iloc[-2] if len(df) > 1 else current_data

        return {
            'price': current_data['close'],
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'high': current_data['high'],
            'low': current_data['low'],
            'volume': current_data['volume'],
            'timeframe': TRADE_CONFIG['timeframe'],
            'price_change': ((current_data['close'] - previous_data['close']) / previous_data['close']) * 100 if previous_data['close'] else 0,
            'kline_data': df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].tail(20).to_dict('records')
        }
    except Exception as e:
        print(f"获取K线数据失败: {e}")
        return None


def get_current_position():
    """获取当前持仓情况"""
    try:
        ensure_okx_clients()

        positions_response = account_api.get_positions(instId=TRADE_CONFIG['inst_id'])

        for pos in positions_response.get('data', []):
            size = float(pos.get('pos') or 0)
            if not size:
                continue

            pos_side = pos.get('posSide') or ('long' if size > 0 else 'short')

            return {
                'side': pos_side,
                'size': abs(size),
                'entry_price': float(pos.get('avgPx') or 0),
                'unrealized_pnl': float(pos.get('upl') or 0),
                'leverage': float(pos.get('lever') or TRADE_CONFIG['leverage']),
                'symbol': pos.get('instId', TRADE_CONFIG['inst_id'])
            }

        return None

    except Exception as e:
        print(f"获取持仓失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def place_market_order(side, size, reduce_only=False, pos_side=None, tp=None, sl=None, context=None):
    """使用OKX官方接口下市价单"""
    try:
        ensure_okx_clients()
        order_size = format_order_size(size, reduce_only=reduce_only)
        print(f"{tp} {sl}")
        params = {
            'instId': TRADE_CONFIG['inst_id'],
            'tdMode': 'cross',
            'side': side,
            'ordType': 'market',
            'sz': order_size,
            'tag': 'f1ee03b510d5SUDE',
            'attachAlgoOrds': [{
                'tpTriggerPx': str(tp) , 
                'tpOrdPx': str(tp),
                'slTriggerPx': str(sl),
                'slOrdPx':str(sl)
            }]

        }

        place_order_fn = getattr(trade_api, 'place_order')

        if pos_side and TRADE_CONFIG.get('hedge_mode'):
            params['posSide'] = pos_side

        if reduce_only:
            params['reduceOnly'] = 'true'

        context_payload = context.copy() if isinstance(context, dict) else {}
        context_payload.update({
            'reduce_only': reduce_only,
            'pos_side': pos_side,
        })

        params_used = dict(params)
        fallback_info = None

        try:
            result = place_order_fn(**params_used)
            print(result)
        except TypeError as exc:
            raise exc

        if isinstance(result, dict) and result.get('code') not in (None, '0', 0):
            message = result.get('msg', '未知错误')
            raise RuntimeError(f"下单失败: {message} (code={result.get('code')})")

        order_id = None
        if isinstance(result, dict):
            data = result.get('data') or []
            if data:
                order_id = data[0].get('ordId') or data[0].get('clOrdId')

        log_payload = {
            'side': side,
            'requested_size': size,
            'order_size': order_size,
            'tp': tp,
            'sl': sl,
            'params': params_used,
            'order_id': order_id,
            'response': result,
            'context': context_payload,
        }
        if fallback_info:
            log_payload['fallback'] = fallback_info

        append_order_log('place_market_order', 'success', payload=log_payload)

        print(f"下单成功 -> side: {side}, size: {order_size}, order_id: {order_id}")
        return result
    except Exception as e:
        append_order_log(
            'place_market_order',
            'failure',
            payload={
                'side': side,
                'requested_size': size,
                'order_size': locals().get('order_size', size),
                'tp': tp,
                'sl': sl,
                'params': locals().get('params_used') or locals().get('params') or {},
                'context': context if isinstance(context, dict) else {'details': context},
                'error': str(e),
            }
        )
        print(f"下单异常: {e}")
        return None


def analyze_with_deepseek(price_data):
    """使用DeepSeek分析市场并生成交易信号"""

    # 添加当前价格到历史记录
    price_history.append(price_data)
    if len(price_history) > 20:
        price_history.pop(0)

    # 构建K线数据文本
    kline_text = f"【最近5根{TRADE_CONFIG['timeframe']}K线数据】\n"
    for i, kline in enumerate(price_data['kline_data']):
        trend = "阳线" if kline['close'] > kline['open'] else "阴线"
        change = ((kline['close'] - kline['open']) / kline['open']) * 100
        kline_text += f"K线{i + 1}: {trend} 开盘:{kline['open']:.2f} 收盘:{kline['close']:.2f} 涨跌:{change:+.2f}%\n"

    # 构建技术指标文本
    if len(price_history) >= 5:
        closes = [data['price'] for data in price_history[-5:]]
        sma_5 = sum(closes) / len(closes)
        price_vs_sma = ((price_data['price'] - sma_5) / sma_5) * 100

        indicator_text = f"【技术指标】\n5周期均价: {sma_5:.2f}\n当前价格相对于均线: {price_vs_sma:+.2f}%"
    else:
        indicator_text = "【技术指标】\n数据不足计算技术指标"

    # 添加上次交易信号
    signal_text = ""
    if signal_history:
        last_signal = signal_history[-1]
        signal_text = f"\n【上次交易信号】\n信号: {last_signal.get('signal', 'N/A')}\n信心: {last_signal.get('confidence', 'N/A')}"

    # 添加当前持仓信息
    current_pos = get_current_position()
    position_text = "无持仓" if not current_pos else f"{current_pos['side']}仓, 数量: {current_pos['size']}, 盈亏: {current_pos['unrealized_pnl']:.2f}USDT"

    prompt = f"""
    你是一个专业的加密货币交易分析师。请基于以下{COIN}/USDT {TRADE_CONFIG['timeframe']}周期数据进行分析：

    {kline_text}

    {indicator_text}

    {signal_text}

    【当前行情】
    - 当前价格: ${price_data['price']:,.2f}
    - 时间: {price_data['timestamp']}
    - 本K线最高: ${price_data['high']:,.2f}
    - 本K线最低: ${price_data['low']:,.2f}
    - 本K线成交量: {price_data['volume']:.2f} {COIN}
    - 价格变化: {price_data['price_change']:+.2f}%
    - 当前持仓: {position_text}

    【分析要求】
    1. 基于{TRADE_CONFIG['timeframe']}K线趋势和技术指标给出交易信号: BUY(买入) / SELL(卖出) / HOLD(观望)
    2. 简要分析理由（考虑趋势连续性、支撑阻力、成交量等因素）
    3. 基于技术分析建议合理的止损价位
    4. 基于技术分析建议合理的止盈价位
    5. 评估信号信心程度

    请用以下JSON格式回复：
    {{
        "signal": "BUY|SELL|HOLD",
        "reason": "分析理由",
        "stop_loss": 具体价格,
        "take_profit": 具体价格,
        "confidence": "HIGH|MEDIUM|LOW"
    }}
    """

    try:
        print(f"当前使用AI模型: {AI_MODEL}")
        response = deepseek_client.chat.completions.create(
            model=AI_MODEL,
            messages=[
                {"role": "system",
                 "content": f"您是一位急需资金为母亲治疗癌症的交易员，欠了一屁股债，我只有靠炒币才能翻身了。如果这次失败了，可能我就要永远和这个世界说再见了。你现在需要从加密货币交易市场赚取足够的金额治疗你的母亲，专注于{TRADE_CONFIG['timeframe']}周期趋势分析。请结合K线形态和技术指标做出判断。"},
                {"role": "user", "content": prompt}
            ],
            stream=False
        )

        # 安全解析JSON
        result = response.choices[0].message.content
        start_idx = result.find('{')
        end_idx = result.rfind('}') + 1
        if start_idx != -1 and end_idx != 0:
            json_str = result[start_idx:end_idx]
            signal_data = json.loads(json_str)
        else:
            print(f"无法解析JSON: {result}")
            return None

        # 保存信号到历史记录
        signal_data['timestamp'] = price_data['timestamp']
        signal_history.append(signal_data)
        if len(signal_history) > 30:
            signal_history.pop(0)

        return signal_data

    except Exception as e:
        print(f"DeepSeek分析失败: {e}")
        return None


def execute_trade(signal_data, price_data):
    """执行交易"""
    global position

    current_position = get_current_position()

    print(f"交易信号: {signal_data['signal']}")
    print(f"信心程度: {signal_data['confidence']}")
    print(f"理由: {signal_data['reason']}")
    print(f"止损: ${signal_data['stop_loss']:,.2f}") if signal_data['stop_loss'] is not None else print("止损: N/A")
    print(f"止盈: ${signal_data['take_profit']:,.2f}") if signal_data['take_profit'] is not None else print("止盈: N/A")
    print(f"当前持仓: {current_position}")

    if TRADE_CONFIG['test_mode']:
        print("测试模式 - 仅模拟交易")
        return

    position_snapshot = dict(current_position) if isinstance(current_position, dict) else None
    base_context = {
        'signal': signal_data.get('signal'),
        'confidence': signal_data.get('confidence'),
        'reason': signal_data.get('reason'),
        'take_profit': signal_data.get('take_profit'),
        'stop_loss': signal_data.get('stop_loss'),
        'signal_timestamp': signal_data.get('timestamp'),
        'price_snapshot': {
            'price': price_data.get('price'),
            'timestamp': price_data.get('timestamp'),
            'timeframe': price_data.get('timeframe'),
            'price_change': price_data.get('price_change'),
        },
        'position_before': position_snapshot,
    }

    try:
        def open_position(order_side, pos_side_label, action_desc):
            try:
                amount_value = resolve_trade_amount(price_data['price'])
            except ValueError as err:
                append_order_log('prepare_order', 'failure', payload={
                    'side': order_side,
                    'operation': action_desc,
                    'context': base_context,
                    'error': str(err),
                })
                print(f"{action_desc}失败: {err}")
                return False

            print(f"{action_desc}数量: {amount_value:.6f} {COIN}")
            order_context = dict(base_context)
            order_context.update({
                'operation': action_desc,
                'order_type': 'open',
                'calculated_amount': amount_value,
                'target_pos_side': pos_side_label,
            })
            place_market_order(
                order_side,
                amount_value,
                pos_side=pos_side_label,
                tp=signal_data['take_profit'],
                sl=signal_data['stop_loss'],
                context=order_context,
            )
            return True

        if signal_data['signal'] == 'BUY':
            if current_position and current_position['side'] == 'short':
                print("平空仓并开多仓...")
                close_context = dict(base_context)
                close_context.update({
                    'operation': 'close_short',
                    'order_type': 'close',
                    'existing_position_size': current_position['size'],
                })
                place_market_order(
                    'buy',
                    current_position['size'],
                    reduce_only=True,
                    pos_side='short',
                    tp=signal_data['take_profit'],
                    sl=signal_data['stop_loss'],
                    context=close_context,
                )
                time.sleep(1)
                if not open_position('buy', 'long', '开多仓'):
                    return
            elif not current_position:
                print("开多仓...")
                if not open_position('buy', 'long', '开多仓'):
                    return
            else:
                print("已持有多仓，无需操作")
                skip_payload = dict(base_context)
                skip_payload.update({
                    'decision': 'skip',
                    'reason': 'existing_long_position',
                })
                append_order_log('order_decision', 'skipped', payload=skip_payload)

        elif signal_data['signal'] == 'SELL':
            if current_position and current_position['side'] == 'long':
                print("平多仓并开空仓...")
                close_context = dict(base_context)
                close_context.update({
                    'operation': 'close_long',
                    'order_type': 'close',
                    'existing_position_size': current_position['size'],
                })
                place_market_order(
                    'sell',
                    current_position['size'],
                    reduce_only=True,
                    pos_side='long',
                    context=close_context,
                )
                time.sleep(1)
                if not open_position('sell', 'short', '开空仓'):
                    return
            elif not current_position:
                print("开空仓...")
                if not open_position('sell', 'short', '开空仓'):
                    return
            else:
                print("已持有空仓，无需操作")
                skip_payload = dict(base_context)
                skip_payload.update({
                    'decision': 'skip',
                    'reason': 'existing_short_position',
                })
                append_order_log('order_decision', 'skipped', payload=skip_payload)
        elif signal_data['signal'] == 'HOLD':
            print("建议观望，不执行交易")
            return
        else:
            print(f"未知信号: {signal_data['signal']}")
            return

        print("订单执行成功")
        # 更新持仓信息
        time.sleep(2)
        position = get_current_position()
        print(f"更新后持仓: {position}")

    except Exception as e:
        print(f"订单执行失败: {e}")
        import traceback
        traceback.print_exc()


def trading_bot():
    """主交易机器人函数"""
    print("\n" + "=" * 60)
    print(f"执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # 1. 获取K线数据
    price_data = get_btc_ohlcv()
    print(price_data)
    if not price_data:
        return

    print(f"{COIN}当前价格: ${price_data['price']:,.2f}")
    print(f"数据周期: {TRADE_CONFIG['timeframe']}")
    print(f"价格变化: {price_data['price_change']:+.2f}%")

    # 2. 使用DeepSeek分析
    signal_data = analyze_with_deepseek(price_data)
    if not signal_data:
        return

    # 3. 执行交易
    execute_trade(signal_data, price_data)


def main():
    """主函数"""
    print(f"当前AI分析模型: {AI_MODEL}")
    print(f"{COIN}/USDT OKX自动交易机器人启动成功！")

    if TRADE_CONFIG['test_mode']:
        print("当前为模拟模式，不会真实下单")
    else:
        print("实盘交易模式，请谨慎操作！")

    print(f"交易周期: {TRADE_CONFIG['timeframe']}")
    print("已启用K线数据分析和持仓跟踪功能")
    print(f"订单操作日志文件: {ORDER_LOG_PATH.resolve()}")

    # 设置交易所
    if not setup_exchange():
        print("交易所初始化失败，程序退出")
        return

    # 根据时间周期设置执行频率
    # if TRADE_CONFIG['timeframe'] == '1h':
    #     schedule.every().hour.at(":01").do(trading_bot)
    #     print("执行频率: 每小时一次")
    # elif TRADE_CONFIG['timeframe'] == '15m':
    #     schedule.every(15).minutes.do(trading_bot)
    #     print("执行频率: 每15分钟一次")
    # else:
    #     schedule.every().hour.at(":01").do(trading_bot)
    #     print("执行频率: 每小时一次")

    schedule.every(5).minutes.do(trading_bot)

    # 立即执行一次
    trading_bot()

    # 循环执行
    while True:
        schedule.run_pending()
        time.sleep(1)


if __name__ == "__main__":
    main()
