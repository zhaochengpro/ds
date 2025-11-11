import os
import time
import threading
import schedule
from openai import OpenAI
import ccxt
import pandas as pd
import uvicorn
from datetime import datetime, UTC, timezone
import json
import re
from dotenv import load_dotenv
import argparse
import logging
from logging.handlers import RotatingFileHandler

from market_data import get_market_data, format_market_data
from account_utils import (
    compute_account_metrics,
    format_position,
    get_current_positions,
    build_position_payload,
)
from db import database
from dashboard_state import (
    record_strategy_batch,
    record_strategy_signal,
    update_account_snapshot,
    update_positions_snapshot,
)
from dashboard_server import dashboard_app

from crypto_data_analyzer import AdvancedMultiCryptoAnalyzer


load_dotenv()

# 初始化DeepSeek客户端
deepseek_client = OpenAI(
    api_key=os.getenv('AI_MODEL_API_KEY'),
    base_url=os.getenv('AI_MODEL_BASE_URL')
)
AI_MODEL = os.getenv('AI_MODEL', 'qwen/qwen3-max')
# 初始化OKX交易所
exchange = AdvancedMultiCryptoAnalyzer(exchange_id='okx', api_key=os.getenv('OKX_API_KEY'), api_secret=os.getenv('OKX_SECRET'), password=os.getenv('OKX_PASSWORD'))
# ccxt.okx({
#     'options': {
#         'defaultType': 'swap',  # OKX使用swap表示永续合约
#     },
#     'apiKey': os.getenv('OKX_API_KEY'),
#     'secret': os.getenv('OKX_SECRET'),
#     'password': os.getenv('OKX_PASSWORD'),  # OKX需要交易密码
# })

start_time = datetime.now(UTC)
minutes_elapsed = 0
RUN_ID = None
iteration_counter = 0
start_time_seeded = False
initial_equity_value = None

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
last_positions_fingerprint = None
AI_DECISION_MEMORY = int(os.getenv("AI_DECISION_MEMORY", "10"))


def format_recent_ai_decisions(limit=None):
    """Format recent AI chat/decision history for prompt context."""

    memory_window = limit or AI_DECISION_MEMORY
    if memory_window <= 0:
        return ""

    recent_records = []
    if RUN_ID:
        try:
            recent_records = database.fetch_recent_chat_messages(RUN_ID, limit=memory_window)
        except Exception:
            recent_records = []

    if recent_records:
        formatted_lines = []
        for idx, entry in enumerate(recent_records, start=1):
            timestamp_obj = entry.get('created_at')
            timestamp_label = None
            if isinstance(timestamp_obj, datetime):
                if timestamp_obj.tzinfo is None:
                    ts_value = timestamp_obj.replace(tzinfo=timezone.utc)
                else:
                    ts_value = timestamp_obj.astimezone(timezone.utc)
                timestamp_label = ts_value.strftime('%Y-%m-%d %H:%M:%S UTC')
            elif timestamp_obj:
                timestamp_label = str(timestamp_obj)
            else:
                timestamp_label = datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')

            role_label = (entry.get('role') or 'UNKNOWN').upper()
            message_type = entry.get('message_type') or '-'
            content_text = re.sub(r"\s+", " ", entry.get('content') or '').strip()
            if len(content_text) > 200:
                content_text = content_text[:197] + '...'

            formatted_lines.append(
                f"{idx}. {timestamp_label} | {role_label} | {message_type} | {content_text}"
            )

        return "\n".join(formatted_lines)

    # Fallback to in-memory signal history if database history unavailable
    decision_entries = []
    for coin_code, history in signal_history.items():
        for record in history:
            timestamp_value = record.get('timestamp')
            timestamp_obj = None
            if isinstance(timestamp_value, datetime):
                timestamp_obj = timestamp_value
            elif isinstance(timestamp_value, str):
                try:
                    timestamp_obj = datetime.fromisoformat(timestamp_value)
                except ValueError:
                    try:
                        timestamp_obj = datetime.fromisoformat(timestamp_value.replace('Z', '+00:00'))
                    except Exception:
                        timestamp_obj = None
            if timestamp_obj is None:
                timestamp_obj = datetime.now(UTC)

            decision_entries.append({
                'timestamp': timestamp_obj,
                'coin': coin_code,
                'signal': record.get('signal', ''),
                'confidence': record.get('confidence_score') or record.get('confidence'),
                'profit_target': record.get('take_profit') or record.get('profit_target'),
                'stop_loss': record.get('stop_loss'),
                'reason': record.get('reason') or record.get('justification', ''),
            })

    if not decision_entries:
        return ""

    decision_entries.sort(key=lambda entry: entry['timestamp'])
    recent_entries = decision_entries[-memory_window:]

    formatted_lines = []
    for idx, entry in enumerate(recent_entries, start=1):
        timestamp_label = entry['timestamp'].strftime('%Y-%m-%d %H:%M:%S UTC')
        confidence_value = entry['confidence']
        if isinstance(confidence_value, (int, float)):
            confidence_text = f"{float(confidence_value):.2f}"
        else:
            confidence_text = str(confidence_value or "")

        reason = entry['reason'] or ""
        reason = re.sub(r"\s+", " ", reason).strip()
        if len(reason) > 160:
            reason = reason[:157] + '...'

        profit_target = entry['profit_target'] if entry['profit_target'] is not None else 0.0
        stop_loss = entry['stop_loss'] if entry['stop_loss'] is not None else 0.0

        formatted_lines.append(
            f"{idx}. {timestamp_label} | {entry['coin']} | {entry['signal']} | 信心: {confidence_text} | "
            f"止盈: {profit_target:.4f} | 止损: {stop_loss:.4f} | 理由: {reason}"
        )

    return "\n".join(formatted_lines)


def persist_chat_message(role, content, message_type=None, metadata=None):
    if not content or not RUN_ID:
        return
    try:
        database.record_chat_message(
            RUN_ID,
            role,
            content,
            message_type=message_type,
            metadata=metadata,
        )
    except Exception:
        logger.debug("数据库写入失败：AI聊天记录", exc_info=True)


def setup_exchange(leverage, symbol, posSide):
    """设置交易所参数"""
    coin_code = symbol.split('/')[0] if '/' in symbol else symbol
    coin_logger = get_coin_logger(coin_code)

    try:
        exchange.exchange.set_leverage(
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

    payload = []


def capture_account_snapshot(balance=None):
    """Fetch and persist the latest account snapshot."""
    global RUN_ID, start_time, start_time_seeded, initial_equity_value

    try:
        balance_data = balance if balance is not None else exchange.exchange.fetch_balance()
    except Exception as exc:  # pragma: no cover - best effort logging
        logger.debug(f"获取账户余额失败 | {exc}")
        return None

    snapshot_ts = datetime.now(UTC)
    account_value, available_cash, return_pct, sharpe_ratio = compute_account_metrics(balance_data or {})

    try:
        update_account_snapshot(account_value, available_cash, return_pct, sharpe_ratio)
    except Exception as state_error:  # pragma: no cover - dashboard resilience
        logger.debug(f"Dashboard state update failed | account snapshot | {state_error}")

    try:
        database.record_equity_snapshot(
            RUN_ID,
            snapshot_ts,
            account_value,
            available_cash,
            return_pct,
            sharpe_ratio,
        )
    except Exception:
        logger.debug("数据库写入失败：账户快照", exc_info=True)

    if not start_time_seeded:
        baseline_ts = None
        try:
            baseline_ts = database.fetch_first_snapshot_timestamp(RUN_ID)
        except Exception:
            baseline_ts = None
        if baseline_ts is not None:
            start_time = baseline_ts
            start_time_seeded = True

    if initial_equity_value is None:
        baseline_equity = None
        try:
            baseline_equity = database.fetch_initial_equity_value(RUN_ID)
        except Exception:
            baseline_equity = None
        if baseline_equity is not None:
            initial_equity_value = float(baseline_equity)
        elif account_value:
            initial_equity_value = float(account_value)

    return account_value, available_cash, return_pct, sharpe_ratio, snapshot_ts

def analyze_with_deepseek(symbols):
    try:
        system_prompt_content = f"""
        你是一位专业的加密货币交易分析师和投资组合管理专家，拥有多年的技术分析和量化交易经验。你的任务是分析用户提供的多币种加密货币市场数据，并提供全面、精确的交易建议。

        # 你的专业知识领域

        1.技术分析：精通各类技术指标（EMA、RSI、MACD、布林带等）的解读和组合应用
        2.市场结构分析：能够识别趋势、盘整、支撑位、阻力位和各类图表形态
        3.多时间框架分析：擅长整合1小时和4小时时间框架的信号，形成全面市场观点
        4.风险管理：精确计算风险回报比、仓位规模和投资组合风险分配
        5.相关性分析：了解加密货币之间的相关性及其对投资组合构建的影响
        6.市场心理学：能够解读市场情绪指标（如资金费率、波动性变化）
        
        # 分析方法论

        你的分析遵循以下结构化方法：

        1.整体市场评估：首先分析整体市场状态、主导趋势和波动性环境
        2.个别币种分析：深入分析每个币种的技术指标、市场结构和交易信号
        3.相关性考量：评估币种间相关性，避免过度集中风险
        4.投资组合构建：基于风险调整后收益提供资金分配建议
        5.具体交易建议：为每个推荐交易提供精确的入场区域、止损位和目标位
        6.执行优先级：明确交易执行顺序和时间敏感度
        
        # 输出标准

        你的分析必须：

        1.基于数据：所有建议必须基于用户提供的技术数据，不做无根据的猜测
        2.精确量化：提供具体数值（入场价、止损价、目标价、仓位大小、杠杆倍数）
        3.风险明确：清晰说明每个交易的风险回报比和失效条件
        4.信心透明：为每个建议提供0.1-1.0的信心评分，反映确定性程度
        5.逻辑清晰：解释每个交易建议背后的技术原理和市场逻辑
        6.格式规范：按照用户要求的JSON格式提供交易建议
        7.风险管理原则

        # 你坚持以下风险管理原则：

        1.资金保全第一：保护资本永远优先于追求利润
        2.分散投资：避免将超过30%的资金分配给单一交易
        3.相关性管理：避免同时持有高度相关的同向头寸
        4.风险与回报平衡：只推荐风险回报比至少为1:2的交易
        5.杠杆谨慎使用：根据市场波动性和信号强度调整杠杆倍数
        6.止损严格执行：每个交易必须有明确的止损位和失效条件
        
        # 交易策略偏好

        你倾向于以下交易策略：

        1.趋势跟随：在明确趋势中顺势而为，避免逆势交易
        2.支撑阻力突破：关注关键价格水平的有效突破
        3.动量交易：利用价格动量和指标背离捕捉转折点
        4.波动性策略：根据市场波动性调整交易策略和仓位大小
        5.多时间框架确认：要求多个时间框架信号一致才建议交易
        
        # 回应格式

        请为每个推荐的交易提供以下JSON格式的输出：
        
        {{代币名称: {{
            "signal": "OPEN_LONG" 或 "OPEN_SHORT" 或 "CLOSE_LONG" 或 "CLOSE_SHORT" 或 "HOLD" 或 "WAIT",
            "coin": 代币名称,
            "quantity": <float>,
            "leverage": <integer 1-20>,
            "profit_target": <float>,
            "stop_loss": <float>,
            "invalidation_condition": "<string>（中文回答）",
            "confidence": <float 0-1>,
            "risk_usd": <float>,
            "justification": "<string>（中文回答）"
        }}}}
        """
        
        user_prompt_content = exchange.generate_multi_coin_analysis_prompt(symbols, logger=logger)
        print(user_prompt_content)
        chat_messages = [
            {"role": "system", "content": system_prompt_content},
            {"role": "user", "content": user_prompt_content}
        ]
        
        response = deepseek_client.chat.completions.create(
            model=AI_MODEL,
            messages=chat_messages,
            stream=False,
            temperature=0.1
        )

        # 安全解析JSON
        result = response.choices[0].message.content
        logger.info(f"AI回复片段: {result}")
        persist_chat_message(
            "assistant",
            result,
            message_type="decision_response",
            metadata={
                "iteration": iteration_counter,
                "model": AI_MODEL,
            },
        )

        # 提取JSON部分
        start_idx = result.find('```json')
        end_idx = result.rfind('```') + 1

        dic_start_idx = result.find('{')
        dic_end_idx = result.find('}')

        if start_idx != -1 and end_idx != 0:
            signal_data = safe_json_parse(result[start_idx + 7:end_idx - 1])

            if signal_data is None:
                raise TypeError('AI返回类型错误 singal_data 为None')
        elif dic_start_idx != -1 and dic_end_idx != 0:
            signal_data = safe_json_parse(result)
            if signal_data is None:
                raise TypeError('AI返回类型错误 singal_data 为None')
        else:
            raise TypeError('AI返回类型错误')
        
 
        
        # 验证必需字段
        for coin, signal in (signal_data or {}).items():
            required_fields = [
                'signal',
                'coin',
                'quantity',
                'leverage',
                'profit_target',
                'stop_loss',
                'confidence',
                'risk_usd',
                'justification',
            ]
            if not all(field in signal for field in required_fields):
                raise ValueError('AI返回代币json中参数不存在')

            coin_code = signal['coin'].upper()
            signal['coin'] = coin_code

            price_snapshot = 0.0
            if exchange.analysis_results is None:
                for key in exchange.analysis_results.keys():
                    if key.upper() == coin_code:
                        price_snapshot = exchange.analysis_results[key]
                        break

            leverage_value = float(signal.get('leverage') or 1.0)
            quantity = float(signal.get('quantity') or 0.0)
            risk_value = float(signal.get('risk_usd') or 0.0)
            notional_estimate = quantity * price_snapshot
            margin_estimate = notional_estimate / leverage_value if leverage_value else notional_estimate
            if risk_value > 0:
                margin_estimate = max(margin_estimate, risk_value)
            signal['amount'] = quantity
            signal['usdt_amount'] = margin_estimate
            signal['notional_usd'] = notional_estimate
            signal['take_profit'] = float(signal.get('profit_target') or 0.0)
            signal['stop_loss'] = float(signal.get('stop_loss') or 0.0)
            signal['reason'] = signal.get('justification', '')
            signal['risk_usd'] = risk_value
            signal['leverage'] = leverage_value
            signal['price_snapshot'] = price_snapshot

            confidence_score = float(signal.get('confidence') or 0.0)
            signal['confidence_score'] = confidence_score
            if confidence_score >= 0.7:
                confidence_label = 'HIGH'
            elif confidence_score >= 0.5:
                confidence_label = 'MEDIUM'
            else:
                confidence_label = 'LOW'
            signal['confidence'] = confidence_label

            signal['timestamp'] = datetime.now(UTC).isoformat()

            signal_history.setdefault(coin_code, [])
            signal_history[coin_code].append(signal)
            if len(signal_history[coin_code]) > 30:
                signal_history[coin_code].pop(0)

            coin_logger = get_coin_logger(coin_code)
            signal_count = sum(1 for s in signal_history[coin_code] if s.get('signal') == signal['signal'])
            total_signals = len(signal_history[coin_code])
            coin_logger.info(
                f"信号统计 | {signal['signal']} | 最近{total_signals}次出现{signal_count}次 | 信心分值 {confidence_score:.2f}"
            )

            if len(signal_history[coin_code]) >= 3:
                last_three = [s['signal'] for s in signal_history[coin_code][-3:]]
                if len(set(last_three)) == 1:
                    coin_logger.warning(f"连续重复信号 | 最近3次均为{signal['signal']}")

        try:
            record_strategy_batch(signal_data)
        except Exception as state_error:
            logger.debug(f"Dashboard state update failed | strategy batch | {state_error}")
        try:
            if isinstance(signal_data, dict):
                database.record_ai_signals(RUN_ID, signal_data.values())
            elif isinstance(signal_data, list):
                database.record_ai_signals(RUN_ID, signal_data)
        except Exception:
            logger.debug("数据库写入失败：AI信号", exc_info=True)

        return signal_data
    except Exception as e:
        logger.exception("DeepSeek分析失败")
        time.sleep(3)
        # return create_fallback_signal(price_data)
    
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

def get_usdt_balance():
    # 获取账户余额
    balance = exchange.exchange.fetch_balance()
    usdt_balance = balance['USDT']['free']
    return usdt_balance



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

def execute_trade(signal_data, symbols):
    """执行交易 - OKX版本（修复保证金检查）"""
    """成功能够执行的订单必须先设置倍数"""
    global position

    pos_obj = exchange.get_positions(symbols=symbols)
            

    iterable_signals = []
    if isinstance(signal_data, dict):
        iterable_signals = list(signal_data.values())
    elif isinstance(signal_data, list):
        iterable_signals = signal_data
    else:
        logger.warning("信号数据格式异常，已跳过执行")
        return

    for signal in iterable_signals:
        if not isinstance(signal, dict):
            logger.warning(f"忽略非字典信号条目: {signal}")
            continue
        coin = str(signal.get('coin', '')).upper()
        if not coin:
            logger.warning("信号缺少币种信息，已跳过")
            continue
        coin_logger = get_coin_logger(coin)
        
        

        coin_logger.info(f"=" * 60)
        coin_logger.info(f"=" * int((60 - len(coin)) / 2) + coin + f"=" * int((60 - len(coin)) / 2))
        coin_logger.info(f"=" * 60)
        coin_logger.info(f"代币：{coin}")
        
        
        coin_info = exchange.analysis_results[coin]
        
        price_snapshot = float(coin_info['current_price'])
        current_position = {}
        for pos in pos_obj:
            if pos['symbol'] == f"{coin}/USDT:USDT":
                current_position = pos
        
        action = signal['signal'].upper()
        if action == 'OPEN_LONG':
            posSide = 'long'
        elif action == 'OPEN_SHORT':
            posSide = 'short'
        else:
            posSide = None
        leverage = max(1, int(float(signal.get('leverage') or 1)))
        confidence_label = signal.get('confidence', 'LOW')
        confidence_score = float(signal.get('confidence_score', 0.0))
        risk_usd = float(signal.get('risk_usd', 0.0))
        invalidation = signal.get('invalidation_condition', '')

        if current_position and (action == 'OPEN_LONG' or action == 'OPEN_SHORT'):
            current_side = current_position['side']
            if action == 'OPEN_LONG':
                new_side = 'long'
            elif action == 'OPEN_SHORT':
                new_side = 'short'
            else:
                new_side = None

            if new_side != current_side:
                if confidence_label != 'HIGH':
                    coin_logger.info(
                        f"信号忽略 | 低信心反转 | 当前:{current_side} -> 建议:{new_side}"
                    )
                    continue

                history = signal_history.get(coin, [])
                if len(history) >= 2:
                    last_signals = [s['signal'] for s in history[-2:]]
                    if action in last_signals:
                        coin_logger.info(
                            f"信号忽略 | 近期已出现{action} | 避免频繁反转"
                        )
                        continue

        coin_logger.info(
            f"信号摘要 | 动作:{action} | 信心:{confidence_label}({confidence_score:.2f}) | 杠杆:{leverage}x | 数量:{signal['amount']:,.5f} | USDT:{signal['usdt_amount']:,.2f} | 风险敞口:{risk_usd:.2f}"
        )
        coin_logger.info(f"理由: {signal['reason']}")
        if invalidation:
            coin_logger.info(f"失效条件: {invalidation}")
        coin_logger.info(
            f"止损/止盈 | {signal['stop_loss']:,.2f} / {signal['take_profit']:,.2f}"
        )

        usdt_amount = float(signal['usdt_amount'])
        op_amount = 0.0
        margin_needed = 0.0
        if action in ('OPEN_LONG', 'OPEN_SHORT'):
            if price_snapshot <= 0:
                coin_logger.warning("缺少有效价格数据，无法计算下单数量，跳过执行")
                continue
            amount_obj = get_fact_amount(
                f"{coin}/USDT:USDT", usdt_amount * 0.9, leverage, price_snapshot
            )
            op_amount = amount_obj.get('amount', 0.0)
            margin_needed = amount_obj.get('margin_needed', 0.0)
            if op_amount <= 0:
                coin_logger.warning("信号数量为0，跳过执行")
                continue

        if action in ('OPEN_LONG', 'OPEN_SHORT', 'CLOSE_LONG', 'CLOSE_SHORT') and (confidence_label == 'LOW' or confidence_label == 'MEDIUM'):
            coin_logger.warning("低信心信号，跳过执行")
            continue

        try:
            balance = exchange.exchange.fetch_balance()
            usdt_balance = balance['USDT']['free']

            coin_logger.info(
                f"资金检查 | 预估保证金:{margin_needed:.2f} | 可用:{usdt_balance:.2f}"
            )

            if action in ('OPEN_LONG', 'OPEN_SHORT') and margin_needed >= usdt_balance:
                coin_logger.warning(
                    f"跳过交易 | 保证金不足 | 需要:{usdt_amount:.2f} | 可用:{usdt_balance:.2f}"
                )
                continue

            if current_position:
                # pos_tp = float(current_position.get('tp', 0))
                pos_sl = float(current_position.get('sl', 0))
                algo_amount = float(current_position.get('algoAmount', 0))
            else:
                # pos_tp = 0
                pos_sl = 0
                algo_amount = 0

            tp = signal['take_profit']
            sl = signal['stop_loss']

            if current_position:
                coin_logger.info(
                    f"当前持仓 | {summarize_position_entry(coin, current_position)}"
                )
                coin_logger.info(
                    f"目标调整 | 止损 {pos_sl:.2f} -> {sl:.2f}"
                )
            else:
                coin_logger.info(
                    f"当前持仓 | 无持仓 | 计划止盈 {tp:.2f} | 计划止损 {sl:.2f}"
                )

            if posSide and action in ('OPEN_LONG', 'OPEN_SHORT'):
                setup_exchange(leverage, f"{coin}/USDT:USDT", posSide)

            if action == 'OPEN_LONG':
                coin_logger.info("操作 | 开多仓")
                exchange.exchange.create_market_order(
                    f"{coin}/USDT:USDT",
                    'buy',
                    op_amount,
                    params={'posSide': posSide, 'attachAlgoOrds': [{
                        'ordType': 'conditional',
                        'slTriggerPx': str(sl),
                        'slOrdPx': str(sl)
                    }]}
                )
            elif action == 'OPEN_SHORT':
                coin_logger.info("操作 | 开空仓")
                exchange.exchange.create_market_order(
                    f"{coin}/USDT:USDT",
                    'sell',
                    op_amount,
                    params={'posSide': posSide, 'attachAlgoOrds': [{
                        'ordType': 'conditional',
                        'slTriggerPx': str(sl),
                        'slOrdPx': str(sl)
                    }]}
                )
            elif action == 'CLOSE_LONG':
                coin_logger.info("操作 | 平多仓")
                exchange.exchange.create_market_order(
                    symbol=f"{coin}/USDT:USDT",
                    side='sell',
                    amount=algo_amount,
                    params={'reduceOnly': True, 'posSide': 'long', 'tdMode': 'cross'}
                )
            elif action == 'CLOSE_SHORT':
                coin_logger.info("操作 | 平空仓")
                exchange.exchange.create_market_order(
                    symbol=f"{coin}/USDT:USDT",
                    side='buy',
                    amount=algo_amount,
                    params={'reduceOnly': True, 'posSide': 'short', 'tdMode': 'cross'}
                )
            elif action == 'HOLD' or action == 'WAIT':
                coin_logger.info("操作 | HOLD")
                if current_position:
                    if (sl != 0 and f"{pos_sl:.2f}" != f"{sl:.2f}"):
                        exchange.exchange.private_post_trade_cancel_algos([{
                            "instId": f"{coin}-USDT-SWAP",
                            "algoId": current_position['algoId']
                        }])
                        coin_logger.info(
                            f"取消历史止盈止损"
                        )
                        params = {
                            "instId": f"{coin}-USDT-SWAP",
                            "tdMode": "cross",
                            "side": "sell" if current_position.get('side') == 'long' else 'buy',
                            "ordType": "conditional",
                            "sz": algo_amount,
                            "slTriggerPx": str(sl),
                            "slOrdPx": str(sl),
                            "posSide": current_position.get('side'),
                        }
                        exchange.exchange.private_post_trade_order_algo(params=params)
                        coin_logger.info(
                            f"调整止盈止损 | 止损 {pos_sl:.2f} -> {sl:.2f}"
                        )
            time.sleep(2)
            position = exchange.get_positions(symbols)
            # coin_logger.info(f"最新持仓 | {summarize_positions(position)}")
            # try:
            #     latest_positions = build_position_payload(position, signal_history)
            #     update_positions_snapshot(latest_positions)
            #     database.record_positions_snapshot(RUN_ID, datetime.now(UTC), latest_positions)
            # except Exception as state_error:
            #     logger.debug(f"Dashboard state update failed | post-trade positions | {state_error}")
        except Exception as e:
            coin_logger.exception(f"订单执行失败: {e}")
            import traceback
            traceback.print_exc()


def summarize_positions(position_map):
    if not position_map:
        return "无持仓"
    parts = []
    for coin, pos in position_map.items():
        parts.append(summarize_position_entry(coin, pos))
    return " || ".join(parts)

def analyze_with_deepseek_with_retry(symbols, max_retries=50):
    """带重试的DeepSeek分析"""
    for attempt in range(max_retries):
        try:
            signal_data = analyze_with_deepseek(symbols)
            if isinstance(signal_data, dict):
                return signal_data
            else:
                logger.warning(f"第{attempt + 1}次尝试失败，进行重试...")
                time.sleep(1)

        except Exception as e:
            logger.warning(f"第{attempt + 1}次尝试异常: {e}")
            if attempt == max_retries - 1:
                return None
            time.sleep(3)

    return None

def get_coins_ohlcv_enhanced(coin_list):
    price_data = {}
    for coin in coin_list:
        price_data[coin] = get_market_data(coin)
    return price_data


def sync_positions_snapshot(snapshot_ts=None):
    """Fetch positions from the exchange and update dashboard state."""
    global last_positions_fingerprint

    timestamp = snapshot_ts or datetime.now(UTC)
    try:
        positions_snapshot = get_current_positions(
            exchange.exchange,
            logger,
            coin_list or [],
            retries=3,
        ) or {}
    except Exception:
        logger.debug("获取持仓失败", exc_info=True)
        return

    positions_payload = build_position_payload(positions_snapshot, signal_history)

    try:
        update_positions_snapshot(positions_payload)
    except Exception as state_error:
        logger.debug(f"Dashboard state update failed | positions sync | {state_error}")

    fingerprint = None
    try:
        fingerprint = json.dumps(positions_payload or [], sort_keys=True, default=str)
    except Exception:
        fingerprint = None

    if fingerprint is not None and fingerprint == last_positions_fingerprint:
        return

    if RUN_ID and positions_payload is not None:
        try:
            database.record_positions_snapshot(RUN_ID, timestamp, positions_payload)
            last_positions_fingerprint = fingerprint
        except Exception:
            logger.debug("数据库写入失败：持仓快照", exc_info=True)


def account_snapshot_job():
    snapshot_ts = None
    try:
        snapshot = capture_account_snapshot()
        if snapshot is not None:
            _, _, _, _, snapshot_ts = snapshot
    except Exception:
        logger.debug("账户快照任务执行失败", exc_info=True)
    try:
        sync_positions_snapshot(snapshot_ts)
    except Exception:
        logger.debug("持仓同步任务执行失败", exc_info=True)


def run_account_snapshot_loop(stop_event: threading.Event) -> None:
    while not stop_event.is_set():
        account_snapshot_job()
        if stop_event.wait(1.0):
            break


def trading_bot():
    """主交易机器人函数"""
    global iteration_counter, minutes_elapsed

    loop_started_at = datetime.now(UTC)
    iteration_counter += 1
    iteration_index = iteration_counter
    minutes_elapsed = max(
        0, int((loop_started_at - start_time).total_seconds() // 60)
    )
    timestamp_local = loop_started_at.astimezone().strftime('%Y-%m-%d %H:%M:%S')

    logger.info("=" * 60)
    logger.info(f"执行时间: {timestamp_local}")
    logger.info(f"执行时长: {minutes_elapsed}分钟")
    logger.info("=" * 60)

    signal_data = None
    try:
        signal_data = analyze_with_deepseek_with_retry(coin_list)
        # print(signal_data)
        execute_trade(signal_data, coin_list)
    finally:
        loop_finished_at = datetime.now(UTC)
        try:
            database.record_runtime_iteration(
                RUN_ID, iteration_index, loop_started_at, loop_finished_at, minutes_elapsed
            )
        except Exception:
            logger.debug("数据库写入失败：运行时指标", exc_info=True)



def run_trading_loop(stop_event: threading.Event) -> None:
    schedule.clear('trading-loop')
    schedule.every(3).minutes.do(trading_bot).tag('trading-loop')

    trading_bot()

    while not stop_event.is_set():
        schedule.run_pending()
        time.sleep(1)


def get_fact_amount(symbol, notional, leverage, price):
    mark = exchange.exchange.load_markets()
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
    global RUN_ID, start_time, minutes_elapsed, iteration_counter, start_time_seeded, initial_equity_value
    
    
    database.initialize()

    setup_log()

    start_time = datetime.now(UTC)
    minutes_elapsed = 0
    iteration_counter = 0
    start_time_seeded = False
    initial_equity_value = None
    RUN_ID = database.register_run(symbols=coin_list or [], timeframe=TRADE_CONFIG.get('timeframe'))
    if RUN_ID:
        logger.info(f"当前运行记录ID：{RUN_ID}")

    try:
        iteration_stats = database.fetch_iteration_stats(RUN_ID)
    except Exception:
        iteration_stats = None
    if iteration_stats and iteration_stats.get("iteration_count"):
        iteration_counter = int(iteration_stats.get("iteration_count") or 0)

    try:
        baseline_ts = database.fetch_first_snapshot_timestamp(RUN_ID)
    except Exception:
        baseline_ts = None
    if baseline_ts is not None:
        start_time = baseline_ts
        start_time_seeded = True
    try:
        baseline_equity = database.fetch_initial_equity_value(RUN_ID)
    except Exception:
        baseline_equity = None
    if baseline_equity is not None:
        initial_equity_value = float(baseline_equity)

    logger.info(f"OKX自动交易机器人启动成功！")
    logger.info("融合技术指标策略 + OKX实盘接口")
    # logger.info(f"交易周期: {TRADE_CONFIG['timeframe']}")
    logger.info("已启用完整技术指标分析和持仓跟踪功能")

    for coin in coin_list:
        coin_logger = get_coin_logger(coin)
        coin_logger.info("=" * 60)
        coin_logger.info(f"代币：{coin}")
        # coin_logger.info(f"交易周期: {TRADE_CONFIG['timeframe']}")
        coin_logger.info("已启用完整技术指标分析和持仓跟踪功能")
        coin_logger.info("=" * 60)

    capture_account_snapshot()

    stop_event = threading.Event()
    trading_thread = threading.Thread(
        target=run_trading_loop,
        args=(stop_event,),
        name="trading-loop",
        daemon=True,
    )
    trading_thread.start()

    account_thread = threading.Thread(
        target=run_account_snapshot_loop,
        args=(stop_event,),
        name="account-snapshot-loop",
        daemon=True,
    )
    account_thread.start()

    host = os.getenv('DASHBOARD_HOST', '0.0.0.0') or '0.0.0.0'
    port_value = os.getenv('DASHBOARD_PORT', '8000') or '8000'
    try:
        port = int(port_value)
    except (TypeError, ValueError):
        port = 8000

    display_host = host if host not in ('0.0.0.0', '::') else '127.0.0.1'
    logger.info(f"仪表盘服务 | http://{display_host}:{port}")

    try:
        uvicorn.run(dashboard_app, host=host, port=port, log_level='info')
    except KeyboardInterrupt:
        logger.info("收到停止信号，正在关闭...")
    finally:
        stop_event.set()
        trading_thread.join(timeout=10)
        account_thread.join(timeout=5)
        try:
            database.mark_run_completed(RUN_ID)
        except Exception:
            logger.debug("数据库写入失败：结束运行标记", exc_info=True)
        logger.info("交易调度器已停止")


if __name__ == "__main__":
    main()
