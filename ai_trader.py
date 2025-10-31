import os
import time
import threading
import schedule
from openai import OpenAI
import ccxt
import pandas as pd
import uvicorn
from datetime import datetime, UTC
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
from dashboard_state import (
    record_strategy_batch,
    record_strategy_signal,
    update_account_snapshot,
    update_positions_snapshot,
)
from dashboard_server import dashboard_app


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

    payload = []
def analyze_with_deepseek(price_data):
    """使用DeepSeek分析市场并生成交易信号（增强版）"""
    
    market_data_prompt = ""
    for coin, data in price_data.items():
        coin_market_text = format_market_data(data)
        market_data_prompt += coin_market_text
    
    positions_snapshot = get_current_positions(exchange, logger, price_data.keys()) or {}
    position_prompt = format_position(positions_snapshot)
    positions_payload = build_position_payload(positions_snapshot, signal_history)
    positions_payload_json = json.dumps(positions_payload, ensure_ascii=False, indent=4)

    balance = exchange.fetch_balance()
    account_value, available_cash, return_pct, sharpe_ratio = compute_account_metrics(balance)
    usdt_balance = available_cash

    try:
        update_account_snapshot(account_value, available_cash, return_pct, sharpe_ratio)
        update_positions_snapshot(positions_payload)
    except Exception as state_error:
        logger.debug(f"Dashboard state update failed | account sync | {state_error}")

    prompt = f"""
    自您开始交易以来已过去{minutes_elapsed}分钟。

    下方为您提供各类状态数据、价格数据及预测信号，助您发掘超额收益。其下为您的当前账户信息、资产价值、业绩表现、持仓情况等。

    ⚠️ **重要提示：以下所有价格或信号数据均按时间排序：最旧 → 最新**

    **时间周期说明：**除非章节标题另有说明，日内系列数据均以**3分钟间隔**提供。若某币种采用不同间隔，将在该币种专属章节中明确标注。

    ---

    ## 所有币种当前市场状态

    {market_data_prompt}

    ## 您的账户信息与表现

    **绩效指标：**
    - 当前总回报率（百分比）：{return_pct:.2f}%
    - 夏普比率：{sharpe_ratio:.2f}

    **账户状态：**
    - 可用现金：${usdt_balance:,.2f}
    - **当前账户价值：** ${account_value:,.2f}

    **当前持仓与业绩：**
{positions_payload_json}

    根据上述数据，请以要求的JSON格式提供您的交易决策。
    """
    
    # print('prompt', prompt)

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
                - **资产池**：{','.join(coin_list)}（永续合约）
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

                1. **开多单**：开立新多头头寸（押注价格上涨）
                - 适用场景：技术面看涨、动能积极、风险回报率利好上涨

                2. **开空单**：建立新空头头寸（押注价格下跌）
                - 适用场景：技术面看跌、动能疲软、风险回报率利空

                3. **持仓**：维持现有仓位不变
                - 适用场景：现有头寸表现符合预期，或无明显优势

                4. **关闭多单**：完全退出现有多仓头寸
                - 适用场景：盈利目标达成、止损触发或交易逻辑失效

                5. **关闭空单**：完全退出现有空仓头寸
                - 适用场景：盈利目标达成、止损触发或交易逻辑失效

                6. **等待**：不做任何操作，等待机会
                - 适用场景：信心不足时，不做任何操作，等到高信息机会


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

                请用以下JSON格式回复，必须包含以下字段：
                
                {{
                    {'|'.join(['"' + coin + '"' for coin in coin_list])}: {{
                        "signal": "OPEN_LONG" | "OPEN_SHORT" | "CLOSE_LONG" | "CLOSE_SHORT" | "HOLD" | "WAIT",
                        "coin": {'|'.join(['"' + coin + '"' for coin in coin_list])},
                        "quantity": <float>,
                        "leverage": <integer 1-20>,
                        "profit_target": <float>,
                        "stop_loss": <float>,
                        "invalidation_condition": "<string>（中文回答）",
                        "confidence": <float 0-1>,
                        "risk_usd": <float>,
                        "justification": "<string>（中文回答）"
                    }}
                }}

                ## 输出验证规则

                - 所有数值字段必须为正数（信号为"持仓"时除外）
                - 止盈价：多单需高于开仓价，空单需低于开仓价
                - 止损价：多单必须低于入场价，空单必须高于入场价
                - 操作说明需简明扼要（最多500字符）
                - 当信号为"持仓"或者"等待"时：设置数量=0，杠杆=1，风险字段使用占位符值

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
            price_obj = price_data.get(coin_code)
            if price_obj is None:
                for key in price_data.keys():
                    if key.upper() == coin_code:
                        price_obj = price_data[key]
                        break
            if price_obj is not None:
                price_snapshot = getattr(price_obj, "current_price", 0.0)

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
            elif confidence_score >= 0.4:
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

        return signal_data

    except Exception as e:
        logger.exception("DeepSeek分析失败")
        return create_fallback_signal(price_data)
    
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
    fallback_signals = []
    timestamp = datetime.now(UTC).isoformat()
    for coin, data in price_data.items():
        current_price = getattr(data, "current_price", 0.0)
        fallback_signals.append(
            {
                "signal": "HOLD",
                "coin": coin,
                "quantity": 0.0,
                "leverage": 1,
                "profit_target": 0.0,
                "stop_loss": 0.0,
                "invalidation_condition": "Fallback hold due to analysis failure",
                "confidence": 0.0,
                "risk_usd": 0.0,
                "justification": "因分析失败，暂时采取保守策略。",
                "reason": "因分析失败，暂时采取保守策略。",
                "take_profit": 0.0,
                "timestamp": timestamp,
                "usdt_amount": 0.0,
                "amount": 0.0,
                "notional_usd": 0.0,
                "confidence_score": 0.0,
                "confidence_label": "LOW",
                "is_fallback": True,
                "price_snapshot": current_price,
            }
        )
        fallback_signals[-1]["confidence"] = "LOW"

    try:
        for signal in fallback_signals:
            record_strategy_signal(str(signal.get("coin", "")).upper(), signal)
    except Exception as state_error:
        logger.debug(f"Dashboard state update failed | fallback strategy | {state_error}")

    return fallback_signals

def get_usdt_balance():
    # 获取账户余额
    balance = exchange.fetch_balance()
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

def execute_trade(signal_data, price_data_obj):
    """执行交易 - OKX版本（修复保证金检查）"""
    """成功能够执行的订单必须先设置倍数"""
    global position

    pos_obj = get_current_positions(exchange, logger, price_data_obj.keys())

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
        coin_key = next((k for k in price_data_obj.keys() if k.upper() == coin), None)
        if coin_key is None:
            coin_logger.error(f"未找到{coin}的行情数据，跳过执行")
            continue

        coin_logger.info(f"=" * 60)
        coin_logger.info(f"=" * int((60 - len(coin)) / 2) + coin + f"=" * int((60 - len(coin)) / 2))
        coin_logger.info(f"=" * 60)
        coin_logger.info(f"代币：{coin}")

        price_data = price_data_obj[coin_key]
        price_snapshot = getattr(price_data, "current_price", 0.0)
        current_position = pos_obj.get(coin_key)
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

        if current_position and (action != 'HOLD' or action != 'WAIT'):
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
                    # continue

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

        if action in ('OPEN_LONG', 'OPEN_SHORT') and confidence_label == 'LOW':
            coin_logger.warning("低信心信号，跳过执行")
            continue

        try:
            balance = exchange.fetch_balance()
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
                pos_tp = float(current_position.get('tp', 0))
                pos_sl = float(current_position.get('sl', 0))
                algo_amount = float(current_position.get('algoAmount', 0))
            else:
                pos_tp = 0
                pos_sl = 0
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

            if posSide and action != 'HOLD':
                setup_exchange(leverage, f"{coin}/USDT:USDT", posSide)

            if action == 'OPEN_LONG':
                coin_logger.info("操作 | 开多仓")
                exchange.create_market_order(
                    f"{coin}/USDT:USDT",
                    'buy',
                    op_amount,
                    params={'posSide': posSide, 'attachAlgoOrds': [{
                        'tpTriggerPx': str(tp),
                        'tpOrdPx': str(tp),
                        'slTriggerPx': str(sl),
                        'slOrdPx': str(sl)
                    }]}
                )
                coin_logger.info("执行完成 | 已提交订单")

            elif action == 'OPEN_SHORT':
                coin_logger.info("操作 | 开空仓")
                exchange.create_market_order(
                    f"{coin}/USDT:USDT",
                    'sell',
                    op_amount,
                    params={'posSide': posSide, 'attachAlgoOrds': [{
                        'tpTriggerPx': str(tp),
                        'tpOrdPx': str(tp),
                        'slTriggerPx': str(sl),
                        'slOrdPx': str(sl)
                    }]}
                )
                coin_logger.info("执行完成 | 已提交订单")
            elif action == 'CLOSE_LONG':
                coin_logger.info("操作 | 平多仓")
                exchange.create_market_order(
                    symbol=f"{coin}/USDT:USDT",
                    side='sell',
                    amount=algo_amount,
                    params={'reduceOnly': True, 'posSide': 'long', 'tdMode': 'cross'}
                )
                coin_logger.info("执行完成 | 已提交订单")
            elif action == 'CLOSE_SHORT':
                coin_logger.info("操作 | 平空仓")
                exchange.create_market_order(
                    symbol=f"{coin}/USDT:USDT",
                    side='buy',
                    amount=algo_amount,
                    params={'reduceOnly': True, 'posSide': 'short', 'tdMode': 'cross'}
                )
                coin_logger.info("执行完成 | 已提交订单")
            elif action == 'HOLD' or action == 'WAIT':
                coin_logger.info("操作 | HOLD")
            time.sleep(2)
            position = get_current_positions(exchange, logger, price_data_obj.keys())
            coin_logger.info(f"最新持仓 | {summarize_positions(position)}")
            try:
                latest_positions = build_position_payload(position, signal_history)
                update_positions_snapshot(latest_positions)
            except Exception as state_error:
                logger.debug(f"Dashboard state update failed | post-trade positions | {state_error}")
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

def analyze_with_deepseek_with_retry(price_data, max_retries=50):
    """带重试的DeepSeek分析"""
    for attempt in range(max_retries):
        try:
            signal_data = analyze_with_deepseek(price_data)
            # print('signal_data', signal_data, isinstance(signal_data, dict))
            if isinstance(signal_data, dict):
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

def get_coins_ohlcv_enhanced(coin_list):
    price_data = {}
    for coin in coin_list:
        price_data[coin] = get_market_data(coin)
    return price_data

def trading_bot():
    """主交易机器人函数"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    minutes_elapsed = datetime.now().minute - start_time.minute
    logger.info("=" * 60)
    logger.info(f"执行时间: {timestamp}")
    logger.info(f"执行时长: {minutes_elapsed}分钟")
    logger.info("=" * 60)

    # 1. 获取增强版K线数据
    price_data = get_coins_ohlcv_enhanced(coin_list)
    if not price_data:
        return
    
    # print(price_data)

    # 2. 使用DeepSeek分析（带重试）
    signal_data = analyze_with_deepseek_with_retry(price_data)

    # # 3. 执行交易
    execute_trade(signal_data, price_data)



def run_trading_loop(stop_event: threading.Event) -> None:
    schedule.clear('trading-loop')
    schedule.every(3).minutes.do(trading_bot).tag('trading-loop')

    trading_bot()

    while not stop_event.is_set():
        schedule.run_pending()
        time.sleep(1)


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

    stop_event = threading.Event()
    trading_thread = threading.Thread(
        target=run_trading_loop,
        args=(stop_event,),
        name="trading-loop",
        daemon=True,
    )
    trading_thread.start()

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
        logger.info("交易调度器已停止")


if __name__ == "__main__":
    main()
