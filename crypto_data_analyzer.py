import os
import pandas as pd
import numpy as np
import ccxt
import time
from datetime import datetime
import talib
import json
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')


class AdvancedMultiCryptoAnalyzer:
    def __init__(self, exchange_id='okx', api_key=None, api_secret=None, password=None, max_workers=4):
        """
        初始化高级多币种加密货币分析器
        
        参数:
        exchange_id (str): 交易所ID，默认为'binance'
        api_key (str): API密钥（如需访问私有API）
        api_secret (str): API密钥（如需访问私有API）
        max_workers (int): 并行处理的最大工作线程数
        """
        # 初始化交易所连接
        exchange_class = getattr(ccxt, exchange_id)
        self.exchange = exchange_class({
            'options': {
                'defaultType': 'swap',  # OKX使用swap表示永续合约
            },
            'apiKey': os.getenv('OKX_API_KEY'),
            'secret': os.getenv('OKX_SECRET'),
            'password': os.getenv('OKX_PASSWORD'),  # OKX需要交易密码
        })
        
        # 设置并行处理的最大工作线程数
        self.max_workers = max_workers
        
        # 存储分析结果的字典
        self.analysis_results = {}
        
        # 相关性矩阵
        self.correlation_matrix = None
        
        # 市场状态
        self.market_state = None
        
        # 波动率排名
        self.volatility_ranking = None
        
        # 动量排名
        self.momentum_ranking = None
        self.exchange.httpsProxy = os.getenv('https_proxy')
        
    
    def fetch_ohlcv_data(self, symbol, timeframe, limit=100):
        """
        获取OHLCV（开高低收成交量）数据
        
        参数:
        symbol (str): 交易对符号，例如 'BTC/USDT'
        timeframe (str): 时间框架，例如 '1h', '4h'
        limit (int): 获取的K线数量
        
        返回:
        pandas.DataFrame: 包含OHLCV数据的DataFrame
        """
        try:
            # 获取OHLCV数据
            ohlcv = self.exchange.fetch_ohlcv(f"{symbol}/USDT:USDT", timeframe, limit=limit)
            
            # 转换为DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # 转换时间戳为可读时间
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df
        except Exception as e:
            print(f"获取{symbol} {timeframe}数据时出错: {e}")
            raise e
    
    def calculate_technical_indicators(self, df):
        """
        计算扩展的技术指标
        
        参数:
        df (pandas.DataFrame): 包含OHLCV数据的DataFrame
        
        返回:
        pandas.DataFrame: 添加了技术指标的DataFrame
        """
        # 确保数据足够计算指标
        if len(df) < 200:
            print("警告: 数据点不足，某些指标可能不准确")
        
        # 创建DataFrame的副本以避免SettingWithCopyWarning
        df = df.copy()
        
        # ===== 趋势指标 =====
        
        # 移动平均线
        df['sma5'] = talib.SMA(df['close'], timeperiod=5)
        df['sma10'] = talib.SMA(df['close'], timeperiod=10)
        df['sma20'] = talib.SMA(df['close'], timeperiod=20)
        df['sma50'] = talib.SMA(df['close'], timeperiod=50)
        df['sma100'] = talib.SMA(df['close'], timeperiod=100)
        df['sma200'] = talib.SMA(df['close'], timeperiod=200)
        
        # 指数移动平均线
        df['ema9'] = talib.EMA(df['close'], timeperiod=9)
        df['ema20'] = talib.EMA(df['close'], timeperiod=20)
        df['ema50'] = talib.EMA(df['close'], timeperiod=50)
        df['ema100'] = talib.EMA(df['close'], timeperiod=100)
        df['ema200'] = talib.EMA(df['close'], timeperiod=200)
        
        # MACD
        macd, macd_signal, macd_hist = talib.MACD(
            df['close'], 
            fastperiod=12, 
            slowperiod=26, 
            signalperiod=9
        )
        df['macd'] = macd
        df['macd_signal'] = macd_signal
        df['macd_hist'] = macd_hist
        
        # 抛物线转向指标 (SAR)
        df['sar'] = talib.SAR(df['high'], df['low'], acceleration=0.02, maximum=0.2)
        
        # 平均方向指数 (ADX) - 趋势强度
        df['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
        df['plus_di'] = talib.PLUS_DI(df['high'], df['low'], df['close'], timeperiod=14)
        df['minus_di'] = talib.MINUS_DI(df['high'], df['low'], df['close'], timeperiod=14)
        
        # 考夫曼自适应移动平均线 (KAMA)
        df['kama'] = talib.KAMA(df['close'], timeperiod=30)
        
        # 三重指数移动平均线 (TEMA)
        df['tema'] = talib.TEMA(df['close'], timeperiod=20)
        
        # ===== 动量指标 =====
        
        # 相对强弱指数 (RSI)
        df['rsi7'] = talib.RSI(df['close'], timeperiod=7)
        df['rsi14'] = talib.RSI(df['close'], timeperiod=14)
        df['rsi21'] = talib.RSI(df['close'], timeperiod=21)
        
        # 随机指标
        slowk, slowd = talib.STOCH(df['high'], df['low'], df['close'], 
                                   fastk_period=14, slowk_period=3, slowk_matype=0, 
                                   slowd_period=3, slowd_matype=0)
        df['stoch_k'] = slowk
        df['stoch_d'] = slowd
        
        # 威廉指标 (Williams %R)
        df['willr'] = talib.WILLR(df['high'], df['low'], df['close'], timeperiod=14)
        
        # 商品通道指数 (CCI)
        df['cci'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=14)
        
        # 动量指标 (MOM)
        df['mom'] = talib.MOM(df['close'], timeperiod=10)
        
        # 变动率指标 (ROC)
        df['roc'] = talib.ROC(df['close'], timeperiod=10)
        
        # 终极波动指标 (Ultimate Oscillator)
        df['ultosc'] = talib.ULTOSC(df['high'], df['low'], df['close'], timeperiod1=7, timeperiod2=14, timeperiod3=28)
        
        # 资金流量指标 (MFI)
        df['mfi'] = talib.MFI(df['high'], df['low'], df['close'], df['volume'], timeperiod=14)
        
        # ===== 波动性指标 =====
        
        # 平均真实波幅 (ATR)
        df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        
        # 相对ATR百分比
        df['atr_percent'] = (df['atr'] / df['close']) * 100
        
        # 布林带
        upper, middle, lower = talib.BBANDS(df['close'], timeperiod=20, nbdevup=2, nbdevdn=2)
        df['bb_upper'] = upper
        df['bb_middle'] = middle
        df['bb_lower'] = lower
        
        # 布林带宽度
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # 肯特纳通道 (Keltner Channel)
        # 使用ATR计算肯特纳通道
        df['kc_middle'] = df['ema20']
        df['kc_upper'] = df['kc_middle'] + 2 * df['atr']
        df['kc_lower'] = df['kc_middle'] - 2 * df['atr']
        
        # 肯特纳通道宽度
        df['kc_width'] = (df['kc_upper'] - df['kc_lower']) / df['kc_middle']
        
        # 标准差
        df['stddev'] = talib.STDDEV(df['close'], timeperiod=20, nbdev=1)
        
        # ===== 成交量指标 =====
        
        # 能量潮指标 (OBV)
        df['obv'] = talib.OBV(df['close'], df['volume'])
        
        # 成交量加权平均价格 (VWAP) - 简化计算
        df['vwap'] = (df['high'] + df['low'] + df['close']) / 3 * df['volume']
        df['vwap'] = df['vwap'].cumsum() / df['volume'].cumsum()
        
        # 资金流量指数 (CMF)
        # 计算资金流量乘数
        df['mfm'] = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
        df['mfm'] = df['mfm'].replace([np.inf, -np.inf], 0)  # 处理除以零的情况
        df['mfv'] = df['mfm'] * df['volume']
        df['cmf'] = df['mfv'].rolling(window=20).sum() / df['volume'].rolling(window=20).sum()
        
        # 价量趋势指标 (PVT)
        df['pvt'] = ((df['close'] - df['close'].shift(1)) / df['close'].shift(1)) * df['volume']
        df['pvt'] = df['pvt'].cumsum()
        
        # ===== 自定义组合指标 =====
        
        # 超级趋势指标 (SuperTrend) - 简化版
        factor = 3.0
        atr_period = 10
        
        # 计算基本上轨和下轨
        df['basic_upperband'] = (df['high'] + df['low']) / 2 + factor * talib.ATR(df['high'], df['low'], df['close'], timeperiod=atr_period)
        df['basic_lowerband'] = (df['high'] + df['low']) / 2 - factor * talib.ATR(df['high'], df['low'], df['close'], timeperiod=atr_period)
        
        # 初始化最终上轨和下轨
        df['final_upperband'] = df['basic_upperband']
        df['final_lowerband'] = df['basic_lowerband']
        
        # 计算超级趋势
        for i in range(1, len(df)):
            if df['close'].iloc[i-1] <= df['final_upperband'].iloc[i-1]:
                df.loc[df.index[i], 'final_upperband'] = min(df['basic_upperband'].iloc[i], df['final_upperband'].iloc[i-1])
            else:
                df.loc[df.index[i], 'final_upperband'] = df['basic_upperband'].iloc[i]
                
            if df['close'].iloc[i-1] >= df['final_lowerband'].iloc[i-1]:
                df.loc[df.index[i], 'final_lowerband'] = max(df['basic_lowerband'].iloc[i], df['final_lowerband'].iloc[i-1])
            else:
                df.loc[df.index[i], 'final_lowerband'] = df['basic_lowerband'].iloc[i]
        
        # 确定趋势方向
        df['supertrend'] = np.nan
        for i in range(len(df)):
            if df['close'].iloc[i] <= df['final_upperband'].iloc[i]:
                df.loc[df.index[i], 'supertrend'] = df['final_upperband'].iloc[i]
            else:
                df.loc[df.index[i], 'supertrend'] = df['final_lowerband'].iloc[i]
        
        # 趋势方向 (1=上升趋势, -1=下降趋势)
        df['supertrend_direction'] = np.where(df['close'] > df['supertrend'], 1, -1)
        
        # 冰山指标 (Iceberg) - 价格与成交量的关系
        df['iceberg'] = df['close'] * df['volume']
        df['iceberg_ma'] = df['iceberg'].rolling(window=20).mean()
        
        # 价格动量发散指标 (PMD)
        df['pmd'] = df['close'] - df['close'].shift(10)
        df['pmd_ma'] = df['pmd'].rolling(window=10).mean()
        
        # 波动率比率 (VR)
        df['high_low_range'] = df['high'] - df['low']
        df['vr'] = df['high_low_range'].rolling(window=10).mean() / df['high_low_range'].rolling(window=30).mean()
        
        # 趋势强度指数 (TSI)
        momentum = df['close'] - df['close'].shift(1)
        abs_momentum = abs(momentum)
        
        # 双重平滑
        smooth_momentum = momentum.ewm(span=25, adjust=False).mean().ewm(span=13, adjust=False).mean()
        smooth_abs_momentum = abs_momentum.ewm(span=25, adjust=False).mean().ewm(span=13, adjust=False).mean()
        
        df['tsi'] = 100 * smooth_momentum / smooth_abs_momentum
        
        # 计算背离
        # MACD背离
        df['macd_divergence'] = np.nan
        for i in range(20, len(df)-1):
            # 寻找局部低点
            if i > 0 and i < len(df)-1:
                if (df['close'].iloc[i-1] > df['close'].iloc[i] < df['close'].iloc[i+1]) and \
                   (df['macd'].iloc[i-1] < df['macd'].iloc[i] > df['macd'].iloc[i+1]):
                    df.loc[df.index[i], 'macd_divergence'] = 1  # 正背离（看涨）
                # 寻找局部高点
                elif (df['close'].iloc[i-1] < df['close'].iloc[i] > df['close'].iloc[i+1]) and \
                     (df['macd'].iloc[i-1] > df['macd'].iloc[i] < df['macd'].iloc[i+1]):
                    df.loc[df.index[i], 'macd_divergence'] = -1  # 负背离（看跌）
        
        # RSI背离
        df['rsi_divergence'] = np.nan
        for i in range(20, len(df)-1):
            # 寻找局部低点
            if i > 0 and i < len(df)-1:
                if (df['close'].iloc[i-1] > df['close'].iloc[i] < df['close'].iloc[i+1]) and \
                   (df['rsi14'].iloc[i-1] < df['rsi14'].iloc[i] > df['rsi14'].iloc[i+1]):
                    df.loc[df.index[i], 'rsi_divergence'] = 1  # 正背离（看涨）
                # 寻找局部高点
                elif (df['close'].iloc[i-1] < df['close'].iloc[i] > df['close'].iloc[i+1]) and \
                     (df['rsi14'].iloc[i-1] > df['rsi14'].iloc[i] < df['rsi14'].iloc[i+1]):
                    df.loc[df.index[i], 'rsi_divergence'] = -1  # 负背离（看跌）
        
        # ===== 市场结构指标 =====
        
        # 计算高点和低点
        df['local_high'] = df['high'].rolling(window=5, center=True).max()
        df['local_low'] = df['low'].rolling(window=5, center=True).min()
        
        # 识别高点和低点
        df['is_high'] = np.where(df['high'] == df['local_high'], 1, 0)
        df['is_low'] = np.where(df['low'] == df['local_low'], 1, 0)
        
        # 计算市场结构
        df['higher_high'] = np.nan
        df['higher_low'] = np.nan
        df['lower_high'] = np.nan
        df['lower_low'] = np.nan
        
        # 寻找最近的高点和低点
        high_points = []
        low_points = []
        
        for i in range(2, len(df)-2):
            # 局部高点
            if df['high'].iloc[i] > df['high'].iloc[i-1] and df['high'].iloc[i] > df['high'].iloc[i-2] and \
               df['high'].iloc[i] > df['high'].iloc[i+1] and df['high'].iloc[i] > df['high'].iloc[i+2]:
                high_points.append((i, df['high'].iloc[i]))
                
                # 检查是否形成更高的高点
                if len(high_points) >= 2 and high_points[-1][1] > high_points[-2][1]:
                    df.loc[df.index[i], 'higher_high'] = 1
                elif len(high_points) >= 2 and high_points[-1][1] < high_points[-2][1]:
                    df.loc[df.index[i], 'lower_high'] = 1
            
            # 局部低点
            if df['low'].iloc[i] < df['low'].iloc[i-1] and df['low'].iloc[i] < df['low'].iloc[i-2] and \
               df['low'].iloc[i] < df['low'].iloc[i+1] and df['low'].iloc[i] < df['low'].iloc[i+2]:
                low_points.append((i, df['low'].iloc[i]))
                
                # 检查是否形成更高的低点
                if len(low_points) >= 2 and low_points[-1][1] > low_points[-2][1]:
                    df.loc[df.index[i], 'higher_low'] = 1
                elif len(low_points) >= 2 and low_points[-1][1] < low_points[-2][1]:
                    df.loc[df.index[i], 'lower_low'] = 1
        
        # ===== 综合评分指标 =====
        
        # 趋势评分
        df['trend_score'] = 0
        
        # EMA趋势评分
        df['trend_score'] += np.where(df['close'] > df['ema20'], 1, -1)
        df['trend_score'] += np.where(df['ema20'] > df['ema50'], 1, -1)
        df['trend_score'] += np.where(df['ema50'] > df['ema200'], 2, -2)
        
        # MACD趋势评分
        df['trend_score'] += np.where(df['macd'] > df['macd_signal'], 1, -1)
        df['trend_score'] += np.where(df['macd_hist'] > 0, 1, -1)
        df['trend_score'] += np.where(df['macd_hist'] > df['macd_hist'].shift(1), 0.5, -0.5)
        
        # ADX趋势强度评分
        df['trend_score'] += np.where(df['adx'] > 25, 1, 0)
        df['trend_score'] += np.where(df['adx'] > 40, 1, 0)
        
        # 超级趋势评分
        df['trend_score'] += df['supertrend_direction']
        
        # 动量评分
        df['momentum_score'] = 0
        
        # RSI动量评分
        df['momentum_score'] += np.where(df['rsi14'] > 50, 1, -1)
        df['momentum_score'] += np.where(df['rsi14'] > 70, -1, 0)  # 超买惩罚
        df['momentum_score'] += np.where(df['rsi14'] < 30, 1, 0)   # 超卖奖励
        
        # 随机指标动量评分
        df['momentum_score'] += np.where(df['stoch_k'] > df['stoch_d'], 1, -1)
        df['momentum_score'] += np.where(df['stoch_k'] > 80, -1, 0)  # 超买惩罚
        df['momentum_score'] += np.where(df['stoch_k'] < 20, 1, 0)   # 超卖奖励
        
        # CCI动量评分
        df['momentum_score'] += np.where(df['cci'] > 0, 0.5, -0.5)
        df['momentum_score'] += np.where(df['cci'] > 100, -0.5, 0)  # 超买惩罚
        df['momentum_score'] += np.where(df['cci'] < -100, 0.5, 0)  # 超卖奖励
        
        # 波动性评分
        df['volatility_score'] = 0
        
        # ATR波动性评分
        df['volatility_score'] += np.where(df['atr_percent'] > df['atr_percent'].rolling(window=20).mean(), 0.5, -0.5)
        
        # 布林带宽度评分
        df['volatility_score'] += np.where(df['bb_width'] < df['bb_width'].rolling(window=20).mean(), 0.5, -0.5)  # 收缩波动性
        
        # 价格相对于布林带位置
        df['volatility_score'] += np.where(df['close'] > df['bb_upper'], -1, 0)  # 超出上轨惩罚
        df['volatility_score'] += np.where(df['close'] < df['bb_lower'], 1, 0)   # 超出下轨奖励
        
        # 成交量评分
        df['volume_score'] = 0
        
        # 成交量趋势评分
        df['volume_score'] += np.where(df['volume'] > df['volume'].rolling(window=20).mean(), 0.5, -0.5)
        
        # OBV成交量评分
        df['volume_score'] += np.where(df['obv'] > df['obv'].shift(1), 0.5, -0.5)
        
        # CMF成交量评分
        df['volume_score'] += np.where(df['cmf'] > 0, 0.5, -0.5)
        
        # 综合技术评分 (范围: -10 到 10)
        df['technical_score'] = (
            df['trend_score'] * 0.4 +  # 趋势权重40%
            df['momentum_score'] * 0.3 +  # 动量权重30%
            df['volatility_score'] * 0.1 +  # 波动性权重10%
            df['volume_score'] * 0.2  # 成交量权重20%
        )
        
        # 标准化评分到-100到100范围
        max_score = 10
        df['technical_score'] = (df['technical_score'] / max_score) * 100
        
        return df
    
    def identify_support_resistance(self, df, window=10, threshold=0.01):
        """
        识别支撑位和阻力位
        
        参数:
        df (pandas.DataFrame): 价格数据
        window (int): 寻找局部极值的窗口大小
        threshold (float): 价格变化阈值，用于过滤微小波动
        
        返回:
        tuple: (支撑位列表, 阻力位列表)
        """
        supports = []
        resistances = []
        
        # 获取高点和低点
        for i in range(window, len(df) - window):
            # 检查是否是局部低点（潜在支撑位）
            if all(df['low'].iloc[i] <= df['low'].iloc[i-j] for j in range(1, window+1)) and \
               all(df['low'].iloc[i] <= df['low'].iloc[i+j] for j in range(1, window+1)):
                supports.append(df['low'].iloc[i])
            
            # 检查是否是局部高点（潜在阻力位）
            if all(df['high'].iloc[i] >= df['high'].iloc[i-j] for j in range(1, window+1)) and \
               all(df['high'].iloc[i] >= df['high'].iloc[i+j] for j in range(1, window+1)):
                resistances.append(df['high'].iloc[i])
        
        # 合并相近的支撑位和阻力位
        supports = self._merge_levels(supports, threshold)
        resistances = self._merge_levels(resistances, threshold)
        
        # 按价格排序
        supports.sort()
        resistances.sort()
        
        # 过滤掉远离当前价格的水平
        current_price = df['close'].iloc[-1]
        supports = [s for s in supports if s < current_price and s > current_price * 0.7]
        resistances = [r for r in resistances if r > current_price and r < current_price * 1.3]
        
        # 取最近的几个水平
        supports = supports[-3:] if supports else []
        resistances = resistances[:3] if resistances else []
        
        return supports, resistances
    
    def identify_fibonacci_levels(self, df, trend='auto', depth=30):
        """
        识别斐波那契回调和扩展水平
        
        参数:
        df (pandas.DataFrame): 价格数据
        trend (str): 'up', 'down', 或 'auto'
        depth (int): 寻找高低点的回溯周期
        
        返回:
        dict: 斐波那契水平
        """
        # 获取最近的价格范围
        recent_df = df.tail(depth)
        
        # 自动检测趋势
        if trend == 'auto':
            if recent_df['close'].iloc[-1] > recent_df['close'].iloc[0]:
                trend = 'up'
            else:
                trend = 'down'
        
        # 根据趋势找到高点和低点
        if trend == 'up':
            high = recent_df['high'].max()
            high_idx = recent_df['high'].idxmax()
            # 在高点之前找低点
            low_df = df.loc[:high_idx]
            if not low_df.empty:
                low = low_df['low'].min()
            else:
                low = recent_df['low'].min()
        else:  # trend == 'down'
            low = recent_df['low'].min()
            low_idx = recent_df['low'].idxmin()
            # 在低点之前找高点
            high_df = df.loc[:low_idx]
            if not high_df.empty:
                high = high_df['high'].max()
            else:
                high = recent_df['high'].max()
        
        # 计算价格范围
        price_range = high - low
        
        # 计算斐波那契水平
        fib_levels = {
            'trend': trend,
            'high': high,
            'low': low,
            'levels': {}
        }
        
        # 回调水平
        fib_ratios = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1]
        
        for ratio in fib_ratios:
            if trend == 'up':
                fib_levels['levels'][ratio] = high - (price_range * ratio)
            else:  # trend == 'down'
                fib_levels['levels'][ratio] = low + (price_range * ratio)
        
        # 扩展水平
        ext_ratios = [1.272, 1.414, 1.618, 2.0, 2.618]
        
        for ratio in ext_ratios:
            if trend == 'up':
                fib_levels['levels'][ratio] = high + (price_range * (ratio - 1))
            else:  # trend == 'down'
                fib_levels['levels'][ratio] = low - (price_range * (ratio - 1))
        
        return fib_levels
    
    def identify_chart_patterns(self, df, window=20):
        """
        识别常见图表形态
        
        参数:
        df (pandas.DataFrame): 价格数据
        window (int): 寻找形态的窗口大小
        
        返回:
        dict: 识别到的图表形态
        """
        patterns = {
            'head_and_shoulders': False,
            'inverse_head_and_shoulders': False,
            'double_top': False,
            'double_bottom': False,
            'triple_top': False,
            'triple_bottom': False,
            'ascending_triangle': False,
            'descending_triangle': False,
            'symmetrical_triangle': False,
            'bullish_flag': False,
            'bearish_flag': False,
            'bullish_pennant': False,
            'bearish_pennant': False
        }
        
        # 使用简化的方法检测形态，不依赖TA-Lib的模式识别函数
        
        # 双顶/双底检测
        recent_highs = df['high'].rolling(window=5).max()
        recent_lows = df['low'].rolling(window=5).min()
        
        if len(recent_highs) >= window:
            # 寻找局部高点
            max_indices = []
            for i in range(5, len(recent_highs)):
                if recent_highs.iloc[i-2] < recent_highs.iloc[i-1] > recent_highs.iloc[i]:
                    max_indices.append(i-1)
            
            # 检查双顶
            if len(max_indices) >= 2:
                # 检查两个高点是否接近
                if abs(df['high'].iloc[max_indices[0]] - df['high'].iloc[max_indices[1]]) / df['high'].iloc[max_indices[0]] < 0.03:
                    patterns['double_top'] = True
        
        if len(recent_lows) >= window:
            # 寻找局部低点
            min_indices = []
            for i in range(5, len(recent_lows)):
                if recent_lows.iloc[i-2] > recent_lows.iloc[i-1] < recent_lows.iloc[i]:
                    min_indices.append(i-1)
            
            # 检查双底
            if len(min_indices) >= 2:
                # 检查两个低点是否接近
                if abs(df['low'].iloc[min_indices[0]] - df['low'].iloc[min_indices[1]]) / df['low'].iloc[min_indices[0]] < 0.03:
                    patterns['double_bottom'] = True
        
        # 三重顶和三重底的简化检测
        if len(recent_highs) >= window and len(recent_lows) >= window:
            # 寻找局部高点
            max_indices = []
            for i in range(5, len(recent_highs)):
                if recent_highs.iloc[i-2] < recent_highs.iloc[i-1] > recent_highs.iloc[i]:
                    max_indices.append(i-1)
            
            # 寻找局部低点
            min_indices = []
            for i in range(5, len(recent_lows)):
                if recent_lows.iloc[i-2] > recent_lows.iloc[i-1] < recent_lows.iloc[i]:
                    min_indices.append(i-1)
            
            # 检查三重顶
            if len(max_indices) >= 3:
                # 检查三个高点是否接近
                if (abs(df['high'].iloc[max_indices[0]] - df['high'].iloc[max_indices[1]]) / df['high'].iloc[max_indices[0]] < 0.03 and
                    abs(df['high'].iloc[max_indices[1]] - df['high'].iloc[max_indices[2]]) / df['high'].iloc[max_indices[1]] < 0.03):
                    patterns['triple_top'] = True
            
            # 检查三重底
            if len(min_indices) >= 3:
                # 检查三个低点是否接近
                if (abs(df['low'].iloc[min_indices[0]] - df['low'].iloc[min_indices[1]]) / df['low'].iloc[min_indices[0]] < 0.03 and
                    abs(df['low'].iloc[min_indices[1]] - df['low'].iloc[min_indices[2]]) / df['low'].iloc[min_indices[1]] < 0.03):
                    patterns['triple_bottom'] = True
        
        # 简化的三角形检测
        if len(df) >= window:
            # 获取窗口内的数据
            window_df = df.tail(window)
            
            # 检查上升三角形
            higher_lows = all(window_df['low'].iloc[i] >= window_df['low'].iloc[i-1] for i in range(1, len(window_df), 3))
            flat_highs = abs(window_df['high'].max() - window_df['high'].min()) / window_df['high'].mean() < 0.03
            
            if higher_lows and flat_highs:
                patterns['ascending_triangle'] = True
            
            # 检查下降三角形
            lower_highs = all(window_df['high'].iloc[i] <= window_df['high'].iloc[i-1] for i in range(1, len(window_df), 3))
            flat_lows = abs(window_df['low'].max() - window_df['low'].min()) / window_df['low'].mean() < 0.03
            
            if lower_highs and flat_lows:
                patterns['descending_triangle'] = True
            
            # 检查对称三角形
            higher_lows = all(window_df['low'].iloc[i] >= window_df['low'].iloc[i-2] for i in range(2, len(window_df), 3))
            lower_highs = all(window_df['high'].iloc[i] <= window_df['high'].iloc[i-2] for i in range(2, len(window_df), 3))
            
            if higher_lows and lower_highs:
                patterns['symmetrical_triangle'] = True
        
        # 旗形和三角旗形的简化检测
        if len(df) >= window * 2:
            # 获取前一个窗口和当前窗口的数据
            prev_window = df.iloc[-window*2:-window]
            curr_window = df.iloc[-window:]
            
            # 检查看涨旗形
            if prev_window['close'].iloc[0] < prev_window['close'].iloc[-1]:  # 前一个窗口是上升趋势
                # 当前窗口是小幅下跌整理
                if (curr_window['high'].max() < prev_window['close'].iloc[-1] and
                    curr_window['low'].min() > prev_window['close'].iloc[0] and
                    curr_window['close'].iloc[-1] < curr_window['close'].iloc[0]):
                    patterns['bullish_flag'] = True
            
            # 检查看跌旗形
            if prev_window['close'].iloc[0] > prev_window['close'].iloc[-1]:  # 前一个窗口是下降趋势
                # 当前窗口是小幅上涨整理
                if (curr_window['low'].min() > prev_window['close'].iloc[-1] and
                    curr_window['high'].max() < prev_window['close'].iloc[0] and
                    curr_window['close'].iloc[-1] > curr_window['close'].iloc[0]):
                    patterns['bearish_flag'] = True
            
            # 检查看涨三角旗
            if prev_window['close'].iloc[0] < prev_window['close'].iloc[-1]:  # 前一个窗口是上升趋势
                # 当前窗口是收敛三角形
                higher_lows = all(curr_window['low'].iloc[i] >= curr_window['low'].iloc[i-2] for i in range(2, len(curr_window), 3))
                lower_highs = all(curr_window['high'].iloc[i] <= curr_window['high'].iloc[i-2] for i in range(2, len(curr_window), 3))
                
                if higher_lows and lower_highs:
                    patterns['bullish_pennant'] = True
            
            # 检查看跌三角旗
            if prev_window['close'].iloc[0] > prev_window['close'].iloc[-1]:  # 前一个窗口是下降趋势
                # 当前窗口是收敛三角形
                higher_lows = all(curr_window['low'].iloc[i] >= curr_window['low'].iloc[i-2] for i in range(2, len(curr_window), 3))
                lower_highs = all(curr_window['high'].iloc[i] <= curr_window['high'].iloc[i-2] for i in range(2, len(curr_window), 3))
                
                if higher_lows and lower_highs:
                    patterns['bearish_pennant'] = True
        
        return patterns
    
    def identify_candlestick_patterns(self, df):
        """
        识别蜡烛图形态
        
        参数:
        df (pandas.DataFrame): 价格数据
        
        返回:
        dict: 最近的蜡烛图形态
        """
        patterns = {}
        
        # 检查TA-Lib中是否有蜡烛图形态识别函数
        try:
            # 尝试使用TA-Lib的蜡烛图形态识别函数
        
            patterns['doji'] = talib.CDLDOJI(df['open'], df['high'], df['low'], df['close'])
            patterns['engulfing'] = talib.CDLENGULFING(df['open'], df['high'], df['low'], df['close'])        
            patterns['hammer'] = talib.CDLHAMMER(df['open'], df['high'], df['low'], df['close'])
            patterns['hanging_man'] = talib.CDLHANGINGMAN(df['open'], df['high'], df['low'], df['close'])
            patterns['shooting_star'] = talib.CDLSHOOTINGSTAR(df['open'], df['high'], df['low'], df['close'])
            patterns['morning_star'] = talib.CDLMORNINGSTAR(df['open'], df['high'], df['low'], df['close'], penetration=0)
            patterns['evening_star'] = talib.CDLEVENINGSTAR(df['open'], df['high'], df['low'], df['close'], penetration=0)
            patterns['three_white_soldiers'] = talib.CDL3WHITESOLDIERS(df['open'], df['high'], df['low'], df['close'])
            patterns['three_black_crows'] = talib.CDL3BLACKCROWS(df['open'], df['high'], df['low'], df['close'])
            patterns['harami'] = talib.CDLHARAMI(df['open'], df['high'], df['low'], df['close'])
            patterns['piercing'] = talib.CDLPIERCING(df['open'], df['high'], df['low'], df['close'])
            patterns['dark_cloud_cover'] = talib.CDLDARKCLOUDCOVER(df['open'], df['high'], df['low'], df['close'], penetration=0)
        
        except Exception as e:
            print(f"蜡烛图形态识别出错: {e}")
            # 如果TA-Lib函数不可用，使用简化的方法
            patterns = self._identify_basic_candlestick_patterns(df)
        
        # 获取最近的形态
        recent_patterns = {}
        for pattern_name, pattern_values in patterns.items():
            if isinstance(pattern_values, pd.Series):
                # 检查最近5根K线
                recent_values = pattern_values.iloc[-5:].values
                if any(recent_values != 0):
                    # 找到最近的非零值
                    for i in range(len(recent_values)-1, -1, -1):
                        if recent_values[i] != 0:
                            recent_patterns[pattern_name] = {
                                'value': int(recent_values[i]),
                                'position': i
                            }
                            break
            else:
                # 如果是使用简化方法生成的字典
                recent_patterns.update(pattern_values)
        
        return recent_patterns
    
    def _identify_basic_candlestick_patterns(self, df):
        """
        使用基本逻辑识别蜡烛图形态（不依赖TA-Lib）
        
        参数:
        df (pandas.DataFrame): 价格数据
        
        返回:
        dict: 蜡烛图形态
        """
        patterns = {}
        
        # 计算蜡烛体和影线
        df = df.copy()
        df['body'] = abs(df['close'] - df['open'])
        df['body_pct'] = df['body'] / ((df['high'] + df['low']) / 2) * 100
        df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
        df['is_bullish'] = df['close'] > df['open']
        df['is_bearish'] = df['close'] < df['open']
        
        # 十字星
        doji_condition = (df['body_pct'] < 0.1) & (df['upper_shadow'] > 0) & (df['lower_shadow'] > 0)
        if doji_condition.iloc[-1]:
            patterns['doji'] = {'value': 1, 'position': 0}
        
        # 锤子线
        hammer_condition = (df['body_pct'] < 0.3) & (df['lower_shadow'] > 2 * df['body']) & (df['upper_shadow'] < 0.1 * df['body'])
        if hammer_condition.iloc[-1]:
            if df['is_bullish'].iloc[-1]:
                patterns['hammer'] = {'value': 1, 'position': 0}
            else:
                patterns['hanging_man'] = {'value': -1, 'position': 0}
        
        # 流星线
        shooting_star_condition = (df['body_pct'] < 0.3) & (df['upper_shadow'] > 2 * df['body']) & (df['lower_shadow'] < 0.1 * df['body'])
        if shooting_star_condition.iloc[-1]:
            patterns['shooting_star'] = {'value': -1, 'position': 0}
        
        # 吞没形态
        if len(df) >= 2:
            bullish_engulfing = (df['is_bearish'].iloc[-2]) & (df['is_bullish'].iloc[-1]) & (df['open'].iloc[-1] < df['close'].iloc[-2]) & (df['close'].iloc[-1] > df['open'].iloc[-2])
            bearish_engulfing = (df['is_bullish'].iloc[-2]) & (df['is_bearish'].iloc[-1]) & (df['open'].iloc[-1] > df['close'].iloc[-2]) & (df['close'].iloc[-1] < df['open'].iloc[-2])
            
            if bullish_engulfing:
                patterns['engulfing'] = {'value': 1, 'position': 0}
            elif bearish_engulfing:
                patterns['engulfing'] = {'value': -1, 'position': 0}
        
        return patterns
    
    def _merge_levels(self, levels, threshold):
        """
        合并相近的价格水平
        
        参数:
        levels (list): 价格水平列表
        threshold (float): 合并阈值
        
        返回:
        list: 合并后的价格水平
        """
        if not levels:
            return []
        
        # 排序价格水平
        sorted_levels = sorted(levels)
        merged = [sorted_levels[0]]
        
        for level in sorted_levels[1:]:
            # 如果当前水平与最后一个合并水平相差不大，则合并
            if (level - merged[-1]) / merged[-1] <= threshold:
                # 更新为平均值
                merged[-1] = (merged[-1] + level) / 2
            else:
                merged.append(level)
        
        return merged
    
    def get_market_data(self, symbol):
        """
        获取市场数据（资金费率、24小时价格变化、未平仓合约）
        
        参数:
        symbol (str): 交易对符号
        
        返回:
        dict: 市场数据字典
        """
        market_data = {}
        
        try:
            # 获取资金费率
            funding_rate_info = self.exchange.fetch_funding_rate(f"{symbol}/USDT:USDT")
            market_data['funding_rate'] = funding_rate_info.get('fundingRate')
        except Exception as e:
            print(f"获取{symbol}资金费率时出错: {e}")
            market_data['funding_rate'] = None
            raise e
        
        try:
            # 获取24小时价格变化
            ticker = self.exchange.fetch_ticker(f"{symbol}/USDT:USDT")
            market_data['price_change_24h'] = ticker.get('percentage')
            market_data['volume_24h'] = ticker.get('info').get('volCcy24h')
        except Exception as e:
            print(f"获取{symbol}24小时价格变化时出错: {e}")
            market_data['price_change_24h'] = None
            market_data['volume_24h'] = None
            raise e
        
        try:
            # 获取未平仓合约数量
            open_interest = self.exchange.fetch_open_interest(f"{symbol}-USDT-SWAP")
            market_data['open_interest'] = open_interest.get('openInterestValue')
            
            # 如果有历史数据，计算变化
            # 这需要存储之前的数据，这里简化处理
            market_data['open_interest_change'] = None
        except Exception as e:
            print(f"获取{symbol}未平仓合约数量时出错: {e}")
            market_data['open_interest'] = None
            market_data['open_interest_change'] = None
            raise e
        
        return market_data
    
    def analyze_single_symbol(self, symbol, max_retries=50):
        """
        分析单个交易对
        
        参数:
        symbol (str): 交易对符号
        
        返回:
        dict: 分析结果
        """
    
        print(f"正在分析 {symbol}...")
        
        # 获取1小时和4小时数据
        df_1h = self.fetch_ohlcv_data(symbol, '1h', limit=200)  # 增加数据量以计算更多指标
        df_4h = self.fetch_ohlcv_data(symbol, '4h', limit=200)
        
        
        if df_1h is None or df_4h is None:
            print(f"无法获取{symbol}的完整数据，跳过分析")
            return None
        
        # 计算技术指标
        df_1h = self.calculate_technical_indicators(df_1h)
        df_4h = self.calculate_technical_indicators(df_4h)
        
        # 识别支撑位和阻力位
        supports_1h, resistances_1h = self.identify_support_resistance(df_1h)
        supports_4h, resistances_4h = self.identify_support_resistance(df_4h)
        
        # 识别斐波那契水平
        fib_levels_1h = self.identify_fibonacci_levels(df_1h)
        fib_levels_4h = self.identify_fibonacci_levels(df_4h)
        
        # 识别图表形态
        chart_patterns_1h = self.identify_chart_patterns(df_1h)
        chart_patterns_4h = self.identify_chart_patterns(df_4h)
        
        # 识别蜡烛图形态
        candlestick_patterns_1h = self.identify_candlestick_patterns(df_1h)
        candlestick_patterns_4h = self.identify_candlestick_patterns(df_4h)
        
        # 获取市场数据
        market_data = self.get_market_data(symbol)
        
        # 当前价格
        current_price = df_1h['close'].iloc[-1]
        
        # 准备分析数据
        analysis_data = {
            'symbol': symbol,
            'current_price': current_price,
            'current_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            
            '4h_data': {
                'price_data': df_4h[['open', 'high', 'low', 'close']].tail(10).values.tolist(),
                'volume': df_4h['volume'].tail(10).tolist(),
                'ema': {
                    '9': df_4h['ema9'].iloc[-1],
                    '20': df_4h['ema20'].iloc[-1],
                    '50': df_4h['ema50'].iloc[-1],
                    '100': df_4h['ema100'].iloc[-1],
                    '200': df_4h['ema200'].iloc[-1]
                },
                'sma': {
                    '20': df_4h['sma20'].iloc[-1],
                    '50': df_4h['sma50'].iloc[-1],
                    '200': df_4h['sma200'].iloc[-1]
                },
                'rsi': {
                    '7': df_4h['rsi7'].iloc[-1],
                    '14': df_4h['rsi14'].iloc[-1],
                    '21': df_4h['rsi21'].iloc[-1]
                },
                'macd': {
                    'line': df_4h['macd'].iloc[-1],
                    'signal': df_4h['macd_signal'].iloc[-1],
                    'histogram': df_4h['macd_hist'].iloc[-1]
                },
                'stochastic': {
                    'k': df_4h['stoch_k'].iloc[-1],
                    'd': df_4h['stoch_d'].iloc[-1]
                },
                'bollinger_bands': {
                    'upper': df_4h['bb_upper'].iloc[-1],
                    'middle': df_4h['bb_middle'].iloc[-1],
                    'lower': df_4h['bb_lower'].iloc[-1],
                    'width': df_4h['bb_width'].iloc[-1]
                },
                'atr': df_4h['atr'].iloc[-1],
                'atr_percent': df_4h['atr_percent'].iloc[-1],
                'adx': df_4h['adx'].iloc[-1],
                'plus_di': df_4h['plus_di'].iloc[-1],
                'minus_di': df_4h['minus_di'].iloc[-1],
                'cci': df_4h['cci'].iloc[-1],
                'mfi': df_4h['mfi'].iloc[-1],
                'obv': df_4h['obv'].iloc[-1],
                'supertrend': {
                    'value': df_4h['supertrend'].iloc[-1],
                    'direction': df_4h['supertrend_direction'].iloc[-1]
                },
                'support_levels': supports_4h,
                'resistance_levels': resistances_4h,
                'fibonacci_levels': fib_levels_4h,
                'chart_patterns': chart_patterns_4h,
                'candlestick_patterns': candlestick_patterns_4h,
                'technical_score': df_4h['technical_score'].iloc[-1],
                'trend_score': df_4h['trend_score'].iloc[-1],
                'momentum_score': df_4h['momentum_score'].iloc[-1],
                'volatility_score': df_4h['volatility_score'].iloc[-1],
                'volume_score': df_4h['volume_score'].iloc[-1]
            },
            
            '1h_data': {
                'price_data': df_1h[['open', 'high', 'low', 'close']].tail(10).values.tolist(),
                'volume': df_1h['volume'].tail(10).tolist(),
                'ema': {
                    '9': df_1h['ema9'].iloc[-1],
                    '20': df_1h['ema20'].iloc[-1],
                    '50': df_1h['ema50'].iloc[-1],
                    '100': df_1h['ema100'].iloc[-1],
                    '200': df_1h['ema200'].iloc[-1]
                },
                'sma': {
                    '20': df_1h['sma20'].iloc[-1],
                    '50': df_1h['sma50'].iloc[-1],
                    '200': df_1h['sma200'].iloc[-1]
                },
                'rsi': {
                    '7': df_1h['rsi7'].iloc[-1],
                    '14': df_1h['rsi14'].iloc[-1],
                    '21': df_1h['rsi21'].iloc[-1]
                },
                'macd': {
                    'line': df_1h['macd'].iloc[-1],
                    'signal': df_1h['macd_signal'].iloc[-1],
                    'histogram': df_1h['macd_hist'].iloc[-1]
                },
                'stochastic': {
                    'k': df_1h['stoch_k'].iloc[-1],
                    'd': df_1h['stoch_d'].iloc[-1]
                },
                'bollinger_bands': {
                    'upper': df_1h['bb_upper'].iloc[-1],
                    'middle': df_1h['bb_middle'].iloc[-1],
                    'lower': df_1h['bb_lower'].iloc[-1],
                    'width': df_1h['bb_width'].iloc[-1]
                },
                'atr': df_1h['atr'].iloc[-1],
                'atr_percent': df_1h['atr_percent'].iloc[-1],
                'adx': df_1h['adx'].iloc[-1],
                'plus_di': df_1h['plus_di'].iloc[-1],
                'minus_di': df_1h['minus_di'].iloc[-1],
                'cci': df_1h['cci'].iloc[-1],
                'mfi': df_1h['mfi'].iloc[-1],
                'obv': df_1h['obv'].iloc[-1],
                'supertrend': {
                    'value': df_1h['supertrend'].iloc[-1],
                    'direction': df_1h['supertrend_direction'].iloc[-1]
                },
                'support_levels': supports_1h,
                'resistance_levels': resistances_1h,
                'fibonacci_levels': fib_levels_1h,
                'chart_patterns': chart_patterns_1h,
                'candlestick_patterns': candlestick_patterns_1h,
                'technical_score': df_1h['technical_score'].iloc[-1],
                'trend_score': df_1h['trend_score'].iloc[-1],
                'momentum_score': df_1h['momentum_score'].iloc[-1],
                'volatility_score': df_1h['volatility_score'].iloc[-1],
                'volume_score': df_1h['volume_score'].iloc[-1]
            },
            
            'market_data': market_data,
            
            # 添加综合技术评分
            'technical_score': (df_1h['technical_score'].iloc[-1] * 0.4 + df_4h['technical_score'].iloc[-1] * 0.6)
        }
        
        # 添加趋势判断
        analysis_data['trend_analysis'] = self.determine_trend(df_1h, df_4h)
        
        # 添加背离分析
        analysis_data['divergence_analysis'] = self.analyze_divergences(df_1h, df_4h)
        
        # 添加波动性分析
        analysis_data['volatility_analysis'] = self.analyze_volatility(df_1h, df_4h)
        
        # 添加市场结构分析
        analysis_data['market_structure'] = self.analyze_market_structure(df_1h, df_4h)
        
        # 添加交易信号
        analysis_data['trading_signals'] = self.generate_trading_signals(df_1h, df_4h, analysis_data)
        
        return analysis_data
    
    def determine_trend(self, df_1h, df_4h):
        """
        确定趋势方向和强度
        
        参数:
        df_1h (pandas.DataFrame): 1小时数据
        df_4h (pandas.DataFrame): 4小时数据
        
        返回:
        dict: 趋势分析结果
        """
        trend_analysis = {}
        
        # 1小时趋势
        close_1h = df_1h['close'].iloc[-1]
        ema20_1h = df_1h['ema20'].iloc[-1]
        ema50_1h = df_1h['ema50'].iloc[-1]
        ema200_1h = df_1h['ema200'].iloc[-1]
        
        # 价格相对于EMA的位置
        if close_1h > ema20_1h > ema50_1h > ema200_1h:
            trend_1h = "强烈上升"
        elif close_1h > ema20_1h > ema50_1h:
            trend_1h = "上升"
        elif close_1h > ema20_1h:
            trend_1h = "弱上升"
        elif close_1h < ema20_1h < ema50_1h < ema200_1h:
            trend_1h = "强烈下降"
        elif close_1h < ema20_1h < ema50_1h:
            trend_1h = "下降"
        elif close_1h < ema20_1h:
            trend_1h = "弱下降"
        else:
            trend_1h = "盘整"
        
        # 4小时趋势
        close_4h = df_4h['close'].iloc[-1]
        ema20_4h = df_4h['ema20'].iloc[-1]
        ema50_4h = df_4h['ema50'].iloc[-1]
        ema200_4h = df_4h['ema200'].iloc[-1]
        
        # 价格相对于EMA的位置
        if close_4h > ema20_4h > ema50_4h > ema200_4h:
            trend_4h = "强烈上升"
        elif close_4h > ema20_4h > ema50_4h:
            trend_4h = "上升"
        elif close_4h > ema20_4h:
            trend_4h = "弱上升"
        elif close_4h < ema20_4h < ema50_4h < ema200_4h:
            trend_4h = "强烈下降"
        elif close_4h < ema20_4h < ema50_4h:
            trend_4h = "下降"
        elif close_4h < ema20_4h:
            trend_4h = "弱下降"
        else:
            trend_4h = "盘整"
        
        # 趋势强度
        adx_1h = df_1h['adx'].iloc[-1]
        adx_4h = df_4h['adx'].iloc[-1]
        
        if adx_1h >= 30:
            strength_1h = "强"
        elif adx_1h >= 20:
            strength_1h = "中"
        else:
            strength_1h = "弱"
            
        if adx_4h >= 30:
            strength_4h = "强"
        elif adx_4h >= 20:
            strength_4h = "中"
        else:
            strength_4h = "弱"
        
        # 趋势一致性
        if (trend_1h.endswith("上升") and trend_4h.endswith("上升")) or \
           (trend_1h.endswith("下降") and trend_4h.endswith("下降")):
            consistency = "高"
        elif trend_1h == "盘整" or trend_4h == "盘整":
            consistency = "中"
        else:
            consistency = "低"
        
        # 趋势动量
        macd_1h = df_1h['macd'].iloc[-1]
        macd_signal_1h = df_1h['macd_signal'].iloc[-1]
        macd_hist_1h = df_1h['macd_hist'].iloc[-1]
        
        macd_4h = df_4h['macd'].iloc[-1]
        macd_signal_4h = df_4h['macd_signal'].iloc[-1]
        macd_hist_4h = df_4h['macd_hist'].iloc[-1]
        
        if macd_1h > macd_signal_1h and macd_hist_1h > 0 and macd_hist_1h > df_1h['macd_hist'].iloc[-2]:
            momentum_1h = "上升"
        elif macd_1h < macd_signal_1h and macd_hist_1h < 0 and macd_hist_1h < df_1h['macd_hist'].iloc[-2]:
            momentum_1h = "下降"
        else:
            momentum_1h = "中性"
            
        if macd_4h > macd_signal_4h and macd_hist_4h > 0 and macd_hist_4h > df_4h['macd_hist'].iloc[-2]:
            momentum_4h = "上升"
        elif macd_4h < macd_signal_4h and macd_hist_4h < 0 and macd_hist_4h < df_4h['macd_hist'].iloc[-2]:
            momentum_4h = "下降"
        else:
            momentum_4h = "中性"
        
        # 超级趋势方向
        supertrend_direction_1h = "上升" if df_1h['supertrend_direction'].iloc[-1] == 1 else "下降"
        supertrend_direction_4h = "上升" if df_4h['supertrend_direction'].iloc[-1] == 1 else "下降"
        
        trend_analysis = {
            '1h_trend': trend_1h,
            '1h_strength': strength_1h,
            '1h_momentum': momentum_1h,
            '1h_supertrend': supertrend_direction_1h,
            '4h_trend': trend_4h,
            '4h_strength': strength_4h,
            '4h_momentum': momentum_4h,
            '4h_supertrend': supertrend_direction_4h,
            'consistency': consistency,
            'overall_trend': trend_4h if consistency == "高" else "混合"
        }
        
        return trend_analysis
    
    def analyze_divergences(self, df_1h, df_4h):
        """
        分析价格与指标之间的背离
        
        参数:
        df_1h (pandas.DataFrame): 1小时数据
        df_4h (pandas.DataFrame): 4小时数据
        
        返回:
        dict: 背离分析结果
        """
        divergence_analysis = {
            '1h': {
                'rsi_bullish': False,
                'rsi_bearish': False,
                'macd_bullish': False,
                'macd_bearish': False
            },
            '4h': {
                'rsi_bullish': False,
                'rsi_bearish': False,
                'macd_bullish': False,
                'macd_bearish': False
            }
        }
        
        # 检查1小时RSI背离
        if 'rsi_divergence' in df_1h.columns and not pd.isna(df_1h['rsi_divergence'].iloc[-5:]).all():
            recent_rsi_div = df_1h['rsi_divergence'].iloc[-5:].dropna()
            if len(recent_rsi_div) > 0:
                last_div = recent_rsi_div.iloc[-1]
                if last_div == 1:
                    divergence_analysis['1h']['rsi_bullish'] = True
                elif last_div == -1:
                    divergence_analysis['1h']['rsi_bearish'] = True
        
        # 检查1小时MACD背离
        if 'macd_divergence' in df_1h.columns and not pd.isna(df_1h['macd_divergence'].iloc[-5:]).all():
            recent_macd_div = df_1h['macd_divergence'].iloc[-5:].dropna()
            if len(recent_macd_div) > 0:
                last_div = recent_macd_div.iloc[-1]
                if last_div == 1:
                    divergence_analysis['1h']['macd_bullish'] = True
                elif last_div == -1:
                    divergence_analysis['1h']['macd_bearish'] = True
        
        # 检查4小时RSI背离
        if 'rsi_divergence' in df_4h.columns and not pd.isna(df_4h['rsi_divergence'].iloc[-5:]).all():
            recent_rsi_div = df_4h['rsi_divergence'].iloc[-5:].dropna()
            if len(recent_rsi_div) > 0:
                last_div = recent_rsi_div.iloc[-1]
                if last_div == 1:
                    divergence_analysis['4h']['rsi_bullish'] = True
                elif last_div == -1:
                    divergence_analysis['4h']['rsi_bearish'] = True
        
        # 检查4小时MACD背离
        if 'macd_divergence' in df_4h.columns and not pd.isna(df_4h['macd_divergence'].iloc[-5:]).all():
            recent_macd_div = df_4h['macd_divergence'].iloc[-5:].dropna()
            if len(recent_macd_div) > 0:
                last_div = recent_macd_div.iloc[-1]
                if last_div == 1:
                    divergence_analysis['4h']['macd_bullish'] = True
                elif last_div == -1:
                    divergence_analysis['4h']['macd_bearish'] = True
        
        # 添加背离强度评估
        divergence_analysis['strength'] = "无"
        
        # 计算背离强度
        bullish_count = sum([
            divergence_analysis['1h']['rsi_bullish'],
            divergence_analysis['1h']['macd_bullish'],
            divergence_analysis['4h']['rsi_bullish'],
            divergence_analysis['4h']['macd_bullish']
        ])
        
        bearish_count = sum([
            divergence_analysis['1h']['rsi_bearish'],
            divergence_analysis['1h']['macd_bearish'],
            divergence_analysis['4h']['rsi_bearish'],
            divergence_analysis['4h']['macd_bearish']
        ])
        
        if bullish_count >= 2:
            divergence_analysis['strength'] = "强烈看涨"
        elif bullish_count == 1:
            divergence_analysis['strength'] = "轻微看涨"
        elif bearish_count >= 2:
            divergence_analysis['strength'] = "强烈看跌"
        elif bearish_count == 1:
            divergence_analysis['strength'] = "轻微看跌"
        
        return divergence_analysis
    
    def analyze_volatility(self, df_1h, df_4h):
        """
        分析价格波动性
        
        参数:
        df_1h (pandas.DataFrame): 1小时数据
        df_4h (pandas.DataFrame): 4小时数据
        
        返回:
        dict: 波动性分析结果
        """
        volatility_analysis = {}
        
        # 1小时ATR
        atr_1h = df_1h['atr'].iloc[-1]
        atr_percent_1h = df_1h['atr_percent'].iloc[-1]
        
        # 4小时ATR
        atr_4h = df_4h['atr'].iloc[-1]
        atr_percent_4h = df_4h['atr_percent'].iloc[-1]
        
        # 波动性趋势
        atr_trend_1h = "上升" if df_1h['atr'].iloc[-1] > df_1h['atr'].iloc[-5:].mean() else "下降"
        atr_trend_4h = "上升" if df_4h['atr'].iloc[-1] > df_4h['atr'].iloc[-5:].mean() else "下降"
        
        # 布林带宽度
        bb_width_1h = df_1h['bb_width'].iloc[-1]
        bb_width_4h = df_4h['bb_width'].iloc[-1]
        
        # 布林带宽度趋势
        bb_width_trend_1h = "扩张" if df_1h['bb_width'].iloc[-1] > df_1h['bb_width'].iloc[-5:].mean() else "收缩"
        bb_width_trend_4h = "扩张" if df_4h['bb_width'].iloc[-1] > df_4h['bb_width'].iloc[-5:].mean() else "收缩"
        
        # 价格相对于布林带位置
        if df_1h['close'].iloc[-1] > df_1h['bb_upper'].iloc[-1]:
            bb_position_1h = "上轨之上"
        elif df_1h['close'].iloc[-1] < df_1h['bb_lower'].iloc[-1]:
            bb_position_1h = "下轨之下"
        else:
            bb_position_1h = "带内"
            
        if df_4h['close'].iloc[-1] > df_4h['bb_upper'].iloc[-1]:
            bb_position_4h = "上轨之上"
        elif df_4h['close'].iloc[-1] < df_4h['bb_lower'].iloc[-1]:
            bb_position_4h = "下轨之下"
        else:
            bb_position_4h = "带内"
        
        # 波动性评级
        if atr_percent_4h > 5:
            volatility_rating = "极高"
        elif atr_percent_4h > 3:
            volatility_rating = "高"
        elif atr_percent_4h > 1.5:
            volatility_rating = "中"
        else:
            volatility_rating = "低"
        
        # 波动性预期
        if atr_trend_4h == "上升" and bb_width_trend_4h == "扩张":
            volatility_expectation = "增加"
        elif atr_trend_4h == "下降" and bb_width_trend_4h == "收缩":
            volatility_expectation = "减少"
        else:
            volatility_expectation = "稳定"
        
        volatility_analysis = {
            '1h': {
                'atr': atr_1h,
                'atr_percent': atr_percent_1h,
                'atr_trend': atr_trend_1h,
                'bb_width': bb_width_1h,
                'bb_width_trend': bb_width_trend_1h,
                'bb_position': bb_position_1h
            },
            '4h': {
                'atr': atr_4h,
                'atr_percent': atr_percent_4h,
                'atr_trend': atr_trend_4h,
                'bb_width': bb_width_4h,
                'bb_width_trend': bb_width_trend_4h,
                'bb_position': bb_position_4h
            },
            'rating': volatility_rating,
            'expectation': volatility_expectation
        }
        
        return volatility_analysis
    
    def analyze_market_structure(self, df_1h, df_4h):
        """
        分析市场结构
        
        参数:
        df_1h (pandas.DataFrame): 1小时数据
        df_4h (pandas.DataFrame): 4小时数据
        
        返回:
        dict: 市场结构分析结果
        """
        market_structure = {}
        
        # 检查1小时市场结构
        higher_highs_1h = False
        higher_lows_1h = False
        lower_highs_1h = False
        lower_lows_1h = False
        
        if 'higher_high' in df_1h.columns:
            higher_highs_1h = df_1h['higher_high'].iloc[-10:].sum() > 0
        if 'higher_low' in df_1h.columns:
            higher_lows_1h = df_1h['higher_low'].iloc[-10:].sum() > 0
        if 'lower_high' in df_1h.columns:
            lower_highs_1h = df_1h['lower_high'].iloc[-10:].sum() > 0
        if 'lower_low' in df_1h.columns:
            lower_lows_1h = df_1h['lower_low'].iloc[-10:].sum() > 0
        
        if higher_highs_1h and higher_lows_1h:
            structure_1h = "上升趋势"
        elif lower_highs_1h and lower_lows_1h:
            structure_1h = "下降趋势"
        elif higher_lows_1h and lower_highs_1h:
            structure_1h = "收敛三角形"
        elif higher_highs_1h and lower_lows_1h:
            structure_1h = "发散三角形"
        else:
            structure_1h = "盘整"
        
        # 检查4小时市场结构
        higher_highs_4h = False
        higher_lows_4h = False
        lower_highs_4h = False
        lower_lows_4h = False
        
        if 'higher_high' in df_4h.columns:
            higher_highs_4h = df_4h['higher_high'].iloc[-10:].sum() > 0
        if 'higher_low' in df_4h.columns:
            higher_lows_4h = df_4h['higher_low'].iloc[-10:].sum() > 0
        if 'lower_high' in df_4h.columns:
            lower_highs_4h = df_4h['lower_high'].iloc[-10:].sum() > 0
        if 'lower_low' in df_4h.columns:
            lower_lows_4h = df_4h['lower_low'].iloc[-10:].sum() > 0
        
        if higher_highs_4h and higher_lows_4h:
            structure_4h = "上升趋势"
        elif lower_highs_4h and lower_lows_4h:
            structure_4h = "下降趋势"
        elif higher_lows_4h and lower_highs_4h:
            structure_4h = "收敛三角形"
        elif higher_highs_4h and lower_lows_4h:
            structure_4h = "发散三角形"
        else:
            structure_4h = "盘整"
        
        # 价格位置
        close_1h = df_1h['close'].iloc[-1]
        high_20_1h = df_1h['high'].iloc[-20:].max()
        low_20_1h = df_1h['low'].iloc[-20:].min()
        range_1h = high_20_1h - low_20_1h
        
        if range_1h > 0:
            position_in_range_1h = (close_1h - low_20_1h) / range_1h
        else:
            position_in_range_1h = 0.5
        
        close_4h = df_4h['close'].iloc[-1]
        high_20_4h = df_4h['high'].iloc[-20:].max()
        low_20_4h = df_4h['low'].iloc[-20:].min()
        range_4h = high_20_4h - low_20_4h
        
        if range_4h > 0:
            position_in_range_4h = (close_4h - low_20_4h) / range_4h
        else:
            position_in_range_4h = 0.5
        
        # 价格位置描述
        if position_in_range_1h > 0.8:
            position_desc_1h = "接近区间顶部"
        elif position_in_range_1h < 0.2:
            position_desc_1h = "接近区间底部"
        else:
            position_desc_1h = "区间中部"
            
        if position_in_range_4h > 0.8:
            position_desc_4h = "接近区间顶部"
        elif position_in_range_4h < 0.2:
            position_desc_4h = "接近区间底部"
        else:
            position_desc_4h = "区间中部"
        
        market_structure = {
            '1h': {
                'structure': structure_1h,
                'higher_highs': higher_highs_1h,
                'higher_lows': higher_lows_1h,
                'lower_highs': lower_highs_1h,
                'lower_lows': lower_lows_1h,
                'position_in_range': position_in_range_1h,
                'position_description': position_desc_1h
            },
            '4h': {
                'structure': structure_4h,
                'higher_highs': higher_highs_4h,
                'higher_lows': higher_lows_4h,
                'lower_highs': lower_highs_4h,
                'lower_lows': lower_lows_4h,
                'position_in_range': position_in_range_4h,
                'position_description': position_desc_4h
            }
        }
        
        return market_structure
    
    def generate_trading_signals(self, df_1h, df_4h, analysis_data):
        """
        生成交易信号
        
        参数:
        df_1h (pandas.DataFrame): 1小时数据
        df_4h (pandas.DataFrame): 4小时数据
        analysis_data (dict): 分析数据
        
        返回:
        dict: 交易信号
        """
        signals = {
            'long': {
                'strength': 0,
                'reasons': []
            },
            'short': {
                'strength': 0,
                'reasons': []
            },
            'recommendation': 'WAIT',
            'confidence': 0.0
        }
        
        # 获取当前价格
        current_price = analysis_data['current_price']
        
        # 趋势信号
        trend_analysis = analysis_data['trend_analysis']
        
        # 多头趋势信号
        if trend_analysis['4h_trend'].endswith('上升'):
            signals['long']['strength'] += 2
            signals['long']['reasons'].append(f"4小时趋势{trend_analysis['4h_trend']}")
        elif trend_analysis['4h_trend'].endswith('下降'):
            signals['short']['strength'] += 2
            signals['short']['reasons'].append(f"4小时趋势{trend_analysis['4h_trend']}")
            
        if trend_analysis['1h_trend'].endswith('上升'):
            signals['long']['strength'] += 1
            signals['long']['reasons'].append(f"1小时趋势{trend_analysis['1h_trend']}")
        elif trend_analysis['1h_trend'].endswith('下降'):
            signals['short']['strength'] += 1
            signals['short']['reasons'].append(f"1小时趋势{trend_analysis['1h_trend']}")
        
        # 趋势一致性
        if trend_analysis['consistency'] == '高':
            if trend_analysis['4h_trend'].endswith('上升'):
                signals['long']['strength'] += 1
                signals['long']['reasons'].append("趋势一致性高")
            elif trend_analysis['4h_trend'].endswith('下降'):
                signals['short']['strength'] += 1
                signals['short']['reasons'].append("趋势一致性高")
        
        # 超级趋势信号
        if trend_analysis['4h_supertrend'] == '上升':
            signals['long']['strength'] += 1
            signals['long']['reasons'].append("4小时超级趋势看涨")
        elif trend_analysis['4h_supertrend'] == '下降':
            signals['short']['strength'] += 1
            signals['short']['reasons'].append("4小时超级趋势看跌")
            
        if trend_analysis['1h_supertrend'] == '上升':
            signals['long']['strength'] += 0.5
            signals['long']['reasons'].append("1小时超级趋势看涨")
        elif trend_analysis['1h_supertrend'] == '下降':
            signals['short']['strength'] += 0.5
            signals['short']['reasons'].append("1小时超级趋势看跌")
        
        # 动量信号
        if trend_analysis['4h_momentum'] == '上升':
            signals['long']['strength'] += 1
            signals['long']['reasons'].append("4小时动量上升")
        elif trend_analysis['4h_momentum'] == '下降':
            signals['short']['strength'] += 1
            signals['short']['reasons'].append("4小时动量下降")
            
        if trend_analysis['1h_momentum'] == '上升':
            signals['long']['strength'] += 0.5
            signals['long']['reasons'].append("1小时动量上升")
        elif trend_analysis['1h_momentum'] == '下降':
            signals['short']['strength'] += 0.5
            signals['short']['reasons'].append("1小时动量下降")
        
        # 背离信号
        divergence_analysis = analysis_data['divergence_analysis']
        
        if divergence_analysis['4h']['rsi_bullish'] or divergence_analysis['4h']['macd_bullish']:
            signals['long']['strength'] += 1.5
            signals['long']['reasons'].append("4小时正背离")
        elif divergence_analysis['4h']['rsi_bearish'] or divergence_analysis['4h']['macd_bearish']:
            signals['short']['strength'] += 1.5
            signals['short']['reasons'].append("4小时负背离")
            
        if divergence_analysis['1h']['rsi_bullish'] or divergence_analysis['1h']['macd_bullish']:
            signals['long']['strength'] += 0.5
            signals['long']['reasons'].append("1小时正背离")
        elif divergence_analysis['1h']['rsi_bearish'] or divergence_analysis['1h']['macd_bearish']:
            signals['short']['strength'] += 0.5
            signals['short']['reasons'].append("1小时负背离")
        
        # 波动性信号
        volatility_analysis = analysis_data['volatility_analysis']
        
        if volatility_analysis['4h']['bb_position'] == '下轨之下':
            signals['long']['strength'] += 1
            signals['long']['reasons'].append("价格位于4小时布林带下轨之下")
        elif volatility_analysis['4h']['bb_position'] == '上轨之上':
            signals['short']['strength'] += 1
            signals['short']['reasons'].append("价格位于4小时布林带上轨之上")
            
        if volatility_analysis['1h']['bb_position'] == '下轨之下':
            signals['long']['strength'] += 0.5
            signals['long']['reasons'].append("价格位于1小时布林带下轨之下")
        elif volatility_analysis['1h']['bb_position'] == '上轨之上':
            signals['short']['strength'] += 0.5
            signals['short']['reasons'].append("价格位于1小时布林带上轨之上")
        
        # 市场结构信号
        market_structure = analysis_data['market_structure']
        
        if market_structure['4h']['structure'] == '上升趋势':
            signals['long']['strength'] += 1
            signals['long']['reasons'].append("4小时市场结构为上升趋势")
        elif market_structure['4h']['structure'] == '下降趋势':
            signals['short']['strength'] += 1
            signals['short']['reasons'].append("4小时市场结构为下降趋势")
            
        if market_structure['1h']['structure'] == '上升趋势':
            signals['long']['strength'] += 0.5
            signals['long']['reasons'].append("1小时市场结构为上升趋势")
        elif market_structure['1h']['structure'] == '下降趋势':
            signals['short']['strength'] += 0.5
            signals['short']['reasons'].append("1小时市场结构为下降趋势")
        
        # 支撑阻力信号
        supports_1h = analysis_data['1h_data']['support_levels']
        resistances_1h = analysis_data['1h_data']['resistance_levels']
        
        # 检查价格是否接近支撑位
        if supports_1h and min(supports_1h, key=lambda x: abs(x - current_price)) / current_price > 0.98:
            signals['long']['strength'] += 1
            signals['long']['reasons'].append("价格接近1小时支撑位")
        
        # 检查价格是否接近阻力位
        if resistances_1h and min(resistances_1h, key=lambda x: abs(x - current_price)) / current_price < 1.02:
            signals['short']['strength'] += 1
            signals['short']['reasons'].append("价格接近1小时阻力位")
        
        # 技术指标信号
        # RSI信号
        rsi_14_1h = analysis_data['1h_data']['rsi']['14']
        rsi_14_4h = analysis_data['4h_data']['rsi']['14']
        
        if rsi_14_4h < 30:
            signals['long']['strength'] += 1
            signals['long']['reasons'].append(f"4小时RSI超卖({rsi_14_4h:.1f})")
        elif rsi_14_4h > 70:
            signals['short']['strength'] += 1
            signals['short']['reasons'].append(f"4小时RSI超买({rsi_14_4h:.1f})")
            
        if rsi_14_1h < 30:
            signals['long']['strength'] += 0.5
            signals['long']['reasons'].append(f"1小时RSI超卖({rsi_14_1h:.1f})")
        elif rsi_14_1h > 70:
            signals['short']['strength'] += 0.5
            signals['short']['reasons'].append(f"1小时RSI超买({rsi_14_1h:.1f})")
        
        # 随机指标信号
        stoch_k_1h = analysis_data['1h_data']['stochastic']['k']
        stoch_d_1h = analysis_data['1h_data']['stochastic']['d']
        stoch_k_4h = analysis_data['4h_data']['stochastic']['k']
        stoch_d_4h = analysis_data['4h_data']['stochastic']['d']
        
        if stoch_k_4h < 20 and stoch_k_4h > stoch_d_4h:
            signals['long']['strength'] += 1
            signals['long']['reasons'].append("4小时随机指标超卖区金叉")
        elif stoch_k_4h > 80 and stoch_k_4h < stoch_d_4h:
            signals['short']['strength'] += 1
            signals['short']['reasons'].append("4小时随机指标超买区死叉")
            
        if stoch_k_1h < 20 and stoch_k_1h > stoch_d_1h:
            signals['long']['strength'] += 0.5
            signals['long']['reasons'].append("1小时随机指标超卖区金叉")
        elif stoch_k_1h > 80 and stoch_k_1h < stoch_d_1h:
            signals['short']['strength'] += 0.5
            signals['short']['reasons'].append("1小时随机指标超买区死叉")
        
        # 综合技术评分
        technical_score = analysis_data['technical_score']
        
        if technical_score > 50:
            signals['long']['strength'] += 1
            signals['long']['reasons'].append(f"综合技术评分看涨({technical_score:.1f})")
        elif technical_score < -50:
            signals['short']['strength'] += 1
            signals['short']['reasons'].append(f"综合技术评分看跌({technical_score:.1f})")
        
        # 确定最终推荐
        long_strength = signals['long']['strength']
        short_strength = signals['short']['strength']
        
        # 计算信心分数（0-1范围）
        max_strength = 10  # 假设的最大强度值
        
        if long_strength > short_strength:
            signals['confidence'] = min(long_strength / max_strength, 0.95)
            
            if signals['confidence'] > 0.7:
                signals['recommendation'] = 'OPEN_LONG'
            elif signals['confidence'] > 0.5:
                signals['recommendation'] = 'WAIT_LONG'  # 等待更好的多头入场点
            else:
                signals['recommendation'] = 'WAIT'
        elif short_strength > long_strength:
            signals['confidence'] = min(short_strength / max_strength, 0.95)
            
            if signals['confidence'] > 0.7:
                signals['recommendation'] = 'OPEN_SHORT'
            elif signals['confidence'] > 0.5:
                signals['recommendation'] = 'WAIT_SHORT'  # 等待更好的空头入场点
            else:
                signals['recommendation'] = 'WAIT'
        else:
            signals['confidence'] = 0.3
            signals['recommendation'] = 'WAIT'
        
        # 添加止损和目标价格
        if signals['recommendation'] == 'OPEN_LONG':
            # 为多头设置止损和目标
            supports = analysis_data['1h_data']['support_levels']
            resistances = analysis_data['1h_data']['resistance_levels']
            atr = analysis_data['1h_data']['atr']
            
            # 止损：使用最近的支撑位或ATR的1.5倍
            if supports and supports[0] < current_price:
                stop_loss = supports[0] * 0.995  # 略低于支撑位
            else:
                stop_loss = current_price - (1.5 * atr)
            
            # 目标：使用最近的阻力位或ATR的2-3倍
            if resistances and resistances[0] > current_price:
                target1 = resistances[0]
                if len(resistances) > 1:
                    target2 = resistances[1]
                else:
                    target2 = current_price + (3 * atr)
            else:
                target1 = current_price + (2 * atr)
                target2 = current_price + (3 * atr)
            
            signals['stop_loss'] = stop_loss
            signals['targets'] = [target1, target2]
            
        elif signals['recommendation'] == 'OPEN_SHORT':
            # 为空头设置止损和目标
            supports = analysis_data['1h_data']['support_levels']
            resistances = analysis_data['1h_data']['resistance_levels']
            atr = analysis_data['1h_data']['atr']
            
            # 止损：使用最近的阻力位或ATR的1.5倍
            if resistances and resistances[0] > current_price:
                stop_loss = resistances[0] * 1.005  # 略高于阻力位
            else:
                stop_loss = current_price + (1.5 * atr)
            
            # 目标：使用最近的支撑位或ATR的2-3倍
            if supports and supports[0] < current_price:
                target1 = supports[0]
                if len(supports) > 1:
                    target2 = supports[1]
                else:
                    target2 = current_price - (3 * atr)
            else:
                target1 = current_price - (2 * atr)
                target2 = current_price - (3 * atr)
            
            signals['stop_loss'] = stop_loss
            signals['targets'] = [target1, target2]
        
        return signals
    
    def analyze_multiple_symbols(self, symbols):
        """
        并行分析多个交易对
        
        参数:
        symbols (list): 交易对符号列表
        
        返回:
        dict: 分析结果字典
        """
        results = {}
        
        # 使用线程池并行处理
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有分析任务
            future_to_symbol = {executor.submit(self.analyze_single_symbol, symbol): symbol for symbol in symbols}
            
            # 收集结果
            for future in future_to_symbol:
                symbol = future_to_symbol[future]
                try:
                    result = future.result()
                    if result:
                        results[symbol] = result
                except Exception as e:
                    print(f"分析{symbol}时出错: {e}")
        
        self.analysis_results = results
        return results
    
    def calculate_correlation_matrix(self, timeframe='4h', period=30):
        """
        计算币种之间的相关性矩阵
        
        参数:
        timeframe (str): 时间框架
        period (int): 计算相关性的周期
        
        返回:
        pandas.DataFrame: 相关性矩阵
        """
        if not self.analysis_results:
            print("没有分析结果，无法计算相关性矩阵")
            return None
        
        symbols = list(self.analysis_results.keys())
        price_data = {}
        
        # 获取每个币种的收盘价
        for symbol in symbols:
            df = self.fetch_ohlcv_data(symbol, timeframe, limit=period)
            if df is not None:
                price_data[symbol] = df['close']
        
        # 创建价格DataFrame
        price_df = pd.DataFrame(price_data)
        
        # 计算相关性矩阵
        correlation_matrix = price_df.corr(method='pearson')
        
        self.correlation_matrix = correlation_matrix
        return correlation_matrix
    
    def analyze_market_state(self):
        """
        分析整体市场状态
        
        返回:
        dict: 市场状态分析
        """
        if not self.analysis_results:
            print("没有分析结果，无法分析市场状态")
            return None
        
        # 计算平均技术评分
        avg_technical_score = np.mean([data['technical_score'] for data in self.analysis_results.values()])
        
        # 计算趋势一致性
        trend_counts = {
            '上升': 0,
            '下降': 0,
            '盘整': 0
        }
        
        for data in self.analysis_results.values():
            trend = data['trend_analysis']['overall_trend']
            if '上升' in trend:
                trend_counts['上升'] += 1
            elif '下降' in trend:
                trend_counts['下降'] += 1
            else:
                trend_counts['盘整'] += 1
        
        total_coins = len(self.analysis_results)
        trend_percentages = {k: v / total_coins * 100 for k, v in trend_counts.items()}
        
        # 确定主导趋势
        dominant_trend = max(trend_percentages.items(), key=lambda x: x[1])
        
        # 计算平均波动性
        avg_volatility = np.mean([data['volatility_analysis']['4h']['atr_percent'] for data in self.analysis_results.values()])
        
        # 确定市场状态
        if avg_technical_score > 30 and dominant_trend[0] == '上升' and dominant_trend[1] > 60:
            market_state = "强烈看涨"
        elif avg_technical_score > 10 and dominant_trend[0] == '上升' and dominant_trend[1] > 50:
            market_state = "看涨"
        elif avg_technical_score < -30 and dominant_trend[0] == '下降' and dominant_trend[1] > 60:
            market_state = "强烈看跌"
        elif avg_technical_score < -10 and dominant_trend[0] == '下降' and dominant_trend[1] > 50:
            market_state = "看跌"
        elif trend_percentages['盘整'] > 50 or abs(avg_technical_score) < 10:
            market_state = "盘整"
        else:
            market_state = "混合"
        
        # 波动性状态
        if avg_volatility > 4:
            volatility_state = "高波动"
        elif avg_volatility > 2:
            volatility_state = "中等波动"
        else:
            volatility_state = "低波动"
        
        self.market_state = {
            'state': market_state,
            'avg_technical_score': avg_technical_score,
            'trend_distribution': trend_percentages,
            'dominant_trend': dominant_trend[0],
            'volatility_state': volatility_state,
            'avg_volatility': avg_volatility
        }
        
        return self.market_state
    
    def rank_by_volatility(self):
        """
        按波动性对币种进行排名
        
        返回:
        list: 按波动性排序的币种列表
        """
        if not self.analysis_results:
            print("没有分析结果，无法按波动性排名")
            return []
        
        volatility_data = []
        
        for symbol, data in self.analysis_results.items():
            volatility_data.append({
                'symbol': symbol,
                'atr_percent': data['volatility_analysis']['4h']['atr_percent'],
                'bb_width': data['volatility_analysis']['4h']['bb_width'],
                'volatility_rating': data['volatility_analysis']['rating']
            })
        
        # 按ATR百分比排序
        volatility_data.sort(key=lambda x: x['atr_percent'], reverse=True)
        
        self.volatility_ranking = volatility_data
        return volatility_data
    
    def rank_by_momentum(self):
        """
        按动量对币种进行排名
        
        返回:
        list: 按动量排序的币种列表
        """
        if not self.analysis_results:
            print("没有分析结果，无法按动量排名")
            return []
        
        momentum_data = []
        
        for symbol, data in self.analysis_results.items():
            momentum_data.append({
                'symbol': symbol,
                'technical_score': data['technical_score'],
                'rsi_14': data['4h_data']['rsi']['14'],
                'macd_hist': data['4h_data']['macd']['histogram'],
                'price_change_24h': data['market_data']['price_change_24h'] if data['market_data']['price_change_24h'] else 0
            })
        
        # 按技术评分排序
        momentum_data.sort(key=lambda x: abs(x['technical_score']), reverse=True)
        
        self.momentum_ranking = momentum_data
        return momentum_data
    
    def get_account_info(self):
        """
        获取账户信息
        
        返回:
        dict: 包含账户总值和可用保证金的字典
        """
        try:
            balance = self.exchange.fetch_balance()
            return {
                'total_value': balance['total']['USDT'],
                'free_margin': balance['free']['USDT']
            }
        except Exception as e:
            print(f"获取账户信息时出错: {e}")
            return None
    
    def get_positions(self, symbols):
        """
        获取当前持仓
        
        返回:
        list: 持仓列表
        """
        try:
            target_symbols = {coin: f"{coin}/USDT:USDT" for coin in symbols}
            positions = self.exchange.fetch_positions(symbols=[ f"{s}-USDT-SWAP" for s in symbols], params={"instType": "SWAP"})
            
            active_positions = []
            
            for position in positions:
                orders = self.exchange.fetch_open_orders(position['symbol'], params={"ordType": "conditional"})
                open_order = orders[0] if orders else {}
                order_info = open_order.get("info", {}) if isinstance(open_order, dict) else {}
                sl = order_info.get("slTriggerPx")
                tp = order_info.get("tpTriggerPx")
                algo_id = open_order.get("id") if isinstance(open_order, dict) else None
                algo_amount = open_order.get("amount") if isinstance(open_order, dict) else 0.0
                info = position.get('info')
                if info and float(position['contracts']) > 0:
                    active_positions.append({
                        'symbol': position['symbol'],
                        'side': position['side'],
                        'entry_price': float(position['entryPrice']),
                        'amount': float(position['contracts']),
                        'leverage': float(position['leverage']),
                        'unrealized_pnl': float(position['unrealizedPnl']),
                        'liquidation_price': float(position['liquidationPrice']) if 'liquidationPrice' in position and position['liquidationPrice'] else None,
                        "tp": None if tp == '' else tp,
                        "sl": None if sl == '' else sl,
                        "algoId": algo_id,
                        "algoAmount": algo_amount,
                    })
            
            return active_positions
        except Exception as e:
            print(f"获取持仓信息时出错: {e}")
            return None
    
    def calculate_position_size(self, account_value, risk_percentage, entry_price, stop_loss, leverage=1):
        """
        计算仓位大小
        
        参数:
        account_value (float): 账户总值
        risk_percentage (float): 风险百分比（如2%则为0.02）
        entry_price (float): 入场价格
        stop_loss (float): 止损价格
        leverage (float): 杠杆倍数
        
        返回:
        float: 建议的仓位大小（以基础货币计）
        """
        # 计算风险金额
        risk_amount = account_value * risk_percentage
        
        # 计算每单位的风险
        price_risk = abs(entry_price - stop_loss)
        
        # 如果止损距离为0，返回0（无效输入）
        if price_risk == 0:
            return 0
        
        # 计算仓位大小（考虑杠杆）
        position_size = (risk_amount / price_risk) * leverage
        
        return position_size
    
    def calculate_risk_reward_ratio(self, entry_price, stop_loss, take_profit):
        """
        计算风险回报比
        
        参数:
        entry_price (float): 入场价格
        stop_loss (float): 止损价格
        take_profit (float): 获利价格
        
        返回:
        float: 风险回报比
        """
        # 计算风险和回报
        risk = abs(entry_price - stop_loss)
        reward = abs(entry_price - take_profit)
        
        # 如果风险为0，返回0（无效输入）
        if risk == 0:
            return 0
        
        # 计算风险回报比
        risk_reward_ratio = reward / risk
        
        return risk_reward_ratio
    
    def rank_trading_opportunities(self, min_score=30, max_correlation=0.7, min_risk_reward=2.0):
        """
        根据技术评分和相关性排名交易机会
        
        参数:
        min_score (float): 最小技术评分（绝对值）
        max_correlation (float): 最大允许相关性
        min_risk_reward (float): 最小风险回报比
        
        返回:
        list: 排序后的交易机会列表
        """
        if not self.analysis_results:
            print("没有分析结果，无法排名交易机会")
            return []
        
        if self.correlation_matrix is None:
            print("相关性矩阵未计算，将不考虑相关性")
        
        opportunities = []
        
        for symbol, data in self.analysis_results.items():
            # 获取交易信号
            signals = data['trading_signals']
            
            # 只考虑明确的开仓信号
            if signals['recommendation'] in ['OPEN_LONG', 'OPEN_SHORT']:
                opportunity = {
                    'symbol': symbol,
                    'action': signals['recommendation'],
                    'confidence': signals['confidence'],
                    'current_price': data['current_price'],
                    'stop_loss': signals['stop_loss'],
                    'targets': signals['targets'],
                    'reasons': signals['long']['reasons'] if signals['recommendation'] == 'OPEN_LONG' else signals['short']['reasons'],
                    'trend_1h': data['trend_analysis']['1h_trend'],
                    'trend_4h': data['trend_analysis']['4h_trend'],
                    'consistency': data['trend_analysis']['consistency'],
                    'technical_score': data['technical_score']
                }
                
                # 计算风险回报比
                risk_reward = self.calculate_risk_reward_ratio(
                    data['current_price'], 
                    signals['stop_loss'], 
                    signals['targets'][0]  # 使用第一个目标
                )
                
                opportunity['risk_reward'] = risk_reward
                
                # 只添加风险回报比合理的机会
                if risk_reward >= min_risk_reward and signals['confidence'] >= 0.6:
                    opportunities.append(opportunity)
        
        # 如果有相关性矩阵，过滤高相关性的机会
        if self.correlation_matrix is not None and len(opportunities) > 1:
            filtered_opportunities = []
            selected_symbols = []
            
            # 按信心排序
            sorted_opportunities = sorted(opportunities, key=lambda x: x['confidence'], reverse=True)
            
            for opportunity in sorted_opportunities:
                symbol = opportunity['symbol']
                direction = 'long' if opportunity['action'] == 'OPEN_LONG' else 'short'
                
                # 检查与已选币种的相关性
                correlated = False
                for selected in selected_symbols:
                    try:
                        correlation = abs(self.correlation_matrix.loc[symbol, selected['symbol']])
                        
                        # 如果相关性高且方向相同，则跳过
                        if correlation > max_correlation and direction == selected['direction']:
                            correlated = True
                            break
                    except KeyError:
                        # 如果相关性矩阵中没有该币种，则跳过相关性检查
                        pass
                
                if not correlated:
                    filtered_opportunities.append(opportunity)
                    selected_symbols.append({'symbol': symbol, 'direction': direction})
            
            opportunities = filtered_opportunities
        
        # 按信心排序
        opportunities.sort(key=lambda x: x['confidence'], reverse=True)
        
        return opportunities
    
    def suggest_portfolio_allocation(self, opportunities, max_positions=5, max_allocation=0.8):
        """
        建议投资组合资金分配
        
        参数:
        opportunities (list): 交易机会列表
        max_positions (int): 最大持仓数量
        max_allocation (float): 最大资金分配比例（0-1）
        
        返回:
        dict: 资金分配建议
        """
        if not opportunities:
            return {
                'allocations': [],
                'total_allocation': 0,
                'reserve': 1.0
            }
        
        # 限制持仓数量
        top_opportunities = opportunities[:max_positions]
        
        # 计算总信心分数
        total_confidence = sum(opp['confidence'] for opp in top_opportunities)
        
        # 计算每个机会的资金分配比例
        allocations = []
        total_allocation = 0
        
        for opp in top_opportunities:
            # 根据信心分数分配资金
            weight = opp['confidence'] / total_confidence
            
            # 调整权重，使总分配不超过最大分配比例
            allocation = weight * max_allocation
            
            # 添加到分配列表
            allocations.append({
                'symbol': opp['symbol'],
                'action': opp['action'],
                'allocation': allocation,
                'confidence': opp['confidence'],
                'risk_reward': opp['risk_reward']
            })
            
            total_allocation += allocation
        
        # 计算保留资金
        reserve = 1.0 - total_allocation
        
        return {
            'allocations': allocations,
            'total_allocation': total_allocation,
            'reserve': reserve
        }
    
    def generate_multi_coin_analysis_prompt(self, symbols, include_correlation=True, max_retries=50, logger=None):
        """
        生成多币种分析提示
        
        参数:
        symbols (list): 交易对符号列表
        include_correlation (bool): 是否包含相关性分析
        
        返回:
        str: 格式化的分析提示
        """
        
        for attempt in range(max_retries):
            try:
                # 分析指定的币种
                self.analyze_multiple_symbols(symbols)
                
                # 如果需要，计算相关性矩阵
                if include_correlation:
                    self.calculate_correlation_matrix()
                
                # 分析市场状态
                market_state = self.analyze_market_state()
                
                # 获取波动性排名
                volatility_ranking = self.rank_by_volatility()
                
                # 获取动量排名
                momentum_ranking = self.rank_by_momentum()
                
                # 获取交易机会排名
                opportunities = self.rank_trading_opportunities()
                
                # 建议投资组合分配
                portfolio_allocation = self.suggest_portfolio_allocation(opportunities)
                
                # 获取账户信息
                account_info = self.get_account_info()
                positions = self.get_positions(symbols)
                
                # 生成提示
                prompt = f"""
                # 用户提示词：多币种加密货币合约交易分析

                请根据以下市场数据分析多个加密货币的市场状况，并提供基于1小时和4小时时间框架的合约交易建议。我需要一个全面的市场分析和明确的交易决策。

                ## 分析时间
                - 当前时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

                ## 市场整体状态
                - 市场状态：{market_state['state']}
                - 平均技术评分：{market_state['avg_technical_score']:.2f}
                - 主导趋势：{market_state['dominant_trend']} ({market_state['trend_distribution'][market_state['dominant_trend']]:.1f}%)
                - 波动性状态：{market_state['volatility_state']} (平均ATR百分比: {market_state['avg_volatility']:.2f}%)

                ## 账户信息
                """

                if account_info:
                    prompt += f"""- 账户总值：{account_info['total_value']} USDT
                    - 可用保证金：{account_info['free_margin']} USDT
                    """
                else:
                    prompt += "- 账户信息不可用\n"

                # 添加持仓信息
                prompt += f"""
                ## 当前持仓
                """
                if positions and len(positions) > 0:
                    for pos in positions:
                        prompt += f"""
                        - 币种：{pos['symbol'].split('/')[0]}
                        * 方向：{'多单' if pos['side'] == 'long' else '空单'}
                        * 开仓价格：{pos['entry_price']}
                        * 当前盈亏：{pos['unrealized_pnl']} USDT
                        * 杠杆倍数：{pos['leverage']}
                        * 清算价格：{pos['liquidation_price']}
                        """
                else:
                    prompt += "- 当前无持仓\n"

                # 添加相关性矩阵信息
                if include_correlation and self.correlation_matrix is not None:
                    prompt += f"""
                    ## 币种相关性分析
                    以下是主要币种之间的相关性（1表示完全正相关，-1表示完全负相关）：
                    
                    
                    """
                    
                    # 格式化相关性矩阵（选择主要币种）
                    main_coins = [s for s in symbols if 'BTC' in s or 'ETH' in s][:5]  # 限制数量
                    if main_coins:
                        matrix = self.correlation_matrix.loc[main_coins, main_coins].round(2)
                        prompt += f"""
                        {matrix.to_string()}
                        
                        
                        """
                    
                    prompt += f"""
                    交易建议应考虑币种相关性，避免同时持有高度相关的同向头寸。
                    """

                # 添加波动性排名
                prompt += f"""
                ## 波动性排名（前5名）
                """
                if volatility_ranking:
                    for i, coin in enumerate(volatility_ranking[:5], 1):
                        prompt += f"""
                        {i}. {coin['symbol']} - ATR百分比: {coin['atr_percent']:.2f}%, 波动性评级: {coin['volatility_rating']}
                        """
                else:
                    prompt += f"""
                    波动性排名不可用
                    """

                # 添加动量排名
                prompt += f"""
                ## 动量排名（前5名）
                """
                if momentum_ranking:
                    for i, coin in enumerate(momentum_ranking[:5], 1):
                        prompt += f"""
                        {i}. {coin['symbol']} - 技术评分: {coin['technical_score']:.2f}, RSI: {coin['rsi_14']:.2f}, 24h变化: {coin['price_change_24h']:.2f}%
                        """
                else:
                    prompt += f"""
                    动量排名不可用
                    """

                # 添加交易机会排名
                prompt += f"""
                ## 交易机会排名
                """
                if opportunities:
                    prompt += f"""
                    根据技术分析评分、趋势一致性和风险回报比，以下是排名靠前的交易机会：
                    
                    """
                    
                    for i, opp in enumerate(opportunities[:5], 1):  # 限制显示前5个
                        prompt += f"""
                        {i}. {opp['symbol']} - {opp['action']} (信心: {opp['confidence']:.2f})
                        """
                        prompt += f"""
                            * 当前价格: {opp['current_price']}
                        """
                        prompt += f"""   
                            * 止损价格: {opp['stop_loss']}
                        """
                        prompt += f"""
                            * 目标价格: {', '.join([str(t) for t in opp['targets']])}
                        """
                        prompt += f"""   
                            * 风险回报比: 1:{opp['risk_reward']:.2f}
                        """
                        prompt += f"""   
                            * 1小时趋势: {opp['trend_1h']}
                        """
                        prompt += f"""   
                            * 4小时趋势: {opp['trend_4h']}
                        """
                        prompt += f"""   
                            * 趋势一致性: {opp['consistency']}
                        """
                        prompt += f"""   
                        * 信号原因: {', '.join(opp['reasons'][:3])}
                        
                        """
                else:
                    prompt += f"""
                    当前没有符合条件的高质量交易机会。
                    """

                # 添加投资组合分配建议
                prompt += f"""
                ## 投资组合分配建议
                """
                if portfolio_allocation['allocations']:
                    prompt += f"""
                    建议总资金分配: {portfolio_allocation['total_allocation']*100:.1f}%，保留资金: {portfolio_allocation['reserve']*100:.1f}%
                    
                    """
                    
                    for alloc in portfolio_allocation['allocations']:
                        prompt += f"""
                        - {alloc['symbol']} ({alloc['action']}): 账户资金的 {alloc['allocation']*100:.1f}%
                        """
                else:
                    prompt += f"""
                    当前没有推荐的资金分配。
                    """

                # 为每个币种添加详细数据
                prompt += f"""
                ## 各币种详细数据
                """
                
                for symbol in symbols:
                    if symbol in self.analysis_results:
                        data = self.analysis_results[symbol]
                        
                        prompt += f"""
                        ### {symbol}
                        """
                        prompt += f"""
                        - 当前价格：{data['current_price']}
                        """
                        prompt += f"""
                        - 技术评分：{data['technical_score']:.1f} (正值=看涨，负值=看跌)
                        """
                        prompt += f"""
                        - 交易信号：{data['trading_signals']['recommendation']} (信心: {data['trading_signals']['confidence']:.2f})
                        """
                        prompt += f"""
                        - 1小时趋势：{data['trend_analysis']['1h_trend']} ({data['trend_analysis']['1h_strength']})
                        """
                        prompt += f"""
                        - 4小时趋势：{data['trend_analysis']['4h_trend']} ({data['trend_analysis']['4h_strength']})
                        """
                        prompt += f"""
                        - 趋势一致性：{data['trend_analysis']['consistency']}
                        """
                        
                        # 添加背离信息
                        div_analysis = data['divergence_analysis']
                        if div_analysis['strength'] != "无":
                            prompt += f"""
                            - 背离：{div_analysis['strength']}
                            """
                        
                        # 添加波动性信息
                        vol_analysis = data['volatility_analysis']
                        prompt += f"""
                        - 波动性：{vol_analysis['rating']} (ATR百分比: {vol_analysis['4h']['atr_percent']:.2f}%)
                        """
                        
                        # 添加市场结构
                        market_struct = data['market_structure']
                        prompt += f"""
                        - 市场结构：1小时={market_struct['1h']['structure']}, 4小时={market_struct['4h']['structure']}
                        """
                        
                        prompt += f"""
                        #### 4小时图表数据
                        """
                        prompt += f"""
                        - EMA数据：20 EMA = {data['4h_data']['ema']['20']:.2f}, 50 EMA = {data['4h_data']['ema']['50']:.2f}, 200 EMA = {data['4h_data']['ema']['200']:.2f}
                        """
                        prompt += f"""
                        - RSI(14)：{data['4h_data']['rsi']['14']:.2f}
                        """
                        prompt += f"""
                        - MACD：线 = {data['4h_data']['macd']['line']:.6f}, 信号 = {data['4h_data']['macd']['signal']:.6f}, 柱状图 = {data['4h_data']['macd']['histogram']:.6f}
                        """
                        prompt += f"""
                        - 随机指标：K = {data['4h_data']['stochastic']['k']:.2f}, D = {data['4h_data']['stochastic']['d']:.2f}
                        """
                        prompt += f"""
                        - ADX：{data['4h_data']['adx']:.2f} (DI+ = {data['4h_data']['plus_di']:.2f}, DI- = {data['4h_data']['minus_di']:.2f})
                        """
                        prompt += f"""
                        - 布林带：中轨 = {data['4h_data']['bollinger_bands']['middle']:.2f}, 宽度 = {data['4h_data']['bollinger_bands']['width']:.4f}
                        """
                        prompt += f"""
                        - 超级趋势：方向 = {data['trend_analysis']['4h_supertrend']}
                        """
                        prompt += f"""
                        - 支撑位：{data['4h_data']['support_levels']}
                        """
                        prompt += f"""
                        - 阻力位：{data['4h_data']['resistance_levels']}
                        """
                        
                        prompt += f"""
                        #### 1小时图表数据
                        """
                        prompt += f"""
                        - EMA数据：20 EMA = {data['1h_data']['ema']['20']:.2f}, 50 EMA = {data['1h_data']['ema']['50']:.2f}, 200 EMA = {data['1h_data']['ema']['200']:.2f}
                        """
                        prompt += f"""
                        - RSI(14)：{data['1h_data']['rsi']['14']:.2f}
                        """
                        prompt += f"""
                        - MACD：线 = {data['1h_data']['macd']['line']:.6f}, 信号 = {data['1h_data']['macd']['signal']:.6f}, 柱状图 = {data['1h_data']['macd']['histogram']:.6f}
                        """
                        prompt += f"""
                        - 随机指标：K = {data['1h_data']['stochastic']['k']:.2f}, D = {data['1h_data']['stochastic']['d']:.2f}
                        """
                        prompt += f"""
                        - ADX：{data['1h_data']['adx']:.2f} (DI+ = {data['1h_data']['plus_di']:.2f}, DI- = {data['1h_data']['minus_di']:.2f})
                        """
                        prompt += f"""
                        - 布林带：中轨 = {data['1h_data']['bollinger_bands']['middle']:.2f}, 宽度 = {data['1h_data']['bollinger_bands']['width']:.4f}
                        """
                        prompt += f"""
                        - 超级趋势：方向 = {data['trend_analysis']['1h_supertrend']}
                        """
                        prompt += f"""
                        - 支撑位：{data['1h_data']['support_levels']}
                        """
                        prompt += f"""
                        - 阻力位：{data['1h_data']['resistance_levels']}
                        """
                        
                        prompt += f"""
                        #### 市场数据
                        """
                        prompt += f"""
                        - 资金费率：{data['market_data']['funding_rate']}
                        """
                        prompt += f"""
                        - 24小时价格变化：{data['market_data']['price_change_24h']}%
                        """
                        prompt += f"""
                        - 24小时成交量：{data['market_data']['volume_24h']} USDT
                        """
                        if data['market_data']['open_interest']:
                            prompt += f"""
                            - 未平仓合约：{data['market_data']['open_interest']} USDT
                            """
                
                # 添加分析需求
                prompt += f"""
                ## 需要的分析内容

                请基于上述数据提供以下分析：
                **注意：分析结果不作为输出内容
                1. **市场整体分析**：
                - 主要币种的市场结构和趋势
                - 币种之间的相关性分析及其交易影响
                - 整体市场情绪评估

                2. **个别币种分析**：
                - 对每个币种的技术指标和市场结构进行解读
                - 识别最佳交易机会和应避免的币种
                - 关键支撑位和阻力位的重要性

                3. **投资组合建议**：
                - 建议的资金分配比例
                - 多元化策略和相关性管理
                - 总体风险敞口控制

                4. **具体交易决策**：
                - 为每个推荐的交易提供明确的行动建议（开多/开空/平多/平空/持仓/等待）
                - 提供具体的入场区域、止损位和目标位
                - 如有现有仓位，提供持仓管理建议

                5. **风险管理**：
                - 每个交易的建议仓位规模（账户百分比）
                - 建议的杠杆倍数
                - 风险回报比计算
                - 投资组合级别的风险控制

                6. **执行优先级**：
                - 交易执行的优先顺序
                - 时间敏感度评估
                - 建议的观察周期
                """

                return prompt
            except Exception as e:
                logger.warning(f"第{attempt + 1}次尝试异常: {e}")
                if attempt == max_retries - 1:
                    break
                time.sleep(3)
                
        