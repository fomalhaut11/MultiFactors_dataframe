"""
Alpha191 核心计算器

重构自原始 GTJA_191 类，修复 pandas API 兼容性问题
"""
from scipy.stats import rankdata
import scipy as sp
import numpy as np
import pandas as pd
from typing import Dict, Optional, Union, Any
import logging
import warnings

logger = logging.getLogger(__name__)

# 忽略 pandas 性能警告
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)


class Alpha191Calculator:
    """
    Alpha191 因子计算器
    
    重构自原始代码，修复了 pandas API 兼容性问题
    """
    
    def __init__(self):
        """初始化计算器"""
        self.open_price = None
        self.close = None
        self.low = None
        self.high = None
        self.avg_price = None
        self.prev_close = None
        self.volume = None
        self.amount = None
        self.benchmark_open_price = None
        self.benchmark_close_price = None
        
    def prepare_data(self, data: Dict[str, pd.DataFrame], 
                    benchmark_data: Optional[pd.DataFrame] = None):
        """
        准备计算数据
        
        Parameters
        ----------
        data : dict
            包含价格和成交量数据的字典
        benchmark_data : pd.DataFrame, optional
            基准数据
        """
        # 基础价格数据
        self.open_price = data['o']
        self.close = data['c']
        self.low = data['l']
        self.high = data['h']
        self.avg_price = data['vwap']
        self.volume = data['v']
        self.amount = data.get('amount', self.volume * self.avg_price)
        self.prev_close = data.get('prev_close', self.close.shift(1))
        
        # 基准数据（如果提供）
        if benchmark_data is not None:
            self.benchmark_open_price = benchmark_data.get('o')
            self.benchmark_close_price = benchmark_data.get('c')
    
    # ==================== 辅助函数 ====================
    
    @staticmethod
    def func_rank(na):
        """排名函数"""
        if len(na) == 0 or na.isna().all():
            return 0.5
        return rankdata(na, nan_policy='omit')[-1] / len(na[~na.isna()])
    
    @staticmethod
    def func_decaylinear(na):
        """线性衰减权重函数"""
        n = len(na)
        if n == 0:
            return 0
        decay_weights = np.arange(1, n + 1, 1)
        decay_weights = decay_weights / decay_weights.sum()
        return (na * decay_weights).sum()
    
    @staticmethod
    def func_highday(na):
        """最高价距今天数"""
        if len(na) == 0 or na.isna().all():
            return len(na)
        return len(na) - na.values.argmax()
    
    @staticmethod
    def func_lowday(na):
        """最低价距今天数"""
        if len(na) == 0 or na.isna().all():
            return len(na)
        return len(na) - na.values.argmin()
    
    # ==================== Alpha 因子实现 ====================
    
    def alpha_001(self):
        """
        -1 * CORR(RANK(DELTA(LOG(VOLUME),1)), RANK(((CLOSE-OPEN)/OPEN)), 6)
        """
        try:
            volume_delta = np.log(self.volume).diff(1)
            price_ratio = (self.close - self.open_price) / self.open_price
            
            data1 = volume_delta.rank(axis=1, pct=True)
            data2 = price_ratio.rank(axis=1, pct=True)
            
            # 计算6期相关性
            result = -1 * data1.rolling(window=6).corr(data2)
            return result.iloc[-1, :].dropna()
        except Exception as e:
            logger.error(f"Alpha001 计算错误: {e}")
            return pd.Series(dtype=float)
    
    def alpha_002(self):
        """
        -1 * DELTA((((CLOSE-LOW)-(HIGH-CLOSE))/((HIGH-LOW))), 1)
        """
        try:
            numerator = (self.close - self.low) - (self.high - self.close)
            denominator = self.high - self.low
            
            # 避免除零
            ratio = numerator / denominator.replace(0, np.nan)
            result = -1 * ratio.diff(1)
            
            return result.iloc[-1, :].dropna()
        except Exception as e:
            logger.error(f"Alpha002 计算错误: {e}")
            return pd.Series(dtype=float)
    
    def alpha_003(self):
        """
        SUM((CLOSE=DELAY(CLOSE,1)?0:CLOSE-(CLOSE>DELAY(CLOSE,1)?
        MIN(DELAY(CLOSE,1),LOW):MAX(DELAY(CLOSE,1),HIGH))), 6)
        """
        try:
            delay1 = self.close.shift(1)
            
            condition_equal = (self.close == delay1)
            condition_up = (self.close > delay1)
            condition_down = (self.close < delay1)
            
            result = self.close.copy() * 0  # 初始化为0
            
            # 上涨时：CLOSE - MIN(DELAY(CLOSE,1), LOW)
            up_part = self.close - np.minimum(delay1, self.low)
            result[condition_up] = up_part[condition_up]
            
            # 下跌时：CLOSE - MAX(DELAY(CLOSE,1), HIGH)  
            down_part = self.close - np.maximum(delay1, self.high)
            result[condition_down] = down_part[condition_down]
            
            # 平盘时为0（已初始化）
            result[condition_equal] = 0
            
            return result.iloc[-6:, :].sum().dropna()
        except Exception as e:
            logger.error(f"Alpha003 计算错误: {e}")
            return pd.Series(dtype=float)
    
    def alpha_004(self):
        """
        ((((SUM(CLOSE, 8) / 8) + STD(CLOSE, 8)) < (SUM(CLOSE, 2) / 2)) ? (-1 * 1) : 
        (((SUM(CLOSE, 2) / 2) < (SUM(CLOSE, 8) / 8 - STD(CLOSE, 8))) ? 1 : 
        (((1 < (VOLUME / MEAN(VOLUME, 20))) || ((VOLUME / MEAN(VOLUME, 20)) == 1)) ? 1 : (-1 * 1))))
        """
        try:
            sma_8 = self.close.rolling(window=8).mean()
            std_8 = self.close.rolling(window=8).std()
            sma_2 = self.close.rolling(window=2).mean()
            vol_ratio = self.volume / self.volume.rolling(window=20).mean()
            
            condition1 = (sma_8 + std_8) < sma_2
            condition2 = sma_2 < (sma_8 - std_8)
            condition3 = vol_ratio >= 1
            
            result = pd.DataFrame(np.ones(self.close.shape), 
                                index=self.close.index, columns=self.close.columns)
            
            # 应用条件
            result[condition1] = -1
            result[~condition1 & condition2] = 1  
            result[~condition1 & ~condition2 & condition3] = 1
            result[~condition1 & ~condition2 & ~condition3] = -1
            
            return result.iloc[-1, :].dropna()
        except Exception as e:
            logger.error(f"Alpha004 计算错误: {e}")
            return pd.Series(dtype=float)
    
    def alpha_005(self):
        """
        -1 * TSMAX(CORR(TSRANK(VOLUME, 5), TSRANK(HIGH, 5), 5), 3)
        """
        try:
            # 计算5期时序排名
            ts_volume = self.volume.rolling(window=5).apply(self.func_rank)
            ts_high = self.high.rolling(window=5).apply(self.func_rank)
            
            # 计算5期相关性
            corr_ts = ts_volume.rolling(window=5).corr(ts_high)
            
            # 计算3期最大值
            result = -1 * corr_ts.rolling(window=3).max()
            
            return result.iloc[-1, :].dropna()
        except Exception as e:
            logger.error(f"Alpha005 计算错误: {e}")
            return pd.Series(dtype=float)
    
    def alpha_006(self):
        """
        -1 * RANK(SIGN(DELTA((OPEN * 0.85 + HIGH * 0.15), 4)))
        """
        try:
            composite = self.open_price * 0.85 + self.high * 0.15
            delta = composite.diff(4)
            
            # 计算符号
            sign_delta = np.sign(delta)
            
            # 截面排名
            result = -1 * sign_delta.rank(axis=1, pct=True)
            
            return result.iloc[-1, :].dropna()
        except Exception as e:
            logger.error(f"Alpha006 计算错误: {e}")
            return pd.Series(dtype=float)
    
    def alpha_007(self):
        """
        ((ADRANK > 3) && (RANK(VWAP - CLOSE) > 3)) ? -1 : 
        ((ADRANK < 3) && (RANK(VWAP - CLOSE) < 3)) ? 1 : 
        ((RANK(DELTA(VOLUME, 3)) * RANK((VWAP - CLOSE))))
        """
        try:
            vwap_close_diff = self.avg_price - self.close
            volume_delta = self.volume.diff(3)
            
            rank1 = np.maximum(vwap_close_diff, 3).rank(axis=1, pct=True)
            rank2 = np.minimum(vwap_close_diff, 3).rank(axis=1, pct=True)  
            rank3 = volume_delta.rank(axis=1, pct=True)
            
            result = rank1 + rank2 * rank3
            
            return result.iloc[-1, :].dropna()
        except Exception as e:
            logger.error(f"Alpha007 计算错误: {e}")
            return pd.Series(dtype=float)
    
    def alpha_008(self):
        """
        -1 * RANK(DELTA(((HIGH + LOW) / 2 * 0.2 + VWAP * 0.8), 4))
        """
        try:
            composite = (self.high + self.low) / 2 * 0.2 + self.avg_price * 0.8
            delta = composite.diff(4)
            result = -1 * delta.rank(axis=1, pct=True)
            
            return result.iloc[-1, :].dropna()
        except Exception as e:
            logger.error(f"Alpha008 计算错误: {e}")
            return pd.Series(dtype=float)
    
    # ==================== 更多 Alpha 因子 ====================
    # 由于篇幅限制，这里只实现前8个因子作为示例
    # 其他因子将在后续添加
    
    def get_available_alphas(self):
        """获取已实现的 Alpha 因子列表"""
        alphas = []
        for attr in dir(self):
            if attr.startswith('alpha_') and callable(getattr(self, attr)):
                alpha_num = int(attr.split('_')[1])
                alphas.append(alpha_num)
        return sorted(alphas)
    
    def calculate_alpha(self, alpha_num: int) -> pd.Series:
        """
        计算指定的 Alpha 因子
        
        Parameters
        ----------
        alpha_num : int
            Alpha 因子编号
            
        Returns
        -------
        pd.Series
            计算结果
        """
        method_name = f'alpha_{alpha_num:03d}'
        
        if not hasattr(self, method_name):
            raise NotImplementedError(f"Alpha{alpha_num:03d} 尚未实现")
        
        method = getattr(self, method_name)
        try:
            return method()
        except Exception as e:
            logger.error(f"Alpha{alpha_num:03d} 计算失败: {e}")
            return pd.Series(dtype=float)

    # ==================== 更多 Alpha 因子实现占位符 ====================
    
    def alpha_009(self):
        raise NotImplementedError("Alpha009 待实现")
    
    def alpha_010(self):
        raise NotImplementedError("Alpha010 待实现")
        
    # ... 其他 Alpha 因子将逐步实现