"""
风险度量基类

定义风险度量的统一接口和通用功能
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Any, List, Tuple, Union
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from scipy import stats

from .exceptions import (
    DataFormatError,
    InsufficientDataError,
    InvalidParameterError,
    CalculationError
)

logger = logging.getLogger(__name__)


class MetricsBase(ABC):
    """
    风险度量抽象基类
    
    定义了所有风险度量工具必须实现的接口和通用功能
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化风险度量器
        
        Parameters
        ----------
        config : Dict[str, Any], optional
            度量器配置参数
        """
        self.config = config or {}
        self.calculation_history = []
        
        # 默认配置
        self.confidence_levels = self.config.get('confidence_levels', [0.95, 0.99])
        self.return_freq = self.config.get('return_freq', 252)  # 年化频率
        self.min_observations = self.config.get('min_observations', 30)
        
        logger.info(f"Initialized {self.__class__.__name__} with config: {self.config}")
    
    @abstractmethod
    def calculate_risk_metrics(self,
                              returns: pd.Series,
                              weights: Optional[pd.Series] = None,
                              **kwargs) -> Dict[str, float]:
        """
        计算风险指标
        
        Parameters
        ----------
        returns : pd.Series
            收益率序列或组合收益率
        weights : pd.Series, optional
            组合权重（如果计算组合风险）
        **kwargs : dict
            其他参数
            
        Returns
        -------
        Dict[str, float]
            风险指标字典
        """
        pass
    
    def calculate_volatility(self, 
                           returns: pd.Series,
                           annualized: bool = True,
                           method: str = 'sample') -> float:
        """
        计算波动率
        
        Parameters
        ----------
        returns : pd.Series
            收益率序列
        annualized : bool
            是否年化
        method : str
            计算方法 {'sample', 'ewm', 'garch'}
            
        Returns
        -------
        float
            波动率
        """
        self.validate_returns(returns)
        
        if method == 'sample':
            vol = returns.std()
        elif method == 'ewm':
            # 指数加权移动标准差
            span = self.config.get('ewm_span', 30)
            vol = returns.ewm(span=span).std().iloc[-1]
        elif method == 'garch':
            # GARCH模型（简化实现）
            vol = self._calculate_garch_volatility(returns)
        else:
            raise InvalidParameterError('method', method, "{'sample', 'ewm', 'garch'}")
        
        if annualized:
            vol *= np.sqrt(self.return_freq)
        
        return vol
    
    def calculate_var(self,
                     returns: pd.Series,
                     confidence_level: float = 0.95,
                     method: str = 'historical') -> float:
        """
        计算在险价值(VaR)
        
        Parameters
        ----------
        returns : pd.Series
            收益率序列
        confidence_level : float
            置信水平
        method : str
            计算方法 {'historical', 'parametric', 'monte_carlo'}
            
        Returns
        -------
        float
            VaR值（负数表示损失）
        """
        self.validate_returns(returns)
        
        if not 0 < confidence_level < 1:
            raise InvalidParameterError('confidence_level', confidence_level, "(0, 1)")
        
        if method == 'historical':
            return np.percentile(returns, (1 - confidence_level) * 100)
        
        elif method == 'parametric':
            # 假设收益率服从正态分布
            mean = returns.mean()
            std = returns.std()
            z_score = stats.norm.ppf(1 - confidence_level)
            return mean + z_score * std
        
        elif method == 'monte_carlo':
            # 蒙特卡洛模拟
            n_simulations = self.config.get('mc_simulations', 10000)
            simulated_returns = self._monte_carlo_simulation(returns, n_simulations)
            return np.percentile(simulated_returns, (1 - confidence_level) * 100)
        
        else:
            raise InvalidParameterError('method', method, "{'historical', 'parametric', 'monte_carlo'}")
    
    def calculate_cvar(self,
                      returns: pd.Series,
                      confidence_level: float = 0.95,
                      method: str = 'historical') -> float:
        """
        计算条件在险价值(CVaR/Expected Shortfall)
        
        Parameters
        ----------
        returns : pd.Series
            收益率序列
        confidence_level : float
            置信水平
        method : str
            计算方法
            
        Returns
        -------
        float
            CVaR值
        """
        var = self.calculate_var(returns, confidence_level, method)
        
        if method == 'historical':
            # 计算低于VaR的收益率的平均值
            tail_returns = returns[returns <= var]
            if len(tail_returns) == 0:
                return var
            return tail_returns.mean()
        
        elif method == 'parametric':
            # 正态分布下的解析解
            mean = returns.mean()
            std = returns.std()
            z_score = stats.norm.ppf(1 - confidence_level)
            density = stats.norm.pdf(z_score)
            return mean - std * density / (1 - confidence_level)
        
        else:
            # 对于其他方法，使用历史方法
            return self.calculate_cvar(returns, confidence_level, 'historical')
    
    def calculate_downside_deviation(self,
                                   returns: pd.Series,
                                   target_return: float = 0.0,
                                   annualized: bool = True) -> float:
        """
        计算下行偏差
        
        Parameters
        ----------
        returns : pd.Series
            收益率序列
        target_return : float
            目标收益率
        annualized : bool
            是否年化
            
        Returns
        -------
        float
            下行偏差
        """
        self.validate_returns(returns)
        
        # 计算低于目标收益率的偏差
        downside_returns = returns[returns < target_return]
        if len(downside_returns) == 0:
            return 0.0
        
        downside_deviation = np.sqrt(np.mean((downside_returns - target_return) ** 2))
        
        if annualized:
            downside_deviation *= np.sqrt(self.return_freq)
        
        return downside_deviation
    
    def calculate_maximum_drawdown(self, returns: pd.Series) -> Dict[str, Any]:
        """
        计算最大回撤
        
        Parameters
        ----------
        returns : pd.Series
            收益率序列
            
        Returns
        -------
        Dict[str, Any]
            最大回撤信息 {
                'max_drawdown': float,      # 最大回撤
                'start_date': datetime,     # 回撤开始日期
                'end_date': datetime,       # 回撤结束日期
                'recovery_date': datetime,  # 回撤恢复日期
                'duration': int             # 回撤持续天数
            }
        """
        self.validate_returns(returns)
        
        # 计算累计收益
        cumulative = (1 + returns).cumprod()
        
        # 计算滚动最高点
        peak = cumulative.expanding().max()
        
        # 计算回撤
        drawdown = (cumulative - peak) / peak
        
        # 找到最大回撤
        max_dd_idx = drawdown.idxmin()
        max_drawdown = drawdown.loc[max_dd_idx]
        
        # 找到回撤开始点
        start_idx = None
        for i in range(len(drawdown.loc[:max_dd_idx])):
            if drawdown.iloc[i] == 0:
                start_idx = drawdown.index[i]
        
        # 找到回撤恢复点
        recovery_idx = None
        try:
            recovery_series = drawdown.loc[max_dd_idx:]
            recovery_zero = recovery_series[recovery_series >= -1e-6]  # 允许小的数值误差
            if len(recovery_zero) > 0:
                recovery_idx = recovery_zero.index[0]
        except:
            pass
        
        # 计算持续时间
        duration = None
        if start_idx and isinstance(returns.index, pd.DatetimeIndex):
            end_date = max_dd_idx
            duration = (end_date - start_idx).days
        
        return {
            'max_drawdown': abs(max_drawdown),
            'start_date': start_idx,
            'end_date': max_dd_idx,
            'recovery_date': recovery_idx,
            'duration': duration,
            'drawdown_series': drawdown
        }
    
    def calculate_risk_ratios(self,
                            returns: pd.Series,
                            risk_free_rate: float = 0.0,
                            benchmark_returns: Optional[pd.Series] = None) -> Dict[str, float]:
        """
        计算风险调整收益比率
        
        Parameters
        ----------
        returns : pd.Series
            收益率序列
        risk_free_rate : float
            无风险收益率
        benchmark_returns : pd.Series, optional
            基准收益率
            
        Returns
        -------
        Dict[str, float]
            风险比率字典
        """
        self.validate_returns(returns)
        
        # 基础统计量
        mean_return = returns.mean() * self.return_freq
        volatility = self.calculate_volatility(returns, annualized=True)
        
        # 夏普比率
        sharpe_ratio = (mean_return - risk_free_rate) / volatility if volatility > 0 else 0
        
        # 索提诺比率
        downside_dev = self.calculate_downside_deviation(returns, target_return=risk_free_rate/self.return_freq)
        sortino_ratio = (mean_return - risk_free_rate) / downside_dev if downside_dev > 0 else 0
        
        # 卡尔马比率
        max_dd_info = self.calculate_maximum_drawdown(returns)
        max_drawdown = max_dd_info['max_drawdown']
        calmar_ratio = mean_return / max_drawdown if max_drawdown > 0 else 0
        
        # 信息比率（如果有基准）
        information_ratio = 0
        if benchmark_returns is not None:
            active_returns = returns - benchmark_returns
            tracking_error = self.calculate_volatility(active_returns, annualized=True)
            if tracking_error > 0:
                information_ratio = active_returns.mean() * self.return_freq / tracking_error
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'information_ratio': information_ratio
        }
    
    def calculate_tail_risk_metrics(self, 
                                  returns: pd.Series,
                                  confidence_levels: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        计算尾部风险指标
        
        Parameters
        ----------
        returns : pd.Series
            收益率序列
        confidence_levels : List[float], optional
            置信水平列表
            
        Returns
        -------
        Dict[str, Any]
            尾部风险指标
        """
        if confidence_levels is None:
            confidence_levels = self.confidence_levels
        
        tail_metrics = {}
        
        for cl in confidence_levels:
            cl_str = f"{int(cl*100)}%"
            
            # VaR和CVaR
            var_hist = self.calculate_var(returns, cl, 'historical')
            var_param = self.calculate_var(returns, cl, 'parametric')
            cvar_hist = self.calculate_cvar(returns, cl, 'historical')
            
            tail_metrics.update({
                f'var_{cl_str}_hist': var_hist,
                f'var_{cl_str}_param': var_param,
                f'cvar_{cl_str}_hist': cvar_hist
            })
        
        # 极值理论指标
        try:
            extreme_metrics = self._calculate_extreme_value_metrics(returns)
            tail_metrics.update(extreme_metrics)
        except Exception as e:
            logger.warning(f"Failed to calculate extreme value metrics: {e}")
        
        return tail_metrics
    
    def validate_returns(self, returns: pd.Series) -> bool:
        """
        验证收益率数据
        
        Parameters
        ----------
        returns : pd.Series
            收益率数据
            
        Returns
        -------
        bool
            验证是否通过
        """
        if not isinstance(returns, pd.Series):
            raise DataFormatError("pandas Series", f"{type(returns).__name__}")
        
        if not pd.api.types.is_numeric_dtype(returns.dtype):
            raise DataFormatError("numeric dtype", f"{returns.dtype}")
        
        if len(returns) < self.min_observations:
            raise InsufficientDataError(self.min_observations, len(returns), "return observations")
        
        # 检查极端值
        extreme_threshold = 1.0  # 100%的日收益率
        extreme_returns = returns.abs() > extreme_threshold
        if extreme_returns.any():
            extreme_count = extreme_returns.sum()
            logger.warning(f"Found {extreme_count} extreme returns (>{extreme_threshold*100}%)")
        
        return True
    
    def _calculate_garch_volatility(self, returns: pd.Series) -> float:
        """
        计算GARCH模型波动率（简化版）
        
        Parameters
        ----------
        returns : pd.Series
            收益率序列
            
        Returns
        -------
        float
            GARCH波动率
        """
        try:
            # 简化的GARCH(1,1)模型
            # 实际实现中可以使用arch包
            
            # 计算平方收益率
            squared_returns = returns ** 2
            
            # 指数加权移动平均
            alpha = 0.1  # GARCH参数
            beta = 0.85  # GARCH参数
            omega = squared_returns.var() * (1 - alpha - beta)  # 长期方差
            
            # 递推计算条件方差
            conditional_var = [squared_returns.iloc[0]]
            
            for i in range(1, len(returns)):
                var_t = (omega + 
                        alpha * squared_returns.iloc[i-1] + 
                        beta * conditional_var[-1])
                conditional_var.append(var_t)
            
            return np.sqrt(conditional_var[-1])
        
        except Exception as e:
            logger.warning(f"GARCH calculation failed, using sample volatility: {e}")
            return returns.std()
    
    def _monte_carlo_simulation(self, returns: pd.Series, n_simulations: int) -> np.ndarray:
        """
        蒙特卡洛模拟
        
        Parameters
        ----------
        returns : pd.Series
            历史收益率
        n_simulations : int
            模拟次数
            
        Returns
        -------
        np.ndarray
            模拟收益率
        """
        # 估计分布参数
        mu = returns.mean()
        sigma = returns.std()
        
        # 正态分布模拟
        np.random.seed(42)
        simulated_returns = np.random.normal(mu, sigma, n_simulations)
        
        return simulated_returns
    
    def _calculate_extreme_value_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """
        计算极值理论指标
        
        Parameters
        ----------
        returns : pd.Series
            收益率序列
            
        Returns
        -------
        Dict[str, float]
            极值指标
        """
        try:
            from scipy.stats import genextreme
            
            # 使用块最大值方法
            block_size = 21  # 约一个月
            blocks = [returns.iloc[i:i+block_size] for i in range(0, len(returns), block_size)]
            block_minima = [block.min() for block in blocks if len(block) == block_size]
            
            if len(block_minima) < 10:
                return {}
            
            # 拟合广义极值分布
            params = genextreme.fit(block_minima)
            shape, loc, scale = params
            
            # 计算极值VaR
            extreme_var_95 = genextreme.ppf(0.05, *params)
            extreme_var_99 = genextreme.ppf(0.01, *params)
            
            return {
                'extreme_var_95': extreme_var_95,
                'extreme_var_99': extreme_var_99,
                'gev_shape': shape,
                'gev_location': loc,
                'gev_scale': scale
            }
        
        except Exception as e:
            logger.warning(f"Extreme value calculation failed: {e}")
            return {}
    
    def get_calculation_history(self) -> List[Dict[str, Any]]:
        """
        获取计算历史
        
        Returns
        -------
        List[Dict[str, Any]]
            计算历史记录
        """
        return self.calculation_history.copy()
    
    def track_calculation(self, metrics: Dict[str, Any], method: str):
        """
        跟踪计算结果
        
        Parameters
        ----------
        metrics : Dict[str, Any]
            计算的风险指标
        method : str
            计算方法
        """
        record = {
            'timestamp': datetime.now(),
            'method': method,
            'metrics': metrics.copy()
        }
        
        self.calculation_history.append(record)
        
        # 保留最近50次计算记录
        if len(self.calculation_history) > 50:
            self.calculation_history = self.calculation_history[-50:]