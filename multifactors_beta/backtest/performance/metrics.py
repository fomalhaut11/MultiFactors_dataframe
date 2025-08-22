"""
绩效指标计算模块

提供各种投资组合绩效指标的计算功能
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
import logging
from scipy import stats

logger = logging.getLogger(__name__)

class PerformanceMetrics:
    """
    绩效指标计算器
    
    计算各种投资组合绩效和风险指标
    """
    
    def __init__(self, risk_free_rate: float = 0.025):
        """
        初始化绩效计算器
        
        Parameters
        ----------
        risk_free_rate : float
            无风险利率（年化）
        """
        self.risk_free_rate = risk_free_rate
        
    def calculate_all_metrics(self, 
                            returns: pd.Series,
                            benchmark_returns: Optional[pd.Series] = None) -> Dict[str, float]:
        """
        计算所有绩效指标
        
        Parameters
        ----------
        returns : pd.Series
            收益率序列
        benchmark_returns : pd.Series, optional
            基准收益率序列
            
        Returns
        -------
        Dict[str, float]
            所有绩效指标
        """
        metrics = {}
        
        # 基础收益指标
        metrics.update(self.calculate_return_metrics(returns))
        
        # 风险指标
        metrics.update(self.calculate_risk_metrics(returns))
        
        # 风险调整收益指标
        metrics.update(self.calculate_risk_adjusted_metrics(returns))
        
        # 如果有基准，计算相对指标
        if benchmark_returns is not None:
            metrics.update(self.calculate_relative_metrics(returns, benchmark_returns))
        
        return metrics
    
    def calculate_return_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """
        计算收益相关指标
        
        Parameters
        ----------
        returns : pd.Series
            收益率序列
            
        Returns
        -------
        Dict[str, float]
            收益指标
        """
        if len(returns) == 0:
            return {}
        
        # 累计收益率
        total_return = (1 + returns).prod() - 1
        
        # 年化收益率
        years = len(returns) / 252.0
        annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        # 平均日收益率
        mean_daily_return = returns.mean()
        
        # 几何平均收益率
        geometric_mean = (1 + returns).prod() ** (1 / len(returns)) - 1
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'mean_daily_return': mean_daily_return,
            'geometric_mean': geometric_mean
        }
    
    def calculate_risk_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """
        计算风险相关指标
        
        Parameters
        ----------
        returns : pd.Series
            收益率序列
            
        Returns
        -------
        Dict[str, float]
            风险指标
        """
        if len(returns) == 0:
            return {}
        
        # 波动率
        daily_volatility = returns.std()
        annual_volatility = daily_volatility * np.sqrt(252)
        
        # 最大回撤
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdowns = (cumulative - running_max) / running_max
        max_drawdown = drawdowns.min()
        
        # 回撤持续时间
        drawdown_periods = self._calculate_drawdown_periods(drawdowns)
        max_drawdown_duration = max(drawdown_periods) if drawdown_periods else 0
        avg_drawdown_duration = np.mean(drawdown_periods) if drawdown_periods else 0
        
        # VaR和CVaR
        var_95 = np.percentile(returns, 5)  # 95% VaR
        var_99 = np.percentile(returns, 1)  # 99% VaR
        cvar_95 = returns[returns <= var_95].mean()
        cvar_99 = returns[returns <= var_99].mean()
        
        # 下行偏差
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        
        # 偏度和峰度
        skewness = returns.skew()
        kurtosis = returns.kurtosis()
        
        return {
            'daily_volatility': daily_volatility,
            'annual_volatility': annual_volatility,
            'max_drawdown': max_drawdown,
            'max_drawdown_duration': max_drawdown_duration,
            'avg_drawdown_duration': avg_drawdown_duration,
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,
            'cvar_99': cvar_99,
            'downside_deviation': downside_deviation,
            'skewness': skewness,
            'kurtosis': kurtosis
        }
    
    def calculate_risk_adjusted_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """
        计算风险调整收益指标
        
        Parameters
        ----------
        returns : pd.Series
            收益率序列
            
        Returns
        -------
        Dict[str, float]
            风险调整指标
        """
        if len(returns) == 0:
            return {}
        
        # 基础指标
        annual_return = returns.mean() * 252
        annual_volatility = returns.std() * np.sqrt(252)
        
        # 夏普比率
        sharpe_ratio = (annual_return - self.risk_free_rate) / annual_volatility if annual_volatility > 0 else 0
        
        # 最大回撤
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdowns = (cumulative - running_max) / running_max
        max_drawdown = drawdowns.min()
        
        # Calmar比率
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Sortino比率（相对于下行偏差）
        downside_returns = returns[returns < self.risk_free_rate / 252]
        downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = (annual_return - self.risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
        
        # 信息比率（相对于0收益）
        excess_returns = returns
        information_ratio = excess_returns.mean() / excess_returns.std() if excess_returns.std() > 0 else 0
        
        # Omega比率
        omega_ratio = self._calculate_omega_ratio(returns, threshold=0)
        
        # 胜率
        win_rate = (returns > 0).mean()
        
        # 盈亏比
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        profit_loss_ratio = (positive_returns.mean() / abs(negative_returns.mean()) 
                           if len(negative_returns) > 0 and negative_returns.mean() != 0 else np.inf)
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'calmar_ratio': calmar_ratio,
            'sortino_ratio': sortino_ratio,
            'information_ratio': information_ratio,
            'omega_ratio': omega_ratio,
            'win_rate': win_rate,
            'profit_loss_ratio': profit_loss_ratio
        }
    
    def calculate_relative_metrics(self, 
                                 returns: pd.Series, 
                                 benchmark_returns: pd.Series) -> Dict[str, float]:
        """
        计算相对基准的指标
        
        Parameters
        ----------
        returns : pd.Series
            组合收益率序列
        benchmark_returns : pd.Series
            基准收益率序列
            
        Returns
        -------
        Dict[str, float]
            相对指标
        """
        # 对齐数据
        aligned_data = pd.concat([returns, benchmark_returns], axis=1, join='inner')
        if aligned_data.empty:
            return {}
        
        portfolio_ret = aligned_data.iloc[:, 0]
        benchmark_ret = aligned_data.iloc[:, 1]
        
        # 超额收益
        excess_returns = portfolio_ret - benchmark_ret
        
        # 跟踪误差
        tracking_error = excess_returns.std() * np.sqrt(252)
        
        # 信息比率
        excess_annual_return = excess_returns.mean() * 252
        information_ratio = excess_annual_return / tracking_error if tracking_error > 0 else 0
        
        # Beta
        covariance = portfolio_ret.cov(benchmark_ret)
        benchmark_variance = benchmark_ret.var()
        beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
        
        # Alpha (CAPM)
        portfolio_annual_return = portfolio_ret.mean() * 252
        benchmark_annual_return = benchmark_ret.mean() * 252
        alpha = portfolio_annual_return - (self.risk_free_rate + beta * (benchmark_annual_return - self.risk_free_rate))
        
        # Treynor比率
        treynor_ratio = (portfolio_annual_return - self.risk_free_rate) / beta if beta != 0 else 0
        
        # 相关系数
        correlation = portfolio_ret.corr(benchmark_ret)
        
        # 上行捕获率和下行捕获率
        up_capture, down_capture = self._calculate_capture_ratios(portfolio_ret, benchmark_ret)
        
        return {
            'excess_annual_return': excess_annual_return,
            'tracking_error': tracking_error,
            'information_ratio': information_ratio,
            'beta': beta,
            'alpha': alpha,
            'treynor_ratio': treynor_ratio,
            'correlation': correlation,
            'up_capture_ratio': up_capture,
            'down_capture_ratio': down_capture
        }
    
    def _calculate_drawdown_periods(self, drawdowns: pd.Series) -> list:
        """计算回撤持续期间"""
        periods = []
        current_period = 0
        
        for dd in drawdowns:
            if dd < 0:
                current_period += 1
            else:
                if current_period > 0:
                    periods.append(current_period)
                current_period = 0
        
        # 如果序列结束时仍在回撤中
        if current_period > 0:
            periods.append(current_period)
        
        return periods
    
    def _calculate_omega_ratio(self, returns: pd.Series, threshold: float = 0) -> float:
        """计算Omega比率"""
        excess_returns = returns - threshold
        positive_returns = excess_returns[excess_returns > 0]
        negative_returns = excess_returns[excess_returns <= 0]
        
        if len(negative_returns) == 0:
            return np.inf
        
        gains = positive_returns.sum()
        losses = abs(negative_returns.sum())
        
        return gains / losses if losses > 0 else np.inf
    
    def _calculate_capture_ratios(self, 
                                portfolio_returns: pd.Series, 
                                benchmark_returns: pd.Series) -> Tuple[float, float]:
        """计算上行和下行捕获率"""
        # 上行市场
        up_market_mask = benchmark_returns > 0
        if up_market_mask.sum() > 0:
            portfolio_up = portfolio_returns[up_market_mask].mean()
            benchmark_up = benchmark_returns[up_market_mask].mean()
            up_capture = portfolio_up / benchmark_up if benchmark_up != 0 else 0
        else:
            up_capture = 0
        
        # 下行市场
        down_market_mask = benchmark_returns < 0
        if down_market_mask.sum() > 0:
            portfolio_down = portfolio_returns[down_market_mask].mean()
            benchmark_down = benchmark_returns[down_market_mask].mean()
            down_capture = portfolio_down / benchmark_down if benchmark_down != 0 else 0
        else:
            down_capture = 0
        
        return up_capture, down_capture
    
    def rolling_metrics(self, 
                       returns: pd.Series, 
                       window: int = 252,
                       metrics: list = None) -> pd.DataFrame:
        """
        计算滚动窗口绩效指标
        
        Parameters
        ----------
        returns : pd.Series
            收益率序列
        window : int
            滚动窗口大小（天数）
        metrics : list, optional
            要计算的指标列表
            
        Returns
        -------
        pd.DataFrame
            滚动指标结果
        """
        if metrics is None:
            metrics = ['annual_return', 'annual_volatility', 'sharpe_ratio', 'max_drawdown']
        
        rolling_results = {}
        
        for metric in metrics:
            rolling_values = []
            
            for i in range(window, len(returns) + 1):
                window_returns = returns.iloc[i-window:i]
                
                if metric == 'annual_return':
                    value = window_returns.mean() * 252
                elif metric == 'annual_volatility':
                    value = window_returns.std() * np.sqrt(252)
                elif metric == 'sharpe_ratio':
                    annual_ret = window_returns.mean() * 252
                    annual_vol = window_returns.std() * np.sqrt(252)
                    value = (annual_ret - self.risk_free_rate) / annual_vol if annual_vol > 0 else 0
                elif metric == 'max_drawdown':
                    cumulative = (1 + window_returns).cumprod()
                    running_max = cumulative.expanding().max()
                    drawdowns = (cumulative - running_max) / running_max
                    value = drawdowns.min()
                else:
                    value = np.nan
                
                rolling_values.append(value)
            
            rolling_results[metric] = pd.Series(
                rolling_values, 
                index=returns.index[window-1:]
            )
        
        return pd.DataFrame(rolling_results)

# 便捷函数
def calculate_metrics(returns: pd.Series, 
                     benchmark_returns: Optional[pd.Series] = None,
                     risk_free_rate: float = 0.025) -> Dict[str, float]:
    """
    快速计算绩效指标的便捷函数
    
    Parameters
    ----------
    returns : pd.Series
        收益率序列
    benchmark_returns : pd.Series, optional
        基准收益率序列
    risk_free_rate : float
        无风险利率
        
    Returns
    -------
    Dict[str, float]
        绩效指标
    """
    calculator = PerformanceMetrics(risk_free_rate=risk_free_rate)
    return calculator.calculate_all_metrics(returns, benchmark_returns)