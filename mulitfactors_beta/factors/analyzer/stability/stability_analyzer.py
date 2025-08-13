"""
因子稳定性分析器
评估因子在时间序列上的稳定性和持续性
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from datetime import datetime, timedelta
import warnings
from scipy import stats
from statsmodels.stats.diagnostic import breaks_cusumolw

from ..base import AnalyzerBase, BatchAnalyzerMixin
from ...tester.base import TestResult

logger = logging.getLogger(__name__)


class StabilityAnalyzer(AnalyzerBase, BatchAnalyzerMixin):
    """
    因子稳定性分析器
    
    功能：
    1. IC时间序列稳定性分析
    2. 滚动窗口性能评估
    3. 结构突变点检测
    4. 不同市场状态下的表现
    5. 因子衰减速度分析
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化稳定性分析器
        
        Parameters
        ----------
        config : Dict, optional
            配置参数
        """
        super().__init__(name="StabilityAnalyzer", config=config)
        
        # 稳定性分析配置
        self.rolling_window = self.config.get('rolling_window', 60)  # 滚动窗口天数
        self.min_window = self.config.get('min_window', 20)  # 最小窗口
        self.stability_threshold = self.config.get('stability_threshold', 0.5)  # 稳定性阈值
        self.breakpoint_pvalue = self.config.get('breakpoint_pvalue', 0.05)  # 结构突变p值
        
        # 市场状态定义
        self.market_states = self.config.get('market_states', {
            'bull': {'threshold': 0.2, 'window': 60},  # 牛市：60天涨幅>20%
            'bear': {'threshold': -0.2, 'window': 60},  # 熊市：60天跌幅>20%
            'volatile': {'std_threshold': 0.03, 'window': 20},  # 震荡：20天波动率>3%
        })
        
    def analyze(self,
               data: Union[TestResult, pd.Series, pd.DataFrame],
               market_data: Optional[pd.Series] = None,
               **kwargs) -> Dict[str, Any]:
        """
        执行稳定性分析
        
        Parameters
        ----------
        data : Union[TestResult, pd.Series, pd.DataFrame]
            分析数据，可以是：
            - TestResult: 单因子测试结果
            - pd.Series: IC时间序列
            - pd.DataFrame: 多个指标的时间序列
        market_data : pd.Series, optional
            市场数据（用于市场状态分析）
        **kwargs : dict
            其他参数
            
        Returns
        -------
        Dict[str, Any]
            稳定性分析结果
        """
        self.analysis_time = datetime.now()
        
        # 提取时间序列数据
        time_series = self._extract_time_series(data)
        
        if time_series.empty:
            logger.warning("No valid time series data for stability analysis")
            return {}
        
        results = {
            'analysis_time': self.analysis_time,
            'data_length': len(time_series),
            'data_start': time_series.index[0],
            'data_end': time_series.index[-1]
        }
        
        # 1. 基础稳定性指标
        logger.info("Calculating basic stability metrics")
        results['basic_metrics'] = self._calculate_basic_stability(time_series)
        
        # 2. 滚动窗口分析
        logger.info("Performing rolling window analysis")
        results['rolling_analysis'] = self._rolling_window_analysis(time_series)
        
        # 3. 结构突变检测
        logger.info("Detecting structural breaks")
        results['structural_breaks'] = self._detect_structural_breaks(time_series)
        
        # 4. 市场状态分析（如果提供了市场数据）
        if market_data is not None:
            logger.info("Analyzing performance across market states")
            results['market_state_analysis'] = self._analyze_market_states(
                time_series, market_data
            )
        
        # 5. 因子衰减分析
        if 'ic_decay' in kwargs:
            logger.info("Analyzing factor decay")
            results['decay_analysis'] = self._analyze_factor_decay(kwargs['ic_decay'])
        
        # 6. 稳定性评分
        results['stability_score'] = self._calculate_stability_score(results)
        
        # 7. 生成摘要
        results['summary'] = self._generate_stability_summary(results)
        
        logger.info(f"Stability analysis completed")
        
        return results
    
    def validate_input(self, data: Any) -> bool:
        """
        验证输入数据
        
        Parameters
        ----------
        data : Any
            输入数据
            
        Returns
        -------
        bool
            数据是否有效
        """
        if isinstance(data, TestResult):
            return data.ic_result is not None
        elif isinstance(data, (pd.Series, pd.DataFrame)):
            return not data.empty
        return False
    
    def _extract_time_series(self, 
                           data: Union[TestResult, pd.Series, pd.DataFrame]) -> pd.Series:
        """
        从输入数据中提取时间序列
        
        Parameters
        ----------
        data : Union[TestResult, pd.Series, pd.DataFrame]
            输入数据
            
        Returns
        -------
        pd.Series
            时间序列数据
        """
        if isinstance(data, TestResult):
            # 从测试结果中提取IC序列
            if data.ic_result and data.ic_result.ic_series is not None:
                return data.ic_result.ic_series
            else:
                return pd.Series()
                
        elif isinstance(data, pd.Series):
            return data
            
        elif isinstance(data, pd.DataFrame):
            # 如果是DataFrame，默认使用第一列
            if not data.empty:
                return data.iloc[:, 0]
            else:
                return pd.Series()
        
        return pd.Series()
    
    def _calculate_basic_stability(self, ts: pd.Series) -> Dict[str, float]:
        """
        计算基础稳定性指标
        
        Parameters
        ----------
        ts : pd.Series
            时间序列
            
        Returns
        -------
        Dict[str, float]
            基础稳定性指标
        """
        metrics = {}
        
        # 基础统计
        metrics['mean'] = ts.mean()
        metrics['std'] = ts.std()
        metrics['cv'] = metrics['std'] / abs(metrics['mean']) if metrics['mean'] != 0 else np.inf
        metrics['skewness'] = ts.skew()
        metrics['kurtosis'] = ts.kurtosis()
        
        # 稳定性指标
        metrics['autocorr_lag1'] = ts.autocorr(lag=1) if len(ts) > 1 else 0
        metrics['autocorr_lag5'] = ts.autocorr(lag=5) if len(ts) > 5 else 0
        
        # 变化率统计
        changes = ts.diff().dropna()
        if len(changes) > 0:
            metrics['change_mean'] = changes.mean()
            metrics['change_std'] = changes.std()
            metrics['max_drawdown'] = self._calculate_max_drawdown(ts.cumsum())
        
        # 正值比例（对于IC等指标）
        metrics['positive_ratio'] = (ts > 0).mean()
        
        # 趋势性（线性回归斜率）
        if len(ts) > 2:
            x = np.arange(len(ts))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, ts.values)
            metrics['trend_slope'] = slope
            metrics['trend_r2'] = r_value ** 2
            metrics['trend_pvalue'] = p_value
        
        return metrics
    
    def _rolling_window_analysis(self, ts: pd.Series) -> Dict[str, Any]:
        """
        滚动窗口分析
        
        Parameters
        ----------
        ts : pd.Series
            时间序列
            
        Returns
        -------
        Dict[str, Any]
            滚动窗口分析结果
        """
        window = min(self.rolling_window, len(ts) // 2)
        
        if window < self.min_window:
            logger.warning(f"Window size {window} is too small for rolling analysis")
            return {}
        
        # 计算滚动统计
        rolling_mean = ts.rolling(window=window, min_periods=self.min_window).mean()
        rolling_std = ts.rolling(window=window, min_periods=self.min_window).std()
        rolling_sharpe = rolling_mean / rolling_std * np.sqrt(252)
        
        # 稳定性指标
        results = {
            'window_size': window,
            'rolling_mean': rolling_mean,
            'rolling_std': rolling_std,
            'rolling_sharpe': rolling_sharpe,
            'mean_stability': 1 - rolling_mean.std() / abs(rolling_mean.mean()) if rolling_mean.mean() != 0 else 0,
            'std_stability': 1 - rolling_std.std() / rolling_std.mean() if rolling_std.mean() != 0 else 0,
        }
        
        # 计算稳定期和不稳定期
        mean_threshold = rolling_mean.mean()
        stable_periods = (rolling_mean > mean_threshold * 0.5) & (rolling_mean < mean_threshold * 1.5)
        results['stable_ratio'] = stable_periods.mean()
        
        # 识别表现最好和最差的时期
        if len(rolling_mean.dropna()) > 0:
            results['best_period'] = rolling_mean.idxmax()
            results['worst_period'] = rolling_mean.idxmin()
            results['best_value'] = rolling_mean.max()
            results['worst_value'] = rolling_mean.min()
        
        return results
    
    def _detect_structural_breaks(self, ts: pd.Series) -> Dict[str, Any]:
        """
        检测结构突变点
        
        Parameters
        ----------
        ts : pd.Series
            时间序列
            
        Returns
        -------
        Dict[str, Any]
            结构突变检测结果
        """
        if len(ts) < 30:  # 样本太少，无法检测
            return {'detected': False, 'message': 'Insufficient data for break detection'}
        
        try:
            # 使用CUSUM检验
            test_stat, pvalue, crit_values = breaks_cusumolw(ts.values)
            
            results = {
                'detected': pvalue < self.breakpoint_pvalue,
                'test_statistic': test_stat,
                'p_value': pvalue,
                'critical_values': crit_values,
            }
            
            # 如果检测到突变，尝试定位突变点
            if results['detected']:
                results['breakpoints'] = self._locate_breakpoints(ts)
            
            return results
            
        except Exception as e:
            logger.warning(f"Structural break detection failed: {e}")
            return {'detected': False, 'error': str(e)}
    
    def _locate_breakpoints(self, ts: pd.Series) -> List[datetime]:
        """
        定位结构突变点
        
        使用简单的滑动窗口方法检测均值显著变化的点
        
        Parameters
        ----------
        ts : pd.Series
            时间序列
            
        Returns
        -------
        List[datetime]
            突变点时间列表
        """
        breakpoints = []
        window = max(20, len(ts) // 10)
        
        for i in range(window, len(ts) - window):
            before = ts.iloc[i-window:i]
            after = ts.iloc[i:i+window]
            
            # t检验比较前后均值
            t_stat, p_value = stats.ttest_ind(before, after)
            
            if p_value < 0.01:  # 显著性水平1%
                breakpoints.append(ts.index[i])
        
        # 合并相近的突变点
        if breakpoints:
            filtered = [breakpoints[0]]
            for bp in breakpoints[1:]:
                if (bp - filtered[-1]).days > 30:  # 至少相隔30天
                    filtered.append(bp)
            breakpoints = filtered
        
        return breakpoints
    
    def _analyze_market_states(self, 
                              factor_ts: pd.Series,
                              market_ts: pd.Series) -> Dict[str, Any]:
        """
        分析不同市场状态下的因子表现
        
        Parameters
        ----------
        factor_ts : pd.Series
            因子时间序列（如IC）
        market_ts : pd.Series
            市场时间序列（如指数收益）
            
        Returns
        -------
        Dict[str, Any]
            市场状态分析结果
        """
        # 对齐数据
        factor_ts, market_ts = factor_ts.align(market_ts, join='inner')
        
        if len(factor_ts) < self.min_window:
            return {'message': 'Insufficient data for market state analysis'}
        
        # 识别市场状态
        states = self._identify_market_states(market_ts)
        
        # 分析每个状态下的因子表现
        state_performance = {}
        
        for state_name, state_mask in states.items():
            if state_mask.sum() > 0:
                state_data = factor_ts[state_mask]
                state_performance[state_name] = {
                    'count': len(state_data),
                    'mean': state_data.mean(),
                    'std': state_data.std(),
                    'sharpe': state_data.mean() / state_data.std() * np.sqrt(252) if state_data.std() > 0 else 0,
                    'positive_ratio': (state_data > 0).mean(),
                    'periods': self._get_state_periods(state_mask)
                }
        
        # 计算状态间差异
        if len(state_performance) > 1:
            state_means = {k: v['mean'] for k, v in state_performance.items()}
            state_performance['cross_state_stability'] = 1 - np.std(list(state_means.values())) / abs(np.mean(list(state_means.values())))
        
        return state_performance
    
    def _identify_market_states(self, market_ts: pd.Series) -> Dict[str, pd.Series]:
        """
        识别市场状态
        
        Parameters
        ----------
        market_ts : pd.Series
            市场时间序列
            
        Returns
        -------
        Dict[str, pd.Series]
            各状态的布尔掩码
        """
        states = {}
        
        # 计算滚动收益率
        bull_window = self.market_states['bull']['window']
        bull_threshold = self.market_states['bull']['threshold']
        rolling_return = market_ts.rolling(window=bull_window).apply(lambda x: (x[-1] / x[0] - 1))
        
        # 牛市状态
        states['bull'] = rolling_return > bull_threshold
        
        # 熊市状态
        bear_threshold = self.market_states['bear']['threshold']
        states['bear'] = rolling_return < bear_threshold
        
        # 震荡市场（高波动）
        vol_window = self.market_states['volatile']['window']
        vol_threshold = self.market_states['volatile']['std_threshold']
        rolling_vol = market_ts.pct_change().rolling(window=vol_window).std()
        states['volatile'] = rolling_vol > vol_threshold
        
        # 平稳市场（其他）
        states['neutral'] = ~(states['bull'] | states['bear'] | states['volatile'])
        
        return states
    
    def _get_state_periods(self, state_mask: pd.Series) -> List[Tuple[datetime, datetime]]:
        """
        获取状态持续时期
        
        Parameters
        ----------
        state_mask : pd.Series
            状态布尔掩码
            
        Returns
        -------
        List[Tuple[datetime, datetime]]
            状态时期列表
        """
        periods = []
        in_state = False
        start = None
        
        for date, is_state in state_mask.items():
            if is_state and not in_state:
                start = date
                in_state = True
            elif not is_state and in_state:
                if start:
                    periods.append((start, date))
                in_state = False
        
        # 处理最后一个时期
        if in_state and start:
            periods.append((start, state_mask.index[-1]))
        
        return periods
    
    def _analyze_factor_decay(self, ic_decay: pd.Series) -> Dict[str, Any]:
        """
        分析因子衰减特性
        
        Parameters
        ----------
        ic_decay : pd.Series
            IC衰减序列（lag为索引）
            
        Returns
        -------
        Dict[str, Any]
            衰减分析结果
        """
        if len(ic_decay) < 2:
            return {'message': 'Insufficient decay data'}
        
        results = {}
        
        # 计算半衰期
        initial_ic = ic_decay.iloc[0]
        half_ic = initial_ic / 2
        
        half_life = None
        for lag, ic in ic_decay.items():
            if ic <= half_ic:
                half_life = lag
                break
        
        results['half_life'] = half_life if half_life else len(ic_decay)
        
        # 衰减速度（线性回归斜率）
        x = np.array(ic_decay.index)
        y = ic_decay.values
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        results['decay_rate'] = -slope  # 负斜率表示衰减
        results['decay_r2'] = r_value ** 2
        results['decay_significance'] = p_value
        
        # 有效预测期（IC显著大于0的最大lag）
        significance_threshold = 0.02  # IC阈值
        effective_period = 1
        for lag, ic in ic_decay.items():
            if ic > significance_threshold:
                effective_period = lag
            else:
                break
        
        results['effective_period'] = effective_period
        
        # 衰减模式分类
        if results['decay_r2'] > 0.8:
            results['decay_pattern'] = 'linear'
        elif half_life and half_life < len(ic_decay) / 3:
            results['decay_pattern'] = 'exponential'
        else:
            results['decay_pattern'] = 'irregular'
        
        return results
    
    def _calculate_max_drawdown(self, cumulative_ts: pd.Series) -> float:
        """
        计算最大回撤
        
        Parameters
        ----------
        cumulative_ts : pd.Series
            累计序列
            
        Returns
        -------
        float
            最大回撤
        """
        expanding_max = cumulative_ts.expanding().max()
        drawdown = (cumulative_ts - expanding_max) / expanding_max
        return drawdown.min()
    
    def _calculate_stability_score(self, results: Dict[str, Any]) -> float:
        """
        计算综合稳定性评分
        
        Parameters
        ----------
        results : Dict[str, Any]
            分析结果
            
        Returns
        -------
        float
            稳定性评分（0-100）
        """
        score = 50  # 基础分
        
        # 基础稳定性贡献（权重30%）
        if 'basic_metrics' in results:
            metrics = results['basic_metrics']
            # CV越小越稳定
            if 'cv' in metrics and metrics['cv'] < 1:
                score += 15 * (1 - metrics['cv'])
            # 正值比例贡献
            if 'positive_ratio' in metrics:
                score += 15 * metrics['positive_ratio']
        
        # 滚动窗口稳定性（权重30%）
        if 'rolling_analysis' in results:
            rolling = results['rolling_analysis']
            if 'stable_ratio' in rolling:
                score += 30 * rolling['stable_ratio']
        
        # 结构突变（权重20%）
        if 'structural_breaks' in results:
            breaks = results['structural_breaks']
            if not breaks.get('detected', True):
                score += 20
            elif 'breakpoints' in breaks:
                # 突变点越少越稳定
                num_breaks = len(breaks['breakpoints'])
                score += max(0, 20 - num_breaks * 5)
        
        # 市场状态稳定性（权重20%）
        if 'market_state_analysis' in results:
            market = results['market_state_analysis']
            if 'cross_state_stability' in market:
                score += 20 * market['cross_state_stability']
        
        return min(100, max(0, score))
    
    def _generate_stability_summary(self, results: Dict[str, Any]) -> str:
        """
        生成稳定性分析摘要
        
        Parameters
        ----------
        results : Dict[str, Any]
            分析结果
            
        Returns
        -------
        str
            文本摘要
        """
        summary_lines = [
            f"Stability Analysis Summary",
            f"=" * 50,
            f"Analysis Time: {results['analysis_time'].strftime('%Y-%m-%d %H:%M:%S')}",
            f"Data Period: {results.get('data_start', 'N/A')} to {results.get('data_end', 'N/A')}",
            f"Data Points: {results.get('data_length', 'N/A')}",
            f"",
            f"Overall Stability Score: {results.get('stability_score', 0):.1f}/100",
            f"",
        ]
        
        # 基础指标
        if 'basic_metrics' in results:
            metrics = results['basic_metrics']
            summary_lines.extend([
                "Basic Stability Metrics:",
                f"  Mean: {metrics.get('mean', 0):.4f}",
                f"  Std: {metrics.get('std', 0):.4f}",
                f"  CV: {metrics.get('cv', 0):.4f}",
                f"  Positive Ratio: {metrics.get('positive_ratio', 0):.2%}",
                f"  Trend Slope: {metrics.get('trend_slope', 0):.6f}",
                f"  Max Drawdown: {metrics.get('max_drawdown', 0):.2%}",
                "",
            ])
        
        # 滚动窗口分析
        if 'rolling_analysis' in results:
            rolling = results['rolling_analysis']
            summary_lines.extend([
                f"Rolling Window Analysis ({rolling.get('window_size', 'N/A')} days):",
                f"  Stable Periods Ratio: {rolling.get('stable_ratio', 0):.2%}",
                f"  Mean Stability: {rolling.get('mean_stability', 0):.4f}",
                f"  Best Period: {rolling.get('best_period', 'N/A')} ({rolling.get('best_value', 0):.4f})",
                f"  Worst Period: {rolling.get('worst_period', 'N/A')} ({rolling.get('worst_value', 0):.4f})",
                "",
            ])
        
        # 结构突变
        if 'structural_breaks' in results:
            breaks = results['structural_breaks']
            if breaks.get('detected'):
                summary_lines.extend([
                    "Structural Breaks Detected:",
                    f"  P-value: {breaks.get('p_value', 'N/A'):.4f}",
                ])
                if 'breakpoints' in breaks:
                    summary_lines.append(f"  Breakpoints: {len(breaks['breakpoints'])} detected")
                    for bp in breaks['breakpoints'][:5]:  # 显示前5个
                        summary_lines.append(f"    - {bp}")
            else:
                summary_lines.append("No Structural Breaks Detected")
            summary_lines.append("")
        
        # 市场状态分析
        if 'market_state_analysis' in results:
            market = results['market_state_analysis']
            if isinstance(market, dict) and 'cross_state_stability' in market:
                summary_lines.extend([
                    "Market State Performance:",
                    f"  Cross-state Stability: {market.get('cross_state_stability', 0):.4f}",
                ])
                for state in ['bull', 'bear', 'volatile', 'neutral']:
                    if state in market:
                        state_data = market[state]
                        summary_lines.append(
                            f"  {state.capitalize()}: Mean={state_data['mean']:.4f}, "
                            f"Sharpe={state_data['sharpe']:.2f}, "
                            f"Count={state_data['count']}"
                        )
                summary_lines.append("")
        
        # 衰减分析
        if 'decay_analysis' in results:
            decay = results['decay_analysis']
            if isinstance(decay, dict) and 'half_life' in decay:
                summary_lines.extend([
                    "Factor Decay Analysis:",
                    f"  Half-life: {decay.get('half_life', 'N/A')} periods",
                    f"  Decay Rate: {decay.get('decay_rate', 0):.6f}",
                    f"  Effective Period: {decay.get('effective_period', 'N/A')} periods",
                    f"  Decay Pattern: {decay.get('decay_pattern', 'N/A')}",
                ])
        
        return "\n".join(summary_lines)