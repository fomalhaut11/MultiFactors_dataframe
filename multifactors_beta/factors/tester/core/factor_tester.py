"""
单因子测试核心引擎
实现回归分析、分组测试、IC分析等核心功能
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, Any, List, Union
import logging
import statsmodels.api as sm
from scipy.stats import spearmanr
from dataclasses import dataclass

from core.utils import OutlierHandler, Normalizer
from ..base.test_result import TestResult, RegressionResult, GroupResult, ICResult

logger = logging.getLogger(__name__)


class FactorTester:
    """单因子测试器"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化测试器
        
        Parameters
        ----------
        config : Dict, optional
            测试配置
        """
        self.config = config or {}
        self.group_nums = self.config.get('group_nums', 10)
        # 使用因子测试阶段的配置参数
        self.outlier_method = self.config.get('outlier_method') or self._get_testing_config('outlier_method')
        self.outlier_param = self.config.get('outlier_param') or self._get_testing_config('outlier_param')
        self.normalization_method = self.config.get('normalization_method') or self._get_testing_config('normalization_method')
        
        # 新增：股票池支持
        self.stock_universe = None
        
        logger.info(f"FactorTester初始化完成 - outlier_param: {self.outlier_param} (测试阶段配置)")
        
    def test(self, test_data: Dict[str, Any], stock_universe: Optional[pd.Series] = None) -> TestResult:
        """
        执行完整的单因子测试
        
        Parameters
        ----------
        test_data : Dict
            测试数据（由DataManager准备）
        stock_universe : pd.Series, optional
            股票池，MultiIndex[TradingDates, StockCodes] Series格式，值为1
            
        Returns
        -------
        TestResult
            测试结果
        """
        # 设置股票池（保持向后兼容性）
        self.stock_universe = stock_universe
        # 创建结果对象
        result = TestResult(
            factor_name=test_data.get('data_info', {}).get('factor_name', 'unknown'),
            config_snapshot=self.config.copy(),
            data_info=test_data.get('data_info', {})
        )
        
        # 检查数据
        if 'factor' not in test_data or test_data['factor'].empty:
            result.errors.append("因子数据为空")
            return result
        
        if 'returns' not in test_data or test_data['returns'].empty:
            result.errors.append("收益率数据为空")
            return result
        
        try:
            # 合并数据
            merged_data = self._prepare_merged_data(test_data)
            if merged_data.empty:
                result.errors.append("数据合并失败")
                return result
            
            # 执行测试
            logger.info("开始回归分析测试")
            regression_result = self._regression_test(merged_data)
            result.regression_result = regression_result
            
            logger.info("开始分组测试")
            group_result = self._group_test(merged_data)
            result.group_result = group_result
            
            logger.info("开始IC分析")
            ic_result = self._ic_test(merged_data)
            result.ic_result = ic_result
            
            # 保存处理后的因子
            if 'newfactor' in merged_data.columns:
                result.processed_factor = merged_data['newfactor']
            
            # 换手率分析
            logger.info("开始换手率分析")
            turnover_result = self._turnover_test(merged_data)
            result.turnover_result = turnover_result
            
            # 计算性能指标
            result.calculate_performance_metrics()
            
            logger.info("测试完成")
            
        except Exception as e:
            logger.error(f"测试过程出错: {e}")
            result.errors.append(str(e))
        
        return result
    
    def _prepare_merged_data(self, test_data: Dict[str, Any]) -> pd.DataFrame:
        """
        准备合并的数据
        
        Parameters
        ----------
        test_data : Dict
            测试数据
            
        Returns
        -------
        pd.DataFrame
            合并后的数据
        """
        # 获取数据
        factor = test_data['factor']
        returns = test_data['returns']
        control_vars = test_data.get('control_variables', pd.DataFrame())
        
        # 确保factor是DataFrame
        if isinstance(factor, pd.Series):
            factor = factor.to_frame(name='factor')
        
        # 确保returns是DataFrame
        if isinstance(returns, pd.Series):
            returns_df = returns.to_frame('LogReturn')
        else:
            returns_df = returns
            if 'LogReturn' not in returns_df.columns:
                returns_df.columns = ['LogReturn']
        
        # 合并数据
        if not control_vars.empty:
            merged = pd.concat([factor, returns_df, control_vars], 
                             axis=1, join='inner')
        else:
            merged = pd.concat([factor, returns_df], 
                             axis=1, join='inner')
        
        # 删除缺失值
        merged = merged.dropna(subset=['LogReturn'])
        
        # 新增：应用股票池过滤
        if self.stock_universe is not None:
            merged = self._apply_stock_universe_filter(merged, self.stock_universe)
        
        # 按日期分组处理
        processed_list = []
        for date, daily_data in merged.groupby(level=0):
            if len(daily_data) < 10:  # 样本太少，跳过
                continue
            
            # 处理因子
            processed = self._process_daily_factor(daily_data)
            if processed is not None and not processed.empty:
                processed_list.append(processed)
        
        if processed_list:
            return pd.concat(processed_list)
        else:
            return pd.DataFrame()
    
    def _process_daily_factor(self, daily_data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        处理单日因子数据
        
        Parameters
        ----------
        daily_data : pd.DataFrame
            单日数据
            
        Returns
        -------
        pd.DataFrame
            处理后的数据
        """
        try:
            data = daily_data.copy()
            
            # 去极值
            factor_values = OutlierHandler.remove_outlier(
                data['factor'], 
                method=self.outlier_method, 
                param=self.outlier_param
            )
            
            # 标准化
            factor_values = Normalizer.normalize(
                factor_values, 
                method=self.normalization_method
            )
            
            # 填充缺失值
            factor_values = factor_values.fillna(0)
            
            # 检查标准差
            if factor_values.std() <= 1e-6:
                # 标准差太小，使用原始值重新标准化
                factor_values = Normalizer.normalize(data['factor'], method=self.normalization_method)
            
            data['processed_factor'] = factor_values
            
            # 如果有控制变量，进行正交化
            control_cols = [col for col in data.columns 
                          if col not in ['factor', 'LogReturn', 'processed_factor']]
            
            if control_cols:
                # 准备回归数据
                X = data[control_cols].fillna(0)
                
                # 安全删除全0列
                valid_cols_mask = (X != 0).any(axis=0)
                X_valid = X.loc[:, valid_cols_mask]
                
                # 检查是否有有效列
                if X_valid.empty or len(X_valid.columns) == 0:
                    logger.warning(f"日期{data.index[0][0] if len(data) > 0 else 'unknown'}:所有控制变量为零，跳过正交化")
                    data['newfactor'] = data['processed_factor']
                    return data
                
                y = data['processed_factor']
                
                # 检查矩阵秩
                X_with_const = sm.add_constant(X_valid)
                try:
                    rank = np.linalg.matrix_rank(X_with_const)
                    
                    if rank < X_with_const.shape[1]:
                        # 矩阵不满秩，使用岭回归
                        try:
                            from sklearn.linear_model import Ridge
                            ridge = Ridge(alpha=1e-4)
                            ridge.fit(X_valid, y)
                            residuals = y - ridge.predict(X_valid)
                            logger.info(f"使用岭回归处理不满秩矩阵")
                        except (ImportError, Exception) as e:
                            # sklearn不可用或版本问题，使用原始因子
                            logger.warning(f"sklearn回归失败({e})，跳过正交化")
                            residuals = y
                    else:
                        # 矩阵满秩，使用OLS
                        model = sm.OLS(y, X_with_const)
                        result = model.fit()
                        residuals = result.resid
                    
                    # 标准化残差作为新因子
                    newfactor = Normalizer.normalize(residuals, method='zscore')
                    data['newfactor'] = newfactor
                    
                except np.linalg.LinAlgError as e:
                    logger.warning(f"矩阵计算错误: {e}，使用原始因子")
                    data['newfactor'] = data['processed_factor']
                    
            else:
                # 没有控制变量，直接使用处理后的因子
                data['newfactor'] = data['processed_factor']
            
            return data
            
        except Exception as e:
            logger.warning(f"处理单日因子失败: {e}")
            return None
    
    def _regression_test(self, merged_data: pd.DataFrame) -> RegressionResult:
        """
        执行回归分析测试
        
        Parameters
        ----------
        merged_data : pd.DataFrame
            合并后的数据
            
        Returns
        -------
        RegressionResult
            回归分析结果
        """
        daily_results = []
        
        for date, daily_data in merged_data.groupby(level=0):
            if len(daily_data) < 10:
                continue
            
            try:
                # 准备数据
                y = daily_data['LogReturn']
                
                # 构建自变量
                X_cols = ['newfactor']
                control_cols = [col for col in daily_data.columns 
                              if col not in ['factor', 'LogReturn', 'processed_factor', 'newfactor']]
                X_cols.extend(control_cols)
                
                X = daily_data[X_cols].fillna(0)
                X = sm.add_constant(X)
                
                # OLS回归
                model = sm.OLS(y, X)
                result = model.fit()
                
                # 收集结果
                daily_result = {
                    'date': date,
                    'factor_return': result.params.get('newfactor', 0),
                    'factor_tvalue': result.tvalues.get('newfactor', 0),
                    'factor_pvalue': result.pvalues.get('newfactor', 0),
                    'rsquared_adj': result.rsquared_adj
                }
                daily_results.append(daily_result)
                
            except Exception as e:
                logger.warning(f"日期{date}回归失败: {e}")
                continue
        
        # 汇总结果
        if daily_results:
            results_df = pd.DataFrame(daily_results).set_index('date')
            
            return RegressionResult(
                params=results_df['factor_return'],
                tvalues=results_df['factor_tvalue'],
                pvalues=results_df['factor_pvalue'],
                resid=pd.Series(),  # 暂不保存残差
                rsquared_adj=results_df['rsquared_adj'].mean(),
                factor_return=results_df['factor_return'],
                cumulative_return=results_df['factor_return'].cumsum()
            )
        else:
            # 返回空结果
            return RegressionResult(
                params=pd.Series(),
                tvalues=pd.Series(),
                pvalues=pd.Series(),
                resid=pd.Series(),
                rsquared_adj=0,
                factor_return=pd.Series(),
                cumulative_return=pd.Series()
            )
    
    def _group_test(self, merged_data: pd.DataFrame) -> GroupResult:
        """
        执行分组测试
        
        Parameters
        ----------
        merged_data : pd.DataFrame
            合并后的数据
            
        Returns
        -------
        GroupResult
            分组测试结果
        """
        group_returns_list = []
        group_counts_list = []
        
        for date, daily_data in merged_data.groupby(level=0):
            if len(daily_data) < self.group_nums:
                continue
            
            try:
                # 按因子值分组
                daily_data['group'] = pd.qcut(
                    daily_data['newfactor'], 
                    self.group_nums, 
                    labels=False, 
                    duplicates='drop'
                )
                
                # 计算各组收益
                group_stats = daily_data.groupby('group').agg({
                    'LogReturn': 'mean',
                    'factor': 'count'  # 股票数量
                })
                
                # 确保所有组都有值
                group_returns = pd.Series(index=range(self.group_nums), dtype=float)
                group_counts = pd.Series(index=range(self.group_nums), dtype=float)
                
                for g in group_stats.index:
                    group_returns[g] = group_stats.loc[g, 'LogReturn']
                    group_counts[g] = group_stats.loc[g, 'factor']
                
                group_returns = group_returns.fillna(0)
                group_counts = group_counts.fillna(0)
                
                # 添加日期索引
                group_returns.name = date
                group_counts.name = date
                
                group_returns_list.append(group_returns)
                group_counts_list.append(group_counts)
                
            except Exception as e:
                logger.warning(f"日期{date}分组测试失败: {e}")
                continue
        
        # 汇总结果
        if group_returns_list:
            group_returns_df = pd.DataFrame(group_returns_list)
            group_counts_df = pd.DataFrame(group_counts_list)
            
            # 计算多空收益
            long_short = group_returns_df[self.group_nums - 1] - group_returns_df[0]
            
            # 计算单调性得分（Spearman相关系数）
            monotonicity_scores = []
            for _, row in group_returns_df.iterrows():
                if row.notna().sum() >= 3:  # 至少3个有效值
                    score, _ = spearmanr(range(len(row)), row.fillna(row.mean()))
                    monotonicity_scores.append(score)
            
            monotonicity = np.mean(monotonicity_scores) if monotonicity_scores else 0
            
            # 计算累计收益
            group_cumulative = group_returns_df.cumsum()
            
            return GroupResult(
                group_returns=group_returns_df,
                group_counts=group_counts_df,
                long_short_return=long_short,
                group_cumulative_returns=group_cumulative,
                monotonicity_score=monotonicity
            )
        else:
            # 返回空结果
            return GroupResult(
                group_returns=pd.DataFrame(),
                group_counts=pd.DataFrame(),
                long_short_return=pd.Series(),
                group_cumulative_returns=pd.DataFrame(),
                monotonicity_score=0
            )
    
    def _ic_test(self, merged_data: pd.DataFrame) -> ICResult:
        """
        执行IC分析
        
        Parameters
        ----------
        merged_data : pd.DataFrame
            合并后的数据
            
        Returns
        -------
        ICResult
            IC分析结果
        """
        ic_list = []
        rank_ic_list = []
        
        for date, daily_data in merged_data.groupby(level=0):
            if len(daily_data) < 10:
                continue
            
            try:
                # 计算IC（Pearson相关系数）
                ic = daily_data['newfactor'].corr(daily_data['LogReturn'])
                
                # 计算Rank IC（Spearman相关系数）
                rank_ic, _ = spearmanr(daily_data['newfactor'], daily_data['LogReturn'])
                
                ic_list.append({'date': date, 'ic': ic, 'rank_ic': rank_ic})
                
            except Exception as e:
                logger.warning(f"日期{date}IC计算失败: {e}")
                continue
        
        # 汇总结果
        if ic_list:
            ic_df = pd.DataFrame(ic_list).set_index('date')
            
            # 计算IC衰减（未来N期的IC）
            ic_decay = self._calculate_ic_decay(merged_data, periods=5)
            
            return ICResult(
                ic_series=ic_df['ic'],
                rank_ic_series=ic_df['rank_ic'],
                ic_mean=ic_df['ic'].mean(),
                ic_std=ic_df['ic'].std(),
                icir=ic_df['ic'].mean() / ic_df['ic'].std() if ic_df['ic'].std() > 0 else 0,
                rank_ic_mean=ic_df['rank_ic'].mean(),
                rank_icir=ic_df['rank_ic'].mean() / ic_df['rank_ic'].std() if ic_df['rank_ic'].std() > 0 else 0,
                ic_decay=ic_decay
            )
        else:
            # 返回空结果
            return ICResult(
                ic_series=pd.Series(),
                rank_ic_series=pd.Series(),
                ic_mean=0,
                ic_std=0,
                icir=0,
                rank_ic_mean=0,
                rank_icir=0,
                ic_decay=pd.Series()
            )
    
    def _calculate_ic_decay(self, merged_data: pd.DataFrame, periods: int = 5) -> pd.Series:
        """
        计算IC衰减
        
        Parameters
        ----------
        merged_data : pd.DataFrame
            合并后的数据
        periods : int
            计算周期数
            
        Returns
        -------
        pd.Series
            IC衰减序列
        """
        ic_decay = pd.Series(index=range(1, periods + 1), dtype=float)
        
        # 获取所有交易日期
        dates = sorted(merged_data.index.get_level_values(0).unique())
        
        for lag in range(1, periods + 1):
            ic_values = []
            
            for i, date in enumerate(dates[:-lag]):
                # 当期因子值
                current_data = merged_data.loc[date]
                if len(current_data) < 10:
                    continue
                    
                # 未来第lag期的收益
                future_date = dates[i + lag]
                if future_date in merged_data.index.get_level_values(0):
                    future_data = merged_data.loc[future_date]
                    
                    # 匹配相同股票
                    common_stocks = current_data.index.intersection(future_data.index)
                    if len(common_stocks) >= 10:
                        current_factor = current_data.loc[common_stocks, 'newfactor']
                        future_return = future_data.loc[common_stocks, 'LogReturn']
                        
                        # 计算IC
                        ic = current_factor.corr(future_return)
                        if not np.isnan(ic):
                            ic_values.append(ic)
            
            # 计算该滞后期的平均IC
            if ic_values:
                ic_decay[lag] = np.mean(ic_values)
            else:
                ic_decay[lag] = 0
        
        return ic_decay
    
    def _turnover_test(self, merged_data: pd.DataFrame) -> Dict[str, Any]:
        """
        执行换手率分析
        
        Parameters
        ----------
        merged_data : pd.DataFrame
            合并后的数据
            
        Returns
        -------
        Dict
            换手率分析结果
        """
        # 初始化结果
        turnover_results = {
            'daily_turnover': [],  # 每日换手率
            'group_turnover': {},   # 各组换手率
            'avg_turnover': 0,      # 平均换手率
            'turnover_cost': [],    # 换手成本估算
        }
        
        # 按日期排序
        dates = sorted(merged_data.index.get_level_values(0).unique())
        
        # 记录前一期的分组
        prev_groups = {}
        
        for i, date in enumerate(dates):
            daily_data = merged_data.loc[date]
            
            if len(daily_data) < self.group_nums:
                continue
            
            try:
                # 按因子值分组
                daily_data_copy = daily_data.copy()
                daily_data_copy['group'] = pd.qcut(
                    daily_data_copy['newfactor'], 
                    self.group_nums, 
                    labels=False, 
                    duplicates='drop'
                )
                
                # 获取每组的股票列表
                current_groups = {}
                for g in range(self.group_nums):
                    group_stocks = daily_data_copy[daily_data_copy['group'] == g].index.tolist()
                    current_groups[g] = set(group_stocks)
                
                # 计算换手率（如果有前一期数据）
                if prev_groups:
                    daily_turnover = []
                    
                    for g in range(self.group_nums):
                        if g in prev_groups and g in current_groups:
                            prev_stocks = prev_groups[g]
                            curr_stocks = current_groups[g]
                            
                            # 计算该组的换手率
                            # 换手率 = (卖出股票数 + 买入股票数) / (2 * 平均持仓数)
                            stocks_sold = len(prev_stocks - curr_stocks)
                            stocks_bought = len(curr_stocks - prev_stocks)
                            avg_holdings = (len(prev_stocks) + len(curr_stocks)) / 2
                            
                            if avg_holdings > 0:
                                group_turnover = (stocks_sold + stocks_bought) / (2 * avg_holdings)
                                daily_turnover.append(group_turnover)
                                
                                # 记录各组换手率
                                if g not in turnover_results['group_turnover']:
                                    turnover_results['group_turnover'][g] = []
                                turnover_results['group_turnover'][g].append({
                                    'date': date,
                                    'turnover': group_turnover,
                                    'stocks_sold': stocks_sold,
                                    'stocks_bought': stocks_bought
                                })
                    
                    # 计算当日平均换手率
                    if daily_turnover:
                        avg_daily_turnover = np.mean(daily_turnover)
                        turnover_results['daily_turnover'].append({
                            'date': date,
                            'turnover': avg_daily_turnover
                        })
                        
                        # 估算换手成本（假设单边成本0.15%）
                        cost_rate = 0.0015
                        turnover_cost = avg_daily_turnover * cost_rate * 2  # 双边成本
                        turnover_results['turnover_cost'].append({
                            'date': date,
                            'cost': turnover_cost
                        })
                
                # 更新前一期分组
                prev_groups = current_groups
                
            except Exception as e:
                logger.warning(f"日期{date}换手率计算失败: {e}")
                continue
        
        # 计算汇总统计
        if turnover_results['daily_turnover']:
            turnovers = [x['turnover'] for x in turnover_results['daily_turnover']]
            turnover_results['avg_turnover'] = np.mean(turnovers)
            turnover_results['max_turnover'] = np.max(turnovers)
            turnover_results['min_turnover'] = np.min(turnovers)
            turnover_results['turnover_std'] = np.std(turnovers)
            
            # 计算总换手成本
            if turnover_results['turnover_cost']:
                costs = [x['cost'] for x in turnover_results['turnover_cost']]
                turnover_results['total_cost'] = np.sum(costs)
                turnover_results['avg_cost'] = np.mean(costs)
        
        return turnover_results
    
    def _apply_stock_universe_filter(self, data: pd.DataFrame, stock_universe: pd.Series) -> pd.DataFrame:
        """
        应用股票池过滤
        
        Parameters
        ----------
        data : pd.DataFrame
            待过滤的数据，MultiIndex格式 (date, stock_code)
        stock_universe : pd.Series
            股票池，MultiIndex[TradingDates, StockCodes] Series格式，值为1
            支持时变股票池（不同交易日可以有不同的股票组合）
            
        Returns
        -------
        pd.DataFrame
            过滤后的数据
        """
        if data.empty:
            return data
        
        if stock_universe is None:
            return data
            
        try:
            # 统一使用MultiIndex Series格式处理
            return self._apply_multiindex_universe_filter(data, stock_universe)
                
        except Exception as e:
            logger.warning(f"股票池过滤失败，将使用原始数据: {e}")
            return data
    
    def _apply_multiindex_universe_filter(self, data: pd.DataFrame, stock_universe: pd.Series) -> pd.DataFrame:
        """
        应用MultiIndex Series格式股票池过滤（时变股票池）
        
        Parameters
        ----------
        data : pd.DataFrame
            待过滤的数据，MultiIndex[TradingDates, StockCodes]
        stock_universe : pd.Series
            股票池，MultiIndex[TradingDates, StockCodes] Series，值为1
        
        Returns
        -------
        pd.DataFrame
            过滤后的数据，只保留股票池中存在的日期-股票对
        """
        if not isinstance(data.index, pd.MultiIndex):
            logger.warning("数据不是MultiIndex格式，无法使用时变股票池过滤")
            return data
            
        if not isinstance(stock_universe.index, pd.MultiIndex):
            logger.warning("股票池不是MultiIndex格式，无法进行时变过滤")
            return data
        
        # 获取股票池的索引
        universe_index = stock_universe.index
        
        # 计算交集：只保留同时在data和stock_universe中存在的(日期, 股票)对
        common_index = data.index.intersection(universe_index)
        
        if len(common_index) == 0:
            logger.warning("数据与股票池没有交集，返回空DataFrame")
            return pd.DataFrame(columns=data.columns)
        
        # 过滤数据
        filtered_data = data.loc[common_index]
        
        # 记录过滤效果
        original_records = len(data)
        filtered_records = len(filtered_data)
        original_dates = len(data.index.get_level_values(0).unique())
        filtered_dates = len(filtered_data.index.get_level_values(0).unique()) if not filtered_data.empty else 0
        original_stocks = len(data.index.get_level_values(1).unique())
        filtered_stocks = len(filtered_data.index.get_level_values(1).unique()) if not filtered_data.empty else 0
        
        logger.info(f"时变股票池过滤完成：")
        logger.info(f"  日期范围：{original_dates} → {filtered_dates} 个交易日")
        logger.info(f"  股票范围：{original_stocks} → {filtered_stocks} 只股票")
        logger.info(f"  数据量：{original_records} → {filtered_records} 条记录")
        logger.info(f"  过滤率：{(1 - filtered_records/original_records)*100:.1f}%")
        
        return filtered_data
    
    def _get_testing_config(self, param: str):
        """
        获取因子测试阶段的配置参数
        
        Parameters:
        -----------
        param : str
            参数名称
            
        Returns:
        --------
        配置参数值
        """
        from config import get_config
        
        try:
            # 优先使用测试阶段配置
            return get_config(f'main.data_processing.factor_testing.{param}')
        except Exception:
            try:
                # 降级到默认配置
                return get_config(f'main.data_processing.{param}')
            except Exception:
                # 硬编码默认值
                defaults = {
                    'outlier_method': 'IQR',
                    'outlier_param': 3,  # 测试阶段使用更严格的参数
                    'normalization_method': 'zscore'
                }
                return defaults.get(param)