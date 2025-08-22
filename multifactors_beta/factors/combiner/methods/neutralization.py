"""
中性化方法

实现因子的风格中性化和行业中性化
"""

from typing import Dict, Optional, Any, List, Union
import pandas as pd
import numpy as np
import logging
from sklearn.linear_model import LinearRegression
import warnings

logger = logging.getLogger(__name__)


class NeutralizationCombiner:
    """
    中性化组合器
    
    提供多种中性化方法，消除因子与风格/行业的相关性
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化中性化组合器
        
        Parameters
        ----------
        config : Dict[str, Any], optional
            配置参数
        """
        self.config = config or {}
        self.method = self.config.get('neutralization_method', 'regression')  # 'regression' or 'demean'
        self.handle_missing = self.config.get('handle_missing', 'drop')
        self.normalize_after = self.config.get('normalize_after', True)
        self.min_observations = self.config.get('min_observations', 10)
        self.industry_neutral = self.config.get('industry_neutral', True)
        self.style_neutral = self.config.get('style_neutral', True)
        self.market_neutral = self.config.get('market_neutral', False)
    
    def neutralize(self,
                  factors: Dict[str, pd.Series],
                  risk_factors: Optional[Dict[str, pd.Series]] = None,
                  industry_data: Optional[pd.DataFrame] = None,
                  style_factors: Optional[Dict[str, pd.Series]] = None) -> Dict[str, pd.Series]:
        """
        中性化因子
        
        Parameters
        ----------
        factors : Dict[str, pd.Series]
            原始因子字典
        risk_factors : Dict[str, pd.Series], optional
            风险因子（如市值、Beta等）
        industry_data : pd.DataFrame, optional
            行业分类数据（列为行业，值为0/1）
        style_factors : Dict[str, pd.Series], optional
            风格因子
            
        Returns
        -------
        Dict[str, pd.Series]
            中性化后的因子
        """
        if not factors:
            raise ValueError("No factors to neutralize")
        
        # 对齐因子
        aligned_factors = self._align_factors(factors)
        
        # 构建中性化因子集
        neutralization_factors = self._build_neutralization_factors(
            risk_factors, industry_data, style_factors
        )
        
        if not neutralization_factors:
            logger.warning("No neutralization factors provided, returning original factors")
            return aligned_factors
        
        # 执行中性化
        if self.method == 'regression':
            return self._regression_neutralization(aligned_factors, neutralization_factors)
        elif self.method == 'demean':
            return self._demean_neutralization(aligned_factors, neutralization_factors)
        else:
            raise ValueError(f"Unknown neutralization method: {self.method}")
    
    def _align_factors(self, factors: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        对齐因子索引
        """
        # 找到公共索引
        common_index = None
        for factor in factors.values():
            if common_index is None:
                common_index = factor.index
            else:
                common_index = common_index.intersection(factor.index)
        
        if len(common_index) == 0:
            raise ValueError("No common index found among factors")
        
        # 对齐
        aligned = {}
        for name, factor in factors.items():
            aligned[name] = factor.reindex(common_index)
        
        return aligned
    
    def _build_neutralization_factors(self,
                                     risk_factors: Optional[Dict[str, pd.Series]],
                                     industry_data: Optional[pd.DataFrame],
                                     style_factors: Optional[Dict[str, pd.Series]]) -> Dict[str, pd.Series]:
        """
        构建中性化因子集
        """
        neutralization_factors = {}
        
        # 添加风险因子
        if risk_factors and self.style_neutral:
            for name, factor in risk_factors.items():
                neutralization_factors[f'risk_{name}'] = factor
        
        # 添加行业因子
        if industry_data is not None and self.industry_neutral:
            # 将DataFrame转换为Series字典
            for col in industry_data.columns:
                neutralization_factors[f'industry_{col}'] = industry_data[col]
        
        # 添加风格因子
        if style_factors and self.style_neutral:
            for name, factor in style_factors.items():
                neutralization_factors[f'style_{name}'] = factor
        
        # 添加市场因子（全为1的向量）
        if self.market_neutral and neutralization_factors:
            # 使用第一个因子的索引
            first_factor = next(iter(neutralization_factors.values()))
            market_factor = pd.Series(1, index=first_factor.index, name='market')
            neutralization_factors['market'] = market_factor
        
        return neutralization_factors
    
    def _regression_neutralization(self,
                                  factors: Dict[str, pd.Series],
                                  neutralization_factors: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        回归中性化
        """
        neutralized_factors = {}
        
        for factor_name, factor in factors.items():
            # 获取日期
            dates = factor.index.get_level_values(0).unique()
            
            # 存储中性化后的结果
            neutralized_results = []
            
            for date in dates:
                try:
                    # 获取当日数据
                    factor_day = factor.xs(date, level=0)
                    
                    # 构建中性化因子矩阵
                    X_list = []
                    valid_stocks = factor_day.index
                    
                    for neut_name, neut_factor in neutralization_factors.items():
                        if isinstance(neut_factor, pd.Series):
                            if neut_factor.index.nlevels == 2:  # MultiIndex
                                try:
                                    neut_day = neut_factor.xs(date, level=0)
                                    common_stocks = valid_stocks.intersection(neut_day.index)
                                    if len(common_stocks) > 0:
                                        X_list.append(neut_day.reindex(common_stocks).fillna(0).values)
                                        valid_stocks = common_stocks
                                except KeyError:
                                    continue
                            else:  # 单层索引（如行业数据）
                                common_stocks = valid_stocks.intersection(neut_factor.index)
                                if len(common_stocks) > 0:
                                    X_list.append(neut_factor.reindex(common_stocks).fillna(0).values)
                                    valid_stocks = common_stocks
                    
                    if not X_list or len(valid_stocks) < self.min_observations:
                        continue
                    
                    # 准备数据
                    X = np.column_stack(X_list) if len(X_list) > 1 else X_list[0].reshape(-1, 1)
                    y = factor_day.reindex(valid_stocks).values
                    
                    # 处理缺失值
                    valid_mask = ~np.isnan(y)
                    if self.handle_missing == 'drop':
                        # 删除任何包含缺失值的行
                        for i in range(X.shape[1]):
                            valid_mask &= ~np.isnan(X[:, i])
                    
                    if valid_mask.sum() < self.min_observations:
                        continue
                    
                    X_valid = X[valid_mask]
                    y_valid = y[valid_mask]
                    
                    # 回归
                    reg = LinearRegression()
                    reg.fit(X_valid, y_valid)
                    
                    # 计算残差
                    predictions = reg.predict(X)
                    residuals = y - predictions
                    
                    # 标准化残差（可选）
                    if self.normalize_after:
                        std = np.nanstd(residuals)
                        if std > 0:
                            residuals = (residuals - np.nanmean(residuals)) / std
                    
                    # 保存结果
                    for i, stock in enumerate(valid_stocks):
                        if not np.isnan(residuals[i]):
                            neutralized_results.append((date, stock, residuals[i]))
                            
                except Exception as e:
                    logger.warning(f"Failed to neutralize {factor_name} on {date}: {e}")
                    continue
            
            # 构建中性化后的Series
            if neutralized_results:
                index = pd.MultiIndex.from_tuples(
                    [(d, s) for d, s, _ in neutralized_results]
                )
                values = [v for _, _, v in neutralized_results]
                neutralized_factors[factor_name] = pd.Series(
                    values,
                    index=index,
                    name=f'{factor_name}_neutralized'
                )
            else:
                neutralized_factors[factor_name] = pd.Series(
                    [],
                    index=pd.MultiIndex.from_tuples([]),
                    name=f'{factor_name}_neutralized'
                )
        
        logger.info(f"Neutralized {len(factors)} factors using regression method")
        return neutralized_factors
    
    def _demean_neutralization(self,
                              factors: Dict[str, pd.Series],
                              neutralization_factors: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        去均值中性化
        """
        neutralized_factors = {}
        
        # 找出行业因子
        industry_factors = {k: v for k, v in neutralization_factors.items() 
                          if k.startswith('industry_')}
        
        if not industry_factors:
            logger.warning("No industry factors found for demean neutralization")
            return factors
        
        for factor_name, factor in factors.items():
            # 获取日期
            dates = factor.index.get_level_values(0).unique()
            
            # 存储中性化后的结果
            neutralized_results = []
            
            for date in dates:
                try:
                    # 获取当日数据
                    factor_day = factor.xs(date, level=0)
                    
                    # 对每个行业进行去均值
                    for ind_name, ind_factor in industry_factors.items():
                        # 获取行业分类
                        if ind_factor.index.nlevels == 2:
                            try:
                                ind_day = ind_factor.xs(date, level=0)
                            except KeyError:
                                continue
                        else:
                            ind_day = ind_factor
                        
                        # 找出属于该行业的股票
                        industry_stocks = ind_day[ind_day == 1].index
                        common_stocks = factor_day.index.intersection(industry_stocks)
                        
                        if len(common_stocks) > 0:
                            # 计算行业均值
                            industry_mean = factor_day.loc[common_stocks].mean()
                            
                            # 去均值
                            for stock in common_stocks:
                                original_value = factor_day.loc[stock]
                                if not np.isnan(original_value):
                                    neutralized_value = original_value - industry_mean
                                    neutralized_results.append((date, stock, neutralized_value))
                    
                except Exception as e:
                    logger.warning(f"Failed to demean {factor_name} on {date}: {e}")
                    continue
            
            # 构建中性化后的Series
            if neutralized_results:
                # 去重（一个股票可能属于多个行业）
                unique_results = {}
                for date, stock, value in neutralized_results:
                    key = (date, stock)
                    if key not in unique_results:
                        unique_results[key] = value
                
                index = pd.MultiIndex.from_tuples(list(unique_results.keys()))
                values = list(unique_results.values())
                
                # 标准化（可选）
                if self.normalize_after:
                    values = np.array(values)
                    std = np.std(values)
                    if std > 0:
                        values = (values - np.mean(values)) / std
                
                neutralized_factors[factor_name] = pd.Series(
                    values,
                    index=index,
                    name=f'{factor_name}_neutralized'
                )
            else:
                neutralized_factors[factor_name] = pd.Series(
                    [],
                    index=pd.MultiIndex.from_tuples([]),
                    name=f'{factor_name}_neutralized'
                )
        
        logger.info(f"Neutralized {len(factors)} factors using demean method")
        return neutralized_factors
    
    def sector_neutralize(self,
                         factor: pd.Series,
                         sector_data: pd.DataFrame,
                         method: str = 'regression') -> pd.Series:
        """
        板块中性化
        
        Parameters
        ----------
        factor : pd.Series
            原始因子
        sector_data : pd.DataFrame
            板块分类数据
        method : str
            中性化方法
            
        Returns
        -------
        pd.Series
            中性化后的因子
        """
        # 将板块数据转换为因子格式
        sector_factors = {}
        for col in sector_data.columns:
            sector_factors[col] = sector_data[col]
        
        # 执行中性化
        result = self.neutralize(
            {'factor': factor},
            industry_data=sector_data
        )
        
        return result.get('factor', factor)
    
    def style_neutralize(self,
                        factor: pd.Series,
                        style_factors: Dict[str, pd.Series],
                        method: str = 'regression') -> pd.Series:
        """
        风格中性化
        
        Parameters
        ----------
        factor : pd.Series
            原始因子
        style_factors : Dict[str, pd.Series]
            风格因子
        method : str
            中性化方法
            
        Returns
        -------
        pd.Series
            中性化后的因子
        """
        # 执行中性化
        result = self.neutralize(
            {'factor': factor},
            style_factors=style_factors
        )
        
        return result.get('factor', factor)
    
    def calculate_neutralization_stats(self,
                                      original_factors: Dict[str, pd.Series],
                                      neutralized_factors: Dict[str, pd.Series]) -> Dict[str, Dict[str, float]]:
        """
        计算中性化统计信息
        
        Parameters
        ----------
        original_factors : Dict[str, pd.Series]
            原始因子
        neutralized_factors : Dict[str, pd.Series]
            中性化后的因子
            
        Returns
        -------
        Dict[str, Dict[str, float]]
            统计信息
        """
        stats = {}
        
        for factor_name in original_factors:
            if factor_name not in neutralized_factors:
                continue
            
            original = original_factors[factor_name]
            neutralized = neutralized_factors[factor_name]
            
            # 对齐索引
            common_index = original.index.intersection(neutralized.index)
            if len(common_index) == 0:
                continue
            
            original_aligned = original.reindex(common_index)
            neutralized_aligned = neutralized.reindex(common_index)
            
            # 计算统计量
            stats[factor_name] = {
                'original_mean': original_aligned.mean(),
                'original_std': original_aligned.std(),
                'neutralized_mean': neutralized_aligned.mean(),
                'neutralized_std': neutralized_aligned.std(),
                'correlation': original_aligned.corr(neutralized_aligned),
                'r2_reduction': 1 - (neutralized_aligned.var() / original_aligned.var())
            }
        
        return stats