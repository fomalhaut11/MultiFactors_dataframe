"""稳健回归工具模块

处理行业中性化中的奇异矩阵和数据清洗问题
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class RobustIndustryNeutralizer:
    """稳健的行业中性化处理器"""
    
    def __init__(self, min_stocks_per_industry: int = 3, use_ridge: bool = True, ridge_alpha: float = 1e-4):
        """
        初始化稳健行业中性化器
        
        Parameters
        ----------
        min_stocks_per_industry : int
            每个行业最少股票数，低于此数的行业将被忽略
        use_ridge : bool
            是否优先使用岭回归
        ridge_alpha : float
            岭回归正则化参数
        """
        self.min_stocks = min_stocks_per_industry
        self.use_ridge = use_ridge
        self.ridge_alpha = ridge_alpha
    
    def neutralize(self, factor_data: pd.Series, industry_onehot: pd.DataFrame, date: Optional[str] = None) -> pd.Series:
        """
        稳健的行业中性化
        
        Parameters
        ----------
        factor_data : pd.Series
            因子数据
        industry_onehot : pd.DataFrame
            行业独热编码数据
        date : str, optional
            日期（用于日志）
            
        Returns
        -------
        pd.Series
            中性化后的因子
        """
        date_str = str(date) if date else "unknown"
        
        try:
            # 1. 数据对齐
            common_index = factor_data.index.intersection(industry_onehot.index)
            if len(common_index) < 10:
                logger.warning(f"{date_str}: 有效样本过少({len(common_index)})，返回原始因子")
                return factor_data
            
            # 2. 检查行业有效性
            industry_counts = industry_onehot.loc[common_index].sum(axis=0)
            valid_industries = industry_counts >= self.min_stocks
            
            if valid_industries.sum() == 0:
                logger.warning(f"{date_str}: 无有效行业，返回原始因子")
                return factor_data
            
            # 3. 构建回归矩阵
            X = industry_onehot.loc[common_index, valid_industries]
            y = factor_data.loc[common_index]
            
            # 4. 处理无行业股票
            no_industry_mask = X.sum(axis=1) == 0
            if no_industry_mask.any():
                # 剔除无行业股票
                X = X[~no_industry_mask]
                y = y[~no_industry_mask]
                logger.info(f"{date_str}: 剔除{no_industry_mask.sum()}只无行业分类的股票")
            
            if len(X) < 10:
                logger.warning(f"{date_str}: 回归样本过少({len(X)})，返回原始因子")
                return factor_data
            
            # 5. 稳健回归
            residuals = self._robust_regression(X, y, date_str)
            
            # 6. 返回中性化后的因子
            neutralized = pd.Series(index=factor_data.index, dtype=float)
            neutralized.loc[residuals.index] = residuals
            neutralized = neutralized.fillna(factor_data)  # 未处理的保持原值
            
            logger.info(f"{date_str}: 行业中性化完成，使用{len(X)}只股票，{X.shape[1]}个行业")
            return neutralized
            
        except Exception as e:
            logger.error(f"{date_str} 行业中性化失败: {e}")
            return factor_data
    
    def _robust_regression(self, X: pd.DataFrame, y: pd.Series, date_str: str) -> pd.Series:
        """
        执行稳健回归
        
        Parameters
        ----------
        X : pd.DataFrame
            自变量矩阵
        y : pd.Series
            因变量
        date_str : str
            日期字符串
            
        Returns
        -------
        pd.Series
            回归残差
        """
        try:
            # 检查矩阵秩
            rank = np.linalg.matrix_rank(X)
            
            if self.use_ridge or rank < X.shape[1]:
                # 使用岭回归
                try:
                    from sklearn.linear_model import Ridge
                    ridge = Ridge(alpha=self.ridge_alpha)
                    ridge.fit(X, y)
                    residuals = y - ridge.predict(X)
                    logger.debug(f"{date_str}: 使用岭回归(alpha={self.ridge_alpha})")
                    return pd.Series(residuals, index=y.index)
                    
                except (ImportError, Exception) as e:
                    logger.warning(f"{date_str}: sklearn回归失败({e})，尝试OLS回归")
            
            # 尝试OLS回归
            X_with_const = sm.add_constant(X)
            model = sm.OLS(y, X_with_const)
            result = model.fit()
            logger.debug(f"{date_str}: 使用OLS回归，R²={result.rsquared:.4f}")
            return pd.Series(result.resid, index=y.index)
            
        except Exception as e:
            logger.warning(f"{date_str}: 回归失败({e})，返回零残差")
            return pd.Series(0, index=y.index)
    
    def batch_neutralize(self, factor_data: pd.Series, industry_onehot: pd.DataFrame) -> pd.Series:
        """
        批量中性化（按日期分组）
        
        Parameters
        ----------
        factor_data : pd.Series
            MultiIndex因子数据 (date, stock)
        industry_onehot : pd.DataFrame
            MultiIndex行业数据 (date, stock)
            
        Returns
        -------
        pd.Series
            中性化后的因子
        """
        results = []
        
        for date in factor_data.index.get_level_values(0).unique():
            # 获取当日数据
            daily_factor = factor_data.loc[date]
            daily_industry = industry_onehot.loc[date] if date in industry_onehot.index.get_level_values(0) else pd.DataFrame()
            
            if daily_industry.empty:
                logger.warning(f"{date}: 无行业数据，保持原始因子")
                results.append(daily_factor)
                continue
            
            # 中性化
            neutralized_daily = self.neutralize(daily_factor, daily_industry, date)
            results.append(neutralized_daily)
        
        # 合并结果
        return pd.concat(results)


class DataAlignmentHelper:
    """数据对齐辅助工具"""
    
    @staticmethod
    def align_factor_and_industry(
        factor_data: pd.Series, 
        industry_data: pd.DataFrame,
        min_observations: int = 10
    ) -> Dict[str, Any]:
        """
        对齐因子和行业数据
        
        Parameters
        ----------
        factor_data : pd.Series
            因子数据
        industry_data : pd.DataFrame
            行业数据
        min_observations : int
            最少观察值
            
        Returns
        -------
        Dict
            对齐结果
        """
        result = {
            'aligned_factor': pd.Series(dtype=float),
            'aligned_industry': pd.DataFrame(),
            'valid': False,
            'info': {}
        }
        
        try:
            # 基础检查
            if factor_data.empty or industry_data.empty:
                result['info']['error'] = "输入数据为空"
                return result
            
            # 索引对齐
            common_index = factor_data.index.intersection(industry_data.index)
            if len(common_index) < min_observations:
                result['info']['error'] = f"共同样本过少: {len(common_index)}"
                return result
            
            # 对齐数据
            aligned_factor = factor_data.loc[common_index]
            aligned_industry = industry_data.loc[common_index]
            
            # 检查因子有效性
            valid_factor_mask = aligned_factor.notna() & np.isfinite(aligned_factor)
            if valid_factor_mask.sum() < min_observations:
                result['info']['error'] = f"有效因子观察值过少: {valid_factor_mask.sum()}"
                return result
            
            # 检查行业有效性
            industry_coverage = (aligned_industry != 0).any(axis=0)
            valid_industries = industry_coverage.sum()
            if valid_industries == 0:
                result['info']['error'] = "无有效行业"
                return result
            
            # 最终过滤
            final_index = common_index[valid_factor_mask]
            result['aligned_factor'] = aligned_factor.loc[final_index]
            result['aligned_industry'] = aligned_industry.loc[final_index, industry_coverage]
            result['valid'] = True
            result['info'] = {
                'samples': len(final_index),
                'industries': valid_industries,
                'coverage_ratio': len(final_index) / len(factor_data)
            }
            
            return result
            
        except Exception as e:
            result['info']['error'] = f"对齐失败: {e}"
            return result


# 向后兼容的快捷函数
def neutralize_factor_with_industry(
    factor: pd.Series, 
    industry_onehot: pd.DataFrame,
    **kwargs
) -> pd.Series:
    """
    快捷的行业中性化函数
    
    Parameters
    ----------
    factor : pd.Series
        因子数据
    industry_onehot : pd.DataFrame
        行业独热编码
    **kwargs
        传递给RobustIndustryNeutralizer的参数
        
    Returns
    -------
    pd.Series
        中性化后的因子
    """
    neutralizer = RobustIndustryNeutralizer(**kwargs)
    
    if isinstance(factor.index, pd.MultiIndex):
        return neutralizer.batch_neutralize(factor, industry_onehot)
    else:
        return neutralizer.neutralize(factor, industry_onehot)