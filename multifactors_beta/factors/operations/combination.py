#!/usr/bin/env python3
"""
因子组合模块

提供因子组合和变换功能：
- 线性组合：多个因子的加权平均
- 正交化：去除因子间的相关性
- 残差化：对控制变量做回归取残差
- 因子运算：加减乘除等基本运算

所有函数都接受MultiIndex Series格式的数据，并返回相同格式的结果。

Author: AI Assistant
Date: 2025-08-26
"""

import pandas as pd
import numpy as np
from typing import Union, List, Dict, Optional, Tuple
import logging
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def linear_combine(factors: Union[List[pd.Series], Dict[str, pd.Series]],
                  weights: Optional[Union[List[float], Dict[str, float]]] = None,
                  normalize_weights: bool = True,
                  method: str = 'equal') -> pd.Series:
    """
    因子线性组合
    
    将多个因子进行加权平均组合
    
    Parameters
    ----------
    factors : list of pd.Series or dict of pd.Series
        要组合的因子列表或字典，每个因子都是MultiIndex格式[TradingDates, StockCodes]
    weights : list of float or dict of float, optional
        对应的权重。如果不指定，根据method参数确定权重
    normalize_weights : bool, default True
        是否将权重归一化使其和为1
    method : str, default 'equal'
        默认权重方法：
        - 'equal': 等权重
        - 'variance': 按方差倒数加权（低方差权重高）
        - 'ic': 按IC加权（需要提供收益率数据）
        
    Returns
    -------
    pd.Series
        组合后的因子，MultiIndex格式与输入因子相同
        
    Examples
    --------
    >>> # 等权重组合
    >>> combined = linear_combine([factor1, factor2, factor3])
    >>> # 指定权重组合
    >>> combined = linear_combine([factor1, factor2], weights=[0.3, 0.7])
    >>> # 字典形式组合
    >>> combined = linear_combine({'value': factor1, 'quality': factor2}, 
    ...                          weights={'value': 0.4, 'quality': 0.6})
    """
    # 参数验证和处理
    if isinstance(factors, dict):
        factor_names = list(factors.keys())
        factor_series = list(factors.values())
    elif isinstance(factors, list):
        factor_names = [f'factor_{i}' for i in range(len(factors))]
        factor_series = factors
    else:
        raise ValueError("factors必须是Series列表或字典")
    
    if len(factor_series) == 0:
        raise ValueError("至少需要一个因子")
    
    # 验证所有因子的格式
    for i, factor in enumerate(factor_series):
        if not isinstance(factor, pd.Series) or not isinstance(factor.index, pd.MultiIndex):
            raise ValueError(f"第{i}个因子必须是MultiIndex Series格式")
    
    try:
        # 对齐所有因子
        aligned_factors = pd.concat(factor_series, axis=1, keys=factor_names, join='inner')
        
        if aligned_factors.empty:
            logger.warning("因子对齐后数据为空")
            return pd.Series(dtype=float, index=pd.MultiIndex.from_tuples([], names=['TradingDates', 'StockCodes']))
        
        # 确定权重
        if weights is None:
            if method == 'equal':
                weights = [1.0 / len(factor_series)] * len(factor_series)
            elif method == 'variance':
                # 按方差倒数加权
                variances = aligned_factors.var()
                inv_var = 1.0 / (variances + 1e-8)  # 加小数避免除零
                weights = (inv_var / inv_var.sum()).values
            else:
                raise ValueError(f"不支持的权重方法: {method}")
        else:
            if isinstance(weights, dict):
                weights = [weights.get(name, 0.0) for name in factor_names]
            elif not isinstance(weights, (list, tuple)):
                raise ValueError("weights必须是列表、元组或字典")
            
            if len(weights) != len(factor_series):
                raise ValueError("权重数量必须与因子数量一致")
        
        # 权重归一化
        if normalize_weights:
            total_weight = sum(weights)
            if total_weight == 0:
                raise ValueError("权重总和不能为0")
            weights = [w / total_weight for w in weights]
        
        # 计算加权平均
        weighted_factors = aligned_factors * weights
        result = weighted_factors.sum(axis=1)
        
        # 设置名称
        result.name = 'combined_factor'
        
        logger.debug(f"因子线性组合完成，组合{len(factor_series)}个因子，权重: {weights}")
        return result
        
    except Exception as e:
        logger.error(f"因子线性组合失败: {e}")
        raise


def orthogonalize(target_factor: pd.Series,
                 base_factors: Union[pd.Series, List[pd.Series], pd.DataFrame],
                 method: str = 'ols',
                 standardize: bool = True) -> pd.Series:
    """
    因子正交化
    
    将目标因子对基础因子进行回归，返回残差作为正交化后的因子
    
    Parameters
    ----------
    target_factor : pd.Series
        目标因子，MultiIndex格式[TradingDates, StockCodes]
    base_factors : pd.Series, list of pd.Series, or pd.DataFrame
        基础因子，用于正交化
    method : str, default 'ols'
        回归方法：'ols'（普通最小二乘）
    standardize : bool, default True
        是否在回归前标准化因子
        
    Returns
    -------
    pd.Series
        正交化后的残差因子，MultiIndex格式与目标因子相同
        
    Examples
    --------
    >>> # 对市值因子正交化
    >>> ortho_factor = orthogonalize(factor, market_cap)
    >>> # 对多个基础因子正交化
    >>> ortho_factor = orthogonalize(factor, [market_cap, beta, momentum])
    """
    if not isinstance(target_factor, pd.Series) or not isinstance(target_factor.index, pd.MultiIndex):
        raise ValueError("目标因子必须是MultiIndex Series格式")
    
    # 处理基础因子格式
    if isinstance(base_factors, pd.Series):
        base_df = pd.DataFrame({'base_0': base_factors})
    elif isinstance(base_factors, list):
        base_df = pd.DataFrame({f'base_{i}': factor for i, factor in enumerate(base_factors)})
    elif isinstance(base_factors, pd.DataFrame):
        base_df = base_factors.copy()
    else:
        raise ValueError("基础因子必须是Series、Series列表或DataFrame")
    
    try:
        # 对齐数据
        aligned_data = pd.concat([target_factor, base_df], axis=1, join='inner')
        
        if aligned_data.empty:
            logger.warning("数据对齐后为空")
            return pd.Series(dtype=float, index=target_factor.index)
        
        target_col = aligned_data.columns[0]  # 目标因子列
        base_cols = aligned_data.columns[1:]  # 基础因子列
        
        def _orthogonalize_group(group):
            """对单个日期的数据进行正交化"""
            # 去除缺失值
            clean_data = group.dropna()
            if len(clean_data) < len(base_cols) + 2:  # 需要足够的样本
                return group[target_col] * np.nan
            
            y = clean_data[target_col].values
            X = clean_data[base_cols].values
            
            # 标准化（可选）
            if standardize:
                scaler_y = StandardScaler()
                scaler_X = StandardScaler()
                y = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
                X = scaler_X.fit_transform(X)
            
            # 回归
            try:
                reg = LinearRegression(fit_intercept=True)
                reg.fit(X, y)
                residuals = y - reg.predict(X)
                
                # 构造结果
                result = pd.Series(np.nan, index=group.index)
                result.loc[clean_data.index] = residuals
                
                return result
                
            except Exception as e:
                logger.warning(f"回归失败: {e}")
                return group[target_col] * np.nan
        
        # 按日期分组进行正交化
        result = aligned_data.groupby(level=0).apply(_orthogonalize_group)
        
        # 处理多层索引
        if isinstance(result.index, pd.MultiIndex) and result.index.nlevels > 2:
            result.index = result.index.droplevel(0)
        
        # 保持原有名称
        result.name = f"{target_factor.name}_ortho" if target_factor.name else "orthogonalized_factor"
        
        logger.debug(f"因子正交化完成，基础因子数量: {len(base_cols)}")
        return result
        
    except Exception as e:
        logger.error(f"因子正交化失败: {e}")
        raise


def residualize(target_factor: pd.Series,
               control_factors: Union[pd.Series, List[pd.Series], pd.DataFrame],
               add_constant: bool = True) -> pd.Series:
    """
    因子残差化
    
    将目标因子对控制因子进行截面回归，返回残差
    （与orthogonalize类似，但专注于截面回归）
    
    Parameters
    ----------
    target_factor : pd.Series
        目标因子，MultiIndex格式[TradingDates, StockCodes]
    control_factors : pd.Series, list of pd.Series, or pd.DataFrame
        控制因子，用于回归
    add_constant : bool, default True
        是否添加常数项
        
    Returns
    -------
    pd.Series
        残差因子，MultiIndex格式与目标因子相同
        
    Examples
    --------
    >>> # 获取去除市值影响的因子
    >>> residual_factor = residualize(factor, market_cap)
    >>> # 去除多个控制变量的影响
    >>> residual_factor = residualize(factor, [size, beta, momentum])
    """
    # 调用orthogonalize，两者实现基本相同
    return orthogonalize(
        target_factor=target_factor,
        base_factors=control_factors,
        method='ols',
        standardize=False  # residualize通常不标准化
    )


def factor_operation(factor1: pd.Series,
                    factor2: Optional[pd.Series] = None,
                    operation: str = 'add',
                    constant: Optional[float] = None) -> pd.Series:
    """
    因子基本运算
    
    对因子进行基本的数学运算
    
    Parameters
    ----------
    factor1 : pd.Series
        第一个因子，MultiIndex格式[TradingDates, StockCodes]
    factor2 : pd.Series, optional
        第二个因子，如果不提供则使用constant
    operation : str, default 'add'
        运算类型：'add', 'subtract', 'multiply', 'divide', 'power'
    constant : float, optional
        常数值，当factor2不提供时使用
        
    Returns
    -------
    pd.Series
        运算结果，MultiIndex格式与输入因子相同
        
    Examples
    --------
    >>> # 因子相加
    >>> sum_factor = factor_operation(factor1, factor2, 'add')
    >>> # 因子乘以常数
    >>> scaled_factor = factor_operation(factor1, operation='multiply', constant=2.0)
    >>> # 因子幂运算
    >>> power_factor = factor_operation(factor1, operation='power', constant=2)
    """
    if not isinstance(factor1, pd.Series) or not isinstance(factor1.index, pd.MultiIndex):
        raise ValueError("factor1必须是MultiIndex Series格式")
    
    if factor2 is not None and (not isinstance(factor2, pd.Series) or not isinstance(factor2.index, pd.MultiIndex)):
        raise ValueError("factor2必须是MultiIndex Series格式或None")
    
    if factor2 is None and constant is None:
        raise ValueError("必须提供factor2或constant中的一个")
    
    if operation not in ['add', 'subtract', 'multiply', 'divide', 'power']:
        raise ValueError("operation必须是'add', 'subtract', 'multiply', 'divide', 'power'中的一个")
    
    try:
        if factor2 is not None:
            # 因子间运算
            factor1_aligned, factor2_aligned = factor1.align(factor2, join='inner')
            
            if operation == 'add':
                result = factor1_aligned + factor2_aligned
            elif operation == 'subtract':
                result = factor1_aligned - factor2_aligned
            elif operation == 'multiply':
                result = factor1_aligned * factor2_aligned
            elif operation == 'divide':
                result = factor1_aligned / factor2_aligned.replace(0, np.nan)
            elif operation == 'power':
                result = factor1_aligned ** factor2_aligned
                
        else:
            # 与常数运算
            if operation == 'add':
                result = factor1 + constant
            elif operation == 'subtract':
                result = factor1 - constant
            elif operation == 'multiply':
                result = factor1 * constant
            elif operation == 'divide':
                if constant == 0:
                    raise ValueError("不能除以0")
                result = factor1 / constant
            elif operation == 'power':
                result = factor1 ** constant
        
        # 处理无穷值
        result = result.replace([np.inf, -np.inf], np.nan)
        
        # 设置名称
        if factor2 is not None:
            result.name = f"{factor1.name}_{operation}_{factor2.name}"
        else:
            result.name = f"{factor1.name}_{operation}_{constant}"
        
        logger.debug(f"因子运算完成，操作: {operation}")
        return result
        
    except Exception as e:
        logger.error(f"因子运算失败: {e}")
        raise


# 便捷别名和函数
combine = linear_combine
orthogonalize_factor = orthogonalize
residualize_factor = residualize

# 便捷运算函数
def add_factors(factor1: pd.Series, factor2: pd.Series) -> pd.Series:
    """因子相加"""
    return factor_operation(factor1, factor2, 'add')

def subtract_factors(factor1: pd.Series, factor2: pd.Series) -> pd.Series:
    """因子相减"""
    return factor_operation(factor1, factor2, 'subtract')

def multiply_factors(factor1: pd.Series, factor2: pd.Series) -> pd.Series:
    """因子相乘"""
    return factor_operation(factor1, factor2, 'multiply')

def divide_factors(factor1: pd.Series, factor2: pd.Series) -> pd.Series:
    """因子相除"""
    return factor_operation(factor1, factor2, 'divide')

def scale_factor(factor: pd.Series, scale: float) -> pd.Series:
    """因子缩放"""
    return factor_operation(factor, operation='multiply', constant=scale)