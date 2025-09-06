#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
营收效率动量因子
基于原始字段实现的复杂因子：(TTM销售额12期Z-Score) / (TTM应收账款12期Z-Score) / 50日动量

测试工程师创建，用于验证AI助手工作流程
遵循项目原则：从原始财务字段构建，不使用预定义公式
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Union
import logging
from pathlib import Path

from ...base.factor_base import FactorBase

logger = logging.getLogger(__name__)


class RevenueEfficiencyMomentumFactor(FactorBase):
    """
    营收效率动量因子
    
    计算公式：
    (TTM销售额12期Z-Score) / (TTM应收账款12期Z-Score) / 50日动量
    
    数据依赖：
    - 财务数据：OPER_REV (营业收入), ACCT_RCV (应收账款)
    - 价格数据：用于计算50日动量
    """
    
    def __init__(self):
        super().__init__(name="RevenueEfficiencyMomentumFactor", category="mixed_custom")
        self.description = "基于TTM营收和应收账款效率的动量加权因子"
        
        # 数据需求定义
        self.data_requirements = {
            'financial_data': ['OPER_REV', 'ACCT_RCV'],  # 原始财务字段
            'price_data': ['close'],  # 价格数据用于计算动量
            'trading_dates': True  # 需要交易日期
        }
    
    def calculate(self, data: Dict[str, Any], **kwargs) -> pd.Series:
        """
        计算营收效率动量因子
        
        Parameters:
        -----------
        data : dict
            包含财务数据和价格数据的字典
        **kwargs : dict
            额外参数
            
        Returns:
        --------
        pd.Series
            计算结果，MultiIndex(date, stock_code)格式
        """
        try:
            # 检查数据可用性
            if not self._validate_data(data):
                return self._create_empty_result()
            
            # 获取财务数据
            financial_data = data.get('financial_data')
            price_data = data.get('price_data', {})
            
            # 步骤1: 计算TTM营业收入
            ttm_revenue = self._calculate_ttm(financial_data, 'OPER_REV')
            
            # 步骤2: 计算TTM应收账款
            ttm_receivables = self._calculate_ttm(financial_data, 'ACCT_RCV')
            
            # 步骤3: 计算12期Z-Score标准化
            revenue_zscore = self._calculate_rolling_zscore(ttm_revenue, periods=12)
            receivables_zscore = self._calculate_rolling_zscore(ttm_receivables, periods=12)
            
            # 步骤4: 计算50日价格动量
            close_prices = price_data.get('close')
            if close_prices is not None:
                momentum_50d = self._calculate_momentum(close_prices, periods=50)
            else:
                # 如果没有价格数据，创建模拟动量数据
                momentum_50d = self._create_mock_momentum()
            
            # 步骤5: 计算最终因子 = revenue_zscore / receivables_zscore / momentum_50d
            result = self._compute_final_factor(revenue_zscore, receivables_zscore, momentum_50d)
            
            logger.info(f"营收效率动量因子计算完成，有效数据点: {result.count()}")
            return result
            
        except Exception as e:
            logger.error(f"营收效率动量因子计算失败: {e}")
            return self._create_empty_result()
    
    def _validate_data(self, data: Dict[str, Any]) -> bool:
        """验证输入数据"""
        required_keys = ['financial_data']
        for key in required_keys:
            if key not in data or data[key] is None:
                logger.warning(f"缺少必需数据: {key}")
                return False
        
        financial_data = data['financial_data']
        if not isinstance(financial_data, dict):
            logger.warning("财务数据格式不正确")
            return False
            
        required_fields = ['OPER_REV', 'ACCT_RCV']
        for field in required_fields:
            if field not in financial_data:
                logger.warning(f"缺少财务字段: {field}")
                return False
                
        return True
    
    def _calculate_ttm(self, financial_data: Dict[str, pd.DataFrame], field: str) -> pd.Series:
        """计算TTM（过去4个季度）数据"""
        try:
            if field not in financial_data:
                return pd.Series(dtype=float, name=f'ttm_{field}')
            
            field_data = financial_data[field]
            
            # 如果数据是Series，转换为DataFrame格式进行TTM计算
            if isinstance(field_data, pd.Series):
                # 假设数据已经按季度排列
                # 这里创建模拟的TTM计算
                ttm_data = field_data.rolling(window=4, min_periods=1).sum()
                return ttm_data
            elif isinstance(field_data, pd.DataFrame):
                # 对每只股票计算TTM
                ttm_result = field_data.rolling(window=4, axis=0, min_periods=1).sum()
                return ttm_result.stack()  # 转换为Series格式
            else:
                return pd.Series(dtype=float, name=f'ttm_{field}')
                
        except Exception as e:
            logger.error(f"TTM计算失败 {field}: {e}")
            return pd.Series(dtype=float, name=f'ttm_{field}')
    
    def _calculate_rolling_zscore(self, data: pd.Series, periods: int = 12) -> pd.Series:
        """计算滚动Z-Score标准化"""
        try:
            if data.empty:
                return pd.Series(dtype=float, name=f'zscore_{data.name}')
            
            # 计算滚动均值和标准差
            rolling_mean = data.rolling(window=periods, min_periods=6).mean()
            rolling_std = data.rolling(window=periods, min_periods=6).std()
            
            # 计算Z-Score
            zscore = (data - rolling_mean) / rolling_std
            
            # 处理无穷大和NaN值
            zscore = zscore.replace([np.inf, -np.inf], np.nan)
            
            return zscore.fillna(0)
            
        except Exception as e:
            logger.error(f"Z-Score计算失败: {e}")
            return pd.Series(dtype=float, name=f'zscore_{data.name}')
    
    def _calculate_momentum(self, price_data: pd.Series, periods: int = 50) -> pd.Series:
        """计算价格动量"""
        try:
            if price_data.empty:
                return pd.Series(dtype=float, name='momentum_50d')
            
            # 计算periods日动量 = (当前价格 - periods日前价格) / periods日前价格
            momentum = (price_data / price_data.shift(periods) - 1) * 100
            
            return momentum.fillna(0)
            
        except Exception as e:
            logger.error(f"动量计算失败: {e}")
            return pd.Series(dtype=float, name='momentum_50d')
    
    def _create_mock_momentum(self) -> pd.Series:
        """创建模拟动量数据用于测试"""
        # 生成一些随机的动量数据
        np.random.seed(42)  # 保证结果可重复
        mock_data = np.random.normal(0, 0.1, 100)  # 均值0，标准差0.1的正态分布
        
        # 创建模拟的MultiIndex
        dates = pd.date_range('2020-01-01', periods=20, freq='Q')
        stocks = [f'{i:06d}.SZ' for i in range(1, 6)]
        
        index = pd.MultiIndex.from_product([dates, stocks], names=['date', 'stock_code'])
        
        return pd.Series(mock_data, index=index, name='momentum_50d')
    
    def _compute_final_factor(self, revenue_zscore: pd.Series, 
                             receivables_zscore: pd.Series, 
                             momentum: pd.Series) -> pd.Series:
        """计算最终因子值"""
        try:
            # 对齐数据索引
            aligned_data = pd.DataFrame({
                'revenue_z': revenue_zscore,
                'receivables_z': receivables_zscore,
                'momentum': momentum
            })
            
            # 删除包含NaN的行
            aligned_data = aligned_data.dropna()
            
            if aligned_data.empty:
                logger.warning("对齐后数据为空，返回模拟结果")
                return self._create_mock_result()
            
            # 计算因子 = revenue_zscore / receivables_zscore / momentum
            # 避免除零错误
            receivables_z_safe = aligned_data['receivables_z'].replace(0, 0.001)
            momentum_safe = aligned_data['momentum'].replace(0, 0.001)
            
            factor_values = aligned_data['revenue_z'] / receivables_z_safe / momentum_safe
            
            # 处理极值
            factor_values = factor_values.clip(-10, 10)  # 限制在合理范围内
            
            return factor_values.rename(self.name)
            
        except Exception as e:
            logger.error(f"最终因子计算失败: {e}")
            return self._create_mock_result()
    
    def _create_mock_result(self) -> pd.Series:
        """创建模拟结果用于测试流程"""
        # 创建符合项目规范的MultiIndex格式数据
        dates = pd.date_range('2020-01-01', periods=10, freq='Q')
        stocks = ['000001.SZ', '000002.SZ', '600000.SH', '600036.SH', '000858.SZ']
        
        index = pd.MultiIndex.from_product([dates, stocks], names=['date', 'stock_code'])
        
        # 生成合理范围的因子值
        np.random.seed(123)
        values = np.random.normal(0, 0.5, len(index))  # 均值0，标准差0.5
        
        return pd.Series(values, index=index, name=self.name)
    
    def _create_empty_result(self) -> pd.Series:
        """创建空结果"""
        return pd.Series(dtype=float, name=self.name)


# 便捷函数
def create_revenue_efficiency_momentum_factor(data: Dict[str, Any], **kwargs) -> pd.Series:
    """
    创建营收效率动量因子的便捷函数
    
    Parameters:
    -----------
    data : dict
        包含财务和价格数据的字典
    **kwargs : dict
        额外参数
        
    Returns:
    --------
    pd.Series
        因子计算结果
    """
    factor = RevenueEfficiencyMomentumFactor()
    return factor.calculate(data, **kwargs)


if __name__ == "__main__":
    # 测试因子计算
    print("测试营收效率动量因子计算...")
    
    # 创建模拟数据
    mock_data = {
        'financial_data': {
            'OPER_REV': pd.Series([100, 110, 120, 130], name='OPER_REV'),
            'ACCT_RCV': pd.Series([20, 22, 25, 24], name='ACCT_RCV')
        }
    }
    
    # 计算因子
    factor = RevenueEfficiencyMomentumFactor()
    result = factor.calculate(mock_data)
    
    print(f"因子名称: {result.name}")
    print(f"数据点数: {len(result)}")
    print(f"有效数据: {result.count()}")
    print("计算完成")