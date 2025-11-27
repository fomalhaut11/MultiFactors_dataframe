#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
5日收益率因子（基础因子）

使用close-to-close计算方式，确保时间方向正确性
作为基础因子，可以被其他复合因子调用

Author: AI Assistant
Date: 2025-09-07
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional
from pathlib import Path

from ...base.factor_base import FactorBase
from ...utils.data_loader import FactorDataLoader
from config import get_config

logger = logging.getLogger(__name__)


class Returns5DFactor(FactorBase):
    """5日收益率因子
    
    计算close-to-close的5日滚动收益率
    作为基础因子供其他因子调用
    """
    
    def __init__(self):
        """初始化5日收益率因子"""
        super().__init__(
            name="Returns_5D_C2C",
            category="technical"
        )
        self.description = "5日close-to-close滚动收益率"
        
    def _load_price_data(self) -> pd.DataFrame:
        """加载价格数据"""
        try:
            # 尝试从配置获取数据路径
            config = get_config('main')
            data_root = config.get('paths', {}).get('data_root') if config else None
            
            if data_root:
                price_file = Path(data_root) / 'Price.pkl'
            else:
                # 使用相对路径
                project_root = Path(__file__).parent.parent.parent.parent.parent
                price_file = project_root / 'StockData' / 'Price.pkl'
            
            if not price_file.exists():
                raise FileNotFoundError(f"价格数据文件不存在: {price_file}")
            
            logger.info(f"加载价格数据: {price_file}")
            price_data = pd.read_pickle(price_file)
            
            # 验证数据格式
            required_columns = ['c', 'adjfactor']  # close price and adjustment factor
            missing_cols = [col for col in required_columns if col not in price_data.columns]
            if missing_cols:
                raise ValueError(f"价格数据缺少必需列: {missing_cols}")
            
            logger.info(f"价格数据加载成功: {price_data.shape}")
            return price_data
            
        except Exception as e:
            logger.error(f"加载价格数据失败: {e}")
            raise
            
    def _calculate_daily_c2c_returns(self, price_data: pd.DataFrame) -> pd.Series:
        """计算日度c2c收益率"""
        try:
            # 按日期排序
            price_data = price_data.sort_index()
            
            # 计算复权收盘价
            price_data['adjusted_close'] = price_data['c'] * price_data['adjfactor']
            
            # 重新整理数据格式以便分组
            if isinstance(price_data.index, pd.MultiIndex):
                # 如果已经是MultiIndex (TradingDates, StockCodes)
                # 按股票分组计算收益率
                def calculate_returns_for_stock(group):
                    # 确保按日期排序
                    group = group.sort_index()
                    # 计算对数收益率 ln(P_t / P_{t-1})
                    returns = np.log(group / group.shift(1))
                    return returns.dropna()
                
                daily_returns = price_data.groupby(level=1)['adjusted_close'].apply(
                    calculate_returns_for_stock
                )
            else:
                # 如果是单层索引，需要先转换格式
                # 假设索引是日期，列包含股票代码信息
                logger.warning("价格数据不是MultiIndex格式，可能需要重新整理数据结构")
                raise ValueError("价格数据必须是MultiIndex格式[TradingDates, StockCodes]")
            
            daily_returns.name = "Returns_Daily_C2C"
            logger.info(f"日度c2c收益率计算完成: {daily_returns.shape}")
            return daily_returns
            
        except Exception as e:
            logger.error(f"计算日度c2c收益率失败: {e}")
            raise
            
    def _calculate_5d_rolling_returns(self, daily_returns: pd.Series) -> pd.Series:
        """计算5日滚动收益率"""
        try:
            logger.info("计算5日滚动收益率...")
            
            # 按股票分组计算5日滚动收益率
            def rolling_5d_returns(group):
                # 确保按日期排序
                group = group.sort_index()
                # 计算5日滚动收益率（向前看5日，即使用历史数据）
                rolling_returns = group.rolling(window=5, min_periods=5).sum()
                return rolling_returns.dropna()
            
            # 按股票代码分组计算
            returns_5d = daily_returns.groupby(level=1).apply(rolling_5d_returns)
            
            # 重新整理索引结构
            if isinstance(returns_5d.index, pd.MultiIndex) and len(returns_5d.index.levels) == 3:
                # 如果有多层索引，需要重新整理
                returns_5d = returns_5d.droplevel(1)  # 移除中间层
            
            returns_5d.name = "Returns_5D_C2C"
            logger.info(f"5日滚动收益率计算完成: {returns_5d.shape}")
            
            return returns_5d
            
        except Exception as e:
            logger.error(f"计算5日滚动收益率失败: {e}")
            raise
            
    def calculate(self) -> pd.Series:
        """计算5日收益率因子"""
        try:
            logger.info("开始计算5日收益率因子...")
            
            # 1. 加载价格数据
            price_data = self._load_price_data()
            
            # 2. 计算日度c2c收益率
            daily_returns = self._calculate_daily_c2c_returns(price_data)
            
            # 3. 计算5日滚动收益率
            returns_5d = self._calculate_5d_rolling_returns(daily_returns)
            
            # 4. 数据质量检查
            if returns_5d.empty:
                raise ValueError("5日收益率计算结果为空")
            
            # 检查异常值
            q95 = returns_5d.quantile(0.95)
            q5 = returns_5d.quantile(0.05)
            outlier_count = ((returns_5d > q95 * 3) | (returns_5d < q5 * 3)).sum()
            
            logger.info(f"因子数据质量检查:")
            logger.info(f"  总数据点: {len(returns_5d):,}")
            logger.info(f"  NaN数量: {returns_5d.isna().sum():,}")
            logger.info(f"  异常值数量: {outlier_count:,}")
            logger.info(f"  数据范围: [{returns_5d.min():.4f}, {returns_5d.max():.4f}]")
            
            logger.info("✅ 5日收益率因子计算完成")
            return returns_5d
            
        except Exception as e:
            logger.error(f"计算5日收益率因子失败: {e}")
            raise
            
    def get_factor_info(self) -> dict:
        """获取因子信息"""
        return {
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "data_requirements": [
                "Price.pkl (包含close price和adjustment factor)"
            ],
            "calculation_method": "close-to-close 5-day rolling log returns",
            "time_direction": "历史数据，向前滚动",
            "output_format": "MultiIndex Series [TradingDates, StockCodes]",
            "frequency": "日频",
            "min_periods": 5
        }
        

def create_returns_5d_factor() -> Returns5DFactor:
    """创建5日收益率因子实例"""
    return Returns5DFactor()


# 测试函数
def test_returns_5d_factor():
    """测试5日收益率因子"""
    try:
        logger.info("测试5日收益率因子...")
        
        factor = create_returns_5d_factor()
        
        # 获取因子信息
        info = factor.get_factor_info()
        logger.info(f"因子信息: {info}")
        
        # 计算因子
        result = factor.calculate()
        
        logger.info(f"因子计算成功!")
        logger.info(f"结果形状: {result.shape}")
        logger.info(f"结果类型: {type(result)}")
        
        # 显示样本数据
        logger.info(f"样本数据:")
        logger.info(result.head(10))
        
        return True
        
    except Exception as e:
        logger.error(f"测试失败: {e}")
        return False


if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 运行测试
    success = test_returns_5d_factor()
    if success:
        print("✅ 5日收益率因子测试成功!")
    else:
        print("❌ 5日收益率因子测试失败!")