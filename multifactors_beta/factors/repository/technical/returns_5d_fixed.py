#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修正版5日收益率因子
修正了MultiIndex处理的错误，确保计算准确性

Author: AI Assistant  
Date: 2025-09-07
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional
from pathlib import Path

from ...base.factor_base import FactorBase
from config import get_config

logger = logging.getLogger(__name__)


class Returns5DFixedFactor(FactorBase):
    """修正版5日收益率因子
    
    使用更简单直接的方法计算，避免MultiIndex处理错误
    """
    
    def __init__(self):
        """初始化修正版5日收益率因子"""
        super().__init__(
            name="Returns_5D_C2C_Fixed",
            category="technical"
        )
        self.description = "修正版5日close-to-close滚动收益率"
        
    def _load_price_data(self) -> pd.DataFrame:
        """加载价格数据"""
        try:
            # 从配置获取数据路径
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
            required_columns = ['c', 'adjfactor']
            missing_cols = [col for col in required_columns if col not in price_data.columns]
            if missing_cols:
                raise ValueError(f"价格数据缺少必需列: {missing_cols}")
            
            logger.info(f"价格数据加载成功: {price_data.shape}")
            return price_data
            
        except Exception as e:
            logger.error(f"加载价格数据失败: {e}")
            raise
    
    def calculate(self) -> pd.Series:
        """计算5日收益率因子 - 使用修正后的方法"""
        try:
            logger.info("开始计算修正版5日收益率因子...")
            
            # 1. 加载价格数据
            price_data = self._load_price_data()
            
            # 2. 计算复权收盘价
            logger.info("计算复权收盘价...")
            price_data['adj_close'] = price_data['c'] * price_data['adjfactor']
            
            # 3. 重置索引，更容易处理
            logger.info("重新整理数据结构...")
            df = price_data.reset_index()
            
            # 4. 确保按日期和股票排序
            df = df.sort_values(['StockCodes', 'TradingDates'])
            
            # 5. 计算每只股票的日收益率
            logger.info("计算日收益率...")
            
            def calc_stock_returns(group):
                """计算单只股票的收益率"""
                # 确保按日期排序
                group = group.sort_values('TradingDates')
                
                # 计算日收益率
                group['daily_return'] = np.log(group['adj_close'] / group['adj_close'].shift(1))
                
                # 计算5日滚动收益率
                group['returns_5d'] = group['daily_return'].rolling(window=5, min_periods=5).sum()
                
                return group[['TradingDates', 'StockCodes', 'returns_5d']].dropna()
            
            # 按股票分组计算
            logger.info("按股票分组计算5日收益率...")
            results = []
            
            # 分批处理以避免内存问题
            unique_stocks = df['StockCodes'].unique()
            batch_size = 500
            
            for i in range(0, len(unique_stocks), batch_size):
                batch_stocks = unique_stocks[i:i+batch_size]
                batch_data = df[df['StockCodes'].isin(batch_stocks)]
                
                batch_results = batch_data.groupby('StockCodes').apply(calc_stock_returns)
                
                if not batch_results.empty:
                    # 重置索引
                    batch_results = batch_results.reset_index(drop=True)
                    results.append(batch_results)
                
                if (i + batch_size) % 2000 == 0:
                    logger.info(f"已处理 {i + batch_size}/{len(unique_stocks)} 只股票")
            
            # 6. 合并所有结果
            logger.info("合并计算结果...")
            final_result = pd.concat(results, ignore_index=True)
            
            # 7. 设置正确的MultiIndex
            logger.info("设置MultiIndex...")
            final_result = final_result.set_index(['TradingDates', 'StockCodes'])
            returns_5d_series = final_result['returns_5d']
            returns_5d_series.name = self.name
            
            # 8. 数据质量检查
            logger.info("数据质量检查...")
            total_count = len(returns_5d_series)
            null_count = returns_5d_series.isna().sum()
            
            if total_count > 0:
                q95 = returns_5d_series.quantile(0.95)
                q5 = returns_5d_series.quantile(0.05)
                outlier_count = ((returns_5d_series > q95 * 3) | (returns_5d_series < q5 * 3)).sum()
                
                logger.info(f"计算完成统计:")
                logger.info(f"  总数据点: {total_count:,}")
                logger.info(f"  NaN数量: {null_count:,}")
                logger.info(f"  异常值数量: {outlier_count:,}")
                logger.info(f"  数据范围: [{returns_5d_series.min():.6f}, {returns_5d_series.max():.6f}]")
                
                # 转换为实际收益率百分比进行验证
                min_pct = (np.exp(returns_5d_series.min()) - 1) * 100
                max_pct = (np.exp(returns_5d_series.max()) - 1) * 100
                logger.info(f"  实际收益率范围: [{min_pct:.2f}%, {max_pct:.2f}%]")
            
            logger.info("修正版5日收益率因子计算完成！")
            return returns_5d_series
            
        except Exception as e:
            logger.error(f"计算修正版5日收益率因子失败: {e}")
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
            "calculation_method": "close-to-close 5-day rolling log returns (fixed algorithm)",
            "time_direction": "历史数据，向前滚动",
            "output_format": "MultiIndex Series [TradingDates, StockCodes]",
            "frequency": "日频",
            "min_periods": 5,
            "version": "fixed_v1.0"
        }


def create_returns_5d_fixed_factor() -> Returns5DFixedFactor:
    """创建修正版5日收益率因子实例"""
    return Returns5DFixedFactor()


# 测试函数
def test_returns_5d_fixed_factor():
    """测试修正版5日收益率因子"""
    try:
        logger.info("测试修正版5日收益率因子...")
        
        factor = create_returns_5d_fixed_factor()
        
        # 获取因子信息
        info = factor.get_factor_info()
        logger.info(f"因子信息: {info}")
        
        # 计算因子
        result = factor.calculate()
        
        logger.info(f"因子计算成功!")
        logger.info(f"结果形状: {result.shape}")
        logger.info(f"结果类型: {type(result)}")
        
        # 验证600770在2024-06-05的值
        if ('2024-06-05', '600770') in result.index:
            test_value = result.loc[('2024-06-05', '600770')]
            test_pct = (np.exp(test_value) - 1) * 100
            logger.info(f"600770在2024-06-05的5日收益率: {test_pct:.2f}% (应该约为-9%)")
        
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
    success = test_returns_5d_fixed_factor()
    if success:
        print("修正版5日收益率因子测试成功!")
    else:
        print("修正版5日收益率因子测试失败!")