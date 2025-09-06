#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
工具函数使用示例
演示如何使用重构后的工具模块
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# 导入工具模块
from core.utils import (
    OutlierHandler, Normalizer, DataCleaner,
    FactorOrthogonalizer, FactorProcessor,
    MarketCapFilter, LiquidityMetrics
)
# 导入技术指标（从新位置）
from factors.generator.technical.indicators import (
    MovingAverageCalculator, VolatilityCalculator, TechnicalIndicators
)


def example_data_cleaning():
    """数据清洗示例"""
    print("\n=== 数据清洗示例 ===")
    
    # 创建示例数据
    np.random.seed(42)
    data = pd.DataFrame({
        'factor1': np.random.randn(100) * 10 + 50,
        'factor2': np.random.randn(100) * 5 + 20,
        'factor3': np.zeros(100)  # 全零列
    })
    
    # 添加一些异常值
    data.iloc[10, 0] = 1000  # 极大值
    data.iloc[20, 1] = -500  # 极小值
    
    print("原始数据统计:")
    print(data.describe())
    
    # 1. 去除异常值
    cleaned_data = data.copy()
    cleaned_data['factor1'] = OutlierHandler.remove_outlier(
        cleaned_data['factor1'], 
        method="IQR", 
        threshold=1.5
    )
    cleaned_data['factor2'] = OutlierHandler.remove_outlier(
        cleaned_data['factor2'], 
        method="mean", 
        threshold=3
    )
    
    print("\n去除异常值后:")
    print(cleaned_data.describe())
    
    # 2. 标准化
    normalized_data = cleaned_data.copy()
    normalized_data['factor1'] = Normalizer.normalize(
        normalized_data['factor1'], 
        method="zscore"
    )
    normalized_data['factor2'] = Normalizer.normalize(
        normalized_data['factor2'], 
        method="robust"
    )
    
    print("\n标准化后:")
    print(normalized_data.describe())
    
    # 3. 清洗DataFrame（删除全零列）
    final_data = DataCleaner.clean_dataframe(normalized_data)
    print(f"\n清洗后的列: {final_data.columns.tolist()}")


def example_technical_indicators():
    """技术指标计算示例"""
    print("\n=== 技术指标计算示例 ===")
    
    # 创建模拟价格数据
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    prices = pd.DataFrame({
        'open': 100 + np.cumsum(np.random.randn(100) * 0.5),
        'high': 100 + np.cumsum(np.random.randn(100) * 0.5) + 1,
        'low': 100 + np.cumsum(np.random.randn(100) * 0.5) - 1,
        'close': 100 + np.cumsum(np.random.randn(100) * 0.5),
        'volume': np.random.randint(1000000, 10000000, 100),
        'amt': np.random.randint(10000000, 100000000, 100)
    }, index=dates)
    
    # 确保high/low的逻辑关系
    prices['high'] = prices[['open', 'high', 'close']].max(axis=1)
    prices['low'] = prices[['open', 'low', 'close']].min(axis=1)
    
    # 1. 移动平均
    prices['sma_20'] = MovingAverageCalculator.simple_moving_average(
        prices['close'], window=20
    )
    prices['ema_20'] = MovingAverageCalculator.exponential_moving_average(
        prices['close'], span=20
    )
    
    # 2. 波动率
    prices['volatility'] = VolatilityCalculator.historical_volatility(
        prices, window=20, method="simple", annualize=True
    )
    
    # 3. 布林带
    middle, upper, lower = TechnicalIndicators.bollinger_bands(
        prices['close'], window=20, num_std=2
    )
    prices['bb_middle'] = middle
    prices['bb_upper'] = upper
    prices['bb_lower'] = lower
    
    # 4. RSI
    prices['rsi'] = TechnicalIndicators.rsi(prices['close'], window=14)
    
    # 5. MACD
    macd_line, signal_line, histogram = TechnicalIndicators.macd(prices['close'])
    prices['macd'] = macd_line
    prices['macd_signal'] = signal_line
    prices['macd_hist'] = histogram
    
    print("技术指标计算结果:")
    print(prices[['close', 'sma_20', 'ema_20', 'volatility', 'rsi']].tail(10))


def example_factor_processing():
    """因子处理示例"""
    print("\n=== 因子处理示例 ===")
    
    # 创建多因子数据
    dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
    stocks = ['000001', '000002', '000003', '000004', '000005']
    
    # 创建MultiIndex
    index = pd.MultiIndex.from_product(
        [dates, stocks], 
        names=['TradingDates', 'StockCodes']
    )
    
    # 创建因子数据
    np.random.seed(42)
    factors = pd.DataFrame({
        'momentum': np.random.randn(len(index)),
        'value': np.random.randn(len(index)),
        'quality': np.random.randn(len(index)),
        'size': np.random.randn(len(index))
    }, index=index)
    
    # 添加一些相关性
    factors['value'] = factors['value'] + 0.3 * factors['momentum']
    factors['quality'] = factors['quality'] + 0.2 * factors['value']
    
    print("原始因子相关性:")
    print(factors.corr())
    
    # 1. 顺序正交化
    orthogonal_factors = FactorOrthogonalizer.sequential_orthogonalize(
        factors, 
        normalize=True, 
        remove_outliers=True
    )
    
    print("\n正交化后因子相关性:")
    print(orthogonal_factors.corr())
    
    # 2. 添加新因子并正交化
    new_factor = pd.DataFrame({
        'growth': np.random.randn(len(index)) + 0.5 * factors['quality']
    }, index=index)
    
    all_factors = FactorOrthogonalizer.add_new_factor_orthogonal(
        orthogonal_factors[['momentum', 'value', 'quality']],
        new_factor,
        normalize=True
    )
    
    print("\n添加新因子后的因子数量:", all_factors.shape[1])


def example_market_microstructure():
    """市场微观结构示例"""
    print("\n=== 市场微观结构示例 ===")
    
    # 创建价格和市值数据
    dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
    stocks = [f"{i:06d}" for i in range(1, 101)]  # 100只股票
    
    index = pd.MultiIndex.from_product(
        [dates, stocks], 
        names=['TradingDates', 'StockCodes']
    )
    
    # 模拟不同市值的股票
    np.random.seed(42)
    price_data = pd.DataFrame(index=index)
    
    # 市值呈对数正态分布
    base_mc = np.exp(np.random.normal(20, 2, len(stocks)))
    
    for i, date in enumerate(dates):
        date_idx = price_data.index.get_level_values('TradingDates') == date
        # 市值随时间小幅波动
        price_data.loc[date_idx, 'MC'] = base_mc * (1 + np.random.randn(len(stocks)) * 0.01)
        price_data.loc[date_idx, 'amt'] = base_mc * np.random.uniform(0.01, 0.1, len(stocks)) * 1e6
        price_data.loc[date_idx, 'close'] = 10 + np.random.randn(len(stocks)) * 2
        price_data.loc[date_idx, 'volume'] = price_data.loc[date_idx, 'amt'] / price_data.loc[date_idx, 'close']
    
    # 1. 市值分组筛选
    small_cap = MarketCapFilter.filter_by_market_cap(
        price_data,
        num_groups=10,
        target_group=0,  # 最小市值组
        min_amount=100000
    )
    
    print(f"小市值股票数量: {len(small_cap.index.get_level_values('StockCodes').unique())}")
    
    # 选择多个组
    mid_cap = MarketCapFilter.filter_by_market_cap(
        price_data,
        num_groups=10,
        target_group=[4, 5],  # 中等市值组
        min_amount=100000
    )
    
    print(f"中市值股票数量: {len(mid_cap.index.get_level_values('StockCodes').unique())}")
    
    # 2. 市值分位数
    quantiles = MarketCapFilter.get_market_cap_quantiles(
        price_data,
        quantiles=[0.1, 0.3, 0.5, 0.7, 0.9]
    )
    
    print("\n市值分位数:")
    print(quantiles.head())
    
    # 3. 流动性指标
    # Amihud非流动性
    illiquidity = LiquidityMetrics.calculate_amihud_illiquidity(
        price_data, 
        window=20
    )
    
    print(f"\nAmihud非流动性指标统计:")
    print(illiquidity.describe())
    
    # 换手率
    turnover = LiquidityMetrics.calculate_turnover_rate(price_data)
    
    print(f"\n换手率统计:")
    print(turnover.describe())


def main():
    """运行所有示例"""
    print("工具函数使用示例")
    print("=" * 50)
    
    # 运行各个示例
    example_data_cleaning()
    example_technical_indicators()
    example_factor_processing()
    example_market_microstructure()
    
    print("\n所有示例运行完成！")


if __name__ == "__main__":
    main()