#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子计算示例
演示如何使用重构后的因子计算模块
"""
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import sys

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入因子计算器
from factors import FactorCalculator, FactorDataLoader
from get_real_trading_dates import create_debug_trading_dates

# 导入数据获取模块（假设已存在）
# from data.fetcher import DataFetcher


def load_sample_data():
    """加载示例数据（实际使用时应从数据库加载）"""
    # 这里创建一些模拟数据用于演示
    
    # 创建日期和股票索引
    # 🎯 使用真实交易日期替代简单的日期范围
    try:
        dates = create_debug_trading_dates('2024-01-01', '2024-12-31')
        print(f"✅ 使用真实交易日期: {len(dates)}个交易日")
    except Exception as e:
        print(f"⚠️ 无法加载真实交易日期，使用工作日: {e}")
        dates = pd.date_range('2024-01-01', '2024-12-31', freq='B')  # Business days
        
    stocks = ['000001', '000002', '000003', '000004', '000005']
    
    # 创建价格数据
    price_data = []
    for date in dates:
        for stock in stocks:
            price_data.append({
                'TradingDates': date,
                'StockCodes': stock,
                'open': np.random.uniform(10, 100),
                'high': np.random.uniform(10, 100),
                'low': np.random.uniform(10, 100),
                'close': np.random.uniform(10, 100),
                'volume': np.random.randint(1000000, 10000000),
                'amt': np.random.randint(10000000, 100000000),
                'adjfactor': 1.0
            })
    
    price_df = pd.DataFrame(price_data)
    price_df['high'] = price_df[['open', 'high', 'close']].max(axis=1)
    price_df['low'] = price_df[['open', 'low', 'close']].min(axis=1)
    price_df = price_df.set_index(['TradingDates', 'StockCodes'])
    
    # 创建财务数据
    report_dates = pd.date_range('2023-03-31', '2024-09-30', freq='Q')
    financial_data = []
    
    for stock in stocks:
        for i, date in enumerate(report_dates):
            financial_data.append({
                'ReportDates': date,
                'StockCodes': stock,
                'DEDUCTEDPROFIT': np.random.uniform(1e8, 1e9),
                'EQY_BELONGTO_PARCOMSH': np.random.uniform(1e9, 1e10),
                'TOT_OPER_REV': np.random.uniform(1e9, 1e10),
                'TOT_OPER_COST': np.random.uniform(5e8, 8e9),
                'TOT_ASSETS': np.random.uniform(1e10, 1e11),
                'TOT_CUR_ASSETS': np.random.uniform(1e9, 1e10),
                'TOT_CUR_LIAB': np.random.uniform(5e8, 5e9),
                'INVENTORIES': np.random.uniform(1e8, 1e9),
                'NET_CASH_FLOWS_OPER_ACT': np.random.uniform(1e8, 1e9),
                'CASH_PAY_ACQ_CONST_FIOLTA': np.random.uniform(1e7, 1e8),
                'GOODWILL': np.random.uniform(1e7, 1e8),
                'd_quarter': (i % 4) + 1
            })
    
    financial_df = pd.DataFrame(financial_data)
    financial_df = financial_df.set_index(['ReportDates', 'StockCodes'])
    
    # 创建市值数据
    market_cap = price_df['close'] * np.random.uniform(1e8, 1e9, size=len(price_df))
    
    # 创建发布日期数据
    release_dates = []
    for stock in stocks:
        for date in report_dates:
            release_dates.append({
                'ReportDates': date,
                'StockCodes': stock,
                'ReleasedDates': date + pd.Timedelta(days=30)  # 假设延迟30天发布
            })
    
    release_df = pd.DataFrame(release_dates)
    release_df = release_df.set_index(['ReportDates', 'StockCodes'])
    
    # 创建基准数据
    benchmark_data = pd.DataFrame({
        'TradingDates': dates,
        'close': 3000 + np.cumsum(np.random.randn(len(dates)) * 10)
    })
    benchmark_data = benchmark_data.set_index('TradingDates')
    
    return {
        'price_data': price_df,
        'financial_data': financial_df,
        'market_cap': market_cap,
        'release_dates': release_df,
        'benchmark_data': benchmark_data,
        'trading_dates': dates
    }


def example_basic_factors():
    """基本因子计算示例"""
    print("\n=== 基本因子计算示例 ===")
    
    # 加载数据
    data = load_sample_data()
    
    # 创建因子计算器
    calculator = FactorCalculator()
    
    # 查看可用因子
    print("\n可用因子列表:")
    factor_info = calculator.list_factors()
    for name, info in factor_info.items():
        print(f"  {name}: {info['description']} (类别: {info['category']})")
    
    # 计算基本面因子
    fundamental_factors = ['EP_ttm', 'BP', 'ROE_ttm', 'PEG']
    
    print(f"\n计算基本面因子: {fundamental_factors}")
    results = calculator.calculate_factors(
        factor_names=fundamental_factors,
        financial_data=data['financial_data'],
        market_cap=data['market_cap'],
        release_dates=data['release_dates'],
        trading_dates=data['trading_dates']
    )
    
    print(f"\n计算结果形状: {results.shape}")
    print(f"因子统计信息:")
    print(results.describe())


def example_technical_factors():
    """技术因子计算示例"""
    print("\n=== 技术因子计算示例 ===")
    
    # 加载数据
    data = load_sample_data()
    
    # 创建因子计算器
    calculator = FactorCalculator()
    
    # 计算技术因子
    technical_factors = ['Momentum_20', 'RSI_14', 'Volatility_20', 'GapReturn']
    
    print(f"\n计算技术因子: {technical_factors}")
    results = calculator.calculate_factors(
        factor_names=technical_factors,
        price_data=data['price_data']
    )
    
    print(f"\n计算结果形状: {results.shape}")
    print(f"因子相关性矩阵:")
    print(results.corr())


def example_risk_factors():
    """风险因子计算示例"""
    print("\n=== 风险因子计算示例 ===")
    
    # 加载数据
    data = load_sample_data()
    
    # 创建因子计算器
    calculator = FactorCalculator()
    
    # 计算风险因子
    risk_factors = ['Beta_252', 'WeightedBeta_252_63']
    
    print(f"\n计算风险因子: {risk_factors}")
    results = calculator.calculate_factors(
        factor_names=risk_factors,
        price_data=data['price_data'],
        benchmark_data=data['benchmark_data']
    )
    
    print(f"\n计算结果形状: {results.shape}")
    print(f"Beta因子分布:")
    print(results['Beta_252'].describe())


def example_custom_factor():
    """自定义因子示例"""
    print("\n=== 自定义因子示例 ===")
    
    from factors.base import FactorBase
    
    class CustomMomentumFactor(FactorBase):
        """自定义动量因子"""
        
        def __init__(self):
            super().__init__(name='CustomMomentum', category='technical')
            self.description = "Custom momentum factor"
            
        def calculate(self, price_data, **kwargs):
            """计算自定义动量"""
            # 计算5日动量和20日动量的差值
            close_price = price_data['close']
            
            momentum_5 = close_price.groupby(level='StockCodes').pct_change(5)
            momentum_20 = close_price.groupby(level='StockCodes').pct_change(20)
            
            custom_momentum = momentum_5 - momentum_20
            
            # 预处理
            custom_momentum = self.preprocess(custom_momentum)
            
            return custom_momentum
    
    # 加载数据
    data = load_sample_data()
    
    # 创建因子计算器并注册自定义因子
    calculator = FactorCalculator()
    calculator.register_factor('CustomMomentum', CustomMomentumFactor())
    
    # 计算自定义因子
    results = calculator.calculate_factors(
        factor_names=['CustomMomentum'],
        price_data=data['price_data']
    )
    
    print(f"\n自定义因子计算结果:")
    print(results.head())


def example_factor_combination():
    """因子组合示例"""
    print("\n=== 因子组合示例 ===")
    
    # 加载数据
    data = load_sample_data()
    
    # 创建因子计算器
    calculator = FactorCalculator()
    
    # 计算多类因子
    all_factors = [
        'EP_ttm', 'ROE_ttm',  # 基本面
        'Momentum_20', 'Volatility_20',  # 技术面
        'Beta_252'  # 风险
    ]
    
    print(f"\n计算因子组合: {all_factors}")
    
    # 保存路径
    save_path = Path('./factor_results')
    save_path.mkdir(exist_ok=True)
    
    results = calculator.calculate_factors(
        factor_names=all_factors,
        financial_data=data['financial_data'],
        price_data=data['price_data'],
        market_cap=data['market_cap'],
        benchmark_data=data['benchmark_data'],
        release_dates=data['release_dates'],
        trading_dates=data['trading_dates'],
        save_path=save_path
    )
    
    print(f"\n因子组合形状: {results.shape}")
    print(f"\n因子间相关性:")
    print(results.corr())
    
    # 因子正交化
    from core.utils import FactorOrthogonalizer
    
    orthogonal_factors = FactorOrthogonalizer.sequential_orthogonalize(
        results,
        normalize=True,
        remove_outliers=True
    )
    
    print(f"\n正交化后的因子相关性:")
    print(orthogonal_factors.corr())


def example_load_factors():
    """加载已保存的因子示例"""
    print("\n=== 加载因子示例 ===")
    
    # 因子文件路径
    factor_path = Path('./factor_results')
    
    if factor_path.exists() and any(factor_path.glob('*.pkl')):
        # 加载因子
        factor_names = ['EP_ttm', 'Momentum_20']
        factors = FactorDataLoader.load_factors(
            factor_names=factor_names,
            data_path=factor_path
        )
        
        print(f"\n成功加载 {len(factors.columns)} 个因子")
        print(f"因子数据形状: {factors.shape}")
        print(f"\n因子头部数据:")
        print(factors.head())
    else:
        print("\n未找到已保存的因子文件，请先运行 example_factor_combination()")


def main():
    """主函数"""
    print("因子计算模块使用示例")
    print("=" * 50)
    
    # 运行各个示例
    example_basic_factors()
    example_technical_factors()
    example_risk_factors()
    example_custom_factor()
    example_factor_combination()
    example_load_factors()
    
    print("\n所有示例运行完成！")


if __name__ == "__main__":
    main()