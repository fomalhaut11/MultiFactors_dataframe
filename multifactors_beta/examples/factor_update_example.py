#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子更新示例
演示如何使用因子更新模块进行全量和增量更新
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 添加项目路径
import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入因子更新模块
from factors.utils.factor_updater import FactorUpdater, UpdateTracker
from get_real_trading_dates import create_debug_trading_dates


def create_sample_data(base_path: Path):
    """创建示例数据，模拟真实的数据更新场景"""
    
    # 确保目录存在
    base_path.mkdir(exist_ok=True)
    
    # 1. 创建财务数据（包含报表发布日期）
    logger.info("Creating sample financial data...")
    
    # 模拟2023年Q1到2024年Q3的财务数据
    report_dates = pd.date_range('2023-03-31', '2024-09-30', freq='Q')
    stocks = ['000001', '000002', '000003', '000004', '000005']
    
    financial_data = []
    for stock in stocks:
        for i, report_date in enumerate(report_dates):
            # 模拟报表发布延迟（通常1-2个月）
            release_delay = np.random.randint(30, 60)
            release_date = report_date + pd.Timedelta(days=release_delay)
            
            financial_data.append({
                'ReportDates': report_date,
                'StockCodes': stock,
                'ReleasedDates': release_date,  # 关键：报表发布日期
                'DEDUCTEDPROFIT': np.random.uniform(1e8, 1e9) * (1 + i * 0.05),  # 模拟增长
                'EQY_BELONGTO_PARCOMSH': np.random.uniform(1e9, 1e10) * (1 + i * 0.03),
                'TOT_OPER_REV': np.random.uniform(1e9, 1e10) * (1 + i * 0.04),
                'TOT_OPER_COST': np.random.uniform(5e8, 8e9) * (1 + i * 0.03),
                'TOT_ASSETS': np.random.uniform(1e10, 1e11) * (1 + i * 0.02),
                'TOT_CUR_ASSETS': np.random.uniform(1e9, 1e10),
                'TOT_CUR_LIAB': np.random.uniform(5e8, 5e9),
                'NET_CASH_FLOWS_OPER_ACT': np.random.uniform(1e8, 1e9),
                'CASH_PAY_ACQ_CONST_FIOLTA': np.random.uniform(1e7, 1e8),
                'd_quarter': (i % 4) + 1
            })
    
    financial_df = pd.DataFrame(financial_data)
    financial_df = financial_df.set_index(['ReportDates', 'StockCodes'])
    
    # 保存初始数据（只包含到2024年Q2）
    initial_financial = financial_df[
        financial_df['ReleasedDates'] <= pd.Timestamp('2024-08-15')
    ]
    initial_financial.to_pickle(base_path / 'financial_data_initial.pkl')
    
    # 保存包含新数据的完整数据
    financial_df.to_pickle(base_path / 'financial_data_updated.pkl')
    
    # 2. 创建价格数据
    logger.info("Creating sample price data...")
    
    # 初始数据：2024年1月到8月
    # 🎯 使用真实交易日期
    try:
        all_trading_dates = create_debug_trading_dates('2024-01-01', '2024-09-30')
        initial_dates = all_trading_dates[all_trading_dates <= '2024-08-31']
        print(f"✅ 使用真实交易日期: {len(initial_dates)}个初始交易日")
    except Exception as e:
        print(f"⚠️ 无法加载真实交易日期，使用工作日: {e}")
        initial_dates = pd.date_range('2024-01-01', '2024-08-31', freq='B')
        
    price_data_initial = []
    
    for date in initial_dates:
        for stock in stocks:
            base_price = 10 + hash(stock) % 90
            price_data_initial.append({
                'TradingDates': date,
                'StockCodes': stock,
                'open': base_price + np.random.uniform(-1, 1),
                'high': base_price + np.random.uniform(0, 2),
                'low': base_price + np.random.uniform(-2, 0),
                'close': base_price + np.random.uniform(-1, 1),
                'volume': np.random.randint(1000000, 10000000),
                'amt': np.random.randint(10000000, 100000000),
                'adjfactor': 1.0
            })
    
    price_df_initial = pd.DataFrame(price_data_initial)
    price_df_initial = price_df_initial.set_index(['TradingDates', 'StockCodes'])
    price_df_initial.to_pickle(base_path / 'Price_initial.pkl')
    
    # 更新数据：添加9月的数据
    # 🎯 使用之前获取的真实交易日期
    try:
        if 'all_trading_dates' not in locals():
            all_trading_dates = create_debug_trading_dates('2024-01-01', '2024-09-30')
        all_dates = all_trading_dates
        print(f"✅ 使用真实交易日期: {len(all_dates)}个总交易日")
    except Exception as e:
        print(f"⚠️ 无法加载真实交易日期，使用工作日: {e}")
        all_dates = pd.date_range('2024-01-01', '2024-09-30', freq='B')
        
    price_data_all = []
    
    for date in all_dates:
        for stock in stocks:
            base_price = 10 + hash(stock) % 90
            price_data_all.append({
                'TradingDates': date,
                'StockCodes': stock,
                'open': base_price + np.random.uniform(-1, 1),
                'high': base_price + np.random.uniform(0, 2),
                'low': base_price + np.random.uniform(-2, 0),
                'close': base_price + np.random.uniform(-1, 1),
                'volume': np.random.randint(1000000, 10000000),
                'amt': np.random.randint(10000000, 100000000),
                'adjfactor': 1.0
            })
    
    price_df_all = pd.DataFrame(price_data_all)
    price_df_all = price_df_all.set_index(['TradingDates', 'StockCodes'])
    price_df_all.to_pickle(base_path / 'Price_updated.pkl')
    
    # 3. 创建市值数据
    market_cap_initial = price_df_initial['close'] * np.random.uniform(1e8, 1e9, size=len(price_df_initial))
    market_cap_initial.to_pickle(base_path / 'MarketCap_initial.pkl')
    
    market_cap_all = price_df_all['close'] * np.random.uniform(1e8, 1e9, size=len(price_df_all))
    market_cap_all.to_pickle(base_path / 'MarketCap_updated.pkl')
    
    logger.info("Sample data created successfully")


def example_full_update():
    """全量更新示例"""
    print("\n" + "="*60)
    print("全量更新示例")
    print("="*60)
    
    # 设置路径
    data_path = Path('./sample_data')
    factor_path = Path('./factor_results')
    factor_path.mkdir(exist_ok=True)
    
    # 创建更新器
    updater = FactorUpdater(data_path, factor_path)
    
    # 加载初始数据
    financial_data = pd.read_pickle(data_path / 'financial_data_initial.pkl')
    price_data = pd.read_pickle(data_path / 'Price_initial.pkl')
    market_cap = pd.read_pickle(data_path / 'MarketCap_initial.pkl')
    
    print(f"\n初始数据统计:")
    print(f"- 财务数据记录数: {len(financial_data)}")
    print(f"- 最新报表发布日期: {financial_data['ReleasedDates'].max()}")
    print(f"- 价格数据记录数: {len(price_data)}")
    print(f"- 最新交易日: {price_data.index.get_level_values('TradingDates').max()}")
    
    # 执行全量更新
    print("\n执行全量更新...")
    
    # 更新基本面因子
    fundamental_factors = ['EP_ttm', 'ROE_ttm', 'CurrentRatio']
    fundamental_results = updater.update_fundamental_factors(
        factor_names=fundamental_factors,
        mode='full',
        financial_data=financial_data,
        market_cap=market_cap,
        release_dates=financial_data[['ReleasedDates']],
        trading_dates=price_data.index.get_level_values('TradingDates').unique()
    )
    
    print(f"\n基本面因子更新完成:")
    for factor_name, factor_data in fundamental_results.items():
        print(f"- {factor_name}: {len(factor_data)} 条记录")
    
    # 更新技术因子
    technical_factors = ['Momentum_20', 'Volatility_20']
    technical_results = updater.update_technical_factors(
        factor_names=technical_factors,
        mode='full',
        price_data=price_data
    )
    
    print(f"\n技术因子更新完成:")
    for factor_name, factor_data in technical_results.items():
        print(f"- {factor_name}: {len(factor_data)} 条记录")
    
    # 查看更新追踪信息
    tracker_info = updater.tracker.status
    print(f"\n更新追踪信息:")
    print(f"- 财务数据最后发布日期: {tracker_info.get('financial', {}).get('last_release_date')}")
    print(f"- 价格数据最后交易日: {tracker_info.get('price', {}).get('last_trading_date')}")


def example_incremental_update():
    """增量更新示例"""
    print("\n" + "="*60)
    print("增量更新示例")
    print("="*60)
    
    # 设置路径
    data_path = Path('./sample_data')
    factor_path = Path('./factor_results')
    
    # 创建更新器
    updater = FactorUpdater(data_path, factor_path)
    
    # 加载包含新数据的文件
    financial_data_new = pd.read_pickle(data_path / 'financial_data_updated.pkl')
    price_data_new = pd.read_pickle(data_path / 'Price_updated.pkl')
    market_cap_new = pd.read_pickle(data_path / 'MarketCap_updated.pkl')
    
    # 检查新增的财务数据
    print("\n检查财务数据更新...")
    has_financial_updates, new_financial = updater.check_financial_updates(financial_data_new)
    
    if has_financial_updates:
        print(f"发现新的财务数据:")
        print(f"- 新增记录数: {len(new_financial)}")
        print(f"- 涉及股票: {new_financial.index.get_level_values('StockCodes').unique().tolist()}")
        print(f"- 新报表发布日期范围: {new_financial['ReleasedDates'].min()} 到 {new_financial['ReleasedDates'].max()}")
        
        # 显示部分新数据
        print("\n新增财务数据示例:")
        print(new_financial.head())
    
    # 检查新增的价格数据
    print("\n检查价格数据更新...")
    has_price_updates, new_price = updater.check_price_updates(price_data_new)
    
    if has_price_updates:
        print(f"发现新的价格数据:")
        print(f"- 新增记录数: {len(new_price)}")
        print(f"- 新交易日范围: {new_price.index.get_level_values('TradingDates').min()} 到 {new_price.index.get_level_values('TradingDates').max()}")
    
    # 执行增量更新
    print("\n执行增量更新...")
    
    # 增量更新基本面因子
    fundamental_factors = ['EP_ttm', 'ROE_ttm', 'CurrentRatio']
    fundamental_results = updater.update_fundamental_factors(
        factor_names=fundamental_factors,
        mode='incremental',
        financial_data=financial_data_new,
        market_cap=market_cap_new,
        release_dates=financial_data_new[['ReleasedDates']],
        trading_dates=price_data_new.index.get_level_values('TradingDates').unique()
    )
    
    print(f"\n基本面因子增量更新完成:")
    for factor_name, factor_data in fundamental_results.items():
        print(f"- {factor_name}: 总记录数 {len(factor_data)}")
    
    # 增量更新技术因子
    technical_factors = ['Momentum_20', 'Volatility_20']
    technical_results = updater.update_technical_factors(
        factor_names=technical_factors,
        mode='incremental',
        price_data=price_data_new
    )
    
    print(f"\n技术因子增量更新完成:")
    for factor_name, factor_data in technical_results.items():
        print(f"- {factor_name}: 总记录数 {len(factor_data)}")
    
    # 查看更新后的追踪信息
    tracker_info = updater.tracker.status
    print(f"\n更新后的追踪信息:")
    print(f"- 财务数据最后发布日期: {tracker_info.get('financial', {}).get('last_release_date')}")
    print(f"- 价格数据最后交易日: {tracker_info.get('price', {}).get('last_trading_date')}")


def example_automated_update():
    """自动化更新示例（模拟日常更新流程）"""
    print("\n" + "="*60)
    print("自动化更新流程示例")
    print("="*60)
    
    # 设置路径
    data_path = Path('./sample_data')
    factor_path = Path('./factor_results')
    
    # 创建更新器
    updater = FactorUpdater(data_path, factor_path)
    
    print("\n开始自动化更新流程...")
    
    # 1. 检查数据源更新
    print("\n1. 检查数据源更新...")
    
    # 假设这是每日运行的脚本，需要检查是否有新数据
    current_date = datetime.now().date()
    print(f"当前日期: {current_date}")
    
    # 2. 下载/同步最新数据（这里使用已准备的数据）
    print("\n2. 同步最新数据...")
    financial_data = pd.read_pickle(data_path / 'financial_data_updated.pkl')
    price_data = pd.read_pickle(data_path / 'Price_updated.pkl')
    market_cap = pd.read_pickle(data_path / 'MarketCap_updated.pkl')
    
    # 3. 执行增量更新
    print("\n3. 执行增量更新...")
    
    try:
        # 更新所有因子
        updater.update_all_factors(
            mode='incremental',
            financial_data=financial_data,
            price_data=price_data,
            market_cap=market_cap,
            release_dates=financial_data[['ReleasedDates']],
            trading_dates=price_data.index.get_level_values('TradingDates').unique()
        )
        
        print("\n[OK] 所有因子更新成功!")
        
    except Exception as e:
        logger.error(f"更新失败: {e}")
        print(f"\n[FAIL] 更新失败: {e}")
    
    # 4. 生成更新报告
    print("\n4. 生成更新报告...")
    generate_update_report(updater)


def generate_update_report(updater: FactorUpdater):
    """生成更新报告"""
    tracker_info = updater.tracker.status
    
    print("\n" + "-"*50)
    print("因子更新报告")
    print("-"*50)
    
    # 财务数据更新情况
    financial_info = tracker_info.get('financial', {})
    if financial_info:
        print(f"\n财务数据:")
        print(f"  - 最后发布日期: {financial_info.get('last_release_date')}")
        print(f"  - 最后更新时间: {financial_info.get('last_update_time')}")
        print(f"  - 总记录数: {financial_info.get('total_records')}")
    
    # 价格数据更新情况
    price_info = tracker_info.get('price', {})
    if price_info:
        print(f"\n价格数据:")
        print(f"  - 最后交易日: {price_info.get('last_trading_date')}")
        print(f"  - 最后更新时间: {price_info.get('last_update_time')}")
        print(f"  - 总记录数: {price_info.get('total_records')}")
    
    # 检查因子文件
    factor_path = updater.factor_path
    factor_files = list(factor_path.glob('*.pkl'))
    
    print(f"\n已更新因子:")
    for factor_file in sorted(factor_files):
        file_stat = factor_file.stat()
        file_size = file_stat.st_size / 1024 / 1024  # MB
        file_mtime = datetime.fromtimestamp(file_stat.st_mtime)
        print(f"  - {factor_file.stem}: {file_size:.2f} MB, 更新时间: {file_mtime}")


def main():
    """主函数"""
    print("因子更新模块使用示例")
    print("=" * 80)
    
    # 创建示例数据
    sample_data_path = Path('./sample_data')
    if not sample_data_path.exists():
        print("\n创建示例数据...")
        create_sample_data(sample_data_path)
    else:
        print("\n使用已存在的示例数据")
    
    # 运行各个示例
    
    # 1. 全量更新（首次运行或重建）
    example_full_update()
    
    # 2. 增量更新（日常更新）
    example_incremental_update()
    
    # 3. 自动化更新流程
    example_automated_update()
    
    print("\n" + "="*80)
    print("所有示例运行完成！")
    print("\n[TIP] 提示:")
    print("1. 增量更新基于报表发布日期(ReleasedDates)判断新数据")
    print("2. 财务因子更新时会重算相关股票的所有历史数据")
    print("3. 技术因子只计算新增交易日的数据")
    print("4. 更新状态保存在 factor_update_tracker.json 中")


if __name__ == "__main__":
    main()