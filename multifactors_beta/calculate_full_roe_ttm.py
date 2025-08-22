#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全量ROE_ttm因子计算脚本
快速计算所有股票的ROE_ttm因子并统计耗时
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import logging
import time
from datetime import datetime

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from factors.financial.fundamental_factors import ROEFactor

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_data():
    """加载所需数据"""
    print("=" * 60)
    print("全量ROE_ttm因子计算")
    print(f"开始时间: {datetime.now()}")
    print("=" * 60)
    
    data_load_start = time.time()
    
    # 数据路径
    data_path = Path(r"E:\Documents\PythonProject\StockProject\StockData")
    auxiliary_path = project_root / "data" / "auxiliary"
    
    print(f"\n📂 数据加载阶段:")
    
    data = {}
    
    try:
        # 加载财务数据
        print(f"   正在加载财务数据...")
        start = time.time()
        financial_path = auxiliary_path / "FinancialData_unified.pkl"
        if financial_path.exists():
            data['financial_data'] = pd.read_pickle(financial_path)
            elapsed = time.time() - start
            print(f"   ✅ 财务数据: {data['financial_data'].shape} (耗时: {elapsed:.2f}秒)")
        else:
            print(f"   ❌ 财务数据文件不存在: {financial_path}")
            return None
        
        # 加载发布日期
        print(f"   正在加载发布日期...")
        start = time.time()
        release_path = auxiliary_path / "ReleaseDates.pkl"
        if release_path.exists():
            data['release_dates'] = pd.read_pickle(release_path)
            elapsed = time.time() - start
            print(f"   ✅ 发布日期: {data['release_dates'].shape} (耗时: {elapsed:.2f}秒)")
        else:
            print(f"   ❌ 发布日期文件不存在: {release_path}")
            return None
        
        # 加载交易日
        print(f"   正在加载交易日...")
        start = time.time()
        trading_path = auxiliary_path / "TradingDates.pkl"
        if trading_path.exists():
            data['trading_dates'] = pd.read_pickle(trading_path)
            elapsed = time.time() - start
            print(f"   ✅ 交易日: {len(data['trading_dates'])} 个 (耗时: {elapsed:.2f}秒)")
        else:
            print(f"   ❌ 交易日文件不存在: {trading_path}")
            return None
        
        data_load_elapsed = time.time() - data_load_start
        print(f"\n📊 数据加载总耗时: {data_load_elapsed:.2f}秒")
        
        return data
        
    except Exception as e:
        print(f"   ❌ 数据加载失败: {e}")
        return None


def analyze_data_scale(data):
    """分析数据规模"""
    print(f"\n📊 数据规模分析:")
    
    # 分析财务数据
    financial_data = data['financial_data']
    stocks = financial_data.index.get_level_values('StockCodes').unique()
    report_dates = financial_data.index.get_level_values('ReportDates').unique()
    
    print(f"   股票数量: {len(stocks):,}")
    print(f"   报告期数量: {len(report_dates):,}")
    print(f"   财务数据记录: {len(financial_data):,}")
    
    # 分析时间范围
    trading_dates = data['trading_dates']
    print(f"   交易日范围: {trading_dates.min()} 至 {trading_dates.max()}")
    print(f"   交易日数量: {len(trading_dates):,}")
    
    # 估算计算量
    estimated_factor_points = len(stocks) * len(trading_dates)
    print(f"   预估因子数据点: {estimated_factor_points:,}")
    
    return len(stocks), len(trading_dates), estimated_factor_points


def calculate_full_roe_ttm(data, save_results=True):
    """计算全量ROE_ttm因子"""
    print(f"\n🔧 因子计算阶段:")
    
    calc_start = time.time()
    
    try:
        # 创建ROE因子实例
        print(f"   创建ROE_ttm因子实例...")
        factor_start = time.time()
        roe_factor = ROEFactor(earnings_method='ttm')
        factor_elapsed = time.time() - factor_start
        print(f"   ✅ ROE因子创建成功 (耗时: {factor_elapsed:.3f}秒)")
        
        # 执行计算
        print(f"   开始全量计算ROE_ttm...")
        print(f"   ⏱️  计算开始时间: {datetime.now().strftime('%H:%M:%S')}")
        
        calculate_start = time.time()
        
        roe_values = roe_factor.calculate(
            financial_data=data['financial_data'],
            release_dates=data['release_dates'],
            trading_dates=data['trading_dates']
        )
        
        calculate_elapsed = time.time() - calculate_start
        print(f"   ✅ ROE_ttm计算完成!")
        print(f"   ⏱️  计算结束时间: {datetime.now().strftime('%H:%M:%S')}")
        print(f"   ⏱️  纯计算耗时: {calculate_elapsed:.2f}秒")
        
        # 分析结果
        print(f"\n📈 计算结果分析:")
        print(f"   结果数据点: {len(roe_values):,}")
        print(f"   有效值数量: {roe_values.count():,}")
        print(f"   有效率: {roe_values.count() / len(roe_values) * 100:.2f}%")
        print(f"   均值: {roe_values.mean():.6f}")
        print(f"   标准差: {roe_values.std():.6f}")
        print(f"   中位数: {roe_values.median():.6f}")
        print(f"   范围: [{roe_values.min():.6f}, {roe_values.max():.6f}]")
        
        # 分位数分析
        quantiles = roe_values.quantile([0.01, 0.05, 0.25, 0.75, 0.95, 0.99])
        print(f"\n   分位数分布:")
        for q, val in quantiles.items():
            print(f"      {q*100:4.0f}%: {val:.6f}")
        
        # 保存结果
        if save_results:
            print(f"\n💾 保存结果:")
            save_start = time.time()
            
            output_path = project_root / "factor_output"
            output_path.mkdir(exist_ok=True)
            
            # 保存为pickle格式（推荐，速度快）
            pkl_file = output_path / f"ROE_ttm_full_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            roe_values.to_pickle(pkl_file)
            
            # 保存为CSV格式（可读性好）
            csv_file = output_path / f"ROE_ttm_full_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            roe_values.to_csv(csv_file)
            
            save_elapsed = time.time() - save_start
            file_size_pkl = pkl_file.stat().st_size / 1024 / 1024  # MB
            file_size_csv = csv_file.stat().st_size / 1024 / 1024  # MB
            
            print(f"   ✅ PKL文件: {pkl_file}")
            print(f"      文件大小: {file_size_pkl:.1f} MB")
            print(f"   ✅ CSV文件: {csv_file}")
            print(f"      文件大小: {file_size_csv:.1f} MB")
            print(f"   💾 保存耗时: {save_elapsed:.2f}秒")
        
        calc_elapsed = time.time() - calc_start
        return roe_values, calculate_elapsed, calc_elapsed
        
    except Exception as e:
        print(f"   ❌ ROE_ttm计算失败: {e}")
        import traceback
        traceback.print_exc()
        return None, 0, 0


def performance_summary(data_load_time, pure_calc_time, total_calc_time, factor_points):
    """性能总结"""
    print(f"\n" + "=" * 60)
    print("性能统计总结")
    print("=" * 60)
    
    total_time = data_load_time + total_calc_time
    
    print(f"⏱️  时间统计:")
    print(f"   数据加载时间: {data_load_time:.2f}秒")
    print(f"   纯因子计算时间: {pure_calc_time:.2f}秒")
    print(f"   总计算时间(含保存): {total_calc_time:.2f}秒")
    print(f"   程序总耗时: {total_time:.2f}秒 ({total_time/60:.1f}分钟)")
    
    print(f"\n🚀 性能指标:")
    if factor_points > 0 and pure_calc_time > 0:
        calc_rate = factor_points / pure_calc_time
        print(f"   因子计算速度: {calc_rate:,.0f} 点/秒")
        print(f"   因子计算速度: {calc_rate * 60:,.0f} 点/分钟")
        
        if calc_rate > 100000:
            performance_level = "🔥 极快"
        elif calc_rate > 50000:
            performance_level = "⚡ 很快"
        elif calc_rate > 10000:
            performance_level = "✅ 良好"
        else:
            performance_level = "🐌 较慢"
            
        print(f"   性能评级: {performance_level}")
    
    print(f"\n💡 优化建议:")
    if pure_calc_time > 60:
        print(f"   - 考虑使用并行计算加速")
        print(f"   - 可以分批计算大型数据集")
    if data_load_time > pure_calc_time:
        print(f"   - 数据加载占用较多时间，考虑优化数据格式")
    
    return total_time


def main():
    """主函数"""
    program_start = time.time()
    
    # 1. 加载数据
    data = load_data()
    if data is None:
        print("❌ 数据加载失败，程序退出")
        return
    
    data_load_time = time.time() - program_start
    
    # 2. 分析数据规模
    stock_count, trading_days, estimated_points = analyze_data_scale(data)
    
    # 3. 询问用户确认
    print(f"\n❓ 确认信息:")
    print(f"   预估计算 {estimated_points:,} 个因子数据点")
    estimated_time = estimated_points / 50000  # 假设每秒5万点
    print(f"   预估耗时: {estimated_time:.1f}秒 ({estimated_time/60:.1f}分钟)")
    
    print(f"\n自动开始全量计算ROE_ttm...")
    # confirm = input(f"\n是否继续全量计算ROE_ttm? (Y/n): ").strip().lower()
    # if confirm == 'n':
    #     print("用户取消，程序退出")
    #     return
    
    # 4. 执行计算
    roe_values, pure_calc_time, total_calc_time = calculate_full_roe_ttm(data, save_results=True)
    
    if roe_values is not None:
        # 5. 性能总结
        actual_points = len(roe_values)
        performance_summary(data_load_time, pure_calc_time, total_calc_time, actual_points)
        
        print(f"\n🎉 全量ROE_ttm因子计算完成!")
        print(f"📊 实际计算了 {actual_points:,} 个数据点")
        print(f"⏱️  总耗时: {(time.time() - program_start):.2f}秒")
    else:
        print(f"\n❌ 计算失败")


if __name__ == "__main__":
    main()