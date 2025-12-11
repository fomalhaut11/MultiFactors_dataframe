#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
并行因子计算脚本
支持多种因子的快速并行计算
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import logging
import time
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from factors.financial.fundamental_factors import ROEFactor, BPFactor, EPFactor
from factors.calculator.factor_calculator import FactorCalculator

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_all_data():
    """加载全部数据"""
    print("=" * 60)
    print("并行因子计算")
    print(f"开始时间: {datetime.now()}")
    print("=" * 60)
    
    data_load_start = time.time()
    
    data_path = Path(r"E:\Documents\PythonProject\StockProject\StockData")
    auxiliary_path = project_root / "data" / "auxiliary"
    
    print(f"\n📂 数据加载:")
    
    data = {}
    file_info = {
        'financial_data': auxiliary_path / "FinancialData_unified.pkl",
        'release_dates': auxiliary_path / "ReleaseDates.pkl", 
        'trading_dates': auxiliary_path / "TradingDates.pkl",
        'market_cap': data_path / "MarketCap.pkl"
    }
    
    for name, filepath in file_info.items():
        if filepath.exists():
            start = time.time()
            data[name] = pd.read_pickle(filepath)
            elapsed = time.time() - start
            
            if hasattr(data[name], 'shape'):
                size_info = f"{data[name].shape}"
            else:
                size_info = f"长度: {len(data[name])}"
            
            print(f"   ✅ {name}: {size_info} ({elapsed:.2f}秒)")
        else:
            print(f"   ❌ {name}: 文件不存在")
            return None
    
    # 处理市值数据格式
    if 'market_cap' in data and isinstance(data['market_cap'], pd.DataFrame):
        data['market_cap'] = data['market_cap'].iloc[:, 0]
    
    data_load_elapsed = time.time() - data_load_start
    print(f"\n📊 数据加载总耗时: {data_load_elapsed:.2f}秒")
    
    return data, data_load_elapsed


def analyze_computation_scale(data):
    """分析计算规模"""
    print(f"\n📊 计算规模分析:")
    
    financial_data = data['financial_data']
    stocks = financial_data.index.get_level_values('StockCodes').unique()
    trading_dates = data['trading_dates']
    
    print(f"   股票数量: {len(stocks):,}")
    print(f"   交易日数量: {len(trading_dates):,}")
    print(f"   预估因子点数: {len(stocks) * len(trading_dates):,}")
    
    return len(stocks), len(trading_dates)


def calculate_single_factor(factor_config):
    """计算单个因子的函数（用于并行）"""
    factor_name, factor_class, params, data = factor_config
    
    try:
        print(f"   🔧 开始计算 {factor_name}...")
        start_time = time.time()
        
        # 创建因子实例
        if params:
            factor = factor_class(**params)
        else:
            factor = factor_class()
        
        # 根据因子类型准备参数
        calc_params = {}
        
        # 基本面因子通常需要这些数据
        if hasattr(factor, 'category') and factor.category == 'fundamental':
            calc_params['financial_data'] = data['financial_data']
            calc_params['release_dates'] = data['release_dates']
            calc_params['trading_dates'] = data['trading_dates']
            
            # EP和BP因子还需要市值数据
            if factor_name.startswith('EP') or factor_name.startswith('BP'):
                calc_params['market_cap'] = data['market_cap']
        
        # 计算因子
        result = factor.calculate(**calc_params)
        
        elapsed = time.time() - start_time
        print(f"   ✅ {factor_name} 完成: {len(result):,}点, {result.count():,}有效 ({elapsed:.2f}秒)")
        
        return factor_name, result, elapsed, None
        
    except Exception as e:
        elapsed = time.time() - start_time if 'start_time' in locals() else 0
        print(f"   ❌ {factor_name} 失败: {e} ({elapsed:.2f}秒)")
        return factor_name, None, elapsed, str(e)


def calculate_factors_parallel(data, factor_configs, max_workers=None):
    """并行计算多个因子"""
    print(f"\n🚀 并行因子计算:")
    
    if max_workers is None:
        max_workers = max(1, mp.cpu_count() - 1)
    
    print(f"   并行工作进程: {max_workers}")
    print(f"   计算因子数量: {len(factor_configs)}")
    
    calc_start = time.time()
    results = {}
    timing_info = {}
    errors = {}
    
    # 准备计算配置（添加数据）
    calc_configs = []
    for name, factor_class, params in factor_configs:
        calc_configs.append((name, factor_class, params, data))
    
    # 使用线程池（因为主要是计算密集型，但避免进程间数据传输开销）
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_factor = {
            executor.submit(calculate_single_factor, config): config[0] 
            for config in calc_configs
        }
        
        # 收集结果
        for future in as_completed(future_to_factor):
            factor_name = future_to_factor[future]
            try:
                name, result, elapsed, error = future.result()
                
                timing_info[name] = elapsed
                
                if result is not None:
                    results[name] = result
                else:
                    errors[name] = error
                    
            except Exception as e:
                errors[factor_name] = str(e)
                print(f"   ❌ {factor_name} 执行异常: {e}")
    
    calc_elapsed = time.time() - calc_start
    
    # 统计结果
    print(f"\n📈 并行计算结果:")
    print(f"   总耗时: {calc_elapsed:.2f}秒")
    print(f"   成功因子: {len(results)}")
    print(f"   失败因子: {len(errors)}")
    
    if results:
        avg_time = sum(timing_info[name] for name in results.keys()) / len(results)
        print(f"   平均单因子耗时: {avg_time:.2f}秒")
        
        total_points = sum(len(result) for result in results.values())
        print(f"   总因子数据点: {total_points:,}")
        
        if calc_elapsed > 0:
            calc_rate = total_points / calc_elapsed
            print(f"   整体计算速度: {calc_rate:,.0f} 点/秒")
    
    if errors:
        print(f"\n❌ 计算失败的因子:")
        for name, error in errors.items():
            print(f"   - {name}: {error}")
    
    return results, timing_info, errors, calc_elapsed


def save_results(results, timing_info):
    """保存计算结果"""
    print(f"\n💾 保存结果:")
    
    save_start = time.time()
    output_path = project_root / "factor_output"
    output_path.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    saved_files = []
    
    for factor_name, factor_data in results.items():
        # 保存单个因子
        pkl_file = output_path / f"{factor_name}_{timestamp}.pkl"
        factor_data.to_pickle(pkl_file)
        
        file_size = pkl_file.stat().st_size / 1024 / 1024
        saved_files.append((factor_name, pkl_file, file_size))
        
        print(f"   ✅ {factor_name}: {pkl_file.name} ({file_size:.1f}MB)")
    
    # 保存合并的因子数据
    if len(results) > 1:
        combined_df = pd.DataFrame(results)
        combined_file = output_path / f"factors_combined_{timestamp}.pkl"
        combined_df.to_pickle(combined_file)
        
        combined_size = combined_file.stat().st_size / 1024 / 1024
        print(f"   ✅ 合并文件: {combined_file.name} ({combined_size:.1f}MB)")
        
        # 保存相关性矩阵
        corr_matrix = combined_df.corr()
        corr_file = output_path / f"factor_correlation_{timestamp}.csv"
        corr_matrix.to_csv(corr_file)
        print(f"   ✅ 相关性矩阵: {corr_file.name}")
    
    # 保存性能统计
    perf_file = output_path / f"performance_log_{timestamp}.csv"
    perf_df = pd.DataFrame([
        {'factor': name, 'elapsed_time': timing, 'data_points': len(results[name])}
        for name, timing in timing_info.items() if name in results
    ])
    perf_df.to_csv(perf_file, index=False)
    print(f"   ✅ 性能日志: {perf_file.name}")
    
    save_elapsed = time.time() - save_start
    print(f"   💾 保存耗时: {save_elapsed:.2f}秒")
    
    return save_elapsed


def main():
    """主函数"""
    program_start = time.time()
    
    # 1. 加载数据
    data_result = load_all_data()
    if data_result is None:
        print("❌ 数据加载失败")
        return
    
    data, data_load_time = data_result
    
    # 2. 分析计算规模
    stock_count, trading_days = analyze_computation_scale(data)
    
    # 3. 定义要计算的因子
    factor_configs = [
        ('ROE_ttm', ROEFactor, {'earnings_method': 'ttm'}),
        ('BP', BPFactor, None),
        ('EP_ttm', EPFactor, {'method': 'ttm'}),
    ]
    
    print(f"\n🎯 计算配置:")
    for name, _, params in factor_configs:
        print(f"   - {name}: {params if params else '默认参数'}")
    
    estimated_points = stock_count * trading_days * len(factor_configs)
    estimated_time = estimated_points / 100000  # 假设每秒10万点
    
    print(f"\n❓ 确认信息:")
    print(f"   计算因子数: {len(factor_configs)}")
    print(f"   预估总数据点: {estimated_points:,}")
    print(f"   预估耗时: {estimated_time:.1f}秒 ({estimated_time/60:.1f}分钟)")
    
    confirm = input(f"\n是否开始并行计算? (Y/n): ").strip().lower()
    if confirm == 'n':
        print("用户取消")
        return
    
    # 4. 并行计算因子
    results, timing_info, errors, calc_time = calculate_factors_parallel(
        data, factor_configs, max_workers=3
    )
    
    if not results:
        print("❌ 没有成功计算的因子")
        return
    
    # 5. 保存结果
    save_time = save_results(results, timing_info)
    
    # 6. 最终总结
    total_time = time.time() - program_start
    
    print(f"\n" + "=" * 60)
    print("最终统计")
    print("=" * 60)
    print(f"⏱️  数据加载: {data_load_time:.2f}秒")
    print(f"⏱️  因子计算: {calc_time:.2f}秒")
    print(f"⏱️  结果保存: {save_time:.2f}秒")
    print(f"⏱️  程序总耗时: {total_time:.2f}秒 ({total_time/60:.1f}分钟)")
    
    if results:
        total_points = sum(len(result) for result in results.values())
        print(f"📊 成功计算 {len(results)} 个因子")
        print(f"📊 总数据点: {total_points:,}")
        
        if calc_time > 0:
            overall_rate = total_points / calc_time
            print(f"🚀 整体速度: {overall_rate:,.0f} 点/秒")
    
    print(f"🎉 批量因子计算完成!")


if __name__ == "__main__":
    main()