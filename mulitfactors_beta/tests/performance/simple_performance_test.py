"""
简单的性能对比测试

比较原始实现和优化实现的性能
"""
import sys
import time
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.processor import PriceDataProcessor, ReturnCalculator
from data.processor.optimized_return_calculator import OptimizedReturnCalculator


def main():
    """主函数"""
    print("="*60)
    print("简单性能对比测试")
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # 准备测试数据
    print("\n准备测试数据...")
    processor = PriceDataProcessor()
    price_df = pd.read_pickle(processor.price_file_path)
    
    # 使用小样本：最近30天，前200只股票
    dates = price_df.index.get_level_values(0).unique()[-30:]
    stocks = price_df.index.get_level_values(1).unique()[:200]
    
    mask = (price_df.index.get_level_values(0).isin(dates) & 
            price_df.index.get_level_values(1).isin(stocks))
    test_data = price_df[mask]
    
    print(f"测试数据形状: {test_data.shape}")
    print(f"日期数: {len(dates)}, 股票数: {len(stocks)}")
    
    # 1. 测试原始实现
    print("\n1. 测试原始实现...")
    calculator_original = ReturnCalculator()
    
    start_time = time.time()
    result_original = calculator_original.calculate_log_return(
        test_data, dates, return_type="o2o"
    )
    time_original = time.time() - start_time
    
    print(f"   耗时: {time_original:.3f}秒")
    print(f"   结果形状: {result_original.shape}")
    print(f"   非空值比例: {(~result_original['LogReturn'].isna()).mean():.2%}")
    
    # 2. 测试向量化实现
    print("\n2. 测试向量化实现...")
    calculator_optimized = OptimizedReturnCalculator(use_parallel=False)
    
    start_time = time.time()
    result_vectorized = calculator_optimized.calculate_log_return_vectorized(
        test_data, dates, return_type="o2o"
    )
    time_vectorized = time.time() - start_time
    
    print(f"   耗时: {time_vectorized:.3f}秒")
    print(f"   结果形状: {result_vectorized.shape}")
    print(f"   非空值比例: {(~result_vectorized['LogReturn'].isna()).mean():.2%}")
    print(f"   加速比: {time_original/time_vectorized:.2f}x")
    
    # 3. 测试滚动收益率计算
    print("\n3. 测试滚动收益率计算...")
    
    # 原始实现
    start_time = time.time()
    rolling_original = calculator_original.calculate_n_days_return(
        result_original, lag=20
    )
    time_rolling_original = time.time() - start_time
    print(f"   原始实现耗时: {time_rolling_original:.3f}秒")
    
    # 优化实现
    start_time = time.time()
    rolling_optimized = calculator_optimized.calculate_n_days_return_optimized(
        result_vectorized, lag=20
    )
    time_rolling_optimized = time.time() - start_time
    print(f"   优化实现耗时: {time_rolling_optimized:.3f}秒")
    print(f"   加速比: {time_rolling_original/time_rolling_optimized:.2f}x")
    
    # 4. 验证结果一致性
    print("\n4. 验证结果一致性...")
    
    # 对齐索引进行比较
    common_index = result_original.index.intersection(result_vectorized.index)
    df1 = result_original.loc[common_index]
    df2 = result_vectorized.loc[common_index]
    
    # 计算差异
    diff = np.abs(df1['LogReturn'].values - df2['LogReturn'].values)
    max_diff = np.nanmax(diff)
    mean_diff = np.nanmean(diff)
    
    print(f"   最大差异: {max_diff:.2e}")
    print(f"   平均差异: {mean_diff:.2e}")
    print(f"   一致性检查: {'通过' if max_diff < 1e-10 else '失败'}")
    
    # 5. 性能总结
    print("\n" + "="*60)
    print("性能总结")
    print("="*60)
    print(f"日收益率计算:")
    print(f"  原始实现: {time_original:.3f}秒")
    print(f"  向量化实现: {time_vectorized:.3f}秒 (加速 {time_original/time_vectorized:.1f}x)")
    print(f"\n滚动收益率计算:")
    print(f"  原始实现: {time_rolling_original:.3f}秒")
    print(f"  优化实现: {time_rolling_optimized:.3f}秒 (加速 {time_rolling_original/time_rolling_optimized:.1f}x)")
    
    # 测试更大的数据集
    print("\n\n测试更大数据集的性能...")
    
    # 使用180天，1000只股票
    dates_large = price_df.index.get_level_values(0).unique()[-180:]
    stocks_large = price_df.index.get_level_values(1).unique()[:1000]
    
    mask_large = (price_df.index.get_level_values(0).isin(dates_large) & 
                  price_df.index.get_level_values(1).isin(stocks_large))
    test_data_large = price_df[mask_large]
    
    print(f"大数据集形状: {test_data_large.shape}")
    print(f"日期数: {len(dates_large)}, 股票数: {len(stocks_large)}")
    
    # 只测试向量化实现
    print("\n测试向量化实现在大数据集上的性能...")
    start_time = time.time()
    result_large = calculator_optimized.calculate_log_return_vectorized(
        test_data_large, dates_large, return_type="o2o"
    )
    time_large = time.time() - start_time
    
    print(f"   耗时: {time_large:.3f}秒")
    print(f"   处理速度: {len(dates_large) * len(stocks_large) / time_large:.0f} 数据点/秒")
    
    print("\n测试完成！")
    

if __name__ == "__main__":
    main()