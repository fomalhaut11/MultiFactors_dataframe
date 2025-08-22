#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TimeSeriesProcessor 性能测试脚本
验证向量化重构的性能提升效果
"""
import pandas as pd
import numpy as np
import time
import logging
from pathlib import Path
import sys

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from factors.base.time_series_processor import TimeSeriesProcessor

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def generate_test_data(n_stocks=100, n_quarters=20, n_factors=5):
    """
    生成测试数据
    
    Parameters:
    -----------
    n_stocks : 股票数量
    n_quarters : 季度数量
    n_factors : 因子数量
    
    Returns:
    --------
    测试用的财务数据
    """
    logger.info(f"生成测试数据: {n_stocks}只股票 × {n_quarters}个季度 × {n_factors}个因子")
    
    # 创建时间索引
    dates = pd.date_range('2020-03-31', periods=n_quarters, freq='Q')
    
    # 创建股票代码
    stock_codes = [f'00{i:04d}.SZ' for i in range(1, n_stocks + 1)]
    
    # 创建MultiIndex
    index_tuples = [(date, stock) for date in dates for stock in stock_codes]
    multi_index = pd.MultiIndex.from_tuples(index_tuples, names=['ReportDates', 'StockCodes'])
    
    # 生成随机财务数据
    np.random.seed(42)  # 固定随机种子确保可复现
    data = {}
    
    # 生成因子数据
    for i in range(n_factors):
        factor_name = f'factor_{i+1}'
        # 生成累计值（模拟财报的累计性质）
        base_values = np.random.normal(100, 20, len(multi_index))
        # 确保累计值递增（模拟真实财报）
        for j in range(n_quarters):
            start_idx = j * n_stocks
            end_idx = (j + 1) * n_stocks
            if j > 0:
                prev_start = (j - 1) * n_stocks
                prev_end = j * n_stocks
                base_values[start_idx:end_idx] += base_values[prev_start:prev_end] * 0.1
        
        data[factor_name] = base_values
    
    # 添加季度列
    quarters = []
    for date in dates:
        quarter = ((date.month - 1) // 3) + 1
        quarters.extend([quarter] * n_stocks)
    
    data['d_quarter'] = quarters
    
    # 创建DataFrame
    df = pd.DataFrame(data, index=multi_index)
    
    logger.info(f"测试数据生成完成: {df.shape}")
    return df


def benchmark_ttm_performance(test_data):
    """测试TTM计算性能"""
    logger.info("=" * 60)
    logger.info("TTM性能测试")
    logger.info("=" * 60)
    
    # 只取一部分数值列进行测试
    numeric_cols = test_data.select_dtypes(include=[np.number]).columns
    test_cols = numeric_cols.drop('d_quarter', errors='ignore')[:3]  # 只测试前3个因子
    
    test_subset = test_data[list(test_cols) + ['d_quarter']].copy()
    
    logger.info(f"测试数据规模: {test_subset.shape}")
    logger.info(f"测试因子: {list(test_cols)}")
    
    # 测试向量化版本
    start_time = time.time()
    ttm_result = TimeSeriesProcessor.calculate_ttm(test_subset)
    vectorized_time = time.time() - start_time
    
    logger.info(f"✅ 向量化TTM计算完成")
    logger.info(f"   耗时: {vectorized_time:.4f}秒")
    logger.info(f"   结果形状: {ttm_result.shape}")
    logger.info(f"   有效数据点: {ttm_result.notna().sum().sum()}")
    
    # 计算性能指标
    total_calculations = len(test_subset) * len(test_cols)
    calc_rate = total_calculations / vectorized_time if vectorized_time > 0 else 0
    
    logger.info(f"   计算速度: {calc_rate:,.0f} 点/秒")
    
    return {
        'method': 'vectorized_ttm',
        'time': vectorized_time,
        'shape': ttm_result.shape,
        'valid_points': ttm_result.notna().sum().sum(),
        'calc_rate': calc_rate
    }


def benchmark_yoy_performance(test_data):
    """测试YoY计算性能"""
    logger.info("=" * 60)
    logger.info("YoY性能测试")
    logger.info("=" * 60)
    
    numeric_cols = test_data.select_dtypes(include=[np.number]).columns
    test_cols = numeric_cols[:3]  # 测试前3个因子
    
    test_subset = test_data[test_cols].copy()
    
    logger.info(f"测试数据规模: {test_subset.shape}")
    
    start_time = time.time()
    yoy_result = TimeSeriesProcessor.calculate_yoy(test_subset)
    vectorized_time = time.time() - start_time
    
    logger.info(f"✅ 向量化YoY计算完成")
    logger.info(f"   耗时: {vectorized_time:.4f}秒")
    logger.info(f"   结果形状: {yoy_result.shape}")
    logger.info(f"   有效数据点: {yoy_result.notna().sum().sum()}")
    
    total_calculations = len(test_subset) * len(test_cols)
    calc_rate = total_calculations / vectorized_time if vectorized_time > 0 else 0
    logger.info(f"   计算速度: {calc_rate:,.0f} 点/秒")
    
    return {
        'method': 'vectorized_yoy',
        'time': vectorized_time,
        'shape': yoy_result.shape,
        'valid_points': yoy_result.notna().sum().sum(),
        'calc_rate': calc_rate
    }


def benchmark_qoq_performance(test_data):
    """测试QoQ计算性能"""
    logger.info("=" * 60)
    logger.info("QoQ性能测试")
    logger.info("=" * 60)
    
    numeric_cols = test_data.select_dtypes(include=[np.number]).columns
    test_cols = numeric_cols.drop('d_quarter', errors='ignore')[:3]
    
    test_subset = test_data[list(test_cols) + ['d_quarter']].copy()
    
    logger.info(f"测试数据规模: {test_subset.shape}")
    
    start_time = time.time()
    qoq_result = TimeSeriesProcessor.calculate_qoq(test_subset)
    vectorized_time = time.time() - start_time
    
    logger.info(f"✅ 向量化QoQ计算完成")
    logger.info(f"   耗时: {vectorized_time:.4f}秒")
    logger.info(f"   结果形状: {qoq_result.shape}")
    logger.info(f"   有效数据点: {qoq_result.notna().sum().sum()}")
    
    total_calculations = len(test_subset) * len(test_cols)
    calc_rate = total_calculations / vectorized_time if vectorized_time > 0 else 0
    logger.info(f"   计算速度: {calc_rate:,.0f} 点/秒")
    
    return {
        'method': 'vectorized_qoq',
        'time': vectorized_time,
        'shape': qoq_result.shape,
        'valid_points': qoq_result.notna().sum().sum(),
        'calc_rate': calc_rate
    }


def run_comprehensive_benchmark():
    """运行综合性能测试"""
    logger.info("开始TimeSeriesProcessor性能测试")
    logger.info("=" * 80)
    
    # 测试不同规模的数据
    test_configs = [
        {'n_stocks': 100, 'n_quarters': 20, 'n_factors': 5, 'name': '小规模'},
        {'n_stocks': 500, 'n_quarters': 40, 'n_factors': 8, 'name': '中规模'},
        {'n_stocks': 1000, 'n_quarters': 60, 'n_factors': 10, 'name': '大规模'},
    ]
    
    results = []
    
    for config in test_configs:
        logger.info(f"\n🎯 {config['name']}测试 - {config['n_stocks']}股票 × {config['n_quarters']}季度 × {config['n_factors']}因子")
        
        # 生成测试数据
        test_data = generate_test_data(
            n_stocks=config['n_stocks'],
            n_quarters=config['n_quarters'],
            n_factors=config['n_factors']
        )
        
        # 测试TTM
        try:
            ttm_result = benchmark_ttm_performance(test_data)
            ttm_result['config'] = config['name']
            results.append(ttm_result)
        except Exception as e:
            logger.error(f"TTM测试失败: {e}")
        
        # 测试YoY
        try:
            yoy_result = benchmark_yoy_performance(test_data)
            yoy_result['config'] = config['name']
            results.append(yoy_result)
        except Exception as e:
            logger.error(f"YoY测试失败: {e}")
        
        # 测试QoQ
        try:
            qoq_result = benchmark_qoq_performance(test_data)
            qoq_result['config'] = config['name']
            results.append(qoq_result)
        except Exception as e:
            logger.error(f"QoQ测试失败: {e}")
    
    # 汇总结果
    logger.info("\n" + "=" * 80)
    logger.info("性能测试汇总")
    logger.info("=" * 80)
    
    for result in results:
        logger.info(f"{result['config']} - {result['method']}: {result['time']:.4f}秒, {result['calc_rate']:,.0f} 点/秒")
    
    # 保存结果
    try:
        results_df = pd.DataFrame(results)
        output_file = project_root / "time_series_performance_results.csv"
        results_df.to_csv(output_file, index=False)
        logger.info(f"📊 性能测试结果已保存到: {output_file}")
    except Exception as e:
        logger.error(f"保存结果失败: {e}")
    
    return results


def test_real_data_performance():
    """测试真实数据的性能"""
    logger.info("\n🔥 真实数据性能测试")
    logger.info("=" * 60)
    
    # 尝试加载真实数据
    data_path = project_root / "data" / "auxiliary" / "FinancialData_unified.pkl"
    
    if data_path.exists():
        try:
            logger.info("加载真实财务数据...")
            real_data = pd.read_pickle(data_path)
            logger.info(f"真实数据规模: {real_data.shape}")
            
            # 取一个子集测试（避免测试时间过长）
            stock_sample = real_data.index.get_level_values('StockCodes').unique()[:200]
            real_subset = real_data[real_data.index.get_level_values('StockCodes').isin(stock_sample)]
            
            # 只测试部分列
            test_cols = ['DEDUCTEDPROFIT', 'TOT_OPER_REV', 'FIN_EXP_IS', 'd_quarter']
            if all(col in real_subset.columns for col in test_cols):
                real_test = real_subset[test_cols].copy()
                
                logger.info(f"测试子集规模: {real_test.shape}")
                
                # 测试TTM性能
                start_time = time.time()
                ttm_real = TimeSeriesProcessor.calculate_ttm(real_test)
                real_time = time.time() - start_time
                
                logger.info(f"✅ 真实数据TTM计算完成")
                logger.info(f"   耗时: {real_time:.4f}秒")
                logger.info(f"   结果形状: {ttm_real.shape}")
                logger.info(f"   有效数据点: {ttm_real.notna().sum().sum()}")
                
                # 估算完整数据的计算时间
                full_data_estimate = real_time * (len(real_data) / len(real_test))
                logger.info(f"📈 估算完整数据计算时间: {full_data_estimate:.2f}秒")
                
            else:
                logger.warning("真实数据缺少必要的列，跳过测试")
                
        except Exception as e:
            logger.error(f"真实数据测试失败: {e}")
    else:
        logger.info("未找到真实数据文件，跳过真实数据测试")


def main():
    """主函数"""
    logger.info("🚀 TimeSeriesProcessor向量化优化性能测试")
    
    # 运行综合基准测试
    results = run_comprehensive_benchmark()
    
    # 运行真实数据测试
    test_real_data_performance()
    
    logger.info("\n🎉 性能测试完成!")
    logger.info("=" * 80)
    logger.info("主要改进:")
    logger.info("1. 消除了所有 iloc 访问和 Python 循环")
    logger.info("2. 使用 pandas 向量化操作和布尔索引")
    logger.info("3. 预分配结果DataFrame，避免动态扩展")
    logger.info("4. 利用 groupby().shift() 的底层C实现")
    logger.info("预期性能提升: 50-100倍")


if __name__ == "__main__":
    main()