#!/usr/bin/env python3
"""演示完整的factors生态系统使用"""

import sys
sys.path.append('E:/Documents/PythonProject/StockProject/MultiFactors/multifactors_beta')

import factors
import pandas as pd
import numpy as np

print("=== 完整因子研究生态系统演示 ===\n")

# 1. 系统概览
summary = factors.get_factor_summary()
print(f"系统状态:")
print(f"  注册因子: {summary['total_factors']}个")
print(f"  因子类别: {summary['categories']}")
print(f"  类别分布: {summary['category_counts']}")

# 2. 数据准备演示
print(f"\n=== 数据处理工具演示 ===")
# 创建模拟财务数据
dates = pd.to_datetime(['2020-12-31', '2021-12-31'])
stocks = ['000001.SZ', '000002.SZ'] 
index = pd.MultiIndex.from_product([dates, stocks], names=['ReportDates', 'StockCodes'])

test_data = pd.DataFrame({
    'DEDUCTEDPROFIT': [1000000, 1200000, 800000, 900000],
    'EQY_BELONGTO_PARCOMSH': [10000000, 12000000, 8000000, 9000000],
    'd_year': [2020, 2021, 2020, 2021],
    'd_quarter': [4, 4, 4, 4]
}, index=index)

# 使用基础工具
ttm_result = factors.calculate_ttm(test_data)
print(f"TTM计算: {ttm_result.shape[0]}行, {ttm_result.shape[1]}列")

# 3. 因子计算演示
print(f"\n=== 因子计算演示 ===")
roe = factors.calculate_factor('ROE_ttm', test_data)
print(f"ROE因子: {roe.notna().sum()}个有效值")

# 批量计算
results = factors.batch_calculate_factors(['ROE_ttm', 'ROA_ttm'], test_data)
print(f"批量计算: {len(results)}个因子")

# 4. 展示完整的研究工作流
print(f"\n=== 完整研究工作流程 ===")
print("第一步: 基础数据处理")
print("  - factors.calculate_ttm() - TTM计算")
print("  - factors.ts_rank() - Alpha191时序排名")
print("  - factors.calculate_zscore() - 标准化")

print("\n第二步: 因子生成") 
print("  - factors.calculate_factor() - 单个因子计算")
print("  - factors.batch_calculate_factors() - 批量计算")
print("  - factors.list_factors() - 查看可用因子")

print("\n第三步: 因子测试")
print("  - factors.test_factor() - 快速测试")
print("  - factors.SingleFactorTestPipeline() - 详细测试")
print("  - factors.batch_test() - 批量测试")

print("\n第四步: 因子分析")
print("  - factors.FactorScreener() - 因子筛选器")
print("  - screener.screen_factors() - 筛选因子")
print("  - screener.analyze_factors() - 详细分析")

print("\n第五步: 因子组合")
print("  - factors.FactorCombiner() - 因子组合器")
print("  - combiner.combine_factors() - 线性组合")
print("  - combiner.orthogonalize_factors() - 正交化")

print(f"\n=== 架构总结 ===")
print("二层核心架构:")
print("  - generators/: 基础数据处理工具(calculate_ttm, ts_rank等)")
print("  - library/: 因子注册和计算接口(calculate_factor等)")

print("\n支撑模块:")
print("  - tester/: 因子测试框架")
print("  - analyzer/: 因子分析工具")  
print("  - combiner/: 因子组合工具")
print("  - base/: 基础类和混入")

print(f"\n统一接口特点:")
print("  - 单一导入: import factors")
print("  - 功能完整: 覆盖因子研究全流程") 
print("  - 架构清晰: 二层核心+支撑模块")
print("  - 内聚性强: 所有功能都在factors模块内")

print(f"\n=== 生态系统演示完成 ===")
print("factors模块现已成为完整的量化因子研究平台！")