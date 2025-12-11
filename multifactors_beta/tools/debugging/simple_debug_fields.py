#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化调试财务数据字段
"""

from factors.utils.data_loader import FactorDataLoader
from factors.generators import calculate_ttm
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)

# 加载财务数据
print("加载财务数据...")
financial_data = FactorDataLoader.load_financial_data()
print(f"财务数据形状: {financial_data.shape}")

# 查看与利润相关的字段
profit_keywords = ['profit', 'income', 'earn', 'net']
profit_fields = []
for col in financial_data.columns:
    if any(keyword in col.lower() for keyword in profit_keywords):
        profit_fields.append(col)

print(f"\n找到与利润相关的字段 ({len(profit_fields)} 个):")
for field in profit_fields:
    print(f"  {field}")

# 查看关键字段的数据情况
key_fields = ['NET_PROFIT_IS', 'TOT_PROFIT', 'NP_BELONGTO_PARCOMSH']
print(f"\n关键利润字段数据情况:")
for field in key_fields:
    if field in financial_data.columns:
        non_null_count = financial_data[field].notna().sum()
        total_count = len(financial_data)
        print(f"  {field}: {non_null_count}/{total_count} ({non_null_count/total_count:.2%})")

# 测试TTM计算
print(f"\n测试TTM计算...")
try:
    # 使用一个小样本测试
    sample_data = financial_data.head(1000)
    ttm_data = calculate_ttm(sample_data)
    print(f"TTM计算成功，返回字段:")
    for col in ttm_data.columns:
        print(f"  {col}")
except Exception as e:
    print(f"TTM计算失败: {e}")
    import traceback
    traceback.print_exc()