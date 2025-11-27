#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试财务数据字段名称
"""

from factors.utils.data_loader import FactorDataLoader
from factors.generators import calculate_ttm
import pandas as pd

# 加载财务数据
financial_data = FactorDataLoader.load_financial_data()
print(f"财务数据形状: {financial_data.shape}")
print(f"财务数据列数: {len(financial_data.columns)}")

# 查看所有字段
print("\n所有字段名称:")
for i, col in enumerate(financial_data.columns):
    print(f"{i+1:3d}: {col}")

# 查看与利润相关的字段
print("\n与利润相关的字段:")
profit_related = [col for col in financial_data.columns if any(keyword in col.lower() for keyword in ['profit', 'income', 'earn', '利润', 'net'])]
for col in profit_related:
    print(f"  {col}")

# 查看与费用相关的字段
print("\n与费用相关的字段:")
expense_related = [col for col in financial_data.columns if any(keyword in col.lower() for keyword in ['expense', 'cost', '费用', 'financial'])]
for col in expense_related:
    print(f"  {col}")

# 查看与存货相关的字段
print("\n与存货相关的字段:")
inventory_related = [col for col in financial_data.columns if any(keyword in col.lower() for keyword in ['inventory', 'stock', '存货', 'invt'])]
for col in inventory_related:
    print(f"  {col}")

# 查看与负债相关的字段
print("\n与负债相关的字段:")
debt_related = [col for col in financial_data.columns if any(keyword in col.lower() for keyword in ['debt', 'liab', '负债', 'current', 'short'])]
for col in debt_related:
    print(f"  {col}")

# 测试TTM计算
print("\n测试TTM计算...")
try:
    ttm_data = calculate_ttm(financial_data)
    print(f"TTM数据计算成功，字段包括:")
    for col in ttm_data.columns:
        print(f"  {col}")
        
    # 查看数据样本
    print(f"\nTTM数据前5行:")
    print(ttm_data.head())
    
except Exception as e:
    print(f"TTM计算失败: {e}")
    
# 查看原始数据样本
print(f"\n原始财务数据前5行:")
print(financial_data.head())