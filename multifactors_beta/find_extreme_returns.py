#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
直接查找5日收益率极值 - 只读取已存储数据，不重新计算
"""

import pandas as pd
import numpy as np
from pathlib import Path

# 直接读取已存储的5日收益率数据
data_file = Path('E:/Documents/PythonProject/StockProject/StockData/factors/technical/Returns_5D_C2C.pkl')

print("正在读取已存储的5日收益率数据...")
returns_data = pd.read_pickle(data_file)

print(f"数据形状: {returns_data.shape}")
print(f"数据范围: [{returns_data.min():.6f}, {returns_data.max():.6f}]")

# 找最大亏损
min_return = returns_data.min()
min_idx = returns_data.idxmin()

print("\n【最大亏损】")
print(f"对数收益率: {min_return:.6f}")
print(f"实际收益率: {(np.exp(min_return) - 1) * 100:.2f}%")
print(f"股票代码: {min_idx[1]}")
print(f"发生日期: {min_idx[0]}")

# 找最大收益  
max_return = returns_data.max()
max_idx = returns_data.idxmax()

print("\n【最大收益】")
print(f"对数收益率: {max_return:.6f}")
print(f"实际收益率: {(np.exp(max_return) - 1) * 100:.2f}%")
print(f"股票代码: {max_idx[1]}")
print(f"发生日期: {max_idx[0]}")

# 更多极值案例
print("\n【最大亏损TOP5】")
worst_5 = returns_data.nsmallest(5)
for i, (idx, value) in enumerate(worst_5.items(), 1):
    actual_return = (np.exp(value) - 1) * 100
    print(f"{i}. {idx[0]} {idx[1]}: {actual_return:+6.2f}%")

print("\n【最大收益TOP5】")  
best_5 = returns_data.nlargest(5)
for i, (idx, value) in enumerate(best_5.items(), 1):
    actual_return = (np.exp(value) - 1) * 100
    print(f"{i}. {idx[0]} {idx[1]}: {actual_return:+6.2f}%")