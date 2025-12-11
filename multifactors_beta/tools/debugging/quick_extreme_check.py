#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速检查5日收益率极值
"""

import pandas as pd
import numpy as np
from pathlib import Path

# 直接读取已存储的5日收益率数据
data_file = Path('E:/Documents/PythonProject/StockProject/StockData/factors/technical/Returns_5D_C2C.pkl')

print("正在读取数据...")
returns_data = pd.read_pickle(data_file)

print(f"数据形状: {returns_data.shape}")

# 快速获取基本统计信息
print("\n基本统计:")
print(f"最小值: {returns_data.min():.6f}")
print(f"最大值: {returns_data.max():.6f}")
print(f"均值: {returns_data.mean():.6f}")

# 转换为实际收益率百分比
min_actual = (np.exp(returns_data.min()) - 1) * 100
max_actual = (np.exp(returns_data.max()) - 1) * 100

print(f"\n实际收益率:")
print(f"最大亏损: {min_actual:.2f}%")
print(f"最大收益: {max_actual:.2f}%")

# 使用sample来找到具体的极值位置（避免全量扫描）
print("\n正在定位极值位置...")

# 直接定位最小值和最大值的索引
min_val = returns_data.min()
max_val = returns_data.max()

# 找到所有等于最小值的位置
min_positions = returns_data[returns_data == min_val]
max_positions = returns_data[returns_data == max_val]

print(f"\n【最大亏损详情】")
print(f"亏损幅度: {min_actual:.2f}%")
print(f"发生次数: {len(min_positions)}")
if len(min_positions) > 0:
    first_min = min_positions.index[0]
    print(f"股票代码: {first_min[1]}")
    print(f"发生日期: {first_min[0]}")

print(f"\n【最大收益详情】")
print(f"收益幅度: {max_actual:.2f}%")
print(f"发生次数: {len(max_positions)}")
if len(max_positions) > 0:
    first_max = max_positions.index[0]
    print(f"股票代码: {first_max[1]}")
    print(f"发生日期: {first_max[0]}")

print("\n完成！")