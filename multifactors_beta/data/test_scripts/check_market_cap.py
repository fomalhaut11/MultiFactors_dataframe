"""
检查市值数据的单位和格式
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

auxiliary_path = Path(__file__).parent.parent / "auxiliary"

# 加载市值数据
market_cap = pd.read_pickle(auxiliary_path / "MarketCap.pkl")

print("=" * 60)
print("市值数据检查")
print("=" * 60)

print(f"\n数据类型: {type(market_cap)}")
print(f"数据形状: {market_cap.shape}")
print(f"索引: {market_cap.index.names}")

# 查看数值范围
print(f"\n市值数据统计:")
print(f"最小值: {market_cap.min():.6f}")
print(f"25分位: {market_cap.quantile(0.25):.6f}")
print(f"中位数: {market_cap.median():.6f}")
print(f"75分位: {market_cap.quantile(0.75):.6f}")
print(f"最大值: {market_cap.max():.6f}")
print(f"均值: {market_cap.mean():.6f}")

# 检查是否是对数值
print(f"\n检查是否是对数市值:")
sample_values = market_cap.dropna().head(10)
print("前10个非空值:")
for i, val in enumerate(sample_values):
    exp_val = np.exp(val)
    print(f"  原值: {val:.6f} -> exp(原值): {exp_val/1e8:.2f}亿元")

# 随机抽样一些大的值
large_values = market_cap[market_cap > market_cap.quantile(0.95)].head(5)
print(f"\n最大的5个值:")
for val in large_values:
    exp_val = np.exp(val)
    print(f"  原值: {val:.6f} -> exp(原值): {exp_val/1e8:.2f}亿元")

# 检查实际的价格数据对比
data_root = Path("E:/Documents/PythonProject/StockProject/StockData")
price_path = data_root / "Price.pkl"
if price_path.exists():
    price_data = pd.read_pickle(price_path)
    print(f"\n价格数据形状: {price_data.shape}")

    # 如果价格数据中有市值字段
    if 'MC' in price_data.columns:
        print(f"\n价格数据中的市值(MC):")
        mc_in_price = price_data['MC']
        print(f"最小值: {mc_in_price.min():.2f}")
        print(f"中位数: {mc_in_price.median():.2f}")
        print(f"最大值: {mc_in_price.max():.2f}")

        # 对比同一日期同一股票
        sample_date = market_cap.index[0][0]
        sample_stock = market_cap.index[0][1]

        if (sample_date, sample_stock) in market_cap.index and (sample_date, sample_stock) in mc_in_price.index:
            mc1 = market_cap.loc[(sample_date, sample_stock)]
            mc2 = mc_in_price.loc[(sample_date, sample_stock)]
            print(f"\n同一股票对比 ({sample_date}, {sample_stock}):")
            print(f"  MarketCap.pkl: {mc1:.6f}")
            print(f"  Price.pkl中的MC: {mc2:.2f}")
            print(f"  是否是对数关系: exp({mc1:.2f}) = {np.exp(mc1):.2f} vs {mc2:.2f}")