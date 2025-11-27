"""
诊断EP_ttm数据问题
确认数据单位和计算逻辑
"""

import pandas as pd
import numpy as np
from pathlib import Path

data_root = Path("E:/Documents/PythonProject/StockProject/StockData")

# 1. 加载EP_ttm数据
ep_ttm_path = data_root / "RawFactors" / "EP_ttm.pkl"
ep_ttm = pd.read_pickle(ep_ttm_path)

print("EP_ttm数据信息:")
print(f"  形状: {ep_ttm.shape}")
print(f"  数据类型: {ep_ttm.dtype}")

# 2. 查看EP_ttm的数值范围
valid_ep = ep_ttm[ep_ttm.notna() & (ep_ttm != 0)]
print(f"\nEP_ttm数值分布:")
print(f"  有效值数量: {len(valid_ep)}")
print(f"  最小值: {valid_ep.min():.6f}")
print(f"  25%分位: {valid_ep.quantile(0.25):.6f}")
print(f"  中位数: {valid_ep.median():.6f}")
print(f"  75%分位: {valid_ep.quantile(0.75):.6f}")
print(f"  最大值: {valid_ep.max():.6f}")
print(f"  平均值: {valid_ep.mean():.6f}")

# 3. EP转PE的分布
pe_from_ep = 1 / valid_ep
pe_from_ep = pe_from_ep[pe_from_ep < 1000]  # 剔除极端值

print(f"\nPE (从EP转换) 数值分布:")
print(f"  有效值数量: {len(pe_from_ep)}")
print(f"  最小值: {pe_from_ep.min():.2f}")
print(f"  25%分位: {pe_from_ep.quantile(0.25):.2f}")
print(f"  中位数: {pe_from_ep.median():.2f}")
print(f"  75%分位: {pe_from_ep.quantile(0.75):.2f}")
print(f"  最大值: {pe_from_ep.max():.2f}")
print(f"  平均值: {pe_from_ep.mean():.2f}")

# 4. 检查特定日期和股票
latest_date = ep_ttm.index.get_level_values('TradingDates').max()
print(f"\n最新日期: {latest_date}")

# 获取最新日期的数据
latest_ep = ep_ttm.xs(latest_date, level='TradingDates')
valid_latest = latest_ep[latest_ep.notna() & (latest_ep != 0)]

print(f"\n最新日期EP分布:")
print(f"  股票数量: {len(valid_latest)}")
print(f"  EP中位数: {valid_latest.median():.6f}")

# 转换为PE
pe_latest = 1 / valid_latest
pe_latest = pe_latest[pe_latest < 1000]
print(f"\n最新日期PE分布:")
print(f"  PE中位数: {pe_latest.median():.2f}")
print(f"  PE平均值: {pe_latest.mean():.2f}")

# 5. 抽样几个股票看看
sample_stocks = ['000001', '000002', '600000', '600036']
print(f"\n样本股票EP和PE值:")

for stock in sample_stocks:
    if (latest_date, stock) in ep_ttm.index:
        ep_val = ep_ttm.loc[(latest_date, stock)]
        if pd.notna(ep_val) and ep_val != 0:
            pe_val = 1 / ep_val
            print(f"  {stock}: EP={ep_val:.6f}, PE={pe_val:.2f}")

# 6. 检查EP_ttm可能的单位问题
print("\n检查EP值是否可能是百分比形式:")
# 如果EP是百分比，那么应该在0-100之间
if valid_ep.max() < 1:
    print("  EP值都小于1，可能是小数形式")
    print("  如果净利润/市值本应该是百分比，现在是小数")
elif valid_ep.max() > 100:
    print("  EP值有大于100的，可能不是百分比")
else:
    print("  EP值在0-100之间，可能是百分比形式")

# 7. 尝试不同的转换方式
print("\n尝试不同的EP到PE转换方式:")
print("1. 直接倒数: PE = 1/EP")
sample_ep = 0.05
print(f"   EP={sample_ep} → PE={1/sample_ep:.2f}")

print("2. 如果EP是百分比: PE = 100/EP")
print(f"   EP={sample_ep} → PE={100/sample_ep:.2f}")

# 8. 加载市值数据对比
price_path = data_root / "Price.pkl"
price_df = pd.read_pickle(price_path)
market_cap = price_df['MC']

# 找一个共同的样本
common_idx = ep_ttm.index.intersection(market_cap.index)
if len(common_idx) > 0:
    sample_idx = common_idx[0]
    sample_date, sample_stock = sample_idx

    ep_val = ep_ttm.loc[sample_idx]
    mc_val = market_cap.loc[sample_idx]

    if pd.notna(ep_val) and pd.notna(mc_val) and ep_val != 0:
        print(f"\n实际数据验证:")
        print(f"  样本: {sample_stock} @ {sample_date}")
        print(f"  市值: {mc_val/1e8:.2f} 亿元")
        print(f"  EP: {ep_val:.6f}")
        print(f"  PE (1/EP): {1/ep_val:.2f}")
        print(f"  隐含净利润 (市值*EP): {mc_val*ep_val/1e8:.2f} 亿元")
        print(f"  隐含净利润 (市值/PE): {mc_val/(1/ep_val)/1e8:.2f} 亿元")