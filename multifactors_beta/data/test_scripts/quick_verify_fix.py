"""
快速验证板块PE计算修复
只计算最新一天的数据
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from data.processor import SectorValuationFromStockPE

print("="*60)
print("快速验证板块PE计算修复")
print("="*60)

# 创建处理器
processor = SectorValuationFromStockPE()

# 只计算最新1天
print("\n计算最新1天的板块估值...")
result = processor.process(date_range=1)

if not result.empty:
    print(f"\n计算完成，共{len(result)}条记录")

    # 显示PE统计
    if 'PE_TTM' in result.columns:
        pe_data = result['PE_TTM'].dropna()
        print(f"\nPE_TTM统计:")
        print(f"  数量: {len(pe_data)}")
        print(f"  最小值: {pe_data.min():.2f}")
        print(f"  25%分位: {pe_data.quantile(0.25):.2f}")
        print(f"  中位数: {pe_data.median():.2f}")
        print(f"  75%分位: {pe_data.quantile(0.75):.2f}")
        print(f"  最大值: {pe_data.max():.2f}")
        print(f"  平均值: {pe_data.mean():.2f}")

    # 显示PB统计
    if 'PB' in result.columns:
        pb_data = result['PB'].dropna()
        print(f"\nPB统计:")
        print(f"  数量: {len(pb_data)}")
        print(f"  最小值: {pb_data.min():.2f}")
        print(f"  25%分位: {pb_data.quantile(0.25):.2f}")
        print(f"  中位数: {pb_data.median():.2f}")
        print(f"  75%分位: {pb_data.quantile(0.75):.2f}")
        print(f"  最大值: {pb_data.max():.2f}")
        print(f"  平均值: {pb_data.mean():.2f}")

    # 显示各板块详细数据
    print("\n各板块估值详情:")
    print("-"*80)
    print(f"{'板块':<20} {'股票数':<8} {'市值(万亿)':<12} {'PE_TTM':<10} {'PB':<10}")
    print("-"*80)

    # 按市值排序
    result_sorted = result.sort_values('TotalMarketCap', ascending=False)

    for _, row in result_sorted.iterrows():
        sector = row['Sector'][:18]  # 截断过长的板块名
        stock_count = row['StockCount']
        market_cap = row['TotalMarketCap'] / 1e12  # 转换为万亿

        pe = row.get('PE_TTM', np.nan)
        pe_str = f"{pe:.2f}" if pd.notna(pe) else "N/A"

        pb = row.get('PB', np.nan)
        pb_str = f"{pb:.2f}" if pd.notna(pb) else "N/A"

        print(f"{sector:<20} {stock_count:<8} {market_cap:<12.2f} {pe_str:<10} {pb_str:<10}")

    print("-"*80)

    # 检查是否有异常值
    print("\n数据质量检查:")
    if 'PE_TTM' in result.columns:
        abnormal_pe = result[(result['PE_TTM'] < 5) | (result['PE_TTM'] > 100)]
        if not abnormal_pe.empty:
            print(f"  发现{len(abnormal_pe)}个板块PE异常（<5或>100）")
            for _, row in abnormal_pe.iterrows():
                print(f"    {row['Sector']}: PE={row['PE_TTM']:.2f}")
        else:
            print("  所有板块PE在合理范围内（5-100）")
else:
    print("计算结果为空！")