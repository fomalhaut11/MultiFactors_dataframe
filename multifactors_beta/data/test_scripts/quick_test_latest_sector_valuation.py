"""
快速测试：只计算最新一天的板块估值
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

def quick_test():
    print("=" * 60)
    print("快速测试：计算最新一天的板块估值")
    print("=" * 60)

    start_time = time.time()

    # 数据路径
    data_root = Path("E:/Documents/PythonProject/StockProject/StockData")

    # 1. 加载数据
    print("\n1. 加载数据...")

    # 价格数据（包含市值）
    price_data = pd.read_pickle(data_root / "Price.pkl")
    market_cap = price_data['MC']  # 市值，单位：元
    print(f"   市值数据: {market_cap.shape}")

    # PE数据
    pe_ttm = pd.read_pickle(data_root / "RawFactors" / "EP_ttm.pkl")
    print(f"   EP_ttm数据: {pe_ttm.shape}")

    # 行业分类
    classification = pd.read_pickle(data_root / "Classificationdata" / "classification_one_hot.pkl")
    print(f"   行业分类: {classification.shape}")

    # 2. 获取最新日期
    latest_date = sorted(classification.index.get_level_values('TradingDates').unique())[-1]
    print(f"\n2. 测试日期: {latest_date}")

    # 3. 获取当日行业分类
    daily_classification = classification.xs(latest_date, level='TradingDates')
    sectors = classification.columns.tolist()[:10]  # 只测试前10个行业

    print(f"\n3. 开始计算板块估值...")
    results = []

    for sector in sectors:
        sector_name = sector.replace('concept_name_', '').replace('(申万)', '').replace('(退市)', '')

        # 找出该行业的股票
        sector_stocks = daily_classification[daily_classification[sector] == 1].index.tolist()

        if len(sector_stocks) == 0:
            continue

        # 计算总市值
        total_mc = 0
        valid_mc_count = 0
        for stock in sector_stocks:
            if (latest_date, stock) in market_cap.index:
                mc = market_cap.loc[(latest_date, stock)]
                if pd.notna(mc) and mc > 0:
                    total_mc += mc
                    valid_mc_count += 1

        if total_mc == 0:
            continue

        # 计算总净利润（通过EP反推）
        total_profit = 0
        valid_pe_count = 0
        for stock in sector_stocks:
            if (latest_date, stock) in market_cap.index and (latest_date, stock) in pe_ttm.index:
                mc = market_cap.loc[(latest_date, stock)]
                ep = pe_ttm.loc[(latest_date, stock)]

                if pd.notna(mc) and pd.notna(ep) and mc > 0 and ep > 0:
                    # 净利润 = 市值 * EP
                    profit = mc * ep
                    total_profit += profit
                    valid_pe_count += 1

        # 计算PE
        if total_profit > 0:
            pe_ttm_val = total_mc / total_profit
        else:
            pe_ttm_val = None

        results.append({
            'Sector': sector_name,
            'StockCount': len(sector_stocks),
            'ValidMC': valid_mc_count,
            'ValidPE': valid_pe_count,
            'TotalMarketCap_亿': total_mc / 1e8,
            'TotalProfit_亿': total_profit / 1e8 if total_profit > 0 else None,
            'PE_TTM': pe_ttm_val
        })

        print(f"   {sector_name:10} - 股票{len(sector_stocks):3}只, "
              f"市值{total_mc/1e10:.1f}万亿, "
              f"PE: {pe_ttm_val:.1f}" if pe_ttm_val else f"PE: N/A")

    # 4. 显示结果
    result_df = pd.DataFrame(results)

    print(f"\n4. 计算完成!")
    print(f"   有效记录: {len(result_df)}条")

    if 'PE_TTM' in result_df.columns:
        pe_valid = result_df['PE_TTM'].notna()
        if pe_valid.any():
            print(f"   PE_TTM范围: {result_df.loc[pe_valid, 'PE_TTM'].min():.1f} - {result_df.loc[pe_valid, 'PE_TTM'].max():.1f}")
            print(f"   PE_TTM中位数: {result_df.loc[pe_valid, 'PE_TTM'].median():.1f}")

    # 显示完整结果表
    print("\n" + "=" * 60)
    print("板块估值结果：")
    print(result_df.to_string(index=False))

    elapsed = time.time() - start_time
    print(f"\n总耗时: {elapsed:.2f}秒")

    # 保存结果
    output_path = data_root / "SectorData" / "quick_test_sector_valuation.csv"
    result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"结果已保存至: {output_path}")

if __name__ == "__main__":
    quick_test()