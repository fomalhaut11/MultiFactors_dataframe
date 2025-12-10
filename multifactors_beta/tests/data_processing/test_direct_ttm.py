"""
直接使用TTM数据计算板块估值（不扩展到日频）
简化版本 - 只计算最新一期的板块估值
"""

import sys
import pandas as pd
from pathlib import Path
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from factors.generators import calculate_ttm

logging.basicConfig(level=logging.INFO, format='%(message)s')

def test_direct():
    print("=" * 60)
    print("直接使用TTM数据计算板块估值")
    print("=" * 60)

    # 路径配置
    data_root = Path("E:/Documents/PythonProject/StockProject/StockData")
    auxiliary_path = Path(__file__).parent.parent / "auxiliary"
    sector_data_path = data_root / "SectorData"
    sector_data_path.mkdir(exist_ok=True)

    # 1. 加载数据
    print("\n加载数据...")
    financial_data = pd.read_pickle(auxiliary_path / "FinancialData_unified.pkl")
    market_cap = pd.read_pickle(auxiliary_path / "MarketCap.pkl")
    classification = pd.read_pickle(data_root / "Classificationdata" / "classification_one_hot.pkl")

    print(f"财务数据: {financial_data.shape}")
    print(f"市值数据: {market_cap.shape}")
    print(f"行业分类: {classification.shape}")

    # 2. 计算TTM净利润
    print("\n计算TTM净利润...")
    time_fields = ['d_quarter', 'd_year']
    net_profit_df = financial_data[['NET_PROFIT_IS'] + time_fields].copy()
    ttm_result = calculate_ttm(net_profit_df)
    net_profit_ttm = ttm_result['NET_PROFIT_IS_ttm']
    print(f"净利润TTM: {net_profit_ttm.shape}")

    # 获取股东权益（最新值）
    total_equity = financial_data['TOT_EQUITY']
    print(f"股东权益: {total_equity.shape}")

    # 3. 找一个最新的日期进行测试
    latest_trading_dates = sorted(classification.index.get_level_values('TradingDates').unique())[-5:]
    test_date = latest_trading_dates[-1]
    print(f"\n测试日期: {test_date}")

    # 4. 获取该日期的行业分类
    daily_classification = classification.xs(test_date, level='TradingDates')
    sectors = classification.columns.tolist()[:10]  # 只测试前10个行业

    # 5. 计算板块估值
    results = []
    for sector in sectors:
        sector_name = sector.replace('concept_name_', '').replace('(申万)', '')
        print(f"\n处理行业: {sector_name}")

        # 找出该行业的股票
        sector_stocks = daily_classification[daily_classification[sector] == 1].index.tolist()
        print(f"  成分股数量: {len(sector_stocks)}")

        if len(sector_stocks) == 0:
            continue

        # 计算总市值
        total_mc = 0
        valid_stocks = []
        for stock in sector_stocks:
            if (test_date, stock) in market_cap.index:
                mc = market_cap.loc[(test_date, stock)]
                if pd.notna(mc) and mc > 0:
                    total_mc += mc
                    valid_stocks.append(stock)

        if total_mc == 0:
            continue

        print(f"  有效股票: {len(valid_stocks)}")
        print(f"  总市值: {total_mc/1e8:.2f}亿")

        # 计算总净利润（使用最新的财报数据）
        total_profit = 0
        profit_count = 0
        for stock in valid_stocks:
            # 查找该股票最新的TTM净利润
            if stock in net_profit_ttm.index.get_level_values('StockCodes'):
                stock_ttm = net_profit_ttm.xs(stock, level='StockCodes')
                if not stock_ttm.empty:
                    latest_profit = stock_ttm.iloc[-1]  # 最新一期
                    if pd.notna(latest_profit):
                        total_profit += latest_profit
                        profit_count += 1

        if total_profit > 0:
            pe_ttm = total_mc / total_profit
            print(f"  总净利润TTM: {total_profit/1e8:.2f}亿")
            print(f"  PE_TTM: {pe_ttm:.2f}")
        else:
            pe_ttm = None
            print(f"  净利润数据不足")

        # 计算PB
        total_eq = 0
        equity_count = 0
        for stock in valid_stocks:
            if stock in total_equity.index.get_level_values('StockCodes'):
                stock_eq = total_equity.xs(stock, level='StockCodes')
                if not stock_eq.empty:
                    latest_eq = stock_eq.iloc[-1]
                    if pd.notna(latest_eq) and latest_eq > 0:
                        total_eq += latest_eq
                        equity_count += 1

        if total_eq > 0:
            pb = total_mc / total_eq
            print(f"  总净资产: {total_eq/1e8:.2f}亿")
            print(f"  PB: {pb:.2f}")
        else:
            pb = None

        results.append({
            'Sector': sector_name,
            'StockCount': len(sector_stocks),
            'ValidStocks': len(valid_stocks),
            'TotalMarketCap': total_mc / 1e8,
            'PE_TTM': pe_ttm,
            'PB': pb
        })

    # 6. 显示结果
    result_df = pd.DataFrame(results)
    print("\n" + "=" * 60)
    print("计算结果:")
    print(result_df.to_string())

    # 保存结果
    output_file = sector_data_path / "sector_valuation_simple.csv"
    result_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n结果已保存至: {output_file}")

if __name__ == "__main__":
    test_direct()