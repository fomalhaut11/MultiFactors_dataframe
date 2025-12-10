"""
测试小规模板块估值计算（只计算最近10天）
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from data.processor.sector_valuation_processor import SectorValuationProcessor

logging.basicConfig(level=logging.INFO, format='%(message)s')

class SmallTestProcessor(SectorValuationProcessor):
    """修改后只处理少量数据的测试处理器"""

    def calculate_sector_valuation(self, daily_financials):
        """重写方法，只计算最近10天"""
        self.logger.info("计算行业板块估值指标（测试模式：最近10天）...")

        results = []

        # 只取最近10天
        latest_dates = sorted(pd.Series(self.trading_dates).unique())[-10:]
        self.logger.info(f"测试日期: {latest_dates[0]} 到 {latest_dates[-1]}")

        for i, date in enumerate(latest_dates):
            self.logger.info(f"处理日期 {i+1}/10: {date}")

            if date in self.classification.index.get_level_values('TradingDates'):
                daily_classification = self.classification.xs(date, level='TradingDates')

                # 只处理前5个行业（测试用）
                sectors = self.classification.columns.tolist()[:5]

                for sector in sectors:
                    sector_stocks = daily_classification[daily_classification[sector] == 1].index.tolist()

                    if len(sector_stocks) == 0:
                        continue

                    sector_result = {
                        'TradingDate': date,
                        'Sector': sector,
                        'StockCount': len(sector_stocks)
                    }

                    try:
                        # 计算总市值
                        sector_market_caps = []
                        for stock in sector_stocks[:50]:  # 每个行业最多50只股票
                            if (date, stock) in self.market_cap.index:
                                mc = self.market_cap.loc[(date, stock)]
                                if pd.notna(mc) and mc > 0:
                                    sector_market_caps.append(mc)

                        if not sector_market_caps:
                            continue

                        total_market_cap = sum(sector_market_caps)
                        sector_result['TotalMarketCap'] = total_market_cap

                        # 计算PE_TTM
                        if 'net_profit_ttm' in daily_financials:
                            sector_profits = []
                            for stock in sector_stocks[:50]:
                                if (date, stock) in daily_financials['net_profit_ttm'].index:
                                    profit = daily_financials['net_profit_ttm'].loc[(date, stock)]
                                    if pd.notna(profit):
                                        sector_profits.append(profit)

                            if sector_profits:
                                total_profit = sum(sector_profits)
                                if total_profit > 0:
                                    sector_result['PE_TTM'] = total_market_cap / total_profit

                        # 计算PB
                        if 'total_equity' in daily_financials:
                            sector_equities = []
                            for stock in sector_stocks[:50]:
                                if (date, stock) in daily_financials['total_equity'].index:
                                    equity = daily_financials['total_equity'].loc[(date, stock)]
                                    if pd.notna(equity) and equity > 0:
                                        sector_equities.append(equity)

                            if sector_equities:
                                total_equity = sum(sector_equities)
                                if total_equity > 0:
                                    sector_result['PB'] = total_market_cap / total_equity

                        results.append(sector_result)

                    except Exception as e:
                        self.logger.debug(f"错误: {e}")
                        continue

        results_df = pd.DataFrame(results)
        self.logger.info(f"测试完成，生成{len(results_df)}条记录")

        return results_df

def test():
    print("=" * 60)
    print("小规模测试板块估值计算")
    print("=" * 60)

    processor = SmallTestProcessor()

    try:
        result = processor.process()

        print(f"\n✅ 计算完成!")
        print(f"数据形状: {result.shape}")
        print(f"列名: {list(result.columns)}")

        # 检查估值指标
        if 'PE_TTM' in result.columns:
            pe_data = result[result['PE_TTM'].notna()]
            if not pe_data.empty:
                print(f"\n✅ 成功计算PE_TTM! 有效记录: {len(pe_data)}")
                print(f"PE_TTM范围: {pe_data['PE_TTM'].min():.2f} - {pe_data['PE_TTM'].max():.2f}")

        if 'PB' in result.columns:
            pb_data = result[result['PB'].notna()]
            if not pb_data.empty:
                print(f"\n✅ 成功计算PB! 有效记录: {len(pb_data)}")
                print(f"PB范围: {pb_data['PB'].min():.2f} - {pb_data['PB'].max():.2f}")

        # 显示样本
        if not result.empty:
            print("\n样本数据:")
            print(result.head(10).to_string())

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test()