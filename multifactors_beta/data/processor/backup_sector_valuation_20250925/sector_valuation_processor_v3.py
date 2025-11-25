"""
行业板块估值指标处理器 V3版本
使用已计算好的个股PE_ttm反推净利润，避免TTM计算和日频扩展

计算逻辑：
1. 个股净利润_TTM = 个股市值 / 个股PE_TTM
2. 板块总净利润_TTM = Σ(个股净利润_TTM)
3. 板块总市值 = Σ(个股市值)
4. 板块PE_TTM = 板块总市值 / 板块总净利润_TTM
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List
import logging
from datetime import datetime
import time

class SectorValuationProcessorV3:
    """
    行业板块估值指标计算器 V3
    使用个股PE反推净利润的高效版本
    """

    def __init__(self):
        """初始化处理器"""
        self.logger = logging.getLogger(__name__)

        # 数据路径配置
        self.data_root = Path("E:/Documents/PythonProject/StockProject/StockData")
        self.factors_path = self.data_root / "RawFactors"

        # 创建输出目录
        self.sector_data_path = self.data_root / "SectorData"
        self.sector_data_path.mkdir(exist_ok=True)
        self.logger.info(f"板块数据将存储在: {self.sector_data_path}")

    def load_data(self) -> Dict:
        """加载所需数据"""
        self.logger.info("加载数据...")

        data = {}

        # 1. 加载价格数据（包含市值）
        price_path = self.data_root / "Price.pkl"
        if price_path.exists():
            data['price'] = pd.read_pickle(price_path)
            self.logger.info(f"价格数据形状: {data['price'].shape}")

            # 提取市值数据（单位：元）
            if 'MC' in data['price'].columns:
                data['market_cap'] = data['price']['MC']
                self.logger.info(f"市值数据形状: {data['market_cap'].shape}")

        # 2. 加载个股PE_ttm数据
        pe_ttm_path = self.factors_path / "EP_ttm.pkl"  # 或者 PE_ttm.pkl
        if not pe_ttm_path.exists():
            pe_ttm_path = self.factors_path / "PE_ttm.pkl"

        if pe_ttm_path.exists():
            data['pe_ttm'] = pd.read_pickle(pe_ttm_path)
            self.logger.info(f"PE_ttm数据形状: {data['pe_ttm'].shape}")
        else:
            self.logger.warning(f"未找到PE_ttm数据，尝试查找EP_ttm")
            ep_ttm_path = self.factors_path / "EP_ttm.pkl"
            if ep_ttm_path.exists():
                ep_ttm = pd.read_pickle(ep_ttm_path)
                # EP转PE：PE = 1/EP
                data['pe_ttm'] = 1 / ep_ttm
                data['pe_ttm'] = data['pe_ttm'].replace([np.inf, -np.inf], np.nan)
                self.logger.info(f"从EP_ttm转换得到PE_ttm，形状: {data['pe_ttm'].shape}")

        # 3. 加载个股PB数据（如果存在）
        pb_path = self.factors_path / "BP.pkl"  # 或者 PB.pkl
        if not pb_path.exists():
            pb_path = self.factors_path / "PB.pkl"

        if pb_path.exists():
            data['pb'] = pd.read_pickle(pb_path)
            self.logger.info(f"PB数据形状: {data['pb'].shape}")
        else:
            bp_path = self.factors_path / "BP.pkl"
            if bp_path.exists():
                bp = pd.read_pickle(bp_path)
                # BP转PB：PB = 1/BP
                data['pb'] = 1 / bp
                data['pb'] = data['pb'].replace([np.inf, -np.inf], np.nan)
                self.logger.info(f"从BP转换得到PB，形状: {data['pb'].shape}")

        # 4. 加载行业分类数据
        classification_path = self.data_root / "Classificationdata" / "classification_one_hot.pkl"
        data['classification'] = pd.read_pickle(classification_path)
        self.logger.info(f"行业分类数据形状: {data['classification'].shape}")

        return data

    def calculate_sector_valuation(self, data: Dict, date_range: int = 252) -> pd.DataFrame:
        """
        计算板块估值指标

        Args:
            data: 数据字典
            date_range: 计算的日期范围（默认252个交易日）

        Returns:
            板块估值DataFrame
        """
        self.logger.info(f"计算板块估值（最近{date_range}个交易日）...")

        market_cap = data['market_cap']
        pe_ttm = data.get('pe_ttm')
        pb = data.get('pb')
        classification = data['classification']

        # 获取交易日期
        trading_dates = sorted(classification.index.get_level_values('TradingDates').unique())
        latest_dates = trading_dates[-date_range:] if len(trading_dates) > date_range else trading_dates

        self.logger.info(f"日期范围: {latest_dates[0]} 到 {latest_dates[-1]}")

        results = []

        for i, date in enumerate(latest_dates):
            if i % 50 == 0:
                self.logger.info(f"处理进度: {i}/{len(latest_dates)}")

            # 获取当日的行业分类
            if date not in classification.index.get_level_values('TradingDates'):
                continue

            daily_classification = classification.xs(date, level='TradingDates')
            sectors = classification.columns.tolist()

            for sector in sectors:
                # 找出该行业的股票
                sector_mask = daily_classification[sector] == 1
                sector_stocks = daily_classification[sector_mask].index.tolist()

                if len(sector_stocks) == 0:
                    continue

                sector_result = {
                    'TradingDate': date,
                    'Sector': sector,
                    'StockCount': len(sector_stocks),
                }

                try:
                    # 计算板块总市值
                    sector_market_caps = []
                    for stock in sector_stocks:
                        if (date, stock) in market_cap.index:
                            mc = market_cap.loc[(date, stock)]
                            if pd.notna(mc) and mc > 0:
                                sector_market_caps.append(mc)

                    if not sector_market_caps:
                        continue

                    total_market_cap = sum(sector_market_caps)
                    sector_result['TotalMarketCap'] = total_market_cap
                    sector_result['AvgMarketCap'] = total_market_cap / len(sector_market_caps)
                    sector_result['ValidStocks'] = len(sector_market_caps)

                    # 计算板块PE_TTM
                    if pe_ttm is not None:
                        sector_profits = []
                        pe_count = 0

                        for stock in sector_stocks:
                            if (date, stock) in market_cap.index and (date, stock) in pe_ttm.index:
                                mc = market_cap.loc[(date, stock)]
                                pe = pe_ttm.loc[(date, stock)]

                                if pd.notna(mc) and pd.notna(pe) and mc > 0 and pe > 0:
                                    # 个股净利润 = 市值 / PE
                                    profit = mc / pe
                                    sector_profits.append(profit)
                                    pe_count += 1

                        if sector_profits:
                            total_profit = sum(sector_profits)
                            if total_profit > 0:
                                sector_result['PE_TTM'] = total_market_cap / total_profit
                                sector_result['TotalProfit_TTM'] = total_profit
                                sector_result['PE_StockCount'] = pe_count

                    # 计算板块PB
                    if pb is not None:
                        sector_equities = []
                        pb_count = 0

                        for stock in sector_stocks:
                            if (date, stock) in market_cap.index and (date, stock) in pb.index:
                                mc = market_cap.loc[(date, stock)]
                                pb_val = pb.loc[(date, stock)]

                                if pd.notna(mc) and pd.notna(pb_val) and mc > 0 and pb_val > 0:
                                    # 个股净资产 = 市值 / PB
                                    equity = mc / pb_val
                                    sector_equities.append(equity)
                                    pb_count += 1

                        if sector_equities:
                            total_equity = sum(sector_equities)
                            if total_equity > 0:
                                sector_result['PB'] = total_market_cap / total_equity
                                sector_result['TotalEquity'] = total_equity
                                sector_result['PB_StockCount'] = pb_count

                    results.append(sector_result)

                except Exception as e:
                    self.logger.debug(f"计算{sector}在{date}的估值失败: {e}")
                    continue

        results_df = pd.DataFrame(results)
        self.logger.info(f"板块估值计算完成，共{len(results_df)}条记录")

        # 添加统计信息
        if not results_df.empty:
            self.logger.info(f"日期数量: {results_df['TradingDate'].nunique()}")
            self.logger.info(f"行业数量: {results_df['Sector'].nunique()}")

            if 'PE_TTM' in results_df.columns:
                pe_valid = results_df['PE_TTM'].notna().sum()
                self.logger.info(f"有效PE_TTM记录: {pe_valid}/{len(results_df)}")

            if 'PB' in results_df.columns:
                pb_valid = results_df['PB'].notna().sum()
                self.logger.info(f"有效PB记录: {pb_valid}/{len(results_df)}")

        return results_df

    def process(self, date_range: int = 252) -> pd.DataFrame:
        """
        主处理函数

        Args:
            date_range: 计算的日期范围（默认252个交易日）

        Returns:
            板块估值DataFrame
        """
        start_time = time.time()
        self.logger.info("=" * 60)
        self.logger.info("开始计算板块估值（V3版本 - 使用个股PE反推）")
        self.logger.info("=" * 60)

        # 1. 加载数据
        data = self.load_data()

        # 2. 计算板块估值
        sector_valuation = self.calculate_sector_valuation(data, date_range)

        # 3. 保存结果
        if not sector_valuation.empty:
            # 保存pkl格式
            output_path = self.sector_data_path / "SectorValuation_v3.pkl"
            sector_valuation.to_pickle(output_path)
            self.logger.info(f"板块估值数据已保存至: {output_path}")

            # 保存CSV格式
            csv_path = self.sector_data_path / "SectorValuation_v3.csv"
            sector_valuation.to_csv(csv_path, index=False, encoding='utf-8-sig')
            self.logger.info(f"板块估值CSV已保存至: {csv_path}")

            # 生成汇总报告
            self.generate_summary_report(sector_valuation)

        elapsed = time.time() - start_time
        self.logger.info(f"处理完成，总耗时: {elapsed:.2f}秒")

        return sector_valuation

    def generate_summary_report(self, sector_valuation: pd.DataFrame):
        """生成汇总报告"""

        # 获取最新日期的数据
        latest_date = sector_valuation['TradingDate'].max()
        latest_data = sector_valuation[sector_valuation['TradingDate'] == latest_date]

        if latest_data.empty:
            return

        summary = {
            'report_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'latest_date': str(latest_date),
            'date_range': f"{sector_valuation['TradingDate'].min()} to {latest_date}",
            'total_records': len(sector_valuation),
            'sectors': sector_valuation['Sector'].nunique(),
            'dates': sector_valuation['TradingDate'].nunique(),
        }

        # PE统计
        if 'PE_TTM' in latest_data.columns:
            pe_data = latest_data['PE_TTM'].dropna()
            if len(pe_data) > 0:
                summary['pe_stats'] = {
                    'count': len(pe_data),
                    'mean': float(pe_data.mean()),
                    'median': float(pe_data.median()),
                    'min': float(pe_data.min()),
                    'max': float(pe_data.max()),
                }

        # PB统计
        if 'PB' in latest_data.columns:
            pb_data = latest_data['PB'].dropna()
            if len(pb_data) > 0:
                summary['pb_stats'] = {
                    'count': len(pb_data),
                    'mean': float(pb_data.mean()),
                    'median': float(pb_data.median()),
                    'min': float(pb_data.min()),
                    'max': float(pb_data.max()),
                }

        # 保存汇总报告
        import json
        summary_path = self.sector_data_path / "sector_valuation_summary_v3.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        self.logger.info(f"汇总报告已保存至: {summary_path}")

        # 打印关键信息
        self.logger.info("=" * 60)
        self.logger.info("板块估值汇总（最新日期）")
        self.logger.info("=" * 60)
        if 'pe_stats' in summary:
            self.logger.info(f"PE_TTM - 均值: {summary['pe_stats']['mean']:.2f}, "
                           f"中位数: {summary['pe_stats']['median']:.2f}, "
                           f"范围: {summary['pe_stats']['min']:.2f}-{summary['pe_stats']['max']:.2f}")
        if 'pb_stats' in summary:
            self.logger.info(f"PB - 均值: {summary['pb_stats']['mean']:.2f}, "
                           f"中位数: {summary['pb_stats']['median']:.2f}, "
                           f"范围: {summary['pb_stats']['min']:.2f}-{summary['pb_stats']['max']:.2f}")