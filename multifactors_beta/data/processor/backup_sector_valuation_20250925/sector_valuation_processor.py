"""
行业板块估值指标处理器
计算各行业板块的整体估值指标：TTM PE、PB等

板块估值计算逻辑：
- PE_TTM = 板块总市值 / 板块总净利润(TTM)
- PB = 板块总市值 / 板块总净资产
- PS_TTM = 板块总市值 / 板块总营收(TTM)
- PCF_TTM = 板块总市值 / 板块总经营现金流(TTM)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Union, List
import warnings
import sys
import logging
from datetime import datetime

# 添加上级目录到路径以便导入factors模块
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 导入因子计算工具
from factors.generators import (
    calculate_ttm,
    expand_to_daily_vectorized,
    FinancialReportProcessor
)

from data.processor.base_processor import BaseDataProcessor

class SectorValuationProcessor(BaseDataProcessor):
    """
    行业板块估值指标计算器

    计算每个行业板块的整体估值指标，使用板块内所有股票的总市值和总财务指标
    """

    def __init__(self):
        """初始化处理器"""
        super().__init__()
        self.logger = logging.getLogger(__name__)

        # 数据路径配置
        self.data_root = Path("E:/Documents/PythonProject/StockProject/StockData")
        self.auxiliary_path = Path(__file__).parent.parent / "auxiliary"

        # 创建行业数据存储目录
        self.sector_data_path = self.data_root / "SectorData"
        self.sector_data_path.mkdir(exist_ok=True)
        self.logger.info(f"行业数据将存储在: {self.sector_data_path}")

        # 财务字段映射
        self.field_mapping = {
            'net_profit': 'NET_PROFIT_IS',      # 净利润
            'total_equity': 'TOT_EQUITY',       # 股东权益(净资产)
            'total_assets': 'TOT_ASSETS',       # 总资产
            'revenue': 'TOT_OPER_REV',          # 营业总收入
            'operating_cash_flow': 'NET_CASH_FLOWS_OPER_ACT'  # 经营现金流净额
        }

    def validate_input(self, **kwargs):
        """验证输入参数"""
        return True

    def load_data(self):
        """加载所需数据"""
        self.logger.info("加载数据...")

        # 1. 加载财务数据
        self.financial_data = pd.read_pickle(self.auxiliary_path / "FinancialData_unified.pkl")
        self.logger.info(f"财务数据形状: {self.financial_data.shape}")

        # 2. 加载市值数据 - 这是关键数据！
        self.market_cap = pd.read_pickle(self.auxiliary_path / "MarketCap.pkl")
        self.logger.info(f"市值数据形状: {self.market_cap.shape}")

        # 3. 加载行业分类数据
        classification_path = self.data_root / "Classificationdata" / "classification_one_hot.pkl"
        self.classification = pd.read_pickle(classification_path)
        self.logger.info(f"行业分类数据形状: {self.classification.shape}")

        # 4. 加载交易日期
        self.trading_dates = pd.read_pickle(self.auxiliary_path / "TradingDates.pkl")
        if isinstance(self.trading_dates, pd.Series):
            self.trading_dates = self.trading_dates.values
        self.logger.info(f"交易日期数量: {len(self.trading_dates)}")

        # 5. 加载财报发布日期
        self.release_dates = pd.read_pickle(self.auxiliary_path / "ReleaseDates.pkl")
        self.logger.info(f"财报发布日期数据形状: {self.release_dates.shape}")

    def calculate_ttm_financials(self) -> Dict[str, pd.Series]:
        """计算TTM财务指标"""
        self.logger.info("计算TTM财务指标...")

        ttm_data = {}

        # 准备必要的时间字段
        time_fields = ['d_quarter', 'd_year'] if 'd_quarter' in self.financial_data.columns else []

        # 计算净利润TTM
        if self.field_mapping['net_profit'] in self.financial_data.columns:
            # calculate_ttm需要包含d_quarter等字段的DataFrame
            fields_to_include = [self.field_mapping['net_profit']] + time_fields
            net_profit_df = self.financial_data[fields_to_include].copy()
            ttm_result = calculate_ttm(net_profit_df)
            # 提取结果Series - 注意：字段名会被加上_ttm后缀
            ttm_field_name = f"{self.field_mapping['net_profit']}_ttm"
            ttm_data['net_profit_ttm'] = ttm_result[ttm_field_name]
            self.logger.info(f"净利润TTM计算完成: {ttm_data['net_profit_ttm'].shape}")

        # 计算营收TTM
        if self.field_mapping['revenue'] in self.financial_data.columns:
            fields_to_include = [self.field_mapping['revenue']] + time_fields
            revenue_df = self.financial_data[fields_to_include].copy()
            ttm_result = calculate_ttm(revenue_df)
            ttm_field_name = f"{self.field_mapping['revenue']}_ttm"
            ttm_data['revenue_ttm'] = ttm_result[ttm_field_name]
            self.logger.info(f"营收TTM计算完成: {ttm_data['revenue_ttm'].shape}")

        # 计算经营现金流TTM
        if self.field_mapping['operating_cash_flow'] in self.financial_data.columns:
            fields_to_include = [self.field_mapping['operating_cash_flow']] + time_fields
            ocf_df = self.financial_data[fields_to_include].copy()
            ttm_result = calculate_ttm(ocf_df)
            ttm_field_name = f"{self.field_mapping['operating_cash_flow']}_ttm"
            ttm_data['ocf_ttm'] = ttm_result[ttm_field_name]
            self.logger.info(f"经营现金流TTM计算完成: {ttm_data['ocf_ttm'].shape}")

        # 获取最新的股东权益（用于PB计算，不需要TTM）
        if self.field_mapping['total_equity'] in self.financial_data.columns:
            ttm_data['total_equity'] = self.financial_data[self.field_mapping['total_equity']]
            self.logger.info(f"股东权益数据获取完成: {ttm_data['total_equity'].shape}")

        return ttm_data

    def expand_financial_to_daily(self, ttm_data: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """将财务数据扩展到日频"""
        self.logger.info("将财务数据扩展到日频...")

        daily_data = {}

        for key, data in ttm_data.items():
            try:
                # expand_to_daily_vectorized需要DataFrame输入，将Series转换为DataFrame
                if isinstance(data, pd.Series):
                    data_df = pd.DataFrame({key: data})
                else:
                    data_df = data

                # 使用expand_to_daily_vectorized函数扩展到日频
                result_df = expand_to_daily_vectorized(
                    factor_data=data_df,
                    release_dates=self.release_dates,
                    trading_dates=self.trading_dates
                )

                # 提取结果Series
                daily_data[key] = result_df[key] if isinstance(result_df, pd.DataFrame) else result_df
                self.logger.info(f"{key}扩展到日频完成: {daily_data[key].shape}")
            except Exception as e:
                self.logger.error(f"扩展{key}到日频失败: {e}")

        return daily_data

    def calculate_sector_valuation_for_date(
        self,
        date: pd.Timestamp,
        daily_classification: pd.DataFrame,
        daily_financials: Dict[str, pd.Series]
    ) -> List[Dict]:
        """计算特定日期的行业板块估值"""

        results = []
        sectors = self.classification.columns.tolist()

        for sector in sectors:
            # 获取该行业的股票列表
            sector_stocks = daily_classification[daily_classification[sector] == 1].index.tolist()

            if len(sector_stocks) == 0:
                continue

            sector_result = {
                'TradingDate': date,
                'Sector': sector,
                'StockCount': len(sector_stocks)
            }

            try:
                # 计算板块总市值（核心！）
                sector_market_caps = []
                for stock in sector_stocks:
                    if (date, stock) in self.market_cap.index:
                        mc = self.market_cap.loc[(date, stock)]
                        if pd.notna(mc) and mc > 0:
                            sector_market_caps.append(mc)

                if not sector_market_caps:
                    continue

                total_market_cap = sum(sector_market_caps)
                sector_result['TotalMarketCap'] = total_market_cap
                sector_result['AvgMarketCap'] = total_market_cap / len(sector_market_caps)

                # 计算PE_TTM = 总市值 / 总净利润
                if 'net_profit_ttm' in daily_financials:
                    sector_profits = []
                    for stock in sector_stocks:
                        if (date, stock) in daily_financials['net_profit_ttm'].index:
                            profit = daily_financials['net_profit_ttm'].loc[(date, stock)]
                            if pd.notna(profit):
                                sector_profits.append(profit)

                    if sector_profits:
                        total_profit = sum(sector_profits)
                        # 只有总净利润为正时才计算PE
                        if total_profit > 0:
                            sector_result['PE_TTM'] = total_market_cap / total_profit
                            sector_result['TotalProfit_TTM'] = total_profit

                # 计算PB = 总市值 / 总净资产
                if 'total_equity' in daily_financials:
                    sector_equities = []
                    for stock in sector_stocks:
                        if (date, stock) in daily_financials['total_equity'].index:
                            equity = daily_financials['total_equity'].loc[(date, stock)]
                            if pd.notna(equity) and equity > 0:
                                sector_equities.append(equity)

                    if sector_equities:
                        total_equity = sum(sector_equities)
                        if total_equity > 0:
                            sector_result['PB'] = total_market_cap / total_equity
                            sector_result['TotalEquity'] = total_equity

                # 计算PS_TTM = 总市值 / 总营收
                if 'revenue_ttm' in daily_financials:
                    sector_revenues = []
                    for stock in sector_stocks:
                        if (date, stock) in daily_financials['revenue_ttm'].index:
                            revenue = daily_financials['revenue_ttm'].loc[(date, stock)]
                            if pd.notna(revenue) and revenue > 0:
                                sector_revenues.append(revenue)

                    if sector_revenues:
                        total_revenue = sum(sector_revenues)
                        if total_revenue > 0:
                            sector_result['PS_TTM'] = total_market_cap / total_revenue
                            sector_result['TotalRevenue_TTM'] = total_revenue

                # 计算PCF_TTM = 总市值 / 总经营现金流
                if 'ocf_ttm' in daily_financials:
                    sector_ocfs = []
                    for stock in sector_stocks:
                        if (date, stock) in daily_financials['ocf_ttm'].index:
                            ocf = daily_financials['ocf_ttm'].loc[(date, stock)]
                            # 经营现金流可能为负，这里只过滤空值
                            if pd.notna(ocf):
                                sector_ocfs.append(ocf)

                    if sector_ocfs:
                        total_ocf = sum(sector_ocfs)
                        # 只有总经营现金流为正时才计算PCF
                        if total_ocf > 0:
                            sector_result['PCF_TTM'] = total_market_cap / total_ocf
                            sector_result['TotalOCF_TTM'] = total_ocf

                results.append(sector_result)

            except Exception as e:
                self.logger.debug(f"计算{sector}在{date}的估值失败: {e}")
                continue

        return results

    def calculate_sector_valuation(self, daily_financials: Dict[str, pd.Series]) -> pd.DataFrame:
        """计算各行业板块的估值指标"""
        self.logger.info("计算行业板块估值指标...")

        results = []

        # 获取最近一年的交易日
        latest_dates = sorted(pd.Series(self.trading_dates).unique())[-252:]

        for i, date in enumerate(latest_dates):
            if i % 50 == 0:
                self.logger.info(f"处理进度: {i}/{len(latest_dates)}")

            # 获取当日的行业分类
            if date in self.classification.index.get_level_values('TradingDates'):
                daily_classification = self.classification.xs(date, level='TradingDates')

                # 计算该日期的板块估值
                daily_results = self.calculate_sector_valuation_for_date(
                    date, daily_classification, daily_financials
                )
                results.extend(daily_results)

        # 转换为DataFrame
        results_df = pd.DataFrame(results)
        self.logger.info(f"板块估值计算完成，共{len(results_df)}条记录")

        # 添加一些统计信息
        if not results_df.empty:
            self.logger.info(f"日期范围: {results_df['TradingDate'].min()} 到 {results_df['TradingDate'].max()}")
            self.logger.info(f"行业数量: {results_df['Sector'].nunique()}")

        return results_df

    def process(self, **kwargs):
        """
        主处理函数

        Returns:
            pd.DataFrame: 行业板块估值指标数据
        """
        self.logger.info("开始计算行业板块估值指标...")

        # 1. 加载数据
        self.load_data()

        # 2. 计算TTM财务指标
        ttm_data = self.calculate_ttm_financials()

        # 3. 将财务数据扩展到日频
        daily_financials = self.expand_financial_to_daily(ttm_data)

        # 4. 计算行业板块估值
        sector_valuation = self.calculate_sector_valuation(daily_financials)

        # 5. 保存结果到标准数据目录
        output_path = self.sector_data_path / "SectorValuation.pkl"
        sector_valuation.to_pickle(output_path)
        self.logger.info(f"板块估值数据已保存至: {output_path}")

        # 同时保存CSV格式便于查看
        csv_path = self.sector_data_path / "SectorValuation.csv"
        sector_valuation.to_csv(csv_path, index=False, encoding='utf-8-sig')
        self.logger.info(f"板块估值CSV已保存至: {csv_path}")

        # 记录处理历史
        self._record_processing(
            operation="sector_valuation",
            params=kwargs,
            result_info={
                "shape": sector_valuation.shape,
                "sectors": sector_valuation['Sector'].nunique() if not sector_valuation.empty else 0,
                "dates": sector_valuation['TradingDate'].nunique() if not sector_valuation.empty else 0,
                "metrics": [col for col in sector_valuation.columns if col.endswith('_TTM') or col in ['PB']]
            }
        )

        return sector_valuation

    def get_latest_valuation(self, sector: Optional[str] = None) -> pd.DataFrame:
        """
        获取最新的板块估值数据

        Args:
            sector: 指定行业名称，None则返回所有行业

        Returns:
            pd.DataFrame: 最新估值数据
        """
        valuation_path = self.sector_data_path / "SectorValuation.pkl"

        if not valuation_path.exists():
            self.logger.warning("板块估值数据不存在，请先运行process()方法")
            return pd.DataFrame()

        valuation = pd.read_pickle(valuation_path)

        # 获取最新日期的数据
        latest_date = valuation['TradingDate'].max()
        latest_valuation = valuation[valuation['TradingDate'] == latest_date]

        if sector:
            latest_valuation = latest_valuation[latest_valuation['Sector'] == sector]

        # 按总市值排序
        return latest_valuation.sort_values('TotalMarketCap', ascending=False)

    def get_sector_history(self, sector: str, metric: str = 'PE_TTM') -> pd.DataFrame:
        """
        获取特定板块的历史估值数据

        Args:
            sector: 行业名称
            metric: 估值指标名称

        Returns:
            pd.DataFrame: 历史估值序列
        """
        valuation_path = self.sector_data_path / "SectorValuation.pkl"

        if not valuation_path.exists():
            self.logger.warning("板块估值数据不存在")
            return pd.DataFrame()

        valuation = pd.read_pickle(valuation_path)

        # 筛选特定板块
        sector_data = valuation[valuation['Sector'] == sector]

        if sector_data.empty:
            self.logger.warning(f"未找到板块: {sector}")
            return pd.DataFrame()

        # 返回时间序列
        return sector_data[['TradingDate', metric, 'TotalMarketCap', 'StockCount']].sort_values('TradingDate')

    def get_valuation_summary(self) -> pd.DataFrame:
        """
        获取所有板块的估值汇总统计

        Returns:
            pd.DataFrame: 估值汇总统计
        """
        valuation_path = self.sector_data_path / "SectorValuation.pkl"

        if not valuation_path.exists():
            self.logger.warning("板块估值数据不存在")
            return pd.DataFrame()

        valuation = pd.read_pickle(valuation_path)

        # 获取最新日期
        latest_date = valuation['TradingDate'].max()
        latest_data = valuation[valuation['TradingDate'] == latest_date]

        # 计算汇总统计
        summary = latest_data.groupby('Sector').agg({
            'TotalMarketCap': 'first',
            'StockCount': 'first',
            'PE_TTM': 'first',
            'PB': 'first',
            'PS_TTM': 'first',
            'PCF_TTM': 'first'
        }).round(2)

        return summary.sort_values('TotalMarketCap', ascending=False)