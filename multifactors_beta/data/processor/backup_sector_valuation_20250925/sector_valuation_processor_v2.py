"""
行业板块估值指标处理器 V2版本
增加中间数据保存功能，提高效率

中间数据保存：
1. TTM财务指标 - 避免重复计算TTM
2. 日频财务数据 - 避免重复扩展到日频
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Union, List
import warnings
import sys
import logging
from datetime import datetime
import time

# 添加上级目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from factors.generators import (
    calculate_ttm,
    expand_to_daily_vectorized,
    FinancialReportProcessor
)

from data.processor.base_processor import BaseDataProcessor

class SectorValuationProcessorV2(BaseDataProcessor):
    """
    行业板块估值指标计算器 V2

    新增功能：
    - 保存TTM中间数据
    - 保存日频财务数据
    - 支持增量计算
    """

    def __init__(self, save_intermediate=True):
        """
        初始化处理器

        Args:
            save_intermediate: 是否保存中间数据
        """
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.save_intermediate = save_intermediate

        # 数据路径配置
        self.data_root = Path("E:/Documents/PythonProject/StockProject/StockData")
        self.auxiliary_path = Path(__file__).parent.parent / "auxiliary"

        # 创建数据存储目录
        self.sector_data_path = self.data_root / "SectorData"
        self.sector_data_path.mkdir(exist_ok=True)

        # 中间数据存储路径
        self.intermediate_path = self.sector_data_path / "intermediate"
        if self.save_intermediate:
            self.intermediate_path.mkdir(exist_ok=True)
            self.logger.info(f"中间数据将存储在: {self.intermediate_path}")

        # 财务字段映射
        self.field_mapping = {
            'net_profit': 'NET_PROFIT_IS',
            'total_equity': 'TOT_EQUITY',
            'total_assets': 'TOT_ASSETS',
            'revenue': 'TOT_OPER_REV',
            'operating_cash_flow': 'NET_CASH_FLOWS_OPER_ACT'
        }

    def load_or_calculate_ttm(self) -> Dict[str, pd.Series]:
        """
        加载或计算TTM财务指标

        Returns:
            TTM财务指标字典
        """
        ttm_file = self.intermediate_path / "ttm_financials.pkl"

        # 尝试加载已有的TTM数据
        if self.save_intermediate and ttm_file.exists():
            try:
                self.logger.info(f"加载已有的TTM数据: {ttm_file}")
                ttm_data = pd.read_pickle(ttm_file)

                # 验证数据完整性
                expected_keys = ['net_profit_ttm', 'revenue_ttm', 'ocf_ttm', 'total_equity']
                if all(key in ttm_data for key in expected_keys):
                    self.logger.info("成功加载TTM数据")
                    return ttm_data
                else:
                    self.logger.warning("TTM数据不完整，重新计算")
            except Exception as e:
                self.logger.warning(f"加载TTM数据失败: {e}")

        # 计算TTM
        self.logger.info("计算TTM财务指标...")
        ttm_data = self.calculate_ttm_financials()

        # 保存TTM数据
        if self.save_intermediate and ttm_data:
            try:
                pd.to_pickle(ttm_data, ttm_file)
                self.logger.info(f"TTM数据已保存至: {ttm_file}")

                # 同时保存为CSV便于查看
                for key, data in ttm_data.items():
                    csv_file = self.intermediate_path / f"{key}.csv"
                    data.to_csv(csv_file, encoding='utf-8-sig')
                    self.logger.info(f"保存{key}到CSV: {csv_file}")
            except Exception as e:
                self.logger.error(f"保存TTM数据失败: {e}")

        return ttm_data

    def load_or_expand_daily(self, ttm_data: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        加载或计算日频财务数据

        Args:
            ttm_data: TTM财务指标

        Returns:
            日频财务数据字典
        """
        daily_file = self.intermediate_path / "daily_financials.pkl"

        # 尝试加载已有的日频数据
        if self.save_intermediate and daily_file.exists():
            try:
                self.logger.info(f"加载已有的日频数据: {daily_file}")
                daily_data = pd.read_pickle(daily_file)

                # 验证数据完整性
                if daily_data and len(daily_data) == len(ttm_data):
                    self.logger.info("成功加载日频数据")
                    return daily_data
                else:
                    self.logger.warning("日频数据不完整，重新计算")
            except Exception as e:
                self.logger.warning(f"加载日频数据失败: {e}")

        # 扩展到日频
        self.logger.info("将财务数据扩展到日频...")
        daily_data = self.expand_financial_to_daily(ttm_data)

        # 保存日频数据
        if self.save_intermediate and daily_data:
            try:
                pd.to_pickle(daily_data, daily_file)
                self.logger.info(f"日频数据已保存至: {daily_file}")
            except Exception as e:
                self.logger.error(f"保存日频数据失败: {e}")

        return daily_data

    def calculate_ttm_financials(self) -> Dict[str, pd.Series]:
        """计算TTM财务指标"""
        self.logger.info("开始计算TTM财务指标...")
        start_time = time.time()

        # 加载财务数据（如果还没加载）
        if not hasattr(self, 'financial_data'):
            self.financial_data = pd.read_pickle(self.auxiliary_path / "FinancialData_unified.pkl")

        ttm_data = {}
        time_fields = ['d_quarter', 'd_year'] if 'd_quarter' in self.financial_data.columns else []

        # 计算各项TTM指标
        for name, field in [
            ('net_profit_ttm', 'net_profit'),
            ('revenue_ttm', 'revenue'),
            ('ocf_ttm', 'operating_cash_flow')
        ]:
            if self.field_mapping[field] in self.financial_data.columns:
                self.logger.info(f"计算{name}...")
                fields_to_include = [self.field_mapping[field]] + time_fields
                df = self.financial_data[fields_to_include].copy()

                ttm_result = calculate_ttm(df)
                ttm_field_name = f"{self.field_mapping[field]}_ttm"
                ttm_data[name] = ttm_result[ttm_field_name]
                self.logger.info(f"{name}计算完成: {ttm_data[name].shape}")

        # 获取最新股东权益
        if self.field_mapping['total_equity'] in self.financial_data.columns:
            ttm_data['total_equity'] = self.financial_data[self.field_mapping['total_equity']]
            self.logger.info(f"股东权益数据获取完成: {ttm_data['total_equity'].shape}")

        elapsed = time.time() - start_time
        self.logger.info(f"TTM计算完成，耗时: {elapsed:.2f}秒")

        return ttm_data

    def expand_financial_to_daily(self, ttm_data: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """将财务数据扩展到日频"""
        self.logger.info("开始扩展财务数据到日频...")
        start_time = time.time()

        # 加载必要数据
        if not hasattr(self, 'release_dates'):
            self.release_dates = pd.read_pickle(self.auxiliary_path / "ReleaseDates.pkl")
        if not hasattr(self, 'trading_dates'):
            self.trading_dates = pd.read_pickle(self.auxiliary_path / "TradingDates.pkl")
            if isinstance(self.trading_dates, pd.Series):
                self.trading_dates = self.trading_dates.values

        daily_data = {}

        for key, data in ttm_data.items():
            try:
                self.logger.info(f"扩展{key}到日频...")

                # Series转DataFrame
                if isinstance(data, pd.Series):
                    data_df = pd.DataFrame({key: data})
                else:
                    data_df = data

                # 扩展到日频
                result_df = expand_to_daily_vectorized(
                    factor_data=data_df,
                    release_dates=self.release_dates,
                    trading_dates=self.trading_dates
                )

                # 提取结果
                daily_data[key] = result_df[key] if isinstance(result_df, pd.DataFrame) else result_df
                self.logger.info(f"{key}扩展完成: {daily_data[key].shape}")

            except Exception as e:
                self.logger.error(f"扩展{key}失败: {e}")

        elapsed = time.time() - start_time
        self.logger.info(f"日频扩展完成，耗时: {elapsed:.2f}秒")

        return daily_data

    def validate_input(self, **kwargs):
        """验证输入参数"""
        return True

    def load_data(self):
        """加载必要的数据"""
        self.logger.info("加载数据...")

        # 1. 加载财务数据
        self.financial_data = pd.read_pickle(self.auxiliary_path / "FinancialData_unified.pkl")
        self.logger.info(f"财务数据形状: {self.financial_data.shape}")

        # 2. 加载市值数据
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

    def calculate_sector_valuation(self, daily_financials: Dict[str, pd.Series],
                                  date_range: int = 252) -> pd.DataFrame:
        """
        计算行业板块估值

        Args:
            daily_financials: 日频财务数据
            date_range: 计算的日期范围（默认252个交易日）
        """
        self.logger.info(f"计算行业板块估值指标（最近{date_range}个交易日）...")

        results = []
        latest_dates = sorted(pd.Series(self.trading_dates).unique())[-date_range:]

        for i, date in enumerate(latest_dates):
            if i % 50 == 0:
                self.logger.info(f"处理进度: {i}/{len(latest_dates)}")

            if date in self.classification.index.get_level_values('TradingDates'):
                daily_classification = self.classification.xs(date, level='TradingDates')

                # 计算每个行业的估值
                daily_results = self.calculate_sector_valuation_for_date(
                    date, daily_classification, daily_financials
                )
                results.extend(daily_results)

        results_df = pd.DataFrame(results)
        self.logger.info(f"板块估值计算完成，共{len(results_df)}条记录")

        return results_df

    def calculate_sector_valuation_for_date(
        self,
        date: pd.Timestamp,
        daily_classification: pd.DataFrame,
        daily_financials: Dict[str, pd.Series]
    ) -> List[Dict]:
        """计算特定日期的板块估值（与原版相同）"""

        results = []
        sectors = self.classification.columns.tolist()

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

                # 计算PE_TTM
                if 'net_profit_ttm' in daily_financials:
                    sector_profits = []
                    for stock in sector_stocks:
                        if (date, stock) in daily_financials['net_profit_ttm'].index:
                            profit = daily_financials['net_profit_ttm'].loc[(date, stock)]
                            if pd.notna(profit):
                                sector_profits.append(profit)

                    if sector_profits:
                        total_profit = sum(sector_profits)
                        if total_profit > 0:
                            sector_result['PE_TTM'] = total_market_cap / total_profit
                            sector_result['TotalProfit_TTM'] = total_profit

                # 计算PB
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

                results.append(sector_result)

            except Exception as e:
                self.logger.debug(f"计算{sector}在{date}的估值失败: {e}")
                continue

        return results

    def process(self, force_recalc=False, date_range=252):
        """
        主处理函数

        Args:
            force_recalc: 是否强制重新计算（忽略缓存）
            date_range: 计算的日期范围
        """
        self.logger.info(f"开始计算行业板块估值指标...")
        self.logger.info(f"保存中间数据: {self.save_intermediate}")
        self.logger.info(f"强制重新计算: {force_recalc}")

        overall_start = time.time()

        # 如果强制重新计算，删除缓存
        if force_recalc and self.intermediate_path.exists():
            self.logger.info("删除旧的中间数据...")
            for file in self.intermediate_path.glob("*.pkl"):
                file.unlink()

        # 1. 加载基础数据
        self.load_data()

        # 2. 加载或计算TTM
        ttm_data = self.load_or_calculate_ttm()

        # 3. 加载或扩展到日频
        daily_financials = self.load_or_expand_daily(ttm_data)

        # 4. 计算板块估值
        sector_valuation = self.calculate_sector_valuation(daily_financials, date_range)

        # 5. 保存结果
        output_path = self.sector_data_path / "SectorValuation.pkl"
        sector_valuation.to_pickle(output_path)
        self.logger.info(f"板块估值数据已保存至: {output_path}")

        csv_path = self.sector_data_path / "SectorValuation.csv"
        sector_valuation.to_csv(csv_path, index=False, encoding='utf-8-sig')
        self.logger.info(f"板块估值CSV已保存至: {csv_path}")

        # 记录总耗时
        total_elapsed = time.time() - overall_start
        self.logger.info(f"总计耗时: {total_elapsed:.2f}秒")

        # 记录处理历史
        self._record_processing(
            operation="sector_valuation_v2",
            params={'force_recalc': force_recalc, 'date_range': date_range},
            result_info={
                "shape": sector_valuation.shape,
                "sectors": sector_valuation['Sector'].nunique() if not sector_valuation.empty else 0,
                "dates": sector_valuation['TradingDate'].nunique() if not sector_valuation.empty else 0,
                "metrics": [col for col in sector_valuation.columns if col.endswith('_TTM') or col in ['PB']],
                "total_time": f"{total_elapsed:.2f}s"
            }
        )

        return sector_valuation