"""
行业板块指标计算器 - 混合架构实现
Sector Metrics Calculator - Hybrid Architecture

设计理念：
1. 智能路由：自动选择最优计算路径（反向计算 vs 正向计算）
2. 高效性：优先使用反向计算（基于已生成的个股因子）
3. 扩展性：支持任意新指标的正向计算
4. 向后兼容：保持与旧版本相同的接口

计算路径：
- 反向计算（高效）：个股因子 → 反推财务指标 → 板块汇总
- 正向计算（通用）：财务报表 → TTM → 日频扩展 → 板块汇总

支持的指标：
- PE_TTM: 市盈率（反向/正向）
- PB: 市净率（反向/正向）
- PS_TTM: 市销率（正向）
- PCF_TTM: 市现率（正向）
- ROE_TTM: 净资产收益率（正向）
- 更多指标可通过配置轻松添加...
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List, Union, Callable
import logging
from datetime import datetime
import time

from .base_processor import BaseDataProcessor


class SectorMetricsCalculator(BaseDataProcessor):
    """
    行业板块指标计算器 - 混合架构

    核心特性：
    1. 智能路由：自动选择反向计算或正向计算
    2. 统一接口：隐藏计算路径差异
    3. 高度扩展：支持任意新指标
    4. 向后兼容：process()方法保持兼容
    """

    def __init__(self):
        """初始化计算器"""
        super().__init__()
        self.logger = logging.getLogger(__name__)

        # 数据路径配置
        self.data_root = Path("E:/Documents/PythonProject/StockProject/StockData")
        self.factors_path = self.data_root / "RawFactors"
        self.auxiliary_path = Path(__file__).parent.parent / "auxiliary"

        # 创建输出目录
        self.sector_data_path = self.data_root / "SectorData"
        self.sector_data_path.mkdir(exist_ok=True)

        # 数据缓存
        self._cache = {}
        self._cache_time = {}

        # 注册反向计算配置（高效路径）
        self._reverse_configs = {
            'PE_TTM': {
                'factors': ['EP_ttm', 'PE_ttm'],  # 优先级顺序
                'reverse_func': self._calc_pe_reverse
            },
            'PB': {
                'factors': ['BP', 'PB'],
                'reverse_func': self._calc_pb_reverse
            },
            # 未来可扩展：
            # 'PS_TTM': {'factors': ['SP_ttm'], 'reverse_func': self._calc_ps_reverse},
            # 'PCF_TTM': {'factors': ['CFP_ttm'], 'reverse_func': self._calc_pcf_reverse},
        }

        # 注册正向计算配置（通用路径）
        self._forward_configs = {
            'PE_TTM': {
                'financial_field': 'NET_PROFIT_IS',
                'ttm': True,
                'description': '市盈率TTM'
            },
            'PB': {
                'financial_field': 'TOT_EQUITY',
                'ttm': False,
                'description': '市净率'
            },
            'PS_TTM': {
                'financial_field': 'TOT_OPER_REV',
                'ttm': True,
                'description': '市销率TTM'
            },
            'PCF_TTM': {
                'financial_field': 'NET_CASH_FLOWS_OPER_ACT',
                'ttm': True,
                'description': '市现率TTM'
            },
        }

        self.logger.info(f"板块指标计算器初始化完成")
        self.logger.info(f"支持反向计算: {list(self._reverse_configs.keys())}")
        self.logger.info(f"支持正向计算: {list(self._forward_configs.keys())}")

    def validate_input(self, **kwargs):
        """验证输入参数"""
        return True

    # ==================== 公共接口 ====================

    def process(self, date_range: int = 252, **kwargs) -> pd.DataFrame:
        """
        主处理函数 - 向后兼容接口

        计算板块估值指标（PE_TTM, PB）

        Args:
            date_range: 计算的日期范围（默认252个交易日）
            **kwargs: 其他参数

        Returns:
            板块估值DataFrame
        """
        self.logger.info("="*60)
        self.logger.info("开始计算板块估值指标...")
        self.logger.info(f"计算范围: 最近{date_range}个交易日")
        self.logger.info("="*60)

        start_time = time.time()

        # 默认计算PE_TTM和PB
        metrics = kwargs.get('metrics', ['PE_TTM', 'PB'])

        # 调用批量计算
        results = self.calculate_multiple_metrics(
            metric_list=metrics,
            date_range=date_range,
            **kwargs
        )

        elapsed = time.time() - start_time
        self.logger.info(f"板块估值计算完成！耗时: {elapsed:.2f}秒")

        return results

    def calculate_metric(self,
                        metric_name: str,
                        date_range: int = 252,
                        **kwargs) -> pd.DataFrame:
        """
        计算单个板块指标 - 智能路由

        Args:
            metric_name: 指标名称（如 'PE_TTM', 'PB', 'PS_TTM'等）
            date_range: 计算的日期范围
            **kwargs: 其他参数

        Returns:
            板块指标DataFrame
        """
        self.logger.info(f"\n计算板块指标: {metric_name}")

        # 检查是否支持反向计算且因子存在
        if self._can_use_reverse(metric_name):
            self.logger.info(f"  → 使用高效反向计算")
            return self._reverse_configs[metric_name]['reverse_func'](
                date_range=date_range, **kwargs
            )

        # 降级到正向计算
        if metric_name in self._forward_configs:
            self.logger.info(f"  → 使用正向计算（从财务报表）")
            return self._forward_calculate(metric_name, date_range, **kwargs)

        raise ValueError(f"不支持的指标: {metric_name}")

    def calculate_multiple_metrics(self,
                                   metric_list: List[str],
                                   date_range: int = 252,
                                   **kwargs) -> pd.DataFrame:
        """
        批量计算多个指标 - 优化数据加载

        Args:
            metric_list: 指标名称列表
            date_range: 计算的日期范围
            **kwargs: 其他参数

        Returns:
            包含所有指标的DataFrame
        """
        self.logger.info(f"批量计算 {len(metric_list)} 个指标: {metric_list}")

        # 按计算路径分组
        reverse_metrics = [m for m in metric_list if self._can_use_reverse(m)]
        forward_metrics = [m for m in metric_list if m not in reverse_metrics]

        self.logger.info(f"  反向计算: {reverse_metrics}")
        self.logger.info(f"  正向计算: {forward_metrics}")

        all_results = []

        # 批量反向计算（共享市值、分类数据）
        if reverse_metrics:
            self.logger.info("\n执行批量反向计算...")
            market_cap = self._load_market_cap()
            classification = self._load_classification()

            for metric in reverse_metrics:
                result = self._reverse_configs[metric]['reverse_func'](
                    date_range=date_range,
                    market_cap=market_cap,
                    classification=classification,
                    **kwargs
                )
                all_results.append(result)

        # 批量正向计算（共享财务数据）
        if forward_metrics:
            self.logger.info("\n执行批量正向计算...")
            # 正向计算会自动加载所需数据
            for metric in forward_metrics:
                result = self._forward_calculate(metric, date_range, **kwargs)
                all_results.append(result)

        # 合并结果
        if all_results:
            final_result = pd.concat(all_results, axis=0, ignore_index=False)
            # 按日期和板块排序
            final_result = final_result.sort_values(['TradingDate', 'Sector'])
            return final_result
        else:
            return pd.DataFrame()

    # ==================== 反向计算引擎 ====================

    def _calc_pe_reverse(self,
                        date_range: int = 252,
                        market_cap: Optional[pd.Series] = None,
                        classification: Optional[pd.DataFrame] = None,
                        **kwargs) -> pd.DataFrame:
        """反向计算PE_TTM（基于EP_ttm或PE_ttm因子）"""
        # 加载因子数据
        pe_ttm = self._load_factor('PE_ttm')
        if pe_ttm is None:
            ep_ttm = self._load_factor('EP_ttm')
            if ep_ttm is not None:
                # EP转PE: PE = 1/EP
                pe_ttm = 1 / ep_ttm
                pe_ttm = pe_ttm.replace([np.inf, -np.inf], np.nan)
                pe_ttm[pe_ttm < 0] = np.nan
                pe_ttm[pe_ttm > 1000] = np.nan
            else:
                raise ValueError("未找到PE_ttm或EP_ttm因子")

        # 加载市值和分类（如果未提供）
        if market_cap is None:
            market_cap = self._load_market_cap()
        if classification is None:
            classification = self._load_classification()

        # 反向推算：净利润 = 市值 / PE
        # 板块PE = Σ市值 / Σ净利润
        return self._aggregate_by_reverse(
            factor=pe_ttm,
            market_cap=market_cap,
            classification=classification,
            metric_name='PE_TTM',
            date_range=date_range
        )

    def _calc_pb_reverse(self,
                        date_range: int = 252,
                        market_cap: Optional[pd.Series] = None,
                        classification: Optional[pd.DataFrame] = None,
                        **kwargs) -> pd.DataFrame:
        """反向计算PB（基于BP或PB因子）"""
        # 加载因子数据
        pb = self._load_factor('PB')
        if pb is None:
            bp = self._load_factor('BP')
            if bp is not None:
                # BP转PB: PB = 1/BP
                pb = 1 / bp
                pb = pb.replace([np.inf, -np.inf], np.nan)
                pb[pb < 0] = np.nan
                pb[pb > 100] = np.nan
            else:
                raise ValueError("未找到PB或BP因子")

        # 加载市值和分类（如果未提供）
        if market_cap is None:
            market_cap = self._load_market_cap()
        if classification is None:
            classification = self._load_classification()

        # 反向推算：净资产 = 市值 / PB
        # 板块PB = Σ市值 / Σ净资产
        return self._aggregate_by_reverse(
            factor=pb,
            market_cap=market_cap,
            classification=classification,
            metric_name='PB',
            date_range=date_range
        )

    def _aggregate_by_reverse(self,
                             factor: pd.Series,
                             market_cap: pd.Series,
                             classification: pd.DataFrame,
                             metric_name: str,
                             date_range: int) -> pd.DataFrame:
        """
        通用的反向聚合逻辑

        公式：
        - 个股财务值 = 市值 / 因子值
        - 板块因子值 = Σ市值 / Σ财务值
        """
        # 获取交易日期
        trading_dates = sorted(classification.index.get_level_values('TradingDates').unique())
        latest_dates = trading_dates[-date_range:] if len(trading_dates) > date_range else trading_dates

        self.logger.info(f"  日期范围: {latest_dates[0]} 到 {latest_dates[-1]}")

        results = []

        for i, date in enumerate(latest_dates):
            if i % 50 == 0:
                self.logger.info(f"  处理进度: {i}/{len(latest_dates)}")

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

                # 计算板块总市值和总财务值
                total_market_cap = 0
                total_financial_value = 0
                valid_count = 0

                for stock in sector_stocks:
                    if (date, stock) in market_cap.index and (date, stock) in factor.index:
                        mc = market_cap.loc[(date, stock)]
                        fv = factor.loc[(date, stock)]

                        if pd.notna(mc) and pd.notna(fv) and mc > 0 and fv > 0:
                            # 个股财务值 = 市值 / 因子值
                            financial_val = mc / fv
                            total_market_cap += mc
                            total_financial_value += financial_val
                            valid_count += 1

                # 计算板块指标
                if total_financial_value > 0 and valid_count > 0:
                    sector_metric = total_market_cap / total_financial_value

                    results.append({
                        'TradingDate': date,
                        'Sector': sector,
                        metric_name: sector_metric,
                        'TotalMarketCap': total_market_cap,
                        f'Total_{metric_name}_FinancialValue': total_financial_value,
                        'StockCount': len(sector_stocks),
                        'ValidStocks': valid_count
                    })

        results_df = pd.DataFrame(results)
        self.logger.info(f"  {metric_name} 计算完成，共 {len(results_df)} 条记录")

        return results_df

    # ==================== 正向计算引擎 ====================

    def _forward_calculate(self,
                          metric_name: str,
                          date_range: int,
                          **kwargs) -> pd.DataFrame:
        """
        正向计算（从财务报表数据）

        流程：
        1. 加载财务数据
        2. 计算TTM（如果需要）
        3. 扩展到日频
        4. 按行业聚合
        """
        config = self._forward_configs[metric_name]

        self.logger.info(f"  正向计算 {metric_name}: {config['description']}")

        # 1. 加载财务数据
        financial_data = self._load_financial_data()
        field = config['financial_field']

        if field not in financial_data.columns:
            raise ValueError(f"财务数据中未找到字段: {field}")

        field_data = financial_data[field]

        # 2. 计算TTM（如果需要）
        if config['ttm']:
            from factors.generators import calculate_ttm
            self.logger.info(f"  计算TTM...")
            field_data = calculate_ttm(field_data)

        # 3. 扩展到日频
        from factors.generators import expand_to_daily_vectorized
        self.logger.info(f"  扩展到日频...")

        release_dates = self._load_release_dates()
        trading_dates = self._load_trading_dates()

        daily_data = expand_to_daily_vectorized(
            factor_data=field_data,
            release_dates=release_dates,
            trading_dates=trading_dates
        )

        # 4. 按行业聚合
        self.logger.info(f"  按行业聚合...")
        market_cap = self._load_market_cap()
        classification = self._load_classification()

        return self._aggregate_by_forward(
            daily_data=daily_data,
            market_cap=market_cap,
            classification=classification,
            metric_name=metric_name,
            date_range=date_range
        )

    def _aggregate_by_forward(self,
                             daily_data: pd.Series,
                             market_cap: pd.Series,
                             classification: pd.DataFrame,
                             metric_name: str,
                             date_range: int) -> pd.DataFrame:
        """
        通用的正向聚合逻辑

        公式：
        - 板块财务值 = Σ个股财务值
        - 板块因子值 = 板块总市值 / 板块财务值
        """
        # 获取交易日期
        trading_dates = sorted(classification.index.get_level_values('TradingDates').unique())
        latest_dates = trading_dates[-date_range:] if len(trading_dates) > date_range else trading_dates

        self.logger.info(f"  日期范围: {latest_dates[0]} 到 {latest_dates[-1]}")

        results = []

        for i, date in enumerate(latest_dates):
            if i % 50 == 0:
                self.logger.info(f"  处理进度: {i}/{len(latest_dates)}")

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

                # 计算板块总市值和总财务值
                total_market_cap = 0
                total_financial_value = 0
                valid_count = 0

                for stock in sector_stocks:
                    if (date, stock) in market_cap.index and (date, stock) in daily_data.index:
                        mc = market_cap.loc[(date, stock)]
                        fv = daily_data.loc[(date, stock)]

                        if pd.notna(mc) and pd.notna(fv) and mc > 0:
                            total_market_cap += mc
                            total_financial_value += fv
                            valid_count += 1

                # 计算板块指标
                if total_financial_value > 0 and valid_count > 0:
                    sector_metric = total_market_cap / total_financial_value

                    results.append({
                        'TradingDate': date,
                        'Sector': sector,
                        metric_name: sector_metric,
                        'TotalMarketCap': total_market_cap,
                        f'Total_{metric_name}_FinancialValue': total_financial_value,
                        'StockCount': len(sector_stocks),
                        'ValidStocks': valid_count
                    })

        results_df = pd.DataFrame(results)
        self.logger.info(f"  {metric_name} 计算完成，共 {len(results_df)} 条记录")

        return results_df

    # ==================== 辅助方法 ====================

    def _can_use_reverse(self, metric_name: str) -> bool:
        """检查是否可以使用反向计算"""
        if metric_name not in self._reverse_configs:
            return False

        # 检查是否有可用的因子文件
        config = self._reverse_configs[metric_name]
        for factor_name in config['factors']:
            factor_path = self.factors_path / f"{factor_name}.pkl"
            if factor_path.exists():
                return True

        return False

    def _load_factor(self, factor_name: str) -> Optional[pd.Series]:
        """加载因子数据（带缓存）"""
        cache_key = f'factor_{factor_name}'

        if cache_key in self._cache:
            return self._cache[cache_key]

        factor_path = self.factors_path / f"{factor_name}.pkl"
        if not factor_path.exists():
            self.logger.debug(f"因子文件不存在: {factor_name}")
            return None

        self.logger.info(f"  加载因子: {factor_name}")
        data = pd.read_pickle(factor_path)
        self._cache[cache_key] = data

        return data

    def _load_market_cap(self) -> pd.Series:
        """加载市值数据（带缓存）"""
        cache_key = 'market_cap'

        if cache_key in self._cache:
            return self._cache[cache_key]

        price_path = self.data_root / "Price.pkl"
        price_data = pd.read_pickle(price_path)

        if 'MC' not in price_data.columns:
            raise ValueError("价格数据中未找到市值(MC)字段")

        market_cap = price_data['MC']
        self._cache[cache_key] = market_cap
        self.logger.info(f"  加载市值数据: {market_cap.shape}")

        return market_cap

    def _load_classification(self) -> pd.DataFrame:
        """加载行业分类数据（带缓存）"""
        cache_key = 'classification'

        if cache_key in self._cache:
            return self._cache[cache_key]

        classification_path = self.data_root / "Classificationdata" / "classification_one_hot.pkl"
        classification = pd.read_pickle(classification_path)
        self._cache[cache_key] = classification
        self.logger.info(f"  加载行业分类: {classification.shape}")

        return classification

    def _load_financial_data(self) -> pd.DataFrame:
        """加载财务数据（带缓存）"""
        cache_key = 'financial_data'

        if cache_key in self._cache:
            return self._cache[cache_key]

        financial_path = self.auxiliary_path / "FinancialData_unified.pkl"
        financial_data = pd.read_pickle(financial_path)
        self._cache[cache_key] = financial_data
        self.logger.info(f"  加载财务数据: {financial_data.shape}")

        return financial_data

    def _load_release_dates(self) -> pd.Series:
        """加载财报发布日期"""
        cache_key = 'release_dates'

        if cache_key in self._cache:
            return self._cache[cache_key]

        release_path = self.auxiliary_path / "ReleaseDates.pkl"
        release_dates = pd.read_pickle(release_path)
        self._cache[cache_key] = release_dates

        return release_dates

    def _load_trading_dates(self) -> pd.Index:
        """加载交易日期"""
        cache_key = 'trading_dates'

        if cache_key in self._cache:
            return self._cache[cache_key]

        trading_path = self.auxiliary_path / "TradingDates.pkl"
        trading_dates = pd.read_pickle(trading_path)
        self._cache[cache_key] = trading_dates

        return trading_dates

    def clear_cache(self):
        """清除缓存"""
        self._cache.clear()
        self._cache_time.clear()
        self.logger.info("缓存已清除")

    # ==================== 向后兼容别名 ====================

    def get_latest_valuation(self, sector: Optional[str] = None) -> pd.DataFrame:
        """获取最新的板块估值数据（向后兼容）"""
        result = self.process(date_range=1)

        if sector:
            result = result[result['Sector'] == sector]

        return result

    def get_sector_history(self, sector: str, metric: str = 'PE_TTM', days: int = 252) -> pd.DataFrame:
        """获取特定板块的历史估值数据（向后兼容）"""
        result = self.calculate_metric(metric, date_range=days)
        result = result[result['Sector'] == sector]

        return result.sort_values('TradingDate')


# 向后兼容别名
SectorValuationFromStockPE = SectorMetricsCalculator
