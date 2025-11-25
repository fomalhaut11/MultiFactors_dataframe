"""
集成数据处理管道
包含所有数据处理功能，包括板块估值计算

完整的数据更新流程：
1. 价格数据处理
2. 收益率计算
3. 财务数据处理
4. 行业分类处理
5. 板块估值计算（新增）
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
import logging
import time
import gc

from .data_processing_pipeline import DataProcessingPipeline
from .sector_metrics_calculator import SectorMetricsCalculator
from config import get_config


class IntegratedDataPipeline(DataProcessingPipeline):
    """
    集成的数据处理管道
    在原有功能基础上增加板块估值计算
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        初始化集成管道

        Args:
            config_path: 配置文件路径
        """
        super().__init__(config_path)

        # 初始化板块指标计算器（使用混合架构）
        self.sector_valuation_processor = SectorMetricsCalculator()

        # 配置板块估值计算参数
        self.sector_valuation_config = {
            'enabled': True,  # 是否启用板块估值计算
            'date_range': 252,  # 默认计算最近252个交易日（一年）
            'save_intermediate': True,  # 保存中间结果
            'output_formats': ['pkl', 'csv', 'json']  # 输出格式
        }

    def run_full_pipeline(self,
                         save_intermediate: bool = True,
                         include_sector_valuation: bool = True,
                         sector_date_range: Optional[int] = None) -> Dict[str, Any]:
        """
        运行完整的数据处理流程（包含板块估值）

        Args:
            save_intermediate: 是否保存中间结果
            include_sector_valuation: 是否包含板块估值计算
            sector_date_range: 板块估值计算的日期范围

        Returns:
            处理结果字典
        """
        self.logger.info("="*60)
        self.logger.info("开始运行集成数据处理管道...")
        self.logger.info("="*60)

        start_time = time.time()

        # 调用父类的处理流程
        results = super().run_full_pipeline(save_intermediate)

        # 添加板块估值计算
        if include_sector_valuation and self.sector_valuation_config['enabled']:
            self.logger.info("\n" + "="*60)
            self.logger.info("步骤6: 计算板块估值指标...")
            self.logger.info("="*60)

            try:
                # 确定计算范围
                date_range = sector_date_range or self.sector_valuation_config['date_range']

                # 执行板块估值计算
                sector_valuation_df = self.sector_valuation_processor.process(
                    date_range=date_range
                )

                if not sector_valuation_df.empty:
                    results['sector_valuation'] = sector_valuation_df

                    # 获取最新的板块估值统计
                    latest_stats = self._get_latest_sector_stats(sector_valuation_df)
                    results['sector_valuation_stats'] = latest_stats

                    self.logger.info(f"板块估值计算完成，共{len(sector_valuation_df)}条记录")
                    self.logger.info(f"最新日期PE中位数: {latest_stats.get('pe_median', 'N/A'):.2f}")
                    self.logger.info(f"最新日期PB中位数: {latest_stats.get('pb_median', 'N/A'):.2f}")
                else:
                    self.logger.warning("板块估值计算结果为空")

            except Exception as e:
                self.logger.error(f"板块估值计算失败: {e}")
                # 板块估值计算失败不影响整个管道

        elapsed = time.time() - start_time
        self.logger.info(f"\n集成数据处理管道执行完成！总耗时: {elapsed:.2f}秒")

        # 记录处理历史
        self._record_processing(
            'integrated_pipeline',
            {
                'save_intermediate': save_intermediate,
                'include_sector_valuation': include_sector_valuation,
                'sector_date_range': sector_date_range
            },
            {
                'status': 'success',
                'steps_completed': 6 if include_sector_valuation else 5,
                'total_time': elapsed
            }
        )

        return results

    def update_sector_valuation(self,
                               date_range: Optional[int] = None,
                               force_update: bool = False) -> pd.DataFrame:
        """
        独立更新板块估值数据

        Args:
            date_range: 计算的日期范围
            force_update: 是否强制更新（忽略缓存）

        Returns:
            板块估值DataFrame
        """
        self.logger.info("单独更新板块估值数据...")

        # 检查是否需要更新
        if not force_update:
            # 检查最新数据日期
            sector_data_path = Path(get_config('main.paths.data_root')) / "SectorData"
            pkl_path = sector_data_path / "sector_valuation_from_stock_pe.pkl"

            if pkl_path.exists():
                # 获取文件修改时间
                file_mtime = datetime.fromtimestamp(pkl_path.stat().st_mtime)
                current_time = datetime.now()

                # 如果文件是今天生成的，且不强制更新，则跳过
                if file_mtime.date() == current_time.date():
                    self.logger.info("板块估值数据已是最新，跳过更新")
                    return pd.read_pickle(pkl_path)

        # 执行更新
        date_range = date_range or self.sector_valuation_config['date_range']
        sector_valuation = self.sector_valuation_processor.process(date_range=date_range)

        return sector_valuation

    def _get_latest_sector_stats(self, sector_valuation_df: pd.DataFrame) -> Dict:
        """
        获取最新的板块估值统计

        Args:
            sector_valuation_df: 板块估值数据

        Returns:
            统计信息字典
        """
        if sector_valuation_df.empty:
            return {}

        # 获取最新日期数据
        latest_date = sector_valuation_df['TradingDate'].max()
        latest_data = sector_valuation_df[
            sector_valuation_df['TradingDate'] == latest_date
        ]

        stats = {
            'latest_date': str(latest_date),
            'sector_count': latest_data['Sector'].nunique(),
            'total_records': len(latest_data)
        }

        # PE统计
        if 'PE_TTM' in latest_data.columns:
            pe_data = latest_data['PE_TTM'].dropna()
            if not pe_data.empty:
                stats['pe_mean'] = pe_data.mean()
                stats['pe_median'] = pe_data.median()
                stats['pe_min'] = pe_data.min()
                stats['pe_max'] = pe_data.max()

        # PB统计
        if 'PB' in latest_data.columns:
            pb_data = latest_data['PB'].dropna()
            if not pb_data.empty:
                stats['pb_mean'] = pb_data.mean()
                stats['pb_median'] = pb_data.median()
                stats['pb_min'] = pb_data.min()
                stats['pb_max'] = pb_data.max()

        # 市值统计
        if 'TotalMarketCap' in latest_data.columns:
            stats['total_market_cap'] = latest_data['TotalMarketCap'].sum()
            stats['avg_sector_market_cap'] = latest_data['TotalMarketCap'].mean()

        return stats

    def configure_sector_valuation(self, config: Dict[str, Any]):
        """
        配置板块估值计算参数

        Args:
            config: 配置字典
                - enabled: 是否启用
                - date_range: 日期范围
                - save_intermediate: 是否保存中间结果
                - output_formats: 输出格式列表
        """
        self.sector_valuation_config.update(config)
        self.logger.info(f"板块估值配置已更新: {self.sector_valuation_config}")


class DataUpdateScheduler:
    """
    数据更新调度器
    管理定时更新任务
    """

    def __init__(self, pipeline: Optional[IntegratedDataPipeline] = None):
        """
        初始化调度器

        Args:
            pipeline: 数据处理管道实例
        """
        self.logger = logging.getLogger(__name__)
        self.pipeline = pipeline or IntegratedDataPipeline()

        # 更新任务配置
        self.update_tasks = {
            'daily': {
                'price_data': True,
                'returns': True,
                'sector_valuation': True,
                'date_range': 1  # 每日更新最近1天
            },
            'weekly': {
                'financial_data': True,
                'sector_classification': True,
                'sector_valuation': True,
                'date_range': 5  # 每周更新最近5天
            },
            'monthly': {
                'full_pipeline': True,
                'sector_valuation': True,
                'date_range': 22  # 每月更新最近22天
            }
        }

    def run_daily_update(self):
        """执行每日更新任务"""
        self.logger.info("执行每日数据更新...")

        try:
            # 更新价格和收益率
            self.pipeline.price_processor.process(save_to_file=True)

            # 更新板块估值（最近1天）
            self.pipeline.update_sector_valuation(
                date_range=self.update_tasks['daily']['date_range']
            )

            self.logger.info("每日更新完成")

        except Exception as e:
            self.logger.error(f"每日更新失败: {e}")
            raise

    def run_weekly_update(self):
        """执行每周更新任务"""
        self.logger.info("执行每周数据更新...")

        try:
            # 更新财务数据
            trading_dates = pd.read_pickle(
                Path(get_config('main.paths.data_root')) / "Price.pkl"
            ).index.get_level_values(0).unique()

            self.pipeline.financial_processor.process_all_financial_data(
                trading_dates=trading_dates
            )

            # 更新板块估值（最近5天）
            self.pipeline.update_sector_valuation(
                date_range=self.update_tasks['weekly']['date_range']
            )

            self.logger.info("每周更新完成")

        except Exception as e:
            self.logger.error(f"每周更新失败: {e}")
            raise

    def run_monthly_update(self):
        """执行每月更新任务"""
        self.logger.info("执行每月完整更新...")

        try:
            # 运行完整管道
            self.pipeline.run_full_pipeline(
                save_intermediate=True,
                include_sector_valuation=True,
                sector_date_range=self.update_tasks['monthly']['date_range']
            )

            self.logger.info("每月更新完成")

        except Exception as e:
            self.logger.error(f"每月更新失败: {e}")
            raise

    def run_custom_update(self,
                         update_price: bool = False,
                         update_financial: bool = False,
                         update_sector_valuation: bool = False,
                         sector_date_range: Optional[int] = None):
        """
        执行自定义更新任务

        Args:
            update_price: 是否更新价格数据
            update_financial: 是否更新财务数据
            update_sector_valuation: 是否更新板块估值
            sector_date_range: 板块估值计算范围
        """
        self.logger.info("执行自定义数据更新...")

        try:
            if update_price:
                self.logger.info("更新价格数据...")
                self.pipeline.price_processor.process(save_to_file=True)

            if update_financial:
                self.logger.info("更新财务数据...")
                trading_dates = pd.read_pickle(
                    Path(get_config('main.paths.data_root')) / "Price.pkl"
                ).index.get_level_values(0).unique()
                self.pipeline.financial_processor.process_all_financial_data(
                    trading_dates=trading_dates
                )

            if update_sector_valuation:
                self.logger.info("更新板块估值...")
                self.pipeline.update_sector_valuation(
                    date_range=sector_date_range
                )

            self.logger.info("自定义更新完成")

        except Exception as e:
            self.logger.error(f"自定义更新失败: {e}")
            raise