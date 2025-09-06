#!/usr/bin/env python3
"""
自定义混合因子
需要财务数据和市值数据的复合因子
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Any
import logging

from factors.base.factor_base import FactorBase
from factors.base.data_processing_mixin import DataProcessingMixin
from factors.base.validation import DataValidator

logger = logging.getLogger(__name__)


class CashflowEfficiencyRatio(FactorBase, DataProcessingMixin):
    """
    现金流效率比率因子
    
    计算公式：
    ((FIN_EXP_CS + DEPR_FA_COGA_DPBA) / CASH_RECP_SG_AND_RS) / BP
    
    其中：
    - FIN_EXP_CS: 财务费用
    - DEPR_FA_COGA_DPBA: 固定资产折旧、油气资产折耗、生产性生物资产折旧
    - CASH_RECP_SG_AND_RS: 销售商品、提供劳务收到的现金
    - BP: 净资产市值比
    
    这个因子衡量企业现金流相对于账面价值的效率
    """
    
    def __init__(self):
        super().__init__(
            name="CashflowEfficiencyRatio",
            category="mixed"
        )
        self.factor_name = "CashflowEfficiencyRatio"
        self.factor_description = "现金流效率比率：(财务费用+折旧)/销售现金流/净资产市值比"
        
        # 设置数据依赖
        self.required_fields = [
            'FIN_EXP_CS',           # 财务费用
            'DEPR_FA_COGA_DPBA',    # 折旧费用
            'CASH_RECP_SG_AND_RS',  # 销售商品收现
            'BP'                    # 净资产市值比
        ]
        
        # 设置字段映射（如果数据库字段名不同）
        self.field_mapping = {
            'fin_expense': 'FIN_EXP_CS',
            'depreciation': 'DEPR_FA_COGA_DPBA', 
            'sales_cash': 'CASH_RECP_SG_AND_RS',
            'book_to_price': 'BP'
        }
    
    def validate_data_requirements(self, data: Dict[str, pd.DataFrame]) -> bool:
        """验证数据完整性"""
        try:
            validator = DataValidator()
            
            # 检查财务数据
            if 'financial_data' not in data:
                logger.error("缺少财务数据")
                return False
            
            financial_data = data['financial_data']
            
            # 检查必需的财务字段
            financial_fields = ['FIN_EXP_CS', 'DEPR_FA_COGA_DPBA', 'CASH_RECP_SG_AND_RS']
            missing_fields = []
            
            for field in financial_fields:
                if field not in financial_data.columns:
                    # 尝试映射字段名
                    mapped_field = self.get_mapped_column(field)
                    if mapped_field not in financial_data.columns:
                        missing_fields.append(field)
            
            if missing_fields:
                logger.error(f"财务数据缺少字段: {missing_fields}")
                return False
            
            # 检查BP数据
            if 'bp_data' not in data and 'BP' not in financial_data.columns:
                logger.error("缺少BP（净资产市值比）数据")
                return False
            
            # 验证数据质量
            try:
                validator.validate_financial_data(
                    financial_data, 
                    ['FIN_EXP_CS', 'DEPR_FA_COGA_DPBA', 'CASH_RECP_SG_AND_RS']
                )
            except Exception as e:
                logger.error(f"财务数据格式不正确: {e}")
                return False
            
            logger.info("数据验证通过")
            return True
            
        except Exception as e:
            logger.error(f"数据验证失败: {e}")
            return False
    
    def calculate(self, data: Dict[str, pd.DataFrame]) -> pd.Series:
        """
        计算现金流效率比率因子
        
        Parameters
        ----------
        data : Dict[str, pd.DataFrame]
            包含财务数据的字典
            - 'financial_data': 财务报表数据（必需）
            - 'bp_data': BP数据（可选，如果financial_data中没有BP列）
            
        Returns
        -------
        pd.Series
            计算得到的因子值，MultiIndex格式(date, stock_code)
        """
        try:
            logger.info("开始计算现金流效率比率因子")
            
            # 验证数据
            if not self.validate_data_requirements(data):
                logger.error("数据验证失败")
                return pd.Series()
            
            financial_data = data['financial_data']
            
            # 提取所需字段
            fin_expense = self._get_field_data(financial_data, 'FIN_EXP_CS', 'fin_expense')
            depreciation = self._get_field_data(financial_data, 'DEPR_FA_COGA_DPBA', 'depreciation')
            sales_cash = self._get_field_data(financial_data, 'CASH_RECP_SG_AND_RS', 'sales_cash')
            
            # 获取BP数据
            if 'bp_data' in data:
                bp = data['bp_data']
                if isinstance(bp, pd.DataFrame) and len(bp.columns) == 1:
                    bp = bp.iloc[:, 0]
            else:
                bp = self._get_field_data(financial_data, 'BP', 'book_to_price')
            
            logger.info(f"原始数据提取完成 - 财务费用: {len(fin_expense)}, 折旧: {len(depreciation)}, "
                       f"销售现金: {len(sales_cash)}, BP: {len(bp)}")
            
            # 🚀 优化：在季度频率上先计算，减少数据填充量
            logger.info("在季度频率上进行财务指标计算...")
            
            # 第一步：在季度频率上计算中间结果
            logger.info("计算季度财务指标...")
            
            # 数据对齐（季度数据内部对齐）
            quarterly_aligned = self._align_data([fin_expense, depreciation, sales_cash])
            if not quarterly_aligned:
                logger.error("季度财务数据对齐失败")
                return pd.Series()
            
            fin_expense_q, depreciation_q, sales_cash_q = quarterly_aligned
            
            # 在季度频率上计算中间指标
            cost_sum_quarterly = fin_expense_q + depreciation_q
            
            # 计算现金流效率比率（季度频率）
            cash_efficiency_quarterly = self._safe_divide(
                cost_sum_quarterly, sales_cash_q, 'cost_sum', 'sales_cash'
            )
            
            if cash_efficiency_quarterly.empty:
                logger.error("季度现金流效率计算失败")
                return pd.Series()
            
            logger.info(f"季度计算完成，有效样本数: {cash_efficiency_quarterly.notna().sum()}")
            
            # 第二步：将计算结果扩展到交易日频率（只扩展一次）
            logger.info("将计算结果扩展到交易日频率...")
            
            from factors.base.time_series_processor import TimeSeriesProcessor
            
            # 准备现金流效率季度数据
            efficiency_df = pd.DataFrame({'cash_efficiency': cash_efficiency_quarterly})
            
            # 获取交易日期（优先使用统一的交易日期工具）
            try:
                from utils.trading_dates_utils import get_trading_dates
                # 从BP数据推断日期范围
                bp_dates = bp.index.get_level_values(0).unique()
                start_date = bp_dates.min().strftime('%Y-%m-%d')
                end_date = bp_dates.max().strftime('%Y-%m-%d')
                trading_dates = get_trading_dates(start_date, end_date)
                logger.info(f"使用统一交易日历，获取 {len(trading_dates)} 个交易日")
            except Exception as e:
                logger.warning(f"使用统一交易日历失败，回退到BP数据提取: {e}")
                trading_dates = bp.index.get_level_values(0).unique().sort_values()
            
            # 准备发布日期数据
            if 'release_dates' in data:
                release_dates_df = data['release_dates']
            else:
                release_dates_df = self._create_default_release_dates(efficiency_df)
            
            # 🔥 使用优化的扩展方法提升性能
            try:
                from factors.base.optimized_time_series_processor import OptimizedTimeSeriesProcessor
                logger.info("使用优化的向量化扩展方法...")
                cash_efficiency_daily = OptimizedTimeSeriesProcessor.expand_to_daily_vectorized(
                    efficiency_df,
                    release_dates_df,
                    trading_dates
                )
            except Exception as opt_error:
                logger.warning(f"优化方法失败，回退到原始方法: {opt_error}")
                # 回退到原始方法
                cash_efficiency_daily = TimeSeriesProcessor.expand_to_daily(
                    efficiency_df,
                    release_dates_df,
                    trading_dates
                )
            
            if cash_efficiency_daily.empty:
                logger.error("现金流效率数据扩展到日频失败")
                return pd.Series()
            
            # 提取日频现金流效率数据
            cash_efficiency_series = cash_efficiency_daily['cash_efficiency']
            
            logger.info(f"扩展完成 - 现金流效率日频数据: {len(cash_efficiency_series)}")
            logger.info(f"  - 现金流效率索引类型: {type(cash_efficiency_series.index)}")
            logger.info(f"  - 现金流效率索引名称: {cash_efficiency_series.index.names}")
            logger.info(f"  - BP数据长度: {len(bp)}")
            logger.info(f"  - BP索引类型: {type(bp.index)}")
            logger.info(f"  - BP索引名称: {bp.index.names}")
            
            # 检查索引兼容性
            if hasattr(cash_efficiency_series.index, 'names') and hasattr(bp.index, 'names'):
                if cash_efficiency_series.index.names != bp.index.names:
                    logger.warning(f"索引名称不匹配: {cash_efficiency_series.index.names} vs {bp.index.names}")
            
            # 采样检查数据质量
            logger.info(f"现金流效率有效值比例: {cash_efficiency_series.notna().mean():.2%}")
            logger.info(f"BP有效值比例: {bp.notna().mean():.2%}")
            
            # 第三步：与日频BP数据对齐并计算最终因子
            logger.info("与BP因子对齐并计算最终结果...")
            
            aligned_data = self._align_data([cash_efficiency_series, bp])
            if not aligned_data:
                logger.error("与BP数据对齐失败")
                return pd.Series()
            
            cash_efficiency_aligned, bp_aligned = aligned_data
            
            # 计算最终因子：现金流效率 / BP
            factor_result = self._safe_divide(cash_efficiency_aligned, bp_aligned, 'cash_efficiency', 'BP')
            
            # 数据清理和异常值处理
            factor_result = self.handle_outliers(factor_result, method='winsorize', quantiles=(0.01, 0.99))
            factor_result = self.fill_missing_values(factor_result, method='median')
            
            factor_result.name = self.factor_name
            
            logger.info(f"✅ 现金流效率比率因子计算完成，有效样本数: {factor_result.notna().sum()}")
            
            return factor_result
            
        except Exception as e:
            logger.error(f"计算现金流效率比率因子失败: {e}")
            return pd.Series()
    
    def _get_field_data(self, data: pd.DataFrame, primary_field: str, mapping_key: str) -> pd.Series:
        """获取字段数据，支持字段映射"""
        if primary_field in data.columns:
            return data[primary_field]
        
        # 尝试使用映射
        mapped_field = self.get_mapped_column(mapping_key)
        if mapped_field in data.columns:
            logger.info(f"使用字段映射: {mapping_key} -> {mapped_field}")
            return data[mapped_field]
        
        raise ValueError(f"无法找到字段: {primary_field} 或其映射字段")
    
    def _align_data(self, series_list):
        """对齐多个Series数据"""
        try:
            # 获取公共索引
            common_index = series_list[0].index
            for series in series_list[1:]:
                common_index = common_index.intersection(series.index)
            
            if len(common_index) < 100:
                logger.warning(f"对齐后数据量较少: {len(common_index)}")
                return None
            
            # 对齐所有数据
            aligned_series = []
            for series in series_list:
                aligned = series.loc[common_index]
                aligned_series.append(aligned)
            
            logger.info(f"数据对齐完成，公共样本数: {len(common_index)}")
            return aligned_series
            
        except Exception as e:
            logger.error(f"数据对齐失败: {e}")
            return None
    
    def _safe_divide(self, numerator: pd.Series, denominator: pd.Series, 
                     num_name: str = "numerator", den_name: str = "denominator") -> pd.Series:
        """安全除法，处理除零和异常值"""
        try:
            # 处理除零情况
            denominator_safe = denominator.replace(0, np.nan)
            
            # 过滤极值
            denominator_safe = denominator_safe.where(
                (denominator_safe.abs() > 1e-6) & (denominator_safe.abs() < 1e10)
            )
            
            result = numerator / denominator_safe
            
            # 记录统计信息
            valid_count = result.notna().sum()
            zero_count = (denominator == 0).sum()
            
            logger.info(f"{num_name} / {den_name}: 有效值 {valid_count}, 分母为零 {zero_count}")
            
            return result
            
        except Exception as e:
            logger.error(f"安全除法计算失败: {e}")
            return pd.Series()
    
    
    def _create_default_release_dates(self, financial_df: pd.DataFrame) -> pd.DataFrame:
        """
        创建默认的发布日期数据
        假设财务数据在报告期结束后3个月内发布
        """
        try:
            # 提取报告日期和股票代码
            report_dates = financial_df.index.get_level_values(0)
            stock_codes = financial_df.index.get_level_values(1)
            
            # 创建发布日期：报告期结束后90天（约3个月）
            release_dates = report_dates + pd.Timedelta(days=90)
            
            # 创建发布日期DataFrame
            release_df = pd.DataFrame({
                'ReleasedDates': release_dates
            }, index=financial_df.index)
            
            logger.info(f"创建默认发布日期，样本数: {len(release_df)}")
            return release_df
            
        except Exception as e:
            logger.error(f"创建默认发布日期失败: {e}")
            # 返回空DataFrame，将使用报告期作为发布日期
            return pd.DataFrame()
    
    def _get_trading_calendar(self, start_date, end_date):
        """
        获取交易日历（如果需要的话）
        这里可以连接到交易所日历API或使用本地数据
        """
        try:
            # 简化实现：生成工作日序列，排除周末
            # 实际使用中应该使用准确的交易日历
            dates = pd.bdate_range(start=start_date, end=end_date)
            logger.info(f"生成交易日历: {len(dates)} 个交易日")
            return dates
        except Exception as e:
            logger.error(f"获取交易日历失败: {e}")
            return pd.DatetimeIndex([])


# 因子工厂函数，方便调用
def create_cashflow_efficiency_ratio() -> CashflowEfficiencyRatio:
    """创建现金流效率比率因子实例"""
    return CashflowEfficiencyRatio()


# 注册因子到元数据系统
def register_factor_metadata():
    """注册因子到元数据系统"""
    try:
        from factors.meta import get_factor_registry, FactorType, NeutralizationCategory
        
        registry = get_factor_registry()
        
        registry.register_factor(
            name="CashflowEfficiencyRatio",
            factor_type=FactorType.DERIVED,
            description="现金流效率比率：(财务费用+折旧)/销售现金流/净资产市值比",
            formula="((FIN_EXP_CS + DEPR_FA_COGA_DPBA) / CASH_RECP_SG_AND_RS) / BP",
            neutralization_category=NeutralizationCategory.OPTIONAL_NEUTRALIZE,
            generator="CashflowEfficiencyRatio",
            tags=["custom", "mixed", "cashflow", "efficiency"],
            category="financial_efficiency",
            priority=5
        )
        
        logger.info("✅ 因子元数据注册成功")
        
    except Exception as e:
        logger.warning(f"因子元数据注册失败: {e}")


if __name__ == "__main__":
    # 示例使用
    logging.basicConfig(level=logging.INFO)
    
    # 注册因子元数据
    register_factor_metadata()
    
    print("现金流效率比率因子开发完成！")
    print("使用方法:")
    print("1. factor = create_cashflow_efficiency_ratio()")
    print("2. result = factor.calculate(data)")