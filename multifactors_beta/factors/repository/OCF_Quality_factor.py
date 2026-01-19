#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
经营现金流质量因子 (Operating Cash Flow Quality)

因子定义：
    OCF_Quality = 经营活动现金流量净额(TTM) / 净利润(TTM)

投资逻辑：
    - 比率 > 1.2: 盈利质量优秀，现金充裕
    - 比率 0.8-1.2: 正常水平
    - 比率 < 0.8: 警惕，盈利可能不实或应收账款高
    - 比率 < 0: 亏损或现金流为负，危险信号

    该因子用于识别"纸面富贵"的公司，高比率表明公司盈利有现金支撑，
    低比率可能存在财务造假或应收账款管理问题。

数据需求：
    - 经营活动现金流量净额: NET_CASH_FLOWS_OPER_ACT (现金流量表)
    - 扣非净利润: DEDUCTEDPROFIT (利润表)

作者: Claude Code
创建时间: 2026-01-19
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import logging

# 添加项目路径
project_root = Path(__file__).parents[2]
sys.path.insert(0, str(project_root))

# 导入项目工具（严格遵守防重复造轮规范）
from factors.generators import (
    calculate_ttm,
    expand_to_daily_vectorized
)
from config import get_config
# 项目使用pandas内置的pickle功能

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OCF_QualityFactor:
    """
    经营现金流质量因子计算器

    严格遵守项目规范：
    1. 使用factors.generators工具集
    2. 不重复实现TTM计算
    3. 输出MultiIndex[TradingDates, StockCodes]格式
    """

    def __init__(self):
        """初始化因子计算器"""
        self.factor_name = 'OCF_Quality'
        self.data_root = get_config('main.paths.data_root')
        self.auxiliary_path = get_config('main.paths.auxiliary_data')
        self.factor_save_path = get_config('main.paths.raw_factors')

    def load_data(self):
        """
        加载所需数据

        Returns:
            tuple: (financial_data, release_dates, trading_dates)
        """
        logger.info("加载财务数据...")

        # 加载统一财务数据
        financial_path = Path(self.auxiliary_path) / 'FinancialData_unified.pkl'
        financial_data = pd.read_pickle(str(financial_path))

        # 加载发布日期
        release_dates_path = Path(self.auxiliary_path) / 'ReleaseDates.pkl'
        release_dates = pd.read_pickle(str(release_dates_path))

        # 加载交易日期
        trading_dates_path = Path(self.auxiliary_path) / 'TradingDates.pkl'
        trading_dates = pd.read_pickle(str(trading_dates_path))

        logger.info(f"财务数据形状: {financial_data.shape}")
        logger.info(f"发布日期形状: {release_dates.shape}")
        logger.info(f"交易日数量: {len(trading_dates)}")

        return financial_data, release_dates, trading_dates

    def calculate(self, financial_data, release_dates, trading_dates):
        """
        计算经营现金流质量因子

        Parameters:
            financial_data: 财务数据 DataFrame
            release_dates: 发布日期 DataFrame
            trading_dates: 交易日期 Index

        Returns:
            pd.Series: 因子值，MultiIndex[TradingDates, StockCodes]格式
        """
        logger.info(f"开始计算{self.factor_name}因子...")

        # 1. 提取所需字段
        logger.info("提取经营现金流和净利润字段...")
        required_fields = ['NET_CASH_FLOWS_OPER_ACT', 'DEDUCTEDPROFIT', 'd_quarter']

        # 检查字段是否存在
        missing_fields = [f for f in required_fields if f not in financial_data.columns]
        if missing_fields:
            raise ValueError(f"缺少字段: {missing_fields}")

        ocf_data = financial_data[['NET_CASH_FLOWS_OPER_ACT', 'd_quarter']].copy()
        profit_data = financial_data[['DEDUCTEDPROFIT', 'd_quarter']].copy()

        # 2. 计算TTM（使用项目工具，禁止重复实现）
        logger.info("计算经营现金流TTM...")
        ocf_ttm = calculate_ttm(ocf_data)

        logger.info("计算净利润TTM...")
        profit_ttm = calculate_ttm(profit_data)

        logger.info(f"经营现金流TTM非空值: {ocf_ttm.count()[0]}/{len(ocf_ttm)}")
        logger.info(f"净利润TTM非空值: {profit_ttm.count()[0]}/{len(profit_ttm)}")

        # 3. 计算比率
        logger.info("计算现金流质量比率...")

        # 合并两个序列
        combined = pd.DataFrame({
            'OCF_TTM': ocf_ttm.iloc[:, 0],
            'Profit_TTM': profit_ttm.iloc[:, 0]
        })

        # 计算比率，处理异常情况
        factor_value = combined['OCF_TTM'] / combined['Profit_TTM']

        # 处理异常值
        # 分母为0或接近0的情况
        factor_value = factor_value.replace([np.inf, -np.inf], np.nan)

        # 过滤极端值（保留-10到10之间的值）
        factor_value = factor_value.clip(-10, 10)

        logger.info(f"因子计算完成，非空值: {factor_value.count()}/{len(factor_value)}")
        logger.info(f"因子统计: min={factor_value.min():.2f}, max={factor_value.max():.2f}, "
                   f"mean={factor_value.mean():.2f}, median={factor_value.median():.2f}")

        # 4. 扩展到日频（使用项目工具）
        logger.info("扩展到日频...")
        factor_daily = expand_to_daily_vectorized(
            factor_value.to_frame(self.factor_name),
            release_dates,
            trading_dates
        )

        logger.info(f"日频因子形状: {factor_daily.shape}")
        logger.info(f"日频因子非空值: {factor_daily.count()}/{len(factor_daily)} "
                   f"({factor_daily.count()/len(factor_daily):.1%})")

        return factor_daily

    def save(self, factor_data):
        """
        保存因子数据

        Parameters:
            factor_data: 因子数据 Series
        """
        save_path = Path(self.factor_save_path) / f'{self.factor_name}.pkl'
        save_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"保存因子到: {save_path}")
        factor_data.to_pickle(str(save_path))

        logger.info(f"✓ 因子保存成功: {self.factor_name}")

    def run(self):
        """
        执行完整的因子计算流程

        Returns:
            pd.Series: 计算好的因子数据
        """
        try:
            # 加载数据
            financial_data, release_dates, trading_dates = self.load_data()

            # 计算因子
            factor_data = self.calculate(financial_data, release_dates, trading_dates)

            # 保存因子
            self.save(factor_data)

            logger.info(f"\n{'='*60}")
            logger.info(f"✅ {self.factor_name}因子计算完成！")
            logger.info(f"{'='*60}")

            return factor_data

        except Exception as e:
            logger.error(f"❌ 因子计算失败: {e}")
            import traceback
            traceback.print_exc()
            raise


def main():
    """主函数"""
    print("\n" + "="*70)
    print("经营现金流质量因子 (OCF_Quality) 计算")
    print("="*70)

    calculator = OCF_QualityFactor()
    factor_data = calculator.run()

    # 打印样本数据
    print(f"\n因子数据样本（前10个非空值）：")
    non_null = factor_data[factor_data.notna()]
    if len(non_null) > 0:
        print(non_null.head(10))

    return factor_data


if __name__ == "__main__":
    factor = main()
