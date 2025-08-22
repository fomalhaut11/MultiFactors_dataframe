#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成测试因子脚本
生成并保存 ROE_ttm, BP, SUE 等因子到本地
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import logging

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 添加项目路径
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from core.config_manager import get_path, get_config
from data.fetcher import BasicDataLocalization
from factors.financial.pure_financial_factors import PureFinancialFactorCalculator
from factors.base import TimeSeriesProcessor
import pickle


def load_auxiliary_data():
    """加载辅助数据"""
    logger.info("加载辅助数据...")
    
    data_path = Path(project_root) / 'data' / 'auxiliary'
    
    # 加载财务数据
    financial_file = data_path / 'FinancialData_unified.pkl'
    if financial_file.exists():
        financial_data = pd.read_pickle(financial_file)
        logger.info(f"加载财务数据: shape={financial_data.shape}")
    else:
        logger.error(f"财务数据文件不存在: {financial_file}")
        return None, None, None, None
    
    # 加载市值数据
    market_cap_file = Path(get_path('data_root')) / 'MarketCap.pkl'
    if market_cap_file.exists():
        market_cap = pd.read_pickle(market_cap_file)
        logger.info(f"加载市值数据: shape={market_cap.shape}")
    else:
        logger.warning("市值数据文件不存在，尝试其他路径")
        # 尝试其他可能的路径
        alt_paths = [
            Path(get_path('data_root')) / 'LogMarketCap.pkl',
            data_path / 'MarketCap.pkl',
        ]
        market_cap = None
        for alt_path in alt_paths:
            if alt_path.exists():
                market_cap = pd.read_pickle(alt_path)
                logger.info(f"从 {alt_path} 加载市值数据")
                break
        
        if market_cap is None:
            logger.error("无法找到市值数据")
            return financial_data, None, None, None
    
    # 加载发布日期
    release_dates_file = data_path / 'ReleaseDates.pkl'
    if release_dates_file.exists():
        release_dates = pd.read_pickle(release_dates_file)
        logger.info(f"加载发布日期: shape={release_dates.shape}")
    else:
        logger.error(f"发布日期文件不存在: {release_dates_file}")
        return financial_data, market_cap, None, None
    
    # 加载交易日期
    trading_dates_file = data_path / 'TradingDates.pkl'
    if trading_dates_file.exists():
        trading_dates = pd.read_pickle(trading_dates_file)
        logger.info(f"加载交易日期: {len(trading_dates)} days")
    else:
        # 尝试从StockData路径加载
        alt_path = Path(get_path('data_root')) / 'TradingDates.pkl'
        if alt_path.exists():
            trading_dates = pd.read_pickle(alt_path)
            logger.info(f"从 {alt_path} 加载交易日期")
        else:
            logger.error("无法找到交易日期数据")
            return financial_data, market_cap, release_dates, None
    
    return financial_data, market_cap, release_dates, trading_dates


def generate_roe_ttm(financial_data, release_dates, trading_dates):
    """生成ROE_ttm因子"""
    logger.info("="*60)
    logger.info("生成ROE_ttm因子...")
    
    try:
        # 创建因子计算器
        calculator = PureFinancialFactorCalculator()
        
        # 计算ROE_ttm
        logger.info("计算ROE_ttm...")
        roe_ttm = calculator.calculate_ROE_ttm(
            financial_data=financial_data,
            release_dates=release_dates,
            trading_dates=trading_dates,
            expand_to_daily=True
        )
        
        if roe_ttm is None or roe_ttm.empty:
            logger.error("ROE_ttm计算结果为空")
            return None
        
        logger.info(f"ROE_ttm计算完成: shape={roe_ttm.shape}")
        logger.info(f"  日期范围: {roe_ttm.index.get_level_values(0).min()} 至 {roe_ttm.index.get_level_values(0).max()}")
        logger.info(f"  股票数量: {len(roe_ttm.index.get_level_values(1).unique())}")
        logger.info(f"  非空值比例: {(~roe_ttm.isna()).mean():.2%}")
        
        # 基本统计
        logger.info(f"  均值: {roe_ttm.mean():.4f}")
        logger.info(f"  中位数: {roe_ttm.median():.4f}")
        logger.info(f"  标准差: {roe_ttm.std():.4f}")
        
        return roe_ttm
        
    except Exception as e:
        logger.error(f"生成ROE_ttm失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def generate_bp(financial_data, market_cap, release_dates, trading_dates):
    """生成BP因子（Book-to-Price）"""
    logger.info("="*60)
    logger.info("生成BP因子...")
    
    try:
        # 获取账面价值（净资产）
        if 'EQY_BELONGTO_PARCOMSH' in financial_data.columns:
            book_value = financial_data['EQY_BELONGTO_PARCOMSH']
        elif 'TOT_SHRHLDR_EQY_INCL_MIN_INT' in financial_data.columns:
            book_value = financial_data['TOT_SHRHLDR_EQY_INCL_MIN_INT']
        else:
            logger.error("找不到净资产字段")
            return None
        
        logger.info(f"账面价值数据: shape={book_value.shape}, 非空值比例={book_value.notna().mean():.2%}")
        
        # 扩展到日频
        processor = TimeSeriesProcessor()
        book_value_daily = processor.expand_to_daily(
            book_value.to_frame('book_value'),
            release_dates,
            trading_dates
        )['book_value']
        
        logger.info(f"日频账面价值: shape={book_value_daily.shape}")
        
        # 处理市值数据
        if isinstance(market_cap, pd.DataFrame):
            if 'MarketCap' in market_cap.columns:
                market_cap = market_cap['MarketCap']
            else:
                market_cap = market_cap.iloc[:, 0]
        
        # 如果是对数市值，转换回原始市值
        if market_cap.name == 'LogMarketCap' or market_cap.median() < 100:
            logger.info("检测到对数市值，转换为原始市值")
            market_cap = np.exp(market_cap)
        
        logger.info(f"市值数据: shape={market_cap.shape}, 非空值比例={market_cap.notna().mean():.2%}")
        
        # 对齐数据
        common_index = book_value_daily.index.intersection(market_cap.index)
        logger.info(f"共同索引数量: {len(common_index)}")
        
        if len(common_index) == 0:
            logger.error("账面价值和市值没有共同索引")
            return None
        
        book_value_aligned = book_value_daily.loc[common_index]
        market_cap_aligned = market_cap.loc[common_index]
        
        # 计算BP = 账面价值 / 市值
        bp = book_value_aligned / market_cap_aligned
        
        # 处理异常值
        bp = bp.replace([np.inf, -np.inf], np.nan)
        
        # 基本清理：BP通常在0-10之间，极端值可能是数据错误
        bp = bp.where((bp > 0) & (bp < 10), np.nan)
        
        logger.info(f"BP因子计算完成: shape={bp.shape}")
        logger.info(f"  日期范围: {bp.index.get_level_values(0).min()} 至 {bp.index.get_level_values(0).max()}")
        logger.info(f"  股票数量: {len(bp.index.get_level_values(1).unique())}")
        logger.info(f"  非空值比例: {(~bp.isna()).mean():.2%}")
        
        # 基本统计
        logger.info(f"  均值: {bp.mean():.4f}")
        logger.info(f"  中位数: {bp.median():.4f}")
        logger.info(f"  标准差: {bp.std():.4f}")
        
        return bp
        
    except Exception as e:
        logger.error(f"生成BP因子失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def generate_sue(financial_data, release_dates, trading_dates):
    """生成SUE因子（Standardized Unexpected Earnings）"""
    logger.info("="*60)
    logger.info("生成SUE因子...")
    
    try:
        # 获取净利润数据
        if 'DEDUCTEDPROFIT' in financial_data.columns:
            earnings = financial_data['DEDUCTEDPROFIT']  # 扣非净利润
        elif 'NET_PROFIT_INCL_MIN_INT_INC' in financial_data.columns:
            earnings = financial_data['NET_PROFIT_INCL_MIN_INT_INC']  # 净利润
        else:
            logger.error("找不到净利润字段")
            return None
        
        logger.info(f"净利润数据: shape={earnings.shape}")
        
        # 计算单季度净利润
        processor = TimeSeriesProcessor()
        single_quarter_earnings = processor.calculate_single_quarter(
            earnings.to_frame('earnings').join(financial_data[['d_quarter']]),
            quarter_col='d_quarter'
        )['earnings']
        
        logger.info(f"单季度净利润: shape={single_quarter_earnings.shape}")
        
        # 计算SUE: (当期净利润 - 去年同期净利润) / 过去8个季度净利润标准差
        sue_list = []
        
        for stock in single_quarter_earnings.index.get_level_values(1).unique():
            stock_data = single_quarter_earnings.xs(stock, level=1)
            
            if len(stock_data) < 12:  # 需要足够的历史数据
                continue
            
            # 计算同比变化
            yoy_change = stock_data - stock_data.shift(4)
            
            # 计算过去8个季度的标准差（滚动）
            rolling_std = stock_data.rolling(window=8, min_periods=6).std()
            
            # 计算SUE
            stock_sue = yoy_change / rolling_std.shift(1)  # 使用上期的标准差
            stock_sue = stock_sue.replace([np.inf, -np.inf], np.nan)
            
            # 重建多重索引
            stock_sue_df = stock_sue.to_frame('SUE')
            stock_sue_df['StockCodes'] = stock
            stock_sue_df = stock_sue_df.reset_index()
            stock_sue_df = stock_sue_df.set_index(['ReportDates', 'StockCodes'])['SUE']
            
            sue_list.append(stock_sue_df)
        
        if not sue_list:
            logger.error("没有足够的数据计算SUE")
            return None
        
        # 合并所有股票的SUE
        sue = pd.concat(sue_list)
        sue = sue.sort_index()
        
        logger.info(f"SUE计算完成（季度频率）: shape={sue.shape}")
        
        # 扩展到日频
        sue_daily = processor.expand_to_daily(
            sue.to_frame('SUE'),
            release_dates,
            trading_dates
        )['SUE']
        
        # 处理异常值：SUE通常在-10到10之间
        sue_daily = sue_daily.where((sue_daily > -10) & (sue_daily < 10), np.nan)
        
        logger.info(f"SUE因子计算完成（日频）: shape={sue_daily.shape}")
        logger.info(f"  日期范围: {sue_daily.index.get_level_values(0).min()} 至 {sue_daily.index.get_level_values(0).max()}")
        logger.info(f"  股票数量: {len(sue_daily.index.get_level_values(1).unique())}")
        logger.info(f"  非空值比例: {(~sue_daily.isna()).mean():.2%}")
        
        # 基本统计
        logger.info(f"  均值: {sue_daily.mean():.4f}")
        logger.info(f"  中位数: {sue_daily.median():.4f}")
        logger.info(f"  标准差: {sue_daily.std():.4f}")
        
        return sue_daily
        
    except Exception as e:
        logger.error(f"生成SUE因子失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def save_factor(factor_data, factor_name):
    """保存因子到本地"""
    if factor_data is None or factor_data.empty:
        logger.error(f"因子 {factor_name} 为空，跳过保存")
        return False
    
    # 保存路径
    save_path = Path(get_path('raw_factors'))
    save_path.mkdir(parents=True, exist_ok=True)
    
    # 保存为pickle
    file_path = save_path / f"{factor_name}.pkl"
    factor_data.to_pickle(file_path)
    logger.info(f"因子 {factor_name} 已保存到: {file_path}")
    
    # 同时保存一份到项目的factor_output目录（便于查看）
    output_path = Path(project_root) / 'factor_output'
    output_path.mkdir(exist_ok=True)
    
    # 保存pickle
    output_file = output_path / f"{factor_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
    factor_data.to_pickle(output_file)
    
    # 保存CSV样本（前1000条）
    sample_file = output_path / f"{factor_name}_sample.csv"
    factor_data.head(1000).to_csv(sample_file)
    logger.info(f"样本数据已保存到: {sample_file}")
    
    return True


def test_factor_with_single_test(factor_name, begin_date='2023-01-01', end_date='2024-06-30'):
    """使用单因子测试模块测试因子"""
    logger.info(f"\n测试因子 {factor_name}...")
    
    try:
        from factors.tester import SingleFactorTestPipeline
        
        pipeline = SingleFactorTestPipeline()
        result = pipeline.run(
            factor_name,
            save_result=True,
            begin_date=begin_date,
            end_date=end_date,
            group_nums=5,
            netral_base=False,  # 先不做中性化
            use_industry=False   # 先不用行业
        )
        
        if result.ic_result:
            logger.info(f"  IC均值: {result.ic_result.ic_mean:.4f}")
            logger.info(f"  ICIR: {result.ic_result.icir:.4f}")
            logger.info(f"  Rank IC: {result.ic_result.rank_ic_mean:.4f}")
        
        if result.group_result:
            logger.info(f"  单调性: {result.group_result.monotonicity_score:.4f}")
        
        if result.errors:
            logger.error(f"  测试错误: {result.errors}")
        
        return result
        
    except Exception as e:
        logger.error(f"测试因子 {factor_name} 失败: {e}")
        return None


def main():
    """主函数"""
    logger.info("="*60)
    logger.info("开始生成测试因子")
    logger.info(f"时间: {datetime.now()}")
    logger.info("="*60)
    
    # 加载数据
    financial_data, market_cap, release_dates, trading_dates = load_auxiliary_data()
    
    if financial_data is None:
        logger.error("无法加载必要的数据，退出")
        return
    
    # 生成的因子列表
    generated_factors = []
    
    # 1. 生成ROE_ttm
    logger.info("\n" + "="*60)
    logger.info("1. 生成ROE_ttm因子")
    roe_ttm = generate_roe_ttm(financial_data, release_dates, trading_dates)
    if roe_ttm is not None:
        if save_factor(roe_ttm, 'ROE_ttm'):
            generated_factors.append('ROE_ttm')
    
    # 2. 生成BP因子
    if market_cap is not None:
        logger.info("\n" + "="*60)
        logger.info("2. 生成BP因子")
        bp = generate_bp(financial_data, market_cap, release_dates, trading_dates)
        if bp is not None:
            if save_factor(bp, 'BP'):
                generated_factors.append('BP')
    else:
        logger.warning("跳过BP因子生成（缺少市值数据）")
    
    # 3. 生成SUE因子
    logger.info("\n" + "="*60)
    logger.info("3. 生成SUE因子")
    sue = generate_sue(financial_data, release_dates, trading_dates)
    if sue is not None:
        if save_factor(sue, 'SUE'):
            generated_factors.append('SUE')
    
    # 汇总
    logger.info("\n" + "="*60)
    logger.info("因子生成完成")
    logger.info(f"成功生成 {len(generated_factors)} 个因子: {generated_factors}")
    
    # 可选：立即测试这些因子
    if generated_factors:
        logger.info("\n" + "="*60)
        logger.info("开始测试生成的因子...")
        
        test_results = {}
        for factor_name in generated_factors:
            result = test_factor_with_single_test(factor_name)
            test_results[factor_name] = result
        
        # 打印测试汇总
        logger.info("\n" + "="*60)
        logger.info("测试结果汇总:")
        for factor_name, result in test_results.items():
            if result and result.ic_result:
                logger.info(f"{factor_name}: IC={result.ic_result.ic_mean:.4f}, ICIR={result.ic_result.icir:.4f}")
            else:
                logger.info(f"{factor_name}: 测试失败")
    
    logger.info("\n" + "="*60)
    logger.info("全部完成！")
    logger.info("="*60)


if __name__ == "__main__":
    main()