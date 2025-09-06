#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速生成BP因子
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
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import get_config


def generate_bp_simple():
    """简单生成BP因子"""
    logger.info("开始生成BP因子...")
    
    try:
        # 1. 加载市值数据
        market_cap_path = Path(get_config('main.paths.data_root')) / 'LogMarketCap.pkl'
        if not market_cap_path.exists():
            market_cap_path = Path(get_config('main.paths.data_root')) / 'MarketCap.pkl'
        
        logger.info(f"加载市值数据: {market_cap_path}")
        market_cap = pd.read_pickle(market_cap_path)
        
        # 如果是DataFrame，取第一列
        if isinstance(market_cap, pd.DataFrame):
            market_cap = market_cap.iloc[:, 0]
        
        # 如果是对数市值，转换回原始值
        if market_cap.median() < 100:
            logger.info("转换对数市值为原始市值")
            market_cap = np.exp(market_cap)
        
        logger.info(f"市值数据: {market_cap.shape}")
        
        # 2. 加载净资产数据
        financial_path = Path(project_root) / 'data' / 'auxiliary' / 'FinancialData_unified.pkl'
        logger.info(f"加载财务数据: {financial_path}")
        financial_data = pd.read_pickle(financial_path)
        
        # 获取净资产
        if 'EQY_BELONGTO_PARCOMSH' in financial_data.columns:
            book_value = financial_data['EQY_BELONGTO_PARCOMSH']
        elif 'TOT_SHRHLDR_EQY_INCL_MIN_INT' in financial_data.columns:
            book_value = financial_data['TOT_SHRHLDR_EQY_INCL_MIN_INT']
        else:
            raise ValueError("找不到净资产字段")
        
        logger.info(f"净资产数据: {book_value.shape}")
        
        # 3. 简单扩展到日频（使用前向填充）
        # 获取所有交易日期
        dates = market_cap.index.get_level_values(0).unique().sort_values()
        
        # 创建日频的book value
        book_value_daily_list = []
        for stock in book_value.index.get_level_values(1).unique()[:500]:  # 先处理前500只股票
            if stock not in market_cap.index.get_level_values(1):
                continue
            
            try:
                stock_bv = book_value.xs(stock, level=1)
                # 重采样到日频
                stock_bv_daily = stock_bv.reindex(dates, method='ffill')
                
                # 添加股票代码索引
                stock_bv_daily = pd.Series(
                    stock_bv_daily.values,
                    index=pd.MultiIndex.from_product([dates, [stock]], names=['TradingDates', 'StockCodes'])
                )
                book_value_daily_list.append(stock_bv_daily)
            except:
                continue
        
        if not book_value_daily_list:
            raise ValueError("无法生成日频净资产数据")
        
        book_value_daily = pd.concat(book_value_daily_list)
        logger.info(f"日频净资产: {book_value_daily.shape}")
        
        # 4. 计算BP = 净资产 / 市值
        common_index = book_value_daily.index.intersection(market_cap.index)
        bp = book_value_daily.loc[common_index] / market_cap.loc[common_index]
        
        # 处理异常值
        bp = bp.replace([np.inf, -np.inf], np.nan)
        bp = bp.where((bp > 0) & (bp < 10), np.nan)
        
        logger.info(f"BP因子生成完成: {bp.shape}")
        logger.info(f"  非空值比例: {(~bp.isna()).mean():.2%}")
        logger.info(f"  均值: {bp.mean():.4f}")
        logger.info(f"  中位数: {bp.median():.4f}")
        
        # 5. 保存
        save_path = Path(get_config('main.paths.raw_factors'))
        save_path.mkdir(parents=True, exist_ok=True)
        
        file_path = save_path / 'BP.pkl'
        bp.to_pickle(file_path)
        logger.info(f"BP因子已保存到: {file_path}")
        
        # 保存一份样本到factor_output
        output_path = Path(project_root) / 'factor_output'
        output_path.mkdir(exist_ok=True)
        
        sample_file = output_path / f"BP_sample_{datetime.now().strftime('%Y%m%d')}.csv"
        bp.head(1000).to_csv(sample_file)
        logger.info(f"样本已保存到: {sample_file}")
        
        return bp
        
    except Exception as e:
        logger.error(f"生成BP因子失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_bp_factor():
    """测试BP因子"""
    logger.info("\n测试BP因子...")
    
    try:
        from factors.tester import SingleFactorTestPipeline
        
        pipeline = SingleFactorTestPipeline()
        result = pipeline.run(
            'BP',
            save_result=True,
            begin_date='2024-01-01',
            end_date='2024-06-30',
            group_nums=5,
            netral_base=False,
            use_industry=False
        )
        
        if result.ic_result:
            logger.info(f"IC均值: {result.ic_result.ic_mean:.4f}")
            logger.info(f"ICIR: {result.ic_result.icir:.4f}")
        
        return result
        
    except Exception as e:
        logger.error(f"测试失败: {e}")
        return None


def main():
    """主函数"""
    logger.info("="*60)
    logger.info("BP因子生成程序")
    logger.info("="*60)
    
    # 生成BP因子
    bp = generate_bp_simple()
    
    if bp is not None:
        logger.info("\nBP因子生成成功！")
        
        # 测试因子
        logger.info("\n开始测试BP因子...")
        result = test_bp_factor()
        
        if result:
            logger.info("测试完成！")
    else:
        logger.error("BP因子生成失败")
    
    logger.info("\n程序结束")


if __name__ == "__main__":
    main()