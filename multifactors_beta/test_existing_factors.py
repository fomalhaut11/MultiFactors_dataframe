#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试已存在的因子并使用筛选模块
"""

import sys
import os
from pathlib import Path
import pandas as pd
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

from factors.tester import SingleFactorTestPipeline
from factors.analyzer import FactorScreener


def test_single_factor(factor_name, begin_date='2024-01-01', end_date='2024-12-31'):
    """测试单个因子"""
    logger.info(f"\n{'='*60}")
    logger.info(f"测试因子: {factor_name}")
    logger.info('='*60)
    
    try:
        pipeline = SingleFactorTestPipeline()
        result = pipeline.run(
            factor_name,
            save_result=True,
            begin_date=begin_date,
            end_date=end_date,
            group_nums=5,
            netral_base=False,
            use_industry=False
        )
        
        if result.ic_result:
            logger.info(f"✓ 测试成功!")
            logger.info(f"  IC均值: {result.ic_result.ic_mean:.4f}")
            logger.info(f"  ICIR: {result.ic_result.icir:.4f}")
            logger.info(f"  Rank IC: {result.ic_result.rank_ic_mean:.4f}")
        
        if result.group_result:
            logger.info(f"  单调性: {result.group_result.monotonicity_score:.4f}")
        
        if result.performance_metrics:
            logger.info(f"  夏普比率: {result.performance_metrics.get('long_short_sharpe', 0):.4f}")
        
        if result.errors:
            logger.error(f"  错误: {result.errors}")
        
        return result
        
    except Exception as e:
        logger.error(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_factor_screening():
    """测试因子筛选功能"""
    logger.info(f"\n{'='*60}")
    logger.info("测试因子筛选模块")
    logger.info('='*60)
    
    try:
        # 创建筛选器
        screener = FactorScreener()
        
        # 加载所有测试结果
        logger.info("\n1. 加载历史测试结果...")
        results = screener.load_all_results()
        logger.info(f"   加载了 {len(results)} 个因子的测试结果")
        
        if results:
            # 显示部分因子
            logger.info("   已有因子:")
            for i, factor_name in enumerate(list(results.keys())[:10]):
                logger.info(f"   {i+1}. {factor_name}")
        
        # 使用默认标准筛选
        logger.info("\n2. 筛选高质量因子...")
        selected = screener.screen_factors({
            'ic_mean_min': 0.01,
            'icir_min': 0.3,
            'monotonicity_min': 0.4
        })
        
        logger.info(f"   筛选出 {len(selected)} 个因子")
        for factor in selected[:10]:
            score = screener.get_factor_score(factor)
            logger.info(f"   - {factor}: 评分 {score:.2f}")
        
        # 获取排名
        logger.info("\n3. 因子排名（按ICIR）...")
        ranking = screener.get_factor_ranking('icir', top_n=10)
        print(ranking.to_string())
        
        # 生成报告
        logger.info("\n4. 生成筛选报告...")
        report = screener.generate_screening_report(
            output_path='factor_screening_report.csv',
            top_n=20
        )
        logger.info(f"   报告已保存，包含 {len(report)} 个因子")
        
        return screener
        
    except Exception as e:
        logger.error(f"筛选测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """主函数"""
    logger.info("="*80)
    logger.info("因子测试和筛选程序")
    logger.info("="*80)
    
    # 定义要测试的因子
    test_factors = ['BP', 'EP_ttm', 'SP_ttm']
    
    # 步骤1：测试因子
    logger.info(f"\n步骤1：测试因子 {test_factors}")
    
    test_results = {}
    for factor_name in test_factors:
        # 检查因子文件是否存在
        factor_path = Path('E:/Documents/PythonProject/StockProject/StockData/RawFactors') / f'{factor_name}.pkl'
        if factor_path.exists():
            logger.info(f"\n因子文件存在: {factor_path}")
            result = test_single_factor(factor_name)
            test_results[factor_name] = result
        else:
            logger.warning(f"因子文件不存在: {factor_path}")
    
    # 汇总测试结果
    logger.info(f"\n{'='*60}")
    logger.info("测试结果汇总:")
    logger.info('='*60)
    for factor_name, result in test_results.items():
        if result and result.ic_result:
            logger.info(f"{factor_name:10s}: IC={result.ic_result.ic_mean:7.4f}, ICIR={result.ic_result.icir:7.4f}")
        else:
            logger.info(f"{factor_name:10s}: 测试失败")
    
    # 步骤2：因子筛选
    if test_results:
        logger.info("\n步骤2：因子筛选分析")
        screener = test_factor_screening()
        
        if screener:
            logger.info("\n✓ 因子筛选模块测试成功!")
    
    logger.info(f"\n{'='*80}")
    logger.info("程序完成!")
    logger.info('='*80)


if __name__ == "__main__":
    main()