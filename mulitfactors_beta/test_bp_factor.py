#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试BP因子并验证筛选模块
"""

import sys
from pathlib import Path
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


def test_bp_factor():
    """测试BP因子"""
    logger.info("="*60)
    logger.info("测试BP因子")
    logger.info("="*60)
    
    try:
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
            logger.info(f"✓ 测试成功!")
            logger.info(f"  IC均值: {result.ic_result.ic_mean:.4f}")
            logger.info(f"  ICIR: {result.ic_result.icir:.4f}")
            logger.info(f"  Rank IC: {result.ic_result.rank_ic_mean:.4f}")
        
        if result.group_result:
            logger.info(f"  单调性: {result.group_result.monotonicity_score:.4f}")
        
        if result.performance_metrics:
            logger.info(f"  夏普比率: {result.performance_metrics.get('long_short_sharpe', 0):.4f}")
        
        return result
        
    except Exception as e:
        logger.error(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_screener():
    """测试筛选模块"""
    logger.info("\n" + "="*60)
    logger.info("测试因子筛选模块")
    logger.info("="*60)
    
    try:
        screener = FactorScreener()
        
        # 加载测试结果
        logger.info("\n1. 加载历史测试结果...")
        results = screener.load_all_results()
        logger.info(f"   加载了 {len(results)} 个因子的测试结果")
        
        if results:
            # 显示部分因子
            logger.info("   已有因子:")
            for i, factor_name in enumerate(list(results.keys())[:5]):
                logger.info(f"   {i+1}. {factor_name}")
        
        # 使用默认标准筛选
        logger.info("\n2. 筛选高质量因子...")
        selected = screener.screen_factors({
            'ic_mean_min': 0.01,
            'icir_min': 0.3,
            'monotonicity_min': 0.4
        })
        
        logger.info(f"   筛选出 {len(selected)} 个因子")
        for factor in selected[:5]:
            logger.info(f"   - {factor}")
        
        return screener
        
    except Exception as e:
        logger.error(f"筛选测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """主函数"""
    logger.info("="*80)
    logger.info("BP因子测试和筛选程序")
    logger.info("="*80)
    
    # 步骤1：测试BP因子
    logger.info("\n步骤1：测试BP因子")
    bp_result = test_bp_factor()
    
    if bp_result:
        logger.info("\n✓ BP因子测试成功!")
    
    # 步骤2：测试筛选模块
    logger.info("\n步骤2：测试筛选模块")
    screener = test_screener()
    
    if screener:
        logger.info("\n✓ 筛选模块测试成功!")
    
    logger.info(f"\n{'='*80}")
    logger.info("程序完成!")
    logger.info('='*80)


if __name__ == "__main__":
    main()