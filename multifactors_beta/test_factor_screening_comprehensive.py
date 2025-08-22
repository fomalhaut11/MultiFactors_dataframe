#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
综合因子筛选测试
测试因子筛选模块的各项功能
"""

import sys
from pathlib import Path
import pandas as pd
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

from factors.analyzer import FactorScreener


def test_comprehensive_screening():
    """综合测试因子筛选功能"""
    logger.info("="*80)
    logger.info("综合因子筛选测试")
    logger.info("="*80)
    
    # 创建筛选器
    screener = FactorScreener()
    
    # 1. 加载所有测试结果
    logger.info("\n1. 加载历史测试结果")
    logger.info("-"*40)
    results = screener.load_all_results()
    logger.info(f"加载了 {len(results)} 个因子的测试结果")
    
    if not results:
        logger.warning("没有找到测试结果，请先运行因子测试")
        return
    
    # 显示所有因子的基本信息
    logger.info("\n因子列表:")
    for i, (factor_name, result) in enumerate(results.items(), 1):
        if hasattr(result, 'ic_result') and result.ic_result:
            logger.info(f"  {i:2d}. {factor_name:15s} IC={result.ic_result.ic_mean:7.4f}  ICIR={result.ic_result.icir:7.4f}")
        else:
            logger.info(f"  {i:2d}. {factor_name:15s} [无IC结果]")
    
    # 2. 测试不同筛选标准
    logger.info("\n2. 测试不同筛选标准")
    logger.info("-"*40)
    
    # 宽松标准
    logger.info("\n宽松标准（loose）:")
    loose_factors = screener.screen_factors(preset='loose')
    logger.info(f"  筛选出 {len(loose_factors)} 个因子: {loose_factors}")
    
    # 正常标准
    logger.info("\n正常标准（normal）:")
    normal_factors = screener.screen_factors(preset='normal')
    logger.info(f"  筛选出 {len(normal_factors)} 个因子: {normal_factors}")
    
    # 严格标准
    logger.info("\n严格标准（strict）:")
    strict_factors = screener.screen_factors(preset='strict')
    logger.info(f"  筛选出 {len(strict_factors)} 个因子: {strict_factors}")
    
    # 自定义标准
    logger.info("\n自定义标准（IC>0.01, ICIR>0.1）:")
    custom_factors = screener.screen_factors({
        'ic_mean_min': 0.01,
        'icir_min': 0.1,
        'monotonicity_min': 0.0  # 不限制单调性
    })
    logger.info(f"  筛选出 {len(custom_factors)} 个因子: {custom_factors}")
    
    # 3. 因子排名
    logger.info("\n3. 因子排名")
    logger.info("-"*40)
    
    # 按ICIR排名
    logger.info("\n按ICIR排名（前5）:")
    icir_ranking = screener.get_factor_ranking('icir', top_n=5)
    if not icir_ranking.empty:
        logger.info("\n" + icir_ranking.to_string())
    else:
        logger.info("  无排名数据")
    
    # 按IC排名
    logger.info("\n按IC均值排名（前5）:")
    ic_ranking = screener.get_factor_ranking('ic_mean', top_n=5)
    if not ic_ranking.empty:
        logger.info("\n" + ic_ranking.to_string())
    else:
        logger.info("  无排名数据")
    
    # 4. 稳定性分析（如果有BP因子）
    if 'BP' in results:
        logger.info("\n4. BP因子稳定性分析")
        logger.info("-"*40)
        stability = screener.analyze_factor_stability('BP', lookback_days=30)
        if stability:
            logger.info(f"  IC稳定性: {stability.get('ic_stability', 'N/A')}")
            logger.info(f"  稳定性等级: {stability.get('stability_grade', 'N/A')}")
            logger.info(f"  最近30天IC均值: {stability.get('recent_ic_mean', 'N/A')}")
            logger.info(f"  最近30天IC标准差: {stability.get('recent_ic_std', 'N/A')}")
    
    # 5. 因子比较
    logger.info("\n5. 因子比较")
    logger.info("-"*40)
    factors_to_compare = list(results.keys())[:3]  # 比较前3个因子
    if len(factors_to_compare) >= 2:
        comparison = screener.compare_factors(factors_to_compare)
        if comparison is not None and not comparison.empty:
            logger.info(f"\n比较因子: {factors_to_compare}")
            logger.info("\n" + comparison.to_string())
    
    # 6. 生成筛选报告
    logger.info("\n6. 生成筛选报告")
    logger.info("-"*40)
    report_path = Path(project_root) / 'factor_output' / 'screening_report.csv'
    report_path.parent.mkdir(exist_ok=True)
    
    report = screener.generate_screening_report(
        output_path=str(report_path),
        top_n=10
    )
    
    if report is not None and not report.empty:
        logger.info(f"  报告已保存到: {report_path}")
        logger.info(f"  包含 {len(report)} 个因子")
        logger.info("\n前5行预览:")
        logger.info("\n" + report.head().to_string())
    
    # 7. 获取因子评分
    logger.info("\n7. 因子综合评分")
    logger.info("-"*40)
    for factor in list(results.keys())[:5]:
        score = screener.get_factor_score(factor)
        logger.info(f"  {factor:15s}: {score:6.2f} 分")
    
    logger.info("\n" + "="*80)
    logger.info("综合测试完成!")
    logger.info("="*80)
    
    return screener


def display_summary_statistics(screener):
    """显示汇总统计"""
    logger.info("\n" + "="*80)
    logger.info("汇总统计")
    logger.info("="*80)
    
    results = screener.all_results
    if not results:
        logger.info("无测试结果")
        return
    
    # IC统计
    ic_values = []
    icir_values = []
    monotonicity_values = []
    
    for factor_name, result in results.items():
        if hasattr(result, 'ic_result') and result.ic_result:
            ic_values.append(result.ic_result.ic_mean)
            icir_values.append(result.ic_result.icir)
        if hasattr(result, 'group_result') and result.group_result:
            monotonicity_values.append(result.group_result.monotonicity_score)
    
    if ic_values:
        logger.info("\nIC统计:")
        logger.info(f"  平均值: {pd.Series(ic_values).mean():.4f}")
        logger.info(f"  中位数: {pd.Series(ic_values).median():.4f}")
        logger.info(f"  最大值: {pd.Series(ic_values).max():.4f}")
        logger.info(f"  最小值: {pd.Series(ic_values).min():.4f}")
    
    if icir_values:
        logger.info("\nICIR统计:")
        logger.info(f"  平均值: {pd.Series(icir_values).mean():.4f}")
        logger.info(f"  中位数: {pd.Series(icir_values).median():.4f}")
        logger.info(f"  最大值: {pd.Series(icir_values).max():.4f}")
        logger.info(f"  最小值: {pd.Series(icir_values).min():.4f}")
    
    if monotonicity_values:
        logger.info("\n单调性统计:")
        logger.info(f"  平均值: {pd.Series(monotonicity_values).mean():.4f}")
        logger.info(f"  中位数: {pd.Series(monotonicity_values).median():.4f}")
        logger.info(f"  最大值: {pd.Series(monotonicity_values).max():.4f}")
        logger.info(f"  最小值: {pd.Series(monotonicity_values).min():.4f}")
    
    # 筛选通过率
    logger.info("\n筛选通过率:")
    total = len(results)
    for preset in ['loose', 'normal', 'strict']:
        selected = screener.screen_factors(preset=preset)
        logger.info(f"  {preset:6s}: {len(selected):2d}/{total:2d} ({len(selected)/total*100:.1f}%)")


def main():
    """主函数"""
    try:
        # 运行综合测试
        screener = test_comprehensive_screening()
        
        if screener and screener.all_results:
            # 显示汇总统计
            display_summary_statistics(screener)
        
        logger.info("\n✅ 所有测试完成!")
        
    except Exception as e:
        logger.error(f"测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()