#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OCF_Quality因子单因子测试
测试经营现金流质量因子的有效性
"""
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from factors.tester import SingleFactorTestPipeline

def main():
    """测试OCF_Quality因子"""
    print("\n" + "="*70)
    print("OCF_Quality因子单因子测试")
    print("="*70)

    try:
        # 创建测试流水线
        pipeline = SingleFactorTestPipeline()

        # 运行测试（使用默认配置）
        print("\n开始测试OCF_Quality因子...")
        result = pipeline.run(
            'OCF_Quality',
            save_processed_factor=True,  # 保存处理后的因子
            save_result=True             # 保存测试结果
        )

        # 输出关键结果
        print("\n" + "="*70)
        print("测试结果摘要")
        print("="*70)

        # IC结果
        print("\n【IC分析】")
        print(f"IC均值: {result.ic_result.ic_mean:.4f}")
        print(f"IC标准差: {result.ic_result.ic_std:.4f}")
        print(f"ICIR: {result.ic_result.icir:.4f}")
        print(f"IC>0占比: {result.ic_result.ic_win_rate:.2%}")
        print(f"Rank IC均值: {result.ic_result.rank_ic_mean:.4f}")
        print(f"Rank ICIR: {result.ic_result.rank_icir:.4f}")

        # 回归结果
        print("\n【回归分析】")
        print(f"因子收益均值: {result.regression_result.factor_return.mean():.4f}")
        print(f"因子t值均值: {result.regression_result.tvalues.mean():.2f}")
        print(f"|t|>2占比: {(result.regression_result.tvalues.abs() > 2).sum() / len(result.regression_result.tvalues):.2%}")

        # 分组结果
        print("\n【分组测试】")
        print(f"多头组年化收益: {result.group_result.group_returns[0] * 252:.2%}")
        print(f"空头组年化收益: {result.group_result.group_returns[-1] * 252:.2%}")
        print(f"多空组合年化收益: {result.group_result.long_short_return * 252:.2%}")
        print(f"多空夏普比率: {result.group_result.long_short_sharpe:.2f}")

        # 性能指标
        print("\n【整体评价】")
        metrics = result.performance_metrics
        print(f"综合评分: {metrics.get('综合评分', 'N/A')}")

        # 保存位置
        print("\n" + "="*70)
        print("结果保存位置")
        print("="*70)
        print(f"测试结果: StockData/SingleFactorTestData/")
        print(f"处理后因子: StockData/OrthogonalizationFactors/")

        print("\n✅ 测试完成！")

        return result

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = main()
