#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AssetTurnover_ttm因子单因子测试脚本
测试优化后的日频扩展方法性能和因子效果
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import logging
from factors.tester import SingleFactorTestPipeline

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_asset_turnover_ttm():
    """测试AssetTurnover_ttm因子"""

    print("\n" + "=" * 60)
    print("🧪 AssetTurnover_ttm因子单因子测试")
    print("=" * 60)

    factor_name = 'AssetTurnover_ttm'

    # 创建测试流水线
    print("\n📋 步骤1: 创建测试流水线")
    print("-" * 60)
    try:
        pipeline = SingleFactorTestPipeline()
        print("✅ 测试流水线创建成功")
    except Exception as e:
        print(f"❌ 测试流水线创建失败: {e}")
        return None

    # 运行单因子测试
    print("\n🚀 步骤2: 运行单因子测试")
    print("-" * 60)
    print(f"因子名称: {factor_name}")
    print(f"测试区间: 2020-01-01 至 2023-12-31")

    start_time = time.time()

    try:
        result = pipeline.run(
            factor_name=factor_name,
            save_result=True,
            begin_date='2020-01-01',
            end_date='2023-12-31'
        )

        end_time = time.time()
        elapsed = end_time - start_time

        print(f"\n⏱️  测试耗时: {elapsed:.2f}秒")

        # 检查测试结果
        if result and not (hasattr(result, 'errors') and result.errors):
            print("\n" + "=" * 60)
            print("✅ 因子测试完成!")
            print("=" * 60)

            # 显示性能指标
            if hasattr(result, 'performance_metrics') and result.performance_metrics:
                metrics = result.performance_metrics
                print("\n📈 性能指标汇总:")
                print("-" * 60)
                print(f"IC均值:      {metrics.get('ic_mean', 'N/A')}")
                print(f"IC标准差:    {metrics.get('ic_std', 'N/A')}")
                print(f"ICIR:        {metrics.get('ic_ir', 'N/A')}")
                print(f"年化收益:    {metrics.get('annual_return', 'N/A')}")
                print(f"夏普比率:    {metrics.get('sharpe_ratio', 'N/A')}")
                print(f"最大回撤:    {metrics.get('max_drawdown', 'N/A')}")
                print("-" * 60)

            # 显示分组回测结果
            if hasattr(result, 'group_returns') and result.group_returns is not None:
                print("\n📊 分组回测结果:")
                print("-" * 60)
                print(result.group_returns)
                print("-" * 60)

            return result
        else:
            error_msg = result.errors if hasattr(result, 'errors') else "未知错误"
            print(f"\n❌ 因子测试失败: {error_msg}")
            return None

    except Exception as e:
        end_time = time.time()
        elapsed = end_time - start_time
        logger.error(f"因子测试失败: {e}")
        print(f"\n❌ 因子测试失败: {e}")
        print(f"⏱️  测试耗时: {elapsed:.2f}秒")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    result = test_asset_turnover_ttm()

    if result:
        print("\n" + "=" * 60)
        print("🎉 测试流程完成!")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("⚠️  测试未能完成，请检查错误信息")
        print("=" * 60)
