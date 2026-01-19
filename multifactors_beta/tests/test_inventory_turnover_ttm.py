#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
InventoryTurnover_ttm因子单因子测试脚本
"""
import sys
import logging
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from factors.tester import SingleFactorTestPipeline

def main():
    """主函数"""
    print("=" * 60)
    print("🧪 InventoryTurnover_ttm因子单因子测试")
    print("=" * 60)

    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    try:
        # 步骤1: 创建测试流水线
        print("\n📋 步骤1: 创建测试流水线")
        print("-" * 60)
        pipeline = SingleFactorTestPipeline()
        print("✅ 测试流水线创建成功")

        # 步骤2: 运行单因子测试
        print("\n🚀 步骤2: 运行单因子测试")
        print("-" * 60)
        print(f"因子名称: InventoryTurnover_ttm")
        print(f"测试区间: 2020-01-01 至 2023-12-31")

        result = pipeline.run(
            factor_name='InventoryTurnover_ttm',
            save_result=True,
            begin_date='2020-01-01',
            end_date='2023-12-31'
        )

        # 步骤3: 显示测试结果
        print("\n📈 步骤3: 测试结果汇总")
        print("-" * 60)
        if result and hasattr(result, 'ic_result') and result.ic_result:
            print(f"IC均值:      {result.ic_result.ic_mean:.6f}")
            print(f"IC标准差:    {result.ic_result.ic_std:.6f}")
            print(f"ICIR:        {result.ic_result.icir:.6f}")
        else:
            print("⚠️ 测试结果中未包含IC分析结果")

        print("\n" + "=" * 60)
        print("✅ 因子测试完成!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
