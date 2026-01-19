#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析OCF_Quality因子测试结果
"""
import sys
import os
import pandas as pd
import json

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    """加载并分析测试结果"""
    print("\n" + "="*70)
    print("OCF_Quality因子测试结果分析")
    print("="*70)

    # 测试结果文件路径
    result_dir = r"E:\Documents\PythonProject\StockProject\StockData\SingleFactorTestData\20260119"
    json_file = os.path.join(result_dir, "OCF_Quality_20260119_141054_d1a1283a_summary.json")

    try:
        # 加载JSON结果
        with open(json_file, 'r', encoding='utf-8') as f:
            result_data = json.load(f)

        print("\n【基本信息】")
        print(f"因子名称: {result_data['factor_name']}")
        print(f"测试时间: {result_data['test_time']}")
        print(f"测试ID: {result_data['test_id']}")

        # 数据信息
        print("\n【数据信息】")
        data_info = result_data['data_info']
        print(f"因子数据量: {data_info['factor_count']}")
        print(f"日期范围: {data_info['date_range']}")
        print(f"股票数量: {data_info['stock_count']}")
        print(f"基准因子: {data_info['base_factors']}")

        # 性能指标（从performance_metrics获取）
        perf = result_data['performance_metrics']

        # IC分析
        print("\n【IC分析】")
        print(f"IC均值: {perf['ic_mean']:.4f}")
        print(f"ICIR: {perf['icir']:.4f}")
        print(f"Rank IC均值: {perf['rank_ic_mean']:.4f}")
        print(f"Rank ICIR: {perf['rank_icir']:.4f}")

        # 回归分析
        print("\n【回归分析】")
        print(f"因子年化收益: {perf['factor_return_annual']:.4f} ({perf['factor_return_annual']:.2%})")
        print(f"因子夏普比率: {perf['factor_return_sharpe']:.2f}")

        # 分组测试
        print("\n【分组测试】")
        print(f"多空年化收益: {perf['long_short_annual_return']:.4f} ({perf['long_short_annual_return']:.2%})")
        print(f"多空夏普比率: {perf['long_short_sharpe']:.2f}")
        print(f"单调性得分: {perf['monotonicity_score']:.4f}")

        # 换手率分析
        print("\n【换手率分析】")
        print(f"平均换手率: {perf['avg_turnover']:.2%}")
        print(f"最大换手率: {perf['max_turnover']:.2%}")
        print(f"换手率标准差: {perf['turnover_std']:.4f}")
        print(f"平均换手成本: {perf['avg_turnover_cost']:.6f}")

        # 综合评价
        print("\n" + "="*70)
        print("综合评价")
        print("="*70)

        # 根据指标给出评价
        ic_mean = perf['ic_mean']
        icir = perf['icir']

        print(f"\n1. IC指标评价:")
        if abs(ic_mean) > 0.03:
            print(f"   ✅ IC均值 {ic_mean:.4f} - 优秀（>0.03）")
        elif abs(ic_mean) > 0.02:
            print(f"   ⭕ IC均值 {ic_mean:.4f} - 良好（0.02-0.03）")
        elif abs(ic_mean) > 0.01:
            print(f"   ⚠️  IC均值 {ic_mean:.4f} - 一般（0.01-0.02）")
        else:
            print(f"   ❌ IC均值 {ic_mean:.4f} - 较弱（<0.01）")

        if icir > 0.5:
            print(f"   ✅ ICIR {icir:.4f} - 优秀（>0.5）")
        elif icir > 0.3:
            print(f"   ⭕ ICIR {icir:.4f} - 良好（0.3-0.5）")
        elif icir > 0.2:
            print(f"   ⚠️  ICIR {icir:.4f} - 一般（0.2-0.3）")
        else:
            print(f"   ❌ ICIR {icir:.4f} - 较弱（<0.2）")

        print(f"\n2. 多空组合评价:")
        ls_annual = perf['long_short_annual_return']
        ls_sharpe = perf['long_short_sharpe']
        if ls_annual > 0.1:
            print(f"   ✅ 年化收益 {ls_annual:.2%} - 优秀（>10%）")
        elif ls_annual > 0.05:
            print(f"   ⭕ 年化收益 {ls_annual:.2%} - 良好（5%-10%）")
        else:
            print(f"   ⚠️  年化收益 {ls_annual:.2%} - 一般（<5%）")

        if ls_sharpe > 1.0:
            print(f"   ✅ 夏普比率 {ls_sharpe:.2f} - 优秀（>1.0）")
        elif ls_sharpe > 0.5:
            print(f"   ⭕ 夏普比率 {ls_sharpe:.2f} - 良好（0.5-1.0）")
        else:
            print(f"   ⚠️  夏普比率 {ls_sharpe:.2f} - 一般（<0.5）")

        print(f"\n3. 单调性评价:")
        mono_score = perf['monotonicity_score']
        if mono_score > 0.8:
            print(f"   ✅ 单调性 {mono_score:.4f} - 优秀（>0.8）")
        elif mono_score > 0.6:
            print(f"   ⭕ 单调性 {mono_score:.4f} - 良好（0.6-0.8）")
        else:
            print(f"   ⚠️  单调性 {mono_score:.4f} - 一般（<0.6）")

        print("\n" + "="*70)
        print("✅ 分析完成！")

    except Exception as e:
        print(f"\n❌ 分析失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
