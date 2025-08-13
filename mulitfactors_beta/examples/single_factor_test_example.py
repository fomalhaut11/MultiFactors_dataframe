"""
单因子测试模块使用示例
展示新的测试框架的各种使用方法
"""

import pandas as pd
import numpy as np
from factors.tester import SingleFactorTestPipeline, ResultManager


def example_basic_test():
    """基础测试示例"""
    print("=" * 50)
    print("1. 基础单因子测试")
    print("=" * 50)
    
    # 创建测试流水线
    pipeline = SingleFactorTestPipeline()
    
    # 测试单个因子
    result = pipeline.run('ROE_ttm', save_result=True)
    
    # 查看测试结果
    print(f"因子名称: {result.factor_name}")
    print(f"测试ID: {result.test_id}")
    print(f"样本数量: {result.data_info.get('factor_count', 0)}")
    
    # 查看性能指标
    print("\n性能指标:")
    for key, value in result.performance_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # 查看IC结果
    if result.ic_result:
        print(f"\nIC均值: {result.ic_result.ic_mean:.4f}")
        print(f"ICIR: {result.ic_result.icir:.4f}")
        print(f"Rank IC均值: {result.ic_result.rank_ic_mean:.4f}")
    
    return result


def example_custom_config():
    """自定义配置测试示例"""
    print("\n" + "=" * 50)
    print("2. 自定义配置测试")
    print("=" * 50)
    
    pipeline = SingleFactorTestPipeline()
    
    # 使用自定义配置
    result = pipeline.run(
        'EP_ttm',
        save_result=True,
        # 覆盖默认配置
        begin_date='2020-01-01',
        end_date='2024-12-31',
        group_nums=20,  # 使用20组而不是默认的10组
        netral_base=False,  # 不使用基准中性化
        use_industry=False  # 不使用行业分类
    )
    
    print(f"测试配置:")
    print(f"  时间范围: {result.config_snapshot.get('begin_date')} 到 {result.config_snapshot.get('end_date')}")
    print(f"  分组数量: {result.config_snapshot.get('group_nums')}")
    print(f"  基准中性化: {result.config_snapshot.get('netral_base')}")
    
    # 查看分组结果
    if result.group_result:
        print(f"\n分组单调性得分: {result.group_result.monotonicity_score:.4f}")
        print(f"多空组合年化收益: {result.group_result.long_short_return.mean() * 252:.2%}")
    
    return result


def example_batch_test():
    """批量测试示例"""
    print("\n" + "=" * 50)
    print("3. 批量因子测试")
    print("=" * 50)
    
    pipeline = SingleFactorTestPipeline()
    
    # 测试多个因子
    factor_list = ['ROE_ttm', 'ROA_ttm', 'EP_ttm', 'BP', 'LogMarketCap']
    
    batch_result = pipeline.batch_run(
        factor_list,
        save_results=True,
        parallel=True,  # 并行执行
        max_workers=3   # 使用3个工作进程
    )
    
    # 查看汇总结果
    print(f"测试完成: {len(batch_result.test_results)} 个因子")
    
    if batch_result.summary_df is not None:
        print("\n测试结果汇总 (按ICIR排序):")
        print(batch_result.summary_df[['factor_name', 'ic_mean', 'icir', 'long_short_sharpe']].head())
    
    return batch_result


def example_parameter_sensitivity():
    """参数敏感性测试示例"""
    print("\n" + "=" * 50)
    print("4. 参数敏感性测试")
    print("=" * 50)
    
    pipeline = SingleFactorTestPipeline()
    
    # 定义参数网格
    param_grid = {
        'group_nums': [5, 10, 20],
        'outlier_param': [3, 5, 7],
        'normalization_method': ['zscore', 'minmax']
    }
    
    # 执行参数敏感性测试
    results_df = pipeline.parameter_sensitivity_test(
        'ROE_ttm',
        param_grid,
        save_results=False  # 不保存每个测试结果
    )
    
    print(f"测试了 {len(results_df)} 种参数组合")
    
    # 找出最佳参数组合
    best_idx = results_df['icir'].idxmax()
    best_params = results_df.loc[best_idx]
    
    print(f"\n最佳参数组合 (ICIR最高):")
    print(f"  分组数: {best_params['group_nums']}")
    print(f"  去极值参数: {best_params['outlier_param']}")
    print(f"  标准化方法: {best_params['normalization_method']}")
    print(f"  ICIR: {best_params['icir']:.4f}")
    
    return results_df


def example_result_analysis():
    """结果分析示例"""
    print("\n" + "=" * 50)
    print("5. 结果分析和比较")
    print("=" * 50)
    
    # 创建结果管理器
    result_manager = ResultManager()
    
    # 加载历史结果
    results = result_manager.load_batch()
    
    if results:
        print(f"加载了 {len(results)} 个历史测试结果")
        
        # 生成汇总表
        summary_df = result_manager.get_summary_table(results)
        
        print("\n表现最好的5个因子 (按ICIR排序):")
        print(summary_df[['factor_name', 'ic_mean', 'icir', 'factor_sharpe']].head())
        
        # 比较结果
        comparison_df = result_manager.compare_results(results[:5])
        
        print("\n因子比较:")
        print(comparison_df[['factor_name', 'icir', 'icir_rank', 'monotonicity', 'monotonicity_rank']])
        
        # 导出到Excel
        result_manager.export_to_excel(results[:10], 'factor_test_results.xlsx')
        print("\n结果已导出到 factor_test_results.xlsx")
    else:
        print("没有找到历史测试结果")


def example_profile_test():
    """使用不同配置文件测试示例"""
    print("\n" + "=" * 50)
    print("6. 多配置测试")
    print("=" * 50)
    
    pipeline = SingleFactorTestPipeline()
    
    # 使用不同的预设配置测试同一因子
    profiles = ['quick_test', 'full_test', 'weekly_test']
    
    results = pipeline.run_with_profiles(
        'ROE_ttm',
        profiles,
        save_results=True
    )
    
    print(f"使用 {len(profiles)} 种配置测试了因子 ROE_ttm")
    
    for profile, result in results.items():
        print(f"\n配置: {profile}")
        print(f"  ICIR: {result.performance_metrics.get('icir', 0):.4f}")
        print(f"  多空夏普: {result.performance_metrics.get('long_short_sharpe', 0):.4f}")


def example_visualization():
    """可视化示例"""
    print("\n" + "=" * 50)
    print("7. 结果可视化")
    print("=" * 50)
    
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 运行测试
    pipeline = SingleFactorTestPipeline()
    result = pipeline.run('ROE_ttm', save_result=False)
    
    if result.regression_result and not result.regression_result.cumulative_return.empty:
        # 绘制累计收益曲线
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # 1. 因子累计收益
        result.regression_result.cumulative_return.plot(ax=axes[0, 0])
        axes[0, 0].set_title('因子累计收益')
        axes[0, 0].set_xlabel('日期')
        axes[0, 0].set_ylabel('累计收益')
        
        # 2. IC时间序列
        if result.ic_result and not result.ic_result.ic_series.empty:
            result.ic_result.ic_series.plot(ax=axes[0, 1])
            axes[0, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
            axes[0, 1].set_title('IC时间序列')
            axes[0, 1].set_xlabel('日期')
            axes[0, 1].set_ylabel('IC')
        
        # 3. 分组收益
        if result.group_result and not result.group_result.group_returns.empty:
            group_mean = result.group_result.group_returns.mean()
            group_mean.plot(kind='bar', ax=axes[1, 0])
            axes[1, 0].set_title('分组平均收益')
            axes[1, 0].set_xlabel('分组')
            axes[1, 0].set_ylabel('平均收益')
        
        # 4. 多空组合累计收益
        if result.group_result and not result.group_result.long_short_return.empty:
            result.group_result.long_short_return.cumsum().plot(ax=axes[1, 1])
            axes[1, 1].set_title('多空组合累计收益')
            axes[1, 1].set_xlabel('日期')
            axes[1, 1].set_ylabel('累计收益')
        
        plt.tight_layout()
        plt.savefig('factor_test_visualization.png')
        print("可视化结果已保存到 factor_test_visualization.png")
        plt.show()


def main():
    """运行所有示例"""
    print("单因子测试模块使用示例")
    print("=" * 50)
    
    # 1. 基础测试
    example_basic_test()
    
    # 2. 自定义配置
    example_custom_config()
    
    # 3. 批量测试
    example_batch_test()
    
    # 4. 参数敏感性测试
    example_parameter_sensitivity()
    
    # 5. 结果分析
    example_result_analysis()
    
    # 6. 多配置测试
    example_profile_test()
    
    # 7. 可视化
    example_visualization()
    
    print("\n" + "=" * 50)
    print("所有示例运行完成！")


if __name__ == "__main__":
    main()