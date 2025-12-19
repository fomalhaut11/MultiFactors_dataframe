"""
因子导出和导入示例 (v2.1)

本示例展示如何使用v2.1新增的因子导出、导入和管理功能。
"""

import pandas as pd
from factors.tester import SingleFactorTestPipeline


def example_1_basic_usage():
    """示例1: 基本使用 - 自动保存处理后的因子"""
    print("=" * 60)
    print("示例1: 基本使用 - 自动保存处理后的因子")
    print("=" * 60)

    pipeline = SingleFactorTestPipeline()

    # 测试因子（自动保存测试结果和处理后的因子）
    result = pipeline.run('ROE_ttm')

    print(f"\n测试完成:")
    print(f"  IC均值: {result.ic_result.ic_mean:.4f}")
    print(f"  ICIR: {result.ic_result.icir:.4f}")
    print(f"  单调性: {result.group_result.monotonicity_score:.4f}")

    print(f"\n处理后的因子已自动保存到:")
    print(f"  OrthogonalizationFactors/neutral_industry_outlier3_zscore/ROE_ttm.pkl")
    print(f"  (配置由系统自动根据测试参数生成)")


def example_2_control_save_behavior():
    """示例2: 控制保存行为"""
    print("\n" + "=" * 60)
    print("示例2: 控制保存行为")
    print("=" * 60)

    pipeline = SingleFactorTestPipeline()

    # 场景1: 只保存测试结果，不保存处理后的因子
    print("\n场景1: 只保存测试结果")
    result = pipeline.run('ROA_ttm', save_processed_factor=False)
    print(f"  IC均值: {result.ic_result.ic_mean:.4f}")

    # 场景2: 只生成因子，不保存测试结果
    print("\n场景2: 只生成处理后的因子")
    result = pipeline.run('EP_ttm', save_result=False, save_processed_factor=True)
    print(f"  因子已生成并保存（测试结果未保存）")

    # 场景3: 都不保存（仅用于临时测试）
    print("\n场景3: 临时测试（都不保存）")
    result = pipeline.run('BP', save_result=False, save_processed_factor=False)
    print(f"  临时测试完成，无文件保存")


def example_3_export_factors():
    """示例3: 导出处理后的因子"""
    print("\n" + "=" * 60)
    print("示例3: 导出处理后的因子")
    print("=" * 60)

    pipeline = SingleFactorTestPipeline()

    # 导出默认配置的因子
    print("\n导出默认配置的因子:")
    path = pipeline.export_processed_factor('ROE_ttm')
    print(f"  已保存: {path}")

    # 导出特定配置的因子
    print("\n导出特定配置的因子:")
    path = pipeline.export_processed_factor(
        'ROE_ttm',
        subfolder='large_cap_pool',
        netral_base=True,
        use_industry=True,
        outlier_param=5
    )
    print(f"  已保存: {path}")

    # 批量导出多个因子
    print("\n批量导出多个因子:")
    factors = ['ROE_ttm', 'ROA_ttm', 'EP_ttm']
    for factor_name in factors:
        path = pipeline.export_processed_factor(
            factor_name,
            subfolder='neutral_factors'
        )
        print(f"  {factor_name} -> {path}")


def example_4_load_factors():
    """示例4: 加载处理后的因子"""
    print("\n" + "=" * 60)
    print("示例4: 加载处理后的因子")
    print("=" * 60)

    pipeline = SingleFactorTestPipeline()

    # 确保因子存在（先导出）
    pipeline.export_processed_factor('ROE_ttm')

    # 方法1: 自动查找因子
    print("\n方法1: 自动查找因子")
    factor = pipeline.load_processed_factor('ROE_ttm')
    if factor is not None:
        print(f"  加载成功: {len(factor)} 条数据")
        print(f"  时间范围: {factor.index.get_level_values(0).min()} 到 "
              f"{factor.index.get_level_values(0).max()}")
        print(f"  股票数量: {len(factor.index.get_level_values(1).unique())}")

    # 方法2: 从特定子文件夹加载
    print("\n方法2: 从特定子文件夹加载")
    factor = pipeline.load_processed_factor('ROE_ttm', subfolder='neutral_factors')
    if factor is not None:
        print(f"  加载成功: {len(factor)} 条数据")

    # 方法3: 使用配置键加载
    print("\n方法3: 使用配置键加载")
    factor = pipeline.load_processed_factor(
        'ROE_ttm',
        config_key='neutral_industry_outlier3_zscore'
    )
    if factor is not None:
        print(f"  加载成功: {len(factor)} 条数据")
        print(f"  因子统计: 均值={factor.mean():.6f}, 标准差={factor.std():.6f}")


def example_5_metadata():
    """示例5: 查看因子元数据"""
    print("\n" + "=" * 60)
    print("示例5: 查看因子元数据")
    print("=" * 60)

    pipeline = SingleFactorTestPipeline()

    # 确保因子和元数据存在
    pipeline.export_processed_factor('ROE_ttm')

    # 加载元数据
    metadata = pipeline.load_factor_metadata('ROE_ttm')

    if metadata:
        print("\n处理配置:")
        config = metadata['processing_config']
        print(f"  去极值方法: {config['outlier_method']}")
        print(f"  去极值参数: {config['outlier_param']}")
        print(f"  标准化方法: {config['normalization_method']}")
        print(f"  是否中性化: {config['netral_base']}")
        print(f"  基准因子: {config['base_factors']}")
        print(f"  是否行业中性: {config['use_industry']}")

        print("\n数据信息:")
        data_info = metadata['data_info']
        print(f"  样本数量: {data_info['sample_count']}")
        print(f"  股票数量: {data_info['stock_count']}")
        print(f"  交易日数: {data_info['date_count']}")

        print("\n性能指标:")
        perf = metadata['performance_summary']
        print(f"  IC均值: {perf.get('ic_mean', 'N/A')}")
        print(f"  ICIR: {perf.get('icir', 'N/A')}")
        print(f"  单调性: {perf.get('monotonicity_score', 'N/A')}")


def example_6_list_factors():
    """示例6: 列出所有处理后的因子"""
    print("\n" + "=" * 60)
    print("示例6: 列出所有处理后的因子")
    print("=" * 60)

    pipeline = SingleFactorTestPipeline()

    # 先导出几个因子
    for factor in ['ROE_ttm', 'ROA_ttm']:
        pipeline.export_processed_factor(factor)

    # 获取所有因子名称列表
    print("\n所有处理后的因子:")
    factors = pipeline.list_processed_factors()
    if factors:
        for i, factor_name in enumerate(factors, 1):
            print(f"  {i}. {factor_name}")
    else:
        print("  (尚无处理后的因子)")

    # 获取详细元数据（DataFrame格式）
    print("\n因子详细信息:")
    factors_df = pipeline.list_processed_factors(return_metadata=True)
    if not factors_df.empty:
        print(factors_df[['factor_name', 'subfolder', 'sample_count']].head())

        # 筛选高质量因子
        if 'ic_mean' in factors_df.columns and 'icir' in factors_df.columns:
            high_quality = factors_df[
                (factors_df['ic_mean'].abs() > 0.02) &
                (factors_df['icir'].abs() > 0.3)
            ]
            print(f"\n高质量因子 (|IC|>0.02, |ICIR|>0.3): {len(high_quality)}个")
            if not high_quality.empty:
                print(high_quality[['factor_name', 'ic_mean', 'icir']])


def example_7_multi_strategy():
    """示例7: 多策略因子管理"""
    print("\n" + "=" * 60)
    print("示例7: 多策略因子管理")
    print("=" * 60)

    pipeline = SingleFactorTestPipeline()

    # 为不同的股票池生成中性化因子
    stock_pools = ['liquid_1000', 'large_cap_500']

    print("\n为不同股票池生成中性化因子:")
    for pool in stock_pools:
        path = pipeline.export_processed_factor(
            'ROE_ttm',
            subfolder=pool,
            stock_universe=pool
        )
        print(f"  {pool}: {path}")

    # 后续使用时直接加载
    print("\n加载不同股票池的因子:")
    for pool in stock_pools:
        factor = pipeline.load_processed_factor('ROE_ttm', subfolder=pool)
        if factor is not None:
            print(f"  {pool}: {len(factor)} 条数据")


def example_8_factor_combination():
    """示例8: 因子组合构建"""
    print("\n" + "=" * 60)
    print("示例8: 因子组合构建")
    print("=" * 60)

    pipeline = SingleFactorTestPipeline()

    # 导出多个因子
    factor_names = ['ROE_ttm', 'ROA_ttm', 'EP_ttm']
    print("\n导出多个因子:")
    for name in factor_names:
        pipeline.export_processed_factor(name)
        print(f"  已导出: {name}")

    # 加载多个处理后的因子用于组合
    print("\n加载因子用于组合:")
    factors_dict = {}
    for name in factor_names:
        factor = pipeline.load_processed_factor(name)
        if factor is not None:
            factors_dict[name] = factor
            print(f"  {name}: {len(factor)} 条数据")

    # 合并所有因子为DataFrame
    if factors_dict:
        factors_df = pd.DataFrame(factors_dict)
        print(f"\n因子组合: {factors_df.shape}")
        print(f"  交易日数: {len(factors_df.index.get_level_values(0).unique())}")
        print(f"  股票数: {len(factors_df.index.get_level_values(1).unique())}")
        print(f"  因子数: {len(factors_df.columns)}")


def example_9_config_comparison():
    """示例9: 不同配置对比"""
    print("\n" + "=" * 60)
    print("示例9: 不同配置对比")
    print("=" * 60)

    pipeline = SingleFactorTestPipeline()

    # 生成不同去极值参数的因子
    outlier_params = [3, 5]

    print("\n测试不同去极值参数的影响:")
    for param in outlier_params:
        result = pipeline.run(
            'ROE_ttm',
            outlier_param=param,
            save_result=False,
            save_processed_factor=True
        )
        print(f"  outlier_param={param}: IC={result.ic_result.ic_mean:.4f}, "
              f"ICIR={result.ic_result.icir:.4f}")

    # 加载并比较不同配置的因子
    print("\n加载不同配置的因子:")
    for param in outlier_params:
        factor = pipeline.load_processed_factor(
            'ROE_ttm',
            config_key=f'neutral_industry_outlier{param}_zscore'
        )
        if factor is not None:
            print(f"  outlier{param}: {len(factor)} 条数据, "
                  f"均值={factor.mean():.6f}, 标准差={factor.std():.6f}")


def main():
    """运行所有示例"""
    print("\n" + "=" * 60)
    print("因子导出和导入示例 (v2.1)")
    print("=" * 60)

    examples = [
        example_1_basic_usage,
        example_2_control_save_behavior,
        example_3_export_factors,
        example_4_load_factors,
        example_5_metadata,
        example_6_list_factors,
        example_7_multi_strategy,
        example_8_factor_combination,
        example_9_config_comparison,
    ]

    for example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"\n错误: {example_func.__name__} 执行失败")
            print(f"  {type(e).__name__}: {e}")

    print("\n" + "=" * 60)
    print("示例运行完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()
