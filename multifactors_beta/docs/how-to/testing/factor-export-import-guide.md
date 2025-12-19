# 处理后因子的导出和导入指南（v2.1）

## 概述

v2.1版本新增了中性化和归一化因子的独立保存和管理功能，使得处理后的因子可以被保存、加载和重用，无需每次都重新计算。

## 因子版本体系

项目采用分层的因子版本管理：

```
因子数据流：
RawFactors/              # 原始因子（未处理）
    └── ROE_ttm.pkl     # 从财务报表直接计算的原始TTM ROE

        ↓ 测试处理（去极值+标准化+中性化）

OrthogonalizationFactors/   # 处理后因子（中性化+归一化）
    ├── neutral_industry_outlier3_zscore/     # 配置1：中性化+行业+去极值3+zscore
    │   ├── ROE_ttm.pkl
    │   ├── ROE_ttm_metadata.json
    │   └── ...
    ├── neutral_outlier5_zscore/              # 配置2：中性化+去极值5+zscore（无行业）
    │   └── ...
    └── default/                              # 默认配置
        └── ...

        ↓ 用于测试或建模

SingleFactorTestData/        # 测试结果（含因子快照和分析结果）
    └── 20250112/
        ├── ROE_ttm_test_result.pkl
        └── ...
```

## 自动保存处理后的因子

### 基本使用

默认情况下，运行测试会自动保存处理后的因子：

```python
from factors.tester import SingleFactorTestPipeline

pipeline = SingleFactorTestPipeline()

# 自动保存处理后的因子到OrthogonalizationFactors目录
result = pipeline.run('ROE_ttm')

# 因子保存位置自动由配置决定：
# - netral_base=True → 添加 'neutral' 标识
# - use_industry=True → 添加 'industry' 标识
# - outlier_param=3 → 添加 'outlier3' 标识
# - normalization_method='zscore' → 添加 'zscore' 标识
#
# 最终保存到：OrthogonalizationFactors/neutral_industry_outlier3_zscore/ROE_ttm.pkl
```

### 控制保存行为

```python
# 只保存测试结果，不保存处理后的因子
result = pipeline.run('ROE_ttm', save_processed_factor=False)

# 只生成因子，不保存测试结果
result = pipeline.run('ROE_ttm', save_result=False, save_processed_factor=True)

# 都不保存（仅用于临时测试）
result = pipeline.run('ROE_ttm', save_result=False, save_processed_factor=False)
```

## 导出处理后的因子

使用`export_processed_factor`方法可以生成特定配置的因子：

```python
# 导出默认配置的因子
path = pipeline.export_processed_factor('ROE_ttm')
print(f"因子已保存: {path}")

# 导出特定配置的因子
path = pipeline.export_processed_factor(
    'ROE_ttm',
    subfolder='large_cap_pool',  # 自定义子文件夹
    netral_base=True,
    use_industry=True,
    outlier_param=5
)

# 批量导出多个因子
factors = ['ROE_ttm', 'ROA_ttm', 'EP_ttm']
for factor_name in factors:
    pipeline.export_processed_factor(
        factor_name,
        subfolder='neutral_factors'
    )
```

## 加载处理后的因子

### 基本加载

```python
# 自动查找因子（在所有子文件夹中搜索）
processed_factor = pipeline.load_processed_factor('ROE_ttm')
print(f"加载了 {len(processed_factor)} 条因子数据")
print(f"时间范围: {processed_factor.index.get_level_values(0).min()} 到 "
      f"{processed_factor.index.get_level_values(0).max()}")

# 从特定子文件夹加载
processed_factor = pipeline.load_processed_factor(
    'ROE_ttm',
    subfolder='large_cap_pool'
)

# 使用配置键加载
processed_factor = pipeline.load_processed_factor(
    'ROE_ttm',
    config_key='neutral_industry_outlier3_zscore'
)

# 直接使用加载的因子进行分析或建模
# processed_factor是标准的MultiIndex[TradingDates, StockCodes] Series
```

## 查看因子元数据

每个处理后的因子都保存了完整的元数据：

```python
# 加载元数据
metadata = pipeline.load_factor_metadata('ROE_ttm')

# 查看处理配置
print("处理配置:")
print(f"  去极值方法: {metadata['processing_config']['outlier_method']}")
print(f"  去极值参数: {metadata['processing_config']['outlier_param']}")
print(f"  标准化方法: {metadata['processing_config']['normalization_method']}")
print(f"  是否中性化: {metadata['processing_config']['netral_base']}")
print(f"  基准因子: {metadata['processing_config']['base_factors']}")
print(f"  是否行业中性: {metadata['processing_config']['use_industry']}")

# 查看数据信息
print("\n数据信息:")
print(f"  样本数量: {metadata['data_info']['sample_count']}")
print(f"  股票数量: {metadata['data_info']['stock_count']}")
print(f"  交易日数: {metadata['data_info']['date_count']}")

# 查看性能摘要
print("\n性能指标:")
print(f"  IC均值: {metadata['performance_summary']['ic_mean']:.4f}")
print(f"  ICIR: {metadata['performance_summary']['icir']:.4f}")
print(f"  单调性: {metadata['performance_summary']['monotonicity_score']:.4f}")
```

## 列出所有处理后的因子

```python
# 获取所有因子名称列表
factors = pipeline.list_processed_factors()
print(f"共有 {len(factors)} 个处理后的因子")
print(factors)

# 获取详细元数据（DataFrame格式）
factors_df = pipeline.list_processed_factors(return_metadata=True)
print(factors_df[['factor_name', 'subfolder', 'ic_mean', 'icir', 'sample_count']])

# 筛选高质量因子
high_quality = factors_df[
    (factors_df['ic_mean'].abs() > 0.03) &
    (factors_df['icir'].abs() > 0.5)
]
print(f"高质量因子: {len(high_quality)}个")
print(high_quality[['factor_name', 'ic_mean', 'icir']])

# 查看特定子文件夹的因子
large_cap_factors = pipeline.list_processed_factors(
    subfolder='large_cap_pool',
    return_metadata=True
)
```

## 使用场景示例

### 场景1：多策略因子管理

为不同的股票池生成中性化因子：

```python
# 为不同的股票池生成中性化因子
stock_pools = ['liquid_1000', 'large_cap_500', 'mid_cap_300']

for pool in stock_pools:
    # 导出针对该股票池的中性化因子
    path = pipeline.export_processed_factor(
        'ROE_ttm',
        subfolder=pool,
        stock_universe=pool  # 使用特定股票池
    )
    print(f"{pool} 因子已保存: {path}")

# 后续使用时直接加载
for pool in stock_pools:
    factor = pipeline.load_processed_factor('ROE_ttm', subfolder=pool)
    # 使用该因子进行策略回测或建模
```

### 场景2：因子组合构建

加载多个处理后的因子用于组合：

```python
# 加载多个处理后的因子
factor_names = ['ROE_ttm', 'ROA_ttm', 'EP_ttm', 'BP']
factors_dict = {}

for name in factor_names:
    factor = pipeline.load_processed_factor(name)
    if factor is not None:
        factors_dict[name] = factor
        print(f"加载 {name}: {len(factor)} 条数据")

# 合并所有因子为DataFrame
import pandas as pd
factors_df = pd.DataFrame(factors_dict)
print(f"因子组合: {factors_df.shape}")

# 进行因子组合分析或建模
# ...
```

### 场景3：不同配置对比

生成不同去极值参数的因子并对比：

```python
# 生成不同去极值参数的因子
outlier_params = [3, 5, 7]

for param in outlier_params:
    result = pipeline.run(
        'ROE_ttm',
        outlier_param=param,
        save_result=False,
        save_processed_factor=True
    )
    print(f"outlier_param={param}, IC={result.ic_result.ic_mean:.4f}")

# 加载并对比不同配置的因子
for param in outlier_params:
    factor = pipeline.load_processed_factor(
        'ROE_ttm',
        config_key=f'neutral_industry_outlier{param}_zscore'
    )
    # 分析不同配置的影响
    print(f"outlier{param}: {len(factor)} 条数据")
```

## 最佳实践

### 1. 组织因子存储

建议使用有意义的子文件夹名称组织因子：

```python
# 按股票池组织
pipeline.export_processed_factor('ROE_ttm', subfolder='liquid_1000')
pipeline.export_processed_factor('ROE_ttm', subfolder='large_cap_500')

# 按策略组织
pipeline.export_processed_factor('ROE_ttm', subfolder='value_strategy')
pipeline.export_processed_factor('ROE_ttm', subfolder='momentum_strategy')

# 按配置组织（自动）
result = pipeline.run('ROE_ttm')  # 自动按配置保存
```

### 2. 定期更新因子

建议定期更新处理后的因子以包含最新数据：

```python
# 批量更新因子
factors_to_update = ['ROE_ttm', 'ROA_ttm', 'EP_ttm', 'BP']

for factor_name in factors_to_update:
    print(f"更新因子: {factor_name}")
    pipeline.export_processed_factor(factor_name)
```

### 3. 版本记录

利用元数据记录因子的版本和配置信息：

```python
# 导出因子时记录版本信息
pipeline.export_processed_factor(
    'ROE_ttm',
    subfolder='version_2025_01',
    netral_base=True,
    use_industry=True
)

# 查看历史版本
metadata = pipeline.load_factor_metadata('ROE_ttm')
print(f"因子生成时间: {metadata['test_time']}")
print(f"测试ID: {metadata['test_id']}")
```

## 处理后因子的内容

处理后的因子包含以下处理步骤：

1. **去极值处理**: 使用IQR方法去除极端值（默认参数3）
2. **标准化**: zscore标准化，均值0、标准差1
3. **基准因子中性化**: 如果`netral_base=True`，去除基准因子（市值、估值等）的影响
4. **行业中性化**: 如果`use_industry=True`，去除行业影响
5. **重新标准化**: 中性化后再次标准化

## FAQ

### Q1: 处理后的因子可以直接用于建模吗？

A: 可以。处理后的因子已经完成了：
- 去极值
- 归一化
- 中性化（如果配置了）

可以直接用于机器学习模型或线性回归。

### Q2: 如何查看某个因子有哪些版本？

A: 使用 `list_processed_factors` 方法查看所有版本：

```python
factors_df = pipeline.list_processed_factors(return_metadata=True)
roe_versions = factors_df[factors_df['factor_name'] == 'ROE_ttm']
print(roe_versions[['subfolder', 'test_time', 'ic_mean']])
```

### Q3: 处理后的因子可以在不同机器上共享吗？

A: 可以。处理后的因子是标准的pickle文件，可以复制到其他机器使用：

```python
# 在机器A上导出
path = pipeline.export_processed_factor('ROE_ttm')

# 将整个OrthogonalizationFactors目录复制到机器B

# 在机器B上加载
factor = pipeline.load_processed_factor('ROE_ttm')
```

### Q4: 如何删除不再需要的处理后因子？

A: 直接删除对应的文件夹或文件即可：

```bash
# 删除特定配置的因子
rm -rf StockData/OrthogonalizationFactors/neutral_industry_outlier5_zscore/

# 删除特定因子的所有版本
find StockData/OrthogonalizationFactors/ -name "ROE_ttm.pkl" -delete
find StockData/OrthogonalizationFactors/ -name "ROE_ttm_metadata.json" -delete
```

### Q5: 处理后的因子数据格式是什么？

A: 处理后的因子是标准的MultiIndex Series：

```python
factor = pipeline.load_processed_factor('ROE_ttm')
print(type(factor))  # pandas.core.series.Series
print(factor.index.names)  # ['TradingDates', 'StockCodes']
print(factor.dtype)  # float64
```

## 总结

v2.1的因子导出和导入功能带来了以下好处：

1. **复用性**: 处理后的因子可以重复使用，避免重复计算
2. **一致性**: 保证相同配置下的因子处理结果一致
3. **可追溯性**: 元数据记录完整的处理配置和性能指标
4. **灵活性**: 支持多种配置组织和管理方式
5. **效率提升**: 大幅减少因子处理的计算时间
