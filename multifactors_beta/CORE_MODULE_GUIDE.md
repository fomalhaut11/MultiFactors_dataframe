# Core模块使用指南

## 概述

`core`模块是项目的统一入口，提供了访问核心功能的便捷接口。通过core模块，您可以快速进行单因子测试、因子筛选、批量处理等操作，无需深入了解底层模块结构。

## 快速开始

### 导入core模块

```python
from core import (
    test_single_factor,     # 单因子测试
    screen_factors,         # 因子筛选
    batch_test_factors,     # 批量测试
    load_factor_data,       # 加载因子数据
    get_factor_test_result, # 获取测试结果
    get_config,            # 获取配置
    get_path               # 获取路径
)
```

## 核心功能

### 1. 单因子测试

最简单的因子测试只需一行代码：

```python
from core import test_single_factor

# 测试BP因子
result = test_single_factor('BP', begin_date='2024-01-01', end_date='2024-12-31')

# 查看结果
print(f"IC: {result.ic_result.ic_mean:.4f}")
print(f"ICIR: {result.ic_result.icir:.4f}")
```

### 2. 批量测试因子

同时测试多个因子：

```python
from core import batch_test_factors

# 批量测试
results = batch_test_factors(['BP', 'EP_ttm', 'ROE_ttm', 'SP_ttm'])

# 查看结果
for factor, result in results.items():
    if result:
        print(f"{factor}: IC={result.ic_result.ic_mean:.4f}")
```

### 3. 因子筛选

从已测试的因子中筛选高质量因子：

```python
from core import screen_factors

# 使用预设标准
top_factors = screen_factors(preset='normal')  # 'strict', 'normal', 'loose'

# 使用自定义标准
custom_factors = screen_factors(criteria={
    'ic_mean_min': 0.02,
    'icir_min': 0.5,
    'monotonicity_min': 0.6
})
```

### 4. 加载因子数据

直接加载因子数据进行分析：

```python
from core import load_factor_data

# 加载BP因子
bp_factor = load_factor_data('BP')
print(f"因子形状: {bp_factor.shape}")
print(f"均值: {bp_factor.mean():.4f}")
```

### 5. 获取历史测试结果

查看之前的测试结果：

```python
from core import get_factor_test_result

# 获取最新测试结果
result = get_factor_test_result('BP')
if result:
    print(f"测试时间: {result.test_time}")
    print(f"IC: {result.ic_result.ic_mean:.4f}")
```

## 完整示例

### 示例1: 因子研究工作流

```python
from core import test_single_factor, screen_factors, batch_test_factors

# Step 1: 测试新因子
new_factor_result = test_single_factor('NEW_FACTOR')

# Step 2: 批量测试候选因子
candidates = ['BP', 'EP_ttm', 'ROE_ttm', 'SP_ttm']
all_results = batch_test_factors(candidates)

# Step 3: 筛选高质量因子
top_factors = screen_factors(preset='strict')
print(f"高质量因子: {top_factors}")
```

### 示例2: 因子对比分析

```python
from core import load_factor_data, test_single_factor

# 加载并对比两个因子
bp_data = load_factor_data('BP')
ep_data = load_factor_data('EP_ttm')

# 测试并对比
bp_result = test_single_factor('BP', begin_date='2024-01-01', end_date='2024-06-30')
ep_result = test_single_factor('EP_ttm', begin_date='2024-01-01', end_date='2024-06-30')

print(f"BP - IC: {bp_result.ic_result.ic_mean:.4f}, ICIR: {bp_result.ic_result.icir:.4f}")
print(f"EP - IC: {ep_result.ic_result.ic_mean:.4f}, ICIR: {ep_result.ic_result.icir:.4f}")
```

## 配置管理

Core模块也提供配置管理功能：

```python
from core import get_config, get_path

# 获取配置
db_config = get_config('database')
test_config = get_config('factor_test')

# 获取路径
factor_path = get_path('raw_factors')  # 因子数据路径
test_path = get_path('single_factor_test')  # 测试结果路径
```

## 参数说明

### test_single_factor 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| factor_name | str | 必需 | 因子名称 |
| begin_date | str | None | 开始日期 (YYYY-MM-DD) |
| end_date | str | None | 结束日期 (YYYY-MM-DD) |
| group_nums | int | 5 | 分组数量 |
| save_result | bool | True | 是否保存结果 |
| netral_base | bool | False | 是否市值中性化 |
| use_industry | bool | False | 是否使用行业信息 |

### screen_factors 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| criteria | dict | None | 自定义筛选标准 |
| preset | str | None | 预设标准 ('strict', 'normal', 'loose') |

### 筛选标准字段

| 字段 | 说明 | 建议值 |
|------|------|--------|
| ic_mean_min | IC均值最小值 | > 0.02 |
| icir_min | ICIR最小值 | > 0.5 |
| monotonicity_min | 单调性最小值 | > 0.6 |
| sharpe_min | 夏普比率最小值 | > 1.0 |
| t_value_min | t值最小值 | > 2.0 |

## 错误处理

```python
from core import test_single_factor

try:
    result = test_single_factor('UNKNOWN_FACTOR')
except FileNotFoundError:
    print("因子不存在")
except Exception as e:
    print(f"测试失败: {e}")
```

## 性能优化建议

1. **批量处理**: 使用`batch_test_factors`而不是循环调用`test_single_factor`
2. **缓存利用**: Core模块会自动缓存常用数据
3. **并行计算**: 批量测试会自动优化计算顺序

## 扩展功能

Core模块是可扩展的，您可以在`core/__init__.py`中添加自定义功能：

```python
# core/__init__.py
def my_custom_function(factor_name):
    """您的自定义功能"""
    pass

# 然后在其他地方使用
from core import my_custom_function
```

## 相关文档

- [单因子测试模块详细文档](factors/tester/README.md)
- [因子筛选模块详细文档](docs/因子筛选分析模块使用指南.md)
- [项目快速开始指南](QUICK_START.md)

## 更新日志

- **2025-08-11**: 创建core模块统一入口
- **2025-08-11**: 添加单因子测试快捷函数
- **2025-08-11**: 添加因子筛选快捷函数
- **2025-08-11**: 添加批量处理功能

---
*通过core模块，让因子研究更简单高效！*