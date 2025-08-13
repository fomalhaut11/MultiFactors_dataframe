# Core模块 - 统一入口

## 概述

Core模块是整个多因子量化系统的核心入口，提供了统一的API接口来访问系统的主要功能，包括单因子测试、因子筛选、配置管理和数据库连接等。

## 主要功能

### 1. 单因子测试
- `test_single_factor()` - 测试单个因子
- `batch_test_factors()` - 批量测试多个因子
- `get_factor_test_result()` - 获取历史测试结果

### 2. 因子筛选
- `screen_factors()` - 根据标准筛选高质量因子
- `load_factor_data()` - 加载因子数据

### 3. 配置管理
- `get_config()` - 获取配置项
- `get_path()` - 获取路径配置

### 4. 数据库管理
- `DatabaseManager` - 数据库连接管理
- `SQLExecutor` - SQL执行器

## 快速开始

### 安装导入

```python
from core import test_single_factor, screen_factors, load_factor_data
```

### 测试单个因子

```python
from core import test_single_factor

# 测试BP因子
result = test_single_factor(
    'BP',
    begin_date='2024-01-01',
    end_date='2024-12-31',
    group_nums=5
)

# 查看结果
if result:
    print(f"IC: {result.ic_result.ic_mean:.4f}")
    print(f"ICIR: {result.ic_result.icir:.4f}")
    print(f"夏普比率: {result.performance_metrics.get('long_short_sharpe', 0):.4f}")
```

### 批量测试因子

```python
from core import batch_test_factors

# 批量测试多个因子
factors = ['BP', 'EP_ttm', 'ROE_ttm', 'SP_ttm']
results = batch_test_factors(
    factors,
    begin_date='2024-01-01',
    end_date='2024-12-31'
)

# 显示结果汇总
for factor, result in results.items():
    if result:
        print(f"{factor}: IC={result.ic_result.ic_mean:.4f}")
```

### 筛选高质量因子

```python
from core import screen_factors

# 使用预设标准筛选
top_factors = screen_factors(preset='normal')
print(f"筛选出 {len(top_factors)} 个高质量因子")

# 使用自定义标准
custom_factors = screen_factors({
    'ic_mean_min': 0.02,
    'icir_min': 0.5,
    'monotonicity_min': 0.6
})
```

### 加载因子数据

```python
from core import load_factor_data

# 加载BP因子数据
bp_factor = load_factor_data('BP')
print(f"因子形状: {bp_factor.shape}")
print(f"因子均值: {bp_factor.mean():.4f}")
```

### 获取测试结果

```python
from core import get_factor_test_result

# 获取BP因子的最新测试结果
result = get_factor_test_result('BP')
if result:
    print(f"测试时间: {result.test_time}")
    print(f"IC均值: {result.ic_result.ic_mean:.4f}")
```

## API参考

### 因子测试函数

#### `test_single_factor(factor_name, **kwargs)`
测试单个因子的预测能力

**参数:**
- `factor_name` (str): 因子名称
- `begin_date` (str): 开始日期，格式'YYYY-MM-DD'
- `end_date` (str): 结束日期
- `group_nums` (int): 分组数量，默认5
- `save_result` (bool): 是否保存结果，默认True
- `netral_base` (bool): 是否中性化，默认False
- `use_industry` (bool): 是否使用行业，默认False

**返回:**
- `TestResult`: 测试结果对象

#### `batch_test_factors(factor_list, **kwargs)`
批量测试多个因子

**参数:**
- `factor_list` (list): 因子名称列表
- `**kwargs`: 同`test_single_factor`的参数

**返回:**
- `dict`: 因子名称到测试结果的映射

#### `screen_factors(criteria=None, preset=None)`
筛选高质量因子

**参数:**
- `criteria` (dict): 自定义筛选标准
  - `ic_mean_min`: IC均值最小阈值
  - `icir_min`: ICIR最小阈值
  - `monotonicity_min`: 单调性最小阈值
- `preset` (str): 预设标准 ('strict', 'normal', 'loose')

**返回:**
- `list`: 筛选出的因子名称列表

### 数据管理函数

#### `load_factor_data(factor_name)`
加载因子数据

**参数:**
- `factor_name` (str): 因子名称

**返回:**
- `pd.Series`: 因子数据

**异常:**
- `FileNotFoundError`: 因子文件不存在

#### `get_factor_test_result(factor_name)`
获取因子的最新测试结果

**参数:**
- `factor_name` (str): 因子名称

**返回:**
- `TestResult` or `None`: 测试结果对象

### 配置管理函数

#### `get_config(section=None)`
获取配置信息

**参数:**
- `section` (str): 配置节名称，如'database', 'paths'等

**返回:**
- `dict`: 配置字典

#### `get_path(key)`
获取路径配置

**参数:**
- `key` (str): 路径键名
  - `'data_root'`: 数据根目录
  - `'raw_factors'`: 因子存储目录
  - `'single_factor_test'`: 测试结果目录

**返回:**
- `str`: 路径字符串

## 完整示例

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用Core模块进行因子研究的完整流程
"""

from core import (
    test_single_factor,
    screen_factors,
    batch_test_factors,
    load_factor_data,
    get_factor_test_result
)

def main():
    # 1. 测试新因子
    print("步骤1: 测试BP因子")
    result = test_single_factor('BP', begin_date='2024-01-01', end_date='2024-06-30')
    if result:
        print(f"  IC: {result.ic_result.ic_mean:.4f}")
        print(f"  ICIR: {result.ic_result.icir:.4f}")
    
    # 2. 批量测试
    print("\n步骤2: 批量测试因子")
    factors = ['BP', 'EP_ttm', 'SP_ttm']
    results = batch_test_factors(factors, begin_date='2024-01-01', end_date='2024-06-30')
    
    # 3. 筛选因子
    print("\n步骤3: 筛选高质量因子")
    top_factors = screen_factors(preset='normal')
    print(f"  筛选出: {top_factors}")
    
    # 4. 加载因子数据进行分析
    print("\n步骤4: 加载因子数据")
    if 'BP' in top_factors:
        bp_data = load_factor_data('BP')
        print(f"  BP因子数据点: {len(bp_data):,}")
    
    # 5. 查看历史测试结果
    print("\n步骤5: 查看历史测试结果")
    for factor in top_factors[:3]:
        result = get_factor_test_result(factor)
        if result:
            print(f"  {factor}: IC={result.ic_result.ic_mean:.4f}")

if __name__ == "__main__":
    main()
```

## 目录结构

```
core/
├── __init__.py           # 统一入口，API定义
├── README.md            # 本文档
├── config_manager.py    # 配置管理
├── database/           # 数据库相关
│   ├── connection_manager.py
│   └── sql_executor.py
├── utils/              # 工具函数
│   ├── data_cleaning.py
│   ├── factor_processing.py
│   └── technical_indicators.py
├── cache/              # 缓存目录
└── logs/               # 日志目录
```

## 配置说明

Core模块使用项目根目录的`config.yaml`作为主配置文件：

```yaml
paths:
  data_root: 'E:/Documents/PythonProject/StockProject/StockData'
  raw_factors: '.../RawFactors'
  single_factor_test: '.../SingleFactorTestData'

database:
  host: '${DB_HOST}'
  user: '${DB_USER}'
  password: '${DB_PASSWORD}'
  database: 'stock_data'
```

## 最佳实践

1. **统一入口**: 优先使用core模块的API，而不是直接导入子模块
2. **错误处理**: 所有函数都包含了错误处理，返回None表示失败
3. **批量操作**: 使用批量函数可以提高效率
4. **配置管理**: 通过core模块统一管理配置，避免硬编码

## 相关文档

- [单因子测试模块](../factors/tester/README.md)
- [因子筛选模块](../factors/analyzer/README.md)
- [配置管理指南](../docs/配置管理指南.md)

## 版本历史

- v1.0.0 (2025-08): 初始版本，提供基础API
- v1.1.0 (2025-08): 添加统一入口功能

---
*最后更新: 2025-08-12*