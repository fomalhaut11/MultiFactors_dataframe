# Factors模块编码规范

## 📋 目录

1. [模块结构规范](#1-模块结构规范)
2. [代码风格规范](#2-代码风格规范)
3. [数据格式规范](#3-数据格式规范)
4. [命名规范](#4-命名规范)
5. [配置管理规范](#5-配置管理规范)
6. [错误处理规范](#6-错误处理规范)
7. [文档规范](#7-文档规范)
8. [测试规范](#8-测试规范)
9. [性能规范](#9-性能规范)
10. [版本管理规范](#10-版本管理规范)

---

## 1. 模块结构规范

### 1.1 目录组织

```
factors/
├── generator/          # 因子生成模块
│   ├── financial/      # 财务因子
│   ├── technical/      # 技术因子
│   ├── risk/          # 风险因子
│   └── alternative/    # 另类因子（新增时）
├── tester/            # 因子测试模块
├── analyzer/          # 因子分析模块
├── combiner/          # 因子组合模块（新增时）
├── base/              # 基础类和混入类
└── utils/             # 工具函数
```

### 1.2 模块接口设计

每个模块的`__init__.py`必须包含：

```python
"""
模块说明文档

描述模块的主要功能和用途
"""

# 导入公共接口
from .core.main_class import MainClass
from .utils.helper import helper_function

# 明确声明公共接口
__all__ = [
    'MainClass',
    'helper_function',
]

# 版本信息
__version__ = '1.0.0'

# 便捷函数（可选）
def quick_function(**kwargs):
    """便捷函数说明"""
    return MainClass(**kwargs).run()
```

---

## 2. 代码风格规范

### 2.1 文件头部

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模块名称

模块详细说明，包括：
- 主要功能
- 使用场景
- 注意事项

Author: [作者名]
Date: [创建日期]
"""
```

### 2.2 导入规范

```python
# 标准库
import os
import sys
from pathlib import Path
from typing import Optional, Dict, List, Union, Any

# 第三方库
import pandas as pd
import numpy as np

# 项目内部模块
from ..base import FactorBase
from core.config_manager import config
```

### 2.3 类设计规范

#### 基础因子类

```python
class NewFactor(FactorBase):
    """
    因子说明
    
    计算逻辑：
    1. 步骤1
    2. 步骤2
    
    Attributes
    ----------
    param1 : type
        参数说明
    """
    
    def __init__(self, param1: int = 20, **kwargs):
        """
        初始化
        
        Parameters
        ----------
        param1 : int
            参数说明
        """
        # 设置默认值
        kwargs.setdefault('name', 'NewFactor')
        kwargs.setdefault('category', 'technical')
        super().__init__(**kwargs)
        
        # 因子特定参数
        self.param1 = param1
        
    def calculate(self, data: pd.Series, **kwargs) -> pd.Series:
        """
        计算因子值
        
        Parameters
        ----------
        data : pd.Series
            输入数据，MultiIndex[TradingDates, StockCodes]
            
        Returns
        -------
        pd.Series
            因子值，MultiIndex格式与输入数据一致
        """
        # 实现计算逻辑
        return result
```

#### 生成器类

```python
class NewFactorGenerator(FactorGenerator):
    """生成器说明"""
    
    def __init__(self, **kwargs):
        super().__init__(factor_type='new_type', **kwargs)
        
    def generate(self, factor_name: str, data: pd.Series, **kwargs) -> pd.Series:
        """生成因子"""
        # 实现生成逻辑
        return factor_data
        
    def get_available_factors(self) -> List[str]:
        """返回可用因子列表"""
        return ['Factor1', 'Factor2']
```

---

## 3. 数据格式规范

### 3.1 标准数据格式：MultiIndex Series

```python
# 统一格式：MultiIndex Series
# - 第一级索引：TradingDates（交易日期）
# - 第二级索引：StockCodes（股票代码，如'000001.SZ'）
# - values: 数值型数据（因子值）

# 创建示例
dates = pd.date_range('2024-01-01', periods=3)
stocks = ['000001.SZ', '000002.SZ']
index = pd.MultiIndex.from_product([dates, stocks], 
                                  names=['TradingDates', 'StockCodes'])
factor_data = pd.Series([100, 200, 101, 201, 102, 202], index=index)
```

### 3.2 输入输出格式

```python
# 因子数据：MultiIndex Series
# 收益率数据：MultiIndex Series
# 测试结果：TestResult对象
# 分析结果：字典或MultiIndex Series

# 验证数据格式
def validate_factor_format(factor_data):
    assert isinstance(factor_data, pd.Series)
    assert isinstance(factor_data.index, pd.MultiIndex)
    assert factor_data.index.names == ['TradingDates', 'StockCodes']
    return True
```

### 3.3 DataFrame兼容性转换

```python
# 从DataFrame转换为MultiIndex Series
def dataframe_to_multiindex(df: pd.DataFrame) -> pd.Series:
    """
    将DataFrame（日期为index，股票为columns）转换为MultiIndex Series
    """
    # Stack操作：将列转换为第二级索引
    series = df.stack()
    series.index.names = ['TradingDates', 'StockCodes']
    return series

# 从MultiIndex Series转换为DataFrame（仅在需要时）
def multiindex_to_dataframe(series: pd.Series) -> pd.DataFrame:
    """
    将MultiIndex Series转换为DataFrame格式
    """
    return series.unstack(level='StockCodes')
```

### 3.4 缺失值处理

```python
# MultiIndex Series缺失值处理
def handle_missing_data(data: pd.Series) -> pd.Series:
    # 1. 记录缺失情况
    missing_ratio = data.isna().sum() / len(data)
    logger.info(f"缺失值比例: {missing_ratio:.2%}")
    
    # 2. 按日期分组处理
    def process_daily(group):
        # 可选策略：
        # - 向前填充：group.fillna(method='ffill')
        # - 均值填充：group.fillna(group.mean())
        # - 删除：group.dropna()
        return group.fillna(group.mean())
    
    # 按交易日期分组处理
    processed_data = data.groupby(level='TradingDates').apply(process_daily)
    
    return processed_data
```

---

## 4. 命名规范

### 4.1 因子命名

| 类型 | 格式 | 示例 |
|------|------|------|
| 基础因子 | `FactorName` | `ROE`, `Beta` |
| TTM因子 | `FactorName_ttm` | `ROE_ttm`, `ROA_ttm` |
| 窗口因子 | `FactorName_Nd` | `Momentum_20d`, `Volatility_60d` |
| 年度因子 | `FactorName_Ny` | `Growth_3y`, `Revenue_1y` |

### 4.2 代码命名

```python
# 类名：PascalCase
class FactorCalculator:
    pass

# 函数名：snake_case
def calculate_factor():
    pass

# 常量：UPPER_CASE
DEFAULT_WINDOW = 20
MAX_ITERATIONS = 1000

# 私有成员：前缀下划线
def _internal_function():
    pass

class MyClass:
    def __init__(self):
        self._private_attr = None
```

### 4.3 文件命名

```
factor_base.py          # 模块文件：snake_case
test_factor_base.py     # 测试文件：test_前缀
README.md              # 文档文件：大写
config.yaml            # 配置文件：小写
```

---

## 5. 配置管理规范

### 5.1 配置层级

```python
# 1. 全局配置（config.yaml）
global_config = {
    'database': {...},
    'paths': {...}
}

# 2. 模块配置（module/config.py）
MODULE_CONFIG = {
    'default_window': 20,
    'min_samples': 100
}

# 3. 实例配置（运行时）
factor = NewFactor(window=30)  # 覆盖默认值
```

### 5.2 路径管理

```python
from core.config_manager import get_path

# 统一使用配置管理器获取路径
factor_dir = Path(get_path('factors'))
test_dir = Path(get_path('single_factor_test'))

# 确保目录存在
factor_dir.mkdir(parents=True, exist_ok=True)
```

### 5.3 参数配置

```python
# 使用数据类管理参数
from dataclasses import dataclass

@dataclass
class FactorConfig:
    """因子配置"""
    window: int = 20
    min_periods: int = 10
    method: str = 'rolling'
    
# 使用配置
config = FactorConfig(window=30)
factor = NewFactor(**config.__dict__)
```

---

## 6. 错误处理规范

### 6.1 日志使用

```python
import logging
logger = logging.getLogger(__name__)

# 日志级别使用
logger.debug("详细调试信息")
logger.info(f"开始计算因子: {factor_name}")
logger.warning(f"数据缺失率高: {missing_ratio:.2%}")
logger.error(f"计算失败: {e}")
logger.critical("严重错误，程序终止")
```

### 6.2 异常处理

```python
def calculate_factor(data: pd.Series) -> pd.Series:
    """标准异常处理模式"""
    
    # 输入验证
    if data.empty:
        raise ValueError("输入数据为空")
    
    if not isinstance(data, pd.Series):
        raise TypeError(f"期望MultiIndex Series，得到{type(data)}")
    
    if not isinstance(data.index, pd.MultiIndex):
        raise ValueError("数据必须是MultiIndex格式[TradingDates, StockCodes]")
    
    try:
        # 核心计算逻辑
        result = complex_calculation(data)
        
    except KeyError as e:
        logger.error(f"缺少必要的列: {e}")
        raise
        
    except Exception as e:
        logger.error(f"计算过程出错: {e}")
        # 返回空结果而不是崩溃
        return pd.DataFrame()
        
    return result
```

### 6.3 数据验证

```python
def validate_data(data: pd.Series) -> bool:
    """数据验证"""
    
    # 基础检查
    if data.empty:
        logger.error("数据为空")
        return False
    
    # MultiIndex格式检查
    if not isinstance(data.index, pd.MultiIndex):
        logger.error("数据不是MultiIndex格式")
        return False
    
    # 索引名称检查
    if data.index.names != ['TradingDates', 'StockCodes']:
        logger.error(f"索引名称错误: {data.index.names}")
        return False
        
    # 数据类型检查
    if not np.issubdtype(data.dtype, np.number):
        logger.error("包含非数值数据")
        return False
        
    # 数据质量检查
    missing_ratio = data.isna().sum() / len(data)
    if missing_ratio > 0.5:
        logger.warning(f"缺失值过多: {missing_ratio:.2%}")
        
    return True
```

---

## 7. 文档规范

### 7.1 Docstring格式（NumPy风格）

```python
def function_name(param1: int, param2: str = 'default') -> pd.DataFrame:
    """
    函数简要说明（一行）
    
    详细说明（可选），可以包含：
    - 算法描述
    - 使用场景
    - 注意事项
    
    Parameters
    ----------
    param1 : int
        第一个参数的说明
    param2 : str, optional
        第二个参数的说明，默认值为'default'
        
    Returns
    -------
    pd.DataFrame
        返回值的说明
        
    Raises
    ------
    ValueError
        当param1小于0时抛出
        
    See Also
    --------
    related_function : 相关函数
    
    Notes
    -----
    额外说明信息
    
    Examples
    --------
    >>> result = function_name(10, 'test')
    >>> print(result.shape)
    (100, 50)
    """
    pass
```

### 7.2 模块文档

每个模块必须包含README.md：

```markdown
# 模块名称

## 概述
模块功能简介

## 功能特性
- 特性1
- 特性2

## 使用方法
```python
from module import Function
result = Function()
```

## API文档
详细的接口说明

## 更新日志
- v1.0.0: 初始版本
```

---

## 8. 测试规范

### 8.1 单元测试

```python
# test_new_factor.py
import pytest
import pandas as pd
import numpy as np
from factors.generator.technical import NewFactor

class TestNewFactor:
    """NewFactor测试类"""
    
    @pytest.fixture
    def sample_data(self):
        """准备测试数据"""
        return pd.DataFrame(
            np.random.randn(100, 10),
            index=pd.date_range('2024-01-01', periods=100),
            columns=[f'stock_{i}' for i in range(10)]
        )
    
    def test_calculate(self, sample_data):
        """测试计算功能"""
        factor = NewFactor()
        result = factor.calculate(sample_data)
        
        # 验证输出格式
        assert isinstance(result, pd.DataFrame)
        assert result.shape == sample_data.shape
        
    def test_edge_cases(self):
        """测试边界情况"""
        factor = NewFactor()
        
        # 空数据
        empty_data = pd.DataFrame()
        with pytest.raises(ValueError):
            factor.calculate(empty_data)
            
    def test_performance(self, sample_data):
        """性能测试"""
        import time
        factor = NewFactor()
        
        start = time.time()
        result = factor.calculate(sample_data)
        elapsed = time.time() - start
        
        assert elapsed < 1.0  # 应在1秒内完成
```

### 8.2 集成测试

```python
def test_end_to_end_workflow():
    """端到端工作流测试"""
    
    # 1. 生成因子
    from factors import generate
    factor_data = generate('NewFactor', test_data)
    
    # 2. 测试因子
    from factors import test
    test_result = test('NewFactor')
    
    # 3. 分析因子
    from factors import analyze
    analysis = analyze(['NewFactor'])
    
    # 验证完整流程
    assert factor_data is not None
    assert test_result.ic_result is not None
    assert 'NewFactor' in analysis
```

---

## 9. 性能规范

### 9.1 优化原则

```python
# ✅ 好的实践：向量化操作
result = data.rolling(window).mean()

# ❌ 避免：Python循环
result = []
for i in range(len(data)):
    result.append(data[i:i+window].mean())
```

### 9.2 内存管理

```python
# 大数据分块处理
def process_large_data(data: pd.DataFrame, chunk_size: int = 10000):
    """分块处理大数据"""
    results = []
    
    for i in range(0, len(data), chunk_size):
        chunk = data.iloc[i:i+chunk_size]
        result = process_chunk(chunk)
        results.append(result)
        
        # 及时释放内存
        del chunk
        
    return pd.concat(results)
```

### 9.3 性能监控

```python
from functools import wraps
import time

def timer(func):
    """性能计时装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logger.info(f"{func.__name__} 耗时: {elapsed:.2f}秒")
        return result
    return wrapper

@timer
def calculate_complex_factor(data):
    # 复杂计算
    pass
```

---

## 10. 版本管理规范

### 10.1 版本号规则

遵循语义化版本 2.0.0：`MAJOR.MINOR.PATCH`

- **MAJOR**: 不兼容的API变更
- **MINOR**: 向后兼容的功能新增
- **PATCH**: 向后兼容的问题修复

### 10.2 版本兼容性

```python
import warnings

# 废弃警告
def deprecated_function():
    warnings.warn(
        "deprecated_function将在v2.0.0中移除，请使用new_function",
        DeprecationWarning,
        stacklevel=2
    )
    return new_function()

# 版本检查
def check_version(required_version: str):
    from packaging import version
    current = version.parse(__version__)
    required = version.parse(required_version)
    
    if current < required:
        raise RuntimeError(f"需要版本 {required_version}，当前版本 {__version__}")
```

### 10.3 变更日志

维护CHANGELOG.md：

```markdown
# 变更日志

## [2.0.0] - 2025-08-12
### 变更
- 重构factors模块结构
- 统一接口设计

### 新增
- 添加SUE因子
- 实现因子生成器基类

### 修复
- 修复IC计算错误
```

---

## 📋 检查清单

开发新功能前，请确认：

- [ ] 遵循目录结构规范
- [ ] 继承正确的基类
- [ ] 实现必要的抽象方法
- [ ] 添加完整的docstring
- [ ] 编写单元测试
- [ ] 更新模块的`__init__.py`
- [ ] 添加到`__all__`列表
- [ ] 更新版本号
- [ ] 更新文档
- [ ] 通过代码审查

---

*最后更新：2025-08-12*