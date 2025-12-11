# 模块接口设计规范

## 核心原则

### 1. 接口与实现分离
- 所有公共接口必须在模块的`__init__.py`中导出
- 内部实现细节不应该被外部直接访问
- 使用`__all__`明确声明公共接口

### 2. 稳定的API契约
- 一旦接口发布，保持向后兼容
- 内部实现可以自由重构，但接口保持稳定
- 废弃的接口使用`warnings`模块发出警告

## 标准模块结构

```python
# module/__init__.py
"""
模块说明文档

描述模块的主要功能和用途
"""

# 从内部模块导入公共接口
from .core.main_class import MainClass
from .utils.helper import helper_function
from .config import get_config

# 明确声明公共接口
__all__ = [
    'MainClass',
    'helper_function', 
    'get_config',
]

# 版本信息
__version__ = '1.0.0'

# 可选：提供便捷函数
def quick_start(**kwargs):
    """提供快速开始的便捷函数"""
    return MainClass(**kwargs).run()
```

## 最佳实践示例

### 1. factors.tester模块

```python
# factors/tester/__init__.py
"""单因子测试模块"""

from .core.pipeline import SingleFactorTestPipeline
from .base.test_result import TestResult
from .core.result_manager import ResultManager

__all__ = [
    'SingleFactorTestPipeline',  # 主要接口
    'TestResult',                # 结果类
    'ResultManager',             # 结果管理
]

# 便捷函数
def test_factor(factor_name, **kwargs):
    """快速测试单个因子"""
    pipeline = SingleFactorTestPipeline()
    return pipeline.run(factor_name, **kwargs)
```

### 2. factors.analyzer模块

```python
# factors/analyzer/__init__.py
"""因子分析模块"""

from .screening import FactorScreener
from .config import get_analyzer_config

__all__ = [
    'FactorScreener',       # 筛选器
    'get_analyzer_config',  # 配置
]

# 便捷函数
def screen_factors(preset='normal'):
    """快速筛选因子"""
    screener = FactorScreener()
    return screener.screen_factors(preset=preset)
```

## 调用示例

### 好的实践 ✅

```python
# 使用模块的公共接口
from factors.tester import SingleFactorTestPipeline
from factors.analyzer import FactorScreener

# 或者使用便捷函数
from factors.tester import test_factor
result = test_factor('BP')
```

### 不好的实践 ❌

```python
# 直接访问内部实现
from factors.tester.core.pipeline import SingleFactorTestPipeline  # 避免
from factors.analyzer.screening.factor_screener import FactorScreener  # 避免
```

## 重构示例

### 重构前的调用
```python
# 其他模块的代码
from factors.tester.core.pipeline import SingleFactorTestPipeline
from factors.tester.core.data_manager import DataManager
```

### 重构后的调用
```python
# 其他模块的代码（不需要改变）
from factors.tester import SingleFactorTestPipeline, DataManager
```

### 内部重构示例
即使我们重新组织了内部文件结构：
```
# 从
factors/tester/core/pipeline.py
# 改为
factors/tester/engine/test_pipeline.py
```

只需要更新`__init__.py`：
```python
# 原来
from .core.pipeline import SingleFactorTestPipeline
# 改为
from .engine.test_pipeline import SingleFactorTestPipeline
```

外部调用代码完全不需要修改！

## 接口版本管理

### 1. 添加新功能
```python
# __init__.py
__all__ = [
    'OldClass',
    'NewClass',  # 新增
]
```

### 2. 废弃旧接口
```python
import warnings

def deprecated_function():
    warnings.warn(
        "deprecated_function is deprecated, use new_function instead",
        DeprecationWarning,
        stacklevel=2
    )
    return new_function()
```

### 3. 版本兼容性
```python
# 保持向后兼容
try:
    from .new_module import NewClass
except ImportError:
    from .old_module import OldClass as NewClass
```

## 文档要求

每个模块的`__init__.py`应包含：

1. **模块文档字符串**
   - 简要说明模块功能
   - 列出主要组件

2. **导入语句**
   - 只导入公共接口
   - 使用相对导入

3. **__all__列表**
   - 明确列出所有公共接口
   - 按重要性排序

4. **版本信息**
   - `__version__`变量

5. **便捷函数**（可选）
   - 提供快速使用的函数

## 检查清单

- [ ] 所有公共接口都在`__init__.py`中导出？
- [ ] 使用了`__all__`列表？
- [ ] 有模块文档字符串？
- [ ] 内部实现可以自由重构而不影响外部？
- [ ] 提供了便捷函数（如果适用）？
- [ ] 有版本信息？

## 模块间依赖

### 正确的依赖方式
```python
# factors/builder/factor_builder.py
from factors.tester import SingleFactorTestPipeline  # 通过公共接口
from factors.analyzer import FactorScreener

class FactorBuilder:
    def test_new_factor(self, factor_name):
        pipeline = SingleFactorTestPipeline()
        return pipeline.run(factor_name)
```

### 避免循环依赖
```python
# 使用延迟导入避免循环依赖
def process_factor():
    from factors.analyzer import FactorScreener  # 在函数内导入
    screener = FactorScreener()
    return screener.screen()
```

## 实施计划

1. **审查现有模块**
   - 检查所有模块的`__init__.py`
   - 识别未导出的公共接口

2. **统一接口**
   - 将所有公共接口添加到`__all__`
   - 添加便捷函数

3. **更新调用代码**
   - 修改直接访问内部实现的代码
   - 使用模块级别的导入

4. **文档更新**
   - 更新各模块的README
   - 添加接口变更日志

## 总结

通过在`__init__.py`中管理模块接口，我们可以：
- ✅ 提供清晰的API边界
- ✅ 支持内部重构而不影响外部
- ✅ 简化模块使用
- ✅ 便于版本管理和向后兼容

这是Python模块设计的最佳实践，应该在所有模块中坚持使用。