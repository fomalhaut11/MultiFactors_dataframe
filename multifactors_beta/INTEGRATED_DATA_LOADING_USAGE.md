# 集成数据加载接口使用指南

## 概述

现在项目提供了统一的数据加载接口，支持从硬盘加载股票池数据并直接用于核心业务流程。

## 核心功能

### 1. 统一的股票池数据加载

```python
from factors.utils.data_loader import FactorDataLoader, get_stock_universe

# 方法1：使用类方法
universe = FactorDataLoader.load_stock_universe('liquid_1000')

# 方法2：使用便捷函数  
universe = get_stock_universe('liquid_1000')
```

### 2. 支持的股票池类型

```python
# 列出所有可用的股票池
available = FactorDataLoader.list_available_universes()
```

支持的股票池包括：
- `all` / `full`: 全市场股票
- `liquid_1000`: 流动性前1000只股票
- `liquid_300` / `index_300`: 沪深300相关股票池
- `liquid_500` / `index_500`: 中证500相关股票池
- `large_cap_500`: 大盘股前500只
- `main_board`: 主板股票（排除ST）

### 3. 与核心业务流程集成

#### 单因子测试中的使用

```python
from factors.utils.data_loader import get_stock_universe
from factors.tester.core.pipeline import SingleFactorTestPipeline

# 初始化测试流水线
pipeline = SingleFactorTestPipeline()

# 通过统一数据接口加载股票池
stock_universe = get_stock_universe('liquid_1000')

# 在单因子测试中使用
result = pipeline.run(
    factor_name='ROE_ttm',
    stock_universe=stock_universe  # 直接传入MultiIndex Series格式
)
```

#### 批量测试中的使用

```python
# 批量测试不同股票池
universes = ['liquid_300', 'liquid_500', 'liquid_1000']
results = {}

for universe_name in universes:
    universe = get_stock_universe(universe_name)
    result = pipeline.run('ROE_ttm', stock_universe=universe)
    results[universe_name] = result
```

## 数据格式规范

### 输入格式支持
- 股票池管理器支持多种输入格式（list, 文件路径等）

### 输出格式统一
- **统一输出**: MultiIndex Series格式 `[TradingDates, StockCodes]`，值为1
- **时间变化支持**: 支持时变股票池（不同日期包含不同股票）

## 性能优化

### 缓存机制
```python
# 数据会自动缓存，重复调用无需重新计算
universe1 = get_stock_universe('liquid_1000')  # 首次加载，较慢
universe2 = get_stock_universe('liquid_1000')  # 从缓存加载，很快

# 查看缓存状态
cache_info = FactorDataLoader.get_cache_info()

# 清空缓存（如需要）
FactorDataLoader.clear_cache()
```

### 格式转换
- 自动处理不同格式间的转换
- List → MultiIndex Series 转换在后台自动完成

## 数据注册器集成

### 查看已注册的股票池数据集
```python
from core.data_registry import get_data_registry

registry = get_data_registry()
datasets = registry.list_all_datasets()

# 查看股票池相关数据集
universe_datasets = datasets[datasets['name'].str.contains('universe')]
print(universe_datasets)
```

### 已注册的股票池数据集
- `universe_liquid_300`: 沪深300流动性股票池
- `universe_liquid_500`: 中证500流动性股票池  
- `universe_liquid_1000`: 中证1000流动性股票池
- `universe_custom_universe`: 自定义股票池数据

## 配置与个性化

### 自定义股票池
```python
# 加载自定义股票池（需要预先准备文件）
custom_universe = get_stock_universe('my_custom_pool')

# 传递额外参数给股票池管理器
universe = get_stock_universe(
    'liquid_1000',
    min_market_cap=1e9,  # 最小市值筛选
    exclude_st=True      # 排除ST股票
)
```

### 与现有数据加载器的协调
```python
from factors.utils.data_loader import (
    get_daily_returns,    # 加载收益率数据
    get_price_data,       # 加载价格数据  
    get_market_cap,       # 加载市值数据
    get_stock_universe    # 加载股票池数据 - 新增
)

# 统一的数据加载接口
returns = get_daily_returns()
prices = get_price_data()
market_cap = get_market_cap()
universe = get_stock_universe('liquid_1000')
```

## 错误处理

### 常见错误及解决方案

1. **股票池不存在**
```python
try:
    universe = get_stock_universe('non_existent_pool')
except Exception as e:
    print(f"股票池加载失败: {e}")
    # 使用默认股票池
    universe = get_stock_universe('all')
```

2. **数据格式问题** - 自动处理，无需手动干预

3. **内存不足** - 对于大股票池，使用分块处理
```python
# 对于全市场股票池，建议使用较小的股票池进行测试
universe = get_stock_universe('liquid_1000')  # 而不是 'all'
```

## 最佳实践

1. **优先使用预设股票池**: 如`liquid_1000`而不是`all`
2. **利用缓存机制**: 在同一会话中重复使用相同股票池
3. **数据类型一致性**: 始终使用MultiIndex Series格式进行后续计算
4. **错误处理**: 总是包含适当的错误处理逻辑

## 示例：完整的单因子分析工作流

```python
from factors.utils.data_loader import get_stock_universe
from factors.tester.core.pipeline import SingleFactorTestPipeline

def analyze_factor_on_different_universes(factor_name):
    """在不同股票池上分析因子"""
    
    pipeline = SingleFactorTestPipeline()
    results = {}
    
    # 定义要测试的股票池
    universes = {
        'large_cap': 'liquid_300',
        'mid_cap': 'liquid_500', 
        'small_cap': 'liquid_1000'
    }
    
    for label, universe_name in universes.items():
        print(f"分析 {factor_name} 在 {label} 股票池上的表现...")
        
        # 加载股票池
        stock_universe = get_stock_universe(universe_name)
        
        # 执行单因子测试
        result = pipeline.run(
            factor_name=factor_name,
            stock_universe=stock_universe
        )
        
        results[label] = result
        
        print(f"{label} 完成: IC={result.ic_mean:.4f}")
    
    return results

# 使用示例
results = analyze_factor_on_different_universes('ROE_ttm')
```

通过这个集成的数据加载接口，你现在可以：
- 轻松从硬盘加载各种股票池数据
- 在核心业务流程中直接使用统一格式的股票池
- 享受自动缓存和格式转换的便利
- 保持与现有系统的完全兼容性