# 高成交额股票池使用指南

## 📊 股票池信息

**股票池名称**: `high_turnover_5d_20m`  
**描述**: 5日平均成交额超过2000万元的股票池  
**格式**: MultiIndex Series [TradingDates, StockCodes]，值为1  
**数据点**: 9,484,389 个  
**覆盖期间**: 2014-01-06 到 2025-08-28 (2813个交易日)  
**涉及股票**: 5665只  
**日均股票数**: 3371.6只  

## 🚀 使用方法

### 1. 通过FactorDataLoader（推荐）

```python
from factors.utils.data_loader import get_stock_universe

# 加载高成交额股票池
universe = get_stock_universe('high_turnover_5d_20m')

print(f"数据类型: {type(universe)}")  # pandas.Series
print(f"数据形状: {universe.shape}")   # (15,946,975,) 
print(f"股票数量: {len(universe.index.get_level_values(1).unique())}")  # 5665
```

### 2. 通过StockUniverseManager

```python
from factors.tester.stock_universe_manager import StockUniverseManager

manager = StockUniverseManager()

# MultiIndex Series格式（时变股票池）
universe = manager.get_universe('high_turnover_5d_20m', format_type='multiindex')

# List格式（股票代码列表）
stocks = manager.get_universe('high_turnover_5d_20m', format_type='list')
```

### 3. 在单因子测试中使用

```python
from factors.tester.core.pipeline import SingleFactorTestPipeline
from factors.utils.data_loader import get_stock_universe

# 初始化测试流水线
pipeline = SingleFactorTestPipeline()

# 加载股票池
universe = get_stock_universe('high_turnover_5d_20m')

# 运行单因子测试
result = pipeline.run(
    factor_name='your_factor_name',
    stock_universe=universe  # 传入MultiIndex Series
)
```

### 4. 直接通过字符串使用

```python
from factors.tester.core.pipeline import SingleFactorTestPipeline

pipeline = SingleFactorTestPipeline()

# 直接使用股票池名称
result = pipeline.run(
    factor_name='your_factor_name',
    stock_universe='high_turnover_5d_20m'  # 系统会自动加载
)
```

## 📁 文件位置

- **股票池文件**: `{stock_universe}/high_turnover_5d_20m.pkl`
- **元数据文件**: `{stock_universe}/high_turnover_5d_20m.json`

其中 `{stock_universe}` 是配置的股票池专用目录：
```
E:\Documents\PythonProject\StockProject\StockData\stock_universe\
```

这个路径在 `config/main.yaml` 中通过 `main.paths.stock_universe` 配置。

## 🔍 查看可用股票池

```python
from factors.utils.data_loader import FactorDataLoader

# 列出所有可用股票池
available = FactorDataLoader.list_available_universes()
for name, description in available.items():
    print(f"{name}: {description}")
```

输出包含：
```
high_turnover_5d_20m: 高成交额股票池（5日平均成交额超过2000万元）
```

## ⚡ 性能特征

### 优化点
- **缓存机制**: 首次加载后自动缓存，后续访问极快
- **格式统一**: 直接返回MultiIndex Series，无需转换
- **时变支持**: 支持不同日期包含不同股票的动态股票池

### 数据特征  
- **高流动性**: 确保所选股票具有足够的成交额
- **动态调整**: 股票池成员随时间变化，反映市场流动性变化
- **覆盖全面**: 涵盖2014年以来的完整历史数据

## 🛠️ 技术实现

### 生成逻辑
1. 读取Price.pkl中的成交额数据(`amt`列)
2. 计算每只股票的5日滚动平均成交额
3. 筛选平均成交额 > 2000万元的股票
4. 生成MultiIndex Series格式的时变股票池

### 数据格式
```python
MultiIndex: [TradingDates, StockCodes]
Values: 1 (表示该股票在该日期满足条件)

# 示例
TradingDates  StockCodes
2014-01-06    000001        1
              000002        1  
              600000        1
2014-01-07    000001        1
              000002        1
              600036        1
...
```

## 📈 应用场景

### 1. 单因子测试
在高流动性股票上测试因子表现，避免流动性偏差

### 2. 组合构建
构建投资组合时确保成份股具有足够流动性

### 3. 回测验证
在历史回测中使用动态的流动性股票池

### 4. 风险管理
避免投资流动性不足的股票

## 💡 最佳实践

1. **优先使用**: 在需要高流动性股票的场景中优先使用此股票池
2. **与其他股票池对比**: 可以与全市场股票池对比，分析流动性对因子的影响
3. **定期更新**: 虽然已包含最新数据，建议定期重新生成以包含最新的市场情况
4. **缓存利用**: 在同一会话中重复使用时会自动利用缓存，提高性能

## ✅ 验证方法

```python
# 快速验证
from factors.utils.data_loader import get_stock_universe
universe = get_stock_universe('high_turnover_5d_20m')
print(f"✅ 加载成功: {universe.shape}")
```

这个股票池已经完全集成到系统中，可以在所有支持股票池的地方直接使用！