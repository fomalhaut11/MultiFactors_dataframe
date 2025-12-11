# factors.generators 工具集完整指南

## 工具集概览

**factors.generators是项目中所有数据处理和计算的官方工具库，包含四大核心模块。**

---

## 目录结构

```
factors/generators/
├── __init__.py              # 统一导入接口
├── financial/               # 财务数据处理工具
│   ├── __init__.py          # 财务工具便捷导入
│   └── financial_report_processor.py  # 核心财务处理器
├── technical/               # 技术指标计算工具
│   ├── __init__.py          # 技术指标便捷导入
│   ├── moving_average.py    # 移动平均计算器
│   ├── oscillator.py        # 振荡器指标（RSI、MACD等）
│   └── volatility.py        # 波动率计算器
├── mixed/                   # 混合数据处理工具
│   ├── __init__.py          # 混合处理便捷导入
│   └── mixed_data_processor.py  # 财务+市场数据处理
└── alpha191/                # Alpha191因子计算工具
    ├── __init__.py          # Alpha191便捷导入
    └── alpha191_ops.py      # Alpha191操作函数库
```

---

## 一、财务数据处理工具 (financial)

### 1.1 核心功能
专门处理财务报表数据，提供TTM、YoY、QoQ等标准财务计算。

### 1.2 主要工具

#### FinancialReportProcessor（核心处理器）
```python
from factors.generators import FinancialReportProcessor

# 主要功能：
# - calculate_ttm()          # TTM（滚动12个月）计算
# - calculate_single_quarter()  # 单季度转换
# - calculate_yoy()          # 同比增长率
# - calculate_qoq()          # 环比增长率
# - calculate_zscore()       # Z-Score标准化
# - expand_to_daily_vectorized()  # 日频扩展
```

#### 便捷函数（推荐使用）
```python
from factors.generators import (
    calculate_ttm,           # TTM计算
    calculate_single_quarter,# 单季度计算
    calculate_yoy,           # 同比增长
    calculate_qoq,           # 环比增长
    calculate_zscore,        # Z-Score标准化
    expand_to_daily_vectorized  # 日频扩展
)
```

### 1.3 使用示例

#### TTM计算
```python
# 输入：财务数据（季度）
financial_data = load_financial_data()  # DataFrame with columns like 'net_income', 'revenue'

# TTM计算
ttm_data = calculate_ttm(financial_data)
# 返回：DataFrame，每行是股票+季度，值为该季度对应的TTM值
```

#### 同比/环比增长
```python
# 同比增长（当前季度 vs 去年同季度）
yoy_growth = calculate_yoy(financial_data)

# 环比增长（当前季度 vs 上季度）
qoq_growth = calculate_qoq(financial_data)
```

#### 日频扩展
```python
# 将季度因子值扩展到日频
daily_factor = expand_to_daily_vectorized(
    factor_data=quarterly_factor,
    release_dates=financial_calendar,
    trading_dates=trading_calendar
)
```

---

## 二、技术指标计算工具 (technical)

### 2.1 核心功能
计算各种技术分析指标，包括移动平均、振荡器、波动率等。

### 2.2 主要工具

#### MovingAverageCalculator
```python
from factors.generators import MovingAverageCalculator

calc = MovingAverageCalculator()
# 功能：
# - calculate_sma()    # 简单移动平均
# - calculate_ema()    # 指数移动平均
# - calculate_wma()    # 加权移动平均
```

#### TechnicalIndicators
```python
from factors.generators import TechnicalIndicators

indicators = TechnicalIndicators()
# 功能：
# - calculate_rsi()    # RSI相对强弱指标
# - calculate_macd()   # MACD指标
# - calculate_kdj()    # KDJ指标
```

#### VolatilityCalculator
```python
from factors.generators import VolatilityCalculator

vol_calc = VolatilityCalculator()
# 功能：
# - calculate_rolling_volatility()  # 滚动波动率
# - calculate_garch_volatility()    # GARCH波动率
# - calculate_realized_volatility() # 已实现波动率
```

### 2.3 使用示例

#### 移动平均
```python
price_data = load_price_data()  # 价格序列

# 20日简单移动平均
sma_20 = MovingAverageCalculator().calculate_sma(price_data, window=20)

# 12日指数移动平均
ema_12 = MovingAverageCalculator().calculate_ema(price_data, span=12)
```

#### RSI指标
```python
# 14期RSI
rsi_14 = TechnicalIndicators().calculate_rsi(price_data, window=14)
```

---

## 三、混合数据处理工具 (mixed)

### 3.1 核心功能
处理财务数据与市场数据的结合，提供数据对齐、比率计算等功能。

### 3.2 主要工具

#### MixedDataProcessor
```python
from factors.generators import MixedDataProcessor

# 核心功能：
# - align_data()                    # 数据对齐
# - calculate_financial_market_ratio()  # 财务-市场比率
# - normalize_cross_sectional()     # 截面标准化
```

#### 便捷函数
```python
from factors.generators import (
    align_financial_with_market,        # 财务与市场数据对齐
    calculate_relative_ratio,           # 相对比率计算
    calculate_market_cap_weighted_ratio # 市值加权比率
)
```

### 3.3 使用示例

#### 数据对齐
```python
financial_data = load_financial_data()  # 季度财务数据
market_data = load_market_data()        # 日度市场数据

# 对齐财务和市场数据
aligned_financial, aligned_market = align_financial_with_market(
    financial_data, market_data
)
```

#### 比率计算
```python
# 计算财务指标相对市场指标的比率
ratio = calculate_relative_ratio(
    numerator=financial_indicator,
    denominator=market_indicator
)
```

---

## 四、Alpha191工具集 (alpha191)

### 4.1 核心功能
实现Alpha191因子库中定义的标准操作函数。

### 4.2 主要工具

#### Alpha191操作函数
```python
from factors.generators import (
    convert_to_alpha191_format,  # 数据格式转换
    prepare_alpha191_data,       # Alpha191数据准备
    ts_rank,                     # 时序排序
    ts_mean,                     # 时序均值
    delta,                       # 差值计算
    rank,                        # 截面排序
    scale                        # 标准化
)
```

### 4.3 使用示例

#### Alpha191格式转换
```python
# 将MultiIndex数据转换为Alpha191格式
alpha191_data = convert_to_alpha191_format(multiindex_data)

# 数据准备
prepared_data = prepare_alpha191_data(raw_data)
```

#### Alpha191操作
```python
# 时序排序（过去N期的排序）
ts_ranked = ts_rank(data, window=20)

# 截面排序（当期所有股票排序）
cross_ranked = rank(data)

# 标准化处理
scaled_data = scale(data)
```

---

## 统一导入方式

### 最佳实践
```python
# 推荐：从顶级模块导入
from factors.generators import (
    calculate_ttm,                      # 财务：TTM计算
    calculate_yoy,                      # 财务：同比增长
    MovingAverageCalculator,            # 技术：移动平均
    TechnicalIndicators,                # 技术：技术指标
    align_financial_with_market,        # 混合：数据对齐
    ts_rank,                           # Alpha191：时序排序
    scale                              # Alpha191：标准化
)
```

### 子模块导入（可选）
```python
# 按需从子模块导入
from factors.generators.financial import FinancialReportProcessor
from factors.generators.technical import VolatilityCalculator
from factors.generators.mixed import MixedDataProcessor
```

---

## 工具选择指南

### 财务因子开发
```python
# 基础财务比率（如PE、ROE等）
from factors.generators import calculate_ttm, calculate_yoy

# 复杂财务处理
from factors.generators import FinancialReportProcessor
```

### 技术因子开发
```python
# 价格趋势因子
from factors.generators import MovingAverageCalculator

# 动量振荡因子
from factors.generators import TechnicalIndicators

# 波动率因子
from factors.generators import VolatilityCalculator
```

### 混合因子开发
```python
# 财务+市场数据结合
from factors.generators import align_financial_with_market, calculate_relative_ratio
```

### Alpha191因子开发
```python
# Alpha191标准操作
from factors.generators import ts_rank, delta, rank, scale
```

---

## 数据格式要求

### 输入数据格式
- **财务数据**: MultiIndex[ReportDate, StockCodes] DataFrame
- **价格数据**: MultiIndex[TradingDates, StockCodes] Series/DataFrame
- **市场数据**: MultiIndex[TradingDates, StockCodes] Series

### 输出数据格式
- **因子数据**: MultiIndex[TradingDates, StockCodes] Series
- **中间结果**: 根据具体函数而定，通常保持输入格式

---

## 性能优化建议

### 1. 向量化优先
```python
# ✅ 推荐：使用工具集的向量化方法
ttm_data = calculate_ttm(financial_data)

# ❌ 避免：循环处理
# for stock in stocks:
#     stock_ttm = calculate_ttm_single(stock_data)
```

### 2. 批处理
```python
# ✅ 推荐：一次处理多个指标
all_indicators = TechnicalIndicators().calculate_multiple([
    'rsi', 'macd', 'kdj'
], price_data)

# ❌ 避免：逐个计算
# rsi = calculate_rsi(price_data)
# macd = calculate_macd(price_data)
```

### 3. 内存管理
```python
# ✅ 推荐：使用工具集的chunked处理
for chunk in FinancialReportProcessor.process_in_chunks(large_data):
    result = calculate_ttm(chunk)
```

---

## 错误处理

### 常见错误和解决方案

#### 1. 数据格式错误
```python
# 错误：索引格式不正确
# 解决：使用工具集的格式检查
from factors.generators import convert_to_alpha191_format
formatted_data = convert_to_alpha191_format(raw_data)
```

#### 2. 缺失值处理
```python
# 工具集自动处理NaN值
ttm_data = calculate_ttm(financial_data)  # 自动跳过NaN
```

#### 3. 日期对齐问题
```python
# 使用混合工具的对齐功能
aligned_data = align_financial_with_market(financial_data, market_data)
```

---

## 扩展开发指南

### 如何添加新工具
1. **确定模块**: 选择合适的子模块（financial/technical/mixed/alpha191）
2. **实现函数**: 遵循现有函数的接口规范
3. **添加导入**: 在对应的__init__.py中添加导入
4. **更新文档**: 更新本指南文档

### 函数接口规范
```python
def new_calculation_function(data: pd.DataFrame, **kwargs) -> pd.Series:
    """
    新计算函数的标准格式
    
    Parameters:
    -----------
    data : pd.DataFrame
        输入数据，MultiIndex格式
    **kwargs : dict
        其他参数
        
    Returns:
    --------
    pd.Series
        计算结果，MultiIndex格式
    """
    # 实现逻辑
    pass
```

---

**重要提醒: factors.generators是项目的核心工具库，所有因子开发都应该基于这些工具进行，严禁重复实现其中已有的功能。**