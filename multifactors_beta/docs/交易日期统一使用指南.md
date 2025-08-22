# 交易日期统一使用指南

## 📋 概述

为了避免在代码中硬编码 `pd.date_range` 生成交易日期，项目现在提供了统一的交易日期获取接口。

## 🎯 为什么需要统一？

### 问题：
- 各处代码使用 `pd.date_range('2024-01-01', '2024-12-31', freq='D')` 等硬编码
- 生成的日期包含周末、节假日，不是真实的交易日期
- 不同模块使用不同的日期生成方式，导致不一致

### 解决方案：
- 优先使用 `TradingDates.pkl` 中的真实A股交易日历
- 提供回退机制，确保在任何情况下都能获取交易日期
- 统一的API接口，便于维护和更新

## 🛠️ 使用方法

### 1. 推荐方式（新）

```python
from utils.trading_dates_utils import get_trading_dates

# 获取指定范围的交易日期
trading_dates = get_trading_dates('2024-01-01', '2024-12-31')

# 获取2024年的交易日期
trading_dates = get_trading_dates('2024-01-01', '2024-12-31')

# 获取最近一年的交易日期（自动推断范围）
trading_dates = get_trading_dates()
```

### 2. 便捷函数

```python
from utils.trading_dates_utils import get_year_trading_dates, get_recent_trading_dates

# 获取2024年交易日期
trading_dates_2024 = get_year_trading_dates(2024)

# 获取最近252个交易日
recent_dates = get_recent_trading_dates(252)
```

### 3. 向后兼容方式

```python
from get_real_trading_dates import create_debug_trading_dates

# 仍然可用，但推荐使用新接口
trading_dates = create_debug_trading_dates('2024-01-01', '2024-12-31')
```

## 📊 数据源优先级

1. **TradingDates.pkl** - 真实的A股交易日历（推荐）
   - 路径: `data/auxiliary/TradingDates.pkl`
   - 包含2014-2025年的交易日期
   - 已排除周末和节假日

2. **Price数据** - 从实际价格数据中提取
   - 路径: `core/cache/StockData_price.pkl`
   - 使用 `tradingday` 列的唯一值

3. **工作日** - 回退方案
   - 使用 `pd.date_range(freq='B')` 生成
   - 排除周末但可能包含节假日

## 🔧 已更新的文件

### 示例文件
- `examples/factor_calculation_example.py` ✅
- `examples/factor_update_example.py` ✅

### 测试/调试文件  
- `test_direct_call.py` ✅
- `debug_expand_to_daily_smart.py` ✅

### 待更新文件
- `factors/base/testable_mixin.py`
- `tests/integration/test_refactored_factors.py`

## 🚨 迁移指南

### 旧代码模式：
```python
# ❌ 不推荐 - 硬编码日期生成
trading_dates = pd.date_range('2024-01-01', '2024-12-31', freq='D')
trading_dates = pd.date_range('2024-01-01', '2024-12-31', freq='B')
```

### 新代码模式：
```python
# ✅ 推荐 - 使用统一接口
from utils.trading_dates_utils import get_trading_dates
trading_dates = get_trading_dates('2024-01-01', '2024-12-31')
```

## 🧪 测试验证

运行测试脚本验证交易日期的正确性：

```bash
python utils/trading_dates_utils.py
```

输出示例：
```
2024年:
  交易日数量: 242
  日期范围: 2024-01-02 to 2024-12-31
  包含周末: 否
  长假期间隔:
    2024-02-08 -> 2024-02-19 (11天)  # 春节
    2024-04-29 -> 2024-05-06 (7天)   # 劳动节
```

## 📝 注意事项

1. **日期格式**：使用字符串格式 `'YYYY-MM-DD'`
2. **错误处理**：函数会自动处理数据源不可用的情况
3. **性能考虑**：真实交易日期加载有轻微的I/O开销，但数据会被缓存
4. **调试模式**：在开发阶段可以临时使用工作日模式以加快速度

## 🔮 未来改进

1. 支持更多交易所（港股、美股等）
2. 自动更新交易日历
3. 添加节假日信息标注
4. 支持缓存机制优化性能

---

**重要提醒**：新代码应该使用 `utils.trading_dates_utils.get_trading_dates`，避免硬编码日期生成方式。