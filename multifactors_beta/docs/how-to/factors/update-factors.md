# 因子更新模块使用指南

## [TARGET] 核心功能

因子更新模块支持**全量更新**和**增量更新**两种模式，通过追踪原始数据中的报表发布日期（ReleasedDates）来识别新数据。

## [KEY] 关键特性

1. **智能增量更新**
   - 通过 ReleasedDates 列追踪财务报表的发布日期
   - 通过交易日期追踪价格数据更新
   - 自动识别需要重算的股票和时间段

2. **状态持久化**
   - 更新状态保存在 `factor_update_tracker.json`
   - 记录最后处理的发布日期和交易日期
   - 支持断点续传

3. **财务因子特殊处理**
   - 当某股票有新财务数据时，重算该股票的所有历史因子
   - 确保 TTM、YoY 等时间序列计算的准确性

## [DATA] 快速使用

### 1. 初始化更新器

```python
from factors.utils import FactorUpdater

# 创建更新器
updater = FactorUpdater(
    data_path='./data',      # 原始数据路径
    factor_path='./factors'  # 因子保存路径
)
```

### 2. 全量更新（首次运行）

```python
# 第一次运行时使用全量更新
updater.update_all_factors(
    mode='full',
    financial_data=financial_df,
    price_data=price_df,
    market_cap=market_cap_series,
    release_dates=release_dates_df,  # 包含 ReleasedDates 列
    trading_dates=trading_dates_list
)
```

### 3. 增量更新（日常更新）

```python
# 日常运行使用增量更新
# 1. 检查是否有新数据
has_financial_updates, new_financial = updater.check_financial_updates(financial_df)
has_price_updates, new_price = updater.check_price_updates(price_df)

# 2. 执行增量更新
if has_financial_updates or has_price_updates:
    updater.update_all_factors(
        mode='incremental',
        financial_data=financial_df,
        price_data=price_df,
        market_cap=market_cap_series,
        release_dates=release_dates_df,
        trading_dates=trading_dates_list
    )
```

### 4. 更新特定因子

```python
# 只更新基本面因子
fundamental_results = updater.update_fundamental_factors(
    factor_names=['EP_ttm', 'ROE_ttm', 'CurrentRatio'],
    mode='incremental',
    financial_data=financial_df,
    market_cap=market_cap_series
)

# 只更新技术因子
technical_results = updater.update_technical_factors(
    factor_names=['Momentum_20', 'Volatility_20'],
    mode='incremental',
    price_data=price_df
)
```

## [DATE] 数据格式要求

### 财务数据格式
```python
# MultiIndex: (ReportDates, StockCodes)
# 必须包含 ReleasedDates 列表示报表发布日期
financial_df = pd.DataFrame({
    'ReleasedDates': release_dates,  # 报表发布日期
    'DEDUCTEDPROFIT': [...],
    'EQY_BELONGTO_PARCOMSH': [...],
    # ... 其他财务指标
})
```

### 价格数据格式
```python
# MultiIndex: (TradingDates, StockCodes)
price_df = pd.DataFrame({
    'open': [...],
    'high': [...],
    'low': [...],
    'close': [...],
    'volume': [...],
    'amt': [...]
})
```

## [TOOL] 高级配置

### 自定义追踪文件位置
```python
from factors.utils import UpdateTracker

# 自定义追踪文件路径
tracker = UpdateTracker(tracker_file='./my_tracker.json')
updater = FactorUpdater(data_path, factor_path)
updater.tracker = tracker
```

### 查看更新状态
```python
# 获取当前更新状态
status = updater.tracker.status

# 查看财务数据最后更新信息
financial_status = status.get('financial', {})
print(f"最后发布日期: {financial_status.get('last_release_date')}")
print(f"最后更新时间: {financial_status.get('last_update_time')}")

# 查看价格数据最后更新信息
price_status = status.get('price', {})
print(f"最后交易日: {price_status.get('last_trading_date')}")
```

## [TIP] 最佳实践

1. **定期运行增量更新**
   - 建议每日收盘后运行一次
   - 使用调度器（如 cron）自动化运行

2. **监控更新日志**
   - 配置日志级别查看详细更新信息
   - 记录更新失败的情况

3. **数据验证**
   - 更新前检查数据完整性
   - 更新后验证因子值的合理性

4. **备份管理**
   - 定期备份 factor_update_tracker.json
   - 保留历史因子数据快照

## [ALERT] 注意事项

1. **ReleasedDates 列必须存在**
   - 财务数据必须包含报表发布日期列
   - 列名可以是：ReleasedDates、reportday、releaseddate

2. **增量更新的重算逻辑**
   - 财务因子：有新数据的股票会重算所有历史值
   - 技术因子：只计算新增交易日的值

3. **内存使用**
   - 全量更新可能占用较多内存
   - 建议分批处理大量股票

## [BOOK] 完整示例

参见 `examples/factor_update_example.py` 获取完整的使用示例，包括：
- 创建示例数据
- 全量更新流程
- 增量更新流程
- 自动化更新脚本
- 更新报告生成