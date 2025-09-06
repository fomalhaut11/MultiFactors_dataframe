# 数据依赖关系图谱

## 🎯 依赖关系总览

### 数据流向图

```
                        📊 数据库源头
                             │
           ┌─────────────────┼─────────────────┐
           │                 │                 │
    ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
    │ 价格数据库   │  │ 财务数据库   │  │ 市场数据库   │
    │ StockDB     │  │ StockDB     │  │ StockDB     │  
    └─────────────┘  └─────────────┘  └─────────────┘
           │                 │                 │
           ▼                 ▼                 ▼
    ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
    │📦 增量更新器 │  │📦 增量更新器 │  │📦 增量更新器 │
    │PriceUpdater │  │FinancialUpd │  │MarketUpdater│
    └─────────────┘  └─────────────┘  └─────────────┘
           │                 │                 │
           ▼                 ▼                 ▼
    ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
    │💾 基础文件   │  │💾 基础文件   │  │💾 基础文件   │
    │ Price.pkl   │  │ fzb/lrb/    │  │ ST/Stop/    │
    │TradableDF   │  │ xjlb.pkl    │  │Sector.pkl   │
    └─────────────┘  └─────────────┘  └─────────────┘
           │                 │                 │
           └─────────┬───────┴─────────┬───────┘
                     ▼                 ▼
              ┌─────────────┐  ┌─────────────┐
              │🔧 数据处理器 │  │🔧 数据处理器 │
              │DataPipeline │  │Classification│
              └─────────────┘  └─────────────┘
                     │                 │
                     ▼                 ▼
              ┌─────────────┐  ┌─────────────┐
              │📈 衍生数据   │  │📊 分类数据   │
              │LogReturn_*  │  │StockClass_* │
              └─────────────┘  └─────────────┘
```

## 📋 详细依赖表

### Level 0: 数据库表

| 数据库表名 | 主要字段 | 更新频率 | 数据规模 | 描述 |
|-----------|----------|----------|----------|------|
| Price | tradingday, code, o, h, l, c, v, amt | 每日 | 1130万+ | 股票价格数据 |
| fzb | reportday, tradingday, code, 资产负债表字段 | 季度 | 43万+ | 资产负债表 |
| lrb | reportday, tradingday, code, 利润表字段 | 季度 | 43万+ | 利润表 |
| xjlb | reportday, tradingday, code, 现金流量表字段 | 季度 | 43万+ | 现金流量表 |
| StopPrice | tradingday, code, limit_up, limit_down | 每日 | 1450万+ | 涨跌停价格 |
| ST_stocks | tradingday, code, st_type, change_type | 不定期 | 68万+ | ST股票状态 |
| SectorChanges | sel_day, code, concept_code, action | 不定期 | 625 | 板块调整 |

### Level 1: 基础数据文件

| 文件名 | 更新器 | 依赖源 | 字段数 | 索引结构 | 更新触发条件 |
|--------|--------|--------|--------|----------|-------------|
| Price.pkl | PriceDataUpdater | Price表 | 16 | (TradingDates, StockCodes) | 本地日期 < 数据库日期 |
| TradableDF.pkl | PriceDataUpdater | Price表(trade_status) | 4 | (TradingDates, StockCodes) | 与Price.pkl同步 |
| fzb.pkl | FinancialDataUpdater | fzb表 | 140 | 单层索引 | local_date检查 |
| lrb.pkl | FinancialDataUpdater | lrb表 | 140 | 单层索引 | local_date检查 |
| xjlb.pkl | FinancialDataUpdater | xjlb表 | 140 | 单层索引 | local_date检查 |
| StopPrice.pkl | StopPriceDataUpdater | StopPrice表 | 4 | (TradingDates, StockCodes) | 本地日期 < 数据库日期 |
| ST_stocks.pkl | STDataUpdater | ST_stocks表 | 4 | 单层索引 | 30天更新间隔 |
| SectorChanges_data.pkl | SectorChangesDataUpdater | SectorChanges表 | 多列 | 单层索引 | sel_day检查 |

### Level 2: 处理数据文件

| 文件名 | 处理器 | 直接依赖 | 间接依赖 | 生成条件 | 数据结构 |
|--------|--------|----------|----------|----------|----------|
| Stock3d.pkl | PriceDataProcessor | Price.pkl, TradableDF.pkl | Price表 | Price.pkl更新后 | 3D矩阵 |
| LogReturn_daily_o2o.pkl | ReturnCalculator | Price.pkl | Price表 | 按需触发 | (TradingDates, StockCodes) |
| LogReturn_daily_vwap.pkl | ReturnCalculator | Price.pkl | Price表 | 按需触发 | (TradingDates, StockCodes) |
| LogReturn_5days_o2o.pkl | ReturnCalculator | Price.pkl | Price表 | 按需触发 | (TradingDates, StockCodes) |
| LogReturn_20days_o2o.pkl | ReturnCalculator | Price.pkl | Price表 | 按需触发 | (TradingDates, StockCodes) |
| LogReturn_weekly_o2o.pkl | ReturnCalculator | Price.pkl | Price表 | 按需触发 | (TradingDates, StockCodes) |
| LogReturn_monthly_o2o.pkl | ReturnCalculator | Price.pkl | Price表 | 按需触发 | (TradingDates, StockCodes) |
| StockClassification_*.pkl | SectorClassificationProcessor | SectorChanges_data.pkl | SectorChanges表 | 按需触发 | MultiIndex |

## 🔗 关键依赖链路

### 1. 价格数据链

```
StockDB.Price
    ↓ (IncrementalPriceUpdater)
Price.pkl + TradableDF.pkl
    ↓ (PriceDataProcessor)
Stock3d.pkl
    ↓ (ReturnCalculator)
LogReturn_*.pkl (6个文件)
```

**关键节点**:
- `TradableDF.pkl`: 必须与Price.pkl同步，提供交易状态过滤
- `Stock3d.pkl`: 3D矩阵格式，收益率计算的必要中间格式

### 2. 财务数据链

```
StockDB.{fzb, lrb, xjlb}
    ↓ (IncrementalFinancialUpdater)
{fzb, lrb, xjlb}.pkl
    ↓ (FinancialDataProcessor)
released_dates_df + DateCount_df
    ↓ (因子生成器 - 未来扩展)
FinancialFactors_*.pkl
```

**关键节点**:
- 三表必须同步更新（共享local_date字段）
- `tradingday`字段提供财报发布日期信息

### 3. 分类数据链

```
StockDB.SectorChanges
    ↓ (SectorChangesDataUpdater)
SectorChanges_data.pkl
    ↓ (SectorClassificationProcessor)
StockClassification_*.pkl
```

**关键节点**:
- 板块调整数据更新频率较低
- 分类信息按需计算，不需要实时更新

## ⚠️ 依赖风险与缓解

### 高风险依赖

1. **Price.pkl → TradableDF.pkl**
   - **风险**: TradableDF缺失会导致交易状态过滤失败
   - **缓解**: PriceDataProcessor已修改为可选依赖，缺失时跳过过滤

2. **财务三表同步**
   - **风险**: 某个表更新失败导致数据不一致
   - **缓解**: IncrementalFinancialUpdater使用事务性操作

3. **数据库连接中断**
   - **风险**: 更新过程中数据库连接丢失
   - **缓解**: 连接池自动重连 + 备份机制

### 中风险依赖

1. **Stock3d.pkl格式错误**
   - **风险**: 3D矩阵reshape失败
   - **缓解**: 数据验证 + 错误日志记录

2. **日期格式不一致**
   - **风险**: MultiIndex构建失败
   - **缓解**: 统一的日期处理函数

### 低风险依赖

1. **ST数据更新间隔长**
   - **风险**: 数据可能过时
   - **影响**: 有限，ST状态变化频率低

## 🔧 依赖管理最佳实践

### 1. 更新顺序

推荐按以下顺序执行更新：

```python
# 1. 基础数据更新（可并行）
python scheduled_data_updater.py --data-type price
python scheduled_data_updater.py --data-type financial  
python scheduled_data_updater.py --data-type stop_price

# 2. 市场数据更新（可并行）
python scheduled_data_updater.py --data-type st
python scheduled_data_updater.py --data-type sector_changes

# 3. 处理数据生成（依赖基础数据）
python -c "from data.processor.data_processing_pipeline import DataProcessingPipeline; DataProcessingPipeline().run_full_pipeline()"
```

### 2. 依赖检查

```python
from core.data_registry import get_data_registry

registry = get_data_registry()

# 检查依赖关系
for dataset_name in ['logreturn_daily_o2o']:
    dataset = registry.get_dataset_info(dataset_name)
    if dataset.dependencies:
        print(f"{dataset_name} 依赖: {dataset.dependencies}")
        for dep in dataset.dependencies:
            dep_info = registry.get_dataset_info(dep)
            print(f"  -> {dep}: {'可用' if dep_info.is_available else '不可用'}")
```

### 3. 故障恢复

```bash
# 检查缺失的依赖
python scheduled_data_updater.py --list-data | grep "不可用"

# 强制重新生成依赖数据
python scheduled_data_updater.py --data-type price --force

# 验证数据完整性
python -c "
from core.data_registry import get_data_registry
registry = get_data_registry()
missing = registry.get_missing_datasets()
if missing:
    print('缺失数据集:', missing)
else:
    print('所有数据集可用')
"
```

## 📊 依赖监控

### 监控指标

1. **数据新鲜度**: 检查各文件最后更新时间
2. **依赖完整性**: 验证所有依赖文件是否存在
3. **数据一致性**: 检查关联数据的时间范围一致性
4. **更新成功率**: 统计更新操作的成功/失败比例

### 自动化监控

```python
# 添加到定时任务
def dependency_health_check():
    registry = get_data_registry()
    
    # 检查数据新鲜度
    freshness = registry.check_data_freshness(hours_threshold=24)
    
    # 检查缺失数据
    missing = registry.get_missing_datasets()
    
    # 生成报告
    if missing or not all(freshness.values()):
        # 发送告警
        alert_admin("数据依赖健康检查失败")
```

---

**维护说明**: 
- 新增数据源时，必须更新此依赖图谱
- 修改依赖关系时，需要同步更新注册器配置
- 定期review依赖关系的合理性和必要性