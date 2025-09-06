# 数据更新系统说明文档

## 📋 系统概述

多因子量化投资系统的数据更新模块提供了完整的数据管理和更新功能，包括数据注册器、增量更新器和CLI工具。

**版本**: v2.1 (生产级)  
**创建时间**: 2025-08-29  
**维护状态**: 活跃维护

## 🏗️ 架构设计

### 核心组件

```
数据更新系统
├── core/data_registry.py          # 数据注册器 (中央管理)
├── scheduled_data_updater.py       # 定时更新器 (CLI入口)
├── data/fetcher/                   # 数据获取器模块
│   ├── incremental_price_updater.py       # 价格数据增量更新
│   ├── incremental_financial_updater.py   # 财务数据增量更新  
│   ├── incremental_stop_price_updater.py  # 涨跌停数据增量更新
│   └── data_fetcher.py                     # 基础数据获取器
└── data/processor/                 # 数据处理器模块
    ├── data_processing_pipeline.py        # 数据处理管道
    ├── price_processor.py                 # 价格数据处理器
    ├── financial_processor.py             # 财务数据处理器
    └── return_calculator.py               # 收益率计算器
```

### 数据流向图

```
数据库 → 增量更新器 → 本地文件 → 数据处理器 → 衍生数据
  ↓         ↓           ↓           ↓          ↓
StockDB → Updaters → *.pkl files → Processors → LogReturn_*.pkl
                        ↓
                   数据注册器监控
```

## 📊 数据集分类

### 1. 原始数据（从数据库获取）

| 数据集名称 | 文件名 | 更新器 | 更新频率 | 描述 |
|-----------|--------|--------|----------|------|
| price_data | Price.pkl | PriceDataUpdater | daily | 股票日线价格数据 |
| financial_fzb | fzb.pkl | FinancialDataUpdater | daily | 资产负债表数据 |
| financial_lrb | lrb.pkl | FinancialDataUpdater | daily | 利润表数据 |
| financial_xjlb | xjlb.pkl | FinancialDataUpdater | daily | 现金流量表数据 |
| stop_price | StopPrice.pkl | StopPriceDataUpdater | daily | 涨跌停板数据 |
| st_stocks | ST_stocks.pkl | STDataUpdater | daily | ST股票数据 |
| sector_changes | SectorChanges_data.pkl | SectorChangesDataUpdater | daily | 板块进出调整数据 |

### 2. 处理数据（从原始数据计算）

| 数据集名称 | 文件名 | 处理器 | 依赖关系 | 描述 |
|-----------|--------|--------|----------|------|
| logreturn_daily_o2o | LogReturn_daily_o2o.pkl | DataProcessingPipeline | price_data | 日收益率(开到开) |
| logreturn_daily_vwap | LogReturn_daily_vwap.pkl | DataProcessingPipeline | price_data | 日收益率(VWAP) |
| logreturn_5days_o2o | LogReturn_5days_o2o.pkl | DataProcessingPipeline | price_data | 5天收益率 |
| logreturn_20days_o2o | LogReturn_20days_o2o.pkl | DataProcessingPipeline | price_data | 20天收益率 |
| logreturn_weekly_o2o | LogReturn_weekly_o2o.pkl | DataProcessingPipeline | price_data | 周收益率 |
| logreturn_monthly_o2o | LogReturn_monthly_o2o.pkl | DataProcessingPipeline | price_data | 月收益率 |
| stock_classification | StockClassification_*.pkl | SectorClassificationProcessor | sector_changes | 股票分类信息 |

## 🔗 依赖关系详解

### 数据依赖层级

```
Level 0: 数据库源数据
    ├── StockDB.Price (价格表)
    ├── StockDB.fzb (资产负债表)
    ├── StockDB.lrb (利润表)  
    ├── StockDB.xjlb (现金流量表)
    ├── StockDB.StopPrice (涨跌停表)
    ├── StockDB.ST_stocks (ST股票表)
    └── StockDB.SectorChanges (板块调整表)

Level 1: 基础数据文件 (*.pkl)
    ├── Price.pkl ← StockDB.Price
    ├── fzb.pkl ← StockDB.fzb
    ├── lrb.pkl ← StockDB.lrb
    ├── xjlb.pkl ← StockDB.xjlb
    ├── StopPrice.pkl ← StockDB.StopPrice
    ├── ST_stocks.pkl ← StockDB.ST_stocks
    └── SectorChanges_data.pkl ← StockDB.SectorChanges

Level 2: 处理数据文件
    ├── LogReturn_*.pkl ← Price.pkl
    ├── Stock3d.pkl ← Price.pkl
    └── StockClassification_*.pkl ← SectorChanges_data.pkl

Level 3: 复合数据 (未来扩展)
    └── FactorData_*.pkl ← LogReturn_*.pkl + financial_*.pkl
```

### 关键依赖说明

1. **TradableDF.pkl** - 可交易股票状态文件
   - **生成器**: `StockDataFetcher._fetch_tradable_data()`
   - **依赖**: `StockDB.Price` (trade_status字段)
   - **被依赖**: `PriceDataProcessor` (交易状态过滤)

2. **财务数据三表联动**
   - `fzb.pkl`, `lrb.pkl`, `xjlb.pkl` 必须同步更新
   - 共享字段: `code`, `reportday`, `tradingday`, `d_year`, `d_quarter`

3. **收益率数据链**
   - 所有LogReturn文件都依赖Price.pkl
   - 需要Stock3d.pkl作为中间3D矩阵格式

## 🚀 使用指南

### 1. 健康检查

```bash
# 检查所有数据类型
python scheduled_data_updater.py --data-type all --health-check

# 检查特定数据类型
python scheduled_data_updater.py --data-type financial --health-check
python scheduled_data_updater.py --data-type price --health-check
```

### 2. 数据更新

```bash
# 单类型更新
python scheduled_data_updater.py --data-type price
python scheduled_data_updater.py --data-type financial  
python scheduled_data_updater.py --data-type stop_price

# 强制更新（忽略时间检查）
python scheduled_data_updater.py --data-type all --force

# 静默模式
python scheduled_data_updater.py --data-type price --quiet
```

### 3. 数据注册器管理

```bash
# 查看所有数据集
python scheduled_data_updater.py --list-data

# 查看数据摘要
python scheduled_data_updater.py --data-summary
```

### 4. 程序化使用

```python
from core.data_registry import get_data_registry
from scheduled_data_updater import ScheduledDataUpdater

# 获取数据注册器
registry = get_data_registry()
registry.print_data_summary()

# 创建更新器
updater = ScheduledDataUpdater(['price', 'financial'])
results = updater.run_all_updates()
```

## ⚙️ 配置说明

### 更新时间设置

系统自动判断更新时机：
- **工作日 16:00-23:59**: 允许更新（交易日收盘后）
- **周末全天**: 允许更新（补充数据）
- **其他时间**: 跳过更新（可用--force强制）

### 更新频率配置

```python
# core/data_registry.py 中的配置
UpdateFrequency.DAILY        # 每日更新
UpdateFrequency.WEEKLY       # 每周更新  
UpdateFrequency.MONTHLY      # 每月更新
UpdateFrequency.ON_DEMAND    # 按需更新
```

### 备份策略

- **价格数据**: 保留3天备份
- **财务数据**: 保留7天备份
- **其他数据**: 保留3天备份

## 🔧 故障排除

### 常见问题

1. **数据库连接失败**
   ```
   解决方法: 检查config.yaml中的数据库配置
   ```

2. **文件权限错误**
   ```
   解决方法: 确保数据目录有写入权限
   ```

3. **内存不足**
   ```
   解决方法: 处理大文件时使用分块处理
   ```

4. **数据格式错误**
   ```
   解决方法: 检查MultiIndex格式和日期字段类型
   ```

### 日志查看

```bash
# 查看更新日志
tail -f E:\Documents\PythonProject\StockProject\StockData\logs\data_update_20250829.log

# 查看错误日志
grep ERROR E:\Documents\PythonProject\StockProject\StockData\logs\data_update_*.log
```

## 📈 监控指标

### 关键指标

1. **数据时效性**
   - 价格数据: 应与数据库保持同步
   - 财务数据: T+1更新（报告期后一天）

2. **数据完整性**
   - 记录数量: 与数据库对比
   - 字段完整性: 关键字段非空检查

3. **系统性能**
   - 更新耗时: 大文件<5分钟
   - 内存使用: 峰值<12GB
   - 存储空间: 定期清理备份

### 告警阈值

- 数据延迟 > 2天 → WARNING
- 数据延迟 > 5天 → ERROR  
- 文件缺失 → ERROR
- 更新失败 → ERROR

## 🛠️ 维护建议

### 定期任务

1. **每日**: 运行健康检查
2. **每周**: 清理旧备份文件
3. **每月**: 检查存储空间使用
4. **每季度**: 性能优化评估

### 升级路径

1. **增加新数据源**: 在data_registry.py中注册
2. **修改更新频率**: 调整UpdateFrequency枚举
3. **添加新处理器**: 继承BaseDataProcessor
4. **扩展CLI功能**: 修改scheduled_data_updater.py

---

**文档维护**: 随系统更新同步维护此文档  
**技术支持**: 通过项目Issue反馈问题