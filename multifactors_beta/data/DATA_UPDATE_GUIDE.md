# 数据更新系统使用指南

## 概述

本系统已将板块估值计算完全集成到数据更新流程中，提供了多种更新模式和灵活的配置选项。

## 快速开始

### 1. 每日更新（推荐日常使用）

```bash
# 使用Python脚本
python update_data.py --daily

# 或使用批处理文件（Windows）
run_daily_update.bat
```

每日更新包含：
- 价格数据更新
- 收益率计算
- 板块估值更新（最新1天）

### 2. 每周更新

```bash
# 使用Python脚本
python update_data.py --weekly

# 或使用批处理文件（Windows）
run_weekly_update.bat
```

每周更新包含：
- 财务数据更新
- 行业分类更新
- 板块估值更新（最近5天）

### 3. 完整更新

```bash
python update_data.py --full
```

运行完整的数据处理流程，包括所有数据类型。

## 高级用法

### 自定义更新

可以选择性地更新特定数据：

```bash
# 只更新价格数据
python update_data.py --price

# 只更新板块估值
python update_data.py --sector

# 更新价格和板块估值
python update_data.py --price --sector

# 自定义板块估值计算天数（默认252天）
python update_data.py --sector --sector-days 30
```

### 跳过板块估值

如果不需要板块估值计算：

```bash
python update_data.py --full --no-sector
```

### 强制更新

忽略缓存，强制重新计算：

```bash
python update_data.py --daily --force
```

### 静默模式

只显示错误信息：

```bash
python update_data.py --daily --quiet
```

## Python API 使用

### 基本使用

```python
from data.processor import IntegratedDataPipeline

# 创建管道
pipeline = IntegratedDataPipeline()

# 运行完整更新
results = pipeline.run_full_pipeline(
    save_intermediate=True,
    include_sector_valuation=True,
    sector_date_range=252  # 计算一年的数据
)

# 获取板块估值结果
sector_valuation = results.get('sector_valuation')
```

### 独立更新板块估值

```python
from data.processor import IntegratedDataPipeline

pipeline = IntegratedDataPipeline()

# 更新最近30天的板块估值
sector_data = pipeline.update_sector_valuation(
    date_range=30,
    force_update=False  # 使用缓存
)
```

### 使用调度器

```python
from data.processor import DataUpdateScheduler, IntegratedDataPipeline

# 创建调度器
pipeline = IntegratedDataPipeline()
scheduler = DataUpdateScheduler(pipeline)

# 运行每日更新
scheduler.run_daily_update()

# 运行自定义更新
scheduler.run_custom_update(
    update_price=True,
    update_financial=False,
    update_sector_valuation=True,
    sector_date_range=5
)
```

### 配置板块估值参数

```python
from data.processor import IntegratedDataPipeline

pipeline = IntegratedDataPipeline()

# 修改配置
pipeline.configure_sector_valuation({
    'enabled': True,
    'date_range': 30,
    'save_intermediate': True,
    'output_formats': ['pkl', 'csv', 'json']
})

# 运行更新
pipeline.run_full_pipeline()
```

## 输出文件

板块估值计算会生成以下文件：

### 1. 主数据文件
- `StockData/SectorData/sector_valuation_from_stock_pe.pkl`
  - 二进制格式，包含完整的板块估值时间序列
  - 包含字段：TradingDate, Sector, PE_TTM, PB, TotalMarketCap等

### 2. CSV格式
- `StockData/SectorData/sector_valuation_from_stock_pe.csv`
  - 便于Excel查看和分析
  - UTF-8编码，支持中文

### 3. 汇总报告
- `StockData/SectorData/sector_valuation_summary.json`
  - JSON格式的统计汇总
  - 包含最新日期的PE/PB统计信息

## 数据字段说明

| 字段名 | 说明 | 单位 |
|--------|------|------|
| TradingDate | 交易日期 | - |
| Sector | 行业名称 | - |
| StockCount | 成分股数量 | 个 |
| ValidStocks | 有效股票数（有市值） | 个 |
| TotalMarketCap | 板块总市值 | 元 |
| PE_TTM | 板块市盈率TTM | 倍 |
| TotalProfit_TTM | 板块总净利润TTM | 元 |
| PE_StockCount | 计算PE的有效股票数 | 个 |
| PB | 板块市净率 | 倍 |
| TotalEquity | 板块总净资产 | 元 |
| PB_StockCount | 计算PB的有效股票数 | 个 |

## 配置文件

系统配置文件：`config/data_update.yaml`

主要配置项：
- `sector_valuation.enabled`: 是否启用板块估值计算
- `sector_valuation.default_date_range`: 默认计算天数
- `update_tasks.daily/weekly/monthly`: 不同周期的更新任务配置

## 日志文件

日志保存在：`logs/data_update_YYYYMMDD.log`

## 性能说明

- 计算1天数据：约10-15秒
- 计算30天数据：约1-2分钟
- 计算252天数据：约5-10分钟

内存使用：
- 基础内存需求：2GB
- 完整更新需求：4-6GB

## 常见问题

### Q: 板块PE值异常怎么办？
A: 检查EP_ttm和BP数据是否正确，系统会自动转换EP→PE和BP→PB。

### Q: 如何只计算特定行业？
A: 目前版本计算所有行业，后续版本将支持行业筛选。

### Q: 更新失败如何处理？
A: 查看日志文件了解具体错误，可使用`--force`参数强制重新计算。

## 联系支持

如遇到问题，请查看日志文件或联系系统管理员。