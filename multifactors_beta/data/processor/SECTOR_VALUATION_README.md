# 板块估值计算器使用说明

## 功能概述

`SectorValuationFromStockPE` 是一个高效的行业板块估值计算器，通过个股PE反推净利润的方式计算板块整体估值指标。

## 核心优势

1. **无需重复计算TTM**：直接使用已计算好的个股PE_ttm
2. **避免日频扩展**：无需将财务数据扩展到日频，大幅提升性能
3. **独立性强**：不依赖factors模块，避免循环引用
4. **计算准确**：使用市值加权方式，真实反映板块估值水平

## 计算逻辑

```
个股净利润_TTM = 个股市值 / 个股PE_TTM
板块总净利润_TTM = Σ(个股净利润_TTM)
板块总市值 = Σ(个股市值)
板块PE_TTM = 板块总市值 / 板块总净利润_TTM
```

## 使用方法

### 基本使用

```python
from data.processor import SectorValuationFromStockPE

# 创建处理器
processor = SectorValuationFromStockPE()

# 计算板块估值（默认252个交易日）
result = processor.process()

# 计算最近30天
result = processor.process(date_range=30)
```

### 输出数据

计算结果包含以下字段：

| 字段名 | 说明 |
|--------|------|
| TradingDate | 交易日期 |
| Sector | 行业名称 |
| StockCount | 成分股数量 |
| ValidStocks | 有效股票数（有市值数据） |
| TotalMarketCap | 板块总市值（元） |
| PE_TTM | 板块市盈率TTM |
| TotalProfit_TTM | 板块总净利润TTM（元） |
| PE_StockCount | 计算PE的有效股票数 |
| PB | 板块市净率 |
| TotalEquity | 板块总净资产（元） |
| PB_StockCount | 计算PB的有效股票数 |

### 输出文件

处理结果自动保存到：
- `E:\Documents\PythonProject\StockProject\StockData\SectorData\sector_valuation_from_stock_pe.pkl`
- `E:\Documents\PythonProject\StockProject\StockData\SectorData\sector_valuation_from_stock_pe.csv`
- `E:\Documents\PythonProject\StockProject\StockData\SectorData\sector_valuation_summary.json`

## 数据依赖

需要以下数据文件：

1. **价格数据**：`StockData/Price.pkl`（包含市值MC字段）
2. **个股PE数据**：`StockData/RawFactors/EP_ttm.pkl` 或 `PE_ttm.pkl`
3. **个股PB数据**：`StockData/RawFactors/BP.pkl` 或 `PB.pkl`（可选）
4. **行业分类**：`StockData/Classificationdata/classification_one_hot.pkl`

## 性能优化

- **快速计算**：处理一年数据约需30-60秒
- **内存优化**：逐日计算，避免内存溢出
- **并行潜力**：可改进为并行计算进一步提速

## 示例结果

2025年2月5日部分板块估值：

| 行业 | PE_TTM | 市值(万亿) |
|------|--------|------------|
| 交通运输 | 17.7 | 31.7 |
| 公用事业 | 18.3 | 32.6 |
| 基础化工 | 25.3 | 31.9 |
| 医药生物 | 33.4 | 58.1 |
| 商贸零售 | 41.7 | 9.4 |
| 传媒 | 42.7 | 14.7 |

## 注意事项

1. **数据单位**：市值单位为元，PE、PB为倍数
2. **数据质量**：依赖个股PE数据的质量
3. **更新频率**：建议每日收盘后更新
4. **内存使用**：处理全部历史数据需要约2-4GB内存

## 扩展功能

未来可扩展：
- PS_TTM（市销率）
- PCF_TTM（市现率）
- 板块估值历史分位数
- 板块间估值比较
- 估值异常预警

## 维护说明

该模块完全独立，维护时注意：
1. 不要从factors模块导入任何内容
2. 保持数据路径配置的灵活性
3. 确保与data.processor其他模块的兼容性