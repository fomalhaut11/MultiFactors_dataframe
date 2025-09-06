# 财务报表数据处理器迁移说明

## 迁移内容

**原文件**: `factors/base/time_series_processor.py`  
**新文件**: `factors/generator/financial/financial_report_processor.py`  
**新类名**: `FinancialReportProcessor` (原 `TimeSeriesProcessor`)

## 迁移原因

1. **命名问题**: `time_series_processor.py` 听起来像通用时间序列工具
2. **实际功能**: 专门处理**财务报表数据**的时间序列变换 (TTM、YoY、QoQ等)
3. **位置问题**: 放在 `base/` 目录会误导开发者以为是通用工具
4. **可发现性差**: 开发者难以发现这些重要的财务计算功能

## 新文件功能

`FinancialReportProcessor` 提供财务报表特有的计算功能：

### 核心方法
- `calculate_ttm()` - TTM (Trailing Twelve Months) 计算
- `calculate_yoy()` - YoY (Year-over-Year) 同比增长率
- `calculate_qoq()` - QoQ (Quarter-over-Quarter) 环比增长率  
- `calculate_single_quarter()` - 单季度数据转换
- `calculate_avg()` - 期末期初平均值计算
- `calculate_zscores_timeseries()` - 时间序列标准化
- `calculate_rank_timeseries()` - 滚动排名

### 使用示例

```python
# 新的推荐用法
from factors.generator.financial.financial_report_processor import FinancialReportProcessor

# 计算TTM净利润
ttm_data = FinancialReportProcessor.calculate_ttm(financial_data)
ttm_earnings = ttm_data['net_income']

# 计算同比增长率
yoy_growth = FinancialReportProcessor.calculate_yoy(financial_data)
```

## 向后兼容

为了不破坏现有代码，提供了向后兼容支持：

```python
# 仍然可以使用（向后兼容）
from factors.base import TimeSeriesProcessor

# 或从新位置导入
from factors.generator.financial.financial_report_processor import TimeSeriesProcessor
```

## 建议

**新代码请使用**：
- 文件: `financial_report_processor.py`
- 类名: `FinancialReportProcessor`

这样可以：
1. 提高代码可读性
2. 明确表示这是财务专用工具
3. 避免与通用时间序列工具混淆

## 相关更新

- ✅ `factors/base/__init__.py` - 更新导入路径
- ✅ `factors/__init__.py` - 添加新的导出
- ✅ `factors/generator/financial/ep_ttm_factor.py` - 示例因子已更新
- ✅ `PROJECT_STRUCTURE.md` - 结构文档已更新
- ✅ `ARCHITECTURE_V3.md` - 架构文档已更新

---

**日期**: 2025-09-03  
**影响**: 所有使用财务报表数据时间序列计算的代码  
**状态**: 已完成，提供向后兼容