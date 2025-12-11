# 新因子开发场景完整指南

## 📍 当前场景：新因子开发
**适用情况**: 创建新的因子类、编写因子计算逻辑、实现新的财务或技术指标

**如需切换场景，返回主导航** → @../CLAUDE.md

---

## 🚨 开始前强制检查清单

### 第一步：搜索现有实现（强制执行，不可跳过）
**在写任何代码前，必须完成以下搜索**：

```bash
# 1. 搜索相关功能关键词
grep -r "ttm\|TTM\|trailing.*twelve" factors/generators/
grep -r "同比\|yoy\|YOY" factors/generators/
grep -r "环比\|qoq\|QOQ" factors/generators/

# 2. 检查generators/__init__.py中的可用工具
cat factors/generators/__init__.py

# 3. 搜索类似因子实现  
find factors -name "*.py" -type f | xargs grep -l "你要实现的指标名称"
```

### 第二步：验证必须使用现有工具（不可违反）
- [ ] 确认factors/generators中没有相同功能的工具
- [ ] 确认不会重复实现TTM、YOY、QOQ等财务计算
- [ ] 确认不会使用factors/generator_backup/目录中的代码

**如果发现现有工具，必须停止自己实现** → @anti-duplication-guide.md

---

## 📝 新因子开发标准流程

### 1. 因子设计阶段

#### 1.1 明确因子定义
```python
# 示例：定义你的因子
# 因子名称: [具体名称]
# 计算公式: [详细公式]  
# 经济含义: [解释经济学意义]
# 数据需求: [需要哪些基础数据]
```

#### 1.2 确定因子分类
- **纯财务因子** → 放置在 `factors/repository/financial/`
- **纯技术因子** → 放置在 `factors/repository/technical/`  
- **复合因子** → 放置在 `factors/repository/mixed/`

### 2. 实现准备阶段

#### 2.1 导入必需工具（强制使用）
```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
[因子名称] - 因子说明
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional
from pathlib import Path

# 必须使用的基类
from ...base.factor_base import FactorBase

# 必须使用的工具（根据需要选择）
from factors.generators import (
    calculate_ttm,           # TTM计算
    calculate_yoy,           # 同比增长
    calculate_qoq,           # 环比增长  
    calculate_single_quarter,# 单季度计算
    FinancialReportProcessor # 财报数据处理
)

# 数据加载器（必须使用）
from factors.utils.data_loader import FactorDataLoader

logger = logging.getLogger(__name__)
```

#### 2.2 因子类基础结构（标准模板）
```python
class YourFactorName(FactorBase):
    """
    [因子名称]因子
    
    计算公式: [详细公式]
    经济含义: [解释]
    """
    
    def __init__(self):
        super().__init__(
            name="YourFactorName",
            category="financial"  # 或 "technical" 或 "mixed"
        )
        self.description = "[因子的简短描述]"
    
    def calculate(self) -> pd.Series:
        """
        计算[因子名称]因子
        
        Returns:
        --------
        pd.Series
            MultiIndex[TradingDates, StockCodes]格式的因子值
        """
        try:
            logger.info(f"开始计算{self.name}因子...")
            
            # 1. 加载数据（使用标准加载器）
            financial_data = FactorDataLoader.load_financial_data()
            
            # 2. 使用现有工具进行计算（禁止重复实现）
            # 示例：如果需要TTM计算
            ttm_data = calculate_ttm(financial_data)
            
            # 3. 实现你的特定计算逻辑
            factor_values = self._calculate_specific_logic(ttm_data)
            
            # 4. 数据质量检查
            self._validate_factor_data(factor_values)
            
            logger.info(f"✅ {self.name}因子计算完成")
            return factor_values
            
        except Exception as e:
            logger.error(f"❌ 计算{self.name}因子失败: {e}")
            raise
    
    def _calculate_specific_logic(self, data: pd.DataFrame) -> pd.Series:
        """实现你的具体计算逻辑"""
        # 这里写你的特定计算逻辑
        # 必须返回MultiIndex[TradingDates, StockCodes]格式
        pass
    
    def _validate_factor_data(self, factor_data: pd.Series):
        """验证因子数据质量"""
        if factor_data.empty:
            raise ValueError("因子计算结果为空")
        
        if not isinstance(factor_data.index, pd.MultiIndex):
            raise ValueError("因子数据必须是MultiIndex格式")
        
        if factor_data.index.names != ['TradingDates', 'StockCodes']:
            raise ValueError("索引名称必须为[TradingDates, StockCodes]")
    
    def get_factor_info(self) -> dict:
        """获取因子信息"""
        return {
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "formula": "[你的计算公式]",
            "data_requirements": ["列出需要的数据"],
            "output_format": "MultiIndex Series [TradingDates, StockCodes]"
        }

# 便捷函数（可选）
def create_your_factor() -> YourFactorName:
    """创建因子实例"""
    return YourFactorName()
```

### 3. 常见财务计算使用示例

#### 3.1 TTM计算（必须使用现有工具）
```python
# ❌ 错误：自己实现TTM
# profit_ttm = financial_data.groupby('StockCodes')['PROFIT'].rolling(4).sum()

# ✅ 正确：使用现有工具  
from factors.generators import calculate_ttm
profit_ttm = calculate_ttm(financial_data)
```

#### 3.2 同比增长计算（必须使用现有工具）
```python
# ❌ 错误：自己实现同比
# revenue_yoy = (current_revenue / last_year_revenue - 1) * 100

# ✅ 正确：使用现有工具
from factors.generators import calculate_yoy  
revenue_yoy = calculate_yoy(financial_data)
```

#### 3.3 数据日频扩展（必须使用现有工具）
```python
# ✅ 正确：使用财报处理器进行日频扩展
from factors.generators import FinancialReportProcessor

daily_factor = FinancialReportProcessor.expand_to_daily_vectorized(
    factor_data=quarterly_factor,
    release_dates=release_dates, 
    trading_dates=trading_dates
)
```

---

## 🚨 关键陷阱预警

### 最常见的错误
1. **重复实现TTM计算** 
   - ❌ 使用 `.rolling(4).sum()`
   - ✅ 使用 `calculate_ttm()`

2. **使用backup目录代码**
   - ❌ `from factors.generator_backup.financial import xxx`
   - ✅ `from factors.generators import xxx`

3. **硬编码路径**
   - ❌ `pd.read_pickle('/path/to/data.pkl')`
   - ✅ 使用config/main.yaml中的路径配置

4. **错误的数据格式**
   - ❌ 返回DataFrame或单层索引Series
   - ✅ 返回MultiIndex[TradingDates, StockCodes] Series

### 发现自己在重复造轮子？
**立即停止当前工作** → @anti-duplication-guide.md

---

## 📚 相关文档链接

- **工具详细说明** → @factor-generators-guide.md
- **数据格式规范** → @data-formats-guide.md  
- **代码质量标准** → @code-quality-checklist.md
- **测试你的因子** → @factor-testing-scenario.md

---

## 📋 开发完成检查清单

开发完成后，确认以下各项：

- [ ] 使用了factors.generators中的现有工具，没有重复实现
- [ ] 返回的数据格式为MultiIndex[TradingDates, StockCodes]  
- [ ] 代码中没有硬编码的路径、日期、股票代码
- [ ] 没有使用generator_backup目录中的任何代码
- [ ] 所有字符编码兼容Windows GBK，没有使用emoji
- [ ] 包含了完整的因子信息说明（get_factor_info方法）
- [ ] 添加了适当的日志输出和错误处理

**全部确认后，可以进入测试阶段** → @factor-testing-scenario.md