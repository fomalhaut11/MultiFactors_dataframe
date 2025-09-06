# Generator 模块重构指南

## 重构目标

将 `factors/generator` 模块重构为纯数据处理工具模块，移除所有Factor类定义，保留核心计算逻辑作为可复用的工具函数。

## 重构原则

1. **保留计算逻辑**：所有有价值的因子计算公式都保留为工具函数
2. **移除Factor类**：不再继承FactorBase，不包含name、category等属性
3. **纯函数设计**：所有函数都是纯函数，无副作用
4. **易于调用**：提供简单的函数接口，方便在library/注册系统中使用

## 迁移映射

### 原有Factor类 → 工具函数
```python
# 原来的Factor类
class ROE_ttm_Factor(FactorBase):
    def calculate(self, data):
        # ... 计算逻辑
        return result

# 重构后的工具函数
def calculate_roe_ttm(financial_data: pd.DataFrame, **kwargs) -> pd.Series:
    """计算TTM净资产收益率"""
    # ... 相同的计算逻辑
    return result
```

### 依赖处理
- **financial_report_processor**: 已迁移到 `generators/financial/`
- **数据处理混入**: 计算逻辑直接内联到工具函数中
- **字段映射**: 使用硬编码的实际字段名

## 重构后的目录结构

```
factors/generator/  (保留作为工具函数库)
├── financial/
│   ├── profitability_tools.py    # ROE、ROA等盈利能力计算工具
│   ├── value_tools.py           # PE、PB等估值计算工具
│   ├── quality_tools.py         # 盈利质量计算工具
│   └── __init__.py
├── technical/
│   ├── price_tools.py           # 价格相关计算工具
│   ├── volume_tools.py          # 成交量相关计算工具
│   └── __init__.py
├── mixed/
│   ├── mixed_calculation_tools.py  # 混合因子计算工具
│   └── __init__.py
└── __init__.py

library/  (新的因子注册系统)
├── factor_registry.py          # 因子注册装饰器和管理
├── financial_factors.py        # 财务因子注册
├── technical_factors.py        # 技术因子注册
└── __init__.py
```

## 使用方式对比

### 重构前
```python
from factors.generator.financial.profitability_factors import ROE_ttm_Factor

factor = ROE_ttm_Factor()
result = factor.calculate(financial_data)
```

### 重构后
```python
# 直接使用工具函数
from factors.generator.financial.profitability_tools import calculate_roe_ttm
result = calculate_roe_ttm(financial_data)

# 或者通过注册系统
from factors import get_factor_calculator
calc_func = get_factor_calculator('ROE_ttm')
result = calc_func(financial_data)
```

## 重构步骤

1. ✅ 创建 `generators/` 纯工具模块
2. 🔄 重构 `factors/generator/` 为工具函数库
3. ⏳ 建立 `factors/library/` 注册系统
4. ⏳ 迁移现有因子定义到注册系统
5. ⏳ 更新AI助手接口

## 保持兼容性

在重构过程中，保持向后兼容：
- 原有的Factor类暂时保留，添加deprecation警告
- 提供兼容性包装函数
- 逐步迁移现有代码

## 完成标志

- [ ] 所有计算逻辑转换为纯工具函数
- [ ] 移除所有FactorBase继承
- [ ] 建立完整的函数式接口
- [ ] 注册系统完整可用
- [ ] AI助手能正常调用新接口