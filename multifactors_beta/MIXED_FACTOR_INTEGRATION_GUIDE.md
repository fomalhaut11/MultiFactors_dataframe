# 混合因子与 factor_generator.py 集成指南

## 🎯 概述

本指南说明如何使用新集成的 `MixedFactorGenerator` 类，以及如何通过统一的 `factor_generator.py` 接口来生成混合因子。

## 📋 新增功能

### 1. **MixedFactorGenerator 类**
- 继承自 `FactorGenerator` 基类
- 集成已有的 `MixedFactorManager`
- 支持需要多种数据源的复合因子计算
- 优化的批量计算功能

### 2. **统一工厂接口**
- `create_generator('mixed')` 创建混合因子生成器
- 支持 4 种因子类型：`financial`, `technical`, `risk`, `mixed`

### 3. **增强的数据验证**
- 严格的输入参数验证
- 详细的错误提示信息
- 改进的异常处理机制

## 🚀 使用方法

### 基础用法

```python
from factors.generator.factor_generator import create_generator
import pandas as pd

# 1. 创建混合因子生成器
mixed_generator = create_generator('mixed')

# 2. 查看可用因子
available_factors = mixed_generator.get_available_factors()
print(f"可用混合因子: {available_factors}")
# 输出: ['BP', 'EP_ttm', 'SP_ttm', 'CFP_ttm']

# 3. 查看因子分类
categories = mixed_generator.get_factor_categories()
print(f"因子分类: {categories}")
# 输出: {'valuation': ['BP', 'EP_ttm', 'SP_ttm', 'CFP_ttm']}
```

### 数据准备

**⚠️ 重要：混合因子需要字典格式的数据！**

```python
# 准备数据（必须是字典格式）
data = {
    'financial_data': financial_df,  # 财务数据 DataFrame，MultiIndex[TradingDates, StockCodes]
    'market_cap': market_cap_series   # 市值数据 Series，MultiIndex[TradingDates, StockCodes]
}

# 检查特定因子的数据需求
requirements = mixed_generator.get_data_requirements(['BP', 'EP_ttm'])
print(f"数据需求: {requirements}")
# 输出: ['financial_data', 'market_cap']
```

### 单个因子生成

```python
# 生成单个因子
try:
    bp_factor = mixed_generator.generate('BP', data)
    print(f"BP因子生成成功，数据点数: {len(bp_factor)}")
except Exception as e:
    print(f"生成失败: {e}")
```

### 批量因子生成

```python
# 批量生成多个因子
factor_names = ['BP', 'EP_ttm', 'SP_ttm', 'CFP_ttm']
results = mixed_generator.batch_generate(factor_names, data)

for factor_name, result in results.items():
    if result is not None:
        print(f"✅ {factor_name}: 成功，数据点数 {len(result)}")
    else:
        print(f"❌ {factor_name}: 失败")
```

### 因子保存和加载

```python
# 保存因子
if 'BP' in mixed_generator.generated_factors:
    save_path = mixed_generator.save_factor('BP', format='pkl')
    print(f"BP因子已保存到: {save_path}")

# 加载因子
loaded_bp = mixed_generator.load_factor('BP', format='pkl')
print(f"BP因子加载成功，数据点数: {len(loaded_bp)}")
```

## 📊 数据格式要求

### 财务数据 (financial_data)
```python
# DataFrame 格式，MultiIndex[TradingDates, StockCodes]
financial_data = pd.DataFrame({
    'equity': [...],           # 净资产
    'earnings': [...],         # 净利润
    'revenue': [...],          # 营业收入
    'operating_cashflow': [...], # 经营现金流
    'quarter': [...]           # 季度信息
}, index=pd.MultiIndex.from_product([dates, stocks], names=['TradingDates', 'StockCodes']))
```

### 市值数据 (market_cap)
```python
# Series 格式，MultiIndex[TradingDates, StockCodes]
market_cap = pd.Series(
    data=[...],  # 市值数据
    index=pd.MultiIndex.from_product([dates, stocks], names=['TradingDates', 'StockCodes']),
    name='market_cap'
)
```

## ⚡ 性能优化

### 批量计算优势
- `MixedFactorGenerator` 重写了 `batch_generate` 方法
- 利用 `MixedFactorManager` 的批量计算功能
- 相同计算器类型的因子会被分组优化

### 错误恢复机制
- 批量计算失败时自动回退到单个计算
- 详细的错误日志记录
- 部分成功的结果仍然返回

## 🔧 错误处理

### 常见错误和解决方案

1. **数据格式错误**
```python
# ❌ 错误：传入Series而不是字典
mixed_generator.generate('BP', financial_data)
# ValueError: 混合因子 BP 需要多种数据源，请提供字典格式的数据

# ✅ 正确：传入字典
mixed_generator.generate('BP', {'financial_data': financial_data, 'market_cap': market_cap})
```

2. **缺少必需数据**
```python
# ❌ 错误：缺少market_cap数据
data = {'financial_data': financial_data}  # 缺少market_cap
mixed_generator.generate('BP', data)
# ValueError: 因子 BP 缺少必需数据: ['market_cap']
```

3. **不支持的因子**
```python
# ❌ 错误：因子名称不存在
mixed_generator.generate('UNKNOWN_FACTOR', data)
# ValueError: 不支持的混合因子: UNKNOWN_FACTOR
```

## 🧪 测试脚本

运行测试脚本验证功能：

```bash
python test_mixed_factor_generator.py
```

测试脚本包含：
- 混合因子生成器创建测试
- 模拟数据生成
- 单个和批量因子生成测试
- 因子保存/加载测试
- 错误处理测试
- 与其他生成器的集成测试

## 📈 集成示例

### 完整的工作流程

```python
from factors.generator.factor_generator import create_generator
import pandas as pd
import pickle

# 1. 创建生成器
mixed_gen = create_generator('mixed')

# 2. 加载实际数据
with open('data/auxiliary/FinancialData_unified.pkl', 'rb') as f:
    financial_data = pickle.load(f)
    
with open('MarketCap.pkl', 'rb') as f:
    market_cap = pickle.load(f)

# 3. 准备数据
data = {
    'financial_data': financial_data,
    'market_cap': market_cap
}

# 4. 批量生成估值因子
valuation_factors = ['BP', 'EP_ttm', 'SP_ttm', 'CFP_ttm']
results = mixed_gen.batch_generate(valuation_factors, data)

# 5. 保存因子
for factor_name, factor_data in results.items():
    if factor_data is not None:
        mixed_gen.save_factor(factor_name, factor_data, format='pkl')
        print(f"✅ 因子 {factor_name} 已保存")

print("🎉 混合因子生成完成!")
```

## 🔮 未来扩展

### 待实现的混合因子类型

1. **规模因子 (Size Factors)**
   - Size, LogSize (基于市值)

2. **流动性因子 (Liquidity Factors)**  
   - Turnover, ILLIQ (基于价格和成交量)

3. **质量因子 (Quality Factors)**
   - 财务质量评分 (基于多个财务指标)

### 扩展步骤
1. 在 `factors/generator/mixed/` 目录下创建新的计算器
2. 在 `MixedFactorManager` 中注册新计算器
3. 更新 `factor_config.yaml` 配置文件
4. `MixedFactorGenerator` 会自动支持新因子

## 📝 总结

通过 `MixedFactorGenerator` 的集成，现在可以：

✅ **统一接口**：通过 `create_generator('mixed')` 创建  
✅ **类型安全**：严格的数据验证和错误处理  
✅ **高效计算**：优化的批量计算功能  
✅ **灵活扩展**：易于添加新的混合因子类型  
✅ **完整测试**：全面的测试覆盖

这套架构为混合因子的开发和使用提供了稳固的基础！