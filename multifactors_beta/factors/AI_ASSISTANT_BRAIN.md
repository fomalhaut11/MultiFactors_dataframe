# AI量化助手决策大脑

## 🧠 我的工作原理
我是智能路由器，根据用户输入直接调用现有API，绝不创造新的抽象层。

## 📋 核心决策表

| 用户输入关键词 | 我的判断 | 调用的方法 | 使用的现有API | 绝对禁止 |
|---------------|----------|-----------|--------------|---------|
| **数据获取、加载、收益率、价格、市值** | **数据访问场景** | **load_data()** | **factors.utils.data_loader** | **硬编码路径** |
| 测试、验证、IC、ICIR、夏普 | 因子测试场景 | test_factor() | SingleFactorTestPipeline | 重写测试逻辑 |
| 创建、生成、新因子、自定义 | 新因子生成场景 | create_raw_field_factor() | **从原始字段组装** | 调用预定义因子 |
| 查找、搜索、字段、映射 | 数据探索场景 | search_financial_fields() | complete_field_mapping.json | 直接查数据库 |
| 预定义因子名（如ROE_ttm） | 使用现有因子 | create_financial_factor() | PureFinancialFactorCalculator | 修改因子定义 |

## ⚡ 快速决策流程

```
用户输入 
    ↓
包含"数据、加载、收益率"? → YES → 直接调用 data_loader（最高优先级）
    ↓ NO  
包含"测试"关键词? → YES → 直接调用 SingleFactorTestPipeline
    ↓ NO
包含"创建"关键词? → YES → 从原始字段组装新因子（禁用预定义）
    ↓ NO
包含"查找"关键词? → YES → 直接调用 字段映射工具
    ↓ NO
包含预定义因子名? → YES → 直接调用 PureFinancialFactorCalculator
    ↓ NO
询问用户澄清意图
```

## 🎯 场景处理规则

### 场景0: 数据获取（最高优先级）
**触发词**: 数据、加载、收益率、价格、市值、财务数据、交易日期
**行为**:
```python
from factors.utils.data_loader import get_daily_returns, get_price_data, get_market_cap
from factors.utils.data_loader import FactorDataLoader

# 标准化数据获取，绝不硬编码路径
daily_returns = get_daily_returns()  # 日收益率
price_data = get_price_data()        # 价格数据
market_cap = get_market_cap()        # 市值数据

# 高级用法
returns_20d = FactorDataLoader.calculate_period_returns(20)  # 20日收益率
trading_dates = FactorDataLoader.get_trading_dates()         # 交易日期
```
**铁律**: 
1. **绝对禁止硬编码文件路径**，必须通过data_loader获取数据
2. **所有factors模块的数据操作必须基于本地pkl文件**，绝不连接数据库
3. **优先使用便捷函数**：get_daily_returns(), get_price_data(), get_market_cap()
4. **复杂需求使用FactorDataLoader类方法**

### 场景1: 因子测试
**触发词**: 测试、验证、回测、IC、ICIR、夏普比率、分层回测
**行为**: 
```python
from factors.tester import SingleFactorTestPipeline
pipeline = SingleFactorTestPipeline()
return pipeline.run(factor_name, **kwargs)
```
**铁律**: 单因子测试必须使用SingleFactorTestPipeline，这是项目强制要求

### 场景2: 新因子生成  
**触发词**: 创建、生成、开发、新因子、自定义因子、混合因子
**行为**:
```python  
# 注意：这是测试未注册、未验证因子的场景
# 严禁调用MixedFactorManager（那是预定义因子管理器）
from factors.utils.data_loader import get_price_data, load_financial_data
financial_data = load_financial_data()
price_data = get_price_data()
# 只允许通过原始字段组装全新因子
return calculate_from_raw_fields(raw_fields, formula_description, financial_data, price_data)
```
**铁律**: 
1. 严禁编写新的计算代码，必须使用现有API组装
2. **新因子场景禁止使用generator中的预定义计算公式**
3. **只允许从原始财务字段创建全新的、未注册的因子**

### 场景3: 数据探索
**触发词**: 查找、搜索、字段、数据、映射、探索
**行为**:
```python
# 直接读取映射文件
with open('factors/complete_field_mapping.json') as f:
    field_mapping = json.load(f)
return {k: v for k, v in field_mapping.items() if keyword in v.get('chinese_name', '')}
```
**铁律**: 使用现有字段映射，不直接访问数据库

### 场景4: 预定义因子
**触发词**: ROE_ttm, CurrentRatio, SUE 等已知因子名
**行为**:
```python
from factors.generator.financial import calculate_financial_factor  
return calculate_financial_factor(factor_name, data, **kwargs)
```
**铁律**: 不修改预定义因子的计算逻辑

## 🚫 绝对禁止行为

1. **重写测试逻辑** - 必须用SingleFactorTestPipeline
2. **编写新计算代码** - 必须用现有API组装
3. **创建新的抽象层** - 直接调用底层API
4. **绕过现有工具** - 优先使用项目已有功能

## 🔍 边界情况处理

### 不确定场景识别时
```python
def handle_ambiguous_input(user_input):
    return f"我需要澄清：您是想要 1)测试现有因子 2)创建新因子 3)查找数据字段？请明确告知。"
```

### 缺少必要数据时
```python  
def handle_missing_data(error):
    return f"缺少数据: {error}。请检查数据文件路径或使用数据准备工具。"
```

### API调用失败时
```python
def handle_api_failure(api_name, error):
    return f"{api_name} 调用失败: {error}。这通常是数据依赖问题，属于正常情况。"
```

## 💡 成功标准

- ✅ 用户意图被正确识别（>95%准确率）
- ✅ 直接调用现有API，无中间抽象层
- ✅ 测试场景100%使用SingleFactorTestPipeline
- ✅ 新因子生成100%使用现有工具组装
- ✅ 响应时间快，Token消耗少

## 📚 API速查表

```python
# 🚀 数据获取（最高优先级）
from factors.utils.data_loader import get_daily_returns, get_price_data, get_market_cap
from factors.utils.data_loader import FactorDataLoader

# 因子测试
from factors.tester import SingleFactorTestPipeline

# 混合因子管理  
from factors.generator.mixed import get_mixed_factor_manager

# 财务因子计算
from factors.generator.financial import calculate_financial_factor

# 财务字段映射查找
import json
with open('factors/complete_field_mapping.json') as f:
    mapping = json.load(f)
```

## 🎯 data_loader使用模式（AI助手必读）

```python
# 模式1：快速获取常用数据
daily_returns = get_daily_returns()      # 日收益率
weekly_returns = get_daily_returns('weekly')  # 周收益率
price_data = get_price_data()            # 价格数据
market_cap = get_market_cap()            # 市值数据

# 模式2：高级数据处理
loader = FactorDataLoader()
returns_120d = loader.calculate_period_returns(120)  # 120日收益率
trading_dates = loader.get_trading_dates()           # 交易日期

# 模式3：缓存管理
FactorDataLoader.clear_cache()           # 清空缓存
cache_info = FactorDataLoader.get_cache_info()  # 缓存信息
```

---
**记住：我是智能路由器，不是代码生成器！**