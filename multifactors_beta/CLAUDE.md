# 多因子量化投资系统 - 场景导航中心

## 🚨 通用强制约束（所有场景必读）

### 核心原则
- **严禁重复造轮子** - 必须使用现有工具，禁止重复实现
- **必须使用factors.generators工具集** - 所有财务、技术计算的官方实现
- **禁止使用generator_backup目录** - 那里都是备份文件，非正式组件

### 数据格式规范
- **MultiIndex格式**: 因子数据必须使用`[TradingDates, StockCodes]`格式
- **字符编码**: Windows GBK兼容，严禁emoji和特殊Unicode字符
- **配置路径**: 使用`config/main.yaml`中的路径配置，禁止硬编码

## 🎯 工作场景路由

### 📝 新因子开发场景
**适用情况**: 创建新的因子类或计算逻辑
```
引导路径:
第一步 → @docs/tutorials/02-develop-first-factor.md  (完整开发流程)
工具指南 → @docs/reference/api/generators-api.md      (generators工具集详解)
陷阱预警 → @docs/how-to/factors/avoid-duplication.md  (防重复造轮子)
```

### 🧪 因子测试场景
**适用情况**: 测试因子效果、进行回测分析
```
引导路径:
测试流程 → @docs/how-to/testing/test-single-factor.md  (测试完整流程)
结果分析 → @docs/how-to/factors/analyze-and-screen-factors.md (结果解读方法)
```

### 🔍 代码审查/重构场景
**适用情况**: 修改现有代码、优化性能、代码重构
```
引导路径:
性能优化 → @docs/explanation/best-practices/performance-optimization.md (性能优化指南)
```

### 🔗 系统集成场景
**适用情况**: 模块集成、配置管理、部署相关
```
引导路径:
数据准备 → @docs/how-to/data/prepare-auxiliary-data.md (辅助数据生成)
数据更新 → @docs/how-to/data/update-price-data.md       (价格数据更新)
```

## 🤖 AI助手场景识别指南

### 如何确定当前场景？
- **正在写新的因子计算逻辑？** → 新因子开发场景
- **正在写`class XXXFactor`类？** → 新因子开发场景
- **正在调用测试框架？** → 因子测试场景
- **正在分析因子IC、收益等？** → 因子测试场景
- **正在修改现有的因子代码？** → 代码审查/重构场景
- **正在整合多个模块？** → 系统集成场景

**不确定场景？** 查阅 → @docs/tutorials/ 或 @docs/how-to/ 目录

## ⚡ 快速工具查找

### 财务计算工具速查
```python
# 最常用的工具 - 直接导入使用
from factors.generators import (
    calculate_ttm,           # TTM计算 - 唯一官方实现
    calculate_yoy,           # 同比增长计算
    calculate_qoq,           # 环比增长计算
    calculate_zscore,        # 财报时序zscores
    expand_to_daily_vectorized  # 将财报数据填充到交易日数据， 所有因子都必须是填充到交易日数据的MultiIndex 的series [TradingDates, StockCodes]
)
```
**详细说明** → @docs/reference/api/generators-api.md

### 测试工具速查
```python
# 强制使用的测试工具
from factors.tester import SingleFactorTestPipeline
from factors.analyzer import FactorScreener
```
**详细说明** → @docs/how-to/testing/test-single-factor.md

## 🚨 常见错误预警

### 最频繁的错误
1. **重新实现TTM计算** - 必须使用`calculate_ttm()`
2. **使用backup目录代码** - 只能使用`generators/`正式工具
3. **自写测试逻辑** - 必须使用`SingleFactorTestPipeline`
4. **硬编码路径** - 必须使用配置文件路径
5. **测试文件位置错误** - 所有测试文件必须在`tests/`目录下，禁止在根目录或业务代码目录创建测试文件

### 错误发生时的应对
发现自己在重复造轮子？立即停止 → @docs/how-to/factors/avoid-duplication.md

---

## 📋 项目基本信息
- **项目类型**: 多因子量化投资研究框架
- **架构版本**: v4.0.0 (生产级)
- **主要语言**: Python 3.9+
- **核心理念**: AI助手是系统的主要用户界面

**详细架构信息** → @docs/explanation/concepts/ 目录
