# 项目文档中心

**最后更新**: 2025-12-11
**项目版本**: v4.0.0
**文档体系**: Diataxis 框架（学习、操作、参考、说明）

---

## 📖 文档导航说明

本项目采用 **Diataxis 文档框架**，提供四种类型的文档满足不同需求：

```
📚 Diataxis 框架四大文档类型:

学习导向 (Tutorials)      → 通过实践学习技能
  ↓
目标导向 (How-to Guides)  → 解决具体问题
  ↓
信息导向 (Reference)      → 查阅技术细节
  ↓
理解导向 (Explanation)    → 深入理解概念
```

**📌 如何选择文档**:
- **想学习新技能** → [Tutorials](tutorials/)
- **遇到具体问题** → [How-to Guides](how-to/)
- **查找API信息** → [Reference](reference/)
- **理解原理概念** → [Explanation](explanation/)
- **需要代码示例** → [Examples](examples/)

---

## 🚀 快速入门（根目录文档）

**新用户必读**，建议按顺序阅读：

1. **[README.md](../README.md)** ⭐⭐⭐
   - 5分钟了解项目：功能特性、快速开始、核心架构
   - 适合：所有人

2. **[CLAUDE.md](../CLAUDE.md)** ⭐⭐
   - AI助手使用指南：场景路由、工作场景识别
   - 适合：AI辅助开发用户

3. **[DOCUMENTATION_OPTIMIZATION_PLAN.md](../DOCUMENTATION_OPTIMIZATION_PLAN.md)**
   - 文档优化计划：Diataxis框架实施方案
   - 适合：文档维护者

---

## 📚 Diataxis 文档体系

### 🎓 教程 (Tutorials)

**学习导向** - 通过实践学习

**入门教程**:
1. **[开发第一个因子](tutorials/02-develop-first-factor.md)** ⭐⭐⭐
   - 完整因子开发流程
   - 学习时间：30分钟
   - 前置要求：Python基础

**完整教程索引** → [tutorials/README.md](tutorials/README.md)

---

### 🔧 操作指南 (How-to Guides)

**目标导向** - 解决具体问题

#### 数据操作
- **[准备辅助数据](how-to/data/prepare-auxiliary-data.md)** ⭐⭐
  - 生成TradingDates、ReleaseDates等辅助数据
- **[更新价格数据](how-to/data/update-price-data.md)** ⭐⭐
  - Price数据增量更新

#### 因子操作
- **[防止重复造轮子](how-to/factors/avoid-duplication.md)** ⭐⭐⭐
  - 必读检查清单，避免重复实现
- **[分析和筛选因子](how-to/factors/analyze-and-screen-factors.md)** ⭐⭐
  - 因子筛选分析流程
- **[更新因子数据](how-to/factors/update-factors.md)**
  - 因子增量更新

#### 测试操作
- **[测试单因子](how-to/testing/test-single-factor.md)** ⭐⭐
  - 单因子测试完整流程

**完整操作指南索引** → [how-to/README.md](how-to/README.md)

---

### 📖 参考手册 (Reference)

**信息导向** - 技术文档和规范

#### API参考
- **[Generators API](reference/api/generators-api.md)** ⭐⭐⭐
  - 核心数据处理工具集
  - 财务工具 (TTM, YoY, QoQ等)
  - 技术指标工具
  - Alpha191操作函数

**完整参考文档索引** → [reference/README.md](reference/README.md)

---

### 💡 说明文档 (Explanation)

**理解导向** - 概念和原理

#### 核心概念
- **[财报数据处理](explanation/concepts/financial-data-processing.md)** ⭐⭐
  - 财报字段、TTM/YoY/QoQ概念
- **[交易日期](explanation/concepts/trading-dates.md)** ⭐
  - 交易日历、日期对齐

#### 最佳实践
- **[性能优化](explanation/best-practices/performance-optimization.md)** ⭐⭐
  - 性能优化方法和增量处理

**完整说明文档索引** → [explanation/README.md](explanation/README.md)

---

### 📝 示例代码 (Examples)

**实用导向** - 可运行的代码

#### 高级用法
- **[BP因子案例研究](examples/advanced/bp-factor-case-study.md)** ⭐⭐⭐
  - 完整的估值因子实现案例

**完整示例索引** → [examples/README.md](examples/README.md)

---

## 📂 按角色分类

### 👤 新用户/研究员
**推荐阅读顺序**：
1. [README.md](../README.md) - 项目总览
2. [开发第一个因子](tutorials/02-develop-first-factor.md) - 上手实践
3. [准备辅助数据](how-to/data/prepare-auxiliary-data.md) - 数据准备

### 👨‍💻 因子开发者
**必读文档**：
1. [开发第一个因子](tutorials/02-develop-first-factor.md) ⭐⭐⭐
2. [防止重复造轮子](how-to/factors/avoid-duplication.md) ⭐⭐⭐
3. [Generators API](reference/api/generators-api.md) ⭐⭐⭐
4. [测试单因子](how-to/testing/test-single-factor.md) ⭐⭐

### 🏗️ 架构师/高级开发者
**核心文档**：
1. [财报数据处理](explanation/concepts/financial-data-processing.md)
2. [性能优化](explanation/best-practices/performance-optimization.md)
3. [BP因子案例研究](examples/advanced/bp-factor-case-study.md)

### 🔧 运维/数据管理员
**关键文档**：
1. [准备辅助数据](how-to/data/prepare-auxiliary-data.md)
2. [更新价格数据](how-to/data/update-price-data.md)
3. [配置系统说明](../config/README.md)

---

## 🗂️ 子模块文档

各模块目录下的README文档：

| 模块 | 文档路径 | 说明 |
|------|---------|------|
| **配置管理** | [config/README.md](../config/README.md) | 配置系统说明 ⭐ |
| **核心模块** | [core/README.md](../core/README.md) | 核心工具说明 |
| **数据模块** | [data/README.md](../data/README.md) | 数据获取和处理 |
| **调试工具** | [debug/README.md](../debug/README.md) | 调试工具说明 |
| **测试规范** | [tests/README.md](../tests/README.md) | 测试规范 ⭐ |
| **工具脚本** | [tools/README.md](../tools/README.md) | 辅助工具 |

---

## 🔍 快速查找

### 我想了解...

- **如何开始？** → [README.md](../README.md)
- **如何开发新因子？** → [开发第一个因子](tutorials/02-develop-first-factor.md)
- **如何准备数据？** → [准备辅助数据](how-to/data/prepare-auxiliary-data.md)
- **如何测试因子？** → [测试单因子](how-to/testing/test-single-factor.md)
- **避免重复造轮子？** → [防止重复造轮子](how-to/factors/avoid-duplication.md)
- **查找工具API？** → [Generators API](reference/api/generators-api.md)
- **理解财报处理？** → [财报数据处理](explanation/concepts/financial-data-processing.md)

---

## 📌 文档约定

### 优先级标记
- ⭐⭐⭐ 必读文档
- ⭐⭐ 推荐阅读
- ⭐ 参考文档
- 无标记：专题文档

### 文档状态
- **维护中** - 持续更新
- **稳定** - 无需频繁更新
- **归档** - 历史参考，不再维护

---

## 📚 归档文档 (archived/)

**历史参考，不再维护**：

#### 测试报告 (archived/test-reports/)
- `FACTORS_CONFIG_INTEGRATION_TEST_REPORT.md` - 集成测试报告（2025-09-06）
- `TESTING_REPORT.md` - 测试框架报告（2025-08-29）

#### 迁移指南 (archived/migration-guides/)
- `STOCK_UNIVERSE_PATH_UPDATE.md` - 路径更新说明（已完成）

---

## 📞 获取帮助

### 文档问题
- 文档缺失：提交Issue或联系维护者
- 文档错误：提交Pull Request修正
- 文档建议：在Issue中讨论

### 技术问题
- 参考相关文档后仍无法解决
- 在GitHub Issues中提问
- 提供详细的错误信息和环境描述

---

## 📝 文档维护

### 更新原则
1. **及时更新** - 功能变更时同步更新文档
2. **定期清理** - 每月清理过时文档
3. **版本标注** - 重要文档标注更新日期
4. **结构清晰** - 保持Diataxis框架分类

### 贡献文档
欢迎改进文档：
1. Fork项目
2. 在docs/目录下修改或新增文档
3. 遵循Diataxis框架分类
4. 提交Pull Request

---

## 🎯 Diataxis 框架简介

本项目采用 [Diataxis](https://diataxis.fr/) 文档框架，这是一个成熟的文档组织方法论，被 Django、Gatsby 等大型项目采用。

### 四种文档类型

| 类型 | 导向 | 目标 | 形式 |
|------|------|------|------|
| **Tutorials** | 学习 | 让新手通过实践学会技能 | 手把手教学 |
| **How-to** | 目标 | 解决特定问题 | 问题→步骤 |
| **Reference** | 信息 | 提供准确技术信息 | 干巴巴但准确 |
| **Explanation** | 理解 | 解释概念和原理 | 为什么这样设计 |

### 为什么使用 Diataxis？

✅ **基于成熟框架** - 行业标准，非自创体系
✅ **用户需求导向** - 根据用户意图组织文档
✅ **易于导航** - 清晰的文档分类和索引
✅ **易于维护** - 明确的文档职责划分

---

**文档维护**: MultiFactors Team
**最后更新**: 2025-12-11
**项目版本**: v4.0.0
