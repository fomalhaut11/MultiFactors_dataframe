# 项目文档中心

**最后更新**: 2025-11-27
**项目版本**: v4.0.0
**文档体系**: 三级架构（根目录 → docs/ → 子模块/）

---

## 📖 文档导航说明

本项目采用**三级文档体系**，适合不同角色和需求：

```
L0 (根目录)  → 快速入门文档
L1 (docs/)   → 详细使用文档（本目录）
L2 (子模块/) → 模块专属文档
```

---

## 🚀 快速入门（L0 - 根目录）

**新用户必读**，建议按顺序阅读：

1. **[README.md](../README.md)** ⭐⭐⭐
   - 5分钟了解项目：功能特性、快速开始、核心架构
   - 适合：所有人

2. **[CLAUDE.md](../CLAUDE.md)** ⭐⭐
   - AI助手使用指南：场景路由、工作场景识别
   - 适合：AI辅助开发用户

3. **[DOCUMENTATION_AUDIT_REPORT.md](../DOCUMENTATION_AUDIT_REPORT.md)**
   - 文档结构审查报告：文档重构说明
   - 适合：文档维护者

---

## 📚 详细文档（L1 - docs/）

### 🏗️ 架构设计

**了解项目设计理念和技术架构**：

- **[项目架构v3.0](architecture/ARCHITECTURE_V3.md)** ⭐⭐⭐
  - 四层架构设计（AI助手层 → 策略层 → 因子层 → 数据层 → 基础层）
  - 模块职责划分
  - 架构演进历史

- **[项目结构说明](architecture/PROJECT_STRUCTURE.md)** ⭐⭐
  - 完整目录结构
  - 文件组织方式
  - 命名规范

- **[因子架构最终版](architecture/FACTORS_ARCHITECTURE_FINAL.md)** ⭐
  - 因子模块三层架构（生成→测试→分析）
  - 因子数据流
  - 设计模式

---

### 📖 使用指南

#### 数据处理指南 (guides/data/)

- **[数据更新指南](guides/data/DATA_UPDATE_GUIDE.md)** ⭐⭐⭐
  - 定时更新配置
  - 增量更新机制
  - 健康检查

- **[数据模块完整指南](guides/data/DATA_MODULE_COMPLETE.md)** ⭐⭐
  - 数据获取和处理流程
  - 数据结构规范
  - 数据依赖关系

- **[数据更新API文档](guides/data/DATA_UPDATER_API.md)** ⭐
  - API接口说明
  - 参数详解
  - 使用示例

- **[数据依赖图谱](guides/data/DATA_DEPENDENCY_MAP.md)**
  - 数据流向关系
  - 模块依赖分析

#### 因子开发指南 (guides/factors/)

**⚠️ 因子开发必读三篇**：

1. **[新因子开发完整流程](new-factor-scenario.md)** ⭐⭐⭐
   - 完整开发步骤
   - 标准代码模板
   - 常见陷阱避免

2. **[防重复造轮子指南](anti-duplication-guide.md)** ⭐⭐⭐
   - 强制检查清单
   - 现有工具查找
   - 避免重复实现

3. **[因子生成器工具集](factor-generators-guide.md)** ⭐⭐⭐
   - factors.generators完整API
   - 财务/技术/混合/Alpha191工具
   - 使用示例

**其他因子文档**：

- **[因子快速入门](guides/factors/FACTORS_QUICK_START.md)** ⭐
  - 5分钟生成第一个因子
  - 批量生成工作流

- **[因子仓库指南](guides/factors/FACTOR_REPOSITORY_GUIDE.md)** ⭐
  - 因子注册管理
  - 动态加载机制

- **[混合因子集成指南](guides/factors/MIXED_FACTOR_INTEGRATION_GUIDE.md)**
  - 多数据源协同
  - 混合因子开发示例

#### 因子测试指南

- **[单因子测试模块](单因子测试模块使用指南.md)** ⭐⭐
  - SingleFactorTestPipeline使用
  - IC分析和分组测试

- **[因子筛选分析](因子筛选分析模块使用指南.md)** ⭐⭐
  - FactorScreener使用
  - 筛选策略配置

- **[因子更新模块](因子更新模块使用指南.md)**
  - 因子增量更新
  - 版本管理

#### 回测系统指南 (guides/backtest/)

- **[回测使用指南](guides/backtest/backtest_usage_guide.md)** ⭐
  - 回测引擎使用
  - 策略配置
  - 绩效分析

#### 通用指南 (guides/general/)

- **[快速参考](guides/general/QUICK_REFERENCE.md)** ⭐⭐
  - 常用命令速查
  - 快速问题解决

- **[编码问题解决](guides/general/encoding-guide.md)** ⭐
  - Windows GBK兼容
  - 环境配置
  - 字符编码处理

- **[股票池规范](guides/general/STOCK_UNIVERSE_SPECIFICATION.md)**
  - 股票池构建
  - 筛选条件

- **[高换手率股票池](guides/general/HIGH_TURNOVER_UNIVERSE_USAGE.md)**
  - 高频交易股票池
  - 流动性筛选

---

### 🔧 开发文档 (development/)

- **[版本变更历史](development/CHANGELOG.md)** ⭐⭐
  - 详细版本记录
  - 破坏性变更说明
  - 升级指南

- **[模块开发状态](development/MODULE_DEVELOPMENT_STATUS.md)** ⭐
  - 详细模块清单和API
  - 使用示例
  - 完成度追踪

- **[项目进度报告](development/PROJECT_PROGRESS.md)**
  - 各模块完成度（87%）
  - 已实现功能清单
  - 开发路线图

---

### 📦 归档文档 (archived/)

**历史参考，不再维护**：

#### 测试报告 (archived/test-reports/)
- `FACTORS_CONFIG_INTEGRATION_TEST_REPORT.md` - 集成测试报告（2025-09-06）
- `TESTING_REPORT.md` - 测试框架报告（2025-08-29）

#### 迁移指南 (archived/migration-guides/)
- `STOCK_UNIVERSE_PATH_UPDATE.md` - 路径更新说明（已完成）

---

## 📝 数据处理专题文档

### 财务数据处理
- **[财报数据处理逻辑](财报数据处理逻辑说明.md)**
  - 财报字段说明
  - TTM/YoY/QoQ计算

- **[数据预处理功能](数据预处理功能完整指南.md)**
  - 预处理流程
  - 辅助数据生成

- **[交易日期统一使用](交易日期统一使用指南.md)**
  - 交易日历管理
  - 日期对齐规则

- **[Price数据增量更新](Price数据增量更新使用说明.md)**
  - 增量更新策略
  - 数据完整性检查

### 技术规范
- **[模块接口设计规范](模块接口设计规范.md)** ⭐
  - 接口设计原则
  - 命名约定
  - 参数规范

- **[性能优化和增量处理](性能优化和增量处理说明.md)**
  - 性能优化策略
  - 增量处理方法

### 因子使用案例
- **[BP因子使用指南](BP因子使用指南.md)**
  - 估值因子实现案例
  - 混合因子开发示例

---

## 🗂️ 子模块文档（L2）

各模块目录下的README文档：

| 模块 | 文档路径 | 说明 |
|------|---------|------|
| **配置管理** | [config/README.md](../config/README.md) | 配置系统说明 ⭐ |
| **核心模块** | [core/README.md](../core/README.md) | 核心工具说明 |
| **数据模块** | [data/README.md](../data/README.md) | 数据获取和处理 |
| **调试工具** | [debug/README.md](../debug/README.md) | 调试工具说明 |
| **示例代码** | [examples/README.md](../examples/README.md) | 使用示例 |
| **测试规范** | [tests/README.md](../tests/README.md) | 测试规范 ⭐ |
| **工具脚本** | [tools/README.md](../tools/README.md) | 辅助工具 |

---

## 📂 按角色分类

### 👤 新用户/研究员
**推荐阅读顺序**：
1. [README.md](../README.md) - 项目总览
2. [快速参考](guides/general/QUICK_REFERENCE.md) - 常用命令
3. [因子快速入门](guides/factors/FACTORS_QUICK_START.md) - 5分钟上手
4. [数据更新指南](guides/data/DATA_UPDATE_GUIDE.md) - 数据准备

### 👨‍💻 因子开发者
**必读文档**：
1. [新因子开发场景](new-factor-scenario.md) ⭐⭐⭐
2. [防重复造轮子](anti-duplication-guide.md) ⭐⭐⭐
3. [因子生成器工具集](factor-generators-guide.md) ⭐⭐⭐
4. [单因子测试模块](单因子测试模块使用指南.md) ⭐⭐

### 🏗️ 架构师/高级开发者
**核心文档**：
1. [项目架构v3.0](architecture/ARCHITECTURE_V3.md)
2. [模块开发状态](development/MODULE_DEVELOPMENT_STATUS.md)
3. [模块接口设计规范](模块接口设计规范.md)
4. [性能优化说明](性能优化和增量处理说明.md)

### 🔧 运维/数据管理员
**关键文档**：
1. [数据更新指南](guides/data/DATA_UPDATE_GUIDE.md)
2. [数据依赖图谱](guides/data/DATA_DEPENDENCY_MAP.md)
3. [配置系统说明](../config/README.md)
4. [编码问题解决](guides/general/encoding-guide.md)

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

## 🔍 快速查找

### 我想了解...

- **如何开始？** → [README.md](../README.md)
- **如何开发新因子？** → [新因子开发场景](new-factor-scenario.md)
- **如何更新数据？** → [数据更新指南](guides/data/DATA_UPDATE_GUIDE.md)
- **如何测试因子？** → [单因子测试模块](单因子测试模块使用指南.md)
- **遇到编码问题？** → [编码问题解决](guides/general/encoding-guide.md)
- **项目架构？** → [项目架构v3.0](architecture/ARCHITECTURE_V3.md)
- **版本变更？** → [CHANGELOG](development/CHANGELOG.md)

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
4. **结构清晰** - 保持文档分类合理

### 贡献文档
欢迎改进文档：
1. Fork项目
2. 在docs/目录下修改或新增文档
3. 提交Pull Request
4. 说明改进内容

---

**文档维护**: MultiFactors Team
**最后更新**: 2025-11-27
**项目版本**: v4.0.0
