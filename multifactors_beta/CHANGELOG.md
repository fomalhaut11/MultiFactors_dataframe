# 变更日志 / Changelog

本文档记录多因子量化投资系统的所有重要变更。

版本号遵循[语义化版本控制](https://semver.org/lang/zh-CN/)：主版本号.次版本号.修订号

---

## [4.0.0] - 2025-09-07

### 🚀 重大更新 - AI助手优先架构

#### 架构升级
- **四层架构设计**：新增AI助手层，形成"AI助手层 → 数据层 → 因子层 → 策略层 → 基础层"的完整架构
- **智能路由机制**：AI助手自动识别用户需求，选择最佳工具和策略
- **统一配置管理**：建立config/包体系，支持类型验证、环境隔离和热重载

#### 新增功能
- **AI量化助手V2.0** (`factors/ai_quant_assistant_v2.py`)
  - 自然语言量化研究交互界面
  - 智能因子生成和筛选决策
  - 完整的AI助手配置系统
- **因子管理CLI工具**增强
  - 新增validate、cleanup、backup、restore命令
  - 因子注册表完整性验证
  - 自动化因子管理流程

#### 配置系统重构
- **统一配置体系**：`config/`包
  - `main.yaml` - 系统核心配置（从根目录config.yaml迁移）
  - `factors.yaml` - 因子生成配置（464行，新增）
  - `field_mappings.yaml` - 数据字段映射（351行）
  - `agents.yaml` - AI代理专业配置（186行）
- **配置管理器**：支持分层配置、类型验证、默认值处理

#### 因子架构优化
- **generators重构**：`factors/generator/` → `factors/generators/`（纯工具库）
- **generator_backup**：旧代码移至备份目录，标记为已废弃
- **因子操作工具箱**：新增operations/模块（约2,000行）
  - 截面操作（排序、标准化、去极值）
  - 时序操作（移动平均、滞后、差分）
  - 复合操作（动量、波动率）
  - 因子组合（线性、正交化、残差化）

#### 因子库扩展
- **因子库系统**：`factors/library/`
  - `factor_registry.py` - 因子注册表（424行）
  - `loader.py` - 动态因子加载器（384行）
  - `validator.py` - 因子验证器（444行）
- **因子元数据管理**：`factors/meta/factor_registry.py`（405行）

#### 文档体系完善
- 新增4,000+行架构和使用文档
  - `ARCHITECTURE_V3.md` - 四层架构设计（570行）
  - `PROJECT_STRUCTURE.md` - 项目结构说明（重写，576行）
  - `DATA_DEPENDENCY_MAP.md` - 数据依赖图谱（255行）
  - `DATA_UPDATE_GUIDE.md` - 数据更新指南（269行）
  - `DATA_UPDATER_API.md` - API文档（605行）
- AI助手使用指南和演示
  - `factors/AI_ASSISTANT_BRAIN.md` - AI决策逻辑
  - `examples/demos/demo_ai_assistant_usage.py` - 使用演示

#### 示例和工具重组
- **examples/目录**重新组织
  - `demos/` - AI助手和功能演示
  - `benchmarks/` - 性能基准测试
  - `factor_examples/` - 因子计算示例
  - `workflows/` - 完整工作流示例
- **tools/目录**规范化
  - `helpers/` - 辅助工具（subagent_manager.py, agent_helper.py）
  - `debug/` - 调试工具集合

#### 项目清理
- 删除30+个临时测试文件（约3,000行过时测试代码）
- 移除调试和实验性脚本
- 清理根目录，优化项目结构

#### 破坏性变更 ⚠️
- `config.yaml` 已移动到 `config/main.yaml`
- `factors/generator/` 已废弃，使用 `factors/generators/`
- 部分旧测试文件已删除，迁移至新测试框架

#### 迁移指南
参见 `factors/generator_backup/GENERATOR_REFACTOR_GUIDE.md`

#### 变更统计
- 265个文件变更
- +40,629行新增
- -14,844行删除
- 净增约25,785行代码

---

## [2.1.0-beta] - 2025-08-24

### 🎉 批量因子生成系统 + 因子组合 + 风险模型 + 回测系统

#### 批量因子生成系统（三套方案）
- **快速生成模式** (`quick_generate_factors.py`)
  - 零配置，开箱即用
  - 预设因子集合：core（15个）、basic（8个）、test（4个）
  - 适合新手快速上手
- **配置驱动模式** (`advanced_factor_generator.py`)
  - 基于YAML配置的智能生成
  - 支持因子分组：financial、technical、risk、mixed
  - 精确控制因子生成流程
- **完整批量模式** (`batch_generate_factors.py`)
  - 支持60+因子一键生成
  - 并行计算优化
  - 完整的质量验证

#### 因子质量保障
- **因子验证系统** (`validate_factors.py`)
  - 数据完整性检查（空值、无穷值）
  - 分布特征分析（偏度、峰度、变异系数）
  - 异常值检测（IQR方法）
  - 质量评分：A/B/C/D四级评定
- **因子配置管理** (`factor_config.yaml`)
  - 460+行配置定义
  - 因子分组和依赖管理
  - 自定义因子支持

#### 因子组合系统
- **权重计算方法**（5种）
  - `EqualWeight` - 等权重组合
  - `ICWeight` - IC加权组合
  - `IRWeight` - 信息比率加权
  - `RiskParityWeight` - 风险平价加权
  - `OptimalWeight` - 最优权重（最大化IC）
- **组合方法**（4种）
  - `LinearCombiner` - 线性组合
  - `OrthogonalCombiner` - 正交化组合
  - `PCANeutralization` - PCA中性化
  - `NeutralizationCombiner` - 行业/市值中性化

#### 因子选择系统
- **筛选器**（4种）
  - `PerformanceFilter` - 基于IC、ICIR、收益率筛选
  - `CorrelationFilter` - 相关性控制
  - `StabilityFilter` - 时间序列稳定性筛选
  - `CompositeFilter` - 复合筛选器（AND/OR逻辑）
- **选择策略**（3种）
  - `TopNSelector` - TopN选择
  - `ThresholdSelector` - 阈值选择
  - `ClusteringSelector` - K-means聚类选择

#### 风险模型框架
- **协方差估计器**（4种）
  - `SampleCovarianceEstimator` - 样本协方差
  - `LedoitWolfEstimator` - Ledoit-Wolf收缩（性能最佳）
  - `ExponentialWeightedEstimator` - 指数加权
  - `RobustCovarianceEstimator` - 稳健估计（异常值处理）
- **风险模型**（3种）
  - `CovarianceModel` - 协方差风险模型
  - `BarraModel` - Barra多因子风险模型
  - `FactorModel` - 通用因子模型（PCA、混合）

#### 回测系统框架
- **回测引擎** (`backtest/engine/`)
  - 事件驱动回测框架
  - 多策略并行支持
  - 完整时间管理
- **交易成本模型** (`backtest/cost/`)
  - 佣金模型（多种计算方式、阶梯费率）
  - 市场冲击模型（线性/非线性）
  - 滑点模型（固定/比例）
- **绩效分析** (`backtest/performance/`)
  - 风险指标（Sharpe、最大回撤、Alpha/Beta）
  - 归因分析（因子贡献分解）
  - 可视化报告

#### 因子评估体系
- **五维度评估**
  - 盈利能力（IC均值、ICIR、收益率）
  - 稳定性（时间序列稳定性、IC波动）
  - 及时性（因子延迟、信号衰减）
  - 可交易性（换手率、容量限制）
  - 独特性（相关性、信息贡献）
- **智能评分系统**
  - 综合评分计算
  - 智能诊断报告
  - 改进建议生成

#### 性能优化
- 50只股票建模 < 1秒
- 200只股票建模 0.16-3.75秒
- 并行计算支持（多核加速）
- 内存优化（批处理、分块处理）

#### 新增文档
- 批量因子生成使用指南
- 因子组合和选择教程
- 风险模型API文档
- 回测系统使用指南

---

## [2.0.0] - 2025-08-13

### 🔄 Factors模块架构重构

#### 架构设计
- **三层架构**：生成层 → 测试层 → 分析层
- **MultiIndex数据格式统一**：所有因子采用`[TradingDates, StockCodes]`格式
- **模块解耦**：generator、tester、analyzer职责清晰分离

#### Factors模块重构
- **generator/模块**
  - `financial/` - 纯财务因子（60+个）
  - `technical/` - 技术因子（基础实现）
  - `risk/` - 风险因子（Beta系列）
  - `mixed/` - 混合因子（估值因子）
- **tester/模块**完整实现
  - `SingleFactorTestPipeline` - 单因子测试流水线
  - `DataManager` - 数据管理（因子版本选择）
  - `FactorTester` - 核心测试逻辑（IC分析、分组测试）
  - `ResultManager` - 结果保存和加载
- **analyzer/模块**基础实现
  - `FactorScreener` - 因子筛选（预设条件）
  - `CorrelationAnalyzer` - 相关性分析
  - `StabilityAnalyzer` - 稳定性分析
  - `FactorEvaluator` - 综合评估

#### 测试增强
- 换手率分析
- IC衰减分析
- 滚动窗口测试
- 单调性检验

#### 混合因子系统
- `MixedFactorManager` - 统一管理器
- 估值因子：BP、EP_ttm、SP_ttm、CFP_ttm
- 支持多数据源协同（财务+市场数据）

---

## [1.2.0] - 2025-09-05

### 🐛 性能优化和修复

#### FinancialReportProcessor性能优化
- 优化TTM计算性能（向量化处理）
- 修复因子字段映射问题
- 改进内存使用效率

#### 因子实现修复
- 修正财务因子计算逻辑
- 更新因子字段映射配置
- 完善数据验证流程

---

## [1.0.0] - 2025-02-11

### 🎊 初始发布

#### 核心功能
- 基础因子计算框架
- 单因子测试系统
- 数据获取和预处理
- 配置管理系统

#### 已实现模块
- `core/` - 核心基础设施
- `data/` - 数据获取和处理
- `factors/generator/financial/` - 财务因子（初始版本）
- `factors/tester/` - 单因子测试（基础版本）

#### 数据支持
- 价格数据获取和更新
- 财务数据（三表）获取
- 增量更新机制

---

## 版本规划

### [4.1.0] - 计划中
- 技术因子库扩充（动量、反转、趋势）
- 风险因子库完善（波动率、VaR、CVaR）
- 组合优化器实现（均值方差、风险平价）
- 回测系统完善（策略模板、高级分析）

### [5.0.0] - 未来规划
- 机器学习因子挖掘
- 实时交易接口
- Web管理界面
- 分布式计算支持

---

## 维护信息

- **项目状态**：生产就绪（Production Ready）
- **当前版本**：v4.0.0
- **开发进度**：87%
- **核心特色**：AI助手优先，智能路由，统一配置管理
- **维护团队**：MultiFactors Team
- **许可证**：MIT

---

## 链接

- [项目结构](PROJECT_STRUCTURE.md)
- [快速参考](QUICK_REFERENCE.md)
- [架构设计](ARCHITECTURE_V3.md)
- [贡献指南](CONTRIBUTING.md)

---

*本文档遵循[如何维护更新日志](https://keepachangelog.com/zh-CN/)规范*
