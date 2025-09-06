# 多因子量化投资系统 - 项目配置

## 项目信息
- **项目类型**: 多因子量化投资研究框架
- **架构版本**: v2.1 (生产级)
- **主要语言**: Python 3.9+
- **数据科学导向**: 大量DataFrame和时间序列处理
- **🤖 核心设计理念**: AI助手是系统的主要用户界面

## 🎯 AI助手优先开发原则

**所有factors模块的开发都必须围绕核心目标：让AI助手能够通过`factors/quant_assistant.py`接口高效地帮助用户完成量化研究任务。**

### 开发决策标准
每个新功能、接口设计、代码修改都要问：
1. **AI助手能否轻松理解**这个功能的用途和参数？
2. **AI助手能否准确调用**这个功能并获得结果？
3. **AI助手能否向用户清楚解释**结果和建议后续操作？

### AI助手接口规范
- **核心文件**: `factors/quant_assistant.py` - AI助手与系统交互的唯一入口
- **接口要求**: 参数简单、返回值包含interpretation字段、错误信息友好
- **不要创建**: 面向人类的复杂技术接口，AI助手难以使用

## 核心编码规范

### 字符编码要求 (强制)
**严禁使用emoji符号和特殊Unicode字符**
- 禁止: emoji符号如目标、图表、清单、闪电、庆祝、勾选、叉号、警告等
- 禁止: 特殊Unicode符号如箭头、星号等装饰符号
- 使用: 纯中文汉字、英文字母、阿拉伯数字、标准标点符号
- 替代方案: "成功"代替勾选符号, "失败"代替叉号, "警告"代替警告符号

**Windows GBK编码兼容性要求:**
- 所有Python代码、注释、输出文本必须与GBK编码兼容
- 测试文件和日志输出不得包含GBK无法编码的字符
- 使用简体中文而非繁体中文或特殊汉字

**编码工具使用:**
- 对于包含用户输出的模块，导入编码工具: `from core.utils.encoding_utils import safe_print, clean_emoji_text`
- 使用 `safe_print()` 替代 `print()` 进行输出
- 测试代码和日志输出必须使用编码安全的函数
### 数据处理要求
- **NaN处理**: 所有因子计算必须正确处理NaN值，使用pandas的`.fillna()`, `.dropna()`或`.isna()`
- **向量化优先**: DataFrame操作优先使用向量化而非循环，提升300-500%性能
- **内存优化**: 大数据集处理时使用chunked处理或流式计算
- **时间序列**: 优先使用pandas的时间序列功能，避免手动日期处理

### 数据格式规范
- **统一日期**: 使用`TradingDates.pkl`中的交易日期，不要自定义日期范围
- **财务数据**: 统一使用`data/auxiliary/FinancialData_unified.pkl`
- **MultiIndex**: 因子数据必须使用(TradingDates, StockCodes)的MultiIndex格式
- **数据类型**: 数值数据使用float64，日期使用datetime64

## 项目架构规范

### 配置管理系统
- **统一配置入口**: 使用`config`包进行所有配置管理
- **主配置文件**: `config/main.yaml` - 数据库、路径、系统参数
- **因子配置**: `config/factors.yaml` - 因子生成参数和批次配置
- **字段映射**: `config/field_mappings.yaml` - 财务数据字段映射
- **代理配置**: `config/agents.yaml` - AI助手专业配置
- **配置加载**: `from config import get_config, ConfigManager`

### 核心结构设计
    'ARCHITECTRUE_V3.md'
### 模块化设计
- **基类继承**: 新因子继承`factors.base.FactorBase`
- **混入类**: 使用现有混入类如`DataProcessingMixin`, `TestableMixin`
- **数据适配器**: 优先使用`FlexibleDataAdapter`处理不同数据源
- **字段映射**: 使用`config/field_mappings.yaml`进行字段映射

### 文件组织
- 纯财务因子 → `factors/generator/financial/`
- 混合因子(财务+市值) → `factors/generator/mixed/`
- 技术因子 → `factors/generator/technical/`
- **不要修改**: `factors/base/`中的核心基类

## 测试要求

### 必须测试
- **单元测试**: 新增因子必须包含单元测试
- **数据验证**: 验证因子输出格式和数据质量  
- **性能测试**: 大数据集处理， 请先等待处理进程完成或者报错后介入。

### 强制使用现有工具
- **单因子测试**: 必须使用`SingleFactorTestPipeline()`，禁止重写测试逻辑
- **因子分析**: 优先使用`factors.analyzer`模块中的工具
- **数据处理**: 优先使用`factors.tester.core`中的数据管理器
- **结果管理**: 使用`ResultManager`保存和加载测试结果
- **重要**: 发现现有工具能解决问题时，必须使用现有工具，不得重复开发

### 测试命令
```bash
# 运行所有测试
pytest tests/

# 运行单元测试
pytest tests/unit/

# 运行集成测试
pytest tests/integration/
```

## 常用工作流程

### 数据准备
```bash
# 预处理辅助数据（必须执行）
python data/prepare_auxiliary_data.py

# 增量更新价格数据
python scheduled_data_updater.py --data-type price
```

### 因子生成
```bash
# 批量生成因子
python batch_generate_factors.py

# 快速生成测试
python quick_generate_factors.py

# 验证因子质量
python validate_factors.py
```

## 重要约束

### 不要做的事
- **不要硬编码**路径、日期、股票代码
- **不要修改**核心配置文件(`config/main.yaml`, `config/field_mappings.yaml`)
- **不要忽略**NaN值处理
- **不要使用**循环处理大DataFrame
- **严禁重复开发**现有功能（如单因子测试、因子分析等）
- **新因子创建场景禁止暴露generator中的预定义计算公式** - 这是测试未注册、未验证因子的场景

### 优先使用
- **现有测试工具**: `SingleFactorTestPipeline`, `FactorScreener`, `ResultManager`等
- **现有分析工具**: `factors.analyzer`模块中的所有分析器
- **现有数据工具**: `factors.tester.core`中的数据管理器和处理器
- **工具函数**: `factors/utils/`中的工具函数
- **配置系统**: 配置化的字段映射而非硬编码字段名
- **增量处理**: 优先增量处理而非全量重计算

## 性能要求
- 内存使用: 峰值 < 12GB
- 并行处理: 优先使用pandas的并行特性
## 数据安全
- 不要在代码中暴露数据库密码
- 敏感配置使用环境变量
- 不要提交包含真实数据的文件

---

**重要提醒**: 这是生产级量化系统，代码质量和性能都有严格要求。遇到问题时，优先查看现有的类似实现，遵循已有的设计模式。