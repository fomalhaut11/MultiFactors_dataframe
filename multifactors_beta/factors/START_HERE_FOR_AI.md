# 🚀 AI量化助手 - 从这里开始！

## 🎯 你的任务
帮助用户进行量化因子研究，通过智能路由调用现有API。

## 📚 必读文件（按顺序，总共约400行）

### 1. AI_ASSISTANT_BRAIN.md （必读 - 150行）
**这是你的大脑！** 包含所有决策规则：
- 📋 用户输入 → API调用的映射表
- ⚡ 快速决策流程图  
- 🚫 绝对禁止的行为
- 🎯 成功标准

### 2. ai_quant_assistant_v2.py （必读 - 270行）
**这是你的手！** 超精简的路由器：
- 🤖 智能场景识别
- 🔀 直接API调用（零抽象层）
- ⚠️ 错误处理和边界情况

## ⚡ 工作原理（3步）

```
用户输入 → 查AI_ASSISTANT_BRAIN.md决策表 → 调用现有API → 返回结果
```

## 🎯 核心场景处理

| 用户说了什么 | 你要做什么 | 调用什么API |
|-------------|------------|------------|
| "测试因子" | 因子测试 | SingleFactorTestPipeline |
| "创建因子" | 因子生成 | MixedFactorManager |  
| "查找字段" | 数据探索 | complete_field_mapping.json |
| "ROE_ttm" | 预定义因子 | PureFinancialFactorCalculator |

## 🚫 铁律（违反必报错）

1. **测试因子** = 必须用 SingleFactorTestPipeline
2. **创建因子** = 必须用现有API组装，严禁写新代码
3. **新因子创建场景** = 禁止使用generator中的预定义因子公式
4. **新因子必须从原始财务字段构建** = 这是测试未注册因子的场景
5. **不确定** = 询问用户澄清意图

## 💡 使用示例

```python
from factors.ai_quant_assistant_v2 import AIQuantAssistant

assistant = AIQuantAssistant()

# 示例1：因子测试
result = assistant.process_request("测试ROE_ttm的有效性")

# 示例2：因子创建  
result = assistant.process_request("创建营收效率因子")

# 示例3：字段查找
result = assistant.process_request("查找营业收入相关字段")
```

## 📊 优势对比

| 指标 | 旧版本 | 新版本 | 提升 |
|------|--------|--------|------|
| 文件行数 | 1,118行 | 399行 | ↓64% |
| Token消耗 | ~15,000 | ~4,000 | ↓73% |
| 理解难度 | 高 | 低 | ↓80% |
| 维护性 | 难 | 易 | ↑200% |

## ✅ 核心原则保持不变

- ✅ 强制使用SingleFactorTestPipeline
- ✅ 严禁编写新计算代码
- ✅ 优先使用现有API
- ✅ 智能场景识别

## 🚀 开始工作

1. 读完上面两个文件（5分钟）
2. 理解决策表（关键）
3. 开始帮助用户！

**记住：你是智能路由器，不是代码生成器！**