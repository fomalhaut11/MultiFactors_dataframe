# 示例代码 (Examples)

**文档类型**: 实用导向 - 可运行的代码示例
**适合读者**: 需要快速参考和复用代码的用户
**目标**: 提供可直接使用或修改的代码示例

---

## 📝 示例分类

### 🎓 基础用法 ([basic-usage/](basic-usage/))

常见功能的基础示例：

- **生成因子示例** (计划中) - 如何生成一个简单因子
- **测试因子示例** (计划中) - 如何测试因子效果
- **分析因子示例** (计划中) - 如何分析因子表现

### 🚀 高级用法 ([advanced/](advanced/))

复杂场景和高级功能示例：

- **[BP因子案例研究](advanced/bp-factor-case-study.md)** - 完整的估值因子实现案例
- **自定义因子示例** (计划中) - 开发复杂的自定义因子
- **因子组合示例** (计划中) - 多因子组合策略
- **混合因子示例** (计划中) - 财务+市场数据混合因子

---

## 🎯 如何使用示例代码

### 使用方式

1. **直接运行**: 大部分示例可以直接运行
2. **复制修改**: 复制示例代码，根据需求修改
3. **学习参考**: 理解代码结构和实现方式
4. **调试基础**: 遇到问题时作为对比参考

### 运行示例

```bash
# 1. 确保环境配置正确
python --version  # Python 3.9+

# 2. 确保数据已准备
python data/prepare_auxiliary_data.py

# 3. 运行示例
python examples/basic-usage/generate_factor.py
```

---

## 📖 示例代码规范

### 代码结构

```python
"""
示例名称 - 简短描述

功能说明：
- 功能1
- 功能2

依赖：
- pandas
- numpy
- factors

运行方式：
python examples/xxx.py
"""

# 导入依赖
import pandas as pd
from factors import calculate_ttm

# 配置参数
CONFIG = {
    'data_path': 'path/to/data',
    'output_path': 'path/to/output',
}

def main():
    """主函数"""
    # 1. 加载数据
    data = load_data()

    # 2. 处理数据
    result = process_data(data)

    # 3. 保存结果
    save_result(result)

    print("完成！")

if __name__ == '__main__':
    main()
```

### 代码要求

1. **完整可运行**: 包含所有必要的导入和配置
2. **注释清晰**: 关键步骤都有注释说明
3. **错误处理**: 适当的异常处理
4. **输出明确**: 清楚地输出运行结果
5. **参数可配**: 重要参数可以修改

---

## 📊 示例索引

### 按功能分类

#### 数据处理
- 财务数据TTM计算
- 数据格式转换
- 缺失值处理

#### 因子生成
- 财务因子（ROE、ROA等）
- 技术因子（MA、RSI等）
- 混合因子（EP_TTM等）

#### 因子测试
- 单因子IC测试
- 分组回测
- 因子稳定性分析

#### 因子组合
- 等权重组合
- IC加权组合
- 优化权重组合

### 按难度分类

#### ⭐ 入门级
- 生成简单因子
- 运行基础测试
- 查看分析结果

#### ⭐⭐ 中级
- 自定义因子开发
- 完整的测试流程
- 因子筛选分析

#### ⭐⭐⭐ 高级
- 复杂混合因子
- 多因子组合策略
- 性能优化技巧

---

## 🔍 查找示例

### 我想做什么？

| 需求 | 示例文件 | 难度 |
|------|----------|------|
| 生成ROE因子 | basic-usage/roe_factor.py | ⭐ |
| 计算TTM数据 | basic-usage/ttm_calculation.py | ⭐ |
| 测试因子效果 | basic-usage/test_factor.py | ⭐⭐ |
| 开发混合因子 | advanced/mixed_factor.py | ⭐⭐⭐ |
| BP因子案例 | advanced/bp-factor-case-study.md | ⭐⭐⭐ |

### 我遇到了什么问题？

| 问题 | 示例参考 |
|------|----------|
| 不知道如何开始 | basic-usage/ 所有示例 |
| TTM计算不对 | basic-usage/ttm_calculation.py |
| 测试结果不理想 | basic-usage/test_factor.py |
| 需要复杂因子 | advanced/ 高级示例 |

---

## 💡 贡献示例代码

欢迎贡献新的示例代码！请遵循以下原则：

1. **实用性强**: 解决实际问题的代码
2. **可运行性**: 确保代码可以直接运行
3. **注释清晰**: 关键步骤有详细注释
4. **参数可配**: 重要参数可以轻松修改
5. **结构清晰**: 代码组织合理，易于理解

### 提交示例

1. **选择分类**: 确定示例属于basic还是advanced
2. **编写代码**: 遵循代码规范
3. **测试验证**: 确保代码可运行
4. **编写说明**: 添加README或注释说明
5. **提交PR**: 提交Pull Request

### 示例模板

```python
"""
[示例名称] - [一句话描述]

功能说明：
- 详细说明功能1
- 详细说明功能2

依赖：
- 列出所有依赖包

运行方式：
python examples/category/example_name.py

作者：Your Name
日期：YYYY-MM-DD
"""

import pandas as pd
from factors import xxx

# 配置
CONFIG = {
    'param1': 'value1',
    'param2': 'value2',
}

def load_data():
    """加载数据"""
    pass

def process_data(data):
    """处理数据"""
    pass

def save_result(result):
    """保存结果"""
    pass

def main():
    """主函数"""
    print("开始执行...")

    # 步骤1
    data = load_data()
    print("✓ 数据加载完成")

    # 步骤2
    result = process_data(data)
    print("✓ 数据处理完成")

    # 步骤3
    save_result(result)
    print("✓ 结果保存完成")

    print("全部完成！")

if __name__ == '__main__':
    main()
```

---

## 📚 相关文档

- **教程**: 系统学习 → [tutorials/](../tutorials/)
- **操作指南**: 解决具体问题 → [how-to/](../how-to/)
- **API参考**: 查阅技术细节 → [reference/](../reference/)

---

**维护者**: MultiFactors Team
**最后更新**: 2025-12-11
