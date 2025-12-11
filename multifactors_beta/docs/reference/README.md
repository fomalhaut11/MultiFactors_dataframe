# 参考手册 (Reference)

**文档类型**: 信息导向 - 技术描述和规范
**适合读者**: 开发者、需要查阅技术细节的用户
**目标**: 提供准确、完整的技术信息

---

## 📖 参考文档分类

### 🔌 API参考 ([api/](api/))

模块API的详细说明：

- **[Generators API](api/generators-api.md)** ⭐⭐⭐ - 核心数据处理工具集
  - 财务工具 (TTM, YoY, QoQ等)
  - 技术指标工具
  - 混合数据处理工具
  - Alpha191操作函数
- **Tester API** (计划中) - 因子测试框架API
- **Analyzer API** (计划中) - 因子分析模块API
- **Combiner API** (计划中) - 因子组合器API

### ⚙️ 配置参考 ([config/](config/))

配置文件的详细说明：

- **主配置文件** (计划中) - main.yaml完整配置项
- **模块配置** (计划中) - 各模块的配置选项
- **环境变量** (计划中) - .env文件配置

### 📊 数据格式参考 ([data-formats/](data-formats/))

数据格式和结构规范：

- **因子数据格式** (计划中) - MultiIndex格式规范
- **财务数据格式** (计划中) - 财报数据结构
- **价格数据格式** (计划中) - 行情数据结构

### 💻 CLI参考 ([cli/](cli/))

命令行工具使用说明：

- **数据更新CLI** (计划中) - scheduled_data_updater.py
- **因子管理CLI** (计划中) - factor_manager.py

---

## 🎯 如何使用参考文档

### 查找方式

1. **按模块查找**: 知道要用哪个模块，直接查看对应API
2. **按功能查找**: 知道要实现什么功能，查找相关函数
3. **搜索关键字**: 使用Ctrl+F搜索函数名、参数名

### 阅读建议

- **不需要从头读**: 参考文档供查阅，不是教材
- **关注签名和参数**: 重点看函数签名、参数类型和返回值
- **查看示例**: 如有示例代码，帮助理解用法
- **注意版本**: 确认文档版本与你使用的版本一致

---

## 📖 文档类型对比

| 类型 | 参考手册 (Reference) | 操作指南 (How-to) |
|------|---------------------|-------------------|
| 目标 | 提供技术信息 | 解决具体问题 |
| 结构 | 系统化、分类组织 | 按任务组织 |
| 内容 | 全面、准确、简洁 | 针对性、实用性 |
| 适用场景 | 查阅技术细节 | 遇到具体问题 |

**举例说明**:
- **参考手册**: `calculate_ttm(data, periods=4)` - 函数签名、参数、返回值
- **操作指南**: "如何计算TTM" - 实际使用步骤和注意事项

---

## 📐 API文档规范

### 函数文档格式

```python
def function_name(param1: type, param2: type = default) -> return_type:
    """
    简短的一句话功能描述

    详细说明功能和用途

    Parameters
    ----------
    param1 : type
        参数1的说明
    param2 : type, optional
        参数2的说明 (默认: default)

    Returns
    -------
    return_type
        返回值说明

    Examples
    --------
    >>> result = function_name(value1, value2)
    >>> print(result)
    expected_output

    Notes
    -----
    额外的注意事项

    See Also
    --------
    related_function : 相关功能说明
    """
```

### 类文档格式

```python
class ClassName:
    """
    类的功能描述

    详细说明类的用途和设计

    Attributes
    ----------
    attr1 : type
        属性1说明
    attr2 : type
        属性2说明

    Methods
    -------
    method1(param)
        方法1简短说明
    method2(param)
        方法2简短说明

    Examples
    --------
    >>> obj = ClassName()
    >>> obj.method1(value)
    expected_output
    """
```

---

## 💡 贡献参考文档

欢迎补充和完善参考文档！请遵循以下原则：

1. **准确性**: 信息必须准确无误
2. **完整性**: 覆盖所有公开API
3. **简洁性**: 用最少的文字说清楚
4. **一致性**: 遵循统一的格式规范
5. **可验证**: 代码示例必须可运行

参考文档模板：

```markdown
# 模块名 API参考

**版本**: v4.0.0
**最后更新**: YYYY-MM-DD

## 概述

模块的简短介绍

## 主要类/函数

### 类名/函数名

**签名**: `function(param1, param2)`

**描述**: 功能说明

**参数**:
- `param1` (type): 参数说明
- `param2` (type, optional): 参数说明

**返回**: 返回值说明

**示例**:
```python
# 代码示例
```

**注意事项**:
- 注意点1
- 注意点2

**相关**: [相关函数](#)
```

---

**维护者**: MultiFactors Team
**最后更新**: 2025-12-11
