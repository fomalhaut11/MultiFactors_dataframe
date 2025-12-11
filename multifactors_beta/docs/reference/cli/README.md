# CLI 命令行工具参考

**最后更新**: 2025-12-11
**状态**: 计划中

---

## 📋 计划中的CLI文档

本目录将包含命令行工具的使用说明：

### 数据更新CLI
- **scheduled_data_updater.py** (计划中) - 定时数据更新工具
  - 命令参数说明
  - 使用示例
  - 配置选项

### 因子管理CLI
- **factor_manager.py** (计划中) - 因子管理工具
  - 因子生成命令
  - 因子验证命令
  - 批量操作命令

### 批量生成CLI
- **quick_generate_factors.py** (计划中) - 快速生成工具
- **batch_generate_factors.py** (计划中) - 批量生成工具
- **validate_factors.py** (计划中) - 因子验证工具

---

## 📖 CLI 文档格式

```markdown
# 命令：tool_name.py

**功能**: 命令功能说明
**适用场景**: 使用场景描述

## 语法
```bash
python tool_name.py [options] [arguments]
```

## 参数

### 必需参数
- `--param1`: 参数说明

### 可选参数
- `--option1`: 选项说明（默认值：xxx）
- `--option2`: 选项说明

## 使用示例

### 示例1：基本用法
```bash
python tool_name.py --param1 value1
```

### 示例2：高级用法
```bash
python tool_name.py --param1 value1 --option1 value2
```

## 返回值
- 0: 成功
- 1: 错误

## 相关命令
- [相关命令链接]
```

---

**维护者**: MultiFactors Team
**最后更新**: 2025-12-11
