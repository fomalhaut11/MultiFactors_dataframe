# 统一配置管理

本目录包含多因子系统的所有配置文件，实现了统一的配置管理。

## 📁 配置文件结构

### 核心配置文件

#### main.yaml - 主配置文件
- **功能**: 系统核心配置，包括数据库连接、路径配置、系统参数
- **包含**: 数据库配置、文件路径、数据更新设置、系统参数
- **使用**: 被所有模块引用的基础配置

#### factors.yaml - 因子配置文件  
- **功能**: 批量因子生成配置，定义所有可生成的因子及其参数
- **包含**: 因子分组、计算方法、数据依赖、参数设置
- **使用**: 因子生成脚本、因子管理工具

#### field_mappings.yaml - 字段映射配置
- **功能**: 定义财务数据字段的标准映射关系
- **包含**: 中英文字段名映射、数据表映射、字段类型
- **使用**: 数据处理、因子计算

#### agents.yaml - 智能代理配置
- **功能**: 定义AI助手的专业配置和提示词
- **包含**: 不同领域专家Agent的配置
- **使用**: 智能辅助开发工具

### 扩展配置目录

#### schemas/ - 配置模式定义
- 数据格式验证规则
- 配置文件结构定义
- 类型检查规范

## 🔧 配置管理器

### 统一访问接口
```python
from config.manager import ConfigManager

# 获取配置管理器实例
config = ConfigManager()

# 访问不同配置
db_config = config.get('main.database')
factor_list = config.get('factors.financial.profitability')
field_mapping = config.get('field_mappings.common_fields.revenue')
```

### 配置热更新
```python
# 重新加载配置
config.reload()

# 监听配置文件变化
config.enable_auto_reload()
```

## 📋 迁移说明

### 原配置文件位置 → 新位置
- `config.yaml` → `config/main.yaml`
- `factor_config.yaml` → `config/factors.yaml`  
- `subagent_config.yaml` → `config/agents.yaml`
- `factors/config/field_mapping.yaml` → `config/field_mappings.yaml`

### 代码更新指南
旧的引用方式：
```python
# 旧方式 - 分散的配置加载
from core.config_manager import get_config
config = get_config()
```

新的引用方式：
```python
# 新方式 - 统一配置管理
from config.manager import ConfigManager
config = ConfigManager()
```

## ⚠️ 注意事项

1. **向后兼容**: 原配置文件暂时保留，确保现有代码继续工作
2. **环境变量**: 敏感信息（如数据库密码）仍使用环境变量
3. **配置验证**: 启动时会自动验证配置文件格式
4. **文档同步**: 配置变更时需要同步更新相关文档

## 🚀 最佳实践

1. **配置分离**: 不同功能的配置分别存放
2. **层次化**: 使用YAML的层次结构组织配置
3. **注释完善**: 为每个配置项添加清楚的注释
4. **版本控制**: 配置变更要记录在版本历史中

---
*配置管理统一化完成日期: 2025-08-28*