# Legacy（遗留代码）目录

此目录包含项目的所有旧版本文件，保留用于向后兼容和历史参考。

## 文件说明

### 价格数据更新工具
- `scheduled_price_update.py` - 旧版定时价格更新脚本（完整实现）
- `scheduled_price_update_legacy.py` - 向后兼容包装器（调用新系统）
- `update_price_data.py` - 旧版交互式价格更新工具

### 数据获取工具
- `create_price_pkl.py` - 旧版价格数据创建工具  
- `get_full_price_data.py` - 旧版完整数据获取工具

### 文档
- `README_root_legacy.md` - 原根目录legacy文件夹的README

## 迁移指南

### 推荐使用新版本：
- `scheduled_price_update.py` → `scheduled_data_updater.py`
- `update_price_data.py` → `interactive_data_updater.py`
- `create_price_pkl.py` → 使用新的数据获取模块
- `get_full_price_data.py` → 使用新的数据获取模块

### 使用示例：
```bash
# 旧方式（仍可用但不推荐）
python archive/legacy/scheduled_price_update.py

# 新方式（推荐）  
python scheduled_data_updater.py --data-type price
```

## 兼容性说明

1. **scheduled_price_update_legacy.py** - 提供完全向后兼容的接口，内部调用新系统
2. **其他文件** - 保留原始实现，但建议尽快迁移到新架构

## 清理计划

这些文件将在以下条件满足后逐步移除：
1. 新系统稳定运行至少3个月
2. 所有依赖旧接口的代码完成迁移
3. 充分的文档和迁移指南已完成

## 注意事项

⚠️ 这些代码已不再维护，可能存在以下问题：
- 使用旧的数据库连接方式
- 缺少错误处理和日志记录
- 性能未优化
- 可能与新的数据格式不兼容

建议尽快迁移到新系统以获得更好的稳定性和性能。