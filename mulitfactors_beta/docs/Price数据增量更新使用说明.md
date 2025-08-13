# Price.pkl 增量更新使用说明

## 概述

增量更新系统可以自动检测Price.pkl文件中的最新日期，从数据库获取新的交易数据，并合并到现有文件中，避免重复下载历史数据。

## 核心文件

- `data/fetcher/incremental_price_updater.py` - 增量更新核心模块
- `update_price_data.py` - 交互式更新工具 
- `scheduled_price_update.py` - 定时更新脚本

## 使用方法

### 1. 手动更新（推荐新手使用）

```bash
python update_price_data.py
```

提供菜单界面：
- **查看数据状态** - 检查当前数据情况
- **执行增量更新** - 自动检测并更新新数据
- **强制更新最近数据** - 修复最近几天的数据
- **清理备份文件** - 管理备份文件

### 2. 命令行直接更新

```bash
# 基本更新
python scheduled_price_update.py

# 强制更新（忽略时间限制）
python scheduled_price_update.py --force

# 健康检查
python scheduled_price_update.py --health-check

# 静默模式
python scheduled_price_update.py --quiet
```

### 3. 程序化调用

```python
from data.fetcher.incremental_price_updater import IncrementalPriceUpdater

# 创建更新器
updater = IncrementalPriceUpdater()

# 检查是否需要更新
info = updater.get_update_info()
print(f"需要更新: {info['need_update']}")

# 执行更新
success = updater.update_price_file()
```

## 主要功能

### 智能检测

- 自动读取Price.pkl文件最新日期
- 查询数据库最新可用日期
- 智能判断是否需要更新

### 安全更新

- **自动备份** - 更新前自动备份原文件
- **失败回滚** - 更新失败时自动恢复备份
- **数据验证** - 确保数据格式和完整性

### 增量合并

- 只获取新增的交易日数据
- 自动去重和排序
- 保持数据索引结构一致

## 自动化设置

### Windows 任务计划程序

1. 打开"任务计划程序"
2. 创建基本任务
3. 设置触发器：每日下午5点
4. 设置操作：
   ```
   程序: python
   参数: E:\Documents\PythonProject\StockProject\MultiFactors\mulitfactors_beta\scheduled_price_update.py
   起始于: E:\Documents\PythonProject\StockProject\MultiFactors\mulitfactors_beta
   ```

### 批处理脚本

创建 `daily_update.bat`：
```batch
@echo off
cd /d "E:\Documents\PythonProject\StockProject\MultiFactors\mulitfactors_beta"
python scheduled_price_update.py
pause
```

## 工作原理

### 更新流程

1. **检测阶段**
   - 读取Price.pkl最新日期
   - 查询数据库最新日期
   - 比较确定是否需要更新

2. **备份阶段**
   - 创建原文件备份
   - 保存到backups目录

3. **获取阶段**
   - 使用分块获取器获取增量数据
   - 按30天小块获取，避免超时

4. **合并阶段**
   - 合并新旧数据
   - 去重和排序
   - 计算增强字段

5. **保存阶段**
   - 保存更新后的文件
   - 验证数据完整性

### 时间策略

- **更新时间窗口**: 工作日 16:00-23:59
- **周末补充**: 周末可更新遗漏数据
- **节假日**: 自动跳过无交易日

## 错误处理

### 常见问题

1. **数据库连接失败**
   - 检查网络连接
   - 验证数据库配置

2. **文件权限问题**
   - 确保对数据目录有读写权限
   - 检查文件是否被其他程序占用

3. **内存不足**
   - 关闭其他占用内存的程序
   - 系统会自动使用分块获取

### 数据修复

如果数据出现问题：

1. **使用强制更新**
   ```bash
   python update_price_data.py
   # 选择选项3，强制更新最近几天
   ```

2. **从备份恢复**
   ```bash
   # 手动从backups目录恢复文件
   copy "backups\Price_backup_20250729_180000.pkl" "Price.pkl"
   ```

## 监控和维护

### 日志文件

- 位置: `data_root/logs/price_update_YYYYMMDD.log`
- 包含详细的更新过程和错误信息

### 健康检查

```bash
python scheduled_price_update.py --health-check
```

返回状态码：
- 0: 健康
- 1: 警告（数据稍有延迟）
- 2: 错误（数据严重过期或文件缺失）

### 备份管理

- 自动保留最近7天的备份
- 定期清理过期备份
- 手动清理：选择菜单选项4

## 性能优化

### 建议设置

- **分块大小**: 30天（已优化）
- **备份保留**: 3-7天
- **更新频率**: 每日一次
- **健康检查**: 每小时一次

### 资源使用

- **内存**: 通常<500MB
- **磁盘**: 需要2倍数据文件大小的空闲空间
- **网络**: 取决于新增数据量，通常<50MB

## 故障排除

### 调试模式

```python
import logging
logging.basicConfig(level=logging.DEBUG)

from data.fetcher.incremental_price_updater import IncrementalPriceUpdater
updater = IncrementalPriceUpdater()
updater.update_price_file()
```

### 常用命令

```bash
# 查看当前状态
python -c "from data.fetcher.incremental_price_updater import IncrementalPriceUpdater; print(IncrementalPriceUpdater().get_update_info())"

# 测试数据库连接
python -c "from core.database import test_connection; print(test_connection())"

# 检查文件信息
python -c "import pandas as pd; df=pd.read_pickle('E:/Documents/PythonProject/StockProject/StockData/Price.pkl'); print(f'Shape: {df.shape}, Latest: {df.index.get_level_values(0).max()}')"
```

## 总结

增量更新系统让Price.pkl文件的维护变得简单高效：

- **自动化** - 设置一次，每日自动更新  
- **安全** - 自动备份，失败回滚  
- **高效** - 只下载新数据，节省时间  
- **智能** - 自动检测，按需更新  
- **可靠** - 完善的错误处理和恢复机制  

建议每天在股市收盘后（下午5点后）运行更新，确保数据及时性。