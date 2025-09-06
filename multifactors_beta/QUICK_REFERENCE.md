# 数据更新系统 - 快速参考

## 🚀 快速开始

```bash
# 健康检查
python scheduled_data_updater.py --data-type all --health-check

# 数据更新
python scheduled_data_updater.py --data-type all --force

# 查看数据状态
python scheduled_data_updater.py --data-summary
```

## 📊 数据类型速查

| 数据类型 | CLI参数 | 文件名 | 更新器类 | 更新频率 |
|---------|---------|-------|----------|----------|
| 价格数据 | `price` | Price.pkl | PriceDataUpdater | daily |
| 财务数据 | `financial` | fzb/lrb/xjlb.pkl | FinancialDataUpdater | daily |
| 涨跌停 | `stop_price` | StopPrice.pkl | StopPriceDataUpdater | daily |
| ST股票 | `st` | ST_stocks.pkl | STDataUpdater | daily |
| 板块调整 | `sector_changes` | SectorChanges_data.pkl | SectorChangesDataUpdater | daily |

## 🔗 依赖关系速查

```
数据库 → 基础文件 → 处理数据
  |         |          |
StockDB → *.pkl → LogReturn_*.pkl
```

**关键依赖**:
- `TradableDF.pkl` ← `Price.pkl` (同步生成)
- `LogReturn_*.pkl` ← `Price.pkl` (处理依赖)
- `StockClassification_*.pkl` ← `SectorChanges_data.pkl` (处理依赖)

## ⚙️ 常用命令

### CLI命令

```bash
# 单类型更新
python scheduled_data_updater.py --data-type price
python scheduled_data_updater.py --data-type financial --force

# 健康检查
python scheduled_data_updater.py --data-type financial --health-check

# 数据管理
python scheduled_data_updater.py --list-data
python scheduled_data_updater.py --data-summary
```

### Python API

```python
from scheduled_data_updater import ScheduledDataUpdater

# 创建更新器
updater = ScheduledDataUpdater(['price', 'financial'])

# 健康检查
health = updater.run_health_check()

# 执行更新
results = updater.run_all_updates(force=True)
```

## 🔍 故障排除速查

### 常见错误

| 错误类型 | 症状 | 解决方案 |
|---------|------|----------|
| 数据库连接失败 | `ConnectionError` | 检查config.yaml数据库配置 |
| 文件权限错误 | `PermissionError` | 确保数据目录写权限 |
| 内存不足 | `MemoryError` | 增加系统内存或使用分块处理 |
| 日期格式错误 | `ValueError: time data` | 检查数据库日期字段格式 |
| MultiIndex错误 | `Index contains duplicate` | 检查数据重复，执行去重操作 |

### 快速诊断

```bash
# 检查数据文件状态
ls -la /path/to/data/*.pkl

# 检查最近日志
tail -50 logs/data_update_$(date +%Y%m%d).log

# 验证数据完整性
python -c "
from core.data_registry import get_data_registry
registry = get_data_registry()
missing = registry.get_missing_datasets()
print('缺失:', missing if missing else '无')
"
```

## ⏰ 更新时间策略

### 自动更新时间
- **工作日**: 16:00-23:59 (收盘后)
- **周末**: 全天 (补充数据)
- **其他时间**: 跳过 (使用--force强制)

### 更新频率
- **daily**: 每日检查更新
- **on_demand**: 按需触发更新
- **monthly**: 按月检查 (如ST数据)

## 🎯 性能指标

### 正常表现
- **价格数据更新**: < 2分钟 (1100万+记录)
- **财务数据更新**: < 3分钟 (43万+记录×3表)
- **收益率计算**: < 5分钟 (1500万+记录×6文件)
- **内存使用**: < 12GB峰值

### 告警阈值
- 数据延迟 > 2天: WARNING
- 数据延迟 > 5天: ERROR
- 更新失败: ERROR
- 内存使用 > 16GB: WARNING

## 📈 监控检查

### 日常检查项

```python
# 数据新鲜度
from core.data_registry import get_data_registry
registry = get_data_registry()
freshness = registry.check_data_freshness(hours_threshold=24)
stale_data = [k for k, v in freshness.items() if not v]
print(f"过时数据: {stale_data}")

# 文件大小检查
import os
data_files = {
    'Price.pkl': 1300,      # MB
    'fzb.pkl': 460,
    'lrb.pkl': 210,
    'xjlb.pkl': 300,
    'StopPrice.pkl': 450
}

for filename, expected_mb in data_files.items():
    if os.path.exists(filename):
        actual_mb = os.path.getsize(filename) / 1024**2
        if actual_mb < expected_mb * 0.8:  # 小于期望值80%
            print(f"⚠️  {filename}: {actual_mb:.1f}MB (期望{expected_mb}MB)")
```

## 🔧 维护脚本

### 定时任务设置 (crontab)

```bash
# 每日16:30更新价格数据
30 16 * * 1-5 cd /path/to/project && python scheduled_data_updater.py --data-type price

# 每日17:00更新财务数据
0 17 * * 1-5 cd /path/to/project && python scheduled_data_updater.py --data-type financial

# 每日健康检查
0 18 * * * cd /path/to/project && python scheduled_data_updater.py --data-type all --health-check
```

### 备份清理脚本

```python
import os
import glob
from datetime import datetime, timedelta

def clean_old_backups(backup_dir, keep_days=7):
    """清理旧备份文件"""
    cutoff_date = datetime.now() - timedelta(days=keep_days)
    
    for backup_file in glob.glob(os.path.join(backup_dir, "*_backup_*.pkl")):
        if os.path.getmtime(backup_file) < cutoff_date.timestamp():
            os.remove(backup_file)
            print(f"已清理旧备份: {backup_file}")

# 使用
clean_old_backups("/path/to/backups", keep_days=7)
```

## 📞 获取帮助

### 内置帮助
```bash
python scheduled_data_updater.py --help
python -c "from core.data_registry import get_data_registry; help(get_data_registry)"
```

### 文档链接
- **完整指南**: [DATA_UPDATE_GUIDE.md](./DATA_UPDATE_GUIDE.md)
- **依赖关系**: [DATA_DEPENDENCY_MAP.md](./DATA_DEPENDENCY_MAP.md)
- **API文档**: [DATA_UPDATER_API.md](./DATA_UPDATER_API.md)

### 技术支持
- **GitHub Issues**: 报告问题和功能请求
- **项目文档**: 查看最新版本文档
- **代码注释**: 直接查看源代码注释

---

**版本**: v2.1 | **更新**: 2025-08-29 | **维护**: 活跃