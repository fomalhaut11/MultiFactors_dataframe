# BP因子使用指南

## 1. 快速开始

### 1.1 数据准备

```python
import pandas as pd

# 加载原始数据
fzb = pd.read_pickle('fzb.pkl')  # 负债表
market_cap = pd.read_pickle('MarketCap.pkl')  # 市值

# 转换财务数据格式
fzb['reportday'] = pd.to_datetime(fzb['reportday'])
fzb['tradingday'] = pd.to_datetime(fzb['tradingday'])

# 创建发布日期数据
release_dates = fzb[['reportday', 'code', 'tradingday']].copy()
release_dates = release_dates.rename(columns={'tradingday': 'ReleasedDates'})
release_dates = release_dates.set_index(['reportday', 'code'])
release_dates.index.names = ['ReportDates', 'StockCodes']

# 设置财务数据索引
fzb = fzb.set_index(['reportday', 'code'])
fzb.index.names = ['ReportDates', 'StockCodes']
```

### 1.2 计算BP因子

```python
from factors.utils import FactorCalculator

# 创建计算器
calculator = FactorCalculator()

# 获取交易日期
trading_dates = market_cap.index.get_level_values('TradingDates').unique()

# 计算BP因子
results = calculator.calculate_factors(
    factor_names=['BP'],
    financial_data=fzb,
    market_cap=market_cap,
    release_dates=release_dates,
    trading_dates=trading_dates
)

# 获取结果
bp_factor = results['BP']
```

## 2. 因子更新

### 2.1 全量更新（首次运行）

```python
from factors.utils import FactorUpdater

# 创建更新器
updater = FactorUpdater(
    data_path='./data',      # 数据目录
    factor_path='./factors'  # 因子保存目录
)

# 执行全量更新
results = updater.update_fundamental_factors(
    factor_names=['BP'],
    mode='full',
    financial_data=fzb,
    market_cap=market_cap,
    release_dates=release_dates,
    trading_dates=trading_dates
)

print(f"BP因子计算完成，共 {len(results['BP'])} 条记录")
```

### 2.2 增量更新（日常更新）

```python
# 加载最新数据
latest_financial = pd.read_pickle('latest_financial.pkl')
latest_market_cap = pd.read_pickle('latest_market_cap.pkl')

# 检查是否有新数据
has_updates, new_data = updater.check_financial_updates(latest_financial)

if has_updates:
    print(f"发现 {len(new_data)} 条新财报数据")
    
    # 执行增量更新
    results = updater.update_fundamental_factors(
        factor_names=['BP'],
        mode='incremental',
        financial_data=latest_financial,
        market_cap=latest_market_cap,
        release_dates=latest_release_dates,
        trading_dates=latest_trading_dates
    )
    
    print("增量更新完成!")
else:
    print("没有新数据需要更新")
```

## 3. 完整的更新脚本

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BP因子每日更新脚本
"""
import pandas as pd
from pathlib import Path
from datetime import datetime
from factors.utils import FactorUpdater

def daily_update():
    """执行每日更新"""
    print(f"开始更新 - {datetime.now()}")
    
    # 配置路径
    data_path = Path('./data')
    factor_path = Path('./factors')
    
    try:
        # 1. 创建更新器
        updater = FactorUpdater(data_path, factor_path)
        
        # 2. 加载数据
        fzb = pd.read_pickle(data_path / 'fzb.pkl')
        market_cap = pd.read_pickle(data_path / 'MarketCap.pkl')
        
        # 3. 数据预处理（同上）
        # ... 转换为MultiIndex格式 ...
        
        # 4. 执行更新
        updater.update_all_factors(
            mode='incremental',
            financial_data=fzb,
            market_cap=market_cap,
            release_dates=release_dates,
            trading_dates=trading_dates
        )
        
        print("所有因子更新成功!")
        
    except Exception as e:
        print(f"更新失败: {e}")
        # 可以发送告警邮件等
    
    print(f"更新完成 - {datetime.now()}")

if __name__ == "__main__":
    daily_update()
```

## 4. 关键概念

### 4.1 数据格式要求

- **财务数据**：MultiIndex (ReportDates, StockCodes)
- **市值数据**：Series with MultiIndex (TradingDates, StockCodes)
- **发布日期**：DataFrame with ReleasedDates column

### 4.2 更新模式

- **全量更新 (full)**：重新计算所有数据
- **增量更新 (incremental)**：只处理新数据

### 4.3 更新追踪

- 状态保存在 `factor_update_tracker.json`
- 记录最后处理的发布日期
- 支持断点续传

## 5. 注意事项

1. **首次运行必须使用全量更新**
2. **财务数据必须包含发布日期信息**
3. **增量更新会自动识别需要重算的股票**
4. **定期检查更新日志确保正常运行**

## 6. 解决编码问题

如果遇到编码问题，可以：

### 方法1：设置环境变量
```batch
set PYTHONIOENCODING=utf-8
python your_script.py
```

### 方法2：使用安全打印
```python
# 替换print为安全的字符串处理
def safe_print(text):
    # 移除特殊字符
    text = str(text).encode('ascii', 'ignore').decode('ascii')
    print(text)
```

### 方法3：在代码中避免使用emoji
```python
# 不要使用
print("[OK] 成功")

# 使用
print("[OK] 成功")
```

## 7. 常见问题

**Q: 数据格式不匹配？**
A: 确保财务数据转换为MultiIndex格式

**Q: 增量更新没有找到新数据？**
A: 检查ReleasedDates是否正确，是否有新的发布日期

**Q: 计算结果为空？**
A: 检查数据对齐，确保财务数据和市值数据有重叠的股票和日期

---

更多信息请参考：
- [因子计算模块迁移指南](./因子计算模块迁移指南.md)
- [因子更新模块使用指南](./因子更新模块使用指南.md)