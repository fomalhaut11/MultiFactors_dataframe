# Data模块使用指南

## 📊 模块概述

Data模块负责多因子量化系统的数据获取、处理和存储，采用简单实用的批处理架构，通过文件系统实现与其他模块的解耦。

**设计理念**: 
- 🎯 **简单直接**: 脚本化处理，易于维护和调试
- 📁 **文件驱动**: 使用pkl文件作为数据交换媒介
- 🔄 **批处理友好**: 适合T+1量化研究场景
- 🛠️ **工具化**: 每个脚本专注一个功能

## 📁 目录结构

```
data/
├── README.md                    # 本文档
│
├── 📋 主程序脚本
├── prepare_auxiliary_data.py    # 辅助数据准备（初始化必备）
│
├── 📊 数据获取模块 (fetcher/)
├── fetcher/
│   ├── __init__.py
│   ├── data_fetcher.py          # 数据获取基类和股票数据获取器
│   ├── BasicDataLocalization.py # 基础数据本地化
│   ├── chunked_price_fetcher.py # 分块价格数据获取
│   └── incremental_price_updater.py # 增量价格更新器
│
├── 🔧 数据处理模块 (processor/)
├── processor/
│   ├── __init__.py
│   ├── base_processor.py        # 处理器基类
│   ├── data_processing_pipeline.py # 数据处理管道
│   ├── enhanced_pipeline.py     # 增强处理管道
│   ├── price_processor.py       # 价格数据处理器
│   ├── return_calculator.py     # 收益率计算器
│   ├── financial_processor.py   # 财务数据处理器
│   ├── optimized_return_calculator.py # 优化收益率计算
│   ├── parallel_optimizer.py    # 并行处理优化器
│   └── example_custom_processor.py # 自定义处理器示例
│
├── 💾 数据存储目录 (auxiliary/)
├── auxiliary/                   # 预处理数据存储（生成的pkl文件）
│   ├── FinancialData_unified.pkl    # 合并的财务数据
│   ├── ReleaseDates.pkl             # 财报发布日期
│   ├── StockInfo.pkl                # 股票基本信息
│   ├── TradingDates.pkl             # 交易日期列表
│   ├── data_preparation_summary.json    # 数据准备摘要
│   └── data_preparation_summary_v2.json # 数据准备摘要v2
│
├── 🗃️ 存储接口 (storage/)
│   └── (空目录，保留扩展)
│
├── 📋 数据格式约定
├── schemas.py                   # 数据格式规范和验证器
├── data_bridge.py               # data模块与factors模块的桥接接口
├── DATA_FORMATS.md              # 数据格式约定详细文档
│
└── 📚 使用示例 (examples/)
    └── data_format_examples.py     # 数据格式使用示例
```

## 📋 数据文件说明

| 文件名 | 用途 | 更新频率 | 依赖脚本 |
|--------|------|----------|----------|
| `FinancialData_unified.pkl` | 合并的财务数据（利润表、资产负债表、现金流量表） | 季度 | prepare_auxiliary_data.py |
| `ReleaseDates.pkl` | 财报发布日期数据 | 季度 | prepare_auxiliary_data.py |
| `StockInfo.pkl` | 股票基本信息 | 手动 | prepare_auxiliary_data.py |
| `TradingDates.pkl` | 交易日期列表 | 手动 | prepare_auxiliary_data.py |
| `Price.pkl` | 股票价格数据 | 日 | ../scheduled_data_updater.py |
| `LogReturn_*.pkl` | 各类收益率数据 | 按需 | 数据处理管道 |

## 🚀 快速开始

### 1. 初始化数据环境

**首次使用必须执行以下步骤：**

```bash
# 1. 进入项目根目录
cd mulitfactors_beta/

# 2. 准备辅助数据（必备步骤）
python data/prepare_auxiliary_data.py

# 3. 获取初始价格数据
python scheduled_data_updater.py --data-type price

# 4. 验证数据准备完成
python -c "
import pandas as pd
from pathlib import Path

files = ['FinancialData_unified.pkl', 'ReleaseDates.pkl', 'StockInfo.pkl', 'TradingDates.pkl']
for f in files:
    path = Path(f'data/auxiliary/{f}')
    if path.exists():
        print(f'✓ {f} - 大小: {path.stat().st_size/1024/1024:.1f}MB')
    else:
        print(f'✗ {f} - 缺失')
"
```

### 2. 日常数据更新

```bash
# 更新价格数据（日常）
python scheduled_data_updater.py --data-type price

# 交互式更新（支持多种数据类型）
python interactive_data_updater.py

# 增量更新价格数据
python data/fetcher/incremental_price_updater.py
```

## 💻 使用范例

### 1. 读取基础数据

```python
import pandas as pd
from pathlib import Path

# 读取财务数据
financial_data = pd.read_pickle('data/auxiliary/FinancialData_unified.pkl')
print(f"财务数据形状: {financial_data.shape}")
print(f"包含字段: {list(financial_data.columns)}")

# 读取发布日期
release_dates = pd.read_pickle('data/auxiliary/ReleaseDates.pkl')
print(f"发布日期数据: {release_dates.shape}")

# 读取股票信息
stock_info = pd.read_pickle('data/auxiliary/StockInfo.pkl')
print(f"股票信息: {stock_info.shape}")

# 读取交易日期
trading_dates = pd.read_pickle('data/auxiliary/TradingDates.pkl')
print(f"交易日期: {len(trading_dates)} 个交易日")
```

### 2. 使用数据获取器

```python
from data.fetcher.data_fetcher import StockDataFetcher

# 创建数据获取器
fetcher = StockDataFetcher()

# 获取价格数据
price_data = fetcher.fetch_data('price', begin_date=20240101, end_date=20241231)
print(f"价格数据: {price_data.shape}")

# 获取财务数据
financial_data = fetcher.fetch_data('financial', sheet_type='fzb')  # 资产负债表
print(f"资产负债表: {financial_data.shape}")
```

### 3. 使用数据处理器

```python
from data.processor.data_processing_pipeline import DataProcessingPipeline

# 创建处理管道
pipeline = DataProcessingPipeline()

# 运行完整处理流程
results = pipeline.run_full_pipeline(save_intermediate=True)

# 获取处理结果
price_df = results['price_df']
stock_3d = results['stock_3d']
log_return_daily = results['log_return_daily_o2o']

print(f"价格数据: {price_df.shape}")
print(f"三维股票数据: {stock_3d.shape}")
print(f"日收益率: {log_return_daily.shape}")
```

### 4. 自定义数据处理

```python
from data.processor.base_processor import BaseDataProcessor
import pandas as pd

class CustomDataProcessor(BaseDataProcessor):
    """自定义数据处理器示例"""
    
    def validate_input(self, **kwargs):
        # 自定义验证逻辑
        return True
    
    def process(self, data: pd.DataFrame, **kwargs):
        """自定义处理逻辑"""
        self.logger.info("开始自定义处理...")
        
        # 示例：数据清洗
        cleaned_data = data.dropna()
        
        # 示例：数据转换
        processed_data = self.standardize_data(cleaned_data)
        
        # 记录处理历史
        self._record_processing(
            operation="custom_process",
            params=kwargs,
            result_info={"input_shape": data.shape, "output_shape": processed_data.shape}
        )
        
        return processed_data
    
    def standardize_data(self, data):
        """标准化数据"""
        numeric_cols = data.select_dtypes(include=[float, int]).columns
        data[numeric_cols] = (data[numeric_cols] - data[numeric_cols].mean()) / data[numeric_cols].std()
        return data

# 使用自定义处理器
processor = CustomDataProcessor()
result = processor.process(your_data)
```

### 5. 增量数据更新

```python
from data.fetcher.incremental_price_updater import IncrementalPriceUpdater

# 创建增量更新器
updater = IncrementalPriceUpdater()

# 检查需要更新的数据
update_info = updater.check_update_requirements()
print(f"需要更新: {update_info}")

# 执行增量更新
if update_info['needs_update']:
    updater.update()
    print("增量更新完成")
```

## 🔍 数据质量检查

### 检查数据完整性

```python
import pandas as pd
from datetime import datetime
from pathlib import Path

def check_data_status():
    """检查所有数据文件状态"""
    
    files = {
        'FinancialData_unified.pkl': '财务数据',
        'ReleaseDates.pkl': '发布日期',
        'StockInfo.pkl': '股票信息',
        'TradingDates.pkl': '交易日期'
    }
    
    print("📊 数据状态检查")
    print("=" * 50)
    
    for file, name in files.items():
        path = Path(f'data/auxiliary/{file}')
        if path.exists():
            mtime = datetime.fromtimestamp(path.stat().st_mtime)
            size = path.stat().st_size / 1024 / 1024  # MB
            
            # 读取数据检查形状
            try:
                data = pd.read_pickle(path)
                if isinstance(data, pd.DataFrame):
                    shape_info = f"({data.shape[0]}行, {data.shape[1]}列)"
                else:
                    shape_info = f"({len(data)}项)" if hasattr(data, '__len__') else "未知格式"
                    
                print(f"✓ {name}: {shape_info}")
                print(f"  更新时间: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"  文件大小: {size:.1f}MB")
                
            except Exception as e:
                print(f"⚠️ {name}: 文件损坏 - {e}")
        else:
            print(f"✗ {name}: 文件不存在")
        print()

# 运行检查
check_data_status()
```

### 检查数据质量

```python
def check_data_quality(data_name='FinancialData_unified.pkl'):
    """检查数据质量"""
    
    data = pd.read_pickle(f'data/auxiliary/{data_name}')
    
    print(f"📈 {data_name} 质量报告")
    print("=" * 50)
    
    # 基本信息
    print(f"数据形状: {data.shape}")
    print(f"数据类型: {data.dtypes.value_counts().to_dict()}")
    
    # 缺失值检查
    missing_ratio = data.isnull().sum() / len(data)
    high_missing = missing_ratio[missing_ratio > 0.1]
    
    if not high_missing.empty:
        print(f"\n⚠️ 高缺失值字段 (>10%):")
        for col, ratio in high_missing.head(10).items():
            print(f"  {col}: {ratio:.1%}")
    else:
        print("\n✓ 缺失值检查通过")
    
    # 重复值检查
    if hasattr(data, 'duplicated'):
        dup_count = data.duplicated().sum()
        print(f"\n重复行数: {dup_count} ({dup_count/len(data):.1%})")
    
    # 数值范围检查
    numeric_cols = data.select_dtypes(include=[float, int]).columns
    if len(numeric_cols) > 0:
        print(f"\n数值字段统计 (前5个):")
        print(data[numeric_cols[:5]].describe())

check_data_quality()
```

## 🛠️ 维护指南

### 日常维护任务

```bash
# 每日任务
python scheduled_data_updater.py --data-type price

# 每季度任务
python data/prepare_auxiliary_data.py  # 更新财务数据

# 每月任务 - 清理日志
find data/ -name "*.log" -mtime +30 -delete

# 检查磁盘空间
du -sh data/auxiliary/
```

### 故障排查

1. **数据文件缺失**
   ```bash
   # 重新生成辅助数据
   python data/prepare_auxiliary_data.py
   ```

2. **数据更新失败**
   ```bash
   # 检查数据库连接
   python -c "from core.database import test_connection; test_connection()"
   
   # 手动更新
   python interactive_data_updater.py
   ```

3. **内存不足**
   ```python
   # 使用分块处理
   from data.fetcher.chunked_price_fetcher import ChunkedPriceFetcher
   
   fetcher = ChunkedPriceFetcher(chunk_size=1000)
   data = fetcher.fetch_all()
   ```

### 性能优化建议

1. **启用数据压缩**
   ```python
   # 保存时使用压缩
   pd.to_pickle(data, 'data.pkl.gz', compression='gzip')
   
   # 读取压缩文件
   data = pd.read_pickle('data.pkl.gz')
   ```

2. **使用缓存机制**
   ```python
   # 数据获取器会自动使用缓存
   fetcher = StockDataFetcher()
   data = fetcher.fetch_data('price', cache_hours=24)
   ```

3. **并行处理**
   ```python
   from data.processor.parallel_optimizer import ParallelOptimizer
   
   optimizer = ParallelOptimizer(n_workers=4)
   result = optimizer.process_parallel(data_list)
   ```

## ❓ 常见问题

**Q: 初次运行prepare_auxiliary_data.py很慢怎么办？**
A: 这是正常的，首次需要处理大量财务数据。可以先处理部分数据测试：
```python
# 在prepare_auxiliary_data.py中设置测试模式
TEST_MODE = True  # 只处理部分数据
```

**Q: 如何添加新的数据源？**
A: 继承BaseDataFetcher类：
```python
from data.fetcher.data_fetcher import BaseDataFetcher

class YourDataFetcher(BaseDataFetcher):
    def fetch_data(self, **kwargs):
        # 实现你的数据获取逻辑
        pass
```

**Q: 数据文件太大怎么办？**
A: 可以使用压缩或分片存储：
```python
# 压缩存储
pd.to_pickle(data, 'data.pkl.gz', compression='gzip')

# 分片存储
for i, chunk in enumerate(np.array_split(data, 10)):
    pd.to_pickle(chunk, f'data_chunk_{i}.pkl')
```

**Q: 如何备份数据？**
A: 定期备份auxiliary目录：
```bash
# 创建备份
tar -czf data_backup_$(date +%Y%m%d).tar.gz data/auxiliary/

# 恢复备份
tar -xzf data_backup_20250101.tar.gz
```

## 🔗 数据格式约定

### 数据传递标准

Data模块与factors模块之间采用标准化的数据格式，确保数据传递的一致性：

```python
# 获取标准格式因子数据
from data.data_bridge import get_factor_data

# 价格因子
close_factor = get_factor_data('price', 'c')  # 收盘价
volume_factor = get_factor_data('price', 'v') # 成交量

# 财务因子  
profit_factor = get_factor_data('financial', 'NET_PROFIT')  # 净利润
revenue_factor = get_factor_data('financial', 'REVENUE')    # 营业收入
```

### 标准因子格式

所有传递给factors模块的数据都使用MultiIndex Series格式：

```python
# 标准格式: MultiIndex[TradingDates, StockCodes]
factor_series.index.names = ['TradingDates', 'StockCodes']

# 示例结构:
TradingDates  StockCodes
2024-12-01    000001        10.7
              000002        15.35
2024-12-02    000001        10.8
              000002        15.4
```

### 数据验证

系统提供自动数据验证机制：

```python
from data.data_bridge import validate_data_pipeline

# 验证整个数据管道
if validate_data_pipeline():
    print("✅ 数据管道验证通过")
else:
    print("❌ 数据管道存在问题")

# 查看数据状态
from data.data_bridge import get_data_bridge
bridge = get_data_bridge()
bridge.print_data_status()
```

### 详细文档

- 📋 **完整格式规范**: 查看 `DATA_FORMATS.md`
- 🔧 **验证机制**: 参考 `schemas.py`
- 🌉 **桥接接口**: 使用 `data_bridge.py`
- 💻 **使用示例**: 运行 `examples/data_format_examples.py`

## 📞 技术支持

如有问题，请检查：
1. 日志文件：`core/logs/`
2. 配置文件：`config.yaml`
3. 数据库连接：运行`python -c "from core.database import test_connection; test_connection()"`
4. 数据格式：运行 `python data/examples/data_format_examples.py`

---

**更新时间**: 2025-08-21  
**维护者**: MultiFactors开发团队