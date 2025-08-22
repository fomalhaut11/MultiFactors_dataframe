# 数据格式约定和验证指南

## 📋 概述

本文档定义了从数据库到factors模块的标准数据格式约定，以及相应的验证和转换机制。确保整个系统中数据传递的一致性和可靠性。

## 🎯 设计目标

1. **格式统一**: 定义各种数据的标准格式
2. **自动验证**: 提供数据格式自动验证机制
3. **错误提示**: 清晰的错误信息和修复建议
4. **便捷转换**: 简单易用的数据格式转换工具
5. **质量监控**: 数据质量报告和异常检测

## 📊 数据格式规范

### 1. 价格数据格式 (PRICE_DATA)

**用途**: 股票日频价格数据
**来源**: 数据库 `stock_data.dbo.day5` 表

```python
# 必需字段
required_columns = ['code', 'tradingday', 'c', 'adjfactor']

# 可选字段  
optional_columns = ['o', 'h', 'l', 'v', 'amt', 'total_shares', 'free_float_shares', 'exchange_id']

# 数据类型
data_types = {
    'code': 'string',           # 股票代码
    'tradingday': 'int64',      # 交易日期 (20241201格式)
    'o': 'float64',             # 开盘价
    'h': 'float64',             # 最高价
    'l': 'float64',             # 最低价
    'c': 'float64',             # 收盘价
    'v': 'float64',             # 成交量
    'amt': 'float64',           # 成交额
    'adjfactor': 'float64',     # 复权因子
    'total_shares': 'float64',  # 总股本
    'free_float_shares': 'float64',  # 流通股本
    'exchange_id': 'int64'      # 交易所ID
}

# 约束条件
constraints = {
    'tradingday': {'min': 20100101, 'max': 29991231},
    'c': {'min': 0, 'max': np.inf},
    'adjfactor': {'min': 0, 'max': np.inf},
    'v': {'min': 0, 'max': np.inf}
}
```

**示例数据**:
```
    code  tradingday      o      h      l      c        v        amt  adjfactor
0  000001    20241201  10.50  10.80  10.40  10.70  1000000   10700000      1.0
1  000002    20241201  15.20  15.50  15.10  15.35   800000   12280000      1.0
```

### 2. 财务数据格式 (FINANCIAL_DATA)

**用途**: 财务报表数据
**来源**: 数据库 `stock_data.dbo.{fzb,xjlb,lrb}` 表

```python
# 必需字段
required_columns = ['code', 'reportday', 'd_year', 'd_quarter']

# 数据类型
data_types = {
    'code': 'string',                # 股票代码
    'reportday': 'datetime64[ns]',   # 财报发布日期
    'd_year': 'int64',               # 财报年份
    'd_quarter': 'int64'             # 财报季度
    # 其他财务字段动态验证
}

# 约束条件
constraints = {
    'd_year': {'min': 2000, 'max': 2050},
    'd_quarter': {'min': 1, 'max': 4}
}
```

### 3. 发布日期格式 (RELEASE_DATES)

**用途**: 财报发布日期数据
**来源**: 预处理生成的辅助数据

```python
# 必需字段
required_columns = ['StockCodes', 'ReportPeriod', 'ReleasedDates']

# 数据类型
data_types = {
    'StockCodes': 'string',          # 股票代码
    'ReportPeriod': 'datetime64[ns]', # 财报期间
    'ReleasedDates': 'datetime64[ns]' # 发布日期
}
```

### 4. 标准因子格式 (FACTOR_FORMAT)

**用途**: 传递给factors模块的标准格式
**特点**: MultiIndex Series，索引为[TradingDates, StockCodes]

```python
# 索引规范
index_columns = ['TradingDates', 'StockCodes']

# 数据类型
data_types = {'values': 'float64'}

# 格式示例
factor_series = pd.Series(
    data=[10.5, 15.2, 8.9, 12.3],
    index=pd.MultiIndex.from_tuples([
        ('2024-12-01', '000001'),
        ('2024-12-01', '000002'),
        ('2024-12-02', '000001'),
        ('2024-12-02', '000002')
    ], names=['TradingDates', 'StockCodes'])
)
```

## 🔍 数据验证机制

### 1. 基础验证器

```python
from data.schemas import DataValidator, DataSchemas

# 验证价格数据
is_valid, errors = DataValidator.validate_dataframe(
    price_df, DataSchemas.PRICE_DATA, strict=False
)

if not is_valid:
    print("验证失败:")
    for error in errors:
        print(f"  • {error}")
```

### 2. 便捷验证函数

```python
from data.schemas import validate_price_data, validate_financial_data, validate_factor_format

# 价格数据验证
is_valid, errors = validate_price_data(price_df)

# 财务数据验证  
is_valid, errors = validate_financial_data(financial_df)

# 因子格式验证
is_valid, errors = validate_factor_format(factor_series)
```

### 3. 自动修复建议

验证器会提供具体的错误信息和修复建议：

```python
# 示例错误信息
[
    "缺少必需字段: {'adjfactor'}",
    "字段 tradingday 数据类型不匹配: 期望 int64, 实际 object",
    "字段 c 存在小于最小值的数据: -1.5 < 0",
    "发现 5 行重复数据"
]
```

## 🔄 数据转换机制

### 1. 转换为标准因子格式

```python
from data.schemas import DataConverter

# 价格数据转因子格式
factor_series = DataConverter.price_to_factor_format(
    price_df, 
    value_column='c',           # 收盘价
    date_column='tradingday',   # 日期列
    stock_column='code'         # 股票代码列
)

# 财务数据转因子格式
factor_series = DataConverter.financial_to_factor_format(
    financial_df,
    value_column='NET_PROFIT',  # 净利润
    date_column='reportday',    # 发布日期列
    stock_column='code'         # 股票代码列
)
```

### 2. 便捷转换函数

```python
from data.schemas import convert_to_factor_format

# 通用转换函数
factor_series = convert_to_factor_format(
    df, 
    value_col='target_column',
    date_col='date_column', 
    stock_col='code_column'
)
```

## 🌉 数据桥接接口

### 1. DataBridge 核心接口

```python
from data.data_bridge import DataBridge

# 创建数据桥接器
bridge = DataBridge()

# 获取各种数据
financial_data = bridge.get_financial_data()       # 财务数据
price_data = bridge.get_price_data()               # 价格数据
release_dates = bridge.get_release_dates()         # 发布日期
trading_dates = bridge.get_trading_dates()         # 交易日期
stock_info = bridge.get_stock_info()               # 股票信息
```

### 2. 直接获取因子格式数据

```python
# 价格数据转因子格式
close_factor = bridge.price_to_factor(value_column='c')
volume_factor = bridge.price_to_factor(value_column='v')

# 财务数据转因子格式
profit_factor = bridge.financial_to_factor(value_column='NET_PROFIT')
revenue_factor = bridge.financial_to_factor(value_column='REVENUE')
```

### 3. 便捷函数

```python
from data.data_bridge import get_factor_data

# 获取价格因子
close_prices = get_factor_data('price', 'c')

# 获取财务因子
net_profit = get_factor_data('financial', 'NET_PROFIT')
```

## 📈 数据质量监控

### 1. 质量检查器

```python
from data.schemas import DataQualityChecker

# 生成质量报告
report = DataQualityChecker.generate_quality_report(
    data_df, DataSchemas.PRICE_DATA
)

# 打印报告
DataQualityChecker.print_quality_report(report)
```

### 2. 批量质量检查

```python
from data.data_bridge import DataBridge

bridge = DataBridge()

# 获取所有数据的质量报告
reports = bridge.get_data_quality_report(data_type='all')

# 验证整个数据管道
is_valid = bridge.validate_all_data()
```

### 3. 数据状态概览

```python
# 打印数据状态
bridge.print_data_status()

# 输出示例:
"""
📊 数据状态概览
============================================================
✅ 财务数据: (125,000行, 200列)
   更新时间: 2024-12-01 15:30:00
   文件大小: 45.2MB

✅ 发布日期: (8,500项)
   更新时间: 2024-12-01 15:30:00  
   文件大小: 2.1MB

🔄 缓存状态: 3 个数据集已缓存
   financial_data: 5.2分钟前
   price_data_20240101_0: 2.1分钟前
"""
```

## 💡 使用示例

### 1. 基础数据获取和验证

```python
from data.data_bridge import get_data_bridge, validate_data_pipeline

# 获取数据桥接器
bridge = get_data_bridge()

# 验证整个数据管道
if validate_data_pipeline():
    print("✅ 数据管道验证通过，可以开始使用")
else:
    print("❌ 数据管道验证失败，请检查数据")
```

### 2. 获取标准因子数据

```python
from data.data_bridge import get_factor_data

# 获取收盘价因子（自动验证格式）
close_factor = get_factor_data('price', 'c', 
                              begin_date=20240101, 
                              end_date=20241201)

# 获取净利润因子
profit_factor = get_factor_data('financial', 'NET_PROFIT')

# 检查因子格式
print(f"收盘价因子形状: {close_factor.shape}")
print(f"索引名称: {close_factor.index.names}")
print(f"数据类型: {close_factor.dtype}")
```

### 3. 数据质量监控

```python
# 生成数据质量报告
bridge = get_data_bridge()

# 获取质量报告
reports = bridge.get_data_quality_report('all')

# 检查价格数据质量
if 'price' in reports:
    price_report = reports['price']
    issues = price_report['issues']
    
    if issues:
        print(f"⚠️ 价格数据发现 {len(issues)} 个问题:")
        for issue in issues:
            print(f"  • {issue['description']}")
    else:
        print("✅ 价格数据质量良好")
```

### 4. 在factors模块中使用

```python
# 在因子计算中使用标准接口
from data.data_bridge import get_factor_data

class MyFactor(FactorBase):
    def calculate(self, **kwargs):
        # 获取标准格式的价格数据
        close_prices = get_factor_data('price', 'c')
        volume = get_factor_data('price', 'v')
        
        # 计算因子逻辑
        factor_value = close_prices / volume
        
        # 数据已经是标准格式，可以直接返回
        return factor_value
```

### 5. 错误处理

```python
from data.data_bridge import get_data_bridge
from data.schemas import validate_factor_format

try:
    bridge = get_data_bridge()
    
    # 获取财务数据
    financial_data = bridge.get_financial_data(validate=True)
    
    # 转换为因子格式
    factor = bridge.financial_to_factor('NET_PROFIT')
    
    # 验证因子格式
    is_valid, errors = validate_factor_format(factor)
    if not is_valid:
        raise ValueError(f"因子格式验证失败: {errors}")
    
    print("✅ 数据获取和验证成功")
    
except FileNotFoundError as e:
    print(f"❌ 数据文件不存在: {e}")
    
except ValueError as e:
    print(f"❌ 数据格式错误: {e}")
    
except Exception as e:
    print(f"❌ 未知错误: {e}")
```

## 🛠️ 高级功能

### 1. 自定义数据格式

```python
from data.schemas import DataSchema, DataValidator

# 定义自定义数据格式
my_schema = DataSchema(
    name="my_custom_data",
    required_columns=['id', 'value'],
    optional_columns=['description'],
    index_columns=['id'],
    data_types={'id': 'string', 'value': 'float64'},
    constraints={'value': {'min': 0}},
    description="我的自定义数据格式"
)

# 验证自定义格式
is_valid, errors = DataValidator.validate_dataframe(my_df, my_schema)
```

### 2. 缓存管理

```python
bridge = get_data_bridge()

# 检查缓存状态
bridge.print_data_status()

# 清空缓存（释放内存）
bridge.clear_cache()

# 禁用缓存
financial_data = bridge.get_financial_data(use_cache=False)
```

### 3. 数据版本管理

```python
# 获取特定时间范围的数据
price_data = bridge.get_price_data(
    begin_date=20240101,
    end_date=20241201,
    validate=True
)

# 转换为因子格式
factor = bridge.price_to_factor(
    value_column='c',
    begin_date=20240101,
    end_date=20241201
)
```

## ⚠️ 注意事项

1. **数据类型转换**: 日期字段会自动转换为datetime格式
2. **索引排序**: 转换为因子格式时会自动排序索引
3. **缺失值处理**: 验证器会检查缺失值比例，但不会自动处理
4. **内存管理**: 大数据集建议使用缓存或分块处理
5. **并发安全**: DataBridge是线程安全的，可以在多线程环境使用

## 🔧 故障排查

### 常见问题及解决方案

1. **"数据文件不存在"**
   ```bash
   # 重新生成辅助数据
   python data/prepare_auxiliary_data.py
   ```

2. **"字段缺失"**
   ```python
   # 检查数据库表结构
   from core.database import execute_query
   result = execute_query("SELECT TOP 5 * FROM stock_data.dbo.day5")
   print(result.columns)
   ```

3. **"数据类型不匹配"**
   ```python
   # 查看实际数据类型
   print(df.dtypes)
   
   # 手动转换类型
   df['tradingday'] = df['tradingday'].astype('int64')
   ```

4. **"因子格式验证失败"**
   ```python
   # 检查索引格式
   print(f"索引类型: {type(factor.index)}")
   print(f"索引级别: {factor.index.nlevels}")
   print(f"索引名称: {factor.index.names}")
   
   # 重新转换格式
   factor = convert_to_factor_format(df, 'column_name')
   ```

---

**更新时间**: 2025-08-21  
**维护者**: MultiFactors开发团队