# 股票池数据格式规范

## 📋 总体设计原则

1. **多源支持**：支持实时计算、文件导入、预定义池
2. **格式统一**：内部统一为 `List[str]` 格式
3. **自动转换**：系统自动处理各种输入格式
4. **缓存优化**：避免重复计算，提高性能
5. **元数据记录**：记录股票池创建标准和统计信息

## 🔧 数据来源类型

### 1. 实时计算生成（推荐用于研究）

```python
# 流动性前1000只
universe = get_stock_universe('liquid_1000')

# 大盘股前500只
universe = get_stock_universe('large_cap_500') 

# 主板股票（排除ST）
universe = get_stock_universe('main_board')
```

**优势**：
- 数据实时性强
- 标准化筛选逻辑
- 自动适应市场变化

**适用场景**：学术研究、策略开发、因子测试

### 2. 文件导入（推荐用于生产）

#### 2.1 CSV格式
```csv
stock_code,stock_name,weight
000001,平安银行,0.05
000002,万科A,0.03
600000,浦发银行,0.04
```

#### 2.2 JSON格式（推荐）
```json
{
  "name": "my_universe",
  "description": "自定义股票池",
  "created_date": "2025-01-01",
  "stocks": ["000001", "000002", "600000"],
  "metadata": {
    "criteria": "市值大于100亿",
    "rebalance_frequency": "月度"
  }
}
```

#### 2.3 TXT格式（简单）
```
000001
000002  
600000
601318
```

### 3. 预定义股票池

```python
# 沪深300成分股
universe = get_stock_universe('index_300')

# 中证500成分股
universe = get_stock_universe('index_500')

# A股主板
universe = get_stock_universe('a_share_main')
```

## 📊 股票代码格式约定

### 当前系统格式
```python
# 基于实际数据分析，当前系统使用纯数字格式
stocks = ['000001', '000002', '600000', '601318']
```

### 格式标准化处理
```python
def standardize_stock_codes(stocks: List[str]) -> List[str]:
    """
    标准化股票代码格式
    
    输入格式支持：
    - '000001.SZ' → '000001' 
    - '600000.SH' → '600000'
    - '000001' → '000001' (保持不变)
    """
    standardized = []
    for stock in stocks:
        # 移除交易所后缀，保持纯数字格式
        clean_code = str(stock).split('.')[0].strip()
        if clean_code:
            standardized.append(clean_code)
    return standardized
```

## 🎯 API接口规范

### 核心接口
```python
from factors.tester.stock_universe_manager import get_stock_universe

# 1. 基础用法
stocks = get_stock_universe('liquid_1000')

# 2. 带参数
stocks = get_stock_universe('liquid_1000', lookback_days=30)

# 3. 从文件加载
stocks = get_stock_universe('./data/my_universe.json')

# 4. 全市场（默认）
stocks = get_stock_universe('full')  # 或者 None
```

### 集成到单因子测试
```python
from factors.tester.core.pipeline import SingleFactorTestPipeline

pipeline = SingleFactorTestPipeline()

# 方式1：直接传入股票池名称
result = pipeline.run('ROE_ttm', stock_universe='liquid_1000')

# 方式2：传入股票列表
my_stocks = ['000001', '000002', '600000']
result = pipeline.run('ROE_ttm', stock_universe=my_stocks)

# 方式3：从文件加载
result = pipeline.run('ROE_ttm', stock_universe='./data/my_pool.json')
```

## 📁 文件存储规范

### 目录结构
```
cache/stock_universes/
├── liquid_1000.json              # 缓存的计算结果
├── liquid_1000_metadata.json     # 元数据
├── large_cap_500.json            # 大盘股池
├── custom_pool_20250101.json     # 用户自定义池
└── predefined/                   # 预定义股票池
    ├── index_300.json
    └── index_500.json
```

### 元数据格式
```json
{
  "name": "liquid_1000",
  "description": "流动性前1000只股票",
  "stock_count": 1000,
  "created_date": "2025-01-01T10:30:00",
  "last_updated": "2025-01-01T10:30:00", 
  "data_source": "computed",
  "criteria": {
    "method": "volume_rank",
    "lookback_days": 60,
    "min_volume": 1000000
  },
  "performance": {
    "compute_time": 2.3,
    "cache_hit": false
  }
}
```

## 🔄 缓存机制

### 缓存策略
```python
# 1. 内存缓存（运行时）
manager._universe_cache['liquid_1000_hash123'] = stocks

# 2. 文件缓存（持久化）  
cache/stock_universes/liquid_1000.json

# 3. 智能刷新
stocks = get_stock_universe('liquid_1000', refresh=True)  # 强制刷新
```

### 缓存失效条件
- 计算参数变化
- 基础数据更新
- 手动刷新请求
- 缓存文件过期（可配置）

## ⚡ 性能优化

### 计算性能对比
```python
# 性能测试结果（基于5694只股票）
全市场         : 5694只，无过滤开销，计算量最大
流动性前1000只  : 1000只，计算量减少82%，过滤开销<1秒
大盘股前500只   : 500只，计算量减少91%，过滤开销<0.5秒
自定义50只     : 50只，计算量减少99%，过滤开销<0.1秒
```

### 推荐配置
```python
# 开发阶段：使用小股票池快速测试
DEV_UNIVERSE = 'liquid_100'  # 100只流动性好的股票

# 研究阶段：使用中等规模股票池
RESEARCH_UNIVERSE = 'liquid_1000'  # 1000只股票，平衡计算效率和覆盖面

# 生产阶段：根据策略容量选择
PRODUCTION_UNIVERSE = 'large_cap_500'  # 500只大盘股，适合大资金
```

## 🛡️ 异常处理

### 错误处理机制
```python
def get_stock_universe(name: str) -> List[str]:
    try:
        # 尝试加载股票池
        stocks = _load_universe(name)
        
        # 验证股票池
        if not stocks:
            logger.warning(f"股票池 '{name}' 为空，使用全市场")
            return _get_full_market_universe()
        
        return stocks
        
    except FileNotFoundError:
        logger.error(f"股票池文件不存在: {name}")
        return _get_full_market_universe()  # 降级到全市场
        
    except Exception as e:
        logger.error(f"股票池加载失败: {e}")
        return _get_full_market_universe()  # 降级到全市场
```

### 数据验证
```python
def validate_universe(stocks: List[str]) -> List[str]:
    """验证和清洗股票池数据"""
    valid_stocks = []
    
    for stock in stocks:
        # 格式验证
        if not isinstance(stock, str):
            continue
            
        # 代码格式验证（纯数字，6位）
        clean_code = stock.strip()
        if len(clean_code) == 6 and clean_code.isdigit():
            valid_stocks.append(clean_code)
        else:
            logger.warning(f"股票代码格式异常: {stock}")
    
    return valid_stocks
```

## 📚 使用示例

### 示例1: 研究流程
```python
# 1. 快速原型开发
result = pipeline.run('new_factor', stock_universe='liquid_100')

# 2. 详细研究验证
result = pipeline.run('new_factor', stock_universe='liquid_1000')

# 3. 最终策略测试
result = pipeline.run('new_factor', stock_universe='large_cap_500')
```

### 示例2: 自定义股票池
```python
# 创建行业股票池
finance_stocks = ['000001', '600000', '600036', '601318']

# 保存到文件
manager.save_universe(
    'finance_sector',
    finance_stocks,
    description='金融行业股票池',
    criteria={'sector': 'finance', 'created_by': 'researcher'}
)

# 使用自定义股票池
result = pipeline.run('sector_factor', stock_universe=finance_stocks)
```

### 示例3: 批量测试不同股票池
```python
universes = {
    'small': 'liquid_100',
    'medium': 'liquid_500', 
    'large': 'liquid_1000',
    'full': None
}

results = {}
for name, universe in universes.items():
    results[name] = pipeline.run('test_factor', stock_universe=universe)
    print(f"{name}: IC={results[name].ic_result.ic_mean:.4f}")
```

## 🎯 最佳实践建议

1. **开发阶段**：使用小股票池（50-100只）快速迭代
2. **研究阶段**：使用中等股票池（500-1000只）深入分析  
3. **生产阶段**：根据策略容量选择合适规模
4. **性能监控**：记录股票池大小对计算时间的影响
5. **版本管理**：为重要的股票池保存历史版本
6. **文档记录**：记录股票池的选择标准和业务逻辑

---

**总结**：通过这套规范，我们实现了股票池功能的标准化、高性能和易用性，为量化研究提供了灵活而强大的工具支持。