# 数据预处理功能完整指南

## 功能对比表

| 原始函数 | 新实现位置 | 功能说明 |
|---------|-----------|---------|
| `get_realeaddates_formH5()` | `FinancialDataProcessor.get_released_dates_from_h5()` | 从H5文件获取财报发布日期 |
| `StockDataDF2Matrix()` | `PriceDataProcessor._stock_data_df_to_matrix()` | 将DataFrame转换为3D矩阵 |
| `get_price_data()` | `PriceDataProcessor.process()` | 获取和处理价格数据 |
| `date_serries()` | `PriceDataProcessor.get_date_series()` | 生成日/周/月日期序列 |
| `logreturndf_dateserries()` | `ReturnCalculator.calculate_log_return()` | 计算对数收益率 |
| `released_dates_count()` | `FinancialDataProcessor.calculate_released_dates_count()` | 计算财报发布时间差 |
| `logreturn_after_release()` | `ReturnCalculator.calculate_return_after_release()` | 计算财报发布后收益率 |
| `logreturn_ndays_lag()` | `ReturnCalculator.calculate_n_days_return()` | 计算N天滚动收益率 |
| `run()` | `DataProcessingPipeline.run_full_pipeline()` | 完整处理流程 |

## 完整运行预处理的方法

### 方法1：使用基础管道（与原始实现完全兼容）

```python
from data.processor import DataProcessingPipeline

# 创建管道
pipeline = DataProcessingPipeline()

# 运行完整处理
results = pipeline.run_full_pipeline(save_intermediate=True)
```

### 方法2：使用增强管道（推荐）

```python
from data.processor import EnhancedDataProcessingPipeline

# 创建增强管道
pipeline = EnhancedDataProcessingPipeline(
    use_parallel=True,      # 启用并行处理
    use_incremental=True,   # 启用增量处理
    n_workers=4            # 使用4个工作进程
)

# 运行处理
results = pipeline.run_enhanced_pipeline(
    force_full_update=False,  # 不强制完整更新
    monitor_progress=True     # 显示进度条
)
```

### 方法3：向后兼容的函数接口

```python
# 直接使用原始接口
from data.processor.data_processing_pipeline import run

# 运行（与原始run()函数完全兼容）
run()
```

### 方法4：使用命令行

```bash
# 基础运行
python run_data_processing.py

# 增强运行（推荐）
python run_enhanced_processing.py

# 增量更新（日常使用）
python run_enhanced_processing.py --incremental

# 强制完整更新
python run_enhanced_processing.py --force
```

## 新增预处理项目的方法

### 1. 在现有处理器中添加

如果新功能属于现有类别，直接在相应处理器中添加方法：

#### 示例：添加新的收益率计算

```python
# 在 return_calculator.py 中添加
class ReturnCalculator(BaseDataProcessor):
    
    def calculate_sharpe_ratio(self, returns: pd.DataFrame, 
                              risk_free_rate: float = 0.03) -> pd.DataFrame:
        """
        计算夏普比率
        
        Args:
            returns: 收益率数据
            risk_free_rate: 无风险利率
            
        Returns:
            夏普比率DataFrame
        """
        # 实现计算逻辑
        excess_returns = returns - risk_free_rate / 252
        sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
        return sharpe
```

### 2. 创建新的处理器

如果是全新类型的数据处理，创建新的处理器类：

```python
# 创建新文件：data/processor/factor_processor.py
from .base_processor import BaseDataProcessor

class FactorProcessor(BaseDataProcessor):
    """因子计算处理器"""
    
    def __init__(self, config_path: Optional[str] = None):
        super().__init__(config_path)
        
    def validate_input(self, **kwargs) -> bool:
        # 实现输入验证
        return True
        
    def process(self, **kwargs):
        # 实现处理逻辑
        pass
        
    def calculate_momentum_factor(self, price_df: pd.DataFrame, 
                                lookback: int = 20) -> pd.DataFrame:
        """计算动量因子"""
        # 实现具体计算
        pass
```

### 3. 集成到管道中

在 `data_processing_pipeline.py` 中添加新处理步骤：

```python
class DataProcessingPipeline(BaseDataProcessor):
    
    def __init__(self, config_path: Optional[str] = None):
        super().__init__(config_path)
        # 添加新处理器
        self.factor_processor = FactorProcessor(config_path)
        
    def run_full_pipeline(self, save_intermediate: bool = True) -> Dict[str, Any]:
        # ... 现有处理步骤 ...
        
        # 6. 新增：计算因子
        self.logger.info("步骤6: 计算因子...")
        momentum_factor = self.factor_processor.calculate_momentum_factor(price_df)
        if save_intermediate:
            save_path = self.data_save_path / "momentum_factor.pkl"
            pd.to_pickle(momentum_factor, save_path)
            
        results['momentum_factor'] = momentum_factor
        
        return results
```

### 4. 处理流程扩展示例

```python
# 完整的扩展示例
class ExtendedPipeline(EnhancedDataProcessingPipeline):
    """扩展的数据处理管道"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 添加新的处理器
        self.factor_processor = FactorProcessor()
        self.risk_processor = RiskProcessor()
        
    def run_extended_pipeline(self, **kwargs):
        # 1. 运行基础处理
        results = self.run_enhanced_pipeline(**kwargs)
        
        # 2. 计算技术因子
        self.logger.info("计算技术因子...")
        tech_factors = self._calculate_technical_factors(results['price_df'])
        results['technical_factors'] = tech_factors
        
        # 3. 计算基本面因子
        self.logger.info("计算基本面因子...")
        fundamental_factors = self._calculate_fundamental_factors()
        results['fundamental_factors'] = fundamental_factors
        
        # 4. 风险模型
        self.logger.info("构建风险模型...")
        risk_model = self._build_risk_model(results)
        results['risk_model'] = risk_model
        
        return results
```

## 数据处理流程图

```
输入数据
    │
    ├─> PriceDataProcessor
    │       ├─> 清洗价格数据（剔除退市、北交所）
    │       ├─> 生成3D矩阵
    │       └─> 计算日期序列
    │
    ├─> ReturnCalculator
    │       ├─> 日/周/月收益率
    │       ├─> N天滚动收益率
    │       └─> 财报发布后收益率
    │
    ├─> FinancialDataProcessor
    │       ├─> 读取财报发布日期
    │       └─> 计算时间差特征
    │
    └─> [扩展处理器]
            ├─> 因子计算
            ├─> 风险模型
            └─> 其他特征
                    │
                    ▼
                输出文件
```

## 最佳实践

1. **模块化设计**：每个处理器负责一类相关功能
2. **继承基类**：新处理器应继承`BaseDataProcessor`
3. **统一接口**：保持`validate_input()`和`process()`接口
4. **错误处理**：使用基类的错误处理和日志功能
5. **增量支持**：考虑如何支持增量处理
6. **测试验证**：添加新功能时编写相应的测试

## 输出文件列表

运行完整管道后，会在数据目录生成以下文件：

- `Stock3d.pkl` - 3D价格矩阵
- `LogReturn_daily_o2o.pkl` - 日收益率(开盘到开盘)
- `LogReturn_daily_vwap.pkl` - 日收益率(VWAP)
- `LogReturn_weekly_o2o.pkl` - 周收益率(开盘到开盘)
- `LogReturn_weekly_vwap.pkl` - 周收益率(VWAP)
- `LogReturn_monthly_o2o.pkl` - 月收益率(开盘到开盘)
- `LogReturn_monthly_vwap.pkl` - 月收益率(VWAP)
- `LogReturn_5days_o2o.pkl` - 5天滚动收益率
- `LogReturn_20days_o2o.pkl` - 20天滚动收益率
- `released_dates_df.pkl` - 财报发布日期
- `released_dates_count_df.pkl` - 财报发布时间差
- `lag1_released_logreturn.pkl` - 财报发布后1天收益率
- `lag5_released_logreturn.pkl` - 财报发布后5天收益率
- `lag20_released_logreturn.pkl` - 财报发布后20天收益率
- `lag1_released_alfa_logreturn.pkl` - 财报发布后1天超额收益率
- `lag5_released_alfa_logreturn.pkl` - 财报发布后5天超额收益率
- `lag20_released_alfa_logreturn.pkl` - 财报发布后20天超额收益率