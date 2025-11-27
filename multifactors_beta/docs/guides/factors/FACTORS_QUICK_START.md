# Factors模块快速开始指南

## 🚀 一行导入，全部功能

```python
import factors

# 现在你拥有了完整的因子研究工具包！
```

## 📋 快速使用示例

### 1. 查看可用功能
```python
# 查看所有注册的因子
factor_list = factors.list_factors()
print(factor_list)
# 输出: {'profitability': ['ROE_ttm', 'ROA_ttm', ...], ...}

# 系统概览
summary = factors.get_factor_summary()
print(f"共有 {summary['total_factors']} 个因子")
```

### 2. 数据预处理
```python
# TTM计算（最常用）
ttm_data = factors.calculate_ttm(financial_data)
earnings_ttm = ttm_data['DEDUCTEDPROFIT_ttm']

# 其他财务数据处理
yoy_data = factors.calculate_yoy(financial_data)      # 同比增长
single_q = factors.calculate_single_quarter(data)    # 单季度值
zscore = factors.calculate_zscore(data)               # 标准化

# Alpha191技术运算
price_rank = factors.ts_rank(price_data, window=20)  # 时序排名
price_delta = factors.delta(price_data, period=1)    # 差分
```

### 3. 计算因子
```python
# 单个因子计算
roe = factors.calculate_factor('ROE_ttm', financial_data)
print(f"ROE因子: {roe.name}, 有效值: {roe.notna().sum()}")

# 批量计算多个因子
factor_names = ['ROE_ttm', 'ROA_ttm', 'EP_ttm']
results = factors.batch_calculate_factors(factor_names, financial_data)

for name, result in results.items():
    if result is not None:
        print(f"{name}: {result.notna().sum()} 个有效值")
```

### 4. 因子测试
```python
# 快速测试单个因子
test_result = factors.test_factor('ROE_ttm')
print(f"IC均值: {test_result.ic_result.ic_mean:.4f}")
print(f"IC标准差: {test_result.ic_result.ic_std:.4f}")

# 批量测试
batch_results = factors.batch_test(['ROE_ttm', 'ROA_ttm'])
for name, result in batch_results.items():
    print(f"{name}: IC={result.ic_result.ic_mean:.4f}")
```

### 5. 因子分析
```python
# 因子筛选
screener = factors.FactorScreener()
qualified = screener.screen_factors(preset='normal')  # 'loose', 'normal', 'strict'
print(f"通过筛选的因子: {qualified}")

# 详细分析
analysis = screener.analyze_factors(['ROE_ttm', 'ROA_ttm'])
```

### 6. 因子组合
```python
# 创建因子组合器
combiner = factors.FactorCombiner()

# 线性组合多个因子
factor_dict = {
    'ROE_ttm': factors.calculate_factor('ROE_ttm', financial_data),
    'ROA_ttm': factors.calculate_factor('ROA_ttm', financial_data)
}
combined = combiner.combine_factors(factor_dict, weights=[0.6, 0.4])

# 因子正交化
orthogonal = combiner.orthogonalize_factors(factor_dict)
```

### 7. 因子信息查询
```python
# 获取因子详细信息
info = factors.get_factor_info('ROE_ttm')
print(f"描述: {info['description']}")
print(f"公式: {info['formula']}")
print(f"依赖字段: {info['dependencies']}")

# 搜索相关因子
roe_related = factors.search_factors(keyword='ROE')
profit_factors = factors.search_factors(category='profitability')
```

## 📊 数据格式要求

### 财务数据格式
```python
# MultiIndex DataFrame，索引为 [ReportDates, StockCodes]
financial_data.index.names  # ['ReportDates', 'StockCodes']
financial_data.columns      # ['DEDUCTEDPROFIT', 'EQY_BELONGTO_PARCOMSH', ...]

# 必须包含时间字段
required_cols = ['d_year', 'd_quarter']
```

### 价格数据格式
```python
# 对于Alpha191运算，需要宽格式 (时间 x 股票)
price_wide = pd.DataFrame(
    index=trading_dates,     # 时间索引
    columns=stock_codes      # 股票列
)
```

## 🎯 常用因子介绍

### 盈利能力因子
- **ROE_ttm**: 净资产收益率，衡量股东权益回报
- **ROA_ttm**: 总资产收益率，衡量资产使用效率
- **GrossProfitMargin_ttm**: 毛利率，衡量产品定价能力

### 估值因子
- **EP_ttm**: 盈利收益率（PE倒数），估值指标
- **BP_ttm**: 账面市值比（PB倒数），价值指标

### 质量因子
- **AccrualRatio_ttm**: 应计项目比率，盈利质量指标

## 🔧 扩展开发

### 添加自定义因子
```python
from factors.library import register_factor

@register_factor(
    name='MyFactor',
    category='custom',
    description='我的自定义因子',
    dependencies=['FIELD1', 'FIELD2']
)
def my_custom_factor(data, **kwargs):
    # 使用基础工具
    ttm_data = factors.calculate_ttm(data)
    
    # 自定义计算逻辑
    result = ttm_data['FIELD1_ttm'] / ttm_data['FIELD2_ttm']
    
    return result.replace([np.inf, -np.inf], np.nan)

# 自动注册后即可使用
my_result = factors.calculate_factor('MyFactor', data)
```

### 使用原始工具函数
```python
# 直接使用generators中的基础工具
from factors.generators.financial import calculate_ttm
from factors.generators.alpha191 import ts_rank, delta

# 组合基础工具实现复杂逻辑
ttm_result = calculate_ttm(data)
ranked_result = ts_rank(ttm_result, window=20)
```

## 💡 最佳实践

1. **数据准备**：确保数据包含必要的时间字段（d_year, d_quarter）
2. **因子选择**：使用`list_factors()`和`search_factors()`探索可用因子
3. **批量计算**：对多个因子使用`batch_calculate_factors()`提高效率
4. **错误处理**：因子计算会自动处理异常，返回空Series而不抛出错误
5. **性能优化**：对大数据集，考虑分块处理或使用内存高效的方法

## 🆘 常见问题

**Q: 计算结果为空或NaN很多？**
A: 检查输入数据是否包含因子依赖的字段，使用`get_factor_info()`查看依赖

**Q: 如何查看因子的计算公式？**
A: 使用`get_factor_info('因子名')['formula']`查看

**Q: 可以添加自己的因子吗？**  
A: 可以！使用`@register_factor`装饰器注册新因子

**Q: 支持哪些数据频率？**
A: 主要支持季报数据，部分估值因子需要日频市值数据

---

开始你的因子研究之旅：`import factors` 🚀