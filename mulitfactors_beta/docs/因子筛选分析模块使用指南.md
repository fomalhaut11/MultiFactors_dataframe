# 因子筛选分析模块使用指南

## 模块概述

因子筛选分析模块（`factors.analyzer`）提供了完整的因子研究工具，用于从大量测试过的因子中筛选出高质量的因子。该模块基于磁盘存储的历史测试结果，提供高效的筛选、评估和分析功能。

## 核心功能

### 1. 因子筛选（FactorScreener）
- 从磁盘加载历史测试结果
- 根据多维标准筛选因子
- 支持预设和自定义筛选标准
- 批量因子评分和排名

### 2. 稳定性分析
- 分析因子在不同时期的表现
- 计算稳定性指标
- 评估因子的可靠性

### 3. 因子比较
- 多因子横向对比
- 综合评分计算
- 因子分类管理

## 配置管理

### 配置位置
主配置文件：`config.yaml` 中的 `factor_analyzer` 部分

```yaml
factor_analyzer:
  screening:
    ic_mean_min: 0.02          # IC均值最小阈值
    icir_min: 0.5              # ICIR最小阈值
    monotonicity_min: 0.6      # 单调性最小阈值
    sharpe_min: 1.0            # 夏普比率最小阈值
  stability:
    lookback_window: 30        # 回望窗口（天）
    rolling_window: 252        # 滚动窗口（交易日）
  # ... 更多配置
```

### 配置优先级
1. 运行时参数（最高优先级）
2. 用户自定义配置
3. 项目配置文件
4. 默认值（最低优先级）

## 使用流程

### 基础使用流程

```python
from factors.analyzer import FactorScreener

# 1. 创建筛选器
screener = FactorScreener()

# 2. 加载历史测试结果
results = screener.load_all_results()

# 3. 筛选因子
selected_factors = screener.screen_factors()

# 4. 获取排名
ranking = screener.get_factor_ranking(metric='icir', top_n=10)
```

### 完整工作流程

```python
# 步骤1：测试因子（使用单因子测试模块）
from factors.tester import SingleFactorTestPipeline

pipeline = SingleFactorTestPipeline()
for factor in ['BP', 'EP', 'ROE']:
    result = pipeline.run(factor, save_result=True)

# 步骤2：筛选分析
from factors.analyzer import FactorScreener

screener = FactorScreener()
screener.load_all_results(force_reload=True)

# 使用自定义标准筛选
top_factors = screener.screen_factors({
    'ic_mean_min': 0.02,
    'icir_min': 0.5,
    'monotonicity_min': 0.6
})

# 步骤3：生成报告
report = screener.generate_screening_report(
    output_path='factor_report.csv',
    top_n=20
)
```

## API详解

### FactorScreener类

#### 初始化
```python
screener = FactorScreener(
    test_data_path=None,      # 测试结果路径，默认从配置读取
    config_override=None       # 配置覆盖参数
)
```

#### 加载结果
```python
results = screener.load_all_results(
    date_range=('2024-01-01', '2024-12-31'),  # 日期范围
    factor_names=['BP', 'EP', 'ROE'],         # 因子列表
    force_reload=False                         # 强制重新加载
)
```

#### 筛选因子
```python
# 使用默认标准
selected = screener.screen_factors()

# 使用自定义标准
selected = screener.screen_factors(
    criteria={
        'ic_mean_min': 0.03,
        'icir_min': 0.7,
        'sharpe_min': 1.5
    }
)

# 使用预设标准
selected = screener.screen_factors(preset='strict')  # 'strict', 'normal', 'loose'
```

#### 因子排名
```python
ranking_df = screener.get_factor_ranking(
    metric='icir',      # 排序指标
    top_n=10,          # 前N个
    ascending=False    # 降序
)
```

#### 稳定性分析
```python
stability = screener.analyze_factor_stability(
    factor_name='BP',
    lookback_days=30   # 回望天数
)
```

#### 因子比较
```python
comparison = screener.compare_factors(['BP', 'EP', 'ROE'])
```

#### 生成报告
```python
report = screener.generate_screening_report(
    output_path='report.csv',
    top_n=20
)
```

## 筛选标准说明

### 基础指标
- **ic_mean_min**: IC均值最小阈值（建议 > 0.02）
- **icir_min**: ICIR最小阈值（建议 > 0.5）
- **monotonicity_min**: 单调性最小阈值（建议 > 0.6）
- **sharpe_min**: 夏普比率最小阈值（建议 > 1.0）
- **t_value_min**: t值最小阈值（建议 > 2.0）

### 风险指标
- **max_drawdown_limit**: 最大回撤限制（建议 < 0.3）
- **win_rate_min**: 胜率最小阈值（建议 > 0.55）

### 预设标准

| 预设 | IC均值 | ICIR | 单调性 | 夏普比率 | t值 |
|------|--------|------|--------|----------|-----|
| strict | 0.03 | 0.7 | 0.7 | 1.5 | 2.5 |
| normal | 0.02 | 0.5 | 0.6 | 1.0 | 2.0 |
| loose | 0.01 | 0.3 | 0.4 | 0.5 | 1.5 |

## 稳定性评级

稳定性根据IC稳定性（IC均值/IC标准差）评级：

- **Excellent**: IC稳定性 > 3
- **Good**: IC稳定性 > 2
- **Fair**: IC稳定性 > 1
- **Poor**: IC稳定性 ≤ 1

## 综合评分算法

因子综合评分（0-100分）由以下部分加权组成：

- IC得分（25%）：基于IC均值
- 稳定性得分（20%）：基于IC稳定性
- 单调性得分（20%）：基于分组单调性
- 夏普得分（20%）：基于多空组合夏普比率
- 稳健性得分（15%）：基于多期测试结果

## 最佳实践

### 1. 定期更新测试结果
```python
# 每周/每月运行
pipeline = SingleFactorTestPipeline()
for factor in factor_list:
    pipeline.run(factor, save_result=True)
```

### 2. 分层筛选策略
```python
# 第一层：宽松筛选
candidates = screener.screen_factors(preset='loose')

# 第二层：稳定性筛选
stable_factors = []
for factor in candidates:
    stability = screener.analyze_factor_stability(factor)
    if stability.get('stability_grade') in ['Good', 'Excellent']:
        stable_factors.append(factor)

# 第三层：严格筛选
final_factors = screener.screen_factors(
    criteria={'ic_mean_min': 0.03, 'icir_min': 0.7},
    factor_names=stable_factors
)
```

### 3. 因子组合构建
```python
# 筛选不同类别的顶级因子
categories = ['value', 'growth', 'momentum']
selected_factors = []

for category in categories:
    category_factors = [f for f in all_factors 
                       if screener.config.get_factor_category(f) == category]
    ranking = screener.get_factor_ranking(
        factor_names=category_factors,
        top_n=2
    )
    selected_factors.extend(ranking['factor_name'].tolist())
```

### 4. 监控因子衰减
```python
# 定期检查因子稳定性
for factor in production_factors:
    stability = screener.analyze_factor_stability(factor, lookback_days=60)
    if stability['ic_stability'] < 1.5:
        print(f"警告: {factor} 稳定性下降")
```

## 注意事项

1. **数据依赖**：确保已运行单因子测试并保存结果
2. **缓存管理**：适时使用 `force_reload=True` 更新缓存
3. **路径配置**：确保 `config.yaml` 中的路径配置正确
4. **性能优化**：大量因子时，考虑分批加载和处理

## 故障排除

### 问题1：找不到测试结果
```python
# 检查路径配置
from core.config_manager import get_path
print(get_path('single_factor_test'))

# 手动指定路径
screener = FactorScreener(test_data_path='your/path/here')
```

### 问题2：内存不足
```python
# 分批处理
batch_size = 50
for i in range(0, len(factor_list), batch_size):
    batch = factor_list[i:i+batch_size]
    screener.load_all_results(factor_names=batch)
    # 处理...
```

### 问题3：配置未生效
```python
# 检查配置
from factors.analyzer import get_analyzer_config
config = get_analyzer_config()
print(config.screening)

# 手动覆盖
screener = FactorScreener(config_override={'screening': {'ic_mean_min': 0.03}})
```

## 扩展开发

模块预留了扩展接口，可以添加：

1. **自定义评分算法**：继承 `FactorScreener` 并重写 `get_factor_score`
2. **新的分析方法**：在 `stability/` 目录添加新的分析模块
3. **报告模板**：在 `reports/` 目录添加自定义报告生成器

## 相关模块

- `factors.tester`: 单因子测试模块
- `factors.builder`: 因子构建模块（待开发）
- `factors.selector`: 动态因子选择模块（待开发）
- `portfolio.backtest`: 回测模块（待开发）

## 更新日志

### v1.0.0 (2025-08-07)
- 初始版本发布
- 实现基础筛选功能
- 添加稳定性分析
- 支持配置管理

---

*更多信息请参考项目文档或联系开发团队*