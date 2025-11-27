# 回测系统使用指南

## 📋 目录
1. [核心使用方式](#核心使用方式)
2. [权重数据格式](#权重数据格式)
3. [API调用示例](#api调用示例)
4. [与现有模块集成](#与现有模块集成)
5. [高级用法](#高级用法)

---

## 核心使用方式

### 🎯 回答你的核心问题

**是的，你可以直接将每天的股票权重输入给回测模块！**

回测系统提供了 `run_with_weights()` 接口，专门用于处理预先计算好的每日权重数据。

### 典型工作流程

```
你的策略/模型 → 每日权重计算 → 回测引擎 → 绩效报告
    ↓              ↓            ↓         ↓
  因子信号      权重DataFrame   交易模拟    风险指标
```

---

## 权重数据格式

### 标准格式要求

```python
# 正确的权重DataFrame格式：每行是一天，每列是一只股票
daily_weights_data = [
    [0.333, 0.333, 0.334],  # 2020-01-01的权重分配
    [0.300, 0.400, 0.300],  # 2020-01-02的权重分配  
    [0.250, 0.450, 0.300],  # 2020-01-03的权重分配
    # ... 更多天的权重
]

portfolio_weights = pd.DataFrame(
    daily_weights_data,
    index=pd.date_range('2020-01-01', periods=3, freq='D'),  # 行：日期
    columns=['000001.SZ', '000002.SZ', '000300.SZ']          # 列：股票代码
)

# 数据结构说明：
#                  000001.SZ  000002.SZ  000300.SZ
# 2020-01-01       0.333      0.333      0.334     ← 第一天的权重分配
# 2020-01-02       0.300      0.400      0.300     ← 第二天的权重分配
# 2020-01-03       0.250      0.450      0.300     ← 第三天的权重分配

# 关键要求：
# 1. index必须是DatetimeIndex（交易日期）- 每行代表一天
# 2. columns是股票代码 - 每列代表一只股票
# 3. 每行权重和应该为1.0（每天的权重分配总和为100%）
# 4. 支持权重为0（表示某天不持有该股票）
# 5. 支持权重变化（引擎会自动计算每日调仓需求）
```

### 数据验证

```python
def validate_weights_example():
    """权重数据验证示例"""
    
    # ✅ 正确的权重数据
    good_weights = pd.DataFrame({
        '000001.SZ': [0.4, 0.3, 0.5],
        '000002.SZ': [0.6, 0.7, 0.5],
    }, index=pd.date_range('2020-01-01', periods=3))
    
    # ❌ 常见错误
    # 错误1：权重和不为1
    bad_weights1 = pd.DataFrame({
        '000001.SZ': [0.8, 0.9],  # 权重和 > 1
        '000002.SZ': [0.5, 0.6],
    }, index=pd.date_range('2020-01-01', periods=2))
    
    # 错误2：缺失日期
    bad_weights2 = pd.DataFrame({
        '000001.SZ': [0.5, None, 0.5],  # 有缺失值
        '000002.SZ': [0.5, 0.6, 0.5],
    })
    
    # 回测引擎会自动处理这些问题：
    # - 权重归一化
    # - 缺失值填充
    # - 日期对齐
```

---

## API调用示例

### 1. 基础用法：直接输入权重

```python
from backtest import BacktestEngine
import pandas as pd
import numpy as np

# Step 1: 准备权重数据
dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
stocks = ['000001.SZ', '000002.SZ', '000300.SZ']

# 假设这是你的模型/策略输出的权重
portfolio_weights = pd.DataFrame({
    '000001.SZ': np.random.dirichlet([1, 1, 1], len(dates))[:, 0],
    '000002.SZ': np.random.dirichlet([1, 1, 1], len(dates))[:, 1], 
    '000300.SZ': np.random.dirichlet([1, 1, 1], len(dates))[:, 2],
}, index=dates)

# Step 2: 创建回测引擎
engine = BacktestEngine(
    initial_capital=1_000_000,    # 100万初始资金
    commission_rate=0.001,        # 0.1% 手续费
    slippage_rate=0.0005,         # 0.05% 滑点
    market_impact_model='linear'   # 线性市场冲击模型
)

# Step 3: 执行回测
result = engine.run_with_weights(portfolio_weights)

# Step 4: 查看结果
print("=== 回测结果 ===")
print(f"总收益率: {result.total_return:.2%}")
print(f"年化收益率: {result.annual_return:.2%}")
print(f"夏普比率: {result.sharpe_ratio:.2f}")
print(f"最大回撤: {result.max_drawdown:.2%}")
print(f"年化换手率: {result.turnover_rate:.2f}")
print(f"总交易成本: {result.total_costs:.2f}")

# Step 5: 生成详细报告
result.generate_report('backtest_report.html')
```

### 2. 高级用法：自定义交易成本

```python
# 自定义成本模型
custom_costs = {
    'commission': {
        'rate': 0.001,          # 基础费率
        'min_fee': 5.0,         # 最小手续费
        'max_rate': 0.005       # 最大费率上限
    },
    'slippage': {
        'model': 'sqrt',        # 平方根模型
        'coefficient': 0.1      # 滑点系数
    },
    'market_impact': {
        'temporary': 0.0001,    # 临时冲击
        'permanent': 0.00005    # 永久冲击
    }
}

engine = BacktestEngine(
    initial_capital=1_000_000,
    cost_models=custom_costs
)

result = engine.run_with_weights(
    portfolio_weights,
    rebalance_tolerance=0.01  # 1%权重变化才交易
)
```

### 3. 实时权重生成

```python
def dynamic_strategy(date, current_portfolio, market_data):
    """
    动态权重生成策略
    
    这个函数会在每个交易日被调用，返回当日目标权重
    """
    # 获取最新因子数据
    latest_factors = get_factors_for_date(date)
    
    # 计算预期收益
    expected_returns = calculate_expected_returns(latest_factors)
    
    # 风险模型预测
    risk_model = BarraModel()
    cov_matrix = risk_model.predict_covariance()
    
    # 组合优化
    optimizer = MeanVarianceOptimizer(risk_model)
    result = optimizer.optimize(
        expected_returns=expected_returns,
        constraints={
            'max_weight': 0.1,      # 单股票最大10%
            'max_turnover': 0.2,    # 最大20%换手
            'sector_limits': {      # 行业限制
                'technology': 0.3,
                'finance': 0.2
            }
        }
    )
    
    return result['weights']

# 使用动态策略
result = engine.run_streaming(
    weight_generator=dynamic_strategy,
    start_date='2020-01-01',
    end_date='2023-12-31'
)
```

---

## 与现有模块集成

### 完整的因子投资工作流

```python
def complete_factor_workflow():
    """展示完整的因子投资回测流程"""
    
    # ========================================
    # 第一步：因子生成
    # ========================================
    from factors.generator.financial import PureFinancialFactorCalculator
    from factors.generator.technical import PriceFactorCalculator
    
    # 生成基础因子
    financial_calc = PureFinancialFactorCalculator()
    technical_calc = PriceFactorCalculator()
    
    financial_factors = financial_calc.calculate_multiple([
        'ROE_ttm', 'CurrentRatio', 'DebtToAssets'
    ])
    
    technical_factors = technical_calc.calculate_multiple([
        'Momentum_20d', 'Reversal_5d'
    ])
    
    # ========================================
    # 第二步：因子测试和筛选
    # ========================================
    from factors.tester import SingleFactorTestPipeline
    from factors.analyzer.screening import FactorScreener
    
    # 测试因子有效性
    pipeline = SingleFactorTestPipeline()
    test_results = {}
    
    all_factors = {**financial_factors, **technical_factors}
    for name, factor in all_factors.items():
        test_results[name] = pipeline.run(name, factor_data=factor)
    
    # 筛选有效因子
    screener = FactorScreener()
    good_factors = screener.screen_factors(
        test_results, 
        criteria={'ic_mean_min': 0.02, 'icir_min': 0.5}
    )
    
    # ========================================  
    # 第三步：因子组合
    # ========================================
    from factors.combiner import FactorCombiner
    
    combiner = FactorCombiner(method='ic_weight')
    composite_factor = combiner.combine(
        {name: all_factors[name] for name in good_factors}
    )
    
    # ========================================
    # 第四步：组合优化
    # ========================================
    from factors.risk_model import BarraModel, MeanVarianceOptimizer
    
    # 构建风险模型
    risk_model = BarraModel()
    factor_exposures = build_factor_exposures()  # 假设函数
    stock_returns = load_stock_returns()         # 假设函数
    risk_model.fit(factor_exposures, stock_returns)
    
    # 创建优化器
    optimizer = MeanVarianceOptimizer(risk_model)
    
    # 按日期优化权重
    daily_weights = []
    dates = composite_factor.index.get_level_values(0).unique()
    
    for date in dates:
        # 获取当日因子值作为预期收益
        daily_factors = composite_factor.xs(date, level=0)
        
        # 组合优化
        opt_result = optimizer.optimize(
            expected_returns=daily_factors,
            constraints={
                'max_weight': 0.05,     # 单股票最大5%
                'min_weight': 0.0,      # 不允许做空
                'max_turnover': 0.3,    # 最大30%换手
                'target_risk': 0.15     # 目标15%年化风险
            }
        )
        
        daily_weights.append({
            'date': date,
            'weights': opt_result['weights'],
            'expected_return': opt_result['expected_return'],
            'predicted_risk': opt_result['risk']
        })
    
    # 构建权重DataFrame
    portfolio_weights = pd.DataFrame([w['weights'] for w in daily_weights])
    portfolio_weights.index = [w['date'] for w in daily_weights]
    
    # ========================================
    # 第五步：回测执行
    # ========================================
    engine = BacktestEngine(
        initial_capital=10_000_000,  # 1000万
        commission_rate=0.0008,      # 万8手续费
        slippage_rate=0.0003         # 3bp滑点
    )
    
    backtest_result = engine.run_with_weights(
        portfolio_weights,
        benchmark='000300.SH',  # 沪深300基准
        risk_free_rate=0.025    # 2.5%无风险利率
    )
    
    # ========================================
    # 第六步：结果分析
    # ========================================
    
    print("=== 因子策略回测结果 ===")
    print(f"回测期间: {portfolio_weights.index[0]} - {portfolio_weights.index[-1]}")
    print(f"总收益率: {backtest_result.total_return:.2%}")
    print(f"年化收益率: {backtest_result.annual_return:.2%}")
    print(f"年化波动率: {backtest_result.annual_volatility:.2%}")
    print(f"夏普比率: {backtest_result.sharpe_ratio:.2f}")
    print(f"信息比率: {backtest_result.information_ratio:.2f}")
    print(f"最大回撤: {backtest_result.max_drawdown:.2%}")
    print(f"胜率: {backtest_result.win_rate:.1%}")
    print(f"年化换手率: {backtest_result.annual_turnover:.1f}x")
    
    # 生成详细分析报告
    backtest_result.generate_detailed_report(
        save_path='factor_strategy_report.html',
        include_factor_analysis=True,
        include_risk_attribution=True
    )
    
    return backtest_result

# 执行完整流程
result = complete_factor_workflow()
```

---

## 高级用法

### 1. 多策略组合回测

```python
def multi_strategy_backtest():
    """多策略组合回测"""
    
    # 策略A：动量策略
    momentum_weights = calculate_momentum_weights()
    
    # 策略B：价值策略  
    value_weights = calculate_value_weights()
    
    # 策略C：质量策略
    quality_weights = calculate_quality_weights()
    
    # 组合权重（策略间分配）
    strategy_allocation = {
        'momentum': 0.4,
        'value': 0.4, 
        'quality': 0.2
    }
    
    # 合并策略权重
    combined_weights = (
        momentum_weights * strategy_allocation['momentum'] +
        value_weights * strategy_allocation['value'] +
        quality_weights * strategy_allocation['quality']
    )
    
    # 回测组合策略
    engine = BacktestEngine()
    result = engine.run_with_weights(combined_weights)
    
    return result
```

### 2. 分层回测（按市值、行业等）

```python
def stratified_backtest():
    """分层回测示例"""
    
    # 按市值分层
    market_cap_data = load_market_cap()
    
    results = {}
    for cap_bucket in ['large', 'mid', 'small']:
        # 获取对应市值区间的股票
        bucket_stocks = get_stocks_by_market_cap(cap_bucket)
        
        # 过滤权重数据
        bucket_weights = portfolio_weights[bucket_stocks]
        bucket_weights = bucket_weights.div(bucket_weights.sum(axis=1), axis=0)
        
        # 分别回测
        engine = BacktestEngine()
        results[cap_bucket] = engine.run_with_weights(bucket_weights)
    
    # 比较不同市值区间的表现
    for bucket, result in results.items():
        print(f"{bucket} cap - Return: {result.annual_return:.2%}, "
              f"Sharpe: {result.sharpe_ratio:.2f}")
```

### 3. 滚动回测

```python
def rolling_backtest(window_months=12):
    """滚动回测：模拟真实投资中的样本外测试"""
    
    results = []
    start_date = pd.Timestamp('2020-01-01')
    end_date = pd.Timestamp('2023-12-31')
    
    current_date = start_date
    while current_date + pd.DateOffset(months=window_months) <= end_date:
        # 定义当前窗口
        window_start = current_date
        window_end = current_date + pd.DateOffset(months=window_months)
        
        # 提取窗口内的权重
        window_weights = portfolio_weights[
            (portfolio_weights.index >= window_start) & 
            (portfolio_weights.index < window_end)
        ]
        
        if len(window_weights) > 0:
            # 执行窗口回测
            engine = BacktestEngine()
            window_result = engine.run_with_weights(window_weights)
            
            results.append({
                'period': f"{window_start.strftime('%Y-%m')} - {window_end.strftime('%Y-%m')}",
                'return': window_result.total_return,
                'sharpe': window_result.sharpe_ratio,
                'max_dd': window_result.max_drawdown
            })
        
        # 移动到下一个窗口（重叠50%）
        current_date += pd.DateOffset(months=window_months//2)
    
    # 分析结果稳定性
    results_df = pd.DataFrame(results)
    print(f"平均收益率: {results_df['return'].mean():.2%} ± {results_df['return'].std():.2%}")
    print(f"收益率稳定性: {results_df['return'].std() / results_df['return'].mean():.2f}")
    
    return results_df
```

---

## 关键要点总结

### ✅ 你需要知道的要点

1. **直接权重输入**: 是的，可以直接将每日权重DataFrame传入回测引擎
2. **自动交易计算**: 引擎会自动计算权重变化对应的交易需求
3. **成本考虑**: 全面考虑手续费、滑点、市场冲击等交易成本
4. **灵活配置**: 支持自定义成本模型、约束条件、风险控制等
5. **结果丰富**: 提供详细的绩效指标、归因分析、可视化报告

### 🎯 典型使用场景

- **因子策略回测**: 将因子信号转换为权重进行回测
- **量化策略验证**: 验证优化算法的实际效果
- **风险模型测试**: 测试风险控制和组合约束的有效性
- **成本影响分析**: 分析交易成本对策略收益的影响
- **参数敏感性**: 测试不同参数设置的策略表现

这样的设计让你可以专注于策略和权重的生成，而将繁琐的交易模拟、成本计算、绩效分析等工作交给回测引擎处理。