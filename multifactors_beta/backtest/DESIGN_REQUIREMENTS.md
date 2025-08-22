# 回测模块设计要求文档

## 1. 模块概述

回测模块是MultiFactors项目的核心组件之一，专门用于验证因子策略和投资组合的实际效果。该模块采用事件驱动架构，支持权重驱动和策略驱动两种回测模式，能够模拟真实的交易环境。

## 2. 设计目标

### 2.1 核心目标
- **真实性**: 模拟真实的交易环境，包括交易成本、市场冲击等
- **灵活性**: 支持多种回测模式和参数配置
- **高性能**: 能够处理大规模历史数据的快速回测
- **可扩展性**: 易于扩展新的成本模型和绩效指标

### 2.2 技术目标
- 支持权重驱动的投资组合回测
- 精确计算交易成本（手续费、滑点、市场冲击）
- 提供全面的绩效分析和风险评估
- 支持基准比较和相对绩效分析

## 3. 系统架构

### 3.1 整体架构
```
backtest/
├── engine/                 # 回测引擎
│   ├── __init__.py
│   └── backtest_engine.py  # 核心回测引擎
├── portfolio/              # 组合管理
│   ├── __init__.py
│   └── portfolio_manager.py # 投资组合管理器
├── cost/                   # 成本模型
│   ├── __init__.py
│   ├── commission.py       # 手续费模型
│   ├── slippage.py         # 滑点模型
│   └── market_impact.py    # 市场冲击模型
├── performance/            # 绩效分析
│   ├── __init__.py
│   ├── result.py          # 回测结果
│   └── metrics.py         # 绩效指标计算
├── utils/                  # 工具模块
│   ├── __init__.py
│   └── validation.py       # 数据验证
└── __init__.py
```

### 3.2 核心组件

#### 3.2.1 BacktestEngine (回测引擎)
- **职责**: 统筹整个回测流程，协调各个组件
- **接口**: `run_with_weights()` - 权重驱动回测
- **功能**: 数据验证、时间循环、结果汇总

#### 3.2.2 PortfolioManager (投资组合管理器)
- **职责**: 管理投资组合状态，执行交易指令
- **功能**: 持仓管理、交易执行、现金管理、成本计算

#### 3.2.3 成本模型组件
- **CommissionModel**: 手续费计算（支持分层费率、最小手续费等）
- **SlippageModel**: 滑点成本计算（固定、比例、平方根等多种模型）
- **MarketImpactModel**: 市场冲击建模（临时冲击和永久冲击）

#### 3.2.4 绩效分析组件
- **BacktestResult**: 回测结果存储和计算
- **PerformanceMetrics**: 绩效指标计算器

## 4. 数据格式要求

### 4.1 权重数据格式
```python
# DataFrame格式: 
# - 索引: 日期 (DatetimeIndex)
# - 列: 股票代码 (str)
# - 值: 权重 (float, 每行和为1.0)

weights_data = pd.DataFrame({
    '000001.SZ': [0.3, 0.25, 0.4],
    '000002.SZ': [0.2, 0.35, 0.3], 
    '600000.SH': [0.5, 0.4, 0.3]
}, index=pd.DatetimeIndex(['2023-01-01', '2023-01-02', '2023-01-03']))
```

### 4.2 价格数据格式
```python
# DataFrame格式:
# - 索引: 日期 (DatetimeIndex)  
# - 列: 股票代码 (str)
# - 值: 价格 (float)

price_data = pd.DataFrame({
    '000001.SZ': [10.0, 10.5, 11.0],
    '000002.SZ': [20.0, 19.5, 21.0],
    '600000.SH': [15.0, 15.2, 14.8]
}, index=pd.DatetimeIndex(['2023-01-01', '2023-01-02', '2023-01-03']))
```

## 5. API设计

### 5.1 主要接口

#### 5.1.1 BacktestEngine.run_with_weights()
```python
def run_with_weights(self,
                    weights_data: pd.DataFrame,
                    price_data: pd.DataFrame,
                    start_date: Optional[str] = None,
                    end_date: Optional[str] = None,
                    rebalance_freq: str = 'daily',
                    **kwargs) -> BacktestResult:
```

#### 5.1.2 便捷函数
```python
def run_weights_backtest(weights_data: pd.DataFrame,
                        price_data: pd.DataFrame,
                        initial_capital: float = 1000000.0,
                        commission_rate: float = 0.001,
                        slippage_rate: float = 0.0005,
                        **kwargs) -> BacktestResult:
```

### 5.2 结果获取接口
- `get_portfolio_summary()` - 投资组合汇总
- `get_performance_metrics()` - 详细绩效指标
- `get_trade_statistics()` - 交易统计
- `plot_results()` - 绩效图表
- `save_results()` - 保存结果

## 6. 成本建模要求

### 6.1 手续费模型
- 支持固定费率和分层费率
- 支持最小手续费和最大手续费限制
- 区分买入和卖出手续费（如印花税）
- 支持VIP等级和交易量折扣

### 6.2 滑点模型
- 固定滑点模型
- 比例滑点模型（基于交易量占比）
- 平方根滑点模型
- 组合滑点模型
- 自适应滑点（考虑时间和市场条件）

### 6.3 市场冲击模型
- 临时冲击（交易后逐渐恢复）
- 永久冲击（对价格的持久影响）
- 冲击衰减机制
- 非线性冲击（大额交易）
- 高级冲击模型（考虑流动性、波动率等）

## 7. 绩效分析要求

### 7.1 基础收益指标
- 总收益率、年化收益率
- 平均日收益率、几何平均收益率

### 7.2 风险指标  
- 波动率（日度、年化）
- 最大回撤及持续时间
- VaR和CVaR
- 下行偏差、偏度、峰度

### 7.3 风险调整收益指标
- 夏普比率、索提诺比率
- 卡玛比率、信息比率
- Omega比率、胜率、盈亏比

### 7.4 相对基准指标
- 超额收益、跟踪误差
- Beta、Alpha、Treynor比率
- 相关性、上行/下行捕获率

## 8. 性能要求

### 8.1 时间复杂度
- 单次回测: O(T × N)，其中T为时间长度，N为股票数量
- 内存使用: 线性增长，支持大规模数据处理

### 8.2 数据处理能力
- 支持10年以上历史数据回测
- 支持1000+股票同时回测
- 日级别回测在分钟级别内完成

## 9. 扩展性设计

### 9.1 成本模型扩展
- 基于基类的成本模型设计
- 支持自定义成本计算逻辑
- 预定义常用配置

### 9.2 绩效指标扩展
- 模块化的指标计算器
- 支持自定义指标函数
- 滚动窗口指标计算

### 9.3 策略集成
- 预留策略驱动回测接口
- 支持事件驱动的策略回调

## 10. 数据验证要求

### 10.1 权重数据验证
- 权重和检查（每行和为1.0）
- 缺失值处理
- 日期索引格式检查
- 数值范围检查（0-1之间）

### 10.2 价格数据验证
- 价格合理性检查（正数）
- 缺失值前向填充
- 异常值检测和处理
- 数据对齐检查

## 11. 配置管理

### 11.1 预定义配置
```python
COMMISSION_CONFIGS = {
    'default': {'commission_rate': 0.001, 'min_commission': 5.0},
    'low_cost': {'commission_rate': 0.0005, 'min_commission': 1.0},
    'institutional': {'commission_rate': 0.0003, 'min_commission': 0.0}
}

SLIPPAGE_CONFIGS = {
    'low_impact': {'fixed_slippage': 0.0003},
    'moderate_impact': {'fixed_slippage': 0.0005}, 
    'high_impact': {'fixed_slippage': 0.001}
}
```

### 11.2 配置文件支持
- 支持JSON/YAML配置文件
- 环境变量覆盖
- 运行时参数调整

## 12. 日志和监控

### 12.1 日志要求
- 详细的回测进度日志
- 错误和警告记录
- 性能监控日志
- 调试信息记录

### 12.2 进度监控
- 回测进度百分比
- 实时性能指标
- 内存使用监控

## 13. 质量保证

### 13.1 单元测试要求
- 核心功能100%测试覆盖
- 边界条件测试
- 异常情况处理测试
- 性能基准测试

### 13.2 集成测试
- 端到端回测流程测试
- 不同配置组合测试
- 大数据量压力测试

这份设计文档为回测模块的开发提供了全面的技术要求和实现指导，确保模块能够满足实际使用需求。