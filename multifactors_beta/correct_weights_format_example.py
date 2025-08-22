"""
正确的权重数据格式示例
清楚展示每日股票权重的正确输入方式
"""

import pandas as pd
import numpy as np

def create_correct_weights_format():
    """演示正确的权重数据格式"""
    
    print("=" * 60)
    print("正确的每日股票权重数据格式")
    print("=" * 60)
    
    # 假设我们有5只股票，10个交易日
    dates = pd.date_range('2020-01-01', periods=10, freq='D')
    stocks = ['000001.SZ', '000002.SZ', '000300.SZ', '000858.SZ', '002415.SZ']
    
    print(f"时间范围: {dates[0].date()} 到 {dates[-1].date()}")
    print(f"股票池: {stocks}")
    print()
    
    # 方式1：逐天构建权重
    daily_weights = []
    
    # 第一天：等权重配置
    day1_weights = [0.2, 0.2, 0.2, 0.2, 0.2]
    daily_weights.append(day1_weights)
    print(f"第1天 ({dates[0].date()}):")
    for stock, weight in zip(stocks, day1_weights):
        print(f"  {stock}: {weight:.1%}")
    print(f"  权重总和: {sum(day1_weights):.1f}")
    print()
    
    # 第二天：调整权重（比如基于因子信号）
    day2_weights = [0.3, 0.25, 0.15, 0.2, 0.1]  # 重点配置前两只股票
    daily_weights.append(day2_weights)
    print(f"第2天 ({dates[1].date()}):")
    for stock, weight in zip(stocks, day2_weights):
        print(f"  {stock}: {weight:.1%}")
    print(f"  权重总和: {sum(day2_weights):.1f}")
    print()
    
    # 第三天：进一步调整
    day3_weights = [0.1, 0.1, 0.4, 0.3, 0.1]  # 转向第三、四只股票
    daily_weights.append(day3_weights)
    print(f"第3天 ({dates[2].date()}):")
    for stock, weight in zip(stocks, day3_weights):
        print(f"  {stock}: {weight:.1%}")
    print(f"  权重总和: {sum(day3_weights):.1f}")
    print()
    
    # 剩余天数：随机生成（模拟实际策略输出）
    np.random.seed(42)
    for i in range(3, len(dates)):
        # 生成随机权重并归一化
        raw_weights = np.random.exponential(1, len(stocks))
        normalized_weights = raw_weights / raw_weights.sum()
        daily_weights.append(normalized_weights.tolist())
    
    # 构建正确的DataFrame格式
    portfolio_weights = pd.DataFrame(
        daily_weights,      # 数据：每行是一天的权重分配
        index=dates,        # 行索引：日期
        columns=stocks      # 列索引：股票代码
    )
    
    print("=" * 60)
    print("完整的权重数据DataFrame：")
    print("=" * 60)
    print(portfolio_weights)
    print()
    
    # 验证数据格式
    print("=" * 60)
    print("数据格式验证：")
    print("=" * 60)
    print(f"DataFrame形状: {portfolio_weights.shape}")
    print(f"  - {portfolio_weights.shape[0]} 个交易日")
    print(f"  - {portfolio_weights.shape[1]} 只股票")
    print()
    
    print("每日权重和检查:")
    daily_sums = portfolio_weights.sum(axis=1)
    for date, total in daily_sums.items():
        print(f"  {date.date()}: {total:.6f}")
    print(f"权重和是否都等于1: {np.allclose(daily_sums, 1.0)}")
    print()
    
    print("权重变化示例:")
    print("第1天 → 第2天的变化:")
    weight_changes = portfolio_weights.iloc[1] - portfolio_weights.iloc[0]
    for stock, change in weight_changes.items():
        direction = "↑" if change > 0 else "↓" if change < 0 else "→"
        print(f"  {stock}: {change:+.1%} {direction}")
    
    return portfolio_weights

def demonstrate_backtest_usage(portfolio_weights):
    """演示如何将权重数据输入回测引擎"""
    
    print("\n" + "=" * 60)
    print("回测引擎调用示例：")
    print("=" * 60)
    
    # 模拟回测引擎调用
    print("# 1. 创建回测引擎")
    print("engine = BacktestEngine(")
    print("    initial_capital=1_000_000,  # 100万初始资金")
    print("    commission_rate=0.001,      # 0.1%手续费") 
    print("    slippage_rate=0.0005        # 0.05%滑点")
    print(")")
    print()
    
    print("# 2. 直接输入权重数据进行回测")
    print("result = engine.run_with_weights(portfolio_weights)")
    print()
    
    print("# 3. 回测引擎会自动处理：")
    print("  - 检测每日权重变化")
    print("  - 计算所需的买卖交易")
    print("  - 模拟交易执行和成本")
    print("  - 更新持仓和计算收益")
    print("  - 生成绩效报告")
    print()
    
    # 模拟回测结果
    print("# 4. 查看回测结果")
    print("print(f'总收益率: {result.total_return:.2%}')")
    print("print(f'年化收益率: {result.annual_return:.2%}')")
    print("print(f'夏普比率: {result.sharpe_ratio:.2f}')")
    print("print(f'最大回撤: {result.max_drawdown:.2%}')")

def show_real_world_example():
    """展示真实世界的使用场景"""
    
    print("\n" + "=" * 60)
    print("真实世界使用场景：")
    print("=" * 60)
    
    example_code = """
# 场景：基于多因子模型的权重生成
def generate_daily_weights(date):
    # 1. 获取当日因子数据
    factors = get_factor_data(date)
    
    # 2. 计算预期收益
    expected_returns = factor_model.predict(factors)
    
    # 3. 组合优化
    optimizer_result = portfolio_optimizer.optimize(
        expected_returns=expected_returns,
        constraints={'max_weight': 0.1}
    )
    
    # 4. 返回权重向量
    return optimizer_result['weights']  # pd.Series

# 主流程
dates = get_trading_dates('2020-01-01', '2023-12-31')
all_weights = []

for date in dates:
    daily_weights = generate_daily_weights(date)
    all_weights.append(daily_weights)

# 构建权重DataFrame（这就是输入回测引擎的格式）
portfolio_weights = pd.DataFrame(all_weights, index=dates)

# 回测
result = BacktestEngine().run_with_weights(portfolio_weights)
"""
    
    print(example_code)

if __name__ == "__main__":
    # 创建正确格式的权重数据
    weights = create_correct_weights_format()
    
    # 演示回测调用
    demonstrate_backtest_usage(weights)
    
    # 展示实际使用场景
    show_real_world_example()
    
    print("\n" + "🎯" + " " * 55)
    print("关键理解：")
    print("  - 每一行 = 一天的权重分配方案")
    print("  - 每一列 = 一只股票")  
    print("  - 每行权重和 = 1.0 (100%)")
    print("  - 行与行之间的差异 = 策略调整")
    print("=" * 60)