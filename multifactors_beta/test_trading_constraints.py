"""
测试交易约束功能的简单示例
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 导入回测模块
from backtest import BacktestEngine, create_trading_constraints

def create_sample_data():
    """创建示例数据"""
    # 时间范围
    dates = pd.date_range('2023-01-01', '2023-01-10', freq='D')
    stocks = ['000001.SZ', '300001.SZ', '688001.SH']  # 主板、创业板、科创板各一只
    
    # 权重数据
    np.random.seed(42)
    weights_data = pd.DataFrame(
        np.random.dirichlet([1, 1, 1], len(dates)),  # 确保每行和为1
        index=dates,
        columns=stocks
    )
    
    # 价格数据
    base_prices = {'000001.SZ': 10.0, '300001.SZ': 20.0, '688001.SH': 50.0}
    price_data = pd.DataFrame(index=dates, columns=stocks)
    
    for stock in stocks:
        # 生成随机价格序列（模拟正常交易日）
        returns = np.random.normal(0, 0.02, len(dates))
        prices = [base_prices[stock]]
        for r in returns[1:]:
            prices.append(prices[-1] * (1 + r))
        price_data[stock] = prices
    
    # 市场数据（模拟停牌和涨跌停情况）
    market_data = {}
    
    # 开盘价
    market_data['open'] = price_data.copy()
    
    # 最高价和最低价
    market_data['high'] = price_data * 1.02
    market_data['low'] = price_data * 0.98
    
    # 收盘价
    market_data['close'] = price_data.copy()
    
    # 成交量（模拟停牌：第3天000001.SZ停牌）
    market_data['volume'] = pd.DataFrame(
        np.random.randint(1000000, 10000000, (len(dates), len(stocks))),
        index=dates,
        columns=stocks
    )
    market_data['volume'].loc[dates[2], '000001.SZ'] = 0  # 模拟停牌
    
    # 昨日收盘价
    market_data['prev_close'] = price_data.shift(1).fillna(method='bfill')
    
    # 模拟涨跌停：第5天300001.SZ涨停
    limit_up_date = dates[4]
    prev_close = market_data['prev_close'].loc[limit_up_date, '300001.SZ']
    market_data['open'].loc[limit_up_date, '300001.SZ'] = prev_close * 1.20  # 创业板涨停20%
    
    # 模拟涨跌停：第6天688001.SH跌停
    limit_down_date = dates[5]
    prev_close = market_data['prev_close'].loc[limit_down_date, '688001.SH']
    market_data['open'].loc[limit_down_date, '688001.SH'] = prev_close * 0.80  # 科创板跌停20%
    
    return weights_data, price_data, market_data

def test_trading_constraints():
    """测试交易约束功能"""
    print("=== 测试交易约束功能 ===")
    
    # 1. 创建示例数据
    weights_data, price_data, market_data = create_sample_data()
    
    print(f"测试数据:")
    print(f"时间范围: {weights_data.index[0]} 到 {weights_data.index[-1]}")
    print(f"股票: {list(weights_data.columns)}")
    print(f"权重数据形状: {weights_data.shape}")
    print(f"价格数据形状: {price_data.shape}")
    print()
    
    # 2. 创建交易约束检查器
    trading_constraints = create_trading_constraints('china_a_share')
    
    # 3. 不使用交易约束的回测
    print("--- 无交易约束回测 ---")
    engine_no_constraints = BacktestEngine(
        initial_capital=1000000,
        trading_constraints=None
    )
    
    result_no_constraints = engine_no_constraints.run_with_weights(
        weights_data=weights_data,
        price_data=price_data,
        market_data=market_data
    )
    
    print(f"无约束回测结果:")
    print(f"总收益率: {result_no_constraints.get_total_return():.4f}")
    print(f"最终价值: {result_no_constraints.get_final_portfolio_value():.2f}")
    print()
    
    # 4. 使用交易约束的回测
    print("--- 有交易约束回测 ---")
    engine_with_constraints = BacktestEngine(
        initial_capital=1000000,
        trading_constraints=trading_constraints
    )
    
    result_with_constraints = engine_with_constraints.run_with_weights(
        weights_data=weights_data,
        price_data=price_data,
        market_data=market_data
    )
    
    print(f"有约束回测结果:")
    print(f"总收益率: {result_with_constraints.get_total_return():.4f}")
    print(f"最终价值: {result_with_constraints.get_final_portfolio_value():.2f}")
    print()
    
    # 5. 详细检查交易约束效果
    print("--- 交易约束检查详情 ---")
    for i, date in enumerate(weights_data.index[:6]):  # 只检查前6天
        stocks = list(weights_data.columns)
        tradable_status = trading_constraints.check_trading_availability(
            current_date=date,
            stocks=stocks,
            market_data=market_data
        )
        
        untradable_stocks = [stock for stock, tradable in tradable_status.items() if not tradable]
        if untradable_stocks:
            print(f"{date.date()}: 不可交易股票 {untradable_stocks}")
        else:
            print(f"{date.date()}: 所有股票可交易")
    
    print("\n=== 测试完成 ===")

if __name__ == "__main__":
    test_trading_constraints()