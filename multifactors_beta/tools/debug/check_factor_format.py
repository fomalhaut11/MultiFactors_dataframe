import pickle
import pandas as pd
from pathlib import Path

# 检查BP因子格式
bp_path = Path(r'E:\Documents\PythonProject\StockProject\MultiFactors\MultiFactors_1.0\BP.pkl')
if bp_path.exists():
    with open(bp_path, 'rb') as f:
        bp_data = pickle.load(f)
    print("BP因子格式:")
    print(f"  形状: {bp_data.shape}")
    print(f"  数据类型: {type(bp_data)}")
    if hasattr(bp_data, 'columns'):
        print(f"  列名样本: {list(bp_data.columns[:5])}")
    if hasattr(bp_data, 'index'):
        print(f"  索引样本: {list(bp_data.index[:5])}")
        print(f"  索引类型: {type(bp_data.index[0])}")
    print()

# 检查收益率数据格式
return_path = Path(r'E:\Documents\PythonProject\StockProject\StockData\SingleFactorTestData\LogReturn_daily_o2o.pkl')
if return_path.exists():
    with open(return_path, 'rb') as f:
        return_data = pickle.load(f)
    print("收益率数据格式:")
    print(f"  形状: {return_data.shape}")
    if hasattr(return_data, 'columns'):
        print(f"  列名样本: {list(return_data.columns[:5])}")
    if hasattr(return_data, 'index'):
        print(f"  索引样本: {list(return_data.index[:5])}")
        print(f"  索引类型: {type(return_data.index[0])}")
    print()
    
    # 查找公共股票
    if hasattr(bp_data, 'columns') and hasattr(return_data, 'columns'):
        common_stocks = set(bp_data.columns) & set(return_data.columns)
        print(f"公共股票数量: {len(common_stocks)}")
        if common_stocks:
            print(f"公共股票样本: {list(common_stocks)[:5]}")

# 检查SUE因子格式
sue_path = Path(r'E:\Documents\PythonProject\StockProject\StockData\SingleFactorTestData\SUE.pkl')
if sue_path.exists():
    with open(sue_path, 'rb') as f:
        sue_data = pickle.load(f)
    print("\nSUE因子格式:")
    print(f"  形状: {sue_data.shape}")
    if hasattr(sue_data, 'columns'):
        print(f"  列名样本: {list(sue_data.columns[:5])}")
    if hasattr(sue_data, 'index'):
        print(f"  索引样本: {list(sue_data.index[:5])}")
        print(f"  索引类型: {type(sue_data.index[0])}")