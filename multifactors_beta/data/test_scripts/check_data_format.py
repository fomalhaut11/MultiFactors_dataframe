"""检查数据格式脚本"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent))

# 数据路径
DATA_ROOT = Path("E:/Documents/PythonProject/StockProject/StockData")
AUXILIARY_PATH = Path(__file__).parent.parent / "auxiliary"

def check_data_formats():
    """检查各种数据格式"""

    print("=" * 60)
    print("数据格式检查")
    print("=" * 60)

    # 1. 财务数据
    print("\n1. 财务数据格式:")
    financial_data = pd.read_pickle(AUXILIARY_PATH / "FinancialData_unified.pkl")
    print(f"   形状: {financial_data.shape}")
    print(f"   索引: {financial_data.index.names}")
    print(f"   列数: {len(financial_data.columns)}")
    print(f"   部分列: {list(financial_data.columns[:10])}")

    # 检查是否有净利润和净资产等关键字段
    key_fields = ['NET_PROFIT', 'TOTAL_EQUITY', 'TOTAL_ASSETS', 'TOTAL_LIAB']
    available_fields = [f for f in key_fields if f in financial_data.columns]
    print(f"   关键字段: {available_fields}")

    # 2. 市值数据
    print("\n2. 市值数据格式:")
    market_cap = pd.read_pickle(AUXILIARY_PATH / "MarketCap.pkl")
    print(f"   形状: {market_cap.shape}")
    print(f"   索引: {market_cap.index.names if hasattr(market_cap, 'index') else 'Series'}")
    print(f"   数据类型: {type(market_cap).__name__}")

    # 3. 股票信息
    print("\n3. 股票基本信息:")
    stock_info = pd.read_pickle(AUXILIARY_PATH / "StockInfo.pkl")
    print(f"   形状: {stock_info.shape}")
    print(f"   列: {list(stock_info.columns)}")

    # 4. 行业分类数据
    print("\n4. 行业分类数据:")
    classification_path = DATA_ROOT / "Classificationdata" / "classification_one_hot.pkl"
    classification = pd.read_pickle(classification_path)
    print(f"   形状: {classification.shape}")
    print(f"   索引: {classification.index.names}")
    print(f"   列数(行业数): {len(classification.columns)}")
    print(f"   部分行业: {list(classification.columns[:5])}")

    # 5. 价格数据（如果存在）
    price_path = DATA_ROOT / "Price.pkl"
    if price_path.exists():
        print("\n5. 价格数据:")
        price_data = pd.read_pickle(price_path)
        print(f"   形状: {price_data.shape}")
        print(f"   索引: {price_data.index.names}")
        print(f"   列: {list(price_data.columns) if hasattr(price_data, 'columns') else 'Series'}")

    # 6. 交易日期
    print("\n6. 交易日期数据:")
    trading_dates = pd.read_pickle(AUXILIARY_PATH / "TradingDates.pkl")
    print(f"   类型: {type(trading_dates).__name__}")
    print(f"   长度: {len(trading_dates)}")
    if isinstance(trading_dates, pd.Series):
        print(f"   起止日期: {trading_dates.iloc[0]} - {trading_dates.iloc[-1]}")
    else:
        print(f"   起止日期: {trading_dates[0]} - {trading_dates[-1]}")

    print("\n" + "=" * 60)

if __name__ == "__main__":
    check_data_formats()