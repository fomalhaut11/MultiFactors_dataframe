"""
调试财务数据结构
"""

import sys
import pandas as pd
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 数据路径
AUXILIARY_PATH = Path(__file__).parent.parent / "auxiliary"

def debug_financial_data():
    """调试财务数据结构"""

    print("=" * 60)
    print("调试财务数据结构")
    print("=" * 60)

    # 加载财务数据
    financial_data = pd.read_pickle(AUXILIARY_PATH / "FinancialData_unified.pkl")

    print(f"\n财务数据类型: {type(financial_data)}")
    print(f"形状: {financial_data.shape}")
    print(f"索引: {financial_data.index.names}")

    # 检查是否存在NET_PROFIT_IS字段
    if 'NET_PROFIT_IS' in financial_data.columns:
        net_profit = financial_data['NET_PROFIT_IS']
        print(f"\nNET_PROFIT_IS字段类型: {type(net_profit)}")
        print(f"NET_PROFIT_IS形状: {net_profit.shape}")
        print(f"NET_PROFIT_IS索引: {net_profit.index.names}")

        # 显示部分数据
        print("\n前5条数据:")
        print(net_profit.head())

        # 检查数据类型
        print(f"\n数据dtype: {net_profit.dtype}")
        print(f"非空值数量: {net_profit.notna().sum()}")
        print(f"空值数量: {net_profit.isna().sum()}")

        # 测试转换为DataFrame
        print("\n测试转换为DataFrame:")
        net_profit_df = pd.DataFrame({'NET_PROFIT_IS': net_profit})
        print(f"DataFrame形状: {net_profit_df.shape}")
        print(f"DataFrame索引: {net_profit_df.index.names}")

    else:
        print("\n⚠️ NET_PROFIT_IS字段不存在")
        print("可用字段（前20个）:")
        for i, col in enumerate(financial_data.columns[:20], 1):
            print(f"  {i:2}. {col}")

if __name__ == "__main__":
    debug_financial_data()