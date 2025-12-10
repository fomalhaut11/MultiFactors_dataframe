"""
测试TTM计算函数的输出
"""

import sys
import pandas as pd
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from factors.generators import calculate_ttm

def test_ttm():
    print("=" * 60)
    print("测试TTM计算")
    print("=" * 60)

    # 加载财务数据
    auxiliary_path = Path(__file__).parent.parent / "auxiliary"
    financial_data = pd.read_pickle(auxiliary_path / "FinancialData_unified.pkl")

    # 选择测试字段
    test_fields = ['NET_PROFIT_IS', 'd_quarter', 'd_year']
    test_df = financial_data[test_fields].copy()

    print(f"\n输入DataFrame形状: {test_df.shape}")
    print(f"输入列名: {list(test_df.columns)}")

    # 计算TTM
    print("\n计算TTM...")
    ttm_result = calculate_ttm(test_df)

    print(f"\n输出DataFrame形状: {ttm_result.shape}")
    print(f"输出列名: {list(ttm_result.columns)}")

    # 显示前5行
    print("\nTTM结果前5行:")
    print(ttm_result.head())

    # 检查是否有新的列名
    print("\n新增/改变的列:")
    new_cols = set(ttm_result.columns) - set(test_df.columns)
    print(f"新增列: {new_cols}")

    missing_cols = set(test_df.columns) - set(ttm_result.columns)
    print(f"消失的列: {missing_cols}")

if __name__ == "__main__":
    test_ttm()