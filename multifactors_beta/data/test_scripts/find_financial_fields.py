"""查找财务数据中的净利润和净资产相关字段"""
import sys
import pandas as pd
from pathlib import Path

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent))

# 数据路径
AUXILIARY_PATH = Path(__file__).parent.parent / "auxiliary"

def find_financial_fields():
    """查找财务数据中的关键字段"""

    print("=" * 60)
    print("查找财务数据关键字段")
    print("=" * 60)

    # 加载财务数据
    financial_data = pd.read_pickle(AUXILIARY_PATH / "FinancialData_unified.pkl")

    # 获取所有字段名
    all_fields = list(financial_data.columns)

    # 查找净利润相关字段
    print("\n净利润相关字段:")
    profit_fields = [f for f in all_fields if 'PROFIT' in f.upper() or 'NET' in f.upper() or '利润' in f]
    for field in profit_fields[:20]:  # 只显示前20个
        print(f"  - {field}")

    # 查找净资产/所有者权益相关字段
    print("\n净资产/所有者权益相关字段:")
    equity_fields = [f for f in all_fields if 'EQUITY' in f.upper() or 'SHAREHOLD' in f.upper() or '权益' in f or '资产' in f]
    for field in equity_fields[:20]:
        print(f"  - {field}")

    # 查找总资产相关字段
    print("\n总资产相关字段:")
    asset_fields = [f for f in all_fields if 'ASSET' in f.upper() or 'TOT' in f.upper() and 'ASSET' in f.upper()]
    for field in asset_fields[:20]:
        print(f"  - {field}")

    # 查找负债相关字段
    print("\n负债相关字段:")
    liab_fields = [f for f in all_fields if 'LIAB' in f.upper() or '负债' in f]
    for field in liab_fields[:20]:
        print(f"  - {field}")

    # 查找收入相关字段
    print("\n收入相关字段:")
    revenue_fields = [f for f in all_fields if 'REV' in f.upper() or 'INCOME' in f.upper() or '收入' in f]
    for field in revenue_fields[:20]:
        print(f"  - {field}")

    # 打印所有字段（用于详细检查）
    print("\n" + "=" * 60)
    print("所有字段列表（前50个）:")
    for i, field in enumerate(all_fields[:50], 1):
        print(f"{i:3}. {field}")

    print(f"\n总字段数: {len(all_fields)}")

if __name__ == "__main__":
    find_financial_fields()