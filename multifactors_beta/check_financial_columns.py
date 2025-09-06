#!/usr/bin/env python3
"""
检查财务数据中的列名
"""
import pandas as pd
from pathlib import Path
import sys

# 添加项目根目录到系统路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import get_config

def check_columns():
    """检查财务数据列名"""
    print("检查财务数据列名...")
    
    # 加载财务数据
    data_root = get_config('main.paths.data_root')
    financial_file = Path(data_root) / 'auxiliary' / 'FinancialData_unified.pkl'
    
    if not financial_file.exists():
        print(f"财务数据文件不存在: {financial_file}")
        return
        
    financial_data = pd.read_pickle(financial_file)
    print(f"财务数据形状: {financial_data.shape}")
    
    # 查找包含"净利润"的列
    profit_columns = [col for col in financial_data.columns if '净利润' in col]
    print(f"\n包含'净利润'的列 ({len(profit_columns)}个):")
    for col in sorted(profit_columns):
        print(f"  {col}")
    
    # 查找包含"扣除"的列
    deduct_columns = [col for col in financial_data.columns if '扣除' in col]
    print(f"\n包含'扣除'的列 ({len(deduct_columns)}个):")
    for col in sorted(deduct_columns):
        print(f"  {col}")
    
    # 查找包含"DEDUCTED"的列
    deducted_columns = [col for col in financial_data.columns if 'DEDUCTED' in col.upper()]
    print(f"\n包含'DEDUCTED'的列 ({len(deducted_columns)}个):")
    for col in sorted(deducted_columns):
        print(f"  {col}")
    
    # 检查是否有单季数据
    quarter_columns = [col for col in financial_data.columns if '单季' in col]
    print(f"\n包含'单季'的列 ({len(quarter_columns)}个):")
    for col in sorted(quarter_columns)[:10]:  # 显示前10个
        print(f"  {col}")
    
    if len(quarter_columns) > 10:
        print(f"  ... 还有 {len(quarter_columns) - 10} 个")
    
    # 显示所有列名（前20个）
    print(f"\n所有列名（前20个）:")
    for i, col in enumerate(financial_data.columns[:20]):
        print(f"  {i+1}. {col}")
    
    if len(financial_data.columns) > 20:
        print(f"  ... 还有 {len(financial_data.columns) - 20} 个列")

if __name__ == "__main__":
    check_columns()