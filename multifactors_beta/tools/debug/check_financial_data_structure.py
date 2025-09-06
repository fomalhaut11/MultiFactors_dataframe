#!/usr/bin/env python3
"""
检查财务数据结构
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.fetcher.data_fetcher import DataFetcherManager

def main():
    print("检查财务数据结构...")
    
    # 获取数据
    fetcher = DataFetcherManager()
    
    print("获取财务数据...")
    financial_tables = fetcher.fetch_data('stock', 'financial')
    
    print(f"财务数据类型: {type(financial_tables)}")
    
    if isinstance(financial_tables, dict):
        print(f"包含的表: {list(financial_tables.keys())}")
        
        for table_name, table_data in financial_tables.items():
            print(f"\n=== {table_name} ===")
            print(f"形状: {table_data.shape}")
            print(f"列名 (前20个): {list(table_data.columns[:20])}")
            
            # 查找相关列名
            relevant_cols = []
            for col in table_data.columns:
                if any(keyword in col.upper() for keyword in ['FIN_EXP', 'DEPR', 'CASH_RECP', '财务费用', '折旧', '销售']):
                    relevant_cols.append(col)
            
            if relevant_cols:
                print(f"相关列名: {relevant_cols}")
            else:
                print("未找到相关列名")
    
    else:
        print(f"财务数据形状: {financial_tables.shape}")
        print(f"前20列: {list(financial_tables.columns[:20])}")

if __name__ == "__main__":
    main()