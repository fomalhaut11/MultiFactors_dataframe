#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""测试DataAdapter"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

result_file = "test_adapter_result.txt"

try:
    from factors.utils import DataAdapter
    
    with open(result_file, 'w') as f:
        f.write("Testing DataAdapter...\n")
        
        # 测试加载数据
        data_path = r"E:\Documents\PythonProject\StockProject\StockData"
        prepared_data = DataAdapter.load_and_prepare_data(data_path)
        
        f.write(f"Success!\n")
        f.write(f"Keys: {list(prepared_data.keys())}\n")
        f.write(f"Financial data shape: {prepared_data['financial_data'].shape}\n")
        
except Exception as e:
    with open(result_file, 'w') as f:
        f.write(f"Error: {str(e)}\n")
        import traceback
        f.write(traceback.format_exc())

print(f"Test complete. Check {result_file}")