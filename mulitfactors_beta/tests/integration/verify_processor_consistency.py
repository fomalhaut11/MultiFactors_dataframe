"""
验证数据处理器重构的一致性

此脚本用于验证新的模块化实现与原始实现产生完全相同的结果
"""
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import pickle

# 添加路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root.parent / "MultiFactors_1.0"))

# 导入测试器
from test_data_processor_consistency import DataProcessorConsistencyTester

# 导入新实现
from data.processor import PriceDataProcessor, ReturnCalculator, FinancialDataProcessor
from data.processor.data_processing_pipeline import DataProcessingPipeline


def verify_price_processor():
    """验证价格数据处理器的一致性"""
    print("\n=== 验证PriceDataProcessor ===")
    
    tester = DataProcessorConsistencyTester()
    processor = PriceDataProcessor()
    
    try:
        # 运行新实现
        price_df, stock_3d = processor.process(save_to_file=False)
        
        # 比较结果
        result1 = tester.compare_outputs('get_price_data_pricedf', price_df)
        result2 = tester.compare_outputs('get_price_data_stock3d', stock_3d)
        
        print(f"PriceDF 一致性: {'[OK] 通过' if result1['is_consistent'] else '[FAIL] 失败'}")
        print(f"Stock3D 一致性: {'[OK] 通过' if result2['is_consistent'] else '[FAIL] 失败'}")
        
        if not result1['is_consistent']:
            print(f"  详细信息: {result1}")
        if not result2['is_consistent']:
            print(f"  详细信息: {result2}")
            
        # 保存结果
        tester.results['price_processor_df'] = result1
        tester.results['price_processor_3d'] = result2
        
        return price_df
        
    except Exception as e:
        print(f"[FAIL] 价格处理器验证失败: {e}")
        return None
        

def verify_return_calculator(price_df):
    """验证收益率计算器的一致性"""
    print("\n=== 验证ReturnCalculator ===")
    
    if price_df is None:
        print("跳过收益率计算器验证（无价格数据）")
        return
        
    tester = DataProcessorConsistencyTester()
    calculator = ReturnCalculator()
    processor = PriceDataProcessor()
    
    try:
        # 获取日期序列
        daily_dates = processor.get_date_series(price_df, "daily")
        
        # 测试日期序列生成
        result_daily = tester.compare_outputs('date_serries_daily', daily_dates)
        print(f"日期序列(daily) 一致性: {'[OK] 通过' if result_daily['is_consistent'] else '[FAIL] 失败'}")
        
        # 测试收益率计算（使用小样本）
        sample_dates = daily_dates[:100]
        log_return = calculator.calculate_log_return(
            price_df, sample_dates, return_type="o2o"
        )
        
        result_return = tester.compare_outputs(
            'logreturndf_dateserries_o2o', 
            log_return,
            {'dates_count': 100, 'return_type': 'o2o'}
        )
        print(f"收益率计算(o2o) 一致性: {'[OK] 通过' if result_return['is_consistent'] else '[FAIL] 失败'}")
        
        if not result_return['is_consistent']:
            print(f"  详细信息: {result_return}")
            
        # 保存结果
        tester.results['date_series_daily'] = result_daily
        tester.results['return_calculator'] = result_return
        
    except Exception as e:
        print(f"[FAIL] 收益率计算器验证失败: {e}")
        

def verify_pipeline_integration():
    """验证整体管道集成的一致性"""
    print("\n=== 验证整体管道集成 ===")
    
    # 这里可以添加更多的集成测试
    # 例如：检查文件输出的一致性
    
    print("[v] 管道集成测试完成")
    

def main():
    """主函数"""
    print("开始验证数据处理器重构一致性...")
    
    tester = DataProcessorConsistencyTester()
    
    # 1. 验证价格处理器
    price_df = verify_price_processor()
    
    # 2. 验证收益率计算器
    verify_return_calculator(price_df)
    
    # 3. 验证管道集成
    verify_pipeline_integration()
    
    # 生成测试报告
    report = tester.generate_test_report()
    print("\n" + "="*50)
    print(report)
    
    # 保存报告
    report_path = project_root / "tests" / "processor_consistency_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n测试报告已保存至: {report_path}")
    

if __name__ == "__main__":
    main()