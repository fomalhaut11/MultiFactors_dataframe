"""
小规模数据处理验证测试

测试新的数据处理模块在小数据集上的功能
"""
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import time
from datetime import datetime

# 设置输出编码
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.processor import PriceDataProcessor, ReturnCalculator
from data.processor.data_processing_pipeline import DataProcessingPipeline


def test_price_processor():
    """测试价格数据处理器"""
    print("\n=== 测试价格数据处理器 ===")
    
    try:
        processor = PriceDataProcessor()
        
        # 验证配置加载
        print("√ 配置加载成功")
        print(f"  数据路径: {processor.data_root}")
        print(f"  价格文件: {processor.price_file_path}")
        
        # 验证输入文件
        if processor.validate_input():
            print("√ 输入文件验证通过")
        else:
            print("× 输入文件验证失败")
            return False
            
        # 测试数据加载（不执行完整处理）
        print("\n正在加载价格数据...")
        price_df = pd.read_pickle(processor.price_file_path)
        print(f"√ 价格数据加载成功")
        print(f"  数据形状: {price_df.shape}")
        print(f"  日期范围: {price_df.index.get_level_values(0).min()} 至 {price_df.index.get_level_values(0).max()}")
        print(f"  股票数量: {price_df.index.get_level_values(1).nunique()}")
        
        # 测试日期序列生成
        print("\n测试日期序列生成...")
        daily_dates = processor.get_date_series(price_df, "daily")
        weekly_dates = processor.get_date_series(price_df, "weekly")
        monthly_dates = processor.get_date_series(price_df, "monthly")
        
        print(f"√ 日期序列生成成功")
        print(f"  日频日期数: {len(daily_dates)}")
        print(f"  周频日期数: {len(weekly_dates)}")
        print(f"  月频日期数: {len(monthly_dates)}")
        
        return True, price_df
        
    except Exception as e:
        print(f"× 价格处理器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_return_calculator(price_df):
    """测试收益率计算器"""
    print("\n=== 测试收益率计算器 ===")
    
    if price_df is None:
        print("跳过收益率计算器测试（无价格数据）")
        return False
        
    try:
        calculator = ReturnCalculator()
        processor = PriceDataProcessor()
        
        # 获取小样本数据进行测试
        print("\n准备测试数据...")
        daily_dates = processor.get_date_series(price_df, "daily")
        
        # 使用最近30个交易日的数据
        test_dates = daily_dates[-31:]  # 需要31个日期来计算30个收益率
        test_price_df = price_df.loc[test_dates]
        
        print(f"√ 测试数据准备完成")
        print(f"  测试日期数: {len(test_dates)}")
        print(f"  测试数据形状: {test_price_df.shape}")
        
        # 测试收益率计算
        print("\n计算对数收益率...")
        start_time = time.time()
        
        log_return = calculator.calculate_log_return(
            test_price_df, 
            test_dates, 
            return_type="o2o"
        )
        
        elapsed_time = time.time() - start_time
        print(f"√ 收益率计算成功")
        print(f"  计算耗时: {elapsed_time:.2f}秒")
        print(f"  收益率数据形状: {log_return.shape}")
        print(f"  非空值比例: {(~log_return['LogReturn'].isna()).mean():.2%}")
        
        # 测试N天滚动收益率
        print("\n计算5天滚动收益率...")
        rolling_5d = calculator.calculate_n_days_return(log_return, lag=5)
        print(f"√ 5天滚动收益率计算成功")
        print(f"  数据形状: {rolling_5d.shape}")
        
        return True
        
    except Exception as e:
        print(f"× 收益率计算器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_integrity():
    """测试数据完整性检查"""
    print("\n=== 测试数据完整性 ===")
    
    try:
        # 检查必要的数据文件
        data_root = Path(r"E:\Documents\PythonProject\StockProject\StockData")
        
        files_to_check = [
            ("Price.pkl", "价格数据"),
            ("TradableDF.pkl", "可交易状态"),
            ("financial_v2.h5", "财务数据")
        ]
        
        all_exist = True
        for filename, description in files_to_check:
            filepath = data_root / filename
            if filepath.exists():
                size_mb = filepath.stat().st_size / (1024 * 1024)
                print(f"√ {description}: {filename} ({size_mb:.1f} MB)")
            else:
                print(f"× {description}: {filename} 不存在")
                all_exist = False
                
        return all_exist
        
    except Exception as e:
        print(f"× 数据完整性检查失败: {e}")
        return False


def main():
    """主测试函数"""
    print("="*60)
    print("小规模数据处理验证测试")
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # 1. 检查数据完整性
    if not test_data_integrity():
        print("\n× 数据文件不完整，无法继续测试")
        return False
        
    # 2. 测试价格处理器
    success, price_df = test_price_processor()
    if not success:
        print("\n× 价格处理器测试失败")
        return False
        
    # 3. 测试收益率计算器
    if not test_return_calculator(price_df):
        print("\n× 收益率计算器测试失败")
        return False
        
    print("\n" + "="*60)
    print("√ 所有测试通过！数据处理模块功能正常")
    print("="*60)
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)