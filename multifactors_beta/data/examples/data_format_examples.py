#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据格式约定和验证使用示例

演示如何使用数据格式验证和转换功能，确保data模块到factors模块的数据传递标准化

Author: MultiFactors Team
Date: 2025-08-21
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from datetime import datetime
import logging

# 导入数据格式相关模块
from data.schemas import (
    DataValidator, DataConverter, DataQualityChecker,
    DataSchemas, validate_price_data, validate_financial_data, 
    validate_factor_format, convert_to_factor_format
)
from data.data_bridge import (
    DataBridge, get_data_bridge, get_factor_data, validate_data_pipeline
)

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def example_1_basic_validation():
    """示例1: 基础数据格式验证"""
    print("\n" + "="*60)
    print("示例1: 基础数据格式验证")
    print("="*60)
    
    # 创建示例价格数据
    price_data = pd.DataFrame({
        'code': ['000001', '000002', '000001', '000002'],
        'tradingday': [20241201, 20241201, 20241202, 20241202],
        'o': [10.5, 15.2, 10.6, 15.3],
        'h': [10.8, 15.5, 10.9, 15.6],
        'l': [10.4, 15.1, 10.5, 15.2],
        'c': [10.7, 15.35, 10.8, 15.4],
        'v': [1000000, 800000, 1200000, 900000],
        'amt': [10700000, 12280000, 12960000, 13860000],
        'adjfactor': [1.0, 1.0, 1.0, 1.0]
    })
    
    print("📊 示例价格数据:")
    print(price_data)
    
    # 验证数据格式
    print("\n🔍 数据格式验证:")
    is_valid, errors = validate_price_data(price_data, strict=False)
    
    if is_valid:
        print("✅ 价格数据格式验证通过")
    else:
        print("❌ 价格数据格式验证失败:")
        for error in errors:
            print(f"  • {error}")
    
    # 转换为因子格式
    print("\n🔄 转换为标准因子格式:")
    try:
        factor_series = convert_to_factor_format(
            price_data, 
            value_col='c',
            date_col='tradingday',
            stock_col='code'
        )
        
        print(f"因子数据形状: {factor_series.shape}")
        print(f"索引名称: {factor_series.index.names}")
        print(f"数据类型: {factor_series.dtype}")
        
        # 验证因子格式
        is_factor_valid, factor_errors = validate_factor_format(factor_series)
        
        if is_factor_valid:
            print("✅ 因子格式验证通过")
            print("\n因子数据预览:")
            print(factor_series.head())
        else:
            print("❌ 因子格式验证失败:")
            for error in factor_errors:
                print(f"  • {error}")
                
    except Exception as e:
        print(f"❌ 转换失败: {e}")


def example_2_data_bridge_usage():
    """示例2: 数据桥接器使用"""
    print("\n" + "="*60)
    print("示例2: 数据桥接器使用")
    print("="*60)
    
    try:
        # 获取数据桥接器
        bridge = get_data_bridge()
        
        # 打印数据状态
        print("📊 当前数据状态:")
        bridge.print_data_status()
        
        # 验证数据管道
        print("\n🔍 验证数据管道:")
        pipeline_valid = validate_data_pipeline()
        
        if pipeline_valid:
            print("\n✅ 数据管道验证通过，开始获取数据示例")
            
            # 获取财务数据（如果存在）
            try:
                financial_data = bridge.get_financial_data()
                print(f"\n📈 财务数据: {financial_data.shape}")
                print(f"财务数据字段 (前10个): {list(financial_data.columns)[:10]}")
                
                # 尝试转换财务数据为因子格式
                if 'NET_PROFIT' in financial_data.columns:
                    profit_factor = bridge.financial_to_factor('NET_PROFIT')
                    print(f"净利润因子: {profit_factor.shape}")
                    print(f"净利润因子预览:\n{profit_factor.head()}")
                    
            except FileNotFoundError:
                print("⚠️ 财务数据文件不存在，请先运行 data/prepare_auxiliary_data.py")
            
            # 获取价格数据（如果可用）
            try:
                price_data = bridge.get_price_data(begin_date=20241201, end_date=20241202)
                print(f"\n💰 价格数据: {price_data.shape}")
                
                # 转换为因子格式
                close_factor = bridge.price_to_factor('c', begin_date=20241201, end_date=20241202)
                print(f"收盘价因子: {close_factor.shape}")
                print(f"收盘价因子预览:\n{close_factor.head()}")
                
            except Exception as e:
                print(f"⚠️ 获取价格数据失败: {e}")
        
    except Exception as e:
        print(f"❌ 数据桥接器使用失败: {e}")
        logger.error(f"DataBridge error: {e}", exc_info=True)


def example_3_data_quality_check():
    """示例3: 数据质量检查"""
    print("\n" + "="*60)
    print("示例3: 数据质量检查")
    print("="*60)
    
    # 创建有问题的示例数据
    problematic_data = pd.DataFrame({
        'code': ['000001', '000002', '000001', '000002', '000001'],  # 重复数据
        'tradingday': [20241201, 20241201, 20241202, 20241202, 20241201],
        'c': [10.7, -15.35, np.nan, 15.4, 10.7],  # 负值和缺失值
        'adjfactor': [1.0, 1.0, 0.0, 1.0, 1.0],  # 零值
        'v': [1000000, 800000, 1200000, 900000, 1000000]
    })
    
    print("📊 问题数据示例:")
    print(problematic_data)
    
    # 数据质量检查
    print("\n🔍 数据质量检查:")
    report = DataQualityChecker.generate_quality_report(
        problematic_data, DataSchemas.PRICE_DATA
    )
    
    # 打印质量报告
    DataQualityChecker.print_quality_report(report)
    
    # 详细问题分析
    issues = report['issues']
    if issues:
        print(f"\n⚠️ 发现的具体问题:")
        for i, issue in enumerate(issues, 1):
            print(f"{i}. 类型: {issue['type']}")
            print(f"   字段: {issue['column']}")
            print(f"   描述: {issue['description']}")
    
    # 数据格式验证
    print(f"\n🔍 格式验证:")
    is_valid, errors = validate_price_data(problematic_data, strict=False)
    
    if not is_valid:
        print("❌ 发现格式问题:")
        for error in errors:
            print(f"  • {error}")
    else:
        print("✅ 基础格式验证通过")


def example_4_custom_validation():
    """示例4: 自定义数据验证"""
    print("\n" + "="*60)
    print("示例4: 自定义数据验证和转换")
    print("="*60)
    
    # 创建自定义格式数据
    custom_data = pd.DataFrame({
        'stock_id': ['000001', '000002', '000001', '000002'],
        'trade_date': ['2024-12-01', '2024-12-01', '2024-12-02', '2024-12-02'],
        'close_price': [10.7, 15.35, 10.8, 15.4],
        'volume': [1000000, 800000, 1200000, 900000]
    })
    
    print("📊 自定义格式数据:")
    print(custom_data)
    
    # 自定义转换逻辑
    print("\n🔄 自定义格式转换:")
    
    try:
        # 转换日期格式
        custom_data['trade_date'] = pd.to_datetime(custom_data['trade_date'])
        
        # 转换为因子格式
        factor_series = DataConverter.price_to_factor_format(
            custom_data,
            value_column='close_price',
            date_column='trade_date',
            stock_column='stock_id'
        )
        
        print(f"转换后因子形状: {factor_series.shape}")
        print(f"因子数据:\n{factor_series}")
        
        # 验证转换结果
        is_valid, errors = validate_factor_format(factor_series)
        
        if is_valid:
            print("✅ 自定义转换验证通过")
        else:
            print("❌ 自定义转换验证失败:")
            for error in errors:
                print(f"  • {error}")
                
    except Exception as e:
        print(f"❌ 自定义转换失败: {e}")


def example_5_practical_usage():
    """示例5: 实际使用场景"""
    print("\n" + "="*60)
    print("示例5: 实际使用场景 - 模拟因子计算")
    print("="*60)
    
    try:
        # 模拟在因子计算中的使用
        print("🧮 模拟因子计算流程:")
        
        # 1. 获取数据桥接器
        bridge = get_data_bridge()
        
        # 2. 使用便捷函数获取因子数据
        print("\n第1步: 获取基础数据")
        
        # 模拟获取收盘价数据
        try:
            close_factor = get_factor_data('price', 'c', begin_date=20241201)
            print(f"✅ 获取收盘价因子: {close_factor.shape}")
        except Exception as e:
            print(f"⚠️ 无法获取收盘价数据: {e}")
            # 创建模拟数据
            dates = pd.date_range('2024-12-01', periods=2, freq='D')
            stocks = ['000001', '000002']
            index = pd.MultiIndex.from_product([dates, stocks], names=['TradingDates', 'StockCodes'])
            close_factor = pd.Series([10.7, 15.35, 10.8, 15.4], index=index)
            print(f"✅ 使用模拟收盘价因子: {close_factor.shape}")
        
        # 3. 模拟因子计算
        print("\n第2步: 计算技术因子")
        
        # 计算动量因子（简单示例）
        momentum_factor = close_factor.groupby('StockCodes').pct_change()
        momentum_factor = momentum_factor.dropna()
        
        print(f"动量因子计算完成: {momentum_factor.shape}")
        print(f"动量因子预览:\n{momentum_factor}")
        
        # 4. 验证结果格式
        print("\n第3步: 验证因子格式")
        is_valid, errors = validate_factor_format(momentum_factor)
        
        if is_valid:
            print("✅ 计算结果格式验证通过")
        else:
            print("❌ 计算结果格式验证失败:")
            for error in errors:
                print(f"  • {error}")
        
        # 5. 模拟保存结果
        print("\n第4步: 保存计算结果")
        
        # 模拟保存到文件
        output_path = Path('data/cache/momentum_factor_example.pkl')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存因子数据
        momentum_factor.to_pickle(output_path)
        print(f"✅ 因子数据已保存到: {output_path}")
        
        # 验证保存的数据
        loaded_factor = pd.read_pickle(output_path)
        is_loaded_valid, _ = validate_factor_format(loaded_factor)
        
        if is_loaded_valid:
            print("✅ 保存和加载验证通过")
        else:
            print("⚠️ 保存的数据格式有问题")
        
    except Exception as e:
        print(f"❌ 实际使用场景演示失败: {e}")
        logger.error(f"Practical usage error: {e}", exc_info=True)


def main():
    """主函数：运行所有示例"""
    print("🚀 数据格式约定和验证使用示例")
    print("本示例演示了如何使用data模块的格式约定和验证功能")
    
    # 运行所有示例
    try:
        example_1_basic_validation()
        example_2_data_bridge_usage()
        example_3_data_quality_check()
        example_4_custom_validation()
        example_5_practical_usage()
        
        print("\n" + "="*60)
        print("🎉 所有示例运行完成!")
        print("="*60)
        
        print("\n📚 更多信息:")
        print("• 查看 data/DATA_FORMATS.md 了解详细格式规范")
        print("• 查看 data/README.md 了解模块使用指南")
        print("• 使用 validate_data_pipeline() 验证整个数据管道")
        
    except Exception as e:
        print(f"\n❌ 示例运行出错: {e}")
        logger.error(f"Main execution error: {e}", exc_info=True)


if __name__ == "__main__":
    main()