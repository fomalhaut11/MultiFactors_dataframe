#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试完整的复合盈利能力因子实现
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def test_complex_factor_import():
    """测试复合因子导入"""
    try:
        logger.info("测试复合因子类导入...")
        from factors.repository.mixed.complex_profitability_factor import ComplexProfitabilityFactor
        factor = ComplexProfitabilityFactor()
        
        logger.info(f"因子名称: {factor.name}")
        logger.info(f"因子类别: {factor.category}")
        logger.info(f"因子描述: {factor.description}")
        
        # 获取因子信息
        info = factor.get_factor_info()
        logger.info("因子信息:")
        for key, value in info.items():
            logger.info(f"  {key}: {value}")
        
        return factor
        
    except Exception as e:
        logger.error(f"复合因子导入失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_complex_factor_calculation(factor):
    """测试复合因子计算"""
    try:
        logger.info("=" * 60)
        logger.info("开始测试复合因子计算")
        logger.info("=" * 60)
        
        # 计算因子
        logger.info("开始计算复合盈利能力因子...")
        result = factor.calculate()
        
        logger.info("复合因子计算完成!")
        logger.info(f"结果类型: {type(result)}")
        logger.info(f"结果形状: {result.shape}")
        
        # 数据质量统计
        total_count = len(result)
        valid_count = result.notna().sum()
        
        logger.info(f"数据质量统计:")
        logger.info(f"  总数据点: {total_count:,}")
        logger.info(f"  有效数据点: {valid_count:,}")
        logger.info(f"  有效率: {valid_count/total_count*100:.1f}%")
        
        if valid_count > 0:
            logger.info(f"  数值范围: [{result.min():.4f}, {result.max():.4f}]")
            logger.info(f"  均值: {result.mean():.4f}")
            logger.info(f"  标准差: {result.std():.4f}")
        
        # 显示样本数据
        logger.info("样本数据 (前10行):")
        logger.info(result.head(10))
        
        # 保存结果
        output_path = Path('test_outputs')
        output_path.mkdir(exist_ok=True)
        result_file = output_path / 'complex_profitability_factor_test_result.pkl'
        result.to_pickle(result_file)
        logger.info(f"测试结果已保存到: {result_file}")
        
        return True
        
    except Exception as e:
        logger.error(f"复合因子计算失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("复合盈利能力因子完整测试")
    print("=" * 60)
    print("测试完整的复合因子实现")
    print("=" * 60)
    
    success_count = 0
    total_tests = 2
    
    # 测试1: 因子导入
    logger.info("测试1: 复合因子导入...")
    factor = test_complex_factor_import()
    if factor is not None:
        logger.info("测试1通过")
        success_count += 1
    else:
        logger.error("测试1失败")
        return False
    
    # 测试2: 因子计算
    logger.info("测试2: 复合因子计算...")
    if test_complex_factor_calculation(factor):
        logger.info("测试2通过")
        success_count += 1
    else:
        logger.error("测试2失败")
    
    # 汇总结果
    print("\n" + "=" * 60)
    print(f"测试结果: {success_count}/{total_tests} 通过")
    
    if success_count == total_tests:
        print("复合盈利能力因子完整测试成功！")
        print("因子计算逻辑工作正常，可以进行后续分析和应用")
    else:
        print("部分测试失败，需要检查相关问题")
    
    print("=" * 60)
    
    return success_count == total_tests

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)