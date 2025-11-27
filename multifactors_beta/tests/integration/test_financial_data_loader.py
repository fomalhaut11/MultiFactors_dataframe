#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试新添加的财务数据加载器接口
"""

import sys
from pathlib import Path
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

def test_financial_data_loader():
    """测试财务数据加载器"""
    try:
        logger.info("=" * 60)
        logger.info("测试新的财务数据加载器接口")
        logger.info("=" * 60)
        
        # 方法1: 使用类方法
        logger.info("测试类方法加载...")
        from factors.utils.data_loader import FactorDataLoader
        
        financial_data = FactorDataLoader.load_financial_data()
        logger.info(f"类方法加载成功: {financial_data.shape}")
        
        # 方法2: 使用便捷函数
        logger.info("测试便捷函数加载...")
        from factors.utils.data_loader import get_financial_data
        
        financial_data2 = get_financial_data()
        logger.info(f"便捷函数加载成功: {financial_data2.shape}")
        
        # 验证数据一致性
        if financial_data.equals(financial_data2):
            logger.info("数据一致性验证通过")
        else:
            logger.warning("数据一致性验证失败")
        
        # 检查必需字段
        logger.info("检查必需的财务字段...")
        required_fields = ['DEDUCTEDPROFIT', 'FIN_EXP_IS', 'INVENTORIES', 'ST_BORROW']
        missing_fields = [field for field in required_fields if field not in financial_data.columns]
        
        if missing_fields:
            logger.error(f"缺少必需字段: {missing_fields}")
            logger.info("前20个可用字段:")
            for i, col in enumerate(financial_data.columns[:20]):
                logger.info(f"  {i+1:2d}. {col}")
            return False
        else:
            logger.info("所有必需字段都存在")
        
        # 显示数据摘要
        logger.info(f"财务数据摘要:")
        logger.info(f"  数据形状: {financial_data.shape}")
        logger.info(f"  索引类型: {type(financial_data.index)}")
        
        if hasattr(financial_data.index, 'names'):
            logger.info(f"  索引名称: {financial_data.index.names}")
        
        # 显示样本数据
        logger.info("样本数据:")
        sample_cols = ['DEDUCTEDPROFIT', 'FIN_EXP_IS', 'INVENTORIES', 'ST_BORROW']
        if all(col in financial_data.columns for col in sample_cols):
            sample_data = financial_data[sample_cols].head(5)
            logger.info(sample_data)
        
        # 测试缓存功能
        logger.info("测试缓存功能...")
        financial_data3 = FactorDataLoader.load_financial_data()
        if id(financial_data) == id(financial_data3):
            logger.info("缓存功能工作正常")
        else:
            logger.warning("缓存功能可能有问题")
        
        return True
        
    except Exception as e:
        logger.error(f"测试财务数据加载器失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integration_with_complex_factor():
    """测试与复合因子的集成"""
    try:
        logger.info("=" * 60)
        logger.info("测试与复合因子的集成")
        logger.info("=" * 60)
        
        # 导入复合因子并测试数据加载
        from factors.repository.mixed.complex_profitability_factor import ComplexProfitabilityFactor
        
        factor = ComplexProfitabilityFactor()
        
        # 测试财务数据加载
        logger.info("测试复合因子中的财务数据加载...")
        financial_data = factor._load_financial_data()
        
        logger.info(f"复合因子财务数据加载成功: {financial_data.shape}")
        
        # 验证字段
        required_fields = ['DEDUCTEDPROFIT', 'FIN_EXP_IS', 'INVENTORIES', 'ST_BORROW']
        missing_fields = [field for field in required_fields if field not in financial_data.columns]
        
        if missing_fields:
            logger.error(f"复合因子测试失败，缺少字段: {missing_fields}")
            return False
        
        logger.info("复合因子集成测试通过")
        return True
        
    except Exception as e:
        logger.error(f"复合因子集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("财务数据加载器接口测试")
    print("=" * 60)
    
    success_count = 0
    total_tests = 2
    
    # 测试1: 基础财务数据加载
    logger.info("测试1: 基础财务数据加载...")
    if test_financial_data_loader():
        logger.info("测试1通过")
        success_count += 1
    else:
        logger.error("测试1失败")
    
    # 测试2: 与复合因子集成
    logger.info("测试2: 与复合因子集成...")
    if test_integration_with_complex_factor():
        logger.info("测试2通过")
        success_count += 1
    else:
        logger.error("测试2失败")
    
    # 汇总结果
    print("\n" + "=" * 60)
    print(f"测试结果: {success_count}/{total_tests} 通过")
    
    if success_count == total_tests:
        print("所有测试通过！财务数据加载器接口工作正常")
        print("现在可以在复合因子中使用新的接口了")
    else:
        print("部分测试失败，需要进一步调试")
    
    print("=" * 60)
    
    return success_count == total_tests

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)