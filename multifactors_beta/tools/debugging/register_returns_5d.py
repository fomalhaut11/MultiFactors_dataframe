#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
注册5日收益率因子到系统

将新创建的5日收益率因子注册到因子库中，使其可以被其他因子调用
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

import logging
from factors.library.factor_registry import factor_registry
from factors.repository.technical.returns_5d import Returns5DFactor

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def register_returns_5d_factor():
    """注册5日收益率因子"""
    try:
        # 创建因子实例
        factor = Returns5DFactor()
        
        # 获取因子信息
        info = factor.get_factor_info()
        
        # 注册到因子库
        factor_registry.register_from_file(
            name=info['name'],                      # Returns_5D_C2C
            category=info['category'],              # technical  
            description=info['description'],        # 5日close-to-close滚动收益率
            dependencies=info['data_requirements'], # ["Price.pkl"]
            calculate_func=factor.calculate,        # 计算函数
            file_path=str(Path(__file__).parent / 'factors' / 'repository' / 'technical' / 'returns_5d.py'),
            calculation_method=info['calculation_method'],
            time_direction=info['time_direction'],
            output_format=info['output_format'],
            frequency=info['frequency'],
            min_periods=info['min_periods']
        )
        
        logger.info(f"成功注册因子: {info['name']}")
        logger.info(f"因子类别: {info['category']}")
        logger.info(f"因子描述: {info['description']}")
        
        # 验证注册成功
        registered_func = factor_registry.get(info['name'])
        if registered_func:
            logger.info("因子注册验证成功！")
            
            # 显示注册表状态
            all_factors = factor_registry.list_factors()
            technical_factors = factor_registry.list_factors('technical')
            
            logger.info(f"当前注册因子总数: {len(factor_registry)}")
            logger.info(f"技术类因子数量: {len(technical_factors)}")
            logger.info(f"技术类因子列表: {technical_factors}")
            
            return True
        else:
            logger.error("因子注册验证失败！")
            return False
            
    except Exception as e:
        logger.error(f"注册5日收益率因子失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_registered_factor():
    """测试注册的因子"""
    try:
        factor_name = "Returns_5D_C2C"
        
        # 从注册表获取因子函数
        factor_func = factor_registry.get(factor_name)
        if not factor_func:
            logger.error(f"无法找到注册的因子: {factor_name}")
            return False
        
        # 获取因子信息
        info = factor_registry.get_metadata(factor_name)
        logger.info(f"因子元数据: {info}")
        
        # 尝试调用因子计算（这会使用因子类的calculate方法）
        logger.info("尝试计算因子...")
        result = factor_func()
        
        logger.info(f"因子计算成功!")
        logger.info(f"结果类型: {type(result)}")
        logger.info(f"结果形状: {result.shape if hasattr(result, 'shape') else 'N/A'}")
        logger.info(f"因子名称: {result.name if hasattr(result, 'name') else 'N/A'}")
        
        # 显示样本数据
        if hasattr(result, 'head'):
            logger.info("样本数据:")
            logger.info(result.head(10))
        
        return True
        
    except Exception as e:
        logger.error(f"测试注册的因子失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("注册5日收益率因子到系统")
    print("=" * 60)
    
    # 1. 注册因子
    logger.info("第1步: 注册因子到系统...")
    success = register_returns_5d_factor()
    
    if success:
        print("成功: 5日收益率因子已注册到系统!")
        
        # 2. 测试注册的因子
        logger.info("第2步: 测试注册的因子...")
        test_success = test_registered_factor()
        
        if test_success:
            print("成功: 因子注册和测试都完成!")
            print("\n现在可以使用以下方式调用该因子:")
            print("from factors.library.factor_registry import get_factor")
            print("returns_5d_func = get_factor('Returns_5D_C2C')")
            print("result = returns_5d_func()")
        else:
            print("警告: 因子注册成功但测试失败，请检查具体问题")
    else:
        print("失败: 因子注册失败，请检查错误信息")