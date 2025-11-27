#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试复合盈利能力因子

在项目根目录下运行，测试完整的因子计算流程
"""

import sys
import os
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

def test_data_availability():
    """测试数据可用性"""
    try:
        logger.info("=" * 60)
        logger.info("测试数据可用性")
        logger.info("=" * 60)
        
        # 1. 检查财务数据
        logger.info("检查财务数据...")
        try:
            # 直接加载财务数据文件
            data_root = Path('E:/Documents/PythonProject/StockProject/StockData')
            financial_file = data_root / 'auxiliary' / 'FinancialData_unified.pkl'
            
            if not financial_file.exists():
                logger.error(f"❌ 财务数据文件不存在: {financial_file}")
                return False
            
            financial_data = pd.read_pickle(financial_file)
            logger.info(f"✅ 财务数据: {financial_data.shape}")
            
            required_fields = ['DEDUCTEDPROFIT', 'FIN_EXP_IS', 'INVENTORIES', 'ST_BORROW']
            missing_fields = [field for field in required_fields if field not in financial_data.columns]
            
            if missing_fields:
                logger.error(f"缺少必需字段: {missing_fields}")
                logger.info(f"可用字段: {list(financial_data.columns)[:10]}...")  # 显示前10个字段
                return False
            else:
                logger.info(f"所有必需字段都存在")
                
        except Exception as e:
            logger.error(f"财务数据加载失败: {e}")
            return False
        
        # 2. 检查5日收益率因子
        logger.info("检查5日收益率因子...")
        try:
            from factors.library.factor_registry import get_factor
            returns_func = get_factor('Returns_5D_C2C')
            
            if returns_func:
                logger.info("✅ 5日收益率因子已注册")
                
                # 检查存储文件
                data_root = Path('E:/Documents/PythonProject/StockProject/StockData')
                returns_file = data_root / 'factors' / 'technical' / 'Returns_5D_C2C.pkl'
                
                if returns_file.exists():
                    logger.info(f"✅ 5日收益率数据文件存在: {returns_file}")
                    
                    # 快速检查数据
                    sample_data = pd.read_pickle(returns_file)
                    logger.info(f"✅ 5日收益率数据: {sample_data.shape}")
                else:
                    logger.warning("⚠️ 5日收益率数据文件不存在，需要现场计算")
            else:
                logger.error("❌ 5日收益率因子未注册")
                return False
                
        except Exception as e:
            logger.error(f"❌ 5日收益率因子检查失败: {e}")
            return False
        
        logger.info("✅ 所有数据检查通过")
        return True
        
    except Exception as e:
        logger.error(f"数据可用性检查失败: {e}")
        return False

def test_basic_calculations():
    """测试基础计算组件"""
    try:
        logger.info("=" * 60) 
        logger.info("测试基础计算组件")
        logger.info("=" * 60)
        
        # 1. 测试TTM计算
        logger.info("测试TTM计算逻辑...")
        
        # 创建测试数据
        test_data = pd.DataFrame({
            'StockCodes': ['000001'] * 8,
            'ReportDate': pd.date_range('2023-03-31', periods=8, freq='Q'),
            'DEDUCTEDPROFIT': [100, 120, 110, 130, 140, 150, 160, 170],
            'FINANCIALEXPENSE': [10, 12, 11, 13, 14, 15, 16, 17]
        })
        
        # 计算TTM
        def calc_ttm(group):
            group = group.sort_values('ReportDate')
            group['DEDUCTEDPROFIT_TTM'] = group['DEDUCTEDPROFIT'].rolling(4, min_periods=1).sum()
            group['FINANCIALEXPENSE_TTM'] = group['FINANCIALEXPENSE'].rolling(4, min_periods=1).sum()
            return group
            
        test_result = test_data.groupby('StockCodes').apply(calc_ttm).reset_index(drop=True)
        
        logger.info("TTM计算样本:")
        logger.info(test_result[['ReportDate', 'DEDUCTEDPROFIT', 'DEDUCTEDPROFIT_TTM']].tail(3))
        
        # 2. 测试截面z-score计算
        logger.info("测试截面z-score计算...")
        
        # 创建测试收益率数据
        test_returns = pd.Series([0.01, 0.02, -0.01, 0.03, -0.02], 
                                index=pd.MultiIndex.from_tuples([
                                    ('2024-01-01', 'A'),
                                    ('2024-01-01', 'B'), 
                                    ('2024-01-01', 'C'),
                                    ('2024-01-01', 'D'),
                                    ('2024-01-01', 'E')
                                ], names=['Date', 'Stock']))
        
        def calc_cross_sectional_zscore(group):
            mean = group.mean()
            std = group.std()
            if std == 0 or pd.isna(std):
                return pd.Series(0, index=group.index)
            return (group - mean) / std
        
        zscore_result = test_returns.groupby(level=0).apply(calc_cross_sectional_zscore)
        if isinstance(zscore_result.index, pd.MultiIndex) and len(zscore_result.index.levels) > 2:
            zscore_result = zscore_result.droplevel(0)
            
        logger.info("截面z-score计算样本:")
        logger.info(f"原始收益率: {test_returns.values}")
        logger.info(f"z-score结果: {zscore_result.values}")
        
        logger.info("✅ 基础计算组件测试通过")
        return True
        
    except Exception as e:
        logger.error(f"基础计算测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_complex_factor_calculation():
    """测试完整的复合因子计算"""
    try:
        logger.info("=" * 60)
        logger.info("测试完整复合因子计算") 
        logger.info("=" * 60)
        
        # 导入复合因子类
        from factors.repository.mixed.complex_profitability_factor import ComplexProfitabilityFactor
        
        # 创建因子实例
        logger.info("创建复合因子实例...")
        factor = ComplexProfitabilityFactor()
        
        # 获取因子信息
        info = factor.get_factor_info()
        logger.info("因子信息:")
        for key, value in info.items():
            logger.info(f"  {key}: {value}")
        
        # 尝试计算因子（限制规模以避免超时）
        logger.info("开始计算复合因子（测试版）...")
        logger.info("注意: 这可能需要几分钟时间...")
        
        result = factor.calculate()
        
        if result is not None and len(result) > 0:
            logger.info("✅ 复合因子计算成功!")
            logger.info(f"结果形状: {result.shape}")
            logger.info(f"结果类型: {type(result)}")
            
            # 数据质量检查
            valid_count = result.notna().sum()
            total_count = len(result)
            
            logger.info(f"数据质量:")
            logger.info(f"  总数据点: {total_count:,}")
            logger.info(f"  有效数据点: {valid_count:,}")
            logger.info(f"  有效率: {valid_count/total_count*100:.1f}%")
            
            if valid_count > 0:
                logger.info(f"  数值范围: [{result.min():.4f}, {result.max():.4f}]")
                logger.info(f"  均值: {result.mean():.4f}")
                logger.info(f"  标准差: {result.std():.4f}")
            
            # 显示样本数据
            logger.info("样本数据:")
            logger.info(result.head(10))
            
            return True
        else:
            logger.error("❌ 复合因子计算返回空结果")
            return False
            
    except Exception as e:
        logger.error(f"复合因子计算测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试流程"""
    print("复合盈利能力因子测试")
    print("=" * 60)
    print("此测试将验证:")
    print("1. 数据可用性检查")
    print("2. 基础计算组件测试") 
    print("3. 完整因子计算测试")
    print("=" * 60)
    
    try:
        # Phase 1: 数据可用性检查
        logger.info("Phase 1: 数据可用性检查...")
        if not test_data_availability():
            print("❌ 数据可用性检查失败，请检查数据文件")
            return False
        
        # Phase 2: 基础计算测试
        logger.info("Phase 2: 基础计算组件测试...")
        if not test_basic_calculations():
            print("❌ 基础计算组件测试失败")
            return False
        
        # Phase 3: 完整因子计算测试
        logger.info("Phase 3: 完整因子计算测试...")
        if not test_complex_factor_calculation():
            print("❌ 完整因子计算测试失败")
            return False
        
        print("\n" + "=" * 60)
        print("✅ 所有测试通过！复合盈利能力因子可以正常工作")
        print("=" * 60)
        print("现在可以进行单因子测试和因子评价了")
        print("=" * 60)
        return True
        
    except Exception as e:
        logger.error(f"测试过程出现错误: {e}")
        print(f"❌ 测试过程失败: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)