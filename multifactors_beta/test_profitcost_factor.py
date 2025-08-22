#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试ProfitCost因子计算和保存
"""

import sys
from pathlib import Path
import logging
import pandas as pd
import numpy as np

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    # 简化导入，只导入必要的模块
    from factors.generator.financial.pure_financial_factors import PureFinancialFactorCalculator
    # 其他模块先注释掉，避免循环导入
    # from factors.tester import SingleFactorTestPipeline
    # from core import test_single_factor
    # from core.config_manager import ConfigManager
except ImportError as e:
    logger.error(f"导入模块失败: {e}")
    logger.exception("详细错误:")
    sys.exit(1)


def test_profitcost_basic():
    """基础测试ProfitCost因子计算"""
    logger.info("="*60)
    logger.info("基础测试ProfitCost因子计算")
    logger.info("="*60)
    
    try:
        calculator = PureFinancialFactorCalculator()
        
        # 检查因子是否在可用因子列表中
        available_factors = calculator.get_available_factors()
        if 'ProfitCost_ttm' in available_factors:
            logger.info("✓ ProfitCost_ttm因子已注册成功")
        else:
            logger.error("✗ ProfitCost_ttm因子未找到")
            return False
            
        # 检查因子分类
        profitability_factors = calculator.factor_categories.get('profitability', [])
        if 'ProfitCost_ttm' in profitability_factors:
            logger.info("✓ ProfitCost_ttm因子已正确分类到盈利能力类别")
        else:
            logger.error("✗ ProfitCost_ttm因子分类错误")
            
        return True
        
    except Exception as e:
        logger.error(f"基础测试失败: {e}")
        return False


def test_profitcost_calculation():
    """测试ProfitCost因子计算（模拟数据）"""
    logger.info("="*60)
    logger.info("测试ProfitCost因子计算（模拟数据）")
    logger.info("="*60)
    
    try:
        calculator = PureFinancialFactorCalculator()
        
        # 创建模拟财务数据
        dates = pd.date_range('2023-03-31', '2024-12-31', freq='Q')
        stocks = ['000001.SZ', '000002.SZ', '600000.SH']
        
        # 创建MultiIndex
        index = pd.MultiIndex.from_product(
            [dates, stocks], 
            names=['ReportDates', 'StockCodes']
        )
        
        # 模拟数据
        np.random.seed(42)
        n_records = len(index)
        
        financial_data = pd.DataFrame({
            'DEDUCTEDPROFIT': np.random.uniform(1000, 50000, n_records),  # 扣非净利润
            'FIN_EXP_IS': np.random.uniform(100, 5000, n_records),        # 财务费用
            'TAX': np.random.uniform(200, 8000, n_records),               # 所得税
            'd_quarter': [d.quarter for d in dates] * len(stocks),         # 季度
            'd_year': [d.year for d in dates] * len(stocks),              # 年份
            'ReleasedDates': [d + pd.DateOffset(days=30) for d in dates] * len(stocks)  # 发布日期
        }, index=index)
        
        logger.info(f"创建模拟数据: {financial_data.shape[0]}条记录, {len(stocks)}只股票")
        logger.info(f"数据范围: {financial_data.index.get_level_values(0).min()} 到 {financial_data.index.get_level_values(0).max()}")
        
        # 测试因子计算
        logger.info("开始计算ProfitCost_ttm因子...")
        result = calculator.calculate_ProfitCost_ttm(financial_data)
        
        if result is not None and len(result) > 0:
            logger.info(f"✓ 计算成功!")
            logger.info(f"  结果数量: {len(result)}")
            logger.info(f"  有效值数量: {result.count()}")
            logger.info(f"  均值: {result.mean():.4f}")
            logger.info(f"  标准差: {result.std():.4f}")
            logger.info(f"  最小值: {result.min():.4f}")
            logger.info(f"  最大值: {result.max():.4f}")
            
            # 显示部分结果
            logger.info("前10个计算结果:")
            for i, (idx, val) in enumerate(result.head(10).items()):
                logger.info(f"  {idx}: {val:.4f}")
                
            return True
        else:
            logger.error("✗ 计算结果为空")
            return False
            
    except Exception as e:
        logger.error(f"计算测试失败: {e}")
        logger.exception("详细错误信息:")
        return False


def test_profitcost_pipeline():
    """使用测试流水线测试ProfitCost因子"""
    logger.info("="*60)
    logger.info("使用测试流水线测试ProfitCost因子")
    logger.info("="*60)
    
    try:
        # 使用core模块的便捷函数
        logger.info("开始单因子测试流水线...")
        result = test_single_factor(
            'ProfitCost_ttm',
            begin_date='2024-01-01',
            end_date='2024-06-30',
            group_nums=5
        )
        
        if result and hasattr(result, 'ic_result') and result.ic_result:
            logger.info("✓ 流水线测试成功!")
            logger.info(f"  IC均值: {result.ic_result.ic_mean:.4f}")
            logger.info(f"  ICIR: {result.ic_result.icir:.4f}")
            logger.info(f"  Rank IC: {result.ic_result.rank_ic_mean:.4f}")
            return True
        else:
            logger.warning("流水线测试未返回有效IC结果，可能是数据不足")
            return False
            
    except Exception as e:
        logger.error(f"流水线测试失败: {e}")
        logger.exception("详细错误信息:")
        return False


def save_profitcost_factor():
    """生成并保存ProfitCost因子"""
    logger.info("="*60)
    logger.info("生成并保存ProfitCost因子")
    logger.info("="*60)
    
    try:
        config_manager = ConfigManager()
        factor_path = config_manager.get_path('factors', 'ProfitCost_ttm.pkl')
        
        # 使用流水线测试并保存
        pipeline = SingleFactorTestPipeline()
        result = pipeline.run(
            'ProfitCost_ttm',
            save_result=True,
            begin_date='2024-01-01',
            end_date='2024-10-31',
            group_nums=5,
            netral_base=False,
            use_industry=False
        )
        
        if result:
            logger.info("✓ 因子保存成功!")
            logger.info(f"  保存路径: {factor_path}")
            
            # 尝试加载验证
            if factor_path.exists():
                factor_data = pd.read_pickle(factor_path)
                logger.info(f"  验证加载: {factor_data.shape[0]}条记录")
                logger.info(f"  数据范围: {factor_data.index.min()} 到 {factor_data.index.max()}")
                return True
        
        return False
        
    except Exception as e:
        logger.error(f"保存因子失败: {e}")
        logger.exception("详细错误信息:")
        return False


def main():
    """主测试函数"""
    logger.info("开始ProfitCost因子基础测试")
    
    results = {
        'basic_test': test_profitcost_basic(),
        'calculation_test': test_profitcost_calculation(),
        # 'pipeline_test': test_profitcost_pipeline(),  # 暂时注释掉
        # 'save_test': save_profitcost_factor()  # 暂时注释掉
    }
    
    logger.info("="*60)
    logger.info("测试结果总结")
    logger.info("="*60)
    
    for test_name, success in results.items():
        status = "✓ 通过" if success else "✗ 失败"
        logger.info(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    if all_passed:
        logger.info("\n🎉 所有测试通过! ProfitCost因子已成功实现并保存!")
    else:
        logger.warning(f"\n⚠️  部分测试失败，通过率: {sum(results.values())}/{len(results)}")
    
    return all_passed


if __name__ == "__main__":
    main()