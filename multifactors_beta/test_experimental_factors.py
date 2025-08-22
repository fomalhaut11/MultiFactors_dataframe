#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验性因子测试脚本 - 快速验证新因子想法
"""
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import sys

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from factors.generator.financial.experimental_factors import ExperimentalFactorCalculator

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_sample_data():
    """创建示例财务数据"""
    logger.info("创建示例财务数据...")
    
    # 创建时间和股票索引
    dates = pd.date_range('2020-03-31', periods=16, freq='Q')
    stocks = [f'00000{i}.SZ' for i in range(1, 6)]  # 5只股票
    
    # 创建MultiIndex
    index_tuples = [(date, stock) for date in dates for stock in stocks]
    multi_index = pd.MultiIndex.from_tuples(index_tuples, names=['ReportDates', 'StockCodes'])
    
    # 生成模拟财务数据
    np.random.seed(42)
    n_records = len(multi_index)
    
    data = {
        # 基础财务数据
        'DEDUCTEDPROFIT': np.random.normal(100, 30, n_records),  # 扣非净利润
        'TOT_OPER_REV': np.random.normal(1000, 200, n_records),  # 营业收入
        'NETCASH_OPER': np.random.normal(80, 40, n_records),     # 经营现金流
        'EQY_BELONGTO_PARCOMSH': np.random.normal(2000, 500, n_records),  # 股东权益
        'CASH': np.random.normal(200, 50, n_records),            # 货币资金
        'ST_BORROW': np.random.normal(100, 30, n_records),       # 短期借款
        'FIN_EXP_IS': np.random.normal(20, 10, n_records),       # 财务费用
        'd_quarter': [((date.month - 1) // 3) + 1 for date in dates for _ in stocks]  # 季度
    }
    
    # 模拟累计数据的特征（Q4>Q3>Q2>Q1）
    for i, (date, stock) in enumerate(index_tuples):
        quarter = data['d_quarter'][i]
        # 让累计数据呈递增趋势
        if quarter == 2:
            data['DEDUCTEDPROFIT'][i] = abs(data['DEDUCTEDPROFIT'][i]) * 1.5
            data['TOT_OPER_REV'][i] = abs(data['TOT_OPER_REV'][i]) * 1.5
        elif quarter == 3:
            data['DEDUCTEDPROFIT'][i] = abs(data['DEDUCTEDPROFIT'][i]) * 2
            data['TOT_OPER_REV'][i] = abs(data['TOT_OPER_REV'][i]) * 2
        elif quarter == 4:
            data['DEDUCTEDPROFIT'][i] = abs(data['DEDUCTEDPROFIT'][i]) * 3
            data['TOT_OPER_REV'][i] = abs(data['TOT_OPER_REV'][i]) * 3
    
    df = pd.DataFrame(data, index=multi_index)
    logger.info(f"示例数据创建完成: {df.shape}")
    return df


def create_sample_return_data():
    """创建示例收益率数据用于单因子检验"""
    logger.info("创建示例收益率数据...")
    
    # 创建更密集的交易日数据用于收益率计算
    dates = pd.date_range('2020-01-01', periods=200, freq='B')  # 交易日
    stocks = [f'00000{i}.SZ' for i in range(1, 6)]  # 5只股票
    
    # 创建MultiIndex
    index_tuples = [(date, stock) for date in dates for stock in stocks]
    multi_index = pd.MultiIndex.from_tuples(index_tuples, names=['TradingDates', 'StockCodes'])
    
    # 生成模拟收益率数据
    np.random.seed(123)
    n_records = len(multi_index)
    
    # 模拟日收益率 (均值为0，标准差为2%)
    returns = np.random.normal(0.0005, 0.02, n_records)
    
    # 为不同股票添加一些系统性差异
    for i, (date, stock) in enumerate(index_tuples):
        stock_idx = int(stock.split('.')[0][-1]) - 1
        # 不同股票有不同的期望收益率
        returns[i] += stock_idx * 0.0001
    
    return_data = pd.Series(returns, index=multi_index, name='daily_return')
    logger.info(f"收益率数据创建完成: {return_data.shape}")
    return return_data


def create_sample_factor_data():
    """创建示例因子数据用于单因子检验"""
    logger.info("创建示例因子数据...")
    
    # 创建与收益率数据对应的因子数据（较低频率）
    dates = pd.date_range('2020-01-01', periods=50, freq='4B')  # 每4个交易日一个因子值
    stocks = [f'00000{i}.SZ' for i in range(1, 6)]  # 5只股票
    
    # 创建MultiIndex
    index_tuples = [(date, stock) for date in dates for stock in stocks]
    multi_index = pd.MultiIndex.from_tuples(index_tuples, names=['TradingDates', 'StockCodes'])
    
    # 生成模拟因子数据
    np.random.seed(456)
    n_records = len(multi_index)
    
    # 创建一个有预测能力的因子（与未来收益有微弱正相关）
    factor_values = np.random.normal(0, 1, n_records)
    
    # 为不同股票添加一些趋势性
    for i, (date, stock) in enumerate(index_tuples):
        stock_idx = int(stock.split('.')[0][-1]) - 1
        # 高因子值的股票有略高的预期收益
        factor_values[i] += stock_idx * 0.2
    
    factor_data = pd.Series(factor_values, index=multi_index, name='test_factor')
    logger.info(f"因子数据创建完成: {factor_data.shape}")
    return factor_data


def test_experimental_factors():
    """测试实验性因子"""
    logger.info("🧪 开始测试实验性因子")
    logger.info("=" * 60)
    
    # 1. 创建计算器
    calculator = ExperimentalFactorCalculator()
    
    # 2. 准备数据
    financial_data = create_sample_data()
    
    # 3. 配置列映射（模拟真实的列名映射）
    calculator.set_column_mapping('earnings', 'DEDUCTEDPROFIT')
    calculator.set_column_mapping('revenue', 'TOT_OPER_REV')
    calculator.set_column_mapping('operating_cash_flow', 'NETCASH_OPER')
    calculator.set_column_mapping('equity', 'EQY_BELONGTO_PARCOMSH')
    calculator.set_column_mapping('cash_equivalents', 'CASH')
    calculator.set_column_mapping('short_term_debt', 'ST_BORROW')
    calculator.set_column_mapping('financial_expense', 'FIN_EXP_IS')
    calculator.set_column_mapping('quarter', 'd_quarter')
    
    # 4. 测试单个实验性因子
    print("\n🔬 测试单个实验性因子")
    print("-" * 40)
    
    try:
        # 测试盈利增长质量因子
        factor1 = calculator.calculate_EXPERIMENTAL_ProfitGrowthQuality_ttm(financial_data)
        result1 = calculator.quick_validate_factor(factor1, 'ProfitGrowthQuality')
        
        # 测试债务偿付能力因子
        factor2 = calculator.calculate_EXPERIMENTAL_DebtServiceAbility_ttm(financial_data)
        result2 = calculator.quick_validate_factor(factor2, 'DebtServiceAbility')
        
    except Exception as e:
        logger.error(f"单个因子测试失败: {e}")
    
    # 5. 批量测试所有实验性因子
    print("\n🚀 批量测试所有实验性因子")
    print("-" * 40)
    
    try:
        batch_results = calculator.run_experimental_batch(financial_data)
        
        # 汇总批量测试结果
        print(f"\n📊 批量测试汇总:")
        for name, data in batch_results.items():
            if data is not None:
                valid_count = data.count()
                print(f"   ✅ {name}: {valid_count} 个有效数据点")
            else:
                print(f"   ❌ {name}: 计算失败")
                
    except Exception as e:
        logger.error(f"批量测试失败: {e}")
    
    # 6. 显示验证结果汇总
    print(f"\n📋 因子验证结果汇总:")
    print("-" * 40)
    
    for factor_name, result in calculator.validation_results.items():
        score = result.get('overall_score', 0)
        recommendation = result.get('recommendation', '未评估')
        print(f"   {factor_name}: {score}/100 - {recommendation}")
    
    # 7. 生成迁移到生产环境的代码
    print(f"\n📤 生成迁移代码示例:")
    print("-" * 40)
    
    calculator.export_to_production(
        'calculate_EXPERIMENTAL_ProfitGrowthQuality_ttm',
        'ProfitGrowthQuality',
        'profitability'
    )


def test_factor_template():
    """测试因子模板功能"""
    print("\n📝 生成新因子代码模板")
    print("=" * 60)
    
    from factors.generator.financial.experimental_factors import create_experimental_factor_template
    
    # 生成模板
    template = create_experimental_factor_template(
        factor_name="AssetEfficiency",
        formula_description="TTM营业收入 / 总资产均值",
        economic_meaning="衡量企业资产使用效率",
        hypothesis="高资产效率的企业应该有更好的盈利能力"
    )
    
    print(template)


def test_single_factor_analysis():
    """测试单因子检验功能"""
    logger.info("🔬 开始测试单因子检验功能")
    logger.info("=" * 60)
    
    try:
        # 1. 创建计算器
        calculator = ExperimentalFactorCalculator()
        
        # 2. 创建模拟数据
        factor_data = create_sample_factor_data()
        return_data = create_sample_return_data()
        
        logger.info(f"因子数据: {factor_data.shape}")
        logger.info(f"收益率数据: {return_data.shape}")
        
        # 3. 执行单因子检验
        print("\n🔬 执行完整单因子检验")
        print("-" * 40)
        
        test_results = calculator.single_factor_test(
            factor_data=factor_data,
            return_data=return_data,
            factor_name='TestFactor',
            periods=[1, 5, 10],  # 1天、5天、10天持有期
            quantiles=5,  # 5分组
            save_results=True
        )
        
        # 4. 展示关键结果
        if 'summary' in test_results:
            summary = test_results['summary']
            print(f"\n📊 检验结果摘要:")
            print(f"   因子评分: {summary.get('overall_score', 0):.1f}/100")
            print(f"   因子等级: {summary.get('grade', 'N/A')}")
            print(f"   主要优势: {', '.join(summary.get('strengths', []))}")
            print(f"   主要劣势: {', '.join(summary.get('weaknesses', []))}")
        
        # 5. 测试另一个因子（使用内置的实验性因子）
        print("\n🧪 测试内置实验性因子")
        print("-" * 40)
        
        # 创建财务数据
        financial_data = create_sample_data()
        
        # 配置列映射
        calculator.set_column_mapping('earnings', 'DEDUCTEDPROFIT')
        calculator.set_column_mapping('revenue', 'TOT_OPER_REV')
        calculator.set_column_mapping('operating_cash_flow', 'NETCASH_OPER')
        calculator.set_column_mapping('equity', 'EQY_BELONGTO_PARCOMSH')
        calculator.set_column_mapping('cash_equivalents', 'CASH')
        calculator.set_column_mapping('short_term_debt', 'ST_BORROW')
        calculator.set_column_mapping('financial_expense', 'FIN_EXP_IS')
        calculator.set_column_mapping('quarter', 'd_quarter')
        
        # 计算内置的实验性因子
        try:
            experimental_factor = calculator.calculate_EXPERIMENTAL_DebtServiceAbility_ttm(financial_data)
            
            if not experimental_factor.empty and experimental_factor.count() > 0:
                # 需要将因子数据转换为日频格式进行检验
                # 这里简化处理，直接使用季频数据
                print(f"   计算得到实验性因子: {experimental_factor.count()} 个有效值")
                
                # 简化的检验（使用相同的时间索引）
                simple_validation = calculator.quick_validate_factor(
                    experimental_factor, 'DebtServiceAbility'
                )
                
                print(f"   数据质量评分: {simple_validation.get('overall_score', 0):.1f}/100")
            else:
                print("   实验性因子计算结果为空")
                
        except Exception as e:
            logger.error(f"内置实验性因子测试失败: {e}")
        
        return test_results
        
    except Exception as e:
        logger.error(f"单因子检验测试失败: {e}")
        return None


def test_real_data():
    """如果有真实数据，测试真实数据"""
    data_path = project_root / "data" / "auxiliary" / "FinancialData_unified.pkl"
    
    if data_path.exists():
        print("\n🔥 使用真实数据测试")
        print("=" * 60)
        
        try:
            calculator = ExperimentalFactorCalculator()
            real_data = pd.read_pickle(data_path)
            
            # 取小样本测试
            sample_stocks = real_data.index.get_level_values('StockCodes').unique()[:10]
            real_sample = real_data[real_data.index.get_level_values('StockCodes').isin(sample_stocks)]
            
            print(f"真实数据样本: {real_sample.shape}")
            
            # 测试一个简单的实验性因子
            if 'DEDUCTEDPROFIT' in real_sample.columns and 'd_quarter' in real_sample.columns:
                # 创建一个简单的测试因子
                def test_simple_factor(data):
                    try:
                        # 简单的ROE计算作为测试
                        earnings_data = data[['DEDUCTEDPROFIT', 'd_quarter']].copy()
                        from factors.base.time_series_processor import TimeSeriesProcessor
                        ttm_result = TimeSeriesProcessor.calculate_ttm(earnings_data)
                        return ttm_result.iloc[:, 0] if ttm_result.shape[1] > 0 else pd.Series(dtype=float)
                    except Exception as e:
                        logger.error(f"简单因子测试失败: {e}")
                        return pd.Series(index=data.index, dtype=float)
                
                test_factor = test_simple_factor(real_sample)
                calculator.quick_validate_factor(test_factor, 'SimpleTestFactor')
                
        except Exception as e:
            logger.error(f"真实数据测试失败: {e}")
    else:
        print("未找到真实数据，跳过真实数据测试")


def main():
    """主函数"""
    print("🧪 实验性因子测试系统")
    print("=" * 80)
    
    # 测试实验性因子
    test_experimental_factors()
    
    # 测试单因子检验功能
    test_single_factor_analysis()
    
    # 测试模板生成
    test_factor_template()
    
    # 测试真实数据
    test_real_data()
    
    print("\n🎉 所有测试完成!")
    print("=" * 80)
    print("💡 使用提示:")
    print("1. 复制experimental_factors.py中的模板开始新因子开发")
    print("2. 实现你的因子计算逻辑")
    print("3. 使用 quick_validate_factor 进行基础验证")
    print("4. 使用 single_factor_test 进行完整的单因子检验")
    print("5. 验证通过后使用 export_to_production 迁移到正式环境")
    print("\n📊 单因子检验包括:")
    print("   - IC分析 (信息系数)")
    print("   - 分组分析 (多空收益)")
    print("   - 单调性检验")
    print("   - 统计显著性检验")
    print("   - 因子衰减分析")
    print("   - 综合评分 (A-F等级)")


if __name__ == "__main__":
    main()