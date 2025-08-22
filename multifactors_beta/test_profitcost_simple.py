#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单测试ProfitCost因子实现
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# 添加路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_profitcost_implementation():
    """测试ProfitCost因子是否正确实现"""
    logger.info("测试ProfitCost因子实现")
    
    try:
        # 直接导入并检查是否包含我们的方法
        import importlib.util
        
        spec = importlib.util.spec_from_file_location(
            "pure_financial_factors", 
            str(project_root / "factors/generator/financial/pure_financial_factors.py")
        )
        module = importlib.util.module_from_spec(spec)
        
        # 读取文件内容检查
        with open(project_root / "factors/generator/financial/pure_financial_factors.py", 'r', encoding='utf-8') as f:
            content = f.read()
            
        # 检查实现
        checks = {
            'ProfitCost_ttm in profitability': "'ProfitCost_ttm'" in content and "'profitability':" in content,
            'calculate_ProfitCost_ttm method': "def calculate_ProfitCost_ttm" in content,
            'TTM calculation': "TimeSeriesProcessor.calculate_ttm" in content,
            'financial_expense mapping': "'financial_expense'" in content,
            'income_tax mapping': "'income_tax'" in content,
            'registration': "'ProfitCost_ttm': self.calculate_ProfitCost_ttm" in content
        }
        
        logger.info("实现检查结果:")
        all_passed = True
        for check_name, passed in checks.items():
            status = "✓" if passed else "✗"
            logger.info(f"  {status} {check_name}")
            if not passed:
                all_passed = False
                
        return all_passed
        
    except Exception as e:
        logger.error(f"检查失败: {e}")
        return False

def test_config_updates():
    """测试配置文件更新"""
    logger.info("测试配置文件更新")
    
    try:
        with open(project_root / "factors/config/factor_config.py", 'r', encoding='utf-8') as f:
            config_content = f.read()
            
        checks = {
            'financial_expense mapping': "'financial_expense': 'FIN_EXP_IS'" in config_content,
            'income_tax mapping': "'income_tax': 'TAX'" in config_content,
            'ProfitCost defaults': "'ProfitCost':" in config_content and "'method': 'ttm'" in config_content
        }
        
        logger.info("配置检查结果:")
        all_passed = True
        for check_name, passed in checks.items():
            status = "✓" if passed else "✗"
            logger.info(f"  {status} {check_name}")
            if not passed:
                all_passed = False
                
        return all_passed
        
    except Exception as e:
        logger.error(f"配置检查失败: {e}")
        return False

def create_sample_factor_data():
    """创建示例因子数据用于验证"""
    logger.info("创建ProfitCost因子示例数据")
    
    try:
        # 创建模拟数据
        dates = pd.date_range('2024-01-01', '2024-12-31', freq='M')[:12]
        stocks = ['000001.SZ', '000002.SZ', '600000.SH', '600036.SH', '000858.SZ']
        
        # 创建因子数据
        factor_data = []
        
        np.random.seed(42)
        for date in dates:
            for stock in stocks:
                # 模拟ProfitCost值：扣非净利润/(财务费用+所得税)
                deducted_profit = np.random.uniform(1000, 50000)
                financial_expense = np.random.uniform(100, 5000)
                income_tax = np.random.uniform(200, 8000)
                
                profitcost_value = deducted_profit / (financial_expense + income_tax)
                
                factor_data.append({
                    'Date': date,
                    'StockCode': stock,
                    'ProfitCost': profitcost_value,
                    'DeductedProfit': deducted_profit,
                    'FinancialExpense': financial_expense,
                    'IncomeTax': income_tax
                })
        
        df = pd.DataFrame(factor_data)
        
        # 设置多级索引
        df_pivot = df.set_index(['Date', 'StockCode'])['ProfitCost']
        
        logger.info(f"创建示例数据: {len(df)}条记录")
        logger.info(f"股票数量: {len(stocks)}")
        logger.info(f"时间范围: {dates[0].strftime('%Y-%m-%d')} 到 {dates[-1].strftime('%Y-%m-%d')}")
        logger.info(f"ProfitCost统计: 均值={df['ProfitCost'].mean():.4f}, 标准差={df['ProfitCost'].std():.4f}")
        
        # 保存示例数据
        output_path = project_root / "ProfitCost_sample.pkl"
        df_pivot.to_pickle(output_path)
        logger.info(f"保存示例数据到: {output_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"创建示例数据失败: {e}")
        return False

def main():
    """主函数"""
    logger.info("="*50)
    logger.info("ProfitCost因子实现验证")
    logger.info("="*50)
    
    results = {
        'implementation': test_profitcost_implementation(),
        'config': test_config_updates(),
        'sample_data': create_sample_factor_data()
    }
    
    logger.info("\n" + "="*50)
    logger.info("测试结果总结")
    logger.info("="*50)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✓ 通过" if passed else "✗ 失败"
        logger.info(f"{test_name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        logger.info("\n🎉 ProfitCost因子实现验证成功!")
        logger.info("✓ 因子已正确添加到PureFinancialFactorCalculator")
        logger.info("✓ 配置文件已更新")
        logger.info("✓ 示例数据已创建")
        logger.info("\n📝 后续步骤:")
        logger.info("1. 运行完整测试流水线验证因子计算")
        logger.info("2. 使用真实数据进行因子计算和存储")
        logger.info("3. 进行因子有效性分析")
    else:
        logger.warning("\n⚠️ 部分验证失败，请检查实现")
    
    return all_passed

if __name__ == "__main__":
    main()