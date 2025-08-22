#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成ProfitCost因子并按配置保存

计算公式：TTM扣非净利润 / (TTM财务费用 + TTM所得税)
反映企业扣非净利润相对于财务成本和税收成本的效率
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
from factors.generator.financial.pure_financial_factors import PureFinancialFactorCalculator
# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def load_financial_data():
    """加载财务数据"""
    logger.info("加载财务数据...")
    
    try:
        # 根据项目结构加载数据
        data_path = project_root / "data" / "auxiliary" / "FinancialData_unified.pkl"
        
        if not data_path.exists():
            logger.error(f"财务数据文件不存在: {data_path}")
            return None
            
        financial_data = pd.read_pickle(data_path)
        logger.info(f"✓ 财务数据加载成功: {financial_data.shape}")
        logger.info(f"  数据范围: {financial_data.index.get_level_values(0).min()} 到 {financial_data.index.get_level_values(0).max()}")
        logger.info(f"  股票数量: {financial_data.index.get_level_values(1).nunique()}")
        
        # 检查必要字段
        required_fields = ['DEDUCTEDPROFIT', 'FIN_EXP_IS', 'TAX', 'd_quarter', 'd_year']
        missing_fields = [field for field in required_fields if field not in financial_data.columns]
        
        if missing_fields:
            logger.error(f"缺少必要字段: {missing_fields}")
            logger.info("可用字段:")
            for i, col in enumerate(financial_data.columns):
                if i < 20:  # 只显示前20个字段
                    logger.info(f"  {col}")
                elif i == 20:
                    logger.info(f"  ... (共{len(financial_data.columns)}个字段)")
                    break
            return None
            
        logger.info("✓ 所有必要字段都存在")
        return financial_data
        
    except Exception as e:
        logger.error(f"加载财务数据失败: {e}")
        return None

def calculate_profitcost_factor(financial_data):
    """计算ProfitCost因子"""
    logger.info("开始计算ProfitCost因子...")
    
    try:
        # 导入计算器
        sys.path.append(str(project_root / "factors" / "generator" / "financial"))
        
        # 由于循环导入问题，我们直接实现计算逻辑
        logger.info("使用内嵌计算逻辑...")
        
        # 准备数据
        calc_data = financial_data[['DEDUCTEDPROFIT', 'FIN_EXP_IS', 'TAX', 'd_quarter']].copy()
        
        def calculate_ttm(data, value_col, quarter_col='d_quarter'):
            """计算TTM"""
            result_list = []
            
            for stock in data.index.get_level_values(1).unique():
                stock_data = data.loc[data.index.get_level_values(1) == stock].copy()
                stock_data = stock_data.sort_index()
                
                stock_result = []
                for i in range(len(stock_data)):
                    if i < 4:
                        stock_result.append(np.nan)
                        continue
                        
                    quarter = stock_data[quarter_col].iloc[i]
                    
                    if quarter == 1:
                        # Q1: 当前Q1 + 去年Q4 - 去年Q1
                        ttm_value = (stock_data[value_col].iloc[i] + 
                                   stock_data[value_col].iloc[i-1] - 
                                   stock_data[value_col].iloc[i-4])
                    elif quarter == 2:
                        # Q2: 当前Q2 + 去年Q4 - 去年Q2
                        ttm_value = (stock_data[value_col].iloc[i] + 
                                   stock_data[value_col].iloc[i-2] - 
                                   stock_data[value_col].iloc[i-4])
                    elif quarter == 3:
                        # Q3: 当前Q3 + 去年Q4 - 去年Q3
                        ttm_value = (stock_data[value_col].iloc[i] + 
                                   stock_data[value_col].iloc[i-3] - 
                                   stock_data[value_col].iloc[i-4])
                    else:
                        # Q4: 直接使用当年数据
                        ttm_value = stock_data[value_col].iloc[i]
                        
                    stock_result.append(ttm_value)
                
                # 创建结果DataFrame
                stock_df = pd.DataFrame({
                    value_col + '_TTM': stock_result
                }, index=stock_data.index)
                
                result_list.append(stock_df)
            
            return pd.concat(result_list)
        
        # 计算各项TTM值
        logger.info("  计算扣非净利润TTM...")
        earnings_ttm = calculate_ttm(calc_data, 'DEDUCTEDPROFIT')
        
        logger.info("  计算财务费用TTM...")
        fin_exp_ttm = calculate_ttm(calc_data, 'FIN_EXP_IS')
        
        logger.info("  计算所得税TTM...")
        tax_ttm = calculate_ttm(calc_data, 'TAX')
        
        # 合并数据
        combined = pd.concat([
            earnings_ttm,
            fin_exp_ttm, 
            tax_ttm
        ], axis=1)
        
        # 计算ProfitCost = 扣非净利润TTM / (财务费用TTM + 所得税TTM)
        logger.info("  计算ProfitCost比率...")
        combined['Total_Cost_TTM'] = combined['FIN_EXP_IS_TTM'] + combined['TAX_TTM']
        
        # 避免除零
        combined['Total_Cost_TTM'] = combined['Total_Cost_TTM'].replace(0, np.nan)
        
        # 计算因子值
        profitcost = combined['DEDUCTEDPROFIT_TTM'] / combined['Total_Cost_TTM']
        profitcost = profitcost.replace([np.inf, -np.inf], np.nan)
        
        # 清理数据
        profitcost = profitcost.dropna()
        
        logger.info(f"✓ 计算完成!")
        logger.info(f"  有效数据点: {len(profitcost)}")
        logger.info(f"  均值: {profitcost.mean():.4f}")
        logger.info(f"  标准差: {profitcost.std():.4f}")
        logger.info(f"  最小值: {profitcost.min():.4f}")
        logger.info(f"  最大值: {profitcost.max():.4f}")
        
        return profitcost
        
    except Exception as e:
        logger.error(f"计算ProfitCost因子失败: {e}")
        logger.exception("详细错误:")
        return None

def save_factor_data(factor_data, factor_name='ProfitCost_ttm'):
    """保存因子数据"""
    logger.info(f"保存{factor_name}因子数据...")
    
    try:
        # 创建输出目录
        output_dir = project_root / "factors_data"
        output_dir.mkdir(exist_ok=True)
        
        # 保存路径
        output_path = output_dir / f"{factor_name}.pkl"
        
        # 保存数据
        factor_data.to_pickle(output_path)
        logger.info(f"✓ 因子数据已保存到: {output_path}")
        
        # 验证保存
        test_load = pd.read_pickle(output_path)
        logger.info(f"✓ 验证加载成功: {len(test_load)}条记录")
        
        # 创建CSV版本（用于查看）
        csv_path = output_dir / f"{factor_name}.csv"
        factor_data.to_csv(csv_path)
        logger.info(f"✓ CSV版本已保存到: {csv_path}")
        
        # 生成统计报告
        report_path = output_dir / f"{factor_name}_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"ProfitCost因子统计报告\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"="*50 + "\n\n")
            f.write(f"数据概况:\n")
            f.write(f"  总记录数: {len(factor_data)}\n")
            f.write(f"  有效记录数: {factor_data.count()}\n")
            f.write(f"  缺失率: {(1 - factor_data.count()/len(factor_data))*100:.2f}%\n")
            f.write(f"  数据范围: {factor_data.index.get_level_values(0).min()} 到 {factor_data.index.get_level_values(0).max()}\n")
            f.write(f"  股票数量: {factor_data.index.get_level_values(1).nunique()}\n\n")
            f.write(f"统计指标:\n")
            f.write(f"  均值: {factor_data.mean():.6f}\n")
            f.write(f"  标准差: {factor_data.std():.6f}\n")
            f.write(f"  最小值: {factor_data.min():.6f}\n")
            f.write(f"  25%分位数: {factor_data.quantile(0.25):.6f}\n")
            f.write(f"  中位数: {factor_data.median():.6f}\n")
            f.write(f"  75%分位数: {factor_data.quantile(0.75):.6f}\n")
            f.write(f"  最大值: {factor_data.max():.6f}\n\n")
            f.write(f"因子说明:\n")
            f.write(f"  名称: ProfitCost (扣非净利润成本效率)\n")
            f.write(f"  计算公式: TTM扣非净利润 / (TTM财务费用 + TTM所得税)\n")
            f.write(f"  经济含义: 反映企业扣非净利润相对于财务成本和税收成本的效率\n")
            f.write(f"  数据源: DEDUCTEDPROFIT, FIN_EXP_IS, TAX\n")
            f.write(f"  计算方法: TTM (Trailing Twelve Months)\n")
            
        logger.info(f"✓ 统计报告已保存到: {report_path}")
        
        return output_path
        
    except Exception as e:
        logger.error(f"保存因子数据失败: {e}")
        return None

def main():
    """主函数"""
    logger.info("="*60)
    logger.info("ProfitCost因子生成和保存")
    logger.info("="*60)
    
    # 1. 加载数据
    financial_data = load_financial_data()
    if financial_data is None:
        logger.error("❌ 财务数据加载失败，退出程序")
        return False
    
    # 2. 计算因子
    profitcost_factor = calculate_profitcost_factor(financial_data)
    if profitcost_factor is None:
        logger.error("❌ 因子计算失败，退出程序")
        return False
    
    # 3. 保存因子
    save_path = save_factor_data(profitcost_factor)
    if save_path is None:
        logger.error("❌ 因子保存失败，退出程序")
        return False
    
    logger.info("="*60)
    logger.info("🎉 ProfitCost因子生成完成!")
    logger.info("="*60)
    logger.info("✓ 因子计算公式: TTM扣非净利润 / (TTM财务费用 + TTM所得税)")
    logger.info("✓ 数据已保存并可用于后续分析")
    logger.info("✓ 可以使用因子测试模块进行有效性验证")
    
    return True

if __name__ == "__main__":
    print('test')
    success = main()
    if not success:
        sys.exit(1)