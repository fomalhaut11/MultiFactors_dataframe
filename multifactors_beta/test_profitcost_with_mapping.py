#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用字段映射实现ProfitCost因子
展示如何利用字段映射功能开发新因子
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from factors.generator.financial.experimental_factors import ExperimentalFactorCalculator
from factors.base.time_series_processor import TimeSeriesProcessor


class ProfitCostFactorWithMapping(ExperimentalFactorCalculator):
    """使用字段映射的ProfitCost因子计算器"""
    
    def calculate_EXPERIMENTAL_ProfitCost_ttm(self, financial_data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        实验性因子：ProfitCost (盈利成本比)
        
        计算公式：TTM扣非净利润 / (TTM财务费用 + TTM所得税)
        经济含义：衡量企业盈利相对于财务成本和税负的效率
        假设验证：盈利成本比高的企业应该有更好的投资价值
        
        使用字段映射功能自动验证和说明所需字段
        """
        print("🔍 ProfitCost因子字段验证和说明:")
        print("-" * 50)
        
        # 定义所需字段（使用字段映射验证）
        required_fields = [
            'DEDUCTEDPROFIT',  # 扣非净利润
            'FIN_EXP_IS',      # 财务费用  
            'TAX',             # 所得税
            'd_quarter'        # 季度
        ]
        
        # 验证字段并显示说明
        self.print_field_usage_report(required_fields)
        
        try:
            # 验证数据需求
            if not self.validate_data_requirements(financial_data, required_fields):
                raise ValueError("Required data not available for ProfitCost calculation")
            
            # 提取数据
            extracted_data = self.extract_required_data(financial_data, required_fields)
            
            print("\n📊 开始计算TTM指标...")
            
            # 1. 计算TTM扣非净利润
            earnings_data = extracted_data[['DEDUCTEDPROFIT', 'd_quarter']].copy()
            earnings_ttm = TimeSeriesProcessor.calculate_ttm(earnings_data)
            earnings_series = earnings_ttm.iloc[:, 0] if earnings_ttm.shape[1] > 0 else pd.Series(dtype=float)
            
            # 2. 计算TTM财务费用
            fin_exp_data = extracted_data[['FIN_EXP_IS', 'd_quarter']].copy()
            fin_exp_ttm = TimeSeriesProcessor.calculate_ttm(fin_exp_data)
            fin_exp_series = fin_exp_ttm.iloc[:, 0] if fin_exp_ttm.shape[1] > 0 else pd.Series(dtype=float)
            
            # 3. 计算TTM所得税
            tax_data = extracted_data[['TAX', 'd_quarter']].copy()
            tax_ttm = TimeSeriesProcessor.calculate_ttm(tax_data)
            tax_series = tax_ttm.iloc[:, 0] if tax_ttm.shape[1] > 0 else pd.Series(dtype=float)
            
            # 4. 计算成本总额 = 财务费用 + 所得税
            total_cost = fin_exp_series.abs() + tax_series.abs()  # 取绝对值避免负数影响
            
            # 5. 计算ProfitCost = 净利润 / 总成本
            profitcost = self._safe_division(earnings_series, total_cost)
            
            # 清理异常值
            profitcost = profitcost.replace([np.inf, -np.inf], np.nan)
            
            print(f"✅ ProfitCost因子计算完成:")
            print(f"   数据点数: {len(profitcost):,}")
            print(f"   有效数据: {profitcost.count():,}")
            print(f"   均值: {profitcost.mean():.4f}")
            print(f"   标准差: {profitcost.std():.4f}")
            
            return profitcost
            
        except Exception as e:
            print(f"❌ ProfitCost计算失败: {e}")
            return pd.Series(index=financial_data.index, dtype=float)


def test_profitcost_factor():
    """测试ProfitCost因子"""
    print("💰 测试ProfitCost因子实现")
    print("=" * 80)
    
    # 创建计算器
    calculator = ProfitCostFactorWithMapping()
    
    # 创建模拟数据
    print("\n📝 创建模拟财务数据...")
    dates = pd.date_range('2020-03-31', periods=16, freq='Q')
    stocks = [f'00000{i}.SZ' for i in range(1, 6)]  # 5只股票
    
    # 创建MultiIndex
    index_tuples = [(date, stock) for date in dates for stock in stocks]
    multi_index = pd.MultiIndex.from_tuples(index_tuples, names=['ReportDates', 'StockCodes'])
    
    # 生成模拟数据
    np.random.seed(42)
    n_records = len(multi_index)
    
    data = {
        'DEDUCTEDPROFIT': np.random.normal(100, 30, n_records),  # 扣非净利润
        'FIN_EXP_IS': np.random.normal(20, 10, n_records),       # 财务费用
        'TAX': np.random.normal(15, 8, n_records),               # 所得税
        'd_quarter': [((date.month - 1) // 3) + 1 for date in dates for _ in stocks]
    }
    
    financial_data = pd.DataFrame(data, index=multi_index)
    print(f"模拟数据创建完成: {financial_data.shape}")
    
    # 计算ProfitCost因子
    print("\n🚀 计算ProfitCost因子...")
    profitcost_factor = calculator.calculate_EXPERIMENTAL_ProfitCost_ttm(financial_data)
    
    # 快速验证
    print("\n🔍 因子验证...")
    validation_result = calculator.quick_validate_factor(profitcost_factor, 'ProfitCost')
    
    # 显示部分结果
    print("\n📋 部分计算结果:")
    print(profitcost_factor.dropna().head(10))
    
    return profitcost_factor


def test_field_search_demo():
    """演示字段搜索功能"""
    print("\n🔍 字段搜索功能演示")
    print("=" * 80)
    
    calculator = ProfitCostFactorWithMapping()
    
    # 搜索与成本相关的字段
    print("搜索包含'费用'的字段:")
    cost_fields = calculator.search_similar_fields('费用', 8)
    for field in cost_fields:
        print(f"   {field['field_name']} -> {field['chinese_name']} ({field['table_chinese']})")
    
    print("\n搜索包含'税'的字段:")
    tax_fields = calculator.search_similar_fields('税', 5)
    for field in tax_fields:
        print(f"   {field['field_name']} -> {field['chinese_name']} ({field['table_chinese']})")
    
    print("\n搜索包含'PROFIT'的字段:")
    profit_fields = calculator.search_similar_fields('PROFIT', 8)
    for field in profit_fields:
        print(f"   {field['field_name']} -> {field['chinese_name']} ({field['table_chinese']})")


def main():
    """主函数"""
    print("💰 ProfitCost因子开发演示")
    print("使用字段映射功能进行因子开发")
    print("=" * 80)
    
    try:
        # 测试ProfitCost因子
        profitcost_factor = test_profitcost_factor()
        
        # 演示字段搜索
        test_field_search_demo()
        
        print("\n🎉 演示完成!")
        print("=" * 80)
        print("💡 字段映射功能的优势:")
        print("1. 自动验证字段存在性，避免运行时错误")
        print("2. 提供中文字段说明，便于理解业务含义")
        print("3. 支持字段搜索，快速找到相关字段")
        print("4. 标准化字段使用，提高代码可维护性")
        print("5. 生成字段使用报告，便于文档化")
        
    except Exception as e:
        print(f"❌ 演示失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()