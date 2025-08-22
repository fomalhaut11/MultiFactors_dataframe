#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试字段映射功能集成
"""
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from factors.generator.financial.experimental_factors import ExperimentalFactorCalculator
from factors.config.field_mapper import get_field_mapper


def test_field_mapping_basic():
    """测试基础字段映射功能"""
    print("🔍 测试基础字段映射功能")
    print("=" * 60)
    
    mapper = get_field_mapper()
    
    # 测试常用字段
    test_fields = [
        'DEDUCTEDPROFIT',  # 扣非净利润
        'TOT_OPER_REV',    # 营业收入  
        'NETCASH_OPER',    # 经营现金流
        'FIN_EXP_IS',      # 财务费用
        'INVALID_FIELD'    # 不存在的字段
    ]
    
    for field in test_fields:
        info = mapper.get_field_info(field)
        if info:
            print(f"✅ {field}")
            print(f"   中文名: {info['chinese_name']}")
            print(f"   所属表: {info['table_chinese'] or info['table']}")
        else:
            print(f"❌ {field} - 未找到")
    
    print()


def test_experimental_calculator_integration():
    """测试实验性计算器的字段映射集成"""
    print("🧪 测试实验性计算器的字段映射集成")
    print("=" * 60)
    
    calculator = ExperimentalFactorCalculator()
    
    # 测试字段验证功能
    test_fields = [
        'DEDUCTEDPROFIT',
        'TOT_OPER_REV', 
        'NETCASH_OPER',
        'UNKNOWN_FIELD',
        'd_quarter'
    ]
    
    print("📋 字段验证结果:")
    calculator.print_field_usage_report(test_fields)
    
    # 测试搜索功能
    print("\n🔍 搜索包含'利润'的字段:")
    profit_fields = calculator.search_similar_fields('利润', 5)
    for field in profit_fields:
        print(f"   {field['field_name']} -> {field['chinese_name']}")
    
    print("\n🔍 搜索包含'CASH'的字段:")
    cash_fields = calculator.search_similar_fields('CASH', 5)
    for field in cash_fields:
        print(f"   {field['field_name']} -> {field['chinese_name']}")


def test_profitcost_factor_fields():
    """测试profitcost因子所需字段"""
    print("\n💰 测试profitcost因子字段需求")
    print("=" * 60)
    
    calculator = ExperimentalFactorCalculator()
    
    # profitcost = TTM扣非净利润/(TTM财务费用+TTM所得税)
    profitcost_fields = [
        'DEDUCTEDPROFIT',  # 扣非净利润
        'FIN_EXP_IS',      # 财务费用
        'TAX',             # 所得税
        'd_quarter'        # 季度
    ]
    
    print("ProfitCost因子所需字段:")
    calculator.print_field_usage_report(profitcost_fields)
    
    # 如果有缺失字段，搜索替代方案
    results = calculator.validate_and_explain_fields(profitcost_fields)
    if results['missing_fields']:
        print("\n🔍 寻找替代字段:")
        for missing_field in results['missing_fields']:
            if 'TAX' in missing_field:
                alternatives = calculator.search_similar_fields('税', 3)
                print(f"   {missing_field} 的可能替代字段:")
                for alt in alternatives:
                    print(f"     {alt['field_name']} -> {alt['chinese_name']}")


def test_field_export():
    """测试字段导出功能"""
    print("\n📤 测试字段导出功能")
    print("=" * 60)
    
    mapper = get_field_mapper()
    
    # 导出部分字段列表
    output_path = project_root / "field_list_sample.xlsx"
    success = mapper.export_field_list(output_path, format='excel')
    
    if success:
        print(f"✅ 字段列表已导出到: {output_path}")
    else:
        print("❌ 字段列表导出失败")


def main():
    """主测试函数"""
    print("🧪 字段映射功能集成测试")
    print("=" * 80)
    
    try:
        # 基础功能测试
        test_field_mapping_basic()
        
        # 集成功能测试
        test_experimental_calculator_integration()
        
        # 实际应用测试
        test_profitcost_factor_fields()
        
        # 导出功能测试
        test_field_export()
        
        print("\n🎉 所有测试完成!")
        print("=" * 80)
        print("💡 使用提示:")
        print("1. 使用 calculator.print_field_usage_report(fields) 验证字段")
        print("2. 使用 calculator.search_similar_fields(keyword) 搜索字段")
        print("3. 字段映射配置文件: factors/config/field_mapping.yaml")
        print("4. 使用 mapper.export_field_list() 导出完整字段列表")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()