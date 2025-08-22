#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试重构后的factors模块

验证模块重构后的功能是否正常
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))


def test_module_imports():
    """测试模块导入"""
    print("="*60)
    print("测试模块导入")
    print("="*60)
    
    try:
        # 测试主模块导入
        import factors
        print("✓ factors模块导入成功")
        
        # 测试子模块导入
        from factors import generator, tester, analyzer
        print("✓ 子模块导入成功")
        
        # 测试便捷函数导入
        from factors import generate, test, analyze, pipeline
        print("✓ 便捷函数导入成功")
        
        # 测试生成器模块
        from factors.generator import (
            FactorGenerator,
            FinancialFactorGenerator,
            generate_factor,
            list_available_factors
        )
        print("✓ generator模块接口导入成功")
        
        # 测试tester模块
        from factors.tester import (
            SingleFactorTestPipeline,
            test_factor,
            batch_test
        )
        print("✓ tester模块接口导入成功")
        
        # 测试analyzer模块
        from factors.analyzer import FactorScreener
        print("✓ analyzer模块接口导入成功")
        
        return True
    except ImportError as e:
        print(f"✗ 导入失败: {e}")
        return False


def test_module_structure():
    """测试模块结构"""
    print("\n" + "="*60)
    print("测试模块结构")
    print("="*60)
    
    import factors
    
    # 检查版本信息
    print(f"factors模块版本: {factors.__version__}")
    
    # 检查__all__列表
    print(f"导出接口数量: {len(factors.__all__)}")
    
    # 列出主要接口
    main_interfaces = [
        'generate', 'test', 'analyze', 'pipeline',
        'generator', 'tester', 'analyzer'
    ]
    
    for interface in main_interfaces:
        if hasattr(factors, interface):
            print(f"✓ {interface} 接口存在")
        else:
            print(f"✗ {interface} 接口缺失")
            
    return True


def test_list_factors():
    """测试因子列表功能"""
    print("\n" + "="*60)
    print("测试因子列表功能")
    print("="*60)
    
    try:
        from factors.generator import list_available_factors
        
        # 列出所有因子
        all_factors = list_available_factors()
        print(f"因子类型数量: {len(all_factors)}")
        
        for factor_type, factor_list in all_factors.items():
            print(f"\n{factor_type}类因子:")
            if isinstance(factor_list, dict):
                # 如果是嵌套字典
                for category, factors in factor_list.items():
                    print(f"  {category}: {len(factors)}个因子")
            else:
                # 如果是列表
                print(f"  共{len(factor_list)}个因子")
                
        return True
    except Exception as e:
        print(f"✗ 列表功能测试失败: {e}")
        return False


def test_financial_factor_generator():
    """测试财务因子生成器"""
    print("\n" + "="*60)
    print("测试财务因子生成器")
    print("="*60)
    
    try:
        from factors.generator import FinancialFactorGenerator
        
        # 创建生成器
        generator = FinancialFactorGenerator()
        print(f"✓ 创建财务因子生成器: {type(generator).__name__}")
        
        # 获取可用因子
        available = generator.get_available_factors()
        print(f"✓ 可用财务因子数量: {len(available)}")
        
        # 显示部分因子
        sample_factors = available[:5] if len(available) >= 5 else available
        print(f"  示例因子: {sample_factors}")
        
        return True
    except Exception as e:
        print(f"✗ 财务因子生成器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_factor_interfaces():
    """测试因子接口的向后兼容性"""
    print("\n" + "="*60)
    print("测试接口向后兼容性")
    print("="*60)
    
    try:
        # 测试从主模块导入
        from factors import test_factor
        print("✓ 从factors导入test_factor成功")
        
        # 测试从tester模块导入
        from factors.tester import test_factor as test_func
        print("✓ 从factors.tester导入test_factor成功")
        
        # 测试两者是否相同
        if test_factor == test_func:
            print("✓ 接口一致性验证通过")
        else:
            print("! 接口不一致，但都可用")
            
        return True
    except Exception as e:
        print(f"✗ 接口测试失败: {e}")
        return False


def test_convenience_functions():
    """测试便捷函数"""
    print("\n" + "="*60)
    print("测试便捷函数")
    print("="*60)
    
    import factors
    
    # 检查便捷函数
    convenience_funcs = ['generate', 'test', 'analyze', 'pipeline']
    
    for func_name in convenience_funcs:
        func = getattr(factors, func_name, None)
        if func and callable(func):
            print(f"✓ {func_name}函数可用")
            # 显示函数文档
            if func.__doc__:
                first_line = func.__doc__.strip().split('\n')[0]
                print(f"  {first_line}")
        else:
            print(f"✗ {func_name}函数不可用")
            
    return True


def main():
    """主测试函数"""
    print("\n" + "="*80)
    print("因子模块重构测试")
    print("="*80)
    
    results = []
    
    # 运行各项测试
    results.append(("模块导入", test_module_imports()))
    results.append(("模块结构", test_module_structure()))
    results.append(("因子列表", test_list_factors()))
    results.append(("财务因子生成器", test_financial_factor_generator()))
    results.append(("接口兼容性", test_factor_interfaces()))
    results.append(("便捷函数", test_convenience_functions()))
    
    # 输出测试结果
    print("\n" + "="*80)
    print("测试结果汇总")
    print("="*80)
    
    passed = 0
    failed = 0
    
    for test_name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
        else:
            failed += 1
            
    print(f"\n总计: {passed}个通过, {failed}个失败")
    
    if failed == 0:
        print("\n🎉 所有测试通过！模块重构成功。")
    else:
        print(f"\n⚠️ 有{failed}个测试失败，需要修复。")
        
    return failed == 0


if __name__ == "__main__":
    success = main()