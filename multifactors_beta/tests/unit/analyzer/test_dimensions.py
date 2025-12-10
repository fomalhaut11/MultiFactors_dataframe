"""
测试维度模块的导入和基本功能
"""

def test_import_dimensions():
    """测试所有维度类的导入"""
    try:
        from dimensions import (
            ProfitabilityDimension,
            StabilityDimension,
            TradabilityDimension,
            UniquenessDimension,
            TimelinesDimension
        )
        print("✓ 所有维度类导入成功")
        
        # 测试初始化
        dimensions = {
            'profitability': ProfitabilityDimension(weight=0.35),
            'stability': StabilityDimension(weight=0.25),
            'tradability': TradabilityDimension(weight=0.20),
            'uniqueness': UniquenessDimension(weight=0.10),
            'timeliness': TimelinesDimension(weight=0.10)
        }
        
        print("\n已初始化的维度：")
        for name, dim in dimensions.items():
            print(f"  - {dim.name}: 权重 {dim.weight:.0%}")
        
        # 测试评分等级
        print("\n评分等级映射测试：")
        test_scores = [95, 85, 75, 65, 55, 45, 35, 25]
        for score in test_scores:
            grade = dimensions['profitability'].get_grade(score)
            print(f"  {score:3d}分 -> {grade}级")
        
        return True
    except Exception as e:
        print(f"✗ 导入或初始化失败: {e}")
        return False

def test_evaluator():
    """测试FactorEvaluator的集成"""
    try:
        from factor_evaluator import FactorEvaluator
        
        # 测试不同场景的初始化
        scenarios = ['balanced', 'high_frequency', 'value_investing', 'risk_neutral']
        
        print("\n\n测试FactorEvaluator：")
        for scenario in scenarios:
            evaluator = FactorEvaluator(scenario=scenario)
            print(f"\n场景: {scenario}")
            print("  维度权重配置：")
            for dim_name, dimension in evaluator.dimensions.items():
                print(f"    - {dim_name}: {dimension.weight:.0%}")
        
        return True
    except Exception as e:
        print(f"✗ FactorEvaluator测试失败: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("因子评估维度模块测试")
    print("=" * 60)
    
    # 运行测试
    success = True
    success = test_import_dimensions() and success
    success = test_evaluator() and success
    
    print("\n" + "=" * 60)
    if success:
        print("✓ 所有测试通过！Phase 2开发完成")
    else:
        print("✗ 部分测试失败，请检查代码")
    print("=" * 60)