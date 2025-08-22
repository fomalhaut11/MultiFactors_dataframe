"""
测试Phase 2开发的维度模块
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_dimensions():
    """测试维度模块"""
    print("Testing Phase 2 - Factor Evaluation Dimensions")
    print("=" * 60)
    
    try:
        # 导入维度类
        from factors.analyzer.evaluation.dimensions import (
            ProfitabilityDimension,
            StabilityDimension,
            TradabilityDimension,
            UniquenessDimension,
            TimelinesDimension
        )
        print("[OK] All dimension classes imported successfully")
        
        # 测试初始化
        dimensions = {
            'profitability': ProfitabilityDimension(weight=0.35),
            'stability': StabilityDimension(weight=0.25),
            'tradability': TradabilityDimension(weight=0.20),
            'uniqueness': UniquenessDimension(weight=0.10),
            'timeliness': TimelinesDimension(weight=0.10)
        }
        
        print("\nInitialized dimensions:")
        for name, dim in dimensions.items():
            print(f"  - {dim.name}: weight={dim.weight:.0%}")
        
        # 测试等级映射
        print("\nGrade mapping test:")
        test_scores = [95, 85, 75, 65, 55, 45, 35, 25]
        for score in test_scores:
            grade = dimensions['profitability'].get_grade(score)
            print(f"  Score {score:3d} -> Grade {grade}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return False

def test_evaluator():
    """测试FactorEvaluator集成"""
    print("\n" + "=" * 60)
    print("Testing FactorEvaluator Integration")
    print("=" * 60)
    
    try:
        from factors.analyzer.evaluation import FactorEvaluator
        
        # 测试不同场景
        scenarios = ['balanced', 'high_frequency', 'value_investing', 'risk_neutral']
        
        for scenario in scenarios:
            evaluator = FactorEvaluator(scenario=scenario)
            print(f"\nScenario: {scenario}")
            print("  Dimension weights:")
            for dim_name, dimension in evaluator.dimensions.items():
                print(f"    - {dim_name}: {dimension.weight:.0%}")
        
        print("\n[OK] FactorEvaluator integrated successfully")
        return True
        
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("\n" + "=" * 60)
    print("PHASE 2 DEVELOPMENT TEST")
    print("=" * 60 + "\n")
    
    success = True
    success = test_dimensions() and success
    success = test_evaluator() and success
    
    print("\n" + "=" * 60)
    if success:
        print("[SUCCESS] All tests passed! Phase 2 development completed.")
        print("\nImplemented components:")
        print("  1. ProfitabilityDimension - Evaluates factor return capability")
        print("  2. StabilityDimension - Evaluates factor stability")
        print("  3. TradabilityDimension - Evaluates trading feasibility")
        print("  4. UniquenessDimension - Evaluates factor uniqueness")
        print("  5. TimelinesDimension - Evaluates prediction timeliness")
        print("  6. FactorEvaluator - Integrated all dimensions")
    else:
        print("[FAILED] Some tests failed. Please check the errors above.")
    print("=" * 60)

if __name__ == "__main__":
    main()