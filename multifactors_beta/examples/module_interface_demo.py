#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模块接口设计示例

演示如何通过模块的__init__.py接口进行调用，
以及这种设计如何支持内部重构而不影响外部使用
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))


def good_practice_example():
    """良好实践：通过模块接口调用"""
    print("="*60)
    print("良好实践示例：通过模块接口调用")
    print("="*60)
    
    # 1. 导入单因子测试模块的公共接口
    from factors.tester import (
        SingleFactorTestPipeline,
        TestResult,
        ResultManager
    )
    
    print("✓ 从 factors.tester 导入公共接口")
    print("  - SingleFactorTestPipeline")
    print("  - TestResult")
    print("  - ResultManager")
    
    # 2. 导入因子分析模块的公共接口
    from factors.analyzer import (
        FactorScreener,
        get_analyzer_config
    )
    
    print("\n✓ 从 factors.analyzer 导入公共接口")
    print("  - FactorScreener")
    print("  - get_analyzer_config")
    
    # 3. 导入基础模块的公共接口
    from factors.base import (
        FactorBase,
        TimeSeriesProcessor,
        DataProcessingMixin
    )
    
    print("\n✓ 从 factors.base 导入公共接口")
    print("  - FactorBase")
    print("  - TimeSeriesProcessor")
    print("  - DataProcessingMixin")
    
    # 4. 使用这些接口
    print("\n使用示例:")
    
    # 创建测试流水线
    pipeline = SingleFactorTestPipeline()
    print(f"  创建了测试流水线: {type(pipeline).__name__}")
    
    # 创建筛选器
    screener = FactorScreener()
    print(f"  创建了因子筛选器: {type(screener).__name__}")
    
    # 创建时间序列处理器
    processor = TimeSeriesProcessor()
    print(f"  创建了时间序列处理器: {type(processor).__name__}")
    
    print("\n优点:")
    print("  1. 不需要知道内部文件结构")
    print("  2. 导入语句简洁清晰")
    print("  3. 内部可以自由重构")


def bad_practice_example():
    """不良实践：直接访问内部实现"""
    print("\n" + "="*60)
    print("不良实践示例：直接访问内部实现")
    print("="*60)
    
    # 这些是不好的做法
    print("✗ 直接从内部路径导入:")
    print("  from factors.tester.core.pipeline import SingleFactorTestPipeline")
    print("  from factors.tester.core.result_manager import ResultManager")
    print("  from factors.analyzer.screening.factor_screener import FactorScreener")
    
    print("\n问题:")
    print("  1. 暴露了内部实现细节")
    print("  2. 如果内部文件重组，所有调用代码都要修改")
    print("  3. 导入路径冗长")
    print("  4. 违反了封装原则")


def refactoring_example():
    """重构示例：展示接口稳定性"""
    print("\n" + "="*60)
    print("重构示例：接口保持稳定")
    print("="*60)
    
    print("假设我们要重构 factors.tester 模块:")
    print("\n原始结构:")
    print("  factors/tester/")
    print("    core/")
    print("      pipeline.py  <- SingleFactorTestPipeline")
    print("      result_manager.py  <- ResultManager")
    
    print("\n重构后的结构:")
    print("  factors/tester/")
    print("    engine/")
    print("      test_pipeline.py  <- SingleFactorTestPipeline")
    print("    managers/")
    print("      results.py  <- ResultManager")
    
    print("\n只需要修改 __init__.py:")
    print("  # 原来")
    print("  from .core.pipeline import SingleFactorTestPipeline")
    print("  from .core.result_manager import ResultManager")
    print("\n  # 改为")
    print("  from .engine.test_pipeline import SingleFactorTestPipeline")
    print("  from .managers.results import ResultManager")
    
    print("\n✓ 外部调用代码完全不需要修改！")
    print("  from factors.tester import SingleFactorTestPipeline  # 保持不变")


def convenience_functions_example():
    """便捷函数示例"""
    print("\n" + "="*60)
    print("便捷函数示例")
    print("="*60)
    
    print("模块可以在 __init__.py 中提供便捷函数:")
    
    print("\n示例1: factors.tester 的便捷函数")
    print("""
# factors/tester/__init__.py
def test_factor(factor_name, **kwargs):
    '''快速测试单个因子'''
    pipeline = SingleFactorTestPipeline()
    return pipeline.run(factor_name, **kwargs)

# 使用
from factors.tester import test_factor
result = test_factor('BP')
""")
    
    print("\n示例2: factors.analyzer 的便捷函数")
    print("""
# factors/analyzer/__init__.py
def quick_screen(preset='normal'):
    '''快速筛选因子'''
    screener = FactorScreener()
    return screener.screen_factors(preset=preset)

# 使用
from factors.analyzer import quick_screen
top_factors = quick_screen('strict')
""")


def api_versioning_example():
    """API版本管理示例"""
    print("\n" + "="*60)
    print("API版本管理示例")
    print("="*60)
    
    print("通过 __init__.py 管理API版本:")
    
    print("\n1. 添加新功能（向后兼容）:")
    print("""
# __init__.py
__all__ = [
    'OldClass',       # 保留
    'NewClass',       # 新增
    'ImprovedClass',  # 新增
]
""")
    
    print("\n2. 废弃旧接口:")
    print("""
# __init__.py
import warnings

def deprecated_function():
    warnings.warn(
        "deprecated_function将在v2.0中移除，请使用new_function",
        DeprecationWarning,
        stacklevel=2
    )
    return new_function()
""")
    
    print("\n3. 兼容性别名:")
    print("""
# __init__.py
# 为了向后兼容，提供旧名称的别名
OldName = NewName  # 旧名称指向新实现
""")


def main():
    """主函数"""
    print("\n" + "="*80)
    print("模块接口设计演示")
    print("="*80)
    
    # 1. 良好实践
    good_practice_example()
    
    # 2. 不良实践
    bad_practice_example()
    
    # 3. 重构示例
    refactoring_example()
    
    # 4. 便捷函数
    convenience_functions_example()
    
    # 5. API版本管理
    api_versioning_example()
    
    print("\n" + "="*80)
    print("总结")
    print("="*80)
    print("""
关键要点:
1. 所有公共接口都应该在 __init__.py 中导出
2. 使用 __all__ 明确声明公共接口
3. 内部实现可以自由重构，只要保持接口稳定
4. 提供便捷函数让模块更易用
5. 通过 __init__.py 管理API版本和向后兼容

这种设计模式的好处:
✓ 清晰的API边界
✓ 更好的封装性
✓ 支持内部重构
✓ 简化导入语句
✓ 便于版本管理
""")


if __name__ == "__main__":
    main()