"""
多因子量化系统测试包

包含单元测试、集成测试和性能测试
"""

__version__ = "2.0.0"
__author__ = "Multi-Factor Team"

# 测试配置
TEST_CONFIG = {
    'default_test_stocks': 5,
    'default_test_periods': 4,
    'performance_test_stocks': 50,
    'performance_test_periods': 12,
}

# 导出测试工具
from factors.base.testable_mixin import MockDataProvider, FactorTestSuite

__all__ = [
    'MockDataProvider',
    'FactorTestSuite',
    'TEST_CONFIG'
]