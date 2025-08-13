"""
pytest配置文件 - 全局测试配置和夹具
"""
import pytest
import pandas as pd
import numpy as np
import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from factors.base.testable_mixin import MockDataProvider


@pytest.fixture(scope="session")
def mock_data_provider():
    """会话级别的模拟数据提供者"""
    return MockDataProvider()


@pytest.fixture(scope="function")
def small_sample_data(mock_data_provider):
    """小规模样本数据 - 用于快速单元测试"""
    return {
        'financial_data': mock_data_provider.create_mock_financial_data(n_stocks=3, n_periods=8),  # 增加期数用于TTM
        'market_cap': mock_data_provider.create_mock_market_cap_data(n_stocks=3, n_days=1000, start_date='2022-01-01'),  # 扩大天数到2024年
        'release_dates': mock_data_provider.create_mock_release_dates(n_stocks=3, n_periods=8),
        'trading_dates': mock_data_provider.create_mock_trading_dates('2022-01-01', '2024-12-31')  # 扩大时间范围
    }


@pytest.fixture(scope="function")
def medium_sample_data(mock_data_provider):
    """中等规模样本数据 - 用于集成测试"""
    return {
        'financial_data': mock_data_provider.create_mock_financial_data(n_stocks=10, n_periods=8),
        'market_cap': mock_data_provider.create_mock_market_cap_data(n_stocks=10, n_days=100),
        'release_dates': mock_data_provider.create_mock_release_dates(n_stocks=10, n_periods=8),
        'trading_dates': mock_data_provider.create_mock_trading_dates('2022-01-01', '2023-12-31')
    }


@pytest.fixture(scope="function")
def large_sample_data(mock_data_provider):
    """大规模样本数据 - 用于性能测试"""
    return {
        'financial_data': mock_data_provider.create_mock_financial_data(n_stocks=50, n_periods=12),
        'market_cap': mock_data_provider.create_mock_market_cap_data(n_stocks=50, n_days=300),
        'release_dates': mock_data_provider.create_mock_release_dates(n_stocks=50, n_periods=12),
        'trading_dates': mock_data_provider.create_mock_trading_dates('2020-01-01', '2023-12-31')
    }


@pytest.fixture(autouse=True)
def set_random_seed():
    """自动设置随机种子，确保测试结果可重现"""
    np.random.seed(42)
    

def pytest_configure(config):
    """pytest配置"""
    # 添加自定义标记
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance tests"
    )


def pytest_collection_modifyitems(config, items):
    """修改测试收集，添加标记"""
    for item in items:
        # 为性能测试添加标记
        if "performance" in str(item.fspath):
            item.add_marker(pytest.mark.performance)
            item.add_marker(pytest.mark.slow)
        
        # 为集成测试添加标记
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
            
        # 为包含大数据集的测试添加slow标记
        if "large" in item.name or "benchmark" in item.name:
            item.add_marker(pytest.mark.slow)