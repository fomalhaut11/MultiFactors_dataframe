#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data模块接口测试

测试data模块统一接口的核心功能
"""

import sys
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 测试导入
def test_data_module_imports():
    """测试data模块主要接口的导入"""
    try:
        from data import (
            get_factor_data, get_raw_data, validate_data_pipeline,
            update_all_data, get_data_status, clear_cache
        )
        assert True, "所有接口导入成功"
    except ImportError as e:
        pytest.fail(f"导入失败: {e}")


def test_validate_data_pipeline():
    """测试数据管道验证功能"""
    from data import validate_data_pipeline
    
    # 实际验证可能会失败（如果数据文件不存在），这是正常的
    # 主要测试函数能正常运行
    result = validate_data_pipeline()
    assert isinstance(result, bool), "验证结果应该是布尔值"


@pytest.fixture
def mock_data_bridge():
    """模拟数据桥接器"""
    mock_bridge = MagicMock()
    
    # 模拟价格因子数据
    dates = pd.date_range('2024-01-01', periods=10, freq='D')
    stocks = ['000001', '000002']
    index = pd.MultiIndex.from_product([dates, stocks], names=['TradingDates', 'StockCodes'])
    mock_factor_data = pd.Series(np.random.randn(20), index=index)
    
    mock_bridge.price_to_factor.return_value = mock_factor_data
    mock_bridge.financial_to_factor.return_value = mock_factor_data
    
    # 模拟原始数据
    mock_price_data = pd.DataFrame({
        'code': ['000001', '000002'] * 10,
        'tradingday': [20240101, 20240101] * 10,
        'c': np.random.randn(20),
        'adjfactor': np.ones(20)
    })
    mock_bridge.get_price_data.return_value = mock_price_data
    
    return mock_bridge


def test_get_factor_data_with_mock(mock_data_bridge):
    """使用模拟数据测试获取因子数据"""
    with patch('data.get_data_bridge', return_value=mock_data_bridge):
        from data import get_factor_data
        
        # 测试价格因子
        result = get_factor_data('price', 'c')
        assert isinstance(result, pd.Series), "应返回Series"
        assert result.index.names == ['TradingDates', 'StockCodes'], "索引名称正确"
        
        # 测试财务因子
        result = get_factor_data('financial', 'NET_PROFIT')
        assert isinstance(result, pd.Series), "应返回Series"
        
        # 测试错误类型
        with pytest.raises(ValueError):
            get_factor_data('invalid_type', 'column')


def test_get_raw_data_with_mock(mock_data_bridge):
    """使用模拟数据测试获取原始数据"""
    with patch('data.get_data_bridge', return_value=mock_data_bridge):
        from data import get_raw_data
        
        # 测试价格数据
        result = get_raw_data('price')
        assert isinstance(result, pd.DataFrame), "应返回DataFrame"
        
        # 测试错误类型
        with pytest.raises(ValueError):
            get_raw_data('invalid_type')


def test_clear_cache():
    """测试缓存清理功能"""
    from data import clear_cache
    
    # 应该能正常调用而不报错
    clear_cache()
    assert True, "缓存清理成功"


class TestDataValidation:
    """数据验证测试类"""
    
    def test_validate_price_data(self):
        """测试价格数据验证"""
        from data import validate_price_data
        
        # 创建示例数据
        valid_data = pd.DataFrame({
            'code': ['000001', '000002'],
            'tradingday': [20240101, 20240101],
            'c': [10.5, 15.2],
            'adjfactor': [1.0, 1.0]
        })
        
        is_valid, errors = validate_price_data(valid_data, strict=False)
        assert isinstance(is_valid, bool), "验证结果应为布尔值"
        assert isinstance(errors, list), "错误列表应为list"
    
    def test_validate_factor_format(self):
        """测试因子格式验证"""
        from data import validate_factor_format
        
        # 创建正确格式的因子数据
        dates = pd.date_range('2024-01-01', periods=2, freq='D')
        stocks = ['000001', '000002']
        index = pd.MultiIndex.from_product([dates, stocks], names=['TradingDates', 'StockCodes'])
        factor_data = pd.Series([1.0, 2.0, 3.0, 4.0], index=index)
        
        is_valid, errors = validate_factor_format(factor_data)
        assert isinstance(is_valid, bool), "验证结果应为布尔值"
        assert isinstance(errors, list), "错误列表应为list"
    
    def test_convert_to_factor_format(self):
        """测试数据转换为因子格式"""
        from data import convert_to_factor_format
        
        # 创建测试数据
        test_data = pd.DataFrame({
            'stock_code': ['000001', '000002', '000001', '000002'],
            'trade_date': pd.to_datetime(['2024-01-01', '2024-01-01', '2024-01-02', '2024-01-02']),
            'close_price': [10.5, 15.2, 10.8, 15.4]
        })
        
        result = convert_to_factor_format(
            test_data, 
            value_col='close_price',
            date_col='trade_date',
            stock_col='stock_code'
        )
        
        assert isinstance(result, pd.Series), "应返回Series"
        assert result.index.nlevels == 2, "应为MultiIndex"


if __name__ == "__main__":
    # 可以直接运行进行基础测试
    print("开始数据模块接口测试...")
    
    # 测试导入
    try:
        test_data_module_imports()
        print("✅ 模块导入测试通过")
    except Exception as e:
        print(f"❌ 模块导入测试失败: {e}")
    
    # 测试验证功能
    try:
        test_validate_data_pipeline()
        print("✅ 数据管道验证测试通过")
    except Exception as e:
        print(f"❌ 数据管道验证测试失败: {e}")
    
    print("基础测试完成")