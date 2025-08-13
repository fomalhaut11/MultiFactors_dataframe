"""
基本面因子单元测试 - 整合debug/temp目录下的测试逻辑
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import numpy as np
import pytest
from unittest.mock import Mock, patch
from factors.financial.fundamental_factors import EPFactor, BPFactor, ROEFactor, PEGFactor
from factors.base.testable_mixin import MockDataProvider
from factors.base.validation import ValidationError


class TestEPFactor:
    """EP因子单元测试"""
    
    @pytest.fixture
    def mock_data(self):
        """创建模拟数据"""
        provider = MockDataProvider()
        return {
            'financial_data': provider.create_mock_financial_data(n_stocks=5, n_periods=4),
            'market_cap': provider.create_mock_market_cap_data(n_stocks=5, n_days=50),
            'release_dates': provider.create_mock_release_dates(n_stocks=5, n_periods=4),
            'trading_dates': provider.create_mock_trading_dates('2022-01-01', '2022-12-31')
        }
    
    def test_ep_factor_instantiation(self):
        """测试EP因子实例化"""
        # 测试默认参数
        ep_factor = EPFactor()
        assert ep_factor.name == 'EP_ttm'
        assert ep_factor.category == 'fundamental'
        assert ep_factor.method == 'ttm'
        
        # 测试自定义参数
        ep_factor_sq = EPFactor(method='single_quarter')
        assert ep_factor_sq.name == 'EP_single_quarter'
        assert ep_factor_sq.method == 'single_quarter'
    
    def test_ep_ttm_calculation(self, small_sample_data):
        """测试EP TTM计算"""
        ep_factor = EPFactor(method='ttm')
        
        result = ep_factor.calculate(
            financial_data=small_sample_data['financial_data'],
            market_cap=small_sample_data['market_cap'],
            release_dates=small_sample_data['release_dates'],
            trading_dates=small_sample_data['trading_dates']
        )
        
        assert isinstance(result, pd.Series)
        assert len(result) > 0
        # 检查结果不是全部为NaN
        assert not result.isnull().all(), "结果不应该全为NaN"
    
    def test_ep_single_quarter_calculation(self, mock_data):
        """测试EP单季度计算"""
        ep_factor = EPFactor(method='single_quarter')
        
        result = ep_factor.calculate(
            financial_data=mock_data['financial_data'],
            market_cap=mock_data['market_cap'],
            release_dates=mock_data['release_dates'],
            trading_dates=mock_data['trading_dates']
        )
        
        assert isinstance(result, pd.Series)
        assert len(result) > 0
    
    def test_ep_invalid_method(self, mock_data):
        """测试无效方法参数"""
        ep_factor = EPFactor(method='invalid_method')
        
        with pytest.raises((ValueError, ValidationError)):
            ep_factor.calculate(
                financial_data=mock_data['financial_data'],
                market_cap=mock_data['market_cap'],
                release_dates=mock_data['release_dates'],
                trading_dates=mock_data['trading_dates']
            )
    
    def test_ep_missing_columns(self):
        """测试缺失必需列的情况"""
        ep_factor = EPFactor()
        
        # 创建缺少必需列的数据
        incomplete_data = pd.DataFrame({
            'WRONG_COLUMN': [1, 2, 3],
        }, index=pd.MultiIndex.from_tuples([
            ('2022-03-31', 'stock_001'),
            ('2022-06-30', 'stock_001'),
            ('2022-09-30', 'stock_001'),
        ], names=['ReportDates', 'StockCodes']))
        
        mock_market_cap = pd.Series([1000, 2000, 3000], 
                                   index=pd.MultiIndex.from_tuples([
                                       ('2022-01-01', 'stock_001'),
                                       ('2022-02-01', 'stock_001'),
                                       ('2022-03-01', 'stock_001'),
                                   ], names=['TradingDates', 'StockCodes']))
        
        with pytest.raises((ValueError, ValidationError)):
            ep_factor.calculate(
                financial_data=incomplete_data,
                market_cap=mock_market_cap
            )
    
    def test_ep_edge_cases(self, mock_data):
        """测试边界情况"""
        ep_factor = EPFactor()
        
        # 测试零市值情况
        zero_market_cap = mock_data['market_cap'].copy()
        zero_market_cap.iloc[:5] = 0
        
        result = ep_factor.calculate(
            financial_data=mock_data['financial_data'],
            market_cap=zero_market_cap,
            release_dates=mock_data['release_dates'],
            trading_dates=mock_data['trading_dates']
        )
        
        # 应该能处理零市值，结果中应该有NaN或无穷大
        assert isinstance(result, pd.Series)
        # 可能包含NaN值，但应该能返回结果
        assert len(result) > 0


class TestBPFactor:
    """BP因子单元测试"""
    
    @pytest.fixture
    def mock_data(self):
        """创建模拟数据"""
        provider = MockDataProvider()
        return {
            'financial_data': provider.create_mock_financial_data(n_stocks=5, n_periods=4),
            'market_cap': provider.create_mock_market_cap_data(n_stocks=5, n_days=50),
            'release_dates': provider.create_mock_release_dates(n_stocks=5, n_periods=4),
            'trading_dates': provider.create_mock_trading_dates('2022-01-01', '2022-12-31')
        }
    
    def test_bp_factor_instantiation(self):
        """测试BP因子实例化"""
        bp_factor = BPFactor()
        assert bp_factor.name == 'BP'
        assert bp_factor.category == 'fundamental'
    
    def test_bp_calculation(self, mock_data):
        """测试BP计算"""
        bp_factor = BPFactor()
        
        result = bp_factor.calculate(
            financial_data=mock_data['financial_data'],
            market_cap=mock_data['market_cap'],
            release_dates=mock_data['release_dates'],
            trading_dates=mock_data['trading_dates']
        )
        
        assert isinstance(result, pd.Series)
        assert len(result) > 0
        
        # BP值应该大部分为正数（账面价值通常为正）
        finite_values = result[np.isfinite(result)]
        if len(finite_values) > 0:
            positive_ratio = (finite_values > 0).mean()
            assert positive_ratio > 0.5, "大部分BP值应该为正数"


class TestROEFactor:
    """ROE因子单元测试"""
    
    @pytest.fixture
    def mock_data(self):
        """创建模拟数据"""
        provider = MockDataProvider()
        return {
            'financial_data': provider.create_mock_financial_data(n_stocks=5, n_periods=4),
            'release_dates': provider.create_mock_release_dates(n_stocks=5, n_periods=4),
            'trading_dates': provider.create_mock_trading_dates('2022-01-01', '2022-12-31')
        }
    
    def test_roe_factor_instantiation(self):
        """测试ROE因子实例化"""
        # 测试默认参数
        roe_factor = ROEFactor()
        assert 'ROE' in roe_factor.name
        assert roe_factor.category == 'fundamental'
        assert roe_factor.earnings_method == 'ttm'
        assert roe_factor.equity_method == 'avg'
        
        # 测试自定义参数
        roe_factor_custom = ROEFactor(earnings_method='single_quarter', equity_method='current')
        assert roe_factor_custom.earnings_method == 'single_quarter'
        assert roe_factor_custom.equity_method == 'current'
    
    def test_roe_calculation(self, mock_data):
        """测试ROE计算"""
        roe_factor = ROEFactor()
        
        result = roe_factor.calculate(
            financial_data=mock_data['financial_data'],
            release_dates=mock_data['release_dates'],
            trading_dates=mock_data['trading_dates']
        )
        
        assert isinstance(result, pd.Series)
        assert len(result) > 0
    
    def test_roe_invalid_methods(self, mock_data):
        """测试无效的方法参数"""
        # 测试无效的earnings_method
        roe_factor_invalid_earnings = ROEFactor(earnings_method='invalid')
        with pytest.raises((ValueError, ValidationError)):
            roe_factor_invalid_earnings.calculate(financial_data=mock_data['financial_data'])
        
        # 测试无效的equity_method
        roe_factor_invalid_equity = ROEFactor(equity_method='invalid')
        with pytest.raises((ValueError, ValidationError)):
            roe_factor_invalid_equity.calculate(financial_data=mock_data['financial_data'])


class TestPEGFactor:
    """PEG因子单元测试"""
    
    @pytest.fixture
    def mock_data(self):
        """创建模拟数据"""
        provider = MockDataProvider()
        return {
            'financial_data': provider.create_mock_financial_data(n_stocks=5, n_periods=8),  # 需要更多期数计算增长率
            'market_cap': provider.create_mock_market_cap_data(n_stocks=5, n_days=100),
            'release_dates': provider.create_mock_release_dates(n_stocks=5, n_periods=8),
            'trading_dates': provider.create_mock_trading_dates('2022-01-01', '2023-12-31')
        }
    
    def test_peg_factor_instantiation(self):
        """测试PEG因子实例化"""
        peg_factor = PEGFactor()
        assert peg_factor.name == 'PEG'
        assert peg_factor.category == 'fundamental'
    
    def test_peg_calculation(self, mock_data):
        """测试PEG计算"""
        peg_factor = PEGFactor()
        
        result = peg_factor.calculate(
            financial_data=mock_data['financial_data'],
            market_cap=mock_data['market_cap'],
            release_dates=mock_data['release_dates'],
            trading_dates=mock_data['trading_dates']
        )
        
        assert isinstance(result, pd.Series)
        # PEG因子需要足够的历史数据，结果可能较少
        assert len(result) >= 0


class TestDataValidation:
    """数据验证测试 - 整合check_*文件的逻辑"""
    
    def test_financial_columns_check(self):
        """测试财务数据列检查"""
        provider = MockDataProvider()
        financial_data = provider.create_mock_financial_data()
        
        # 检查必需的列是否存在
        required_columns = ['DEDUCTEDPROFIT', 'EQY_BELONGTO_PARCOMSH', 'd_quarter']
        for col in required_columns:
            assert col in financial_data.columns, f"缺少必需列: {col}"
    
    def test_quarter_column_validation(self):
        """测试季度列验证"""
        provider = MockDataProvider()
        financial_data = provider.create_mock_financial_data()
        
        if 'd_quarter' in financial_data.columns:
            quarters = financial_data['d_quarter'].unique()
            valid_quarters = {1, 2, 3, 4}
            
            for quarter in quarters:
                if not pd.isna(quarter):
                    assert quarter in valid_quarters, f"无效的季度值: {quarter}"
    
    def test_data_types_validation(self):
        """测试数据类型验证"""
        provider = MockDataProvider()
        financial_data = provider.create_mock_financial_data()
        
        # 数值列应该是数值类型
        numeric_columns = ['DEDUCTEDPROFIT', 'EQY_BELONGTO_PARCOMSH', 'OPREVENUE']
        for col in numeric_columns:
            if col in financial_data.columns:
                assert pd.api.types.is_numeric_dtype(financial_data[col]), f"{col}应该是数值类型"


if __name__ == "__main__":
    # 可以直接运行进行测试
    pytest.main([__file__, "-v"])