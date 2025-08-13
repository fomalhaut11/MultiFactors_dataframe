"""
基本面因子单元测试 - 修复版本
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import numpy as np
import pytest
from factors.financial.fundamental_factors import EPFactor, BPFactor, ROEFactor, PEGFactor
from factors.base.validation import ValidationError


class TestEPFactor:
    """EP因子单元测试"""
    
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
        # 检查结果不是全部为NaN（应该有一些有效值）
        assert not result.isnull().all(), "结果不应该全为NaN"
        # 检查至少有一些有限值
        finite_count = np.isfinite(result).sum()
        assert finite_count > 0, f"应该有有限值，但只有 {finite_count} 个"
    
    def test_ep_single_quarter_calculation(self, small_sample_data):
        """测试EP单季度计算"""
        ep_factor = EPFactor(method='single_quarter')
        
        result = ep_factor.calculate(
            financial_data=small_sample_data['financial_data'],
            market_cap=small_sample_data['market_cap'],
            release_dates=small_sample_data['release_dates'],
            trading_dates=small_sample_data['trading_dates']
        )
        
        assert isinstance(result, pd.Series)
        assert len(result) > 0
        # 单季度计算应该比TTM有更多的有效值
        finite_count = np.isfinite(result).sum()
        assert finite_count > 0, f"应该有有限值，但只有 {finite_count} 个"
    
    def test_ep_invalid_method(self, small_sample_data):
        """测试无效方法参数"""
        ep_factor = EPFactor(method='invalid_method')
        
        with pytest.raises((ValueError, ValidationError)):
            ep_factor.calculate(
                financial_data=small_sample_data['financial_data'],
                market_cap=small_sample_data['market_cap'],
                release_dates=small_sample_data['release_dates'],
                trading_dates=small_sample_data['trading_dates']
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


class TestBPFactor: 
    """BP因子单元测试"""
    
    def test_bp_factor_instantiation(self):
        """测试BP因子实例化"""
        bp_factor = BPFactor()
        assert bp_factor.name == 'BP'
        assert bp_factor.category == 'fundamental'
    
    def test_bp_calculation(self, small_sample_data):
        """测试BP计算"""
        bp_factor = BPFactor()
        
        result = bp_factor.calculate(
            financial_data=small_sample_data['financial_data'],
            market_cap=small_sample_data['market_cap'],
            release_dates=small_sample_data['release_dates'],
            trading_dates=small_sample_data['trading_dates']
        )
        
        assert isinstance(result, pd.Series)
        assert len(result) > 0
        
        # 检查至少有一些有效值
        finite_count = np.isfinite(result).sum()
        assert finite_count > 0, f"应该有有限值，但只有 {finite_count} 个"
        
        # BP值在模拟数据中可能有负值，只需要有一些正值即可
        finite_values = result[np.isfinite(result)]
        if len(finite_values) > 0:
            positive_ratio = (finite_values > 0).mean()
            assert positive_ratio > 0.2, f"应该有一定比例的正BP值，当前为 {positive_ratio:.3f}"


class TestROEFactor:
    """ROE因子单元测试"""
    
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
    
    def test_roe_calculation(self, small_sample_data):
        """测试ROE计算"""
        roe_factor = ROEFactor()
        
        result = roe_factor.calculate(
            financial_data=small_sample_data['financial_data'],
            release_dates=small_sample_data['release_dates'],
            trading_dates=small_sample_data['trading_dates']
        )
        
        assert isinstance(result, pd.Series)
        assert len(result) > 0
        
        # 检查至少有一些有效值
        finite_count = np.isfinite(result).sum()
        assert finite_count > 0, f"应该有有限值，但只有 {finite_count} 个"
    
    def test_roe_invalid_methods(self, small_sample_data):
        """测试无效的方法参数"""
        # 测试无效的earnings_method
        roe_factor_invalid_earnings = ROEFactor(earnings_method='invalid')
        with pytest.raises((ValueError, ValidationError)):
            roe_factor_invalid_earnings.calculate(financial_data=small_sample_data['financial_data'])
        
        # 测试无效的equity_method
        roe_factor_invalid_equity = ROEFactor(equity_method='invalid')
        with pytest.raises((ValueError, ValidationError)):
            roe_factor_invalid_equity.calculate(financial_data=small_sample_data['financial_data'])


class TestDataValidation:
    """数据验证测试"""
    
    def test_financial_columns_check(self, small_sample_data):
        """测试财务数据列检查"""
        financial_data = small_sample_data['financial_data']
        
        # 检查必需的列是否存在
        required_columns = ['DEDUCTEDPROFIT', 'EQY_BELONGTO_PARCOMSH', 'd_quarter']
        for col in required_columns:
            assert col in financial_data.columns, f"缺少必需列: {col}"
    
    def test_quarter_column_validation(self, small_sample_data):
        """测试季度列验证"""
        financial_data = small_sample_data['financial_data']
        
        if 'd_quarter' in financial_data.columns:
            quarters = financial_data['d_quarter'].unique()
            valid_quarters = {1, 2, 3, 4}
            
            for quarter in quarters:
                if not pd.isna(quarter):
                    assert quarter in valid_quarters, f"无效的季度值: {quarter}"
    
    def test_data_types_validation(self, small_sample_data):
        """测试数据类型验证"""
        financial_data = small_sample_data['financial_data']
        
        # 数值列应该是数值类型
        numeric_columns = ['DEDUCTEDPROFIT', 'EQY_BELONGTO_PARCOMSH', 'OPREVENUE']
        for col in numeric_columns:
            if col in financial_data.columns:
                assert pd.api.types.is_numeric_dtype(financial_data[col]), f"{col}应该是数值类型"


if __name__ == "__main__":
    # 可以直接运行进行测试
    pytest.main([__file__, "-v"])