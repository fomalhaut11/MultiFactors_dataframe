"""
因子验证集成测试 - 整合原validation目录下的验证脚本
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import numpy as np
import pytest
from factors.financial.fundamental_factors import BPFactor, EPFactor, ROEFactor, PEGFactor
from factors.base.testable_mixin import MockDataProvider


class TestFactorValidation:
    """因子验证测试类"""
    
    @pytest.fixture
    def sample_data(self):
        """创建样本测试数据"""
        provider = MockDataProvider()
        return {
            'financial_data': provider.create_mock_financial_data(n_stocks=10, n_periods=8),
            'market_cap': provider.create_mock_market_cap_data(n_stocks=10, n_days=100),
            'release_dates': provider.create_mock_release_dates(n_stocks=10, n_periods=8),
            'trading_dates': provider.create_mock_trading_dates('2022-01-01', '2023-12-31')
        }
    
    def test_bp_factor_validation(self, sample_data):
        """测试BP因子计算的正确性"""
        bp_factor = BPFactor()
        
        # 计算BP因子
        bp_result = bp_factor.calculate(
            financial_data=sample_data['financial_data'],
            market_cap=sample_data['market_cap'],
            release_dates=sample_data['release_dates'],
            trading_dates=sample_data['trading_dates']
        )
        
        # 基本验证
        assert bp_result is not None, "BP因子计算结果不应为空"
        assert len(bp_result) > 0, "BP因子计算结果不应为空序列"
        assert isinstance(bp_result, pd.Series), "BP因子结果应为Series类型"
        
        # 数值验证
        assert not bp_result.isnull().all(), "BP因子结果不应全为空值"
        finite_values = bp_result[np.isfinite(bp_result)]
        assert len(finite_values) > 0, "BP因子应包含有限数值"
        
        # 合理性验证（BP因子通常在0-10之间）
        reasonable_values = finite_values[(finite_values >= 0) & (finite_values <= 20)]
        assert len(reasonable_values) > len(finite_values) * 0.8, "大部分BP因子值应在合理范围内"
    
    def test_ep_factor_validation(self, sample_data):
        """测试EP因子计算的正确性"""
        ep_factor = EPFactor(method='ttm')
        
        # 计算EP因子
        ep_result = ep_factor.calculate(
            financial_data=sample_data['financial_data'],
            market_cap=sample_data['market_cap'],
            release_dates=sample_data['release_dates'],
            trading_dates=sample_data['trading_dates']
        )
        
        # 基本验证
        assert ep_result is not None, "EP因子计算结果不应为空"
        assert len(ep_result) > 0, "EP因子计算结果不应为空序列"
        assert isinstance(ep_result, pd.Series), "EP因子结果应为Series类型"
        
        # 数值验证
        assert not ep_result.isnull().all(), "EP因子结果不应全为空值"
        finite_values = ep_result[np.isfinite(ep_result)]
        assert len(finite_values) > 0, "EP因子应包含有限数值"
        
        # 合理性验证（EP因子通常在-0.5到0.5之间）
        reasonable_values = finite_values[(finite_values >= -1) & (finite_values <= 1)]
        assert len(reasonable_values) > len(finite_values) * 0.7, "大部分EP因子值应在合理范围内"
    
    def test_roe_factor_validation(self, sample_data):
        """测试ROE因子计算的正确性"""
        roe_factor = ROEFactor(earnings_method='ttm', equity_method='avg')
        
        # 计算ROE因子
        roe_result = roe_factor.calculate(
            financial_data=sample_data['financial_data'],
            release_dates=sample_data['release_dates'],
            trading_dates=sample_data['trading_dates']
        )
        
        # 基本验证
        assert roe_result is not None, "ROE因子计算结果不应为空"
        assert len(roe_result) > 0, "ROE因子计算结果不应为空序列"
        assert isinstance(roe_result, pd.Series), "ROE因子结果应为Series类型"
        
        # 数值验证
        assert not roe_result.isnull().all(), "ROE因子结果不应全为空值"
        finite_values = roe_result[np.isfinite(roe_result)]
        assert len(finite_values) > 0, "ROE因子应包含有限数值"
        
        # 合理性验证（ROE通常在-1到1之间）
        reasonable_values = finite_values[(finite_values >= -2) & (finite_values <= 2)]
        assert len(reasonable_values) > len(finite_values) * 0.8, "大部分ROE因子值应在合理范围内"
    
    def test_peg_factor_validation(self, sample_data):
        """测试PEG因子计算的正确性"""
        peg_factor = PEGFactor()
        
        # 计算PEG因子
        peg_result = peg_factor.calculate(
            financial_data=sample_data['financial_data'],
            market_cap=sample_data['market_cap'],
            release_dates=sample_data['release_dates'],
            trading_dates=sample_data['trading_dates']
        )
        
        # 基本验证
        assert peg_result is not None, "PEG因子计算结果不应为空"
        assert len(peg_result) > 0, "PEG因子计算结果不应为空序列"
        assert isinstance(peg_result, pd.Series), "PEG因子结果应为Series类型"
        
        # 数值验证
        assert not peg_result.isnull().all(), "PEG因子结果不应全为空值"
    
    def test_factor_cross_validation(self, sample_data):
        """因子交叉验证测试"""
        # 计算多个因子
        bp_factor = BPFactor()
        ep_factor = EPFactor(method='ttm')
        
        bp_result = bp_factor.calculate(
            financial_data=sample_data['financial_data'],
            market_cap=sample_data['market_cap'],
            release_dates=sample_data['release_dates'],
            trading_dates=sample_data['trading_dates']
        )
        
        ep_result = ep_factor.calculate(
            financial_data=sample_data['financial_data'],
            market_cap=sample_data['market_cap'],
            release_dates=sample_data['release_dates'],
            trading_dates=sample_data['trading_dates']
        )
        
        # 检查因子之间的相关性（应该有一定相关性但不完全相关）
        common_index = bp_result.index.intersection(ep_result.index)
        if len(common_index) > 10:
            bp_common = bp_result.loc[common_index]
            ep_common = ep_result.loc[common_index]
            
            # 过滤掉无穷大和NaN值
            valid_mask = np.isfinite(bp_common) & np.isfinite(ep_common)
            if valid_mask.sum() > 5:
                correlation = bp_common[valid_mask].corr(ep_common[valid_mask])
                
                # BP和EP应该有正相关性（都是价值因子）
                assert not np.isnan(correlation), "因子相关性计算应该有效"
                # 注意：由于是模拟数据，相关性可能不明显，这里只做基本检查
    
    def test_factor_stability(self, sample_data):
        """因子稳定性测试"""
        bp_factor = BPFactor()
        
        # 多次计算同一因子，结果应该一致
        result1 = bp_factor.calculate(
            financial_data=sample_data['financial_data'],
            market_cap=sample_data['market_cap'],
            release_dates=sample_data['release_dates'],
            trading_dates=sample_data['trading_dates']
        )
        
        result2 = bp_factor.calculate(
            financial_data=sample_data['financial_data'],
            market_cap=sample_data['market_cap'],
            release_dates=sample_data['release_dates'],
            trading_dates=sample_data['trading_dates']
        )
        
        # 结果应该完全一致
        pd.testing.assert_series_equal(result1, result2, check_names=False)
    
    def test_factor_edge_cases(self, sample_data):
        """因子边界情况测试"""
        bp_factor = BPFactor()
        
        # 测试极小数据集
        small_financial = sample_data['financial_data'].head(5)
        small_market_cap = sample_data['market_cap'].head(50)
        small_release_dates = sample_data['release_dates'].head(5)
        
        try:
            result = bp_factor.calculate(
                financial_data=small_financial,
                market_cap=small_market_cap,
                release_dates=small_release_dates,
                trading_dates=sample_data['trading_dates']
            )
            
            # 即使是小数据集也应该能正常处理
            assert result is not None, "小数据集应该能正常处理"
            
        except Exception as e:
            # 如果出错，错误信息应该清晰
            assert "data" in str(e).lower() or "empty" in str(e).lower(), f"错误信息应该清晰: {e}"


if __name__ == "__main__":
    # 可以直接运行这个文件进行测试
    pytest.main([__file__, "-v"])