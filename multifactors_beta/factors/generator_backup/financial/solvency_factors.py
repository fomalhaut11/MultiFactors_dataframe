#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
偿债能力因子实现
包含流动比率、资产负债率等偿债能力相关因子
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
import logging

from ...base.factor_base import FactorBase
from ...base.data_processing_mixin import DataProcessingMixin
from ...base.flexible_data_adapter import ColumnMapperMixin

logger = logging.getLogger(__name__)


class CurrentRatio_Factor(FactorBase, DataProcessingMixin, ColumnMapperMixin):
    """流动比率因子"""
    
    def __init__(self):
        super().__init__(name='CurrentRatio', category='solvency')
        self.description = "流动比率 - 流动资产/流动负债"
        
    def calculate(self, financial_data: pd.DataFrame, **kwargs) -> pd.Series:
        """计算流动比率"""
        if not self.validate_data_requirements(financial_data, ['current_assets', 'current_liabilities']):
            raise ValueError("Required data not available for CurrentRatio calculation")
            
        extracted_data = self.extract_required_data(
            financial_data, required_columns=['current_assets', 'current_liabilities']
        )
        
        current_assets = extracted_data['current_assets']
        current_liabilities = extracted_data['current_liabilities']
        
        current_ratio = self._safe_division(current_assets, current_liabilities)
        
        # 扩展到日频（如果需要）
        if 'release_dates' in kwargs and 'trading_dates' in kwargs:
            current_ratio = self._expand_to_daily_if_needed(current_ratio, 'CurrentRatio', **kwargs)
            
        return current_ratio


class DebtToAssets_Factor(FactorBase, DataProcessingMixin, ColumnMapperMixin):
    """资产负债率因子"""
    
    def __init__(self):
        super().__init__(name='DebtToAssets', category='solvency')
        self.description = "资产负债率 - 负债总额/资产总额"
        
    def calculate(self, financial_data: pd.DataFrame, **kwargs) -> pd.Series:
        """计算资产负债率"""
        if not self.validate_data_requirements(financial_data, ['total_liabilities', 'total_assets']):
            raise ValueError("Required data not available for DebtToAssets calculation")
            
        extracted_data = self.extract_required_data(
            financial_data, required_columns=['total_liabilities', 'total_assets']
        )
        
        total_liabilities = extracted_data['total_liabilities']
        total_assets = extracted_data['total_assets']
        
        debt_to_assets = self._safe_division(total_liabilities, total_assets)
        
        # 扩展到日频（如果需要）
        if 'release_dates' in kwargs and 'trading_dates' in kwargs:
            debt_to_assets = self._expand_to_daily_if_needed(debt_to_assets, 'DebtToAssets', **kwargs)
            
        return debt_to_assets