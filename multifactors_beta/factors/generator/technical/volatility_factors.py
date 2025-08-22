#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Volatility factors calculation module

Calculate various volatility related technical factors
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
import logging

from ...base.factor_base import FactorBase

logger = logging.getLogger(__name__)


class VolatilityFactor(FactorBase):
    """
    Volatility factor base class
    """
    
    def __init__(self, **kwargs):
        kwargs.setdefault('name', 'VolatilityFactor')
        kwargs.setdefault('category', 'technical')
        super().__init__(**kwargs)
        
    def calculate(self, data: pd.DataFrame, window: int = 20, **kwargs) -> pd.DataFrame:
        """
        Calculate historical volatility
        
        Parameters
        ----------
        data : pd.DataFrame
            Price data
        window : int
            Rolling window
            
        Returns
        -------
        pd.DataFrame
            Volatility factor values
        """
        # Calculate returns
        returns = data.pct_change()
        
        # Calculate rolling standard deviation
        volatility = returns.rolling(window=window).std()
        
        # Annualize
        volatility = volatility * np.sqrt(252)
        
        return volatility


# TODO: Implement more volatility factors
# - GARCH volatility
# - Realized volatility
# - Jump volatility
# - Downside volatility