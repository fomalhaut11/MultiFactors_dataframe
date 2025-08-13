"""
因子计算工具模块
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
import logging
from pathlib import Path

from ..base.factor_base import MultiFactorBase
from ..generator.financial.pure_financial_factors import PureFinancialFactorCalculator
from ..generator.technical import (
    price_factors,
    volatility_factors
)
from ..generator.risk import beta_factors

from core.config_manager import config

logger = logging.getLogger(__name__)


class FactorCalculator:
    """因子计算器，统一管理所有因子的计算"""
    
    def __init__(self):
        self.factors = {}
        self._register_default_factors()
        
    def _register_default_factors(self):
        """注册默认因子"""
        # 创建纯财务因子计算器
        financial_calculator = PureFinancialFactorCalculator()
        
        # 注册主要的纯财务因子
        key_financial_factors = [
            'ROE_ttm', 'ROA_ttm', 'GrossProfitMargin_ttm', 'NetProfitMargin_ttm',
            'CurrentRatio', 'QuickRatio', 'DebtToAssets', 'AssetTurnover_ttm'
        ]
        
        for factor_name in key_financial_factors:
            if factor_name in financial_calculator.factor_methods:
                # 创建包装器来适配原有接口
                method = financial_calculator.factor_methods[factor_name]
                
                # 创建包装器类
                class FinancialFactorWrapper:
                    def __init__(self, method, name):
                        self.method = method
                        self.name = name
                        self.category = 'financial'
                        
                    def calculate(self, data, **kwargs):
                        return self.method(data, **kwargs)
                
                self.factors[factor_name] = FinancialFactorWrapper(method, factor_name)
                
        # 注册估值因子（需要市值数据的因子暂时保留原有逻辑）
        # 这些因子在pure_financial_factors中需要市值数据，暂不迁移
        
        # 技术因子
        self.register_factor('Momentum_20', price_factors.MomentumFactor(window=20))
        self.register_factor('Reversal_5', price_factors.ReversalFactor(window=5))
        self.register_factor('MA_5_20', price_factors.MovingAverageFactor(5, 20))
        self.register_factor('RSI_14', price_factors.RSIFactor(window=14))
        self.register_factor('GapReturn', price_factors.GapReturnFactor())
        
        # 波动率因子
        self.register_factor('Volatility_20', volatility_factors.HistoricalVolatilityFactor(window=20))
        self.register_factor('IntradayVolatility_20', volatility_factors.IntradayVolatilityFactor(window=20))
        self.register_factor('DownsideVolatility_20', volatility_factors.DownsideVolatilityFactor(window=20))
        
        # 风险因子
        self.register_factor('Beta_252', beta_factors.BetaFactor(window=252))
        self.register_factor('WeightedBeta_252_63', beta_factors.WeightedBetaFactor(window=252, half_life=63))
        
    def register_factor(self, name: str, factor):
        """注册因子"""
        self.factors[name] = factor
        logger.info(f"Registered factor: {name}")
        
    def calculate_factors(self,
                        factor_names: List[str],
                        financial_data: Optional[pd.DataFrame] = None,
                        price_data: Optional[pd.DataFrame] = None,
                        market_cap: Optional[pd.Series] = None,
                        benchmark_data: Optional[pd.DataFrame] = None,
                        release_dates: Optional[pd.DataFrame] = None,
                        trading_dates: Optional[pd.DatetimeIndex] = None,
                        save_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
        """
        计算指定的因子
        
        Parameters:
        -----------
        factor_names : 要计算的因子名称列表
        financial_data : 财务数据
        price_data : 价格数据
        market_cap : 市值数据
        benchmark_data : 基准指数数据
        release_dates : 财报发布日期
        trading_dates : 交易日序列
        save_path : 保存路径
        
        Returns:
        --------
        因子DataFrame
        """
        results = {}
        
        # 构建参数字典
        kwargs = {}
        if release_dates is not None:
            kwargs['release_dates'] = release_dates
        if trading_dates is not None:
            kwargs['trading_dates'] = trading_dates
            
        for factor_name in factor_names:
            if factor_name not in self.factors:
                logger.warning(f"Factor {factor_name} not registered, skipping...")
                continue
                
            factor = self.factors[factor_name]
            logger.info(f"Calculating factor: {factor_name}")
            
            try:
                # 根据因子类型决定需要的数据
                if factor.category == 'fundamental' or factor.category == 'profitability' or factor.category == 'liquidity':
                    if financial_data is None:
                        logger.error(f"Financial data required for {factor_name}")
                        continue
                        
                    # 部分因子需要市值数据
                    if factor_name in ['EP_ttm', 'BP', 'PEG', 'FreeCashFlow_ttm'] and market_cap is not None:
                        factor_value = factor.calculate(financial_data, market_cap, **kwargs)
                    else:
                        factor_value = factor.calculate(financial_data, **kwargs)
                        
                elif factor.category == 'technical':
                    if price_data is None:
                        logger.error(f"Price data required for {factor_name}")
                        continue
                    factor_value = factor.calculate(price_data, **kwargs)
                    
                elif factor.category == 'risk':
                    if price_data is None or benchmark_data is None:
                        logger.error(f"Price and benchmark data required for {factor_name}")
                        continue
                    factor_value = factor.calculate(price_data, benchmark_data, **kwargs)
                    
                else:
                    logger.warning(f"Unknown factor category: {factor.category}")
                    continue
                    
                results[factor_name] = factor_value
                
                # 保存单个因子
                if save_path:
                    save_file = Path(save_path) / f"{factor_name}.pkl"
                    factor.save(factor_value, save_file)
                    
            except Exception as e:
                logger.error(f"Error calculating {factor_name}: {e}")
                continue
                
        # 合并所有因子
        if results:
            all_factors = pd.DataFrame(results)
            logger.info(f"Successfully calculated {len(results)} factors")
            return all_factors
        else:
            logger.warning("No factors calculated successfully")
            return pd.DataFrame()
            
    def list_factors(self) -> Dict[str, str]:
        """列出所有可用因子"""
        factor_info = {}
        for name, factor in self.factors.items():
            factor_info[name] = {
                'category': factor.category,
                'description': factor.description
            }
        return factor_info


class FactorDataLoader:
    """因子数据加载器"""
    
    @staticmethod
    def load_factors(factor_names: List[str],
                    data_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
        """
        加载因子数据
        
        Parameters:
        -----------
        factor_names : 因子名称列表
        data_path : 数据路径
        
        Returns:
        --------
        因子DataFrame
        """
        if data_path is None:
            data_path = config.get_path('factors')
        else:
            data_path = Path(data_path)
            
        factors = {}
        for factor_name in factor_names:
            factor_file = data_path / f"{factor_name}.pkl"
            if factor_file.exists():
                try:
                    factors[factor_name] = pd.read_pickle(factor_file)
                    logger.info(f"Loaded factor: {factor_name}")
                except Exception as e:
                    logger.error(f"Error loading {factor_name}: {e}")
            else:
                logger.warning(f"Factor file not found: {factor_file}")
                
        if factors:
            return pd.DataFrame(factors)
        else:
            return pd.DataFrame()