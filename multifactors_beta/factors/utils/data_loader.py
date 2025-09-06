#!/usr/bin/env python3
"""
统一的因子数据加载器

解决因子模块中重复计算收益率等基础数据的问题
遵循单一职责原则：data模块负责数据预处理，factors模块专注于因子计算
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Union, Literal, List, Dict, Any
import logging
import os

from config import get_config

logger = logging.getLogger(__name__)


class FactorDataLoader:
    """统一的因子数据加载器
    
    负责加载预处理好的基础数据，避免在因子计算中重复计算收益率等
    """
    
    _cache = {}  # 类级别缓存
    
    @classmethod
    def _get_cache_key(cls, data_type: str, **kwargs) -> str:
        """生成缓存键"""
        key_parts = [data_type]
        for k, v in sorted(kwargs.items()):
            key_parts.append(f"{k}_{v}")
        return "_".join(key_parts)
    
    @classmethod
    def _get_data_path(cls, filename: str) -> Path:
        """获取数据文件路径"""
        try:
            # 首先尝试从配置获取路径
            data_root = get_config('main.paths.data_root')
            
            # auxiliary目录的文件
            auxiliary_files = [
                'LogReturn_daily_o2o.pkl', 'LogReturn_5days_o2o.pkl', 
                'LogReturn_20days_o2o.pkl', 'LogReturn_monthly_o2o.pkl',
                'LogReturn_weekly_o2o.pkl', 'LogReturn_daily_vwap.pkl',
                'MarketCap.pkl', 'ReleaseDates.pkl', 'StockInfo.pkl',
                'TradingDates.pkl', 'FinancialData_unified.pkl'
            ]
            
            if filename in auxiliary_files or filename.startswith('LogReturn'):
                return Path(data_root) / 'auxiliary' / filename
            else:
                return Path(data_root) / filename
        except:
            # 如果配置失败，使用相对路径
            project_root = Path(__file__).parent.parent.parent.parent
            data_root = project_root / 'StockData'
            
            auxiliary_files = [
                'LogReturn_daily_o2o.pkl', 'LogReturn_5days_o2o.pkl', 
                'LogReturn_20days_o2o.pkl', 'LogReturn_monthly_o2o.pkl',
                'LogReturn_weekly_o2o.pkl', 'LogReturn_daily_vwap.pkl',
                'MarketCap.pkl', 'ReleaseDates.pkl', 'StockInfo.pkl',
                'TradingDates.pkl', 'FinancialData_unified.pkl'
            ]
            
            if filename in auxiliary_files or filename.startswith('LogReturn'):
                return data_root / 'auxiliary' / filename
            else:
                return data_root / filename
    
    @classmethod
    def load_returns(cls, 
                    period: Literal['daily', 'weekly', 'monthly', '5days', '20days'] = 'daily',
                    return_type: Literal['o2o', 'vwap'] = 'o2o') -> pd.Series:
        """
        加载预处理的收益率数据
        
        Parameters
        ----------
        period : str
            收益率周期：'daily', 'weekly', 'monthly', '5days', '20days'
        return_type : str
            收益率类型：'o2o' (open to open), 'vwap' (volume weighted average price)
            
        Returns
        -------
        pd.Series
            收益率数据，MultiIndex格式[TradingDates, StockCodes]
        """
        cache_key = cls._get_cache_key('returns', period=period, return_type=return_type)
        
        if cache_key in cls._cache:
            logger.debug(f"从缓存加载收益率数据: {cache_key}")
            return cls._cache[cache_key]
        
        try:
            filename = f"LogReturn_{period}_{return_type}.pkl"
            filepath = cls._get_data_path(filename)
            
            if not filepath.exists():
                raise FileNotFoundError(f"收益率数据文件不存在: {filepath}")
            
            logger.info(f"加载收益率数据: {filename}")
            returns_data = pd.read_pickle(filepath)
            
            # 处理数据格式：如果是DataFrame，转换为Series
            if isinstance(returns_data, pd.DataFrame):
                if returns_data.shape[1] == 1:
                    # 单列DataFrame，转换为Series
                    returns_data = returns_data.iloc[:, 0]
                elif 'LogReturn' in returns_data.columns:
                    # 包含LogReturn列，提取该列
                    returns_data = returns_data['LogReturn']
                else:
                    raise ValueError(f"DataFrame格式的收益率数据应该只有一列或包含'LogReturn'列，实际列：{list(returns_data.columns)}")
            elif not isinstance(returns_data, pd.Series):
                raise ValueError(f"收益率数据格式错误，期望Series或单列DataFrame，实际{type(returns_data)}")
            
            if not isinstance(returns_data.index, pd.MultiIndex):
                raise ValueError("收益率数据必须是MultiIndex格式")
            
            # 排序并缓存
            returns_data = returns_data.sort_index()
            cls._cache[cache_key] = returns_data
            
            logger.info(f"收益率数据加载成功: {returns_data.shape}")
            return returns_data
            
        except Exception as e:
            logger.error(f"加载收益率数据失败 {filename}: {e}")
            raise
    
    @classmethod
    def load_price_data(cls) -> pd.DataFrame:
        """
        加载价格数据（包含adjfactor等）
        
        Returns
        -------
        pd.DataFrame
            价格数据，包含open, high, low, close, volume, adjfactor等列
        """
        cache_key = 'price_data'
        
        if cache_key in cls._cache:
            logger.debug("从缓存加载价格数据")
            return cls._cache[cache_key]
        
        try:
            filepath = cls._get_data_path("Price.pkl")
            
            if not filepath.exists():
                raise FileNotFoundError(f"价格数据文件不存在: {filepath}")
            
            logger.info("加载价格数据: Price.pkl")
            price_data = pd.read_pickle(filepath)
            
            # 验证数据格式
            if not isinstance(price_data, pd.DataFrame):
                raise ValueError(f"价格数据格式错误，期望DataFrame，实际{type(price_data)}")
            
            if not isinstance(price_data.index, pd.MultiIndex):
                raise ValueError("价格数据必须是MultiIndex格式")
            
            # 验证必要的列（使用实际的列名）
            # 实际列名: 'c'=close, 'o'=open, 'h'=high, 'l'=low, 'v'=volume, 'amt'=amount
            required_columns = ['c', 'o', 'h', 'l', 'v']  
            missing_columns = [col for col in required_columns if col not in price_data.columns]
            if missing_columns:
                logger.warning(f"价格数据缺少列: {missing_columns}")
            else:
                logger.info("价格数据列验证通过")
            
            # 排序并缓存
            price_data = price_data.sort_index()
            cls._cache[cache_key] = price_data
            
            logger.info(f"价格数据加载成功: {price_data.shape}")
            return price_data
            
        except Exception as e:
            logger.error(f"加载价格数据失败: {e}")
            raise
    
    @classmethod
    def load_market_cap(cls) -> pd.Series:
        """
        加载市值数据
        
        Returns
        -------
        pd.Series
            市值数据，MultiIndex格式[TradingDates, StockCodes]
        """
        cache_key = 'market_cap'
        
        if cache_key in cls._cache:
            logger.debug("从缓存加载市值数据")
            return cls._cache[cache_key]
        
        try:
            filepath = cls._get_data_path("MarketCap.pkl")
            
            if not filepath.exists():
                raise FileNotFoundError(f"市值数据文件不存在: {filepath}")
            
            logger.info("加载市值数据: MarketCap.pkl")
            market_cap = pd.read_pickle(filepath)
            
            # 验证数据格式
            if not isinstance(market_cap, pd.Series):
                raise ValueError(f"市值数据格式错误，期望Series，实际{type(market_cap)}")
            
            if not isinstance(market_cap.index, pd.MultiIndex):
                raise ValueError("市值数据必须是MultiIndex格式")
            
            # 排序并缓存
            market_cap = market_cap.sort_index()
            cls._cache[cache_key] = market_cap
            
            logger.info(f"市值数据加载成功: {market_cap.shape}")
            return market_cap
            
        except Exception as e:
            logger.error(f"加载市值数据失败: {e}")
            raise
    
    @classmethod
    def load_benchmark_returns(cls, 
                              benchmark: str = 'zz500',
                              period: str = 'daily') -> pd.Series:
        """
        加载基准指数收益率数据
        
        Parameters
        ----------
        benchmark : str
            基准指数名称，如'zz500', 'zz2000'
        period : str
            收益率周期
            
        Returns
        -------
        pd.Series
            基准收益率数据
        """
        # 这里先返回None，等待具体的基准数据文件格式确认
        logger.warning(f"基准收益率数据加载功能待实现: {benchmark}")
        return None
    
    @classmethod
    def get_trading_dates(cls) -> pd.DatetimeIndex:
        """
        获取交易日期序列
        
        Returns
        -------
        pd.DatetimeIndex
            交易日期序列
        """
        cache_key = 'trading_dates'
        
        if cache_key in cls._cache:
            return cls._cache[cache_key]
        
        try:
            # 从配置的auxiliary目录中读取
            data_root = get_config('main.paths.data_root')
            filepath = Path(data_root) / 'auxiliary' / 'TradingDates.pkl'
            
            if filepath.exists():
                trading_dates = pd.read_pickle(filepath)
                cls._cache[cache_key] = trading_dates
                return trading_dates
            else:
                # 从价格数据中提取
                price_data = cls.load_price_data()
                trading_dates = price_data.index.get_level_values(0).unique().sort_values()
                cls._cache[cache_key] = trading_dates
                return trading_dates
                
        except Exception as e:
            logger.error(f"获取交易日期失败: {e}")
            raise
    
    @classmethod
    def calculate_period_returns(cls, periods: int, method: Literal['simple', 'log'] = 'simple') -> pd.Series:
        """
        基于日对数收益率计算任意周期收益率
        
        Parameters
        ----------
        periods : int
            周期天数（如120天）
        method : str
            'simple' - 简单收益率
            'log' - 对数收益率
            
        Returns
        -------
        pd.Series
            周期收益率，MultiIndex格式[TradingDates, StockCodes]
        """
        cache_key = cls._get_cache_key('period_returns', periods=periods, method=method)
        
        if cache_key in cls._cache:
            logger.debug(f"从缓存加载{periods}日{method}收益率")
            return cls._cache[cache_key]
        
        try:
            # 先尝试加载预生成的对应周期收益率
            if periods == 5:
                base_returns = cls.load_returns(period='5days', return_type='o2o')
            elif periods == 20:
                base_returns = cls.load_returns(period='20days', return_type='o2o')
            else:
                # 使用日收益率累积计算
                logger.info(f"使用日对数收益率累积计算{periods}日收益率")
                daily_log_returns = cls.load_returns(period='daily', return_type='o2o')
                
                # 按股票分组进行滚动累积
                cumulative_log_returns = daily_log_returns.groupby(level='StockCodes').rolling(
                    window=periods,
                    min_periods=int(periods * 0.8)  # 允许20%缺失
                ).sum().droplevel(0)
                
                base_returns = cumulative_log_returns
            
            # 根据method转换
            if method == 'simple':
                # 对数收益率转简单收益率: exp(log_return) - 1
                period_returns = np.exp(base_returns) - 1
            else:  # method == 'log'
                period_returns = base_returns
            
            # 缓存结果
            cls._cache[cache_key] = period_returns
            
            logger.info(f"{periods}日{method}收益率计算完成: {period_returns.shape}")
            return period_returns
            
        except Exception as e:
            logger.error(f"计算{periods}日收益率失败: {e}")
            raise
    
    @classmethod
    def clear_cache(cls):
        """清空缓存"""
        cls._cache.clear()
        logger.info("数据加载器缓存已清空")
    
    @classmethod
    def get_cache_info(cls) -> dict:
        """获取缓存信息"""
        return {
            'cached_datasets': list(cls._cache.keys()),
            'cache_size': len(cls._cache)
        }
    
    @classmethod
    def load_stock_universe(cls, 
                          universe_name: str,
                          format_type: str = 'multiindex',
                          refresh: bool = False,
                          **kwargs) -> pd.Series:
        """
        加载股票池数据
        
        Parameters
        ----------
        universe_name : str
            股票池名称，支持：
            - 'all': 全市场股票
            - 'liquid_300': 沪深300流动性股票池
            - 'liquid_500': 中证500流动性股票池  
            - 'liquid_1000': 中证1000流动性股票池
            - 或自定义股票池文件名
        format_type : str
            返回格式类型，'multiindex' 或 'list'
        refresh : bool
            是否强制刷新缓存
        **kwargs : dict
            额外参数传递给股票池管理器
            
        Returns
        -------
        pd.Series
            股票池数据，MultiIndex格式[TradingDates, StockCodes]，值为1
        """
        cache_key = cls._get_cache_key('stock_universe', 
                                     universe=universe_name, 
                                     format=format_type,
                                     **{k: v for k, v in kwargs.items() if isinstance(v, (str, int, float, bool))})
        
        if not refresh and cache_key in cls._cache:
            logger.debug(f"从缓存加载股票池数据: {universe_name}")
            return cls._cache[cache_key]
        
        try:
            # 导入股票池管理器
            from ..tester.stock_universe_manager import StockUniverseManager
            
            # 初始化股票池管理器
            universe_manager = StockUniverseManager()
            
            # 获取股票池数据
            logger.info(f"加载股票池数据: {universe_name}")
            universe_data = universe_manager.get_universe(
                universe_name=universe_name,
                format_type=format_type,
                refresh=refresh,
                **kwargs
            )
            
            # 验证数据格式
            if format_type == 'multiindex':
                if not isinstance(universe_data, pd.Series):
                    raise ValueError(f"股票池数据格式错误，期望Series，实际{type(universe_data)}")
                
                if not isinstance(universe_data.index, pd.MultiIndex):
                    raise ValueError("股票池数据必须是MultiIndex格式")
                
                # 排序并缓存
                universe_data = universe_data.sort_index()
                cls._cache[cache_key] = universe_data
                
                logger.info(f"股票池数据加载成功: {universe_data.shape}")
                return universe_data
            else:
                # 对于list格式，也缓存但不转换
                cls._cache[cache_key] = universe_data
                return universe_data
            
        except Exception as e:
            logger.error(f"加载股票池数据失败 {universe_name}: {e}")
            raise
    
    @classmethod
    def list_available_universes(cls) -> Dict[str, str]:
        """
        列出可用的股票池
        
        Returns
        -------
        Dict[str, str]
            {股票池名称: 描述}的字典
        """
        try:
            from ..tester.stock_universe_manager import StockUniverseManager
            universe_manager = StockUniverseManager()
            return universe_manager.list_available_universes()
        except Exception as e:
            logger.error(f"获取可用股票池列表失败: {e}")
            return {}


# 便捷函数
def get_daily_returns(return_type: str = 'o2o') -> pd.Series:
    """获取日收益率数据的便捷函数"""
    return FactorDataLoader.load_returns('daily', return_type)

def get_price_data() -> pd.DataFrame:
    """获取价格数据的便捷函数"""
    return FactorDataLoader.load_price_data()

def get_market_cap() -> pd.Series:
    """获取市值数据的便捷函数"""
    return FactorDataLoader.load_market_cap()

def get_stock_universe(universe_name: str, **kwargs) -> pd.Series:
    """获取股票池数据的便捷函数"""
    return FactorDataLoader.load_stock_universe(universe_name, **kwargs)