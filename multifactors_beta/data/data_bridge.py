#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据桥接模块

提供data模块和factors模块之间的标准数据传递接口
确保数据格式的一致性和可靠性

Author: MultiFactors Team
Date: 2025-08-21
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime
import logging

from .schemas import (
    DataValidator, DataConverter, DataQualityChecker,
    DataSchemas, validate_price_data, validate_financial_data, 
    validate_factor_format, convert_to_factor_format
)
from .fetcher.data_fetcher import StockDataFetcher
from config import get_config

logger = logging.getLogger(__name__)


class DataBridge:
    """数据桥接器 - data模块和factors模块的标准接口"""
    
    def __init__(self, data_root: Optional[str] = None):
        """
        初始化数据桥接器
        
        Parameters
        ----------
        data_root : str, optional
            数据根目录路径
        """
        self.data_root = Path(data_root or get_config('main.paths.data_root'))
        self.auxiliary_path = self.data_root / 'auxiliary'
        self.fetcher = StockDataFetcher()
        
        # 数据缓存
        self._cache = {}
        self._cache_timestamps = {}
        
        logger.info(f"DataBridge初始化完成，数据路径: {self.data_root}")
    
    def get_financial_data(self, 
                          validate: bool = True,
                          use_cache: bool = True) -> pd.DataFrame:
        """
        获取标准格式的财务数据
        
        Parameters
        ----------
        validate : bool
            是否验证数据格式
        use_cache : bool
            是否使用缓存
            
        Returns
        -------
        pd.DataFrame
            验证过的财务数据
        """
        cache_key = 'financial_data'
        
        # 尝试从缓存获取
        if use_cache and cache_key in self._cache:
            logger.info("从缓存获取财务数据")
            return self._cache[cache_key]
        
        # 从文件读取
        file_path = self.auxiliary_path / 'FinancialData_unified.pkl'
        if not file_path.exists():
            raise FileNotFoundError(f"财务数据文件不存在: {file_path}")
        
        logger.info(f"从文件读取财务数据: {file_path}")
        data = pd.read_pickle(file_path)
        
        # 数据验证
        if validate:
            is_valid, errors = validate_financial_data(data, strict=False)
            if not is_valid:
                logger.warning(f"财务数据格式验证失败: {errors}")
                # 不抛出异常，只记录警告，因为财务数据字段较多且动态
            else:
                logger.info("财务数据格式验证通过")
        
        # 缓存数据
        if use_cache:
            self._cache[cache_key] = data
            self._cache_timestamps[cache_key] = datetime.now()
        
        return data
    
    def get_price_data(self,
                      begin_date: Optional[int] = None,
                      end_date: Optional[int] = None,
                      validate: bool = True,
                      use_cache: bool = True) -> pd.DataFrame:
        """
        获取标准格式的价格数据
        
        Parameters
        ----------
        begin_date : int, optional
            开始日期，格式如20240101
        end_date : int, optional
            结束日期，格式如20241231
        validate : bool
            是否验证数据格式
        use_cache : bool
            是否使用缓存
            
        Returns
        -------
        pd.DataFrame
            验证过的价格数据
        """
        cache_key = f'price_data_{begin_date}_{end_date}'
        
        # 尝试从缓存获取
        if use_cache and cache_key in self._cache:
            logger.info("从缓存获取价格数据")
            return self._cache[cache_key]
        
        # 从数据获取器获取
        logger.info(f"获取价格数据: {begin_date} 到 {end_date}")
        data = self.fetcher.fetch_data(
            'price', 
            begin_date=begin_date or 20200101,
            end_date=end_date or 0
        )
        
        # 数据验证
        if validate:
            is_valid, errors = validate_price_data(data, strict=False)
            if not is_valid:
                raise ValueError(f"价格数据格式验证失败: {errors}")
            logger.info("价格数据格式验证通过")
        
        # 缓存数据
        if use_cache:
            self._cache[cache_key] = data
            self._cache_timestamps[cache_key] = datetime.now()
        
        return data
    
    def get_release_dates(self, use_cache: bool = True) -> pd.DataFrame:
        """
        获取财报发布日期数据
        
        Parameters
        ----------
        use_cache : bool
            是否使用缓存
            
        Returns
        -------
        pd.DataFrame
            发布日期数据
        """
        cache_key = 'release_dates'
        
        # 尝试从缓存获取
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]
        
        # 从文件读取
        file_path = self.auxiliary_path / 'ReleaseDates.pkl'
        if not file_path.exists():
            raise FileNotFoundError(f"发布日期文件不存在: {file_path}")
        
        data = pd.read_pickle(file_path)
        
        # 缓存数据
        if use_cache:
            self._cache[cache_key] = data
            self._cache_timestamps[cache_key] = datetime.now()
        
        return data
    
    def get_trading_dates(self, use_cache: bool = True) -> pd.Series:
        """
        获取交易日期列表
        
        Parameters
        ----------
        use_cache : bool
            是否使用缓存
            
        Returns
        -------
        pd.Series
            交易日期序列
        """
        cache_key = 'trading_dates'
        
        # 尝试从缓存获取
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]
        
        # 从文件读取
        file_path = self.auxiliary_path / 'TradingDates.pkl'
        if not file_path.exists():
            raise FileNotFoundError(f"交易日期文件不存在: {file_path}")
        
        data = pd.read_pickle(file_path)
        
        # 缓存数据
        if use_cache:
            self._cache[cache_key] = data
            self._cache_timestamps[cache_key] = datetime.now()
        
        return data
    
    def get_stock_info(self, use_cache: bool = True) -> pd.DataFrame:
        """
        获取股票基本信息
        
        Parameters
        ----------
        use_cache : bool
            是否使用缓存
            
        Returns
        -------
        pd.DataFrame
            股票信息数据
        """
        cache_key = 'stock_info'
        
        # 尝试从缓存获取
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]
        
        # 从文件读取
        file_path = self.auxiliary_path / 'StockInfo.pkl'
        if not file_path.exists():
            raise FileNotFoundError(f"股票信息文件不存在: {file_path}")
        
        data = pd.read_pickle(file_path)
        
        # 缓存数据
        if use_cache:
            self._cache[cache_key] = data
            self._cache_timestamps[cache_key] = datetime.now()
        
        return data
    
    def price_to_factor(self,
                       value_column: str = 'c',
                       begin_date: Optional[int] = None,
                       end_date: Optional[int] = None,
                       validate_output: bool = True) -> pd.Series:
        """
        获取价格数据并转换为标准因子格式
        
        Parameters
        ----------
        value_column : str
            要提取的价格字段
        begin_date : int, optional
            开始日期
        end_date : int, optional
            结束日期
        validate_output : bool
            是否验证输出格式
            
        Returns
        -------
        pd.Series
            标准因子格式的价格序列
        """
        # 获取价格数据
        price_df = self.get_price_data(begin_date, end_date)
        
        # 转换为因子格式
        factor_series = convert_to_factor_format(
            price_df, 
            value_col=value_column,
            date_col='tradingday',
            stock_col='code'
        )
        
        # 验证输出格式
        if validate_output:
            is_valid, errors = validate_factor_format(factor_series)
            if not is_valid:
                raise ValueError(f"转换后的因子格式验证失败: {errors}")
            logger.info("因子格式验证通过")
        
        return factor_series
    
    def financial_to_factor(self,
                           value_column: str,
                           validate_output: bool = True) -> pd.Series:
        """
        获取财务数据并转换为标准因子格式
        
        Parameters
        ----------
        value_column : str
            要提取的财务字段
        validate_output : bool
            是否验证输出格式
            
        Returns
        -------
        pd.Series
            标准因子格式的财务序列
        """
        # 获取财务数据
        financial_df = self.get_financial_data()
        
        # 检查字段是否存在
        if value_column not in financial_df.columns:
            available_cols = list(financial_df.columns)
            raise ValueError(f"财务数据中不存在字段 '{value_column}'。可用字段: {available_cols[:10]}...")
        
        # 转换为因子格式
        factor_series = DataConverter.financial_to_factor_format(
            financial_df,
            value_column=value_column,
            date_column='reportday',
            stock_column='code'
        )
        
        # 验证输出格式
        if validate_output:
            is_valid, errors = validate_factor_format(factor_series)
            if not is_valid:
                raise ValueError(f"转换后的因子格式验证失败: {errors}")
            logger.info("因子格式验证通过")
        
        return factor_series
    
    def get_data_quality_report(self, data_type: str = 'all') -> Dict[str, Any]:
        """
        获取数据质量报告
        
        Parameters
        ----------
        data_type : str
            数据类型，可选: 'price', 'financial', 'all'
            
        Returns
        -------
        Dict[str, Any]
            数据质量报告
        """
        reports = {}
        
        if data_type in ['price', 'all']:
            try:
                price_data = self.get_price_data(validate=False)
                reports['price'] = DataQualityChecker.generate_quality_report(
                    price_data, DataSchemas.PRICE_DATA
                )
            except Exception as e:
                logger.error(f"生成价格数据质量报告失败: {e}")
        
        if data_type in ['financial', 'all']:
            try:
                financial_data = self.get_financial_data(validate=False)
                reports['financial'] = DataQualityChecker.generate_quality_report(
                    financial_data, DataSchemas.FINANCIAL_DATA
                )
            except Exception as e:
                logger.error(f"生成财务数据质量报告失败: {e}")
        
        return reports
    
    def print_data_status(self):
        """打印数据状态概览"""
        print("\n📊 数据状态概览")
        print("=" * 60)
        
        # 检查各数据文件
        files_info = {
            'FinancialData_unified.pkl': '财务数据',
            'ReleaseDates.pkl': '发布日期',
            'StockInfo.pkl': '股票信息',
            'TradingDates.pkl': '交易日期'
        }
        
        for file_name, description in files_info.items():
            file_path = self.auxiliary_path / file_name
            if file_path.exists():
                mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                size_mb = file_path.stat().st_size / 1024 / 1024
                
                try:
                    # 读取数据获取基本信息
                    data = pd.read_pickle(file_path)
                    if isinstance(data, pd.DataFrame):
                        shape_info = f"({data.shape[0]:,}行, {data.shape[1]}列)"
                    elif isinstance(data, pd.Series):
                        shape_info = f"({len(data):,}项)"
                    else:
                        shape_info = "未知格式"
                    
                    print(f"✅ {description}: {shape_info}")
                    print(f"   更新时间: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
                    print(f"   文件大小: {size_mb:.1f}MB")
                    
                except Exception as e:
                    print(f"⚠️ {description}: 文件损坏 - {e}")
            else:
                print(f"❌ {description}: 文件不存在")
            print()
        
        # 缓存状态
        if self._cache:
            print(f"🔄 缓存状态: {len(self._cache)} 个数据集已缓存")
            for key, timestamp in self._cache_timestamps.items():
                age = (datetime.now() - timestamp).total_seconds() / 60  # 分钟
                print(f"   {key}: {age:.1f}分钟前")
        else:
            print("🔄 缓存状态: 空")
        
        print("=" * 60)
    
    def clear_cache(self):
        """清空缓存"""
        self._cache.clear()
        self._cache_timestamps.clear()
        logger.info("数据缓存已清空")
    
    def validate_all_data(self) -> Dict[str, Tuple[bool, List[str]]]:
        """
        验证所有数据格式
        
        Returns
        -------
        Dict[str, Tuple[bool, List[str]]]
            各数据类型的验证结果
        """
        results = {}
        
        # 验证价格数据
        try:
            price_data = self.get_price_data(validate=False)
            results['price'] = validate_price_data(price_data)
            logger.info(f"价格数据验证: {'通过' if results['price'][0] else '失败'}")
        except Exception as e:
            results['price'] = (False, [f"获取价格数据失败: {e}"])
        
        # 验证财务数据
        try:
            financial_data = self.get_financial_data(validate=False)
            results['financial'] = validate_financial_data(financial_data, strict=False)
            logger.info(f"财务数据验证: {'通过' if results['financial'][0] else '失败'}")
        except Exception as e:
            results['financial'] = (False, [f"获取财务数据失败: {e}"])
        
        return results


# 全局数据桥接器实例
_global_bridge = None

def get_data_bridge() -> DataBridge:
    """获取全局数据桥接器实例"""
    global _global_bridge
    if _global_bridge is None:
        _global_bridge = DataBridge()
    return _global_bridge

# 便捷函数
def get_factor_data(source: str, column: str, **kwargs) -> pd.Series:
    """
    便捷函数：获取标准格式的因子数据
    
    Parameters
    ----------
    source : str
        数据源，'price' 或 'financial'
    column : str
        字段名
    **kwargs
        其他参数
        
    Returns
    -------
    pd.Series
        标准因子格式的数据
    """
    bridge = get_data_bridge()
    
    if source == 'price':
        return bridge.price_to_factor(value_column=column, **kwargs)
    elif source == 'financial':
        return bridge.financial_to_factor(value_column=column, **kwargs)
    else:
        raise ValueError(f"不支持的数据源: {source}")

def validate_data_pipeline() -> bool:
    """
    验证整个数据管道
    
    Returns
    -------
    bool
        验证是否通过
    """
    bridge = get_data_bridge()
    
    print("\n🔍 数据管道验证")
    print("=" * 50)
    
    results = bridge.validate_all_data()
    all_passed = True
    
    for data_type, (is_valid, errors) in results.items():
        status = "✅ 通过" if is_valid else "❌ 失败"
        print(f"{data_type.ljust(10)}: {status}")
        
        if not is_valid:
            all_passed = False
            for error in errors[:3]:  # 只显示前3个错误
                print(f"  • {error}")
            if len(errors) > 3:
                print(f"  • ... 还有 {len(errors) - 3} 个错误")
    
    print(f"\n总体结果: {'✅ 数据管道验证通过' if all_passed else '❌ 数据管道验证失败'}")
    print("=" * 50)
    
    return all_passed