"""
Alpha191 数据适配器工具

提供数据格式转换的纯工具函数，不依赖Factor基类
"""
import pandas as pd
import numpy as np
from typing import Dict, Union, Optional
import logging

logger = logging.getLogger(__name__)


def convert_to_alpha191_format(price_data: pd.Series) -> pd.DataFrame:
    """
    将 MultiIndex 格式的价格数据转换为 Alpha191 所需格式
    
    Parameters
    ----------
    price_data : pd.Series
        MultiIndex格式 [TradingDates, StockCodes] 的价格数据
        
    Returns
    -------
    pd.DataFrame
        宽格式数据，索引为时间，列为股票代码
    """
    if not isinstance(price_data.index, pd.MultiIndex):
        raise ValueError("输入数据必须是 MultiIndex 格式")
    
    # 将长格式转换为宽格式 (时间 x 股票)
    wide_data = price_data.unstack(level='StockCodes')
    
    return wide_data


def prepare_alpha191_data(price_data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    准备 Alpha191 计算所需的所有数据字段
    
    Parameters
    ----------
    price_data : pd.DataFrame
        包含OHLCV的价格数据，MultiIndex格式
        
    Returns
    -------
    dict
        包含各字段的 DataFrame，索引为时间，列为股票代码
    """
    if not isinstance(price_data.index, pd.MultiIndex):
        raise ValueError("输入数据必须是 MultiIndex 格式")
    
    # 必需字段
    required_fields = ['open', 'high', 'low', 'close', 'volume']
    available_fields = price_data.columns.tolist()
    
    # 检查字段映射
    field_mapping = {
        'open': ['open', 'Open', 'OPEN', 'OpenPrice'],
        'high': ['high', 'High', 'HIGH', 'HighPrice'],
        'low': ['low', 'Low', 'LOW', 'LowPrice'], 
        'close': ['close', 'Close', 'CLOSE', 'ClosePrice'],
        'volume': ['volume', 'Volume', 'VOLUME', 'TradingVolume']
    }
    
    result = {}
    
    for field, possible_names in field_mapping.items():
        found_field = None
        for name in possible_names:
            if name in available_fields:
                found_field = name
                break
        
        if found_field:
            # 转换为宽格式
            wide_data = price_data[found_field].unstack(level='StockCodes')
            result[field] = wide_data
        else:
            logger.warning(f"未找到字段 {field}，可能的名称: {possible_names}")
    
    return result


def calculate_returns(close_data: pd.DataFrame) -> pd.DataFrame:
    """
    计算收益率数据
    
    Parameters
    ----------
    close_data : pd.DataFrame
        收盘价数据，索引为时间，列为股票代码
        
    Returns
    -------
    pd.DataFrame
        收益率数据
    """
    returns = close_data.pct_change()
    return returns


def calculate_vwap(high: pd.DataFrame, low: pd.DataFrame, 
                   close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """
    计算成交量加权平均价格 (VWAP)
    
    Parameters
    ----------
    high, low, close : pd.DataFrame
        价格数据
    volume : pd.DataFrame
        成交量数据
        
    Returns
    -------
    pd.DataFrame
        VWAP数据
    """
    typical_price = (high + low + close) / 3
    vwap = typical_price * volume / volume
    return vwap


def calculate_basic_features(data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    计算Alpha191所需的基础特征
    
    Parameters
    ----------
    data_dict : dict
        包含OHLCV数据的字典
        
    Returns
    -------
    dict
        增加了基础特征的数据字典
    """
    result = data_dict.copy()
    
    if all(k in data_dict for k in ['high', 'low', 'close', 'volume']):
        # 计算VWAP
        result['vwap'] = calculate_vwap(
            data_dict['high'], data_dict['low'], 
            data_dict['close'], data_dict['volume']
        )
    
    if 'close' in data_dict:
        # 计算收益率
        result['returns'] = calculate_returns(data_dict['close'])
    
    return result


def validate_alpha191_data(data_dict: Dict[str, pd.DataFrame]) -> bool:
    """
    验证Alpha191数据的完整性和格式
    
    Parameters
    ----------
    data_dict : dict
        数据字典
        
    Returns
    -------
    bool
        数据是否有效
    """
    required_fields = ['open', 'high', 'low', 'close', 'volume']
    
    # 检查必需字段
    missing_fields = [field for field in required_fields if field not in data_dict]
    if missing_fields:
        logger.error(f"缺少必需字段: {missing_fields}")
        return False
    
    # 检查数据形状一致性
    shapes = [(field, df.shape) for field, df in data_dict.items()]
    base_shape = shapes[0][1]
    
    for field, shape in shapes[1:]:
        if shape != base_shape:
            logger.error(f"字段 {field} 的形状 {shape} 与基准 {base_shape} 不一致")
            return False
    
    # 检查数据类型
    for field, df in data_dict.items():
        if not pd.api.types.is_numeric_dtype(df.values.flat[0]):
            logger.warning(f"字段 {field} 包含非数值数据")
    
    logger.info(f"数据验证通过，包含 {len(data_dict)} 个字段，形状 {base_shape}")
    return True