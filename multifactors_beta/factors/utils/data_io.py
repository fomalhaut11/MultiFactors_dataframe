#!/usr/bin/env python3
"""
因子数据输入输出模块

提供因子数据的加载、保存和管理功能：
- 因子数据保存到文件
- 因子数据从文件加载
- 因子数据的版本管理
- 支持多种文件格式

Author: AI Assistant  
Date: 2025-09-03
"""

import pandas as pd
import numpy as np
import os
from typing import Optional, Dict, Any, Union
import logging
import pickle
from datetime import datetime
import json

logger = logging.getLogger(__name__)


def save_factor(factor_data: pd.Series, 
                factor_name: str, 
                path: Optional[str] = None,
                metadata: Optional[Dict[str, Any]] = None,
                format: str = 'pickle') -> str:
    """
    保存因子数据到文件
    
    Parameters
    ----------
    factor_data : pd.Series
        因子数据，MultiIndex格式[TradingDates, StockCodes]
    factor_name : str
        因子名称
    path : str, optional
        保存路径，如果不指定则使用默认路径
    metadata : Dict[str, Any], optional
        因子元数据信息
    format : str, default 'pickle'
        保存格式：'pickle', 'parquet', 'csv'
        
    Returns
    -------
    str
        保存的文件路径
        
    Examples
    --------
    >>> # 保存因子到默认路径
    >>> path = save_factor(my_factor, 'ROE_ttm')
    >>> 
    >>> # 保存到指定路径并包含元数据
    >>> metadata = {'description': 'ROE TTM因子', 'version': '1.0'}
    >>> path = save_factor(my_factor, 'ROE_ttm', metadata=metadata)
    """
    if not isinstance(factor_data, pd.Series):
        raise ValueError("因子数据必须是pandas Series格式")
    
    if not isinstance(factor_data.index, pd.MultiIndex):
        raise ValueError("因子数据必须是MultiIndex格式")
    
    if not factor_name:
        raise ValueError("因子名称不能为空")
    
    try:
        # 确定保存路径
        if path is None:
            try:
                from config import get_config
                factor_dir = get_config('main.paths.factors', 'data/factors')
            except:
                factor_dir = 'data/factors'
            
            os.makedirs(factor_dir, exist_ok=True)
            
            if format == 'pickle':
                path = os.path.join(factor_dir, f"{factor_name}.pkl")
            elif format == 'parquet':
                path = os.path.join(factor_dir, f"{factor_name}.parquet")
            elif format == 'csv':
                path = os.path.join(factor_dir, f"{factor_name}.csv")
            else:
                raise ValueError(f"不支持的保存格式: {format}")
        
        # 准备元数据
        if metadata is None:
            metadata = {}
        
        metadata.update({
            'factor_name': factor_name,
            'created_time': datetime.now().isoformat(),
            'data_shape': factor_data.shape,
            'data_type': str(factor_data.dtype),
            'index_names': list(factor_data.index.names),
            'format': format
        })
        
        # 保存数据
        if format == 'pickle':
            # pickle格式可以保存完整的pandas对象和元数据
            factor_dict = {
                'data': factor_data,
                'metadata': metadata
            }
            with open(path, 'wb') as f:
                pickle.dump(factor_dict, f)
                
        elif format == 'parquet':
            # parquet格式保存数据，元数据单独保存
            factor_df = factor_data.to_frame(factor_name)
            factor_df.to_parquet(path)
            
            # 保存元数据
            metadata_path = path.replace('.parquet', '_metadata.json')
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
                
        elif format == 'csv':
            # CSV格式保存数据，元数据单独保存
            factor_df = factor_data.to_frame(factor_name)
            factor_df.to_csv(path, encoding='utf-8')
            
            # 保存元数据
            metadata_path = path.replace('.csv', '_metadata.json')
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        logger.info(f"因子 '{factor_name}' 保存成功: {path}")
        return path
        
    except Exception as e:
        logger.error(f"保存因子失败: {e}")
        raise RuntimeError(f"保存因子失败: {e}")


def load_factor(factor_name: str, 
                path: Optional[str] = None,
                format: Optional[str] = None) -> Dict[str, Any]:
    """
    从文件加载因子数据
    
    Parameters
    ----------
    factor_name : str
        因子名称
    path : str, optional
        文件路径，如果不指定则从默认路径查找
    format : str, optional
        文件格式，如果不指定则自动检测
        
    Returns
    -------
    Dict[str, Any]
        包含'data'和'metadata'的字典
        - data: 因子数据 (pd.Series)
        - metadata: 因子元数据 (dict)
        
    Examples
    --------
    >>> # 从默认路径加载因子
    >>> result = load_factor('ROE_ttm')
    >>> factor_data = result['data']
    >>> metadata = result['metadata']
    >>> 
    >>> # 从指定路径加载
    >>> result = load_factor('my_factor', path='/path/to/factor.pkl')
    """
    if not factor_name:
        raise ValueError("因子名称不能为空")
    
    try:
        # 确定文件路径
        if path is None:
            try:
                from config import get_config
                factor_dir = get_config('main.paths.factors', 'data/factors')
            except:
                factor_dir = 'data/factors'
            
            # 按优先级顺序尝试不同格式
            for fmt in ['pickle', 'parquet', 'csv']:
                if fmt == 'pickle':
                    candidate_path = os.path.join(factor_dir, f"{factor_name}.pkl")
                elif fmt == 'parquet':
                    candidate_path = os.path.join(factor_dir, f"{factor_name}.parquet")
                elif fmt == 'csv':
                    candidate_path = os.path.join(factor_dir, f"{factor_name}.csv")
                
                if os.path.exists(candidate_path):
                    path = candidate_path
                    format = fmt
                    break
            
            if path is None:
                raise FileNotFoundError(f"找不到因子文件: {factor_name}")
        
        # 自动检测格式
        if format is None:
            if path.endswith('.pkl'):
                format = 'pickle'
            elif path.endswith('.parquet'):
                format = 'parquet'
            elif path.endswith('.csv'):
                format = 'csv'
            else:
                raise ValueError(f"无法检测文件格式: {path}")
        
        # 加载数据
        if format == 'pickle':
            with open(path, 'rb') as f:
                factor_dict = pickle.load(f)
            
            if isinstance(factor_dict, dict) and 'data' in factor_dict:
                return factor_dict
            else:
                # 兼容旧格式（直接保存的Series）
                return {
                    'data': factor_dict,
                    'metadata': {'factor_name': factor_name, 'format': 'pickle'}
                }
                
        elif format == 'parquet':
            factor_df = pd.read_parquet(path)
            if len(factor_df.columns) == 1:
                factor_data = factor_df.iloc[:, 0]
                factor_data.name = factor_name
            else:
                raise ValueError(f"parquet文件包含多列，无法确定因子列: {path}")
            
            # 尝试加载元数据
            metadata_path = path.replace('.parquet', '_metadata.json')
            metadata = {'factor_name': factor_name, 'format': 'parquet'}
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata.update(json.load(f))
            
            return {'data': factor_data, 'metadata': metadata}
            
        elif format == 'csv':
            factor_df = pd.read_csv(path, index_col=[0, 1])
            if len(factor_df.columns) == 1:
                factor_data = factor_df.iloc[:, 0]
                factor_data.name = factor_name
            else:
                raise ValueError(f"CSV文件包含多列，无法确定因子列: {path}")
            
            # 尝试加载元数据
            metadata_path = path.replace('.csv', '_metadata.json')
            metadata = {'factor_name': factor_name, 'format': 'csv'}
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata.update(json.load(f))
            
            return {'data': factor_data, 'metadata': metadata}
        
        logger.info(f"因子 '{factor_name}' 加载成功: {path}")
        
    except FileNotFoundError:
        logger.error(f"因子文件不存在: {factor_name}")
        raise
    except Exception as e:
        logger.error(f"加载因子失败: {e}")
        raise RuntimeError(f"加载因子失败: {e}")


def list_factors(factor_dir: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """
    列出可用的因子
    
    Parameters
    ----------
    factor_dir : str, optional
        因子目录，如果不指定则使用默认目录
        
    Returns
    -------
    Dict[str, Dict[str, Any]]
        因子信息字典，key为因子名，value为因子信息
        
    Examples
    --------
    >>> factors = list_factors()
    >>> for name, info in factors.items():
    ...     print(f"{name}: {info['format']}, {info['size']}")
    """
    if factor_dir is None:
        try:
            from config import get_config
            factor_dir = get_config('main.paths.factors', 'data/factors')
        except:
            factor_dir = 'data/factors'
    
    if not os.path.exists(factor_dir):
        return {}
    
    factors = {}
    
    try:
        for filename in os.listdir(factor_dir):
            if filename.endswith(('_metadata.json',)):
                continue
            
            file_path = os.path.join(factor_dir, filename)
            if not os.path.isfile(file_path):
                continue
            
            # 提取因子名和格式
            if filename.endswith('.pkl'):
                factor_name = filename[:-4]
                format = 'pickle'
            elif filename.endswith('.parquet'):
                factor_name = filename[:-8]
                format = 'parquet'
            elif filename.endswith('.csv'):
                factor_name = filename[:-4]
                format = 'csv'
            else:
                continue
            
            # 获取文件信息
            stat = os.stat(file_path)
            factor_info = {
                'factor_name': factor_name,
                'format': format,
                'path': file_path,
                'size': stat.st_size,
                'modified_time': datetime.fromtimestamp(stat.st_mtime).isoformat()
            }
            
            # 尝试加载元数据
            try:
                if format == 'pickle':
                    with open(file_path, 'rb') as f:
                        data = pickle.load(f)
                    if isinstance(data, dict) and 'metadata' in data:
                        factor_info.update(data['metadata'])
                else:
                    metadata_path = file_path.replace(f'.{format}', '_metadata.json')
                    if os.path.exists(metadata_path):
                        with open(metadata_path, 'r', encoding='utf-8') as f:
                            metadata = json.load(f)
                        factor_info.update(metadata)
            except:
                pass  # 忽略元数据加载错误
            
            factors[factor_name] = factor_info
        
        logger.info(f"找到 {len(factors)} 个因子文件")
        return factors
        
    except Exception as e:
        logger.error(f"列出因子失败: {e}")
        return {}


def delete_factor(factor_name: str, factor_dir: Optional[str] = None) -> bool:
    """
    删除因子文件
    
    Parameters
    ----------
    factor_name : str
        因子名称
    factor_dir : str, optional
        因子目录
        
    Returns
    -------
    bool
        是否删除成功
        
    Examples
    --------
    >>> success = delete_factor('old_factor')
    """
    if factor_dir is None:
        try:
            from config import get_config
            factor_dir = get_config('main.paths.factors', 'data/factors')
        except:
            factor_dir = 'data/factors'
    
    try:
        deleted_files = []
        
        # 删除所有格式的文件
        for fmt in ['pickle', 'parquet', 'csv']:
            if fmt == 'pickle':
                file_path = os.path.join(factor_dir, f"{factor_name}.pkl")
            elif fmt == 'parquet':
                file_path = os.path.join(factor_dir, f"{factor_name}.parquet")
                metadata_path = os.path.join(factor_dir, f"{factor_name}_metadata.json")
            elif fmt == 'csv':
                file_path = os.path.join(factor_dir, f"{factor_name}.csv")
                metadata_path = os.path.join(factor_dir, f"{factor_name}_metadata.json")
            
            if os.path.exists(file_path):
                os.remove(file_path)
                deleted_files.append(file_path)
            
            if fmt != 'pickle' and 'metadata_path' in locals() and os.path.exists(metadata_path):
                os.remove(metadata_path)
                deleted_files.append(metadata_path)
        
        if deleted_files:
            logger.info(f"删除因子 '{factor_name}' 成功，删除文件: {deleted_files}")
            return True
        else:
            logger.warning(f"因子文件不存在: {factor_name}")
            return False
            
    except Exception as e:
        logger.error(f"删除因子失败: {e}")
        return False


__all__ = [
    'save_factor',
    'load_factor', 
    'list_factors',
    'delete_factor'
]