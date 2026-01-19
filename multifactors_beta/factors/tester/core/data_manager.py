"""
单因子测试数据管理器
负责数据加载、预处理和准备
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, List, Any
import logging
import os
from pathlib import Path

from config import get_config, config_manager
from core.utils import OutlierHandler, Normalizer, DataCleaner
from core.utils.factor_processing import FactorOrthogonalizer

logger = logging.getLogger(__name__)


class DataManager:
    """数据管理器"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化数据管理器
        
        Parameters
        ----------
        config : Dict, optional
            配置字典，如果为None则从全局配置读取
        """
        self.config = config or get_config('main.factor_test')
        self.data_cache = {}  # 数据缓存
        self._load_basic_data()
        
    def _load_basic_data(self):
        """加载基础数据"""
        try:
            # 获取数据路径
            data_path = get_config('main.paths.data_root')
            
            # 加载交易日期
            trading_dates_file = os.path.join(data_path, 'TradingDates.pkl')
            if os.path.exists(trading_dates_file):
                self.trading_dates = pd.read_pickle(trading_dates_file)
                logger.info(f"加载交易日期数据: {len(self.trading_dates)} days")
            else:
                logger.warning(f"交易日期文件不存在: {trading_dates_file}")
                self.trading_dates = None
                
        except Exception as e:
            logger.error(f"加载基础数据失败: {e}")
            self.trading_dates = None
    
    def load_return_data(self, return_type: str = 'daily', price_type: str = 'o2o') -> pd.Series:
        """
        加载收益率数据

        Parameters
        ----------
        return_type : str
            收益类型 ('daily', 'weekly', 'monthly')
        price_type : str
            价格类型 ('o2o', 'vwap')

        Returns
        -------
        pd.Series
            收益率数据
        """
        cache_key = f"return_{return_type}_{price_type}"
        if cache_key in self.data_cache:
            return self.data_cache[cache_key]

        try:
            # 优先从factor_test.auxiliary_data_path加载，然后paths.auxiliary_data，最后data_root
            auxiliary_path = self.config.get('auxiliary_data_path') or get_config('main.paths.auxiliary_data')
            data_path = auxiliary_path if auxiliary_path else get_config('main.paths.data_root')
            filename = f"LogReturn_{return_type}_{price_type}.pkl"
            filepath = os.path.join(data_path, filename)
            
            if os.path.exists(filepath):
                return_data = pd.read_pickle(filepath)
                return_data = return_data.sort_index(level=0)
                self.data_cache[cache_key] = return_data
                logger.info(f"加载收益率数据: {filename}")
                return return_data
            else:
                logger.error(f"收益率文件不存在: {filepath}")
                return pd.Series()
                
        except Exception as e:
            logger.error(f"加载收益率数据失败: {e}")
            return pd.Series()
    
    def load_base_factors(self, base_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        加载基准因子数据
        
        Parameters
        ----------
        base_names : List[str], optional
            基准因子名称列表，如果为None则从配置读取
            
        Returns
        -------
        pd.DataFrame
            基准因子数据
        """
        if base_names is None:
            base_names = self.config.get('base_factors', [])
        
        if not base_names or (len(base_names) == 1 and base_names[0] == ''):
            logger.info("无基准因子")
            return pd.DataFrame()
        
        cache_key = f"base_{'_'.join(base_names)}"
        if cache_key in self.data_cache:
            return self.data_cache[cache_key]
        
        try:
            base_data_list = []
            raw_factors_path = get_config('main.paths.raw_factors')
            
            for factor_name in base_names:
                filepath = os.path.join(raw_factors_path, f"{factor_name}.pkl")
                if os.path.exists(filepath):
                    data = pd.read_pickle(filepath)
                    if isinstance(data, pd.Series):
                        data = data.to_frame(name=factor_name)
                    elif isinstance(data, pd.DataFrame) and len(data.columns) == 1:
                        data.columns = [factor_name]
                    base_data_list.append(data)
                    logger.info(f"加载基准因子: {factor_name}")
                else:
                    logger.warning(f"基准因子文件不存在: {filepath}")
            
            if base_data_list:
                base_data = pd.concat(base_data_list, axis=1, join='inner')
                # 正交化处理
                base_data = FactorOrthogonalizer.sequential_orthogonalize(base_data)
                self.data_cache[cache_key] = base_data
                return base_data
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"加载基准因子失败: {e}")
            return pd.DataFrame()
    
    def load_industry_data(self) -> pd.DataFrame:
        """
        加载行业分类数据（独热编码）
        
        Returns
        -------
        pd.DataFrame
            行业独热编码数据
        """
        cache_key = "industry_one_hot"
        if cache_key in self.data_cache:
            return self.data_cache[cache_key]
        
        try:
            classification_name = self.config.get('classification_name', '')
            if not classification_name:
                logger.info("未配置行业分类")
                return pd.DataFrame()
            
            data_path = get_config('main.paths.classification_data')
            filepath = os.path.join(data_path, f"{classification_name}.pkl")
            
            if os.path.exists(filepath):
                industry_data = pd.read_pickle(filepath)
                
                # 数据清洗
                def _clean_industry(slice_data):
                    slice_data = slice_data.fillna(0).astype(float)
                    # 记录原始列名
                    original_cols = slice_data.columns.tolist()
                    # 删除全零列，但保持结构追踪
                    valid_cols_mask = (slice_data != 0).any(axis=0)
                    active_cols = slice_data.columns[valid_cols_mask].tolist()
                    
                    if len(active_cols) > 0:
                        # 保留活跃列
                        slice_data_clean = slice_data[active_cols]
                        # 归一化，使每行和为1
                        row_sum = slice_data_clean.sum(axis=1)
                        row_sum = row_sum.replace(0, 1)  # 防止除零
                        slice_data_clean = slice_data_clean.div(row_sum, axis=0)
                        return slice_data_clean
                    else:
                        # 所有列都是零，返回空DataFrame
                        return pd.DataFrame(index=slice_data.index)
                
                industry_data = industry_data.groupby('TradingDates', group_keys=False).apply(_clean_industry)
                
                # 删除退市股票列
                columns_to_drop = [col for col in industry_data.columns if '(退市)' in col]
                if columns_to_drop:
                    industry_data = industry_data.drop(columns=columns_to_drop)
                
                self.data_cache[cache_key] = industry_data
                logger.info(f"加载行业分类数据: {classification_name}")
                return industry_data
            else:
                logger.warning(f"行业分类文件不存在: {filepath}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"加载行业分类数据失败: {e}")
            return pd.DataFrame()
    
    def load_factor_data(self, factor_name: str, version: str = 'auto') -> pd.Series:
        """
        加载待测试因子数据
        
        Parameters
        ----------
        factor_name : str
            因子名称
        version : str
            因子版本选择
            - 'raw': 只加载原始因子
            - 'orthogonal': 只加载正交化因子
            - 'auto': 自动选择（优先正交化因子）
            
        Returns
        -------
        pd.Series
            因子数据
        """
        # 生成包含版本信息的缓存键
        cache_key = f"factor_{factor_name}_{version}"
        if cache_key in self.data_cache:
            logger.debug(f"从缓存获取因子数据: {factor_name} (version: {version})")
            return self.data_cache[cache_key]
        
        try:
            # 根据版本参数决定搜索策略
            search_paths = []
            
            if version == 'raw':
                # 只搜索原始因子
                search_paths = [
                    (get_config('main.paths.raw_factors'), f"{factor_name}.pkl", "raw"),
                    (get_config('main.paths.raw_factors_alpha191'), f"{factor_name}.pkl", "alpha191"),
                ]
            elif version == 'orthogonal':
                # 只搜索正交化因子
                search_paths = [
                    (get_config('main.paths.orthogonalization_factors'), f"{factor_name}_orth.pkl", "orthogonal"),
                    (get_config('main.paths.orthogonalization_factors'), f"{factor_name}.pkl", "orthogonal_alt"),
                ]
            else:  # auto
                # 优先搜索正交化因子，然后搜索原始因子
                config_use_orth = self.config.get('use_orthogonal_factors', False)
                config_version = self.config.get('factor_version', 'auto')
                
                if config_use_orth or config_version == 'orthogonal':
                    search_paths = [
                        (get_config('main.paths.orthogonalization_factors'), f"{factor_name}_orth.pkl", "orthogonal"),
                        (get_config('main.paths.orthogonalization_factors'), f"{factor_name}.pkl", "orthogonal_alt"),
                        (get_config('main.paths.raw_factors'), f"{factor_name}.pkl", "raw"),
                        (get_config('main.paths.raw_factors_alpha191'), f"{factor_name}.pkl", "alpha191"),
                        (get_config('main.paths.factors'), f"{factor_name}.pkl", "generated"),
                    ]
                else:
                    search_paths = [
                        (get_config('main.paths.raw_factors'), f"{factor_name}.pkl", "raw"),
                        (get_config('main.paths.raw_factors_alpha191'), f"{factor_name}.pkl", "alpha191"),
                        (get_config('main.paths.factors'), f"{factor_name}.pkl", "generated"),
                        (get_config('main.paths.orthogonalization_factors'), f"{factor_name}_orth.pkl", "orthogonal"),
                        (get_config('main.paths.orthogonalization_factors'), f"{factor_name}.pkl", "orthogonal_alt"),
                    ]
            
            # 搜索并加载因子文件
            factor_data = None
            loaded_from = None
            
            for path, filename, source_type in search_paths:
                if not path:
                    continue
                    
                filepath = os.path.join(path, filename)
                if os.path.exists(filepath):
                    try:
                        factor_data = pd.read_pickle(filepath)
                        loaded_from = source_type
                        logger.info(f"从 {source_type} 加载因子: {factor_name} ({filepath})")
                        break
                    except Exception as e:
                        logger.warning(f"加载文件失败 {filepath}: {e}")
                        continue
            
            if factor_data is None:
                logger.error(f"因子文件不存在: {factor_name} (version: {version})")
                return pd.Series()
            
            # 确保是Series格式
            if isinstance(factor_data, pd.DataFrame):
                if len(factor_data.columns) == 1:
                    factor_data = factor_data.iloc[:, 0]
                else:
                    logger.warning(f"因子数据有多列，使用第一列: {list(factor_data.columns)}")
                    factor_data = factor_data.iloc[:, 0]
            
            factor_data.name = 'factor'
            factor_data = factor_data.sort_index(level=0)
            
            # 应用时间过滤
            begin_date = pd.to_datetime(self.config.get('begin_date', '2018-01-01'))
            end_date = pd.to_datetime(self.config.get('end_date', '2025-12-31'))
            factor_data = factor_data.loc[
                (factor_data.index.get_level_values(0) >= begin_date) &
                (factor_data.index.get_level_values(0) <= end_date)
            ]
            
            logger.info(f"✅ 加载因子数据: {factor_name} ({loaded_from}), shape={factor_data.shape}")
            
            # 存储到缓存
            self.data_cache[cache_key] = factor_data
            
            return factor_data
                
        except Exception as e:
            logger.error(f"加载因子数据失败: {e}")
            return pd.Series()
    
    def prepare_test_data(
        self, 
        factor_name: str,
        use_base_factors: bool = True,
        use_industry: bool = True,
        custom_base_factors: Optional[List[str]] = None,
        factor_version: str = 'auto'
    ) -> Dict[str, Any]:
        """
        准备测试数据
        
        Parameters
        ----------
        factor_name : str
            因子名称
        use_base_factors : bool
            是否使用基准因子
        use_industry : bool
            是否使用行业分类
        custom_base_factors : List[str], optional
            自定义基准因子列表
        factor_version : str, default 'auto'
            因子版本选择 ('raw', 'orthogonal', 'auto')
            
        Returns
        -------
        Dict
            包含所有测试所需数据的字典
        """
        test_data = {}
        
        # 加载因子数据
        test_data['factor'] = self.load_factor_data(factor_name, version=factor_version)
        if test_data['factor'].empty:
            logger.error(f"因子数据为空: {factor_name}")
            return test_data
        
        # 加载收益率数据
        backtest_type = self.config.get('backtest_type', 'daily')
        price_type = self.config.get('back_test_trading_price', 'o2o')
        test_data['returns'] = self.load_return_data(backtest_type, price_type)
        
        # 加载基准因子
        if use_base_factors:
            base_factors = custom_base_factors or self.config.get('base_factors', [])
            test_data['base_factors'] = self.load_base_factors(base_factors)
        else:
            test_data['base_factors'] = pd.DataFrame()
        
        # 加载行业数据
        if use_industry:
            test_data['industry'] = self.load_industry_data()
        else:
            test_data['industry'] = pd.DataFrame()
        
        # 合并基准因子和行业数据
        if not test_data['base_factors'].empty and not test_data['industry'].empty:
            # 合并基准因子和行业独热编码
            test_data['control_variables'] = self._merge_base_and_industry(
                test_data['base_factors'], 
                test_data['industry']
            )
        elif not test_data['base_factors'].empty:
            test_data['control_variables'] = test_data['base_factors']
        elif not test_data['industry'].empty:
            test_data['control_variables'] = test_data['industry']
        else:
            test_data['control_variables'] = pd.DataFrame()
        
        # 数据对齐
        test_data = self._align_data(test_data)
        
        # 添加数据信息
        test_data['data_info'] = {
            'factor_name': factor_name,
            'factor_count': len(test_data['factor']),
            'date_range': [
                test_data['factor'].index.get_level_values(0).min().strftime('%Y-%m-%d'),
                test_data['factor'].index.get_level_values(0).max().strftime('%Y-%m-%d')
            ] if len(test_data['factor']) > 0 else ['', ''],
            'stock_count': len(test_data['factor'].index.get_level_values(1).unique()) if len(test_data['factor']) > 0 else 0,
            'base_factors': custom_base_factors or self.config.get('base_factors', []),
            'use_industry': use_industry
        }
        
        return test_data
    
    def _merge_base_and_industry(
        self, 
        base_factors: pd.DataFrame, 
        industry_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        合并基准因子和行业数据
        
        Parameters
        ----------
        base_factors : pd.DataFrame
            基准因子数据
        industry_data : pd.DataFrame
            行业独热编码数据
            
        Returns
        -------
        pd.DataFrame
            合并后的控制变量
        """
        # 合并数据，先去重防止重复索引
        base_factors_clean = base_factors[~base_factors.index.duplicated(keep='first')]
        industry_data_clean = industry_data[~industry_data.index.duplicated(keep='first')]
        merged_data = base_factors_clean.join(industry_data_clean, how='left')
        
        # 前向填充行业数据
        merged_data = merged_data.groupby('StockCodes').fillna(method='ffill')
        
        # 按日期处理
        def _day_processing(slice_data):
            # 分离基准因子和行业
            base_cols = base_factors.columns.tolist()
            industry_cols = [col for col in slice_data.columns if col not in base_cols]
            
            # 处理行业数据
            if industry_cols:
                industry_slice = slice_data[industry_cols].fillna(0)
                # 检查哪些行业列是活跃的
                valid_industry_mask = (industry_slice != 0).any(axis=0)
                active_industry_cols = [col for col, valid in zip(industry_cols, valid_industry_mask) if valid]
                
                if active_industry_cols:
                    # 只保留活跃的行业列，重新构建DataFrame
                    result_data = pd.concat([
                        slice_data[base_cols],
                        slice_data[active_industry_cols]
                    ], axis=1)
                    return result_data
                else:
                    # 所有行业列都无效，只保留基准因子
                    logger.warning(f"所有行业列都为零，只保留基准因子")
                    return slice_data[base_cols]
            else:
                return slice_data
        
        merged_data = merged_data.groupby(level=0, group_keys=False).apply(_day_processing)
        
        return merged_data
    
    def _align_data(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        对齐所有数据
        
        Parameters
        ----------
        data_dict : Dict
            数据字典
            
        Returns
        -------
        Dict
            对齐后的数据字典
        """
        # 找到共同的索引
        valid_dfs = []
        for key in ['factor', 'returns']:
            if key in data_dict and not data_dict[key].empty:
                if isinstance(data_dict[key], pd.Series):
                    valid_dfs.append(data_dict[key].to_frame())
                else:
                    valid_dfs.append(data_dict[key])
        
        if valid_dfs:
            # 获取共同索引
            common_index = valid_dfs[0].index
            for df in valid_dfs[1:]:
                common_index = common_index.intersection(df.index)

            # 对齐所有数据
            for key in data_dict:
                if key in ['factor', 'returns', 'control_variables']:
                    if isinstance(data_dict[key], (pd.DataFrame, pd.Series)) and not data_dict[key].empty:
                        # 获取当前数据的索引与common_index的交集，避免KeyError
                        aligned_index = data_dict[key].index.intersection(common_index)
                        data_dict[key] = data_dict[key].loc[aligned_index]

            logger.info(f"数据对齐完成，共同样本数: {len(common_index)}")
        
        return data_dict
    
    def clear_cache(self):
        """清空缓存"""
        self.data_cache.clear()
        logger.info("数据缓存已清空")