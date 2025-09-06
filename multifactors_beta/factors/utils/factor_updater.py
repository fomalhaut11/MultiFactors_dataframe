"""
因子更新模块 - 支持全量和增量更新
基于原始数据的更新状态进行增量计算
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path
import json
from datetime import datetime, date
import logging

from ..base.time_series_processor import TimeSeriesProcessor
from ..calculator.factor_calculator import FactorCalculator
from config import get_config, config_managernfig

logger = logging.getLogger(__name__)


class UpdateTracker:
    """更新状态追踪器"""
    
    def __init__(self, tracker_file: Optional[Path] = None):
        """
        Parameters:
        -----------
        tracker_file : 状态追踪文件路径
        """
        if tracker_file is None:
            tracker_file = Path(config.get_config('main.paths.data_root')) / 'factor_update_tracker.json'
        
        self.tracker_file = Path(tracker_file)
        self.status = self._load_status()
        
    def _load_status(self) -> Dict:
        """加载更新状态"""
        if self.tracker_file.exists():
            try:
                with open(self.tracker_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load tracker file: {e}")
                return {}
        return {}
    
    def save_status(self):
        """保存更新状态"""
        try:
            self.tracker_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.tracker_file, 'w', encoding='utf-8') as f:
                json.dump(self.status, f, indent=2, default=str)
            logger.info(f"Saved update status to {self.tracker_file}")
        except Exception as e:
            logger.error(f"Failed to save tracker file: {e}")
    
    def get_last_update(self, data_type: str) -> Optional[Dict]:
        """获取某类数据的最后更新信息"""
        return self.status.get(data_type, {})
    
    def update_status(self, data_type: str, info: Dict):
        """更新某类数据的状态"""
        self.status[data_type] = info
        self.save_status()


class FactorUpdater:
    """因子更新器"""
    
    def __init__(self, 
                 data_path: Optional[Path] = None,
                 factor_path: Optional[Path] = None):
        """
        Parameters:
        -----------
        data_path : 原始数据路径
        factor_path : 因子保存路径
        """
        self.data_path = Path(data_path) if data_path else config.get_config('main.paths.data_root')
        self.factor_path = Path(factor_path) if factor_path else config.get_config('main.paths.factors')
        
        self.tracker = UpdateTracker()
        self.calculator = FactorCalculator()
        
    def check_financial_updates(self, 
                              current_financial_data: pd.DataFrame) -> Tuple[bool, pd.DataFrame]:
        """
        检查财务数据更新
        
        Parameters:
        -----------
        current_financial_data : 当前的财务数据，必须包含报表发布日期列
        
        Returns:
        --------
        (has_updates, new_data) : 是否有更新，新增数据
        """
        # 获取上次更新信息
        last_update = self.tracker.get_last_update('financial')
        
        if not last_update:
            logger.info("No previous update record found, treating as full update")
            return True, current_financial_data
        
        # 获取上次处理的最新发布日期
        last_release_date = pd.to_datetime(last_update.get('last_release_date'))
        
        # 假设财务数据有 'ReleasedDates' 或 'reportday' 列表示发布日期
        release_date_col = None
        for col in ['ReleasedDates', 'reportday', 'releaseddate']:
            if col in current_financial_data.columns:
                release_date_col = col
                break
        
        if release_date_col is None:
            logger.error("No release date column found in financial data")
            return False, pd.DataFrame()
        
        # 找出新发布的数据
        current_financial_data[release_date_col] = pd.to_datetime(current_financial_data[release_date_col])
        new_data = current_financial_data[current_financial_data[release_date_col] > last_release_date]
        
        if len(new_data) > 0:
            logger.info(f"Found {len(new_data)} new financial records")
            return True, new_data
        else:
            logger.info("No new financial data found")
            return False, pd.DataFrame()
    
    def check_price_updates(self,
                          current_price_data: pd.DataFrame) -> Tuple[bool, pd.DataFrame]:
        """
        检查价格数据更新
        
        Parameters:
        -----------
        current_price_data : 当前的价格数据
        
        Returns:
        --------
        (has_updates, new_data) : 是否有更新，新增数据
        """
        # 获取上次更新信息
        last_update = self.tracker.get_last_update('price')
        
        if not last_update:
            logger.info("No previous update record found, treating as full update")
            return True, current_price_data
        
        # 获取上次处理的最新交易日
        last_trading_date = pd.to_datetime(last_update.get('last_trading_date'))
        
        # 找出新的交易日数据
        trading_dates = current_price_data.index.get_level_values('TradingDates')
        new_data = current_price_data[trading_dates > last_trading_date]
        
        if len(new_data) > 0:
            logger.info(f"Found {len(new_data)} new price records")
            return True, new_data
        else:
            logger.info("No new price data found")
            return False, pd.DataFrame()
    
    def update_fundamental_factors(self,
                                 factor_names: List[str],
                                 mode: str = 'incremental',
                                 **kwargs) -> Dict[str, pd.Series]:
        """
        更新基本面因子
        
        Parameters:
        -----------
        factor_names : 要更新的因子列表
        mode : 'full' 或 'incremental'
        **kwargs : 传递给calculate_factors的参数
        
        Returns:
        --------
        更新后的因子字典
        """
        logger.info(f"Updating fundamental factors in {mode} mode")
        
        # 加载财务数据
        financial_data = kwargs.get('financial_data')
        if financial_data is None:
            financial_file = self.data_path / 'financial_data.pkl'
            if financial_file.exists():
                financial_data = pd.read_pickle(financial_file)
            else:
                logger.error("Financial data not found")
                return {}
        
        # 检查更新
        has_updates, new_financial_data = self.check_financial_updates(financial_data)
        
        if mode == 'incremental' and not has_updates:
            logger.info("No updates found for incremental update")
            return self._load_existing_factors(factor_names)
        
        # 根据模式选择数据
        if mode == 'incremental' and has_updates:
            # 增量更新：只处理新数据
            logger.info("Processing incremental updates")
            
            # 获取新数据涉及的股票
            updated_stocks = new_financial_data.index.get_level_values('StockCodes').unique()
            
            # 加载现有因子
            existing_factors = self._load_existing_factors(factor_names)
            
            # 对每个更新的股票，重新计算其所有历史因子
            # 因为财务因子依赖历史数据（如TTM需要4个季度）
            stock_financial_data = financial_data[
                financial_data.index.get_level_values('StockCodes').isin(updated_stocks)
            ]
            
            # 计算更新股票的因子
            kwargs['financial_data'] = stock_financial_data
            updated_factors = self.calculator.calculate_factors(
                factor_names=factor_names,
                **kwargs
            )
            
            # 合并更新
            result_factors = {}
            for factor_name in factor_names:
                if factor_name in updated_factors:
                    # 更新相应股票的因子值
                    if factor_name in existing_factors:
                        # 保留未更新股票的因子值
                        result_factors[factor_name] = existing_factors[factor_name].copy()
                        # 更新有新数据的股票
                        for stock in updated_stocks:
                            if stock in updated_factors[factor_name].index.get_level_values('StockCodes'):
                                mask = result_factors[factor_name].index.get_level_values('StockCodes') == stock
                                result_factors[factor_name][mask] = updated_factors[factor_name][
                                    updated_factors[factor_name].index.get_level_values('StockCodes') == stock
                                ]
                    else:
                        result_factors[factor_name] = updated_factors[factor_name]
            
            # 更新追踪信息
            self._update_financial_tracker(financial_data)
            
        else:
            # 全量更新
            logger.info("Processing full update")
            kwargs['financial_data'] = financial_data
            result_factors = self.calculator.calculate_factors(
                factor_names=factor_names,
                save_path=self.factor_path,
                **kwargs
            )
            
            # 更新追踪信息
            self._update_financial_tracker(financial_data)
        
        # 保存因子
        for factor_name, factor_data in result_factors.items():
            factor_file = self.factor_path / f"{factor_name}.pkl"
            factor_data.to_pickle(factor_file)
            logger.info(f"Saved {factor_name} to {factor_file}")
        
        return result_factors
    
    def update_technical_factors(self,
                               factor_names: List[str],
                               mode: str = 'incremental',
                               **kwargs) -> Dict[str, pd.Series]:
        """
        更新技术因子
        
        Parameters:
        -----------
        factor_names : 要更新的因子列表
        mode : 'full' 或 'incremental'
        **kwargs : 传递给calculate_factors的参数
        
        Returns:
        --------
        更新后的因子字典
        """
        logger.info(f"Updating technical factors in {mode} mode")
        
        # 加载价格数据
        price_data = kwargs.get('price_data')
        if price_data is None:
            price_file = self.data_path / 'Price.pkl'
            if price_file.exists():
                price_data = pd.read_pickle(price_file)
            else:
                logger.error("Price data not found")
                return {}
        
        # 检查更新
        has_updates, new_price_data = self.check_price_updates(price_data)
        
        if mode == 'incremental' and not has_updates:
            logger.info("No updates found for incremental update")
            return self._load_existing_factors(factor_names)
        
        if mode == 'incremental' and has_updates:
            # 增量更新技术因子
            logger.info("Processing incremental updates")
            
            # 技术因子通常需要历史数据（如20日动量）
            # 所以需要获取足够的历史数据
            max_lookback = self._get_max_lookback(factor_names)
            
            # 获取新数据的最早日期
            new_start_date = new_price_data.index.get_level_values('TradingDates').min()
            
            # 获取需要的历史数据
            lookback_start = new_start_date - pd.Timedelta(days=max_lookback * 2)
            needed_data = price_data[
                price_data.index.get_level_values('TradingDates') >= lookback_start
            ]
            
            # 计算因子
            kwargs['price_data'] = needed_data
            new_factors = self.calculator.calculate_factors(
                factor_names=factor_names,
                **kwargs
            )
            
            # 只保留新日期的因子值
            result_factors = {}
            for factor_name, factor_data in new_factors.items():
                new_dates_only = factor_data[
                    factor_data.index.get_level_values('TradingDates') >= new_start_date
                ]
                
                # 加载并合并现有因子
                existing_factor = self._load_single_factor(factor_name)
                if existing_factor is not None:
                    # 合并新旧数据
                    result_factors[factor_name] = pd.concat([
                        existing_factor[
                            existing_factor.index.get_level_values('TradingDates') < new_start_date
                        ],
                        new_dates_only
                    ]).sort_index()
                else:
                    result_factors[factor_name] = new_dates_only
            
            # 更新追踪信息
            self._update_price_tracker(price_data)
            
        else:
            # 全量更新
            logger.info("Processing full update")
            kwargs['price_data'] = price_data
            result_factors = self.calculator.calculate_factors(
                factor_names=factor_names,
                save_path=self.factor_path,
                **kwargs
            )
            
            # 更新追踪信息
            self._update_price_tracker(price_data)
        
        # 保存因子
        for factor_name, factor_data in result_factors.items():
            factor_file = self.factor_path / f"{factor_name}.pkl"
            factor_data.to_pickle(factor_file)
            logger.info(f"Saved {factor_name} to {factor_file}")
        
        return result_factors
    
    def update_all_factors(self, mode: str = 'incremental', **kwargs):
        """
        更新所有因子
        
        Parameters:
        -----------
        mode : 'full' 或 'incremental'
        **kwargs : 数据参数
        """
        logger.info(f"Starting {mode} update for all factors")
        
        # 获取所有注册的因子
        all_factors = self.calculator.list_factors()
        
        # 按类别分组
        fundamental_factors = []
        technical_factors = []
        risk_factors = []
        
        for factor_name, info in all_factors.items():
            category = info['category']
            if category in ['fundamental', 'profitability', 'liquidity']:
                fundamental_factors.append(factor_name)
            elif category == 'technical':
                technical_factors.append(factor_name)
            elif category == 'risk':
                risk_factors.append(factor_name)
        
        # 更新基本面因子
        if fundamental_factors:
            self.update_fundamental_factors(fundamental_factors, mode, **kwargs)
        
        # 更新技术因子
        if technical_factors:
            self.update_technical_factors(technical_factors, mode, **kwargs)
        
        # 更新风险因子（通常需要价格和基准数据）
        if risk_factors:
            self.update_technical_factors(risk_factors, mode, **kwargs)
        
        logger.info("All factors update completed")
    
    def _load_existing_factors(self, factor_names: List[str]) -> Dict[str, pd.Series]:
        """加载现有因子"""
        factors = {}
        for factor_name in factor_names:
            factor = self._load_single_factor(factor_name)
            if factor is not None:
                factors[factor_name] = factor
        return factors
    
    def _load_single_factor(self, factor_name: str) -> Optional[pd.Series]:
        """加载单个因子"""
        factor_file = self.factor_path / f"{factor_name}.pkl"
        if factor_file.exists():
            try:
                return pd.read_pickle(factor_file)
            except Exception as e:
                logger.error(f"Failed to load factor {factor_name}: {e}")
        return None
    
    def _update_financial_tracker(self, financial_data: pd.DataFrame):
        """更新财务数据追踪信息"""
        # 找出发布日期列
        release_date_col = None
        for col in ['ReleasedDates', 'reportday', 'releaseddate']:
            if col in financial_data.columns:
                release_date_col = col
                break
        
        if release_date_col:
            last_release_date = financial_data[release_date_col].max()
            self.tracker.update_status('financial', {
                'last_release_date': str(last_release_date),
                'last_update_time': str(datetime.now()),
                'total_records': len(financial_data)
            })
    
    def _update_price_tracker(self, price_data: pd.DataFrame):
        """更新价格数据追踪信息"""
        last_trading_date = price_data.index.get_level_values('TradingDates').max()
        self.tracker.update_status('price', {
            'last_trading_date': str(last_trading_date),
            'last_update_time': str(datetime.now()),
            'total_records': len(price_data)
        })
    
    def _get_max_lookback(self, factor_names: List[str]) -> int:
        """获取因子需要的最大回望期"""
        max_lookback = 20  # 默认值
        
        # 根据因子名称推断回望期
        for factor_name in factor_names:
            if 'Momentum' in factor_name:
                # 从名称中提取数字
                try:
                    lookback = int(factor_name.split('_')[-1])
                    max_lookback = max(max_lookback, lookback)
                except:
                    pass
            elif 'Beta' in factor_name:
                max_lookback = max(max_lookback, 252)
            elif 'Volatility' in factor_name:
                try:
                    lookback = int(factor_name.split('_')[-1])
                    max_lookback = max(max_lookback, lookback)
                except:
                    max_lookback = max(max_lookback, 20)
        
        return max_lookback