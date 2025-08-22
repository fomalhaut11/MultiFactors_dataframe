"""
价格数据处理器

处理股票价格数据的清洗、转换和格式化
保持与原始实现完全一致的算法
"""
import os
import numpy as np
import pandas as pd
import pickle
import gc
from typing import Dict, Tuple, Optional, List
from pathlib import Path

from .base_processor import BaseDataProcessor
from core.config_manager import get_path


class PriceDataProcessor(BaseDataProcessor):
    """价格数据处理器"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化价格数据处理器
        
        Args:
            config_path: 配置文件路径
        """
        super().__init__(config_path)
        self.price_file_path = Path(get_path('data_root')) / "Price.pkl"
        self.tradable_file_path = Path(get_path('data_root')) / "TradableDF.pkl"
        
    def validate_input(self, **kwargs) -> bool:
        """验证输入文件是否存在"""
        if not self.price_file_path.exists():
            self.logger.error(f"价格文件不存在: {self.price_file_path}")
            return False
            
        if not self.tradable_file_path.exists():
            self.logger.error(f"可交易状态文件不存在: {self.tradable_file_path}")
            return False
            
        return True
        
    def _stock_data_df_to_matrix(self, stock_data_df: pd.DataFrame) -> Dict:
        """
        将股票数据DataFrame转换为3D矩阵格式
        完全保持原始算法不变
        
        Args:
            stock_data_df: 股票数据DataFrame
            
        Returns:
            包含3D矩阵的字典
        """
        # 与原始实现完全一致
        levels = [stock_data_df["tradingday"].unique(), stock_data_df["code"].unique()]
        stock_data_df0 = stock_data_df.set_index(["tradingday", "code"])
        stock_data_df1 = stock_data_df0.reindex(
            pd.MultiIndex.from_product(levels), fill_value=np.nan
        )
        pricematrix = stock_data_df1.values.reshape(
            (len(levels[0]), len(levels[1]), np.shape(stock_data_df1.values)[1])
        )
        stock_data_3d_matrix = {
            "TradingDates": levels[0],
            "StockCodes": levels[1],
            "pricematrix": pricematrix,
            "datacolumns": stock_data_df1.columns,
        }
        return stock_data_3d_matrix
        
    def process(self, save_to_file: bool = True) -> Tuple[pd.DataFrame, Dict]:
        """
        处理价格数据
        
        Args:
            save_to_file: 是否保存到文件
            
        Returns:
            (处理后的价格DataFrame, 3D矩阵字典)
        """
        self.logger.info("开始处理价格数据...")
        
        # 加载数据 - 与原始实现保持一致
        price_df = pd.read_pickle(self.price_file_path)
        tradable_df = pd.read_pickle(self.tradable_file_path)
        
        # 剔除北交所 - 与原始实现完全一致
        price_df = price_df[~(price_df["exchange_id"] == "BJ")]
        tradable_df = tradable_df[~(tradable_df["exchange_id"] == "BJ")]
        
        # 合并数据 - 与原始实现完全一致
        price_df = price_df.join(
            tradable_df, how="left", lsuffix="_left", rsuffix="_right"
        )
        
        # 剔除退市股票 - 与原始实现完全一致
        price_df = price_df[~(price_df["trade_status"] == "退市")]
        
        del tradable_df
        
        # 准备3D矩阵数据 - 与原始实现完全一致
        stock_data_df = price_df.copy()
        stock_data_df.index = price_df.index.set_names(["tradingday", "code"])
        stock_data_df = stock_data_df.reset_index()
        stock_data_df["tradingday"] = stock_data_df["tradingday"].dt.strftime("%Y%m%d")
        
        # 选择列 - 与原始实现完全一致
        columns = [
            "tradingday",
            "code",
            "o",
            "h",
            "l",
            "c",
            "v",
            "amt",
            "adjfactor",
            "total_shares",
            "free_float_shares",
            "MC",
            "FMC",
            "turnoverrate",
            "vwap",
            "freeturnoverrate"
        ]
        stock_data_df = stock_data_df[columns]
        
        # 转换为3D矩阵
        stock_3d = self._stock_data_df_to_matrix(stock_data_df.copy(deep=True))
        
        del stock_data_df
        
        # 保存结果
        if save_to_file:
            save_path = Path(get_path('data_root')) / "Stock3d.pkl"
            with open(save_path, "wb") as f:
                pickle.dump(stock_3d, f)
            self.logger.info(f"3D矩阵已保存至: {save_path}")
            
            # 记录保存信息
            self._record_processing(
                'save_stock3d',
                {'save_path': str(save_path)},
                {
                    'status': 'success',
                    'shape': {
                        'dates': len(stock_3d['TradingDates']),
                        'stocks': len(stock_3d['StockCodes']),
                        'features': len(stock_3d['datacolumns'])
                    }
                }
            )
            
        return price_df, stock_3d
        
    def get_date_series(self, price_df: pd.DataFrame, 
                       series_type: str = "daily") -> pd.DatetimeIndex:
        """
        获取日期序列
        完全保持原始算法不变
        
        Args:
            price_df: 价格数据DataFrame
            series_type: 序列类型 ("daily", "weekly", "monthly")
            
        Returns:
            日期序列
        """
        date_series = price_df.index.get_level_values(0).unique()
        
        if series_type == "daily":
            return date_series
        elif series_type == "weekly":
            weekly_mask = date_series.to_series().dt.to_period(
                "W"
            ) != date_series.to_series().shift(1).dt.to_period("W")
            return date_series[weekly_mask]
        elif series_type == "monthly":
            monthly_mask = date_series.to_series().dt.to_period(
                "M"
            ) != date_series.to_series().shift(1).dt.to_period("M")
            return date_series[monthly_mask]
        else:
            raise ValueError(f"不支持的序列类型: {series_type}")
            
    def filter_stocks(self, price_df: pd.DataFrame, 
                     exclude_exchanges: List[str] = None,
                     exclude_status: List[str] = None) -> pd.DataFrame:
        """
        过滤股票
        
        Args:
            price_df: 价格数据
            exclude_exchanges: 要排除的交易所列表
            exclude_status: 要排除的状态列表
            
        Returns:
            过滤后的数据
        """
        filtered_df = price_df.copy()
        
        if exclude_exchanges:
            for exchange in exclude_exchanges:
                mask = filtered_df.get("exchange_id") != exchange
                if mask is not None:
                    filtered_df = filtered_df[mask]
                    
        if exclude_status:
            for status in exclude_status:
                mask = filtered_df.get("trade_status") != status
                if mask is not None:
                    filtered_df = filtered_df[mask]
                    
        return filtered_df