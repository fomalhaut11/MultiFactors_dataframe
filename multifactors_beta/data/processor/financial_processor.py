"""
财报数据处理器

处理财务报表相关数据，包括发布日期和时间特征
使用从数据库获取的财务数据（fzb.pkl、lrb.pkl、xjlb.pkl）
"""
import os
import numpy as np
import pandas as pd
from typing import Optional, Dict
from pathlib import Path

from .base_processor import BaseDataProcessor
from config import get_config


class FinancialDataProcessor(BaseDataProcessor):
    """财报数据处理器"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化财报数据处理器
        
        Args:
            config_path: 配置文件路径
        """
        super().__init__(config_path)
        self.data_root = Path(get_config('main.paths.data_root'))
        self.fzb_file = self.data_root / "fzb.pkl"
        self.lrb_file = self.data_root / "lrb.pkl"
        self.xjlb_file = self.data_root / "xjlb.pkl"
        
    def validate_input(self, **kwargs) -> bool:
        """验证输入文件是否存在"""
        missing_files = []
        for name, file_path in [("资产负债表", self.fzb_file), ("利润表", self.lrb_file), ("现金流量表", self.xjlb_file)]:
            if not file_path.exists():
                missing_files.append(f"{name}: {file_path}")
                
        if missing_files:
            self.logger.error(f"财务数据文件不存在: {', '.join(missing_files)}")
            return False
        return True
        
    def process(self, **kwargs):
        """财报处理器不使用通用process方法"""
        raise NotImplementedError("请使用具体的处理方法")
        
    def get_released_dates_from_pkl(self, datapath: Optional[str] = None) -> pd.DataFrame:
        """
        从PKL文件中获取实际发布日期
        基于数据库获取的财务数据（fzb.pkl）
        
        Args:
            datapath: 数据路径，默认从配置获取
            
        Returns:
            包含发布日期信息的数据框
        """
        if datapath is None:
            datapath = get_config('main.paths.data_root')
            
        self.logger.info("从PKL文件中获取财务数据发布日期")
        
        # 读取资产负债表数据（包含发布日期信息）
        fzb_file = os.path.join(datapath, "fzb.pkl")
        if not os.path.exists(fzb_file):
            raise FileNotFoundError(f"资产负债表文件不存在: {fzb_file}")
            
        fzb_data = pd.read_pickle(fzb_file)
        self.logger.info(f"读取资产负债表数据: {fzb_data.shape}")
        
        # 从数据库财务数据中提取发布日期信息
        # 数据库财务数据包含：reportday（报告期）、tradingday（发布日期）、d_year（年份）、d_quarter（季度）等字段
        if 'tradingday' not in fzb_data.columns or 'reportday' not in fzb_data.columns:
            self.logger.warning("财务数据缺少必要的日期字段，使用默认处理")
            # 创建一个最小化的发布日期DataFrame
            return self._create_minimal_released_dates_df(fzb_data)
        
        # 提取需要的字段
        released_dates_data = fzb_data[['code', 'reportday', 'tradingday', 'd_year', 'd_quarter']].copy()
        
        # 确保日期格式正确
        if not pd.api.types.is_datetime64_any_dtype(released_dates_data['reportday']):
            released_dates_data['reportday'] = pd.to_datetime(released_dates_data['reportday'])
        if not pd.api.types.is_datetime64_any_dtype(released_dates_data['tradingday']):
            released_dates_data['tradingday'] = pd.to_datetime(released_dates_data['tradingday'])
        
        # 去重并排序
        released_dates_data = released_dates_data.drop_duplicates(
            subset=['code', 'reportday', 'd_quarter'], keep='last'
        ).sort_values(['code', 'reportday'])
        
        # 构建MultiIndex
        released_dates_data = released_dates_data.set_index(['code', 'reportday'])
        
        # 创建最终的DataFrame
        realesed_dates_df = pd.DataFrame({
            'ReleasedDates': released_dates_data['tradingday'],
            'Quater': released_dates_data['d_quarter'],
            'Year': released_dates_data['d_year']
        })
        
        realesed_dates_df.index.names = ["StockCodes", "ReportDates"]
        
        self.logger.info(f"发布日期数据处理完成: {realesed_dates_df.shape}")
        return realesed_dates_df
    
    def _create_minimal_released_dates_df(self, fzb_data: pd.DataFrame) -> pd.DataFrame:
        """创建最小化的发布日期DataFrame"""
        self.logger.info("创建最小化的发布日期DataFrame")
        
        # 使用reportday作为发布日期的估计
        if 'reportday' in fzb_data.columns and 'code' in fzb_data.columns:
            minimal_data = fzb_data[['code', 'reportday']].copy()
            if 'd_quarter' in fzb_data.columns:
                minimal_data['d_quarter'] = fzb_data['d_quarter']
            else:
                minimal_data['d_quarter'] = 1  # 默认值
            
            if 'd_year' in fzb_data.columns:
                minimal_data['d_year'] = fzb_data['d_year']
            else:
                minimal_data['d_year'] = 2020  # 默认值
            
            if not pd.api.types.is_datetime64_any_dtype(minimal_data['reportday']):
                minimal_data['reportday'] = pd.to_datetime(minimal_data['reportday'])
            
            minimal_data = minimal_data.drop_duplicates().set_index(['code', 'reportday'])
            
            return pd.DataFrame({
                'ReleasedDates': minimal_data.index.get_level_values(1),  # 使用reportday作为发布日期
                'Quater': minimal_data['d_quarter'],
                'Year': minimal_data['d_year']
            }, index=minimal_data.index)
        else:
            # 创建一个空的DataFrame
            empty_index = pd.MultiIndex.from_tuples([], names=["StockCodes", "ReportDates"])
            return pd.DataFrame({
                'ReleasedDates': pd.Series([], dtype='datetime64[ns]'),
                'Quater': pd.Series([], dtype='int64'),
                'Year': pd.Series([], dtype='int64')
            }, index=empty_index)
    
    def get_released_dates_from_h5(self, datapath: Optional[str] = None) -> pd.DataFrame:
        """
        兼容性方法：从PKL文件获取发布日期数据
        
        Args:
            datapath: 数据路径
            
        Returns:
            发布日期数据框
        """
        self.logger.warning("get_released_dates_from_h5已弃用，使用get_released_dates_from_pkl替代")
        return self.get_released_dates_from_pkl(datapath)
        
    def calculate_released_dates_count(self, 
                                     released_dates_df: pd.DataFrame, 
                                     trading_dates: pd.DataFrame) -> pd.DataFrame:
        """
        计算每个交易日距离最近的财报发布日的时间差
        完全保持原始算法不变
        
        Args:
            released_dates_df: 财报发布日期数据
            trading_dates: 交易日期数据
            
        Returns:
            时间差数据框
        """
        # 内部计算函数 - 与原始实现完全一致
        def row_calc(row, TradingDates):
            indices = np.searchsorted(row, TradingDates.values.flatten(), side="right") - 1
            time_diffs = (
                TradingDates.values.flatten() - row.iloc[indices].values
            ) / np.timedelta64(1, "D")
            time_diffs = pd.Series(time_diffs, index=TradingDates.iloc[:, 0])
            return time_diffs
            
        # 按季度计算 - 与原始实现完全一致
        for i in range(4):
            self.logger.info(f"处理第{i+1}季度数据")
            quater1data = released_dates_df.loc[released_dates_df["Quater"] == 1 + i]
            rd = quater1data["ReleasedDates"].to_frame().unstack()
            time_diffs = rd.apply(row_calc, axis=1, args=(trading_dates,))
            df = time_diffs.reset_index()
            df = df.melt(id_vars="StockCodes", var_name="TradingDates")
            df.set_index(["TradingDates", "StockCodes"], inplace=True)
            columnnames = "Quater" + str(i + 1)
            df.columns = [columnnames]
            
            if i == 0:
                DateCount_df = df
            else:
                DateCount_df = pd.merge(
                    DateCount_df, df, left_index=True, right_index=True
                )
                
        return DateCount_df