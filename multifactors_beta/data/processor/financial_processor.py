"""
财报数据处理器

处理财务报表相关数据，包括发布日期和时间特征
保持与原始实现完全一致的算法
"""
import os
import numpy as np
import pandas as pd
import h5py
from typing import Optional, Dict
from pathlib import Path

from .base_processor import BaseDataProcessor
from core.config_manager import get_path


class FinancialDataProcessor(BaseDataProcessor):
    """财报数据处理器"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化财报数据处理器
        
        Args:
            config_path: 配置文件路径
        """
        super().__init__(config_path)
        self.financial_file_path = Path(get_path('data_root')) / "financial_v2.h5"
        
    def validate_input(self, **kwargs) -> bool:
        """验证输入文件是否存在"""
        if not self.financial_file_path.exists():
            self.logger.error(f"财务数据文件不存在: {self.financial_file_path}")
            return False
        return True
        
    def process(self, **kwargs):
        """财报处理器不使用通用process方法"""
        raise NotImplementedError("请使用具体的处理方法")
        
    def get_released_dates_from_h5(self, datapath: Optional[str] = None) -> pd.DataFrame:
        """
        从H5文件中获取实际发布日期
        完全保持原始算法不变
        
        Args:
            datapath: 数据路径，默认从配置获取
            
        Returns:
            包含发布日期信息的数据框
        """
        if datapath is None:
            datapath = get_path('data_root')
            
        self.logger.info("从H5文件中获取财务数据发布日期")
        
        financialfile = os.path.join(datapath, "financial_v2.h5")
        financialdata = h5py.File(financialfile, "r")
        
        # 获取数据 - 与原始实现完全一致
        stockcodes_in_financialdata = financialdata["uni_code"][()]
        fzb = financialdata.get("fzb")[()]
        public_dates = fzb[:, :, 0]
        report_due_dates = fzb[:, :, 1]
        qnum = fzb[:, :, -3]
        ynum = fzb[:, :, -2]
        
        # 构建股票代码数组 - 与原始实现完全一致
        StockCodes = stockcodes_in_financialdata.repeat(np.shape(report_due_dates[0]))
        StockCodes = [code.decode("utf-8") for code in StockCodes]
        
        # 构建MultiIndex - 与原始实现完全一致
        index1 = pd.MultiIndex.from_arrays(
            [StockCodes, pd.to_datetime(report_due_dates.reshape(-1), format="%Y%m%d")]
        )
        
        # 创建发布日期DataFrame - 与原始实现完全一致
        realesed_dates_df = pd.DataFrame(
            pd.to_datetime(public_dates.reshape(-1), format="%Y%m%d"),
            index=index1,
            columns=["ReleasedDates"],
        )
        
        # 创建季度和年度DataFrame - 与原始实现完全一致
        qnum_df = pd.DataFrame(qnum.reshape(-1), index=index1, columns=["Quater"])
        ynum_df = pd.DataFrame(ynum.reshape(-1), index=index1, columns=["Year"])
        
        # 合并数据 - 与原始实现完全一致
        realesed_dates_df = pd.merge(
            realesed_dates_df, qnum_df, left_index=True, right_index=True
        )
        realesed_dates_df = pd.merge(
            realesed_dates_df, ynum_df, left_index=True, right_index=True
        )
        
        realesed_dates_df.index.names = ["StockCodes", "ReportDates"]
        
        # 关闭H5文件
        financialdata.close()
        
        return realesed_dates_df
        
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