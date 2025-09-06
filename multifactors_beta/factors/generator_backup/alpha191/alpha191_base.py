"""
Alpha191 因子基类和数据适配器
"""
import pandas as pd
import numpy as np
from typing import Optional, Dict, Union, Any
import logging
from abc import abstractmethod

from ...base.factor_base import FactorBase
from core.utils import OutlierHandler, Normalizer

logger = logging.getLogger(__name__)


class Alpha191DataAdapter:
    """
    Alpha191 数据格式适配器
    
    将项目的 MultiIndex 格式转换为 Alpha191 需要的格式
    """
    
    @staticmethod
    def convert_to_alpha191_format(price_data: pd.Series) -> Dict[str, pd.DataFrame]:
        """
        将 MultiIndex 格式的价格数据转换为 Alpha191 所需格式
        
        Parameters
        ----------
        price_data : pd.Series
            MultiIndex格式 [TradingDates, StockCodes] 的价格数据
            
        Returns
        -------
        dict
            包含各字段的 DataFrame，索引为时间，列为股票代码
        """
        if not isinstance(price_data.index, pd.MultiIndex):
            raise ValueError("输入数据必须是 MultiIndex 格式")
        
        # 将长格式转换为宽格式 (时间 x 股票)
        wide_data = price_data.unstack(level='StockCodes')
        
        return wide_data
    
    @staticmethod
    def prepare_alpha191_data(price_data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        准备 Alpha191 计算所需的所有数据字段
        
        Parameters
        ----------
        price_data : pd.DataFrame
            包含价格数据的 DataFrame，MultiIndex [TradingDates, StockCodes]
            需要包含字段: 'o', 'h', 'l', 'c', 'v', 'vwap'
            
        Returns
        -------
        dict
            Alpha191 计算器所需的数据字典
        """
        required_fields = ['o', 'h', 'l', 'c', 'v', 'vwap']
        
        # 检查必需字段
        missing_fields = [field for field in required_fields if field not in price_data.columns]
        if missing_fields:
            raise ValueError(f"缺少必需字段: {missing_fields}")
        
        # 转换每个字段
        alpha191_data = {}
        for field in required_fields:
            field_data = price_data[field]
            alpha191_data[field] = Alpha191DataAdapter.convert_to_alpha191_format(field_data)
        
        # 计算额外字段
        alpha191_data['amount'] = alpha191_data['v'] * alpha191_data['vwap']
        alpha191_data['prev_close'] = alpha191_data['c'].shift(1)
        
        return alpha191_data
    
    @staticmethod
    def convert_from_alpha191_format(data: pd.Series, 
                                   original_index: pd.MultiIndex) -> pd.Series:
        """
        将 Alpha191 输出转换回项目的 MultiIndex 格式
        
        Parameters
        ----------
        data : pd.Series
            Alpha191 输出数据
        original_index : pd.MultiIndex
            原始数据的索引
            
        Returns
        -------
        pd.Series
            MultiIndex 格式的结果
        """
        if isinstance(data, pd.DataFrame) and data.shape[1] == 1:
            data = data.iloc[:, 0]
        
        # 如果是单一值，广播到所有股票
        if isinstance(data, (int, float)):
            result = pd.Series(data, index=original_index)
            return result
        
        # 堆叠为长格式
        if hasattr(data, 'index') and hasattr(data, 'name'):
            # 单个时间点的截面数据
            result_data = []
            result_index = []
            
            for date in original_index.get_level_values('TradingDates').unique():
                for stock in data.index:
                    if (date, stock) in original_index:
                        result_data.append(data.loc[stock])
                        result_index.append((date, stock))
            
            return pd.Series(result_data, index=pd.MultiIndex.from_tuples(
                result_index, names=['TradingDates', 'StockCodes']))
        
        # DataFrame 格式 (时间 x 股票)
        result = data.stack()
        result.index.names = ['TradingDates', 'StockCodes']
        
        # 只保留原始索引中存在的数据点
        result = result.reindex(original_index)
        
        return result


class Alpha191Base(FactorBase):
    """
    Alpha191 因子基类
    
    继承自 FactorBase，提供标准化的因子计算接口
    """
    
    def __init__(self, alpha_num: int, name: str = None, description: str = ""):
        """
        Parameters
        ----------
        alpha_num : int
            Alpha 因子编号 (1-191)
        name : str, optional
            因子名称，默认为 Alpha{num:03d}
        description : str
            因子描述
        """
        if name is None:
            name = f"Alpha{alpha_num:03d}"
        
        super().__init__(name=name, category='alpha191')
        
        self.alpha_num = alpha_num
        self.description = description or f"Alpha191 第 {alpha_num} 号因子"
        
        # Alpha191 特有配置
        self.requires_benchmark = False
        self.min_periods = 250  # 大多数因子需要的最小时间序列长度
    
    def calculate(self, price_data: pd.DataFrame, 
                  benchmark_data: Optional[pd.DataFrame] = None,
                  **kwargs) -> pd.Series:
        """
        标准化的因子计算接口
        
        Parameters
        ----------
        price_data : pd.DataFrame
            价格数据，MultiIndex [TradingDates, StockCodes]
        benchmark_data : pd.DataFrame, optional
            基准数据，某些因子需要
        **kwargs
            其他参数
            
        Returns
        -------
        pd.Series
            计算结果，MultiIndex 格式
        """
        try:
            # 数据验证
            self._validate_input_data(price_data, benchmark_data)
            
            # 准备 Alpha191 格式数据
            alpha191_data = Alpha191DataAdapter.prepare_alpha191_data(price_data)
            
            # 调用具体的 Alpha 计算方法
            result = self._calculate_alpha(alpha191_data, benchmark_data, **kwargs)
            
            # 转换回项目标准格式
            if result is not None and not result.empty:
                result = Alpha191DataAdapter.convert_from_alpha191_format(
                    result, price_data.index
                )
                
                # 应用标准预处理
                result = self.preprocess(result, **kwargs)
                
            return result
            
        except Exception as e:
            logger.error(f"Alpha{self.alpha_num:03d} 计算失败: {e}")
            # 返回空结果但保持索引结构
            return pd.Series(dtype=float, index=price_data.index, name=self.name)
    
    @abstractmethod
    def _calculate_alpha(self, data: Dict[str, pd.DataFrame],
                        benchmark_data: Optional[pd.DataFrame] = None,
                        **kwargs) -> pd.Series:
        """
        具体的 Alpha 因子计算逻辑
        
        Parameters
        ----------
        data : dict
            Alpha191 格式的数据字典
        benchmark_data : pd.DataFrame, optional
            基准数据
        **kwargs
            其他参数
            
        Returns
        -------
        pd.Series
            计算结果
        """
        pass
    
    def _validate_input_data(self, price_data: pd.DataFrame,
                           benchmark_data: Optional[pd.DataFrame] = None):
        """验证输入数据"""
        if not isinstance(price_data, pd.DataFrame):
            raise TypeError("price_data 必须是 DataFrame")
        
        if not isinstance(price_data.index, pd.MultiIndex):
            raise ValueError("price_data 必须使用 MultiIndex")
        
        if self.requires_benchmark and benchmark_data is None:
            raise ValueError(f"Alpha{self.alpha_num:03d} 需要基准数据")
        
        # 检查数据长度
        date_count = len(price_data.index.get_level_values('TradingDates').unique())
        if date_count < self.min_periods:
            logger.warning(f"Alpha{self.alpha_num:03d} 建议至少 {self.min_periods} 个交易日，"
                         f"当前只有 {date_count} 个")
    
    def get_factor_info(self) -> Dict[str, Any]:
        """获取因子信息"""
        return {
            'alpha_num': self.alpha_num,
            'name': self.name,
            'category': self.category,
            'description': self.description,
            'requires_benchmark': self.requires_benchmark,
            'min_periods': self.min_periods
        }