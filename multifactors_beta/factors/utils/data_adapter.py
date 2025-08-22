#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据格式适配器
用于在原始数据格式和因子计算所需格式之间进行转换
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class DataAdapter:
    """数据格式适配器"""
    
    @staticmethod
    def convert_to_multiindex(data: pd.DataFrame,
                            date_col: str,
                            stock_col: str,
                            date_name: str = 'Dates',
                            stock_name: str = 'StockCodes') -> pd.DataFrame:
        """
        将普通DataFrame转换为MultiIndex格式
        
        Parameters:
        -----------
        data : DataFrame
            原始数据
        date_col : str
            日期列名
        stock_col : str
            股票代码列名
        date_name : str
            日期索引名称
        stock_name : str
            股票代码索引名称
            
        Returns:
        --------
        MultiIndex DataFrame
        """
        if date_col not in data.columns or stock_col not in data.columns:
            raise ValueError(f"列 {date_col} 或 {stock_col} 不存在")
            
        # 复制数据避免修改原始数据
        result = data.copy()
        
        # 转换日期格式
        result[date_col] = pd.to_datetime(result[date_col])
        
        # 设置MultiIndex
        result = result.set_index([date_col, stock_col])
        result.index.names = [date_name, stock_name]
        
        return result
    
    @staticmethod
    def _get_report_period_date(year: int, quarter: int) -> pd.Timestamp:
        """
        根据年份和季度获取财报截止日期
        """
        quarter_end_dates = {
            1: f"{year}-03-31",
            2: f"{year}-06-30", 
            3: f"{year}-09-30",
            4: f"{year}-12-31"
        }
        return pd.Timestamp(quarter_end_dates[int(quarter)])
    
    @staticmethod
    def prepare_financial_data(lrb: pd.DataFrame,
                             xjlb: pd.DataFrame,
                             fzb: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        准备财务数据
        
        Parameters:
        -----------
        lrb : DataFrame
            利润表
        xjlb : DataFrame
            现金流量表
        fzb : DataFrame
            资产负债表
            
        Returns:
        --------
        (financial_data, release_dates) : 财务数据和发布日期
        """
        logger.info("准备财务数据...")
        
        financial_data_list = []
        release_dates_list = []
        
        for df, name in [(lrb, 'lrb'), (xjlb, 'xjlb'), (fzb, 'fzb')]:
            # 检查必要列
            required_cols = ['code', 'd_year', 'd_quarter', 'reportday']
            if not all(col in df.columns for col in required_cols):
                logger.warning(f"{name} 缺少必要的列")
                continue
            
            # 创建财报期间
            df['ReportPeriod'] = df.apply(
                lambda row: DataAdapter._get_report_period_date(
                    int(row['d_year']), int(row['d_quarter'])
                ), 
                axis=1
            )
            
            # 保存发布日期（reportday是公布日期）
            df['ReleasedDates'] = pd.to_datetime(df['reportday'])
            
            # 设置索引
            df_indexed = df.set_index(['ReportPeriod', 'code'])
            df_indexed.index.names = ['ReportDates', 'StockCodes']
            
            financial_data_list.append(df_indexed)
            
            # 提取发布日期信息
            release_info = df[['ReportPeriod', 'code', 'ReleasedDates']].copy()
            release_info = release_info.drop_duplicates(['ReportPeriod', 'code'])
            release_info = release_info.set_index(['ReportPeriod', 'code'])
            release_info.index.names = ['ReportDates', 'StockCodes']
            release_dates_list.append(release_info)
        
        # 合并财务数据
        if financial_data_list:
            financial_data = financial_data_list[0]
            for df in financial_data_list[1:]:
                # 只合并新列，避免重复
                common_cols = set(financial_data.columns) & set(df.columns)
                new_cols = [col for col in df.columns if col not in common_cols]
                if new_cols:
                    financial_data = financial_data.join(df[new_cols], how='outer')
            
            # 合并发布日期
            release_dates = pd.concat(release_dates_list, ignore_index=False)
            release_dates = release_dates.groupby(level=[0, 1]).first()  # 去重
            
            logger.info(f"财务数据准备完成，形状: {financial_data.shape}")
            logger.info(f"财报期间范围: {financial_data.index.get_level_values('ReportDates').min()} 到 {financial_data.index.get_level_values('ReportDates').max()}")
            
            return financial_data, release_dates
        else:
            return pd.DataFrame(), pd.DataFrame()
    
    @staticmethod
    def prepare_price_data(price: pd.DataFrame) -> pd.DataFrame:
        """
        准备价格数据
        
        Parameters:
        -----------
        price : DataFrame
            原始价格数据
            
        Returns:
        --------
        MultiIndex价格数据
        """
        logger.info("准备价格数据...")
        
        # 检查是否已经是MultiIndex
        if isinstance(price.index, pd.MultiIndex):
            logger.info("价格数据已经是MultiIndex格式")
            return price
        
        # 转换为MultiIndex
        if 'tradingday' in price.columns and 'code' in price.columns:
            price_mi = DataAdapter.convert_to_multiindex(
                price, 'tradingday', 'code', 'TradingDates', 'StockCodes'
            )
        else:
            raise ValueError("价格数据必须包含 tradingday 和 code 列")
        
        logger.info(f"价格数据准备完成，形状: {price_mi.shape}")
        
        return price_mi
    
    @staticmethod
    def prepare_market_cap(market_cap: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        """
        准备市值数据
        
        Parameters:
        -----------
        market_cap : DataFrame or Series
            原始市值数据
            
        Returns:
        --------
        MultiIndex市值Series
        """
        logger.info("准备市值数据...")
        
        # 如果已经是Series且有MultiIndex
        if isinstance(market_cap, pd.Series) and isinstance(market_cap.index, pd.MultiIndex):
            logger.info("市值数据已经是正确格式")
            return market_cap
        
        # 如果是DataFrame
        if isinstance(market_cap, pd.DataFrame):
            # 查找市值列
            cap_cols = [col for col in market_cap.columns if 'cap' in col.lower() or 'marketvalue' in col.lower()]
            if cap_cols:
                cap_col = cap_cols[0]
            else:
                # 使用第一个数值列
                numeric_cols = market_cap.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    cap_col = numeric_cols[0]
                else:
                    raise ValueError("无法找到市值列")
            
            # 转换为MultiIndex
            if 'tradingday' in market_cap.columns and 'code' in market_cap.columns:
                market_cap_mi = DataAdapter.convert_to_multiindex(
                    market_cap, 'tradingday', 'code', 'TradingDates', 'StockCodes'
                )
                market_cap_series = market_cap_mi[cap_col]
            else:
                raise ValueError("市值数据必须包含 tradingday 和 code 列")
        else:
            # 如果是Series但没有MultiIndex
            market_cap_series = market_cap
        
        logger.info(f"市值数据准备完成，大小: {len(market_cap_series)}")
        
        return market_cap_series
    
    @staticmethod
    def extract_trading_dates(price_data: pd.DataFrame) -> pd.DatetimeIndex:
        """
        从价格数据中提取交易日期
        
        Parameters:
        -----------
        price_data : DataFrame
            价格数据
            
        Returns:
        --------
        交易日期序列
        """
        if isinstance(price_data.index, pd.MultiIndex):
            return price_data.index.get_level_values('TradingDates').unique().sort_values()
        elif 'tradingday' in price_data.columns:
            return pd.to_datetime(price_data['tradingday']).unique()
        else:
            raise ValueError("无法从价格数据中提取交易日期")
    
    @staticmethod
    def load_and_prepare_data(data_path: Union[str, Path]) -> Dict:
        """
        加载并准备所有必要的数据
        
        Parameters:
        -----------
        data_path : str or Path
            数据目录路径
            
        Returns:
        --------
        包含所有准备好的数据的字典
        """
        data_path = Path(data_path)
        logger.info(f"从 {data_path} 加载数据...")
        
        result = {}
        
        try:
            # 加载财务数据
            lrb = pd.read_pickle(data_path / 'lrb.pkl')
            xjlb = pd.read_pickle(data_path / 'xjlb.pkl')
            fzb = pd.read_pickle(data_path / 'fzb.pkl')
            
            # 准备财务数据
            financial_data, release_dates = DataAdapter.prepare_financial_data(lrb, xjlb, fzb)
            result['financial_data'] = financial_data
            result['release_dates'] = release_dates
            
            # 加载并准备价格数据
            price = pd.read_pickle(data_path / 'Price.pkl')
            price_data = DataAdapter.prepare_price_data(price)
            result['price_data'] = price_data
            
            # 提取交易日期
            trading_dates = DataAdapter.extract_trading_dates(price_data)
            result['trading_dates'] = trading_dates
            
            # 加载并准备市值数据
            if (data_path / 'MarketCap.pkl').exists():
                market_cap = pd.read_pickle(data_path / 'MarketCap.pkl')
                market_cap_series = DataAdapter.prepare_market_cap(market_cap)
                result['market_cap'] = market_cap_series
            
            # 加载基准数据（如果存在）
            if (data_path / 'benchmark.pkl').exists():
                benchmark = pd.read_pickle(data_path / 'benchmark.pkl')
                result['benchmark_data'] = benchmark
            
            logger.info("数据加载和准备完成")
            
            return result
            
        except Exception as e:
            logger.error(f"数据加载失败: {e}")
            raise


class FactorDataAdapter:
    """因子数据适配器 - 用于因子计算结果的格式转换"""
    
    @staticmethod
    def convert_to_daily_factor(factor_data: pd.Series,
                              release_dates: pd.DataFrame,
                              trading_dates: pd.DatetimeIndex) -> pd.Series:
        """
        将季度因子转换为日频因子
        
        Parameters:
        -----------
        factor_data : Series
            MultiIndex (ReportDates, StockCodes) 的因子数据
        release_dates : DataFrame
            包含 ReleasedDates 列的发布日期数据
        trading_dates : DatetimeIndex
            交易日期序列
            
        Returns:
        --------
        MultiIndex (TradingDates, StockCodes) 的日频因子
        """
        from ..base.time_series_processor import TimeSeriesProcessor
        
        # 合并因子数据和发布日期
        factor_with_release = pd.DataFrame({'factor': factor_data})
        factor_with_release = factor_with_release.join(release_dates, how='left')
        
        # 使用时间序列处理器扩展到日频
        daily_factor = TimeSeriesProcessor.expand_to_daily(
            factor_with_release,
            release_date_col='ReleasedDates',
            trading_dates=trading_dates
        )
        
        # 只返回因子列，排除 LatestReportDate
        if 'factor' in daily_factor.columns:
            return daily_factor['factor']
        else:
            # 如果是Series，直接返回
            return daily_factor
    
    @staticmethod
    def align_factor_data(factors: Dict[str, pd.Series]) -> pd.DataFrame:
        """
        对齐多个因子数据
        
        Parameters:
        -----------
        factors : dict
            因子名称到因子数据的映射
            
        Returns:
        --------
        对齐后的因子DataFrame
        """
        if not factors:
            return pd.DataFrame()
        
        # 找到共同的索引
        common_index = None
        for name, factor in factors.items():
            if common_index is None:
                common_index = factor.index
            else:
                common_index = common_index.intersection(factor.index)
        
        # 对齐所有因子
        aligned_factors = {}
        for name, factor in factors.items():
            aligned_factors[name] = factor.reindex(common_index)
        
        return pd.DataFrame(aligned_factors)


if __name__ == "__main__":
    # 测试代码
    logger.info("测试数据适配器...")
    
    # 模拟数据路径
    data_path = Path(r"E:\Documents\PythonProject\StockProject\StockData")
    
    try:
        # 加载并准备数据
        prepared_data = DataAdapter.load_and_prepare_data(data_path)
        
        print("\n准备的数据:")
        for key, value in prepared_data.items():
            if isinstance(value, pd.DataFrame):
                print(f"{key}: DataFrame {value.shape}")
            elif isinstance(value, pd.Series):
                print(f"{key}: Series {len(value)}")
            elif isinstance(value, pd.DatetimeIndex):
                print(f"{key}: DatetimeIndex {len(value)} dates")
                
        print("\n[OK] 数据适配器测试成功!")
        
    except Exception as e:
        print(f"\n[FAIL] 数据适配器测试失败: {e}")
        import traceback
        traceback.print_exc()