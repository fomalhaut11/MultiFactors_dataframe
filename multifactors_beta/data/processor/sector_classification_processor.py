#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
股票分类计算处理器

基于板块进出表计算不同时间截面上股票的行业和概念分类信息
通过分析板块调整记录，重构每个时点的股票归属关系

Author: MultiFactors Team
Date: 2025-08-28
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta

from config import get_config
from data.fetcher.data_fetcher import get_stock_data

logger = logging.getLogger(__name__)


class SectorClassificationProcessor:
    """
    股票分类计算处理器
    
    基于板块进出表数据，计算任意时间截面上股票的行业和概念分类信息。
    通过分析"纳入"和"剔除"记录，重构历史时点的股票归属关系。
    """
    
    def __init__(self):
        self.data_root = Path(get_config('main.paths.data_root'))
        # 分类数据直接存储在数据根目录
        self.classification_data_path = self.data_root
        self.sector_changes_file = self.classification_data_path / 'SectorChanges_data.pkl'
        
        # 缓存数据
        self._sector_changes_data = None
        self._last_load_time = None
        
        logger.info("初始化股票分类计算处理器")
    
    def _load_sector_changes_data(self, force_reload: bool = False) -> pd.DataFrame:
        """
        加载板块进出数据
        
        Args:
            force_reload: 是否强制重新加载
            
        Returns:
            板块进出数据DataFrame
        """
        # 检查是否需要重新加载
        current_time = datetime.now()
        if (not force_reload and self._sector_changes_data is not None and 
            self._last_load_time is not None and 
            (current_time - self._last_load_time).total_seconds() < 3600):  # 1小时内不重复加载
            return self._sector_changes_data
        
        if not self.sector_changes_file.exists():
            logger.error(f"板块进出数据文件不存在: {self.sector_changes_file}")
            raise FileNotFoundError(f"板块进出数据文件不存在: {self.sector_changes_file}")
        
        try:
            logger.info(f"加载板块进出数据: {self.sector_changes_file}")
            data = pd.read_pickle(self.sector_changes_file)
            
            # 数据预处理
            data = data.copy()
            data['sel_day'] = pd.to_datetime(data['sel_day'].astype(str), format='%Y%m%d')
            data = data.sort_values(['sel_day', 'concept_code', 'code'])
            
            # 缓存数据
            self._sector_changes_data = data
            self._last_load_time = current_time
            
            logger.info(f"板块进出数据加载完成: {data.shape}, 日期范围: {data['sel_day'].min()} - {data['sel_day'].max()}")
            return data
            
        except Exception as e:
            logger.error(f"加载板块进出数据失败: {e}")
            raise
    
    def calculate_sector_classification_at_date(self, target_date: Union[str, int, datetime], 
                                              concept_types: Optional[List[str]] = None) -> pd.DataFrame:
        """
        计算指定日期的股票分类信息
        
        Args:
            target_date: 目标日期，支持字符串('2024-01-01')、整数(20240101)或datetime对象
            concept_types: 要计算的概念类型列表，None表示计算所有类型
            
        Returns:
            包含股票分类信息的DataFrame，列包括：
            - code: 股票代码
            - concept_code: 概念代码
            - concept_name: 概念名称
            - 指数类型: 指数类型
            - entry_date: 最近一次纳入日期
            - is_active: 是否在该日期有效
        """
        # 统一日期格式
        if isinstance(target_date, str):
            target_date = pd.to_datetime(target_date)
        elif isinstance(target_date, int):
            target_date = pd.to_datetime(str(target_date), format='%Y%m%d')
        
        logger.info(f"计算 {target_date.date()} 的股票分类信息")
        
        # 加载数据
        sector_data = self._load_sector_changes_data()
        
        # 筛选指定日期之前的所有变更记录
        historical_data = sector_data[sector_data['sel_day'] <= target_date].copy()
        
        if historical_data.empty:
            logger.warning(f"指定日期 {target_date.date()} 之前没有板块变更记录")
            return pd.DataFrame(columns=['code', 'concept_code', 'concept_name', 
                                       '指数类型', 'entry_date', 'is_active'])
        
        # 筛选概念类型
        if concept_types:
            historical_data = historical_data[historical_data['指数类型'].isin(concept_types)]
        
        # 按股票和概念分组，获取每个股票在每个概念中的最新状态
        result_records = []
        
        for (code, concept_code), group in historical_data.groupby(['code', 'concept_code']):
            # 按时间排序，获取最新记录
            latest_record = group.sort_values('sel_day').iloc[-1]
            
            # 判断该股票在目标日期是否在该概念中
            is_active = latest_record['纳入_剔除'] == '纳入'
            
            if is_active:
                result_records.append({
                    'code': code,
                    'concept_code': concept_code,
                    'concept_name': latest_record['concept_name'],
                    '指数类型': latest_record['指数类型'],
                    'entry_date': latest_record['sel_day'],
                    'is_active': True
                })
        
        if not result_records:
            logger.warning(f"指定日期 {target_date.date()} 没有有效的股票分类信息")
            return pd.DataFrame(columns=['code', 'concept_code', 'concept_name', 
                                       '指数类型', 'entry_date', 'is_active'])
        
        result_df = pd.DataFrame(result_records)
        result_df = result_df.sort_values(['指数类型', 'concept_code', 'code'])
        
        logger.info(f"计算完成: {target_date.date()} 共有 {len(result_df)} 条有效分类记录")
        return result_df
    
    def calculate_sector_classification_series(self, start_date: Union[str, int, datetime],
                                             end_date: Union[str, int, datetime],
                                             frequency: str = 'M') -> Dict[str, pd.DataFrame]:
        """
        计算时间序列的股票分类信息
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            frequency: 计算频率，'D'(日)、'W'(周)、'M'(月)、'Q'(季)、'Y'(年)
            
        Returns:
            时间序列分类数据字典 {date_str: classification_df}
        """
        # 统一日期格式
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        elif isinstance(start_date, int):
            start_date = pd.to_datetime(str(start_date), format='%Y%m%d')
            
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        elif isinstance(end_date, int):
            end_date = pd.to_datetime(str(end_date), format='%Y%m%d')
        
        logger.info(f"计算时间序列股票分类: {start_date.date()} - {end_date.date()}, 频率: {frequency}")
        
        # 生成日期序列
        date_range = pd.date_range(start=start_date, end=end_date, freq=frequency)
        
        # 计算每个时点的分类
        series_result = {}
        for target_date in date_range:
            date_str = target_date.strftime('%Y-%m-%d')
            try:
                classification_df = self.calculate_sector_classification_at_date(target_date)
                series_result[date_str] = classification_df
                logger.debug(f"完成 {date_str} 的分类计算")
            except Exception as e:
                logger.error(f"计算 {date_str} 分类失败: {e}")
                series_result[date_str] = pd.DataFrame()
        
        logger.info(f"时间序列分类计算完成，共 {len(series_result)} 个时点")
        return series_result
    
    def get_stock_sectors_at_date(self, stock_codes: List[str], 
                                target_date: Union[str, int, datetime],
                                sector_type: Optional[str] = None) -> pd.DataFrame:
        """
        获取指定股票在指定日期的分类信息
        
        Args:
            stock_codes: 股票代码列表
            target_date: 目标日期
            sector_type: 分类类型筛选，如'申万行业板块'等
            
        Returns:
            股票分类信息DataFrame
        """
        # 计算指定日期的完整分类信息
        full_classification = self.calculate_sector_classification_at_date(target_date)
        
        if full_classification.empty:
            return pd.DataFrame()
        
        # 筛选指定股票
        stock_classification = full_classification[
            full_classification['code'].isin(stock_codes)
        ]
        
        # 筛选分类类型
        if sector_type:
            stock_classification = stock_classification[
                stock_classification['指数类型'] == sector_type
            ]
        
        return stock_classification
    
    def get_sector_stocks_at_date(self, concept_code: str,
                                target_date: Union[str, int, datetime]) -> List[str]:
        """
        获取指定概念在指定日期包含的股票列表
        
        Args:
            concept_code: 概念代码
            target_date: 目标日期
            
        Returns:
            股票代码列表
        """
        # 计算指定日期的完整分类信息
        full_classification = self.calculate_sector_classification_at_date(target_date)
        
        if full_classification.empty:
            return []
        
        # 筛选指定概念
        sector_stocks = full_classification[
            full_classification['concept_code'] == concept_code
        ]['code'].tolist()
        
        return sector_stocks
    
    def get_classification_stats(self, target_date: Union[str, int, datetime]) -> Dict[str, int]:
        """
        获取指定日期的分类统计信息
        
        Args:
            target_date: 目标日期
            
        Returns:
            统计信息字典
        """
        classification_df = self.calculate_sector_classification_at_date(target_date)
        
        if classification_df.empty:
            return {}
        
        stats = {
            'total_records': len(classification_df),
            'unique_stocks': classification_df['code'].nunique(),
            'unique_concepts': classification_df['concept_code'].nunique(),
            'concept_types': classification_df['指数类型'].value_counts().to_dict()
        }
        
        return stats
    
    def export_classification_data(self, target_date: Union[str, int, datetime],
                                 output_path: Optional[Path] = None,
                                 format: str = 'pkl') -> Path:
        """
        导出指定日期的分类数据
        
        Args:
            target_date: 目标日期
            output_path: 输出路径，None时使用默认路径
            format: 输出格式，'pkl'、'csv'、'excel'
            
        Returns:
            输出文件路径
        """
        # 统一日期格式
        if isinstance(target_date, str):
            date_str = target_date.replace('-', '')
        elif isinstance(target_date, int):
            date_str = str(target_date)
        else:
            date_str = target_date.strftime('%Y%m%d')
        
        # 计算分类数据
        classification_df = self.calculate_sector_classification_at_date(target_date)
        
        if output_path is None:
            output_path = self.classification_data_path / f'StockClassification_{date_str}.{format}'
        
        # 确保输出目录存在
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 导出数据
        if format == 'pkl':
            classification_df.to_pickle(output_path)
        elif format == 'csv':
            classification_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        elif format == 'excel':
            classification_df.to_excel(output_path, index=False)
        else:
            raise ValueError(f"不支持的输出格式: {format}")
        
        logger.info(f"分类数据已导出: {output_path}")
        return output_path


def create_sector_classification_processor() -> SectorClassificationProcessor:
    """创建股票分类计算处理器实例"""
    return SectorClassificationProcessor()


if __name__ == "__main__":
    # 测试代码
    processor = SectorClassificationProcessor()
    
    try:
        # 测试计算最近日期的分类信息
        recent_classification = processor.calculate_sector_classification_at_date('2024-08-01')
        print(f"测试成功: 2024-08-01 的分类记录数: {len(recent_classification)}")
        
        if not recent_classification.empty:
            print("示例数据:")
            print(recent_classification.head())
            
            # 获取统计信息
            stats = processor.get_classification_stats('2024-08-01')
            print(f"\n统计信息: {stats}")
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()