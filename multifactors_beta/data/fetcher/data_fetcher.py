#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据获取抽象类
提供统一的数据获取接口，支持不同数据源和数据类型的获取
"""

import os
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, date
import logging

from config import get_config
from core.database import execute_stock_data_query, execute_query_to_dataframe, get_db_table_config

logger = logging.getLogger(__name__)


class BaseDataFetcher(ABC):
    """数据获取基类"""
    
    def __init__(self, name: str, config_section: str = None):
        self.name = name
        self.config_section = config_section or name.lower()
        self.cache_enabled = get_config('main.system.cache_enabled') or True
        self.data_root = get_config('main.paths.data_root')
        
        logger.info(f"初始化数据获取器: {self.name}")
    
    @abstractmethod
    def fetch_data(self, **kwargs) -> pd.DataFrame:
        """获取数据的抽象方法"""
        pass
    
    def get_cache_path(self, data_type: str) -> str:
        """获取缓存文件路径"""
        cache_dir = get_config('main.paths.cache', os.path.join(self.data_root, 'cache'))
        os.makedirs(cache_dir, exist_ok=True)
        return os.path.join(cache_dir, f"{self.name}_{data_type}.pkl")
    
    def load_from_cache(self, data_type: str, max_age_hours: int = 24) -> Optional[pd.DataFrame]:
        """从缓存加载数据"""
        if not self.cache_enabled:
            return None
            
        cache_path = self.get_cache_path(data_type)
        
        if not os.path.exists(cache_path):
            return None
        
        # 检查缓存文件年龄
        file_age = (datetime.now().timestamp() - os.path.getmtime(cache_path)) / 3600
        if file_age > max_age_hours:
            logger.info(f"缓存文件过期: {cache_path}")
            return None
        
        try:
            data = pd.read_pickle(cache_path)
            logger.info(f"从缓存加载数据: {cache_path}")
            return data
        except Exception as e:
            logger.error(f"加载缓存失败: {e}")
            return None
    
    def save_to_cache(self, data: pd.DataFrame, data_type: str):
        """保存数据到缓存"""
        if not self.cache_enabled or data.empty:
            return
            
        try:
            cache_path = self.get_cache_path(data_type)
            data.to_pickle(cache_path)
            logger.info(f"数据已缓存: {cache_path}")
        except Exception as e:
            logger.error(f"保存缓存失败: {e}")


class StockDataFetcher(BaseDataFetcher):
    """股票数据获取器"""
    
    def __init__(self):
        super().__init__("StockData", "stock_data")
        self.supported_data_types = [
            'price', 'volume', 'financial', 'macro', 'index', 
            'tradable', 'stop_price', 'concept', 'sector_changes'
        ]
        # 获取数据库表名配置
        self.db_config = get_db_table_config()
    
    def fetch_data(self, data_type: str, **kwargs) -> pd.DataFrame:
        """
        获取股票数据
        
        Args:
            data_type: 数据类型
            **kwargs: 其他参数
            
        Returns:
            DataFrame: 股票数据
        """
        if data_type not in self.supported_data_types:
            raise ValueError(f"不支持的数据类型: {data_type}")
        
        logger.info(f"获取股票数据: {data_type}")
        
        # 尝试从缓存加载
        cached_data = self.load_from_cache(data_type, kwargs.get('cache_hours', 24))
        if cached_data is not None:
            return cached_data
        
        # 根据数据类型调用相应的获取方法
        method_name = f"_fetch_{data_type}_data"
        if hasattr(self, method_name):
            data = getattr(self, method_name)(**kwargs)
            self.save_to_cache(data, data_type)
            return data
        else:
            raise NotImplementedError(f"获取方法未实现: {method_name}")
    
    def _fetch_price_data(self, begin_date: int = 20200101, end_date: int = 0, 
                         incremental: bool = True, **kwargs) -> pd.DataFrame:
        """获取价格数据"""
        price_table = self.db_config.price_table
        
        if end_date == 0:
            sql = (
                "SELECT [code],[tradingday],[o],[h],[l],[c],[v],[amt],[adjfactor],"
                "[total_shares],[free_float_shares],[exchange_id] "
                "FROM {} WHERE tradingday >= %d"
            ).format(price_table) % begin_date
        else:
            sql = (
                "SELECT [code],[tradingday],[o],[h],[l],[c],[v],[amt],[adjfactor],"
                "[total_shares],[free_float_shares],[exchange_id] "
                "FROM {} WHERE tradingday >= %d AND tradingday <= %d"
            ).format(price_table) % (begin_date, end_date)
        
        return execute_stock_data_query(sql, db_name='database')
    
    def _fetch_financial_data(self, sheet_type: str = 'all', **kwargs) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """获取财务数据"""
        tables = {
            'fzb': self.db_config.fzb_table,
            'xjlb': self.db_config.xjlb_table,
            'lrb': self.db_config.lrb_table
        }
        
        if sheet_type == 'all':
            results = {}
            for table_name, table_path in tables.items():
                sql = f"SELECT * FROM {table_path}"
                results[table_name] = execute_stock_data_query(
                    sql, db_name='database', 
                    date_columns=['reportday', 'tradingday']
                )
            return results
        elif sheet_type in tables:
            sql = f"SELECT * FROM {tables[sheet_type]}"
            return execute_stock_data_query(
                sql, db_name='database', 
                date_columns=['reportday', 'tradingday']
            )
        else:
            raise ValueError(f"不支持的财务报表类型: {sheet_type}")
    
    def _fetch_macro_data(self, **kwargs) -> Dict[str, pd.DataFrame]:
        """获取宏观数据"""
        # 行业利润数据
        industry_sql = (
            "SELECT DISTINCT [tradingday],[行业],[利润总额累计值] "
            "FROM {}".format(self.db_config.macro_industry_table)
        )
        industry_data = execute_query_to_dataframe(
            industry_sql, 
            ["tradingday", "行业", "利润总额累计值"],
            db_name='database',
            dtype_mapping={'tradingday': 'datetime'}
        )
        
        # 宏观利率数据
        macro_sql = f"""
        SELECT [tradingday],[美国十年国债收益率],[美国单月同比CPI],[中国十年国债收益率],
               [中国单月同比CPI],[官方外汇储备],[季度GDP累计值],[工业企业利润累计值],
               [社会融资规模_亿],[PPI],[M2],[M2_同比],[GDP_当季值亿],
               [既期10点美元汇率],[写入日期时间]
        FROM {self.db_config.us_treasury_table}
        """
        macro_columns = [
            "tradingday", "美国十年国债收益率", "美国单月同比CPI", "中国十年国债收益率",
            "中国单月同比CPI", "官方外汇储备", "季度GDP累计值", "工业企业利润累计值",
            "社会融资规模_亿", "PPI", "M2", "M2_同比", "GDP_当季值亿",
            "既期10点美元汇率", "写入日期时间"
        ]
        macro_data = execute_query_to_dataframe(
            macro_sql, macro_columns, db_name='database',
            dtype_mapping={'tradingday': 'datetime'}
        )
        
        return {
            'industry_profit': industry_data,
            'macro_rates': macro_data
        }
    
    def _fetch_tradable_data(self, **kwargs) -> pd.DataFrame:
        """获取可交易股票数据"""
        sql = (
            "SELECT [ipo_date],[code],[exchange_id],[last_trade_day],"
            "[tradingday],[trade_status] FROM {}".format(self.db_config.all_stocks_table)
        )
        return execute_stock_data_query(sql, db_name='database')
    
    def _fetch_stop_price_data(self, **kwargs) -> pd.DataFrame:
        """获取涨跌停价格数据"""
        sql = (
            "SELECT DISTINCT [code],[tradingday],[high_limit],[low_limit] "
            "FROM {}".format(self.db_config.stop_price_table)
        )
        columns = ["code", "tradingday", "high_limit", "low_limit"]
        return execute_query_to_dataframe(sql, columns, db_name='database')
    
    def _fetch_index_data(self, index_codes: List[str] = None, 
                         begin_date: int = None, end_date: int = None, **kwargs) -> pd.DataFrame:
        """获取指数数据"""
        sql_base = f"""
        SELECT [bankuai],[tradingday],[exchange_id],[index_name],[code],
               [o],[h],[l],[c],[v],[amt],[writing_day] 
        FROM {self.db_config.wind_index_table} WHERE 1=1
        """
        
        conditions = []
        if begin_date:
            conditions.append(f"tradingday >= {begin_date}")
        if end_date:
            conditions.append(f"tradingday <= {end_date}")
        if index_codes:
            codes_str = "','".join(index_codes)
            conditions.append(f"code IN ('{codes_str}')")
        
        if conditions:
            sql = sql_base + " AND " + " AND ".join(conditions)
        else:
            sql = sql_base
        
        sql += " ORDER BY tradingday"
        
        return execute_stock_data_query(sql, db_name='database')
    
    
    def _fetch_concept_data(self, concept_code: str = None, 
                          trading_date: int = None, **kwargs) -> pd.DataFrame:
        """
        获取概念板块数据
        
        Args:
            concept_code: 概念代码，为None时获取所有概念
            trading_date: 交易日期，格式YYYYMMDD，为None时获取最新数据
            **kwargs: 其他参数
            
        Returns:
            包含概念板块信息的DataFrame
        """
        logger.info(f"获取概念板块数据: concept_code={concept_code}")
        
        if concept_code is None and trading_date is None:
            # 获取所有概念代码和名称
            sql = f"""
            SELECT DISTINCT [concept_code], [concept_name] 
            FROM {self.db_config.concept_table}
            WHERE 指数类型 != '申万行业板块'
            ORDER BY concept_code
            """
            columns = ["concept_code", "concept_name"]
        elif concept_code is None and trading_date is not None:
            # 获取指定日期的所有概念数据
            sql = f"""
            SELECT [tradingday],[sel_day],[指数类型],[concept_code],[concept_name],[code]
            FROM {self.db_config.concept_table}
            WHERE tradingday = {trading_date} AND 指数类型 != '申万行业板块'
            ORDER BY concept_code, code
            """
            columns = ["tradingday", "sel_day", "指数类型", "concept_code", "concept_name", "code"]
        elif concept_code is not None and trading_date is None:
            # 获取指定概念的最新数据
            sql = f"""
            SELECT [tradingday],[sel_day],[指数类型],[concept_code],[concept_name],[code]
            FROM {self.db_config.concept_table}
            WHERE concept_code = '{concept_code}' 
            AND tradingday = (
                SELECT MAX(tradingday) 
                FROM {self.db_config.concept_table} 
                WHERE concept_code = '{concept_code}'
            )
            ORDER BY code
            """
            columns = ["tradingday", "sel_day", "指数类型", "concept_code", "concept_name", "code"]
        else:
            # 获取指定概念和日期的数据
            sql = f"""
            SELECT [tradingday],[sel_day],[指数类型],[concept_code],[concept_name],[code]
            FROM {self.db_config.concept_table}
            WHERE concept_code = '{concept_code}' AND tradingday = {trading_date}
            ORDER BY code
            """
            columns = ["tradingday", "sel_day", "指数类型", "concept_code", "concept_name", "code"]
        
        return execute_query_to_dataframe(sql, columns, db_name='database')
    
    def _fetch_sector_changes_data(self, begin_date: int = None, end_date: int = None, 
                                 concept_code: str = None, **kwargs) -> pd.DataFrame:
        """
        获取板块进出调整数据
        
        Args:
            begin_date: 开始日期，格式YYYYMMDD
            end_date: 结束日期，格式YYYYMMDD
            concept_code: 概念代码，为None时获取所有概念
            **kwargs: 其他参数
            
        Returns:
            包含板块调整信息的DataFrame，按sel_day排序
        """
        logger.info(f"获取板块进出数据: begin_date={begin_date}, end_date={end_date}, concept_code={concept_code}")
        
        # 构建基础SQL
        base_sql = f"""
        SELECT [sel_day], [code], [exchange_id], [code_cn], [纳入_剔除], 
               [concept_code], [concept_exid], [concept_name], [industry], 
               [指数类型], [写入时间]
        FROM {self.db_config.sector_changes_table}
        """
        
        # 构建WHERE条件
        conditions = []
        
        if begin_date is not None:
            conditions.append(f"sel_day >= {begin_date}")
            
        if end_date is not None:
            conditions.append(f"sel_day <= {end_date}")
            
        if concept_code is not None:
            conditions.append(f"concept_code = '{concept_code}'")
        
        # 拼接WHERE条件
        if conditions:
            where_clause = " WHERE " + " AND ".join(conditions)
        else:
            where_clause = ""
        
        # 添加排序
        order_clause = " ORDER BY sel_day, concept_code, code"
        
        sql = base_sql + where_clause + order_clause
        
        columns = ["sel_day", "code", "exchange_id", "code_cn", "纳入_剔除", 
                  "concept_code", "concept_exid", "concept_name", "industry", 
                  "指数类型", "写入时间"]
        
        return execute_query_to_dataframe(sql, columns, db_name='database')


class MarketDataFetcher(BaseDataFetcher):
    """市场数据获取器"""
    
    def __init__(self):
        super().__init__("MarketData", "market_data")
        # 获取数据库表名配置
        self.db_config = get_db_table_config()
    
    def fetch_data(self, data_type: str, **kwargs) -> pd.DataFrame:
        """获取市场数据"""
        logger.info(f"获取市场数据: {data_type}")
        
        if data_type == 'trading_dates':
            return self._fetch_trading_dates(**kwargs)
        elif data_type == 'st_stocks':
            return self._fetch_st_stocks(**kwargs)
        else:
            raise ValueError(f"不支持的市场数据类型: {data_type}")
    
    def _fetch_trading_dates(self, **kwargs) -> pd.DataFrame:
        """获取交易日期"""
        sql = "SELECT [tradingday] FROM tradingday ORDER BY tradingday"
        return execute_query_to_dataframe(sql, ['tradingday'], db_name='database')
    
    def _fetch_st_stocks(self, **kwargs) -> pd.DataFrame:
        """获取ST股票信息"""
        sql = (
            "SELECT [tradingday],[code],[exchange_id],[sec_name] "
            "FROM {} ORDER BY tradingday".format(self.db_config.st_stocks_table)
        )
        columns = ['tradingday', 'code', 'exchange_id', 'sec_name']
        return execute_query_to_dataframe(sql, columns, db_name='database')


class DataFetcherManager:
    """数据获取管理器"""
    
    def __init__(self):
        self.fetchers = {}
        self._register_default_fetchers()
    
    def _register_default_fetchers(self):
        """注册默认的数据获取器"""
        self.register_fetcher('stock', StockDataFetcher())
        self.register_fetcher('market', MarketDataFetcher())
    
    def register_fetcher(self, name: str, fetcher: BaseDataFetcher):
        """注册数据获取器"""
        self.fetchers[name] = fetcher
        logger.info(f"注册数据获取器: {name}")
    
    def get_fetcher(self, name: str) -> BaseDataFetcher:
        """获取数据获取器"""
        if name not in self.fetchers:
            raise ValueError(f"未找到数据获取器: {name}")
        return self.fetchers[name]
    
    def fetch_data(self, fetcher_name: str, data_type: str, **kwargs) -> pd.DataFrame:
        """获取数据的便捷方法"""
        fetcher = self.get_fetcher(fetcher_name)
        return fetcher.fetch_data(data_type, **kwargs)
    
    def list_fetchers(self) -> List[str]:
        """列出所有可用的数据获取器"""
        return list(self.fetchers.keys())


# 全局数据获取管理器实例
data_fetcher_manager = DataFetcherManager()


# 便捷函数
def get_stock_data(data_type: str, **kwargs) -> pd.DataFrame:
    """获取股票数据的便捷函数"""
    return data_fetcher_manager.fetch_data('stock', data_type, **kwargs)


def get_market_data(data_type: str, **kwargs) -> pd.DataFrame:
    """获取市场数据的便捷函数"""
    return data_fetcher_manager.fetch_data('market', data_type, **kwargs)


if __name__ == "__main__":
    # 测试数据获取器
    logger.info("测试数据获取器...")
    
    try:
        # 测试获取交易日期
        trading_dates = get_market_data('trading_dates')
        print(f"[OK] 获取交易日期成功，共 {len(trading_dates)} 个交易日")
        
        # 测试获取价格数据（少量数据）
        price_data = get_stock_data('price', begin_date=20250101, end_date=20250131)
        print(f"[OK] 获取价格数据成功，形状: {price_data.shape}")
        
        # 列出所有获取器
        fetchers = data_fetcher_manager.list_fetchers()
        print(f"可用的数据获取器: {fetchers}")
        
    except Exception as e:
        print(f"[FAIL] 测试失败: {e}")