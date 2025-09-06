""" # -*- coding: utf-8 -*-
从sql或者LAN中获取数据下载到本地
@author: ZhangXi
"""

import os
import numpy as np
import pandas as pd
import pymssql
from datetime import date
from datetime import datetime
import time
import h5py
import sys
import logging

# 导入配置管理器
from config import get_config, config_manager

# 便捷的数据库配置获取函数
def get_database_config():
    """获取数据库配置"""
    return get_config('main.database')

int_today = int(date.today().strftime("%Y%m%d"))
str_today = str(date.today().strftime("%Y%m%d"))

# 设置日志
logger = logging.getLogger(__name__)


def _get_last_trading_date_from_local():
    """
    从本地Price.pkl文件获取最新的交易日期
    
    Returns:
        int: 最新交易日期的整数格式，如果没有本地数据则返回None
    """
    try:
        data_root = get_config('main.paths.data_root')
        price_file = os.path.join(data_root, "Price.pkl")
        
        if not os.path.exists(price_file):
            logger.info("本地Price.pkl文件不存在，将进行全量更新")
            return None
            
        logger.info("正在读取本地Price.pkl数据以确定最新日期...")
        existing_data = pd.read_pickle(price_file)
        
        if existing_data.empty:
            logger.info("本地数据为空，将进行全量更新")
            return None
            
        # 获取最新的交易日期
        latest_date = existing_data.index.get_level_values('TradingDates').max()
        latest_date_int = int(latest_date.strftime('%Y%m%d'))
        
        logger.info(f"本地数据最新日期：{latest_date} ({latest_date_int})")
        
        # 返回最新日期的下一天作为增量更新起点
        next_date = latest_date + pd.Timedelta(days=1)
        next_date_int = int(next_date.strftime('%Y%m%d'))
        
        logger.info(f"增量更新起始日期：{next_date_int}")
        return next_date_int
        
    except Exception as e:
        logger.warning(f"读取本地数据失败，将进行全量更新：{e}")
        return None


def _merge_with_local_data(new_data, data_root):
    """
    将新数据与本地数据合并
    
    Args:
        new_data: DataFrame，新获取的数据
        data_root: 数据根目录
    
    Returns:
        DataFrame: 合并后的数据
    """
    try:
        price_file = os.path.join(data_root, "Price.pkl")
        
        if not os.path.exists(price_file):
            logger.info("本地文件不存在，直接保存新数据")
            return new_data
            
        logger.info("正在合并本地数据和新数据...")
        existing_data = pd.read_pickle(price_file)
        
        if existing_data.empty:
            logger.info("本地数据为空，直接使用新数据")
            return new_data
            
        if new_data.empty:
            logger.info("新数据为空，保持原有数据")
            return existing_data
            
        # 合并数据，去重并按日期排序
        combined_data = pd.concat([existing_data, new_data])
        combined_data = combined_data[~combined_data.index.duplicated(keep='last')]
        combined_data = combined_data.sort_index()
        
        logger.info(f"数据合并完成：原有 {len(existing_data)} 条，新增 {len(new_data)} 条，合并后 {len(combined_data)} 条")
        return combined_data
        
    except Exception as e:
        logger.error(f"数据合并失败：{e}")
        logger.info("使用新数据覆盖")
        return new_data


# 尝试导入板块成份API，如果失败则记录警告
try:
    project_path = get_config('main.paths.project_parent') or r"E:\Documents\PythonProject\StockProject"
    sys.path.append(project_path)
    import lgc_板块成份api as ComData
except ImportError as e:
    logger.warning(f"无法导入lgc_板块成份api: {e}")
    ComData = None


def GetMacroIndexFromSql_save(
        datasavepath=None,
        **db_kwargs
        ):
    """
    从数据库中获取宏观经济数据
    
    Args:
        datasavepath: 数据保存路径，默认从配置获取
        **db_kwargs: 数据库连接参数，会覆盖配置文件
    
    Returns:
        tuple: (行业利润数据, 宏观利率数据)
    """
    # 获取数据保存路径
    if datasavepath is None:
        datasavepath = get_config('main.paths.data_root')
    
    # 获取数据库配置
    db_config = get_database_config()
    db_config.update(db_kwargs)  # 允许参数覆盖

    if not os.path.exists(datasavepath):
        os.makedirs(datasavepath)

    logger.info("从数据库中获取宏观经济数据")
    
    try:
        db = pymssql.connect(
            host=db_config['host'], 
            user=db_config['user'], 
            password=db_config['password'], 
            database=db_config['database']
        )
    except Exception as e:
        logger.error(f"数据库连接失败: {e}")
        raise
    cursor = db.cursor()    
    sql_str = "SELECT distinct [tradingday],[行业],[利润总额累计值] FROM [stock_data].[dbo].[lgc_行业经济数据]"
    cursor.execute(sql_str)
    data1 = cursor.fetchall()
    StockDataDF = pd.DataFrame(data1)
    StockDataDF.columns = ["tradingday", "行业", "利润总额累计值"]
    StockDataDF.set_index(["tradingday", "行业"], inplace=True)
    StockDataDF.sort_index(level=0, inplace=True)
    sql_str = """SELECT [tradingday]
      ,[美国十年国债收益率]
      ,[美国单月同比CPI]
      ,[中国十年国债收益率]
      ,[中国单月同比CPI]
      ,[官方外汇储备]
      ,[季度GDP累计值]
      ,[工业企业利润累计值]
      ,[社会融资规模_亿]
      ,[PPI]
      ,[M2]
      ,[M2_同比]
      ,[GDP_当季值亿]
      ,[既期10点美元汇率]
      ,[写入日期时间]
    FROM [stock_data].[dbo].[美国国债收益率及CPI]"""
    cursor.execute(sql_str)
    data2 = cursor.fetchall()
    cursor.close()
    db.close()
    StockDataDF2 = pd.DataFrame(data2)
    StockDataDF2.columns = [
        "tradingday",
        "美国十年国债收益率",
        "美国单月同比CPI",
        "中国十年国债收益率",
        "中国单月同比CPI",
        "官方外汇储备",
        "季度GDP累计值",
        "工业企业利润累计值",
        "社会融资规模_亿",
        "PPI",
        "M2",
        "M2_同比",
        "GDP_当季值亿",
        "既期10点美元汇率",
        "写入日期时间"
    ]
    StockDataDF2.set_index(["tradingday"], inplace=True)
    StockDataDF2.sort_index(inplace=True)

    # 保存数据
    industry_profit_file = os.path.join(datasavepath, "行业利润总额累计值.pkl")
    macro_rates_file = os.path.join(datasavepath, "中美宏观利率.pkl")
    
    pd.to_pickle(StockDataDF, industry_profit_file)
    pd.to_pickle(StockDataDF2, macro_rates_file)
    
    logger.info(f"宏观数据已保存到: {industry_profit_file}, {macro_rates_file}")
    return StockDataDF, StockDataDF2


def Get3SheetsFromSql(
        datasavepath=None,
        **db_kwargs
        ):
    """
    从数据库中获取3个表格数据
    
    Args:
        datasavepath: 数据保存路径，默认从配置获取
        **db_kwargs: 数据库连接参数，会覆盖配置文件
    
    Returns:
        tuple: (资产负债表, 现金流量表, 利润表)
    """
    logger.info("从数据库中获取3个表格数据")
    
    # 获取数据保存路径
    if datasavepath is None:
        datasavepath = get_config('main.paths.data_root')

    if not os.path.exists(datasavepath):
        os.makedirs(datasavepath)

    # 获取数据库配置
    db_config = get_database_config()
    db_config.update(db_kwargs)
    
    try:
        db = pymssql.connect(
            host=db_config['host'], 
            user=db_config['user'], 
            password=db_config['password'], 
            database=db_config['database']
        )
    except Exception as e:
        logger.error(f"数据库连接失败: {e}")
        raise
    cursor = db.cursor()

    sql_str = "SELECT * FROM [stock_data].[dbo].[fzb]"
    cursor.execute(sql_str)
    data1 = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]
    fzb = pd.DataFrame(data1, columns=columns)
    fzb["reportday"] = pd.to_datetime(
        fzb["reportday"], format="%Y%m%d"
    )
    fzb["tradingday"] = pd.to_datetime(
        fzb["tradingday"], format="%Y%m%d"
    )
    fzb = fzb.replace(123456789.0, np.nan)

    sql_str = "SELECT * FROM [stock_data].[dbo].[xjlb]"
    cursor.execute(sql_str)
    data2 = cursor.fetchall()
    columns2 = [desc[0] for desc in cursor.description]
    xjlb = pd.DataFrame(data2, columns=columns2)
    xjlb["reportday"] = pd.to_datetime(
        xjlb["reportday"], format="%Y%m%d"
    )
    xjlb["tradingday"] = pd.to_datetime(
        xjlb["tradingday"], format="%Y%m%d"
    )
    xjlb = xjlb.replace(123456789.0, np.nan)

    sql_str = "SELECT * FROM [stock_data].[dbo].[lrb]"
    cursor.execute(sql_str)
    data3 = cursor.fetchall()
    columns3 = [desc[0] for desc in cursor.description]
    lrb = pd.DataFrame(data3, columns=columns3)
    lrb["reportday"] = pd.to_datetime(
        lrb["reportday"], format="%Y%m%d"
    )
    lrb["tradingday"] = pd.to_datetime(
        lrb["tradingday"], format="%Y%m%d"
    )
    lrb = lrb.replace(123456789.0, np.nan)

    # 保存数据
    fzb_file = os.path.join(datasavepath, "fzb.pkl")
    xjlb_file = os.path.join(datasavepath, "xjlb.pkl")
    lrb_file = os.path.join(datasavepath, "lrb.pkl")
    
    pd.to_pickle(fzb, fzb_file)
    pd.to_pickle(xjlb, xjlb_file)
    pd.to_pickle(lrb, lrb_file)
    
    cursor.close()
    db.close()
    
    logger.info(f"存储3个表格数据成功: {fzb_file}, {xjlb_file}, {lrb_file}")
    return fzb, xjlb, lrb


def GetAllDayPriceDataFromSql_save(
        datasavepath=None,
        **db_kwargs
        ):
    """
    读取所有股票日线数据，并存储，然后返回沪深交易所的股票数据
    
    Args:
        datasavepath: 数据保存路径，默认从配置获取
        **db_kwargs: 数据库连接参数，会覆盖配置文件
    
    Returns:
        tuple: (价格数据, 可交易股票数据)
    """
    logger.info("从数据库中读取所有股票日线数据")
    
    # 获取数据保存路径
    if datasavepath is None:
        datasavepath = get_config('main.paths.data_root')
    if not os.path.exists(datasavepath):
        os.makedirs(datasavepath)
    
    # 获取数据库配置
    db_config = get_database_config()
    db_config.update(db_kwargs)

    def _PriceDf_storage(datasavepath):
        # 使用增量更新模式获取数据
        PriceDf = GetStockDayDataDFFromSql(
            begindate=20200101,  # 如果是全量更新的起始日期
            enddate=0,
            incremental_update=True,  # 启用增量更新
            **db_config
        )
        
        # 如果GetStockDayDataDFFromSql已经处理了增量更新，直接返回
        if hasattr(PriceDf.index, 'names') and 'TradingDates' in PriceDf.index.names:
            logger.info("数据已经通过增量更新处理完成")
            return PriceDf
        
        # 否则按照原有逻辑处理（用于非增量更新情况）
        if not PriceDf.empty:
            StockDataDF = PriceDf  # 重命名以保持向后兼容
            StockDataDF["tradingday"] = pd.to_datetime(
                StockDataDF["tradingday"], format="%Y%m%d"
            )
            PriceDf = StockDataDF.set_index(["tradingday", "code"])
            PriceDf.index.set_names(["TradingDates", "StockCodes"], inplace=True)
            PriceDf = PriceDf.sort_index()
            
            # 获取涨跌停价格数据
            StockStopPrice = GetStockStopPrice(**db_config)
            if not StockStopPrice.empty:
                StockStopPrice["tradingday"] = pd.to_datetime(
                    StockStopPrice["tradingday"], format="%Y%m%d"
                )
                StockStopPrice = StockStopPrice.set_index(["tradingday", "code"])
                StockStopPrice.index.set_names(["TradingDates", "StockCodes"], inplace=True)
                StockStopPrice = StockStopPrice.sort_index()
                PriceDf = PriceDf.join(StockStopPrice, how="left")

            PriceDf["MC"] = PriceDf["total_shares"] * PriceDf["c"]
            PriceDf["FMC"] = PriceDf["free_float_shares"] * PriceDf["c"]
            PriceDf["turnoverrate"] = PriceDf["v"] / PriceDf["total_shares"]
            PriceDf["vwap"] = PriceDf["amt"] / PriceDf["v"]
            PriceDf["freeturnoverrate"] = PriceDf["v"] / PriceDf["free_float_shares"]
            
            price_file = os.path.join(datasavepath, "Price.pkl")
            PriceDf.to_pickle(price_file)
            logger.info(f"价格数据已保存到: {price_file}")
        
        return PriceDf

    PriceDf = _PriceDf_storage(datasavepath)

    def _tradable_storage(datasavepath):
        StockTradableDF = GetTradableStocksFromSql()
        StockTradableDF["tradingday"] = pd.to_datetime(
            StockTradableDF["tradingday"], format="%Y%m%d"
        )

        StockTradableDF = StockTradableDF.set_index(["tradingday", "code"])
        StockTradableDF.index.set_names(["TradingDates", "StockCodes"], inplace=True)
        StockTradableDF = StockTradableDF.sort_index()
        
        tradable_file = os.path.join(datasavepath, "TradableDF.pkl")
        pd.to_pickle(StockTradableDF, tradable_file)
        logger.info(f"可交易股票数据已保存到: {tradable_file}")
        return StockTradableDF

    StockTradableDF = _tradable_storage(datasavepath)

    return PriceDf, StockTradableDF


def GetStockStopPrice(
        **db_kwargs
        ):
    """
    从数据库中获取所有股票当天涨停价
    
    Args:
        **db_kwargs: 数据库连接参数，会覆盖配置文件
    
    Returns:
        DataFrame: 涨跌停价格数据
    """
    logger.info("从数据库中获取所有股票当天涨停价")
    
    # 获取数据库配置
    db_config = get_database_config()
    db_config.update(db_kwargs)
    
    try:
        db = pymssql.connect(
            host=db_config['host'], 
            user=db_config['user'], 
            password=db_config['password'], 
            database=db_config['database']
        )
    except Exception as e:
        logger.error(f"数据库连接失败: {e}")
        raise
    cursor = db.cursor()
    sql_str = "SELECT distinct [code],[tradingday],[high_limit],[low_limit] FROM [stock_data].[dbo].[lgc_涨跌停板]"
    cursor.execute(sql_str)
    data1 = cursor.fetchall()
    cursor.close()
    db.close()
    StockDataDF = pd.DataFrame(data1)
    StockDataDF.columns = ["code", "tradingday", "high_limit", "low_limit"]
    
    logger.info(f"获取涨跌停价格数据完成，共{len(StockDataDF)}条记录")
    return StockDataDF


def GetIndexNamelistFromSql(
        **db_kwargs
    ):
    """
    从数据库中获取所有板块名称已经对应的板块代码
    
    Args:
        **db_kwargs: 数据库连接参数，会覆盖配置文件
    
    Returns:
        DataFrame: 板块代码和名称
    """
    logger.info("从数据库中获取所有板块名称及对应的板块代码")
    
    # 获取数据库配置
    db_config = get_database_config()
    db_config.update(db_kwargs)
    
    try:
        db = pymssql.connect(
            host=db_config['host'], 
            user=db_config['user'], 
            password=db_config['password'], 
            database=db_config['database']
        )
    except Exception as e:
        logger.error(f"数据库连接失败: {e}")
        raise
    cursor = db.cursor()
    sql_str = "SELECT distinct [concept_code],[concept_name] FROM [stock_data].[dbo].[lgc_板块成份股]"
    cursor.execute(sql_str)
    data1 = cursor.fetchall()
    cursor.close()
    db.close()
    StockDataDF = pd.DataFrame(data1)
    StockDataDF.columns = ["concept_code", "concept_name"]
    
    logger.info(f"获取板块名称数据完成，共{len(StockDataDF)}条记录")
    return StockDataDF


def GetTradableStocksFromSql(
        **db_kwargs
    ):
    """
    从数据库中获取所有股票上市退市时间
    
    Args:
        **db_kwargs: 数据库连接参数，会覆盖配置文件
    
    Returns:
        DataFrame: 股票可交易时间数据
    """
    logger.info("从数据库中获取所有股票上市退市时间")
    
    # 获取数据库配置
    db_config = get_database_config()
    db_config.update(db_kwargs)
    
    try:
        db = pymssql.connect(
            host=db_config['host'], 
            user=db_config['user'], 
            password=db_config['password'], 
            database=db_config['database']
        )
    except Exception as e:
        logger.error(f"数据库连接失败: {e}")
        raise
    cursor = db.cursor()
    sql_str = "SELECT [ipo_date],[code],[exchange_id],[last_trade_day],[tradingday],[trade_status] FROM [stock_data].[dbo].[all_stocks]"
    cursor.execute(sql_str)
    data1 = cursor.fetchall()
    cursor.close()
    db.close()
    StockDataDF = pd.DataFrame(data1)
    StockDataDF.columns = [
        "ipo_date",
        "code",
        "exchange_id",
        "last_trade_day",
        "tradingday",
        "trade_status",
    ]
    
    logger.info(f"从数据库中成功读取上市退市数据，共{len(StockDataDF)}条记录")
    return StockDataDF


def GetIndexComponentFromSql(
        IndexCode="all",
        begindate=20160101,
        **db_kwargs
        ):
    """
    从数据库中获取板块成份信息
    
    Args:
        IndexCode: 指数代码，默认为"all"
        begindate: 开始日期
        **db_kwargs: 数据库连接参数，会覆盖配置文件
    
    Returns:
        DataFrame: 板块成份股信息
    """
    logger.info(f"从数据库中获取板块成份信息：{IndexCode}，日期：{begindate}")
    
    # 获取数据库配置
    db_config = get_database_config()
    db_config.update(db_kwargs)
    
    try:
        db = pymssql.connect(
            host=db_config['host'], 
            user=db_config['user'], 
            password=db_config['password'], 
            database=db_config['database']
        )
    except Exception as e:
        logger.error(f"数据库连接失败: {e}")
        raise
    cursor = db.cursor()
    if IndexCode == "all":
        sql_str = (
            "select [tradingday],[sel_day],[指数类型],[concept_code],[concept_name],[code] from [stock_data].[dbo].[lgc_板块成份股] where tradingday= %d  "
            % begindate
        )
    else:
        sql_str0 = "select [tradingday],[sel_day],[指数类型],[concept_code],[concept_name],[code] from [stock_data].[dbo].[lgc_板块成份股] where tradingday={begindate}  and concept_code={IndexCode}"
        sql_str = sql_str0.format(begindate=begindate, IndexCode=IndexCode)
    cursor.execute(sql_str)
    data1 = cursor.fetchall()
    cursor.close()
    db.close()
    StockDataDF = pd.DataFrame(data1)
    StockDataDF.columns = [
        "tradingday",
        "sel_day",
        "指数类型",
        "concept_code",
        "concept_name",
        "code",
    ]
    
    logger.info(f"从数据库中成功读取板块成份数据，共{len(StockDataDF)}条记录")
    return StockDataDF


def GetIndexPriceFromSql(IndexCodes,
                         begindate,
                         enddate,
                         **db_kwargs
                         ):
    """
    从数据库中获取板块价格信息
    
    Args:
        IndexCodes: 指数代码
        begindate: 开始日期
        enddate: 结束日期
        **db_kwargs: 数据库连接参数，会覆盖配置文件
    
    Returns:
        DataFrame: 板块价格信息
    """
    logger.info(f"从数据库中获取板块价格信息：{IndexCodes}，时间范围：{begindate}-{enddate}")
    
    # 获取数据库配置
    db_config = get_database_config()
    db_config.update(db_kwargs)
    
    try:
        db = pymssql.connect(
            host=db_config['host'], 
            user=db_config['user'], 
            password=db_config['password'], 
            database=db_config['database']
        )
    except Exception as e:
        logger.error(f"数据库连接失败: {e}")
        raise
    cursor = db.cursor()
    sql_str0 = "SELECT [bankuai],[tradingday],[exchange_id],[index_name],[code],[o],[h],[l],[c],[v],[amt],[writing_day] FROM [stock_data].[dbo].[wind_index] where tradingday>={begindate} and tradingday <= {enddate} and code='{IndexCodes}' order by tradingday"
    sql_str = sql_str0.format(
        begindate=begindate, enddate=enddate, IndexCodes=IndexCodes
    )
    cursor.execute(sql_str)
    data1 = cursor.fetchall()
    cursor.close()
    db.close()
    StockDataDF = pd.DataFrame(data1)
    StockDataDF.columns = [
        "bankuai",
        "tradingday",
        "exchange_id",
        "index_name",
        "code",
        "o",
        "h",
        "l",
        "c",
        "v",
        "amt",
        "writing_day",
    ]
    
    logger.info(f"从数据库中成功读取板块价格数据，共{len(StockDataDF)}条记录")
    return StockDataDF


def GetAllTradingDatesFromSql(
        **db_kwargs
    ):
    """
    从数据库中获取所有交易日期
    
    Args:
        **db_kwargs: 数据库连接参数，会覆盖配置文件
    
    Returns:
        ndarray: 交易日期数组
    """
    logger.info("从数据库中获取所有交易日期")
    
    # 获取数据库配置
    db_config = get_database_config()
    db_config.update(db_kwargs)
    
    try:
        db = pymssql.connect(
            host=db_config['host'], 
            user=db_config['user'], 
            password=db_config['password'], 
            database=db_config['database']
        )
    except Exception as e:
        logger.error(f"数据库连接失败: {e}")
        raise
    cursor = db.cursor()
    sql_str = "select [tradingday] from tradingday order by tradingday"
    cursor.execute(sql_str)
    Tradingdates = cursor.fetchall()
    Tradingdates = np.array(Tradingdates)
    cursor.close()
    db.close()
    
    logger.info(f"从数据库中成功获取交易日期，共{len(Tradingdates)}个日期")
    return Tradingdates


def GetAllSTStocksFromSql(
        **db_kwargs
    ):
    """
    从数据库中获取所有ST股票信息
    
    Args:
        **db_kwargs: 数据库连接参数，会覆盖配置文件
    
    Returns:
        ndarray: ST股票信息数组
    """
    logger.info("从数据库中获取所有ST股票信息")
    
    # 获取数据库配置
    db_config = get_database_config()
    db_config.update(db_kwargs)
    
    try:
        db = pymssql.connect(
            host=db_config['host'], 
            user=db_config['user'], 
            password=db_config['password'], 
            database=db_config['database']
        )
    except Exception as e:
        logger.error(f"数据库连接失败: {e}")
        raise
    cursor = db.cursor()
    sql_str = "SELECT [tradingday],[code],[exchange_id],[sec_name] FROM [stock_data].[dbo].[ST] order by tradingday "
    cursor.execute(sql_str)
    STStockname = cursor.fetchall()
    STStockname = np.array(STStockname)
    cursor.close()
    db.close()
    
    logger.info(f"从数据库中成功获取ST股票信息，共{len(STStockname)}条记录")
    return STStockname


def GetStock1minDataFromSql(StockCode, date, **db_kwargs):
    """
    从数据库读取1分钟行情数据
    
    Args:
        StockCode: 股票代码
        date: 日期
        **db_kwargs: 数据库连接参数，会覆盖配置文件
    
    Returns:
        ndarray: 1分钟行情数据
    """
    logger.info(f"从数据库读取1分钟行情数据：{StockCode}，日期：{date}")
    
    # 获取数据库配置，1分钟数据使用不同的数据库
    db_config = get_database_config()
    db_config.update(db_kwargs)
    # 1分钟数据默认使用stock_min1数据库
    if 'database' not in db_kwargs:
        db_config['database'] = 'stock_min1'
    
    try:
        db = pymssql.connect(
            host=db_config['host'], 
            user=db_config['user'], 
            password=db_config['password'], 
            database=db_config['database']
        )
    except Exception as e:
        logger.error(f"数据库连接失败: {e}")
        raise
    cursor = db.cursor()
    StockData = np.array([])
    if isinstance(date, np.int32):
        date = [date]

    for i in range(len(date)):
        sql_str0 = "select [tradingday],[tradingtime],[o],[h],[l],[c],[v],[amt],[adjfactor] from [stock_min1].[dbo].[{StockCode}] where tradingday='{i}' order by tradingtime"
        sql_str = sql_str0.format(StockCode=StockCode, i=date[i])
        cursor.execute(sql_str)
        StockData0 = cursor.fetchall()
        StockData = np.append(StockData, StockData0)
    StockData = np.reshape(StockData, (int(len(StockData) / 9), 9))
    cursor.close()
    db.close()
    
    logger.info(f"从数据库中成功获取1分钟数据，共{len(StockData)}条记录")
    return StockData


def GetStockDayDataDFFromSql(
        begindate=20200101,
        enddate=0,
        batch_size=1000000,
        incremental_update=True,  # 新增增量更新参数
        **db_kwargs
        ):
    """
    从数据库中读取所有股票日线信息，支持增量更新
    
    Args:
        begindate: 开始日期
        enddate: 结束日期，0表示至今
        batch_size: 批次大小
        incremental_update: 是否启用增量更新
        **db_kwargs: 数据库连接参数，会覆盖配置文件
    
    Returns:
        DataFrame: 日线数据
    """
    
    # 获取数据库配置
    db_config = get_database_config()
    db_config.update(db_kwargs)
    
    # 如果启用增量更新，先检查本地数据
    if incremental_update:
        logger.info("启用增量更新模式")
        incremental_begindate = _get_last_trading_date_from_local()
        if incremental_begindate:
            begindate = incremental_begindate
            logger.info(f"从本地数据确定的更新起始日期：{begindate}")
        else:
            logger.info(f"使用默认起始日期：{begindate}")
    
    logger.info(f"从数据库中读取股票日线数据：{begindate} 到 {enddate if enddate != 0 else '今天'}")
    
    try:
        db = pymssql.connect(
            host=db_config['host'], 
            user=db_config['user'], 
            password=db_config['password'], 
            database=db_config['database']
        )
    except Exception as e:
        logger.error(f"数据库连接失败: {e}")
        raise
    cursor = db.cursor()
    
    # 先检查表中是否有数据
    test_sql = "select count(*) from [stock_data].[dbo].[day5]"
    cursor.execute(test_sql)
    total_count = cursor.fetchone()[0]
    logger.info(f"day5表中总共有{total_count}条记录")
    
    # 检查日期范围
    date_range_sql = "select min(tradingday), max(tradingday) from [stock_data].[dbo].[day5]"
    cursor.execute(date_range_sql)
    date_range = cursor.fetchone()
    if date_range:
        logger.info(f"数据日期范围: {date_range[0]} 到 {date_range[1]}")
    
    # 检查2014年的数据是否存在
    check_2014_sql = "select count(*) from [stock_data].[dbo].[day5] where tradingday >= 20140101 and tradingday < 20150101"
    cursor.execute(check_2014_sql)
    count_2014 = cursor.fetchone()[0]
    logger.info(f"2014年数据条数: {count_2014}")
    
    # 移除ORDER BY避免大结果集排序问题，后续在Python中排序
    if enddate == 0:
        sql_str = (
            "select [code],[tradingday],[o],[h],[l],[c],[v],[amt],[adjfactor],[total_shares],[free_float_shares],[exchange_id] from [stock_data].[dbo].[day5] where tradingday >= %d"
            % begindate
        )
    else:
        sql_str = (
            "select [code],[tradingday],[o],[h],[l],[c],[v],[amt],[adjfactor],[total_shares],[free_float_shares],[exchange_id] from [stock_data].[dbo].[day5] where tradingday >= %d and tradingday <= %d"
            % (begindate, enddate)
        )
    logger.info(f"执行SQL查询: {sql_str}")
    logger.info("开始执行查询，请耐心等待...")
    
    import time
    start_time = time.time()
    cursor.execute(sql_str)
    
    logger.info("查询完成，开始获取结果...")
    StockData = cursor.fetchall()
    end_time = time.time()
    
    logger.info(f"查询耗时: {end_time - start_time:.2f} 秒，获取 {len(StockData)} 条记录")
    cursor.close()
    db.close()
    
    # 检查查询结果是否为空
    if not StockData:
        logger.warning(f"数据库查询返回空结果。SQL: {sql_str}")
        # 返回空的DataFrame但包含正确的列结构
        StockDataDF = pd.DataFrame(columns=[
            "code",
            "tradingday", 
            "o",
            "h",
            "l",
            "c",
            "v",
            "amt",
            "adjfactor",
            "total_shares",
            "free_float_shares",
            "exchange_id",
        ])
    else:
        StockDataDF = pd.DataFrame(StockData)
        StockDataDF.columns = [
            "code",
            "tradingday",
            "o",
            "h",
            "l",
            "c",
            "v",
            "amt",
            "adjfactor",
            "total_shares",
            "free_float_shares",
            "exchange_id",
        ]
        
        # 在Python端进行排序，避免数据库端大结果集排序问题
        if len(StockDataDF) > 0:
            StockDataDF = StockDataDF.sort_values(['tradingday', 'code'])
            logger.info("数据排序完成")
    
    logger.info(f"从数据库中成功读取行情数据，共{len(StockDataDF)}条记录")
    
    # 如果是增量更新且有新数据，需要与本地数据合并
    if incremental_update and not StockDataDF.empty:
        logger.info("准备进行数据格式转换和合并...")
        
        # 转换数据格式（与原有逻辑保持一致）
        StockDataDF["tradingday"] = pd.to_datetime(StockDataDF["tradingday"], format="%Y%m%d")
        PriceDf = StockDataDF.set_index(["tradingday", "code"])
        PriceDf.index.set_names(["TradingDates", "StockCodes"], inplace=True)
        PriceDf = PriceDf.sort_index()
        
        # 添加计算字段（与原有逻辑保持一致）
        PriceDf["MC"] = PriceDf["total_shares"] * PriceDf["c"]
        PriceDf["FMC"] = PriceDf["free_float_shares"] * PriceDf["c"]  
        PriceDf["turnoverrate"] = PriceDf["v"] / PriceDf["total_shares"]
        PriceDf["vwap"] = PriceDf["amt"] / PriceDf["v"]
        PriceDf["freeturnoverrate"] = PriceDf["v"] / PriceDf["free_float_shares"]
        
        # 与本地数据合并
        data_root = get_config('main.paths.data_root')
        merged_data = _merge_with_local_data(PriceDf, data_root)
        
        # 保存合并后的数据
        price_file = os.path.join(data_root, "Price.pkl")
        merged_data.to_pickle(price_file)
        logger.info(f"增量更新完成，数据已保存到: {price_file}")
        
        return merged_data
    
    return StockDataDF


def GetForeshowFromSql(
        **db_kwargs
    ):
    """
    从数据库中读取所有股票预报信息
    
    Args:
        **db_kwargs: 数据库连接参数，会覆盖配置文件
    
    Returns:
        DataFrame: 股票预报信息
    """
    logger.info("从数据库中读取所有股票预报信息")
    
    # 获取数据库配置
    db_config = get_database_config()
    db_config.update(db_kwargs)
    
    try:
        db = pymssql.connect(
            host=db_config['host'], 
            user=db_config['user'], 
            password=db_config['password'], 
            database=db_config['database']
        )
    except Exception as e:
        logger.error(f"数据库连接失败: {e}")
        raise
    cursor = db.cursor()
    sql_str = "SELECT [code],[reportday],[tradingday],[l],[h],[lb],[hb],[l_kf],[h_kf],[lb_kf],[hb_kf]   FROM [stock_data].[dbo].[foreshow]"
    cursor.execute(sql_str)
    data1 = cursor.fetchall()
    cursor.close()
    db.close()
    StockDataDF = pd.DataFrame(data1)
    StockDataDF.columns = [
        "code",
        "reportday",
        "tradingday",
        "l",
        "h",
        "lb",
        "hb",
        "l_kf",
        "h_kf",
        "lb_kf",
        "hb_kf",
    ]
    
    logger.info(f"从数据库中成功读取预报数据，共{len(StockDataDF)}条记录")
    return StockDataDF


def GetIPOdateFromSql(
        **db_kwargs
    ):
    """
    从数据库中读取所有IPO日期
    
    Args:
        **db_kwargs: 数据库连接参数，会覆盖配置文件
    
    Returns:
        DataFrame: IPO日期数据
    """
    logger.info("从数据库中读取所有IPO日期")
    
    # 获取数据库配置
    db_config = get_database_config()
    db_config.update(db_kwargs)
    
    try:
        db = pymssql.connect(
            host=db_config['host'], 
            user=db_config['user'], 
            password=db_config['password'], 
            database=db_config['database']
        )
    except Exception as e:
        logger.error(f"数据库连接失败: {e}")
        raise
    cursor = db.cursor()
    sql_str = "select [code],[ipo_date] from [stock_data].[dbo].[all_stocks] where tradingday='20250318'"
    cursor.execute(sql_str)
    StockData = cursor.fetchall()
    cursor.close()
    db.close()
    StockDataDF = pd.DataFrame(StockData)
    
    logger.info(f"从数据库中成功读取IPO日期数据，共{len(StockDataDF)}条记录")
    return StockDataDF


def GetAllConceptNameFromSql(
        **db_kwargs
    ):
    """
    从数据库中获取所有概念板块名称
    
    Args:
        **db_kwargs: 数据库连接参数，会覆盖配置文件
    
    Returns:
        list: 概念板块名称数据
    """
    logger.info("从数据库中获取所有概念板块名称")
    
    # 获取数据库配置，概念数据使用jqdata数据库
    db_config = get_database_config()
    db_config.update(db_kwargs)
    # 概念数据默认使用jqdata数据库
    if 'database' not in db_kwargs:
        db_config['database'] = 'jqdata'
    
    try:
        db = pymssql.connect(
            host=db_config['host'], 
            user=db_config['user'], 
            password=db_config['password'], 
            database=db_config['database']
        )
    except Exception as e:
        logger.error(f"数据库连接失败: {e}")
        raise
    cursor = db.cursor()
    sql_str = "SELECT distinct concept_name  FROM [jqdata].[dbo].[concept_codes]"
    cursor.execute(sql_str)
    StockData = cursor.fetchall()
    cursor.close()
    db.close()
    
    logger.info(f"从数据库中成功读取概念板块名称，共{len(StockData)}条记录")
    return StockData


def GetConceptComponentByNameFromSql(
        ConceptName,
        **db_kwargs
    ):
    """
    从数据库中获取概念板块成份信息
    
    Args:
        ConceptName: 概念板块名称
        **db_kwargs: 数据库连接参数，会覆盖配置文件
    
    Returns:
        list: 概念板块成份股数据
    """
    logger.info(f"从数据库中获取概念板块成份信息：{ConceptName}")
    
    # 获取数据库配置，概念数据使用jqdata数据库
    db_config = get_database_config()
    db_config.update(db_kwargs)
    # 概念数据默认使用jqdata数据库
    if 'database' not in db_kwargs:
        db_config['database'] = 'jqdata'
    
    try:
        db = pymssql.connect(
            host=db_config['host'], 
            user=db_config['user'], 
            password=db_config['password'], 
            database=db_config['database']
        )
    except Exception as e:
        logger.error(f"数据库连接失败: {e}")
        raise
    cursor = db.cursor()
    sql_str = (
        "SELECT [traddingday],[code],[exchangeid] FROM [jqdata].[dbo].[concept_codes] where concept_name='%S' order by tradingday "
        % ConceptName
    )
    cursor.execute(sql_str)
    StockData = cursor.fetchall()
    cursor.close()
    db.close()
    
    logger.info(f"从数据库中成功读取概念板块成份数据，共{len(StockData)}条记录")
    return StockData


def GetWideBaseComponentFromSql(
        IndexCode,
        **db_kwargs
        ):
    """
    获取宽基指数成份股与权重
    
    Args:
        IndexCode: 指数代码，如SH000300, SH000905
        **db_kwargs: 数据库连接参数，会覆盖配置文件
    
    Returns:
        DataFrame: 寽基指数成份股权重数据
    """
    logger.info(f"获取宽基指数成份股与权重：{IndexCode}")
    
    # 获取数据库配置，宽基数据使用jqdata数据库
    db_config = get_database_config()
    db_config.update(db_kwargs)
    # 宽基数据默认使用jqdata数据库
    if 'database' not in db_kwargs:
        db_config['database'] = 'jqdata'
    
    try:
        db = pymssql.connect(
            host=db_config['host'], 
            user=db_config['user'], 
            password=db_config['password'], 
            database=db_config['database']
        )
    except Exception as e:
        logger.error(f"数据库连接失败: {e}")
        raise
    cursor = db.cursor()
    sql_str = (
        "select [tradingday],[index_code] ,[code],[exchange_id],[weight] from [jqdata].[dbo].[index_weights] where index_code='%s' order by tradingday"
        % IndexCode
    )
    cursor.execute(sql_str)
    StockData = cursor.fetchall()
    cursor.close()
    db.close()
    StockDataDF = pd.DataFrame(StockData)
    StockDataDF.columns = [
        "TradingDates",
        "index_code",
        "StockCodes",
        "exchange_id",
        "weight",
    ]
    StockDataDF.set_index(["TradingDates", "StockCodes"], inplace=True)
    StockDataDF.sort_index(inplace=True)
    
    logger.info(f"从数据库中成功获取寽基指数成份股数据，共{len(StockDataDF)}条记录")
    return StockDataDF


def GetAllannouncementFromSql(
        **db_kwargs
    ):
    """
    从数据库中获取所有公告信息
    
    Args:
        **db_kwargs: 数据库连接参数，会覆盖配置文件
    
    Returns:
        DataFrame: 公告信息数据
    """
    logger.info("从数据库中获取所有公告信息")
    
    # 获取数据库配置，公告数据使用Wind数据库
    db_config = get_database_config()
    db_config.update(db_kwargs)
    # 公告数据默认使用Wind数据库
    if 'database' not in db_kwargs:
        db_config['database'] = 'Wind'
    
    sql_str = "select [secCode],[announcementTitle],[tradingday],[tradingtime],[category] from [Wind].[dbo].[jczx_gg1] order by tradingday"
    
    try:
        db = pymssql.connect(
            host=db_config['host'], 
            user=db_config['user'], 
            password=db_config['password'], 
            database=db_config['database']
        )
    except Exception as e:
        logger.error(f"数据库连接失败: {e}")
        raise
    cursor = db.cursor()
    cursor.execute(sql_str)
    StockData = cursor.fetchall()
    cursor.close()
    db.close()
    StockDataDF = pd.DataFrame(StockData)
    StockDataDF.columns = [
        "secCode",
        "announcementTitle",
        "tradingday",
        "tradingtime",
        "category",
    ]
    
    logger.info(f"从数据库中成功获取公告信息，共{len(StockDataDF)}条记录")
    return StockDataDF


def GetFinancialItemFromSql(
        SheetTitle,
        item,
        **db_kwargs
        ):
    """
    获取指定表中指定条目数据
    
    Args:
        SheetTitle: 表名
        item: 数据项名称
        **db_kwargs: 数据库连接参数，会覆盖配置文件
    
    Returns:
        DataFrame: 指定表中的数据项
    """
    logger.info(f"获取指定表中指定条目数据：{SheetTitle}.{item}")
    
    # 获取数据库配置
    db_config = get_database_config()
    db_config.update(db_kwargs)
    
    sql_str0 = "select [code],[reportday],[tradingday],[d_quarter],[d_year],[{item}]  from [stock_data].[dbo].[{SheetTitle}]  order by tradingday"
    sql_str = sql_str0.format(item=item, SheetTitle=SheetTitle)
    
    try:
        db = pymssql.connect(
            host=db_config['host'], 
            user=db_config['user'], 
            password=db_config['password'], 
            database=db_config['database']
        )
    except Exception as e:
        logger.error(f"数据库连接失败: {e}")
        raise
    cursor = db.cursor()
    cursor.execute(sql_str)
    StockData = cursor.fetchall()
    cursor.close()
    db.close()
    StockDataDF = pd.DataFrame(StockData)
    StockDataDF.columns = [
        "code",
        "reportday",
        "tradingday",
        "d_quarter",
        "d_year",
        item,
    ]
    
    logger.info(f"从数据库中成功获取表{SheetTitle}中的{item}数据，共{len(StockDataDF)}条记录")
    return StockDataDF


def GetIndexPointFromSql(
        indexname,
        **db_kwargs
    ):
    """
    获取指数价格信息
    
    Args:
        indexname: 指数名称
        **db_kwargs: 数据库连接参数，会覆盖配置文件
    
    Returns:
        DataFrame: 指数价格数据
    """
    logger.info(f"获取指数价格信息：{indexname}")
    
    # 获取数据库配置
    db_config = get_database_config()
    db_config.update(db_kwargs)
    
    try:
        db = pymssql.connect(
           host=db_config['host'], 
           user=db_config['user'], 
           password=db_config['password'], 
           database=db_config['database']
        )
    except Exception as e:
        logger.error(f"数据库连接失败: {e}")
        raise
    cursor = db.cursor()
    sql_str = (
        "select [tradingday],[o],[h],[l],[c],[v],[amt] from [stock_data].[dbo].[wind_index] where index_name = '%s' order by  tradingday"
        % (indexname)
    )
    cursor.execute(sql_str)
    StockData = cursor.fetchall()
    StockData = pd.DataFrame(StockData)
    StockData.columns = ["tradingday", "o", "h", "l", "c", "v", "amt"]
    cursor.close()
    db.close()
    
    logger.info(f"从数据库中成功获取指数{indexname}价格数据，共{len(StockData)}条记录")
    return StockData


def GetIndustryOneHotFromApi(
            dateslist,
            indextype="申万行业板块",
            industry="sw_l2",
        ):
    print("从API中获取行业独热编码")
    cla = ComData.lgc_板块成份股(指数类型=indextype, industry=industry)
    Industry_one_hot_matrix = []
    if type(dateslist[0]) == pd._libs.tslibs.timestamps.Timestamp:
        dateslist = [int(i.strftime("%Y%m%d")) for i in dateslist]

    for i, date in enumerate(dateslist):
        成份股_data = cla.读取每日成分股(date, date)
        oh = pd.get_dummies(
            成份股_data,
            columns=["concept_name"],
            prefix_sep="_",
            dummy_na=False,
            drop_first=False,
        )
        if i == 0:
            Industry_one_hot_matrix = oh
        else:
            Industry_one_hot_matrix = pd.concat([Industry_one_hot_matrix, oh], axis=0)

    Industry_one_hot_matrix["tradingday"] = pd.to_datetime(
        Industry_one_hot_matrix["tradingday"], format="%Y%m%d"
    )
    Industry_one_hot_matrix["codes"] = Industry_one_hot_matrix["code"].astype(str)
    Industry_one_hot_matrix.rename(
        columns={"tradingday": "TradingDates", "codes": "StockCodes"}, inplace=True
    )
    Industry_one_hot_matrix.set_index(["TradingDates", "StockCodes"], inplace=True)
    classification_one_hot = Industry_one_hot_matrix.drop(
        columns=[
            "sel_day",
            "industry",
            "concept_code",
            "code",
            "code_cn",
            "exchange_id",
        ]
    )

    return classification_one_hot


def GetWideBaseByDateSerriesFromApi(
        dateslist,
        指数类型="沪深交易所核心指数",
        指数code="000905"
        ):
    cla = ComData.lgc_板块成份股(指数类型, 指数code)
    WideBase_matrix = []
    if type(dateslist[0]) == pd._libs.tslibs.timestamps.Timestamp:
        dateslist = [int(i.strftime("%Y%m%d")) for i in dateslist]
    for i, date in enumerate(dateslist):
        print(date)
        成份股_data = cla.读取每日成分股(date, date)
        if i == 0:
            WideBase_matrix = 成份股_data
        else:
            WideBase_matrix = pd.concat([WideBase_matrix, 成份股_data], axis=0)
    WideBase_matrix["tradingday"] = pd.to_datetime(
        WideBase_matrix["tradingday"], format="%Y%m%d"
    )
    WideBase_matrix["codes"] = WideBase_matrix["code"].astype(str)
    WideBase_matrix.rename(
        columns={"tradingday": "TradingDates", "codes": "StockCodes"}, inplace=True
    )
    WideBase_matrix.set_index(["TradingDates", "StockCodes"], inplace=True)
    return WideBase_matrix


def GetFinancialh5fileFromLAN_save(
            sourcepath=None,
            savepath=None
        ):
    """
    从局域网中获得财务数据 h5文件
    
    Args:
        sourcepath: 源文件路径，默认从配置获取
        savepath: 保存路径，默认从配置获取
    
    Returns:
        str: 保存的文件名
    """
    logger.info("从局域网中获取财务数据 h5文件")
    
    # 获取路径配置
    if sourcepath is None:
        sourcepath = get_config('lan.financial_h5_source', r"\\198.16.102.65\lgc\h5")
    if savepath is None:
        savepath = get_config('main.paths.data_root')
    if not os.path.exists(sourcepath):
        raise FileNotFoundError("sourcepath not exists")

    f_index = os.path.join(sourcepath, "financial_v2.h5")
    savefilename = os.path.join(savepath, "financial_v2.h5")
    str_copy = "copy %s %s" % (f_index, savefilename)
    logger.info("从LAN中复制h5文件")
    os.system(str_copy)
    if os.path.exists(savefilename):
        logger.info("复制成功")
    else:
        logger.error("复制失败")
    return savefilename


def date_serries(PriceDf, type="daily"):
    Dateserries = PriceDf.index.get_level_values(0).unique()
    if type == "daily":
        return Dateserries
    if type == "weekly" :
        weekly_mask = Dateserries.to_series().dt.to_period(
            "W"
        ) != Dateserries.to_series().shift(1).dt.to_period("W")
        return Dateserries[weekly_mask]
    if type == "monthly":
        monthly_mask = Dateserries.to_series().dt.to_period(
            "M"
        ) != Dateserries.to_series().shift(1).dt.to_period("M")
        return Dateserries[monthly_mask]


def run():
    """
    运行主函数，使用配置管理系统
    """
    logger.info("开始运行数据获取主程序")
    
    # 从配置获取基础路径
    basedatapath = get_config('main.paths.data_root')
    
    if not os.path.exists(basedatapath):
        os.makedirs(basedatapath)
        logger.info(f"创建数据目录：{basedatapath}")

    logger.info(f"当前工作目录：{os.getcwd()}")
    
    # 使用配置管理系统获取数据
    Get3SheetsFromSql(datasavepath=basedatapath)

    PriceDf, StockTradableDF = GetAllDayPriceDataFromSql_save(
        datasavepath=basedatapath
    )

    monthlyserries = date_serries(PriceDf, type="monthly")
    
    # 检查ComData是否可用
    if ComData is not None:
        IndustryOneHot = GetIndustryOneHotFromApi(
                monthlyserries, indextype="申万行业板块", industry="sw_l1"
                )  # 行业独热编码
        one_hot_datasavepath = os.path.join(basedatapath, "Classificationdata")
        if not os.path.exists(one_hot_datasavepath):
            os.makedirs(one_hot_datasavepath)
        IndustryOneHot.to_pickle(os.path.join(one_hot_datasavepath, "classification_one_hot.pkl"))
        logger.info("行业独热编码数据处理完成")
    else:
        logger.warning("ComData模块不可用，跳过行业独热编码处理")
    
    logger.info("数据获取主程序运行完成")


if __name__ == "__main__":
    print("main")
    run()

