#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分块价格数据获取器
支持分块读取大量股票价格数据，避免内存溢出和数据库超时
"""

import os
import pandas as pd
import numpy as np
import time
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Tuple

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.config_manager import get_path, get_config
from core.database import execute_stock_data_query, execute_query

logger = logging.getLogger(__name__)


class ChunkedPriceFetcher:
    """分块价格数据获取器"""
    
    def __init__(self, chunk_days: int = 90, chunk_stocks: int = 1000):
        """
        初始化分块获取器
        
        Args:
            chunk_days: 每次获取的天数
            chunk_stocks: 每次获取的股票数量
        """
        self.chunk_days = chunk_days
        self.chunk_stocks = chunk_stocks
        self.data_root = get_path('data_root')
        
        logger.info(f"初始化分块价格获取器 - 分块天数: {chunk_days}, 分块股票数: {chunk_stocks}")
    
    def get_stock_list(self, db_name: str = 'database') -> List[str]:
        """获取所有股票代码列表"""
        sql = "SELECT DISTINCT code FROM [stock_data].[dbo].[day5] ORDER BY code"
        results = execute_query(sql, db_name)
        stock_codes = [row[0] for row in results]
        logger.info(f"获取到 {len(stock_codes)} 只股票")
        return stock_codes
    
    def get_date_range(self, begin_date: int, end_date: int) -> List[Tuple[int, int]]:
        """
        将日期范围分块
        
        Args:
            begin_date: 开始日期 (YYYYMMDD)
            end_date: 结束日期 (YYYYMMDD)
            
        Returns:
            日期分块列表 [(start1, end1), (start2, end2), ...]
        """
        if end_date == 0:
            end_date = int(datetime.now().strftime('%Y%m%d'))
        
        start_dt = datetime.strptime(str(begin_date), '%Y%m%d')
        end_dt = datetime.strptime(str(end_date), '%Y%m%d')
        
        date_chunks = []
        current_dt = start_dt
        
        while current_dt <= end_dt:
            chunk_end_dt = min(current_dt + timedelta(days=self.chunk_days), end_dt)
            chunk_start = int(current_dt.strftime('%Y%m%d'))
            chunk_end = int(chunk_end_dt.strftime('%Y%m%d'))
            date_chunks.append((chunk_start, chunk_end))
            current_dt = chunk_end_dt + timedelta(days=1)
        
        logger.info(f"日期范围 {begin_date}-{end_date} 分为 {len(date_chunks)} 个块")
        return date_chunks
    
    def get_stock_chunks(self, stock_codes: List[str]) -> List[List[str]]:
        """
        将股票代码分块
        
        Args:
            stock_codes: 股票代码列表
            
        Returns:
            股票代码分块列表
        """
        chunks = []
        for i in range(0, len(stock_codes), self.chunk_stocks):
            chunk = stock_codes[i:i + self.chunk_stocks]
            chunks.append(chunk)
        
        logger.info(f"{len(stock_codes)} 只股票分为 {len(chunks)} 个块")
        return chunks
    
    def fetch_price_chunk(self, 
                         begin_date: int, 
                         end_date: int, 
                         stock_codes: Optional[List[str]] = None,
                         db_name: str = 'database') -> pd.DataFrame:
        """
        获取指定日期范围和股票的价格数据块
        
        Args:
            begin_date: 开始日期
            end_date: 结束日期
            stock_codes: 股票代码列表，None表示所有股票
            db_name: 数据库名称
            
        Returns:
            价格数据DataFrame
        """
        # 构建SQL查询
        base_sql = """
        SELECT [code],[tradingday],[o],[h],[l],[c],[v],[amt],[adjfactor],
               [total_shares],[free_float_shares],[exchange_id] 
        FROM [stock_data].[dbo].[day5] 
        WHERE tradingday >= {begin_date} AND tradingday <= {end_date}
        """
        
        if stock_codes:
            # 将股票代码转换为SQL IN子句
            codes_str = "','".join(stock_codes)
            stock_filter = f" AND code IN ('{codes_str}')"
            sql = base_sql + stock_filter
        else:
            sql = base_sql
        
        sql = sql.format(begin_date=begin_date, end_date=end_date)
        sql += " ORDER BY tradingday, code"
        
        logger.debug(f"执行SQL查询: 日期 {begin_date}-{end_date}, 股票数 {len(stock_codes) if stock_codes else '全部'}")
        
        start_time = time.time()
        df = execute_stock_data_query(sql, db_name)
        end_time = time.time()
        
        logger.info(f"获取数据块完成: {df.shape}, 耗时 {end_time - start_time:.2f}秒")
        return df
    
    def fetch_price_data_chunked(self, 
                                begin_date: int = 20200101,
                                end_date: int = 0,
                                by_date: bool = True,
                                save_intermediate: bool = True) -> pd.DataFrame:
        """
        分块获取价格数据
        
        Args:
            begin_date: 开始日期
            end_date: 结束日期，0表示今天
            by_date: True表示按日期分块，False表示按股票分块
            save_intermediate: 是否保存中间结果
            
        Returns:
            完整的价格数据DataFrame
        """
        logger.info(f"开始分块获取价格数据: {begin_date}-{end_date}, 分块方式: {'按日期' if by_date else '按股票'}")
        
        all_data = []
        total_chunks = 0
        processed_chunks = 0
        
        if by_date:
            # 按日期分块
            date_chunks = self.get_date_range(begin_date, end_date)
            total_chunks = len(date_chunks)
            
            for i, (chunk_start, chunk_end) in enumerate(date_chunks):
                try:
                    logger.info(f"处理日期块 {i+1}/{total_chunks}: {chunk_start}-{chunk_end}")
                    
                    chunk_data = self.fetch_price_chunk(chunk_start, chunk_end)
                    
                    if not chunk_data.empty:
                        all_data.append(chunk_data)
                        processed_chunks += 1
                        
                        # 保存中间结果
                        if save_intermediate:
                            temp_file = os.path.join(self.data_root, f"temp_price_chunk_{i+1}.pkl")
                            chunk_data.to_pickle(temp_file)
                            logger.debug(f"保存中间结果: {temp_file}")
                    
                    # 进度报告
                    logger.info(f"进度: {i+1}/{total_chunks} ({(i+1)/total_chunks*100:.1f}%)")
                    
                except Exception as e:
                    logger.error(f"处理日期块 {chunk_start}-{chunk_end} 失败: {e}")
                    continue
        
        else:
            # 按股票分块
            stock_codes = self.get_stock_list()
            stock_chunks = self.get_stock_chunks(stock_codes)
            total_chunks = len(stock_chunks)
            
            for i, stock_chunk in enumerate(stock_chunks):
                try:
                    logger.info(f"处理股票块 {i+1}/{total_chunks}: {len(stock_chunk)} 只股票")
                    
                    chunk_data = self.fetch_price_chunk(begin_date, end_date, stock_chunk)
                    
                    if not chunk_data.empty:
                        all_data.append(chunk_data)
                        processed_chunks += 1
                        
                        # 保存中间结果
                        if save_intermediate:
                            temp_file = os.path.join(self.data_root, f"temp_price_chunk_{i+1}.pkl")
                            chunk_data.to_pickle(temp_file)
                            logger.debug(f"保存中间结果: {temp_file}")
                    
                    # 进度报告
                    logger.info(f"进度: {i+1}/{total_chunks} ({(i+1)/total_chunks*100:.1f}%)")
                    
                except Exception as e:
                    logger.error(f"处理股票块失败: {e}")
                    continue
        
        # 合并所有数据
        if all_data:
            logger.info(f"开始合并 {len(all_data)} 个数据块...")
            combined_data = pd.concat(all_data, ignore_index=True)
            
            # 去重和排序
            logger.info("去重和排序数据...")
            combined_data = combined_data.drop_duplicates(['code', 'tradingday'])
            combined_data = combined_data.sort_values(['tradingday', 'code'])
            
            logger.info(f"分块获取完成: 总形状 {combined_data.shape}, 处理了 {processed_chunks}/{total_chunks} 个块")
            
            # 清理临时文件
            if save_intermediate:
                self._cleanup_temp_files()
            
            return combined_data
        else:
            logger.warning("没有获取到任何数据")
            return pd.DataFrame()
    
    def _cleanup_temp_files(self):
        """清理临时文件"""
        try:
            temp_files = [f for f in os.listdir(self.data_root) if f.startswith('temp_price_chunk_')]
            for temp_file in temp_files:
                temp_path = os.path.join(self.data_root, temp_file)
                os.remove(temp_path)
            logger.info(f"清理了 {len(temp_files)} 个临时文件")
        except Exception as e:
            logger.warning(f"清理临时文件失败: {e}")
    
    def save_price_data(self, df: pd.DataFrame, filename: str = "Price.pkl") -> str:
        """
        保存价格数据，包含数据处理和增强字段计算
        
        Args:
            df: 价格数据DataFrame
            filename: 保存文件名
            
        Returns:
            保存的文件路径
        """
        if df.empty:
            logger.warning("数据为空，无法保存")
            return ""
        
        logger.info("开始处理和保存价格数据...")
        
        # 数据处理
        processed_df = df.copy()
        
        # 转换日期格式
        processed_df["tradingday"] = pd.to_datetime(processed_df["tradingday"], format="%Y%m%d")
        
        # 设置索引
        processed_df = processed_df.set_index(["tradingday", "code"])
        processed_df.index.set_names(["TradingDates", "StockCodes"], inplace=True)
        processed_df = processed_df.sort_index()
        
        # 计算增强字段
        logger.info("计算增强字段...")
        processed_df["MC"] = processed_df["total_shares"] * processed_df["c"]  # 总市值
        processed_df["FMC"] = processed_df["free_float_shares"] * processed_df["c"]  # 流通市值
        processed_df["turnoverrate"] = processed_df["v"] / processed_df["total_shares"]  # 换手率
        processed_df["vwap"] = processed_df["amt"] / processed_df["v"]  # 成交均价
        processed_df["freeturnoverrate"] = processed_df["v"] / processed_df["free_float_shares"]  # 流通换手率
        
        # 处理无穷大和NaN值
        processed_df = processed_df.replace([np.inf, -np.inf], np.nan)
        
        # 保存文件
        file_path = os.path.join(self.data_root, filename)
        processed_df.to_pickle(file_path)
        
        logger.info(f"价格数据已保存: {file_path}, 形状: {processed_df.shape}")
        return file_path


def test_chunked_fetcher():
    """测试分块获取功能"""
    logger.info("开始测试分块价格数据获取功能")
    
    # 创建获取器实例
    fetcher = ChunkedPriceFetcher(chunk_days=30, chunk_stocks=500)
    
    try:
        # 测试获取最近30天的数据
        end_date = int(datetime.now().strftime('%Y%m%d'))
        begin_date = int((datetime.now() - timedelta(days=30)).strftime('%Y%m%d'))
        
        logger.info(f"测试获取数据: {begin_date} - {end_date}")
        
        # 按日期分块获取
        data = fetcher.fetch_price_data_chunked(
            begin_date=begin_date, 
            end_date=end_date,
            by_date=True,
            save_intermediate=True
        )
        
        if not data.empty:
            logger.info(f"测试成功！获取数据形状: {data.shape}")
            logger.info(f"股票数量: {data['code'].nunique()}")
            logger.info(f"日期范围: {data['tradingday'].min()} - {data['tradingday'].max()}")
            
            # 保存测试结果
            file_path = fetcher.save_price_data(data, "Price_test.pkl")
            logger.info(f"测试数据已保存到: {file_path}")
            
            return True
        else:
            logger.error("测试失败：没有获取到数据")
            return False
            
    except Exception as e:
        logger.error(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 运行测试
    success = test_chunked_fetcher()
    if success:
        print("[OK] 分块获取功能测试通过")
    else:
        print("[FAIL] 分块获取功能测试失败")