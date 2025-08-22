#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
涨跌停价格数据增量更新器
定期更新StopPrice.pkl文件，添加最新的涨跌停价格数据
"""

import sys
import os

# 配置控制台编码（Windows兼容）
if sys.platform == 'win32':
    import codecs
    if hasattr(sys.stdout, 'encoding') and sys.stdout.encoding.lower() not in ['utf-8', 'utf8']:
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    if hasattr(sys.stderr, 'encoding') and sys.stderr.encoding.lower() not in ['utf-8', 'utf8']:
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict, Any
import shutil

from core.config_manager import get_path
from core.database import execute_stock_data_query
from .BasicDataLocalization import GetStockStopPrice

logger = logging.getLogger(__name__)


class IncrementalStopPriceUpdater:
    """涨跌停价格数据增量更新器"""
    
    def __init__(self):
        self.data_root = get_path('data_root')
        self.stop_price_file = os.path.join(self.data_root, "StopPrice.pkl")
        self.backup_dir = os.path.join(self.data_root, "backups")
        
        # 确保备份目录存在
        os.makedirs(self.backup_dir, exist_ok=True)
        
        logger.info("初始化涨跌停价格数据更新器")
        logger.info(f"涨跌停文件路径: {self.stop_price_file}")
        logger.info(f"备份目录: {self.backup_dir}")
    
    def get_latest_date_from_stop_price_file(self) -> Optional[datetime]:
        """
        从StopPrice.pkl文件获取最新的交易日期
        
        Returns:
            最新交易日期，如果文件不存在或为空则返回None
        """
        if not os.path.exists(self.stop_price_file):
            logger.warning("StopPrice.pkl文件不存在")
            return None
        
        try:
            logger.info("读取StopPrice.pkl文件以确定最新日期...")
            stop_price_data = pd.read_pickle(self.stop_price_file)
            
            if stop_price_data.empty:
                logger.warning("StopPrice.pkl文件为空")
                return None
            
            # 确保tradingday列是datetime格式
            if 'tradingday' in stop_price_data.columns:
                if not pd.api.types.is_datetime64_any_dtype(stop_price_data['tradingday']):
                    stop_price_data['tradingday'] = pd.to_datetime(stop_price_data['tradingday'], format='%Y%m%d')
                
                latest_date = stop_price_data['tradingday'].max()
                logger.info(f"本地涨跌停数据最新日期: {latest_date}")
                return latest_date
            else:
                logger.error("StopPrice.pkl文件缺少tradingday列")
                return None
                
        except Exception as e:
            logger.error(f"读取StopPrice.pkl文件失败: {e}")
            return None
    
    def get_latest_trading_date_from_db(self) -> Optional[datetime]:
        """
        从数据库获取最新的交易日期
        
        Returns:
            数据库中最新的交易日期
        """
        try:
            logger.info("查询数据库涨跌停数据最新日期...")
            sql = "SELECT MAX(tradingday) as latest_date FROM [stock_data].[dbo].[lgc_涨跌停板]"
            result = execute_stock_data_query(sql, db_name='database')
            
            if not result.empty and not pd.isna(result.iloc[0, 0]):
                latest_date = pd.to_datetime(str(result.iloc[0, 0]), format='%Y%m%d')
                logger.info(f"数据库涨跌停数据最新日期: {latest_date}")
                return latest_date
            else:
                logger.warning("数据库中没有涨跌停数据")
                return None
                
        except Exception as e:
            logger.error(f"查询数据库最新日期失败: {e}")
            return None
    
    def needs_update(self) -> bool:
        """
        检查是否需要更新涨跌停数据
        
        Returns:
            如果需要更新返回True，否则返回False
        """
        local_latest = self.get_latest_date_from_stop_price_file()
        db_latest = self.get_latest_trading_date_from_db()
        
        if db_latest is None:
            logger.warning("无法获取数据库最新日期，跳过更新")
            return False
        
        if local_latest is None:
            logger.info("本地文件不存在，需要全量更新")
            return True
        
        if local_latest < db_latest:
            gap_days = (db_latest - local_latest).days
            logger.info(f"发现新涨跌停数据：本地 {local_latest.date()} < 数据库 {db_latest.date()}，相差 {gap_days} 天")
            return True
        else:
            logger.info(f"涨跌停数据已是最新：本地 {local_latest.date()} >= 数据库 {db_latest.date()}")
            return False
    
    def get_incremental_data(self, start_date: datetime) -> pd.DataFrame:
        """
        获取从指定日期开始的增量涨跌停数据
        
        Args:
            start_date: 开始日期
            
        Returns:
            增量的涨跌停数据
        """
        try:
            start_date_int = int(start_date.strftime('%Y%m%d'))
            logger.info(f"获取增量涨跌停数据：从 {start_date_int}")
            
            sql = f"""
            SELECT DISTINCT [code],[tradingday],[high_limit],[low_limit] 
            FROM [stock_data].[dbo].[lgc_涨跌停板] 
            WHERE tradingday >= {start_date_int}
            ORDER BY tradingday, code
            """
            
            data = execute_stock_data_query(sql, db_name='database')
            
            if not data.empty:
                # 设置正确的列名
                data.columns = ["code", "tradingday", "high_limit", "low_limit"]
                
                # 转换日期格式
                data['tradingday'] = pd.to_datetime(data['tradingday'], format='%Y%m%d')
                
                logger.info(f"获取增量涨跌停数据完成：{data.shape}")
                return data
            else:
                logger.info("没有新的涨跌停数据")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"获取增量涨跌停数据失败: {e}")
            return pd.DataFrame()
    
    def backup_existing_file(self):
        """备份现有的涨跌停文件"""
        if os.path.exists(self.stop_price_file):
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_filename = f"StopPrice_backup_{timestamp}.pkl"
            backup_path = os.path.join(self.backup_dir, backup_filename)
            
            shutil.copy2(self.stop_price_file, backup_path)
            logger.info(f"文件已备份到: {backup_path}")
            return backup_path
        return None
    
    def clean_old_backups(self, keep_days: int = 7):
        """清理旧备份文件"""
        try:
            current_time = datetime.now()
            cleaned_count = 0
            
            for filename in os.listdir(self.backup_dir):
                if filename.startswith("StopPrice_backup_") and filename.endswith(".pkl"):
                    file_path = os.path.join(self.backup_dir, filename)
                    file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                    
                    if (current_time - file_time).days > keep_days:
                        os.remove(file_path)
                        cleaned_count += 1
            
            if cleaned_count > 0:
                logger.info(f"清理了 {cleaned_count} 个过期备份文件")
                
        except Exception as e:
            logger.warning(f"清理备份文件失败: {e}")
    
    def update_stop_price_file(self) -> bool:
        """
        更新涨跌停价格文件
        
        Returns:
            更新成功返回True，否则返回False
        """
        try:
            logger.info("开始更新StopPrice.pkl文件")
            
            # 检查是否需要更新
            if not self.needs_update():
                logger.info("涨跌停数据已是最新，无需更新")
                return True
            
            # 备份现有文件
            backup_path = self.backup_existing_file()
            
            local_latest = self.get_latest_date_from_stop_price_file()
            
            if local_latest is None:
                # 全量更新
                logger.info("执行全量更新...")
                try:
                    new_data = GetStockStopPrice()
                    
                    if not new_data.empty:
                        # 转换日期格式
                        if not pd.api.types.is_datetime64_any_dtype(new_data['tradingday']):
                            new_data['tradingday'] = pd.to_datetime(new_data['tradingday'], format='%Y%m%d')
                        
                        # 保存数据
                        new_data.to_pickle(self.stop_price_file)
                        
                        file_size = os.path.getsize(self.stop_price_file) / 1024 / 1024
                        logger.info(f"全量更新完成：文件大小 {file_size:.1f} MB，数据形状 {new_data.shape}")
                        
                        # 清理旧备份
                        self.clean_old_backups()
                        return True
                    else:
                        logger.error("获取的涨跌停数据为空")
                        return False
                        
                except Exception as e:
                    logger.error(f"全量更新失败: {e}")
                    return False
            else:
                # 增量更新
                logger.info("执行增量更新...")
                
                # 读取现有数据
                logger.info("读取现有StopPrice.pkl数据...")
                existing_data = pd.read_pickle(self.stop_price_file)
                logger.info(f"现有数据：{existing_data.shape}")
                
                # 获取增量数据
                next_date = local_latest + timedelta(days=1)
                incremental_data = self.get_incremental_data(next_date)
                
                if incremental_data.empty:
                    logger.info("没有新的涨跌停数据需要更新")
                    return True
                
                logger.info(f"增量数据获取完成：{incremental_data.shape}")
                
                # 合并数据
                logger.info("合并数据...")
                combined_data = pd.concat([existing_data, incremental_data], ignore_index=True)
                
                # 去重（基于code和tradingday）
                before_dedup = len(combined_data)
                combined_data = combined_data.drop_duplicates(subset=['code', 'tradingday'], keep='last')
                after_dedup = len(combined_data)
                
                # 排序
                combined_data = combined_data.sort_values(['tradingday', 'code']).reset_index(drop=True)
                
                logger.info(f"数据合并完成：原有 {len(existing_data):,} + 新增 {len(incremental_data):,} = {before_dedup:,}")
                logger.info(f"去重后：{after_dedup:,} 条记录 (去重 {before_dedup - after_dedup:,} 条)")
                
                # 保存更新后的数据
                logger.info("保存更新后的数据...")
                combined_data.to_pickle(self.stop_price_file)
                
                file_size = os.path.getsize(self.stop_price_file) / 1024 / 1024
                logger.info(f"增量更新完成：文件大小 {file_size:.1f} MB，数据形状 {combined_data.shape}")
                
                # 清理旧备份
                self.clean_old_backups()
                return True
                
        except Exception as e:
            logger.error(f"更新涨跌停价格文件失败: {e}")
            return False
    
    def get_update_info(self) -> Dict[str, Any]:
        """
        获取更新信息
        
        Returns:
            包含更新状态信息的字典
        """
        try:
            local_latest = self.get_latest_date_from_stop_price_file()
            db_latest = self.get_latest_trading_date_from_db()
            
            file_exists = os.path.exists(self.stop_price_file)
            file_size_mb = 0
            
            if file_exists:
                file_size_mb = os.path.getsize(self.stop_price_file) / 1024 / 1024
            
            return {
                'data_type': 'stop_price',
                'stop_price_file_exists': file_exists,
                'file_size_mb': round(file_size_mb, 1),
                'local_latest_date': local_latest.strftime('%Y-%m-%d') if local_latest else None,
                'db_latest_date': db_latest.strftime('%Y-%m-%d') if db_latest else None,
                'need_update': self.needs_update() if db_latest else False,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {
                'data_type': 'stop_price',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }


def main():
    """主函数，用于独立运行测试"""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    updater = IncrementalStopPriceUpdater()
    
    print("涨跌停价格数据更新器")
    print("=" * 50)
    
    # 获取状态信息
    info = updater.get_update_info()
    print("当前状态:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # 检查是否需要更新
    if updater.needs_update():
        print("\n开始更新涨跌停数据...")
        success = updater.update_stop_price_file()
        
        if success:
            print("✅ 涨跌停数据更新成功")
        else:
            print("❌ 涨跌停数据更新失败")
    else:
        print("\n✅ 涨跌停数据已是最新")


if __name__ == "__main__":
    main()