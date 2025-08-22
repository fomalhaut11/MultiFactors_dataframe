#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增量价格数据更新器
定期更新Price.pkl文件，添加最新的交易数据
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
from typing import Optional, Tuple
import shutil

from core.config_manager import get_path
from core.database import execute_stock_data_query
from .chunked_price_fetcher import ChunkedPriceFetcher

logger = logging.getLogger(__name__)


class IncrementalPriceUpdater:
    """增量价格数据更新器"""
    
    def __init__(self):
        self.data_root = get_path('data_root')
        self.price_file = os.path.join(self.data_root, "Price.pkl")
        self.backup_dir = os.path.join(self.data_root, "backups")
        self.fetcher = ChunkedPriceFetcher(chunk_days=30, chunk_stocks=1000)
        
        # 确保备份目录存在
        os.makedirs(self.backup_dir, exist_ok=True)
        
        logger.info("初始化增量价格数据更新器")
        logger.info(f"价格文件路径: {self.price_file}")
        logger.info(f"备份目录: {self.backup_dir}")
    
    def get_latest_date_from_price_file(self) -> Optional[datetime]:
        """
        从Price.pkl文件获取最新的交易日期
        
        Returns:
            最新交易日期，如果文件不存在或为空则返回None
        """
        if not os.path.exists(self.price_file):
            logger.warning("Price.pkl文件不存在")
            return None
        
        try:
            logger.info("读取Price.pkl文件以确定最新日期...")
            
            # 只读取索引信息，不读取全部数据以节省内存
            with pd.HDFStore(self.price_file.replace('.pkl', '.h5'), mode='r') as store:
                if 'price_data' in store:
                    dates = store.select('price_data', columns=[], start=0, stop=1).index.get_level_values('TradingDates')
                else:
                    # 如果是pkl格式，需要读取文件
                    price_data = pd.read_pickle(self.price_file)
                    if price_data.empty:
                        logger.warning("Price.pkl文件为空")
                        return None
                    
                    dates = price_data.index.get_level_values('TradingDates')
            
            latest_date = dates.max()
            logger.info(f"本地数据最新日期: {latest_date}")
            return latest_date
            
        except Exception as e:
            # 回退到直接读取pkl文件
            try:
                logger.info("尝试直接读取pkl文件...")
                price_data = pd.read_pickle(self.price_file)
                
                if price_data.empty:
                    logger.warning("Price.pkl文件为空")
                    return None
                
                latest_date = price_data.index.get_level_values('TradingDates').max()
                logger.info(f"本地数据最新日期: {latest_date}")
                return latest_date
                
            except Exception as e2:
                logger.error(f"读取Price.pkl文件失败: {e2}")
                return None
    
    def get_latest_trading_date_from_db(self) -> Optional[datetime]:
        """
        从数据库获取最新的交易日期
        
        Returns:
            数据库中最新的交易日期
        """
        try:
            logger.info("查询数据库最新交易日期...")
            sql = "SELECT MAX(tradingday) FROM [stock_data].[dbo].[day5]"
            result = execute_stock_data_query(sql, db_name='database')
            
            if result.empty:
                logger.warning("数据库中没有交易数据")
                return None
            
            latest_date_int = result.iloc[0, 0]
            latest_date = pd.to_datetime(str(latest_date_int), format='%Y%m%d')
            
            logger.info(f"数据库最新日期: {latest_date}")
            return latest_date
            
        except Exception as e:
            logger.error(f"查询数据库最新日期失败: {e}")
            return None
    
    def need_update(self) -> Tuple[bool, Optional[datetime], Optional[datetime]]:
        """
        检查是否需要更新
        
        Returns:
            (是否需要更新, 本地最新日期, 数据库最新日期)
        """
        local_latest = self.get_latest_date_from_price_file()
        db_latest = self.get_latest_trading_date_from_db()
        
        if local_latest is None:
            logger.info("本地文件不存在或为空，需要全量更新")
            return True, local_latest, db_latest
        
        if db_latest is None:
            logger.warning("数据库中没有数据，无需更新")
            return False, local_latest, db_latest
        
        # 比较日期（只比较日期部分，不比较时间）
        local_date = local_latest.date()
        db_date = db_latest.date()
        
        if db_date > local_date:
            logger.info(f"发现新数据：本地 {local_date} < 数据库 {db_date}")
            return True, local_latest, db_latest
        else:
            logger.info(f"数据已是最新：本地 {local_date} >= 数据库 {db_date}")
            return False, local_latest, db_latest
    
    def needs_update(self) -> bool:
        """
        检查是否需要更新
        
        Returns:
            是否需要更新
        """
        need_update, _, _ = self.need_update()
        return need_update
    
    def backup_price_file(self) -> str:
        """
        备份现有的Price.pkl文件
        
        Returns:
            备份文件路径
        """
        if not os.path.exists(self.price_file):
            logger.info("没有现有文件需要备份")
            return ""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_file = os.path.join(self.backup_dir, f"Price_backup_{timestamp}.pkl")
        
        try:
            shutil.copy2(self.price_file, backup_file)
            logger.info(f"文件已备份到: {backup_file}")
            return backup_file
        except Exception as e:
            logger.error(f"备份文件失败: {e}")
            return ""
    
    def fetch_incremental_data(self, start_date: datetime, end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        获取增量数据
        
        Args:
            start_date: 开始日期（不包含）
            end_date: 结束日期（包含），None表示到今天
            
        Returns:
            增量数据DataFrame
        """
        # 计算实际的开始日期（下一个交易日）
        begin_date = int((start_date + timedelta(days=1)).strftime('%Y%m%d'))
        
        if end_date is None:
            end_date_int = 0  # 表示到今天
        else:
            end_date_int = int(end_date.strftime('%Y%m%d'))
        
        logger.info(f"获取增量数据：{begin_date} - {end_date_int if end_date_int != 0 else '今天'}")
        
        try:
            # 使用分块获取器获取增量数据
            incremental_data = self.fetcher.fetch_price_data_chunked(
                begin_date=begin_date,
                end_date=end_date_int,
                by_date=True,
                save_intermediate=False  # 增量更新不需要中间文件
            )
            
            logger.info(f"增量数据获取完成：{incremental_data.shape}")
            return incremental_data
            
        except Exception as e:
            logger.error(f"获取增量数据失败: {e}")
            raise
    
    def merge_data(self, existing_data: pd.DataFrame, new_data: pd.DataFrame) -> pd.DataFrame:
        """
        合并现有数据和新数据
        
        Args:
            existing_data: 现有数据
            new_data: 新数据
            
        Returns:
            合并后的数据
        """
        if new_data.empty:
            logger.info("新数据为空，无需合并")
            return existing_data
        
        if existing_data.empty:
            logger.info("现有数据为空，直接使用新数据")
            # 需要对新数据进行格式处理
            return self._format_new_data(new_data)
        
        logger.info("开始合并数据...")
        
        # 确保新数据格式正确
        formatted_new_data = self._format_new_data(new_data)
        
        # 合并数据
        try:
            combined_data = pd.concat([existing_data, formatted_new_data])
            
            # 去重（保留最新的记录）
            original_count = len(combined_data)
            combined_data = combined_data[~combined_data.index.duplicated(keep='last')]
            dedupe_count = len(combined_data)
            
            # 排序
            combined_data = combined_data.sort_index()
            
            logger.info(f"数据合并完成：原有 {len(existing_data):,} + 新增 {len(formatted_new_data):,} = {original_count:,}")
            logger.info(f"去重后：{dedupe_count:,} 条记录 (去重 {original_count - dedupe_count} 条)")
            
            return combined_data
            
        except Exception as e:
            logger.error(f"数据合并失败: {e}")
            raise
    
    def _format_new_data(self, new_data: pd.DataFrame) -> pd.DataFrame:
        """
        格式化新数据，使其与现有Price.pkl格式一致
        
        Args:
            new_data: 原始新数据
            
        Returns:
            格式化后的数据
        """
        if new_data.empty:
            return new_data
        
        formatted_data = new_data.copy()
        
        # 转换日期格式
        if 'tradingday' in formatted_data.columns:
            formatted_data["tradingday"] = pd.to_datetime(formatted_data["tradingday"], format="%Y%m%d")
        
        # 设置索引
        if not isinstance(formatted_data.index, pd.MultiIndex):
            formatted_data = formatted_data.set_index(["tradingday", "code"])
            formatted_data.index.set_names(["TradingDates", "StockCodes"], inplace=True)
        
        # 确保索引名称正确
        if formatted_data.index.names != ["TradingDates", "StockCodes"]:
            formatted_data.index.set_names(["TradingDates", "StockCodes"], inplace=True)
        
        # 计算增强字段（如果不存在）
        if "MC" not in formatted_data.columns:
            formatted_data["MC"] = formatted_data["total_shares"] * formatted_data["c"]
        if "FMC" not in formatted_data.columns:
            formatted_data["FMC"] = formatted_data["free_float_shares"] * formatted_data["c"]
        if "turnoverrate" not in formatted_data.columns:
            formatted_data["turnoverrate"] = formatted_data["v"] / formatted_data["total_shares"]
        if "vwap" not in formatted_data.columns:
            formatted_data["vwap"] = formatted_data["amt"] / formatted_data["v"]
        if "freeturnoverrate" not in formatted_data.columns:
            formatted_data["freeturnoverrate"] = formatted_data["v"] / formatted_data["free_float_shares"]
        
        # 处理无穷大和NaN值
        formatted_data = formatted_data.replace([np.inf, -np.inf], np.nan)
        
        return formatted_data
    
    def update_price_file(self) -> bool:
        """
        执行增量更新
        
        Returns:
            是否更新成功
        """
        logger.info("开始增量更新Price.pkl文件")
        
        try:
            # 检查是否需要更新
            need_update, local_latest, db_latest = self.need_update()
            
            if not need_update:
                logger.info("数据已是最新，无需更新")
                return True
            
            # 备份现有文件
            backup_file = self.backup_price_file()
            
            try:
                # 读取现有数据
                if os.path.exists(self.price_file):
                    logger.info("读取现有Price.pkl数据...")
                    existing_data = pd.read_pickle(self.price_file)
                    logger.info(f"现有数据：{existing_data.shape}")
                else:
                    logger.info("没有现有数据，将创建新文件")
                    existing_data = pd.DataFrame()
                
                # 获取增量数据
                logger.info("获取增量数据...")
                new_data = self.fetch_incremental_data(local_latest or datetime(2014, 1, 1), db_latest)
                
                if new_data.empty:
                    logger.info("没有新数据需要更新")
                    return True
                
                # 合并数据
                logger.info("合并数据...")
                combined_data = self.merge_data(existing_data, new_data)
                
                # 保存更新后的数据
                logger.info("保存更新后的数据...")
                combined_data.to_pickle(self.price_file)
                
                # 验证保存结果
                file_size = os.path.getsize(self.price_file) / 1024 / 1024
                logger.info(f"更新完成：文件大小 {file_size:.1f} MB，数据形状 {combined_data.shape}")
                
                return True
                
            except Exception as e:
                # 如果更新失败，尝试恢复备份
                logger.error(f"更新失败: {e}")
                if backup_file and os.path.exists(backup_file):
                    logger.info("尝试恢复备份文件...")
                    try:
                        shutil.copy2(backup_file, self.price_file)
                        logger.info("备份文件已恢复")
                    except:
                        logger.error("恢复备份文件失败")
                raise
                
        except Exception as e:
            logger.error(f"增量更新失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def clean_old_backups(self, keep_days: int = 7):
        """
        清理旧的备份文件
        
        Args:
            keep_days: 保留的天数
        """
        if not os.path.exists(self.backup_dir):
            return
        
        cutoff_date = datetime.now() - timedelta(days=keep_days)
        
        try:
            backup_files = [f for f in os.listdir(self.backup_dir) if f.startswith('Price_backup_')]
            removed_count = 0
            
            for backup_file in backup_files:
                file_path = os.path.join(self.backup_dir, backup_file)
                file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                
                if file_mtime < cutoff_date:
                    os.remove(file_path)
                    removed_count += 1
            
            if removed_count > 0:
                logger.info(f"清理了 {removed_count} 个过期备份文件")
                
        except Exception as e:
            logger.warning(f"清理备份文件失败: {e}")
    
    def get_update_info(self) -> dict:
        """
        获取更新状态信息
        
        Returns:
            更新状态字典
        """
        need_update, local_latest, db_latest = self.need_update()
        
        # 计算落后天数
        days_behind = 0
        if local_latest and db_latest:
            local_date = local_latest.date()
            db_date = db_latest.date()
            days_behind = max(0, (db_date - local_date).days)
        
        info = {
            'price_file_exists': os.path.exists(self.price_file),
            'local_latest_date': local_latest.strftime('%Y-%m-%d') if local_latest else None,
            'db_latest_date': db_latest.strftime('%Y-%m-%d') if db_latest else None,
            'need_update': need_update,
            'days_behind': days_behind,
            'file_size_mb': os.path.getsize(self.price_file) / 1024 / 1024 if os.path.exists(self.price_file) else 0
        }
        
        return info


def main():
    """主函数 - 执行增量更新"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 60)
    print("Price.pkl 增量更新器")
    print("=" * 60)
    
    updater = IncrementalPriceUpdater()
    
    # 显示当前状态
    info = updater.get_update_info()
    print(f"Price.pkl文件存在: {'是' if info['price_file_exists'] else '否'}")
    print(f"本地最新日期: {info['local_latest_date'] or 'N/A'}")
    print(f"数据库最新日期: {info['db_latest_date'] or 'N/A'}")
    print(f"需要更新: {'是' if info['need_update'] else '否'}")
    print(f"当前文件大小: {info['file_size_mb']:.1f} MB")
    print("=" * 60)
    
    if info['need_update']:
        confirm = input("发现新数据，是否执行更新？(Y/n): ").strip().lower()
        if confirm != 'n':
            print("开始增量更新...")
            start_time = datetime.now()
            
            success = updater.update_price_file()
            
            end_time = datetime.now()
            duration = end_time - start_time
            
            if success:
                print(f"\n[OK] 增量更新完成！耗时: {duration}")
                
                # 清理旧备份
                updater.clean_old_backups()
            else:
                print(f"\n[FAIL] 增量更新失败！")
        else:
            print("更新已取消")
    else:
        print("数据已是最新，无需更新")


if __name__ == "__main__":
    main()