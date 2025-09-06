#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
财务数据增量更新器
基于 local_date 字段实现财务三张表(fzb、xjlb、lrb)的增量更新
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
from typing import Optional, Tuple, Dict, Any, List
import shutil
from pathlib import Path

from config import get_config
from core.database import execute_stock_data_query, get_db_table_config

logger = logging.getLogger(__name__)


class IncrementalFinancialUpdater:
    """财务数据增量更新器"""
    
    def __init__(self):
        self.data_root = get_config('main.paths.data_root')
        self.backup_dir = os.path.join(self.data_root, "backups")
        # 获取数据库表名配置
        self.db_config = get_db_table_config()
        
        # 财务数据表配置
        self.tables = {
            'fzb': {
                'local_file': 'fzb.pkl',
                'db_table': self.db_config.fzb_table,
                'description': '资产负债表'
            },
            'xjlb': {
                'local_file': 'xjlb.pkl', 
                'db_table': self.db_config.xjlb_table,
                'description': '现金流量表'
            },
            'lrb': {
                'local_file': 'lrb.pkl',
                'db_table': self.db_config.lrb_table,
                'description': '利润表'
            }
        }
        
        # 确保备份目录存在
        os.makedirs(self.backup_dir, exist_ok=True)
        
        logger.info("初始化财务数据增量更新器")
        logger.info(f"数据目录: {self.data_root}")
        logger.info(f"备份目录: {self.backup_dir}")
        logger.info(f"管理表: {list(self.tables.keys())}")
    
    def get_latest_local_date_from_file(self, table_name: str) -> Optional[int]:
        """
        从本地pkl文件获取最新的local_date
        
        Args:
            table_name: 表名 (fzb/xjlb/lrb)
            
        Returns:
            最新的local_date，如果文件不存在或为空则返回None
        """
        if table_name not in self.tables:
            logger.error(f"未知的表名: {table_name}")
            return None
            
        file_path = os.path.join(self.data_root, self.tables[table_name]['local_file'])
        
        if not os.path.exists(file_path):
            logger.warning(f"{self.tables[table_name]['description']}文件不存在: {file_path}")
            return None
        
        try:
            logger.info(f"读取{self.tables[table_name]['description']}文件获取最新local_date...")
            df = pd.read_pickle(file_path)
            
            if df.empty or 'local_date' not in df.columns:
                logger.warning(f"{self.tables[table_name]['description']}文件为空或缺少local_date字段")
                return None
            
            # 获取最新的local_date
            latest_date = int(df['local_date'].max())
            logger.info(f"本地{self.tables[table_name]['description']}最新录入时间: {latest_date}")
            return latest_date
            
        except Exception as e:
            logger.error(f"读取{self.tables[table_name]['description']}文件失败: {e}")
            return None
    
    def get_latest_local_date_from_db(self, table_name: str) -> Optional[int]:
        """
        从数据库获取最新的local_date
        
        Args:
            table_name: 表名 (fzb/xjlb/lrb)
            
        Returns:
            数据库中最新的local_date
        """
        if table_name not in self.tables:
            logger.error(f"未知的表名: {table_name}")
            return None
            
        try:
            logger.info(f"查询数据库{self.tables[table_name]['description']}最新录入时间...")
            sql = f"SELECT MAX(local_date) as latest_date FROM {self.tables[table_name]['db_table']}"
            result = execute_stock_data_query(sql, db_name='database')
            
            if not result.empty and not pd.isna(result.iloc[0, 0]):
                latest_date = int(result.iloc[0, 0])
                logger.info(f"数据库{self.tables[table_name]['description']}最新录入时间: {latest_date}")
                return latest_date
            else:
                logger.warning(f"数据库中没有{self.tables[table_name]['description']}数据")
                return None
                
        except Exception as e:
            logger.error(f"查询数据库{self.tables[table_name]['description']}最新时间失败: {e}")
            return None
    
    def needs_update(self, table_name: str = None) -> Dict[str, bool]:
        """
        检查是否需要更新财务数据
        
        Args:
            table_name: 指定表名，None表示检查所有表
            
        Returns:
            字典，包含各表是否需要更新的信息
        """
        tables_to_check = [table_name] if table_name else list(self.tables.keys())
        update_status = {}
        
        for table in tables_to_check:
            if table not in self.tables:
                continue
                
            local_latest = self.get_latest_local_date_from_file(table)
            db_latest = self.get_latest_local_date_from_db(table)
            
            if db_latest is None:
                logger.warning(f"无法获取{table}数据库最新时间，跳过更新")
                update_status[table] = False
                continue
            
            if local_latest is None:
                logger.info(f"{table}本地文件不存在，需要全量更新")
                update_status[table] = True
                continue
            
            if local_latest < db_latest:
                logger.info(f"{table}发现新数据：本地 {local_latest} < 数据库 {db_latest}")
                update_status[table] = True
            else:
                logger.info(f"{table}数据已是最新：本地 {local_latest} >= 数据库 {db_latest}")
                update_status[table] = False
        
        return update_status
    
    def get_incremental_data(self, table_name: str, start_local_date: int) -> pd.DataFrame:
        """
        获取从指定local_date开始的增量财务数据
        
        Args:
            table_name: 表名
            start_local_date: 开始的local_date
            
        Returns:
            增量的财务数据
        """
        if table_name not in self.tables:
            logger.error(f"未知的表名: {table_name}")
            return pd.DataFrame()
            
        try:
            logger.info(f"获取{self.tables[table_name]['description']}增量数据：从 local_date {start_local_date}")
            
            sql = f"""
            SELECT * FROM {self.tables[table_name]['db_table']} 
            WHERE local_date > {start_local_date}
            ORDER BY local_date, code, reportday
            """
            
            data = execute_stock_data_query(sql, db_name='database')
            
            if not data.empty:
                logger.info(f"{self.tables[table_name]['description']}增量数据获取完成：{data.shape}")
                
                # 数据类型转换
                if 'reportday' in data.columns:
                    data['reportday'] = pd.to_datetime(data['reportday'], format='%Y%m%d')
                if 'tradingday' in data.columns:
                    data['tradingday'] = pd.to_datetime(data['tradingday'], format='%Y%m%d')
                    
                return data
            else:
                logger.info(f"没有新的{self.tables[table_name]['description']}数据")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"获取{self.tables[table_name]['description']}增量数据失败: {e}")
            return pd.DataFrame()
    
    def get_full_data(self, table_name: str) -> pd.DataFrame:
        """
        获取指定表的全量数据
        
        Args:
            table_name: 表名
            
        Returns:
            全量财务数据
        """
        if table_name not in self.tables:
            logger.error(f"未知的表名: {table_name}")
            return pd.DataFrame()
            
        try:
            logger.info(f"获取{self.tables[table_name]['description']}全量数据...")
            
            sql = f"SELECT * FROM {self.tables[table_name]['db_table']} ORDER BY local_date, code, reportday"
            data = execute_stock_data_query(sql, db_name='database')
            
            if not data.empty:
                logger.info(f"{self.tables[table_name]['description']}全量数据获取完成：{data.shape}")
                
                # 数据类型转换
                if 'reportday' in data.columns:
                    data['reportday'] = pd.to_datetime(data['reportday'], format='%Y%m%d')
                if 'tradingday' in data.columns:
                    data['tradingday'] = pd.to_datetime(data['tradingday'], format='%Y%m%d')
                    
                return data
            else:
                logger.warning(f"数据库中没有{self.tables[table_name]['description']}数据")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"获取{self.tables[table_name]['description']}全量数据失败: {e}")
            return pd.DataFrame()
    
    def backup_existing_file(self, table_name: str) -> Optional[str]:
        """备份现有的财务文件"""
        if table_name not in self.tables:
            return None
            
        file_path = os.path.join(self.data_root, self.tables[table_name]['local_file'])
        
        if os.path.exists(file_path):
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_filename = f"{table_name}_backup_{timestamp}.pkl"
            backup_path = os.path.join(self.backup_dir, backup_filename)
            
            shutil.copy2(file_path, backup_path)
            logger.info(f"{self.tables[table_name]['description']}已备份到: {backup_path}")
            return backup_path
        return None
    
    def update_single_table(self, table_name: str) -> bool:
        """
        更新单个财务表
        
        Args:
            table_name: 表名
            
        Returns:
            更新成功返回True，否则返回False
        """
        if table_name not in self.tables:
            logger.error(f"未知的表名: {table_name}")
            return False
            
        try:
            logger.info(f"开始更新{self.tables[table_name]['description']}")
            
            # 检查是否需要更新
            update_status = self.needs_update(table_name)
            if not update_status.get(table_name, False):
                logger.info(f"{self.tables[table_name]['description']}已是最新，无需更新")
                return True
            
            # 备份现有文件
            backup_path = self.backup_existing_file(table_name)
            
            local_latest = self.get_latest_local_date_from_file(table_name)
            file_path = os.path.join(self.data_root, self.tables[table_name]['local_file'])
            
            if local_latest is None:
                # 全量更新
                logger.info(f"执行{self.tables[table_name]['description']}全量更新...")
                new_data = self.get_full_data(table_name)
                
                if not new_data.empty:
                    # 保存数据
                    new_data.to_pickle(file_path)
                    
                    file_size = os.path.getsize(file_path) / 1024 / 1024
                    logger.info(f"{self.tables[table_name]['description']}全量更新完成：文件大小 {file_size:.1f} MB，数据形状 {new_data.shape}")
                    return True
                else:
                    logger.error(f"获取的{self.tables[table_name]['description']}数据为空")
                    return False
            else:
                # 增量更新
                logger.info(f"执行{self.tables[table_name]['description']}增量更新...")
                
                # 读取现有数据
                logger.info(f"读取现有{self.tables[table_name]['description']}数据...")
                existing_data = pd.read_pickle(file_path)
                logger.info(f"现有数据：{existing_data.shape}")
                
                # 获取增量数据
                incremental_data = self.get_incremental_data(table_name, local_latest)
                
                if incremental_data.empty:
                    logger.info(f"没有新的{self.tables[table_name]['description']}数据需要更新")
                    return True
                
                logger.info(f"{self.tables[table_name]['description']}增量数据获取完成：{incremental_data.shape}")
                
                # 合并数据
                logger.info(f"合并{self.tables[table_name]['description']}数据...")
                combined_data = pd.concat([existing_data, incremental_data], ignore_index=True)
                
                # 去重（基于code、reportday、tradingday、d_year、d_quarter）
                dedup_cols = ['code', 'reportday', 'd_year', 'd_quarter']
                if 'tradingday' in combined_data.columns:
                    dedup_cols.append('tradingday')
                    
                before_dedup = len(combined_data)
                combined_data = combined_data.drop_duplicates(subset=dedup_cols, keep='last')
                after_dedup = len(combined_data)
                
                # 按local_date排序
                combined_data = combined_data.sort_values(['local_date', 'code', 'reportday']).reset_index(drop=True)
                
                logger.info(f"{self.tables[table_name]['description']}数据合并完成：原有 {len(existing_data):,} + 新增 {len(incremental_data):,} = {before_dedup:,}")
                logger.info(f"去重后：{after_dedup:,} 条记录 (去重 {before_dedup - after_dedup:,} 条)")
                
                # 保存更新后的数据
                logger.info(f"保存更新后的{self.tables[table_name]['description']}数据...")
                combined_data.to_pickle(file_path)
                
                file_size = os.path.getsize(file_path) / 1024 / 1024
                logger.info(f"{self.tables[table_name]['description']}增量更新完成：文件大小 {file_size:.1f} MB，数据形状 {combined_data.shape}")
                
                return True
                
        except Exception as e:
            logger.error(f"更新{self.tables[table_name]['description']}失败: {e}")
            return False
    
    def update_all_tables(self) -> Dict[str, bool]:
        """
        更新所有财务表
        
        Returns:
            字典，包含各表更新结果
        """
        logger.info("开始更新所有财务数据表")
        results = {}
        
        for table_name in self.tables.keys():
            logger.info(f"\n{'='*50}")
            logger.info(f"更新 {self.tables[table_name]['description']} ({table_name})")
            logger.info(f"{'='*50}")
            
            success = self.update_single_table(table_name)
            results[table_name] = success
            
            if success:
                logger.info(f"✅ {self.tables[table_name]['description']}更新成功")
            else:
                logger.error(f"❌ {self.tables[table_name]['description']}更新失败")
        
        # 汇总结果
        success_count = sum(results.values())
        total_count = len(results)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"财务数据更新完成：{success_count}/{total_count} 个表更新成功")
        logger.info(f"{'='*60}")
        
        return results
    
    def clean_old_backups(self, keep_days: int = 7):
        """清理旧备份文件"""
        try:
            current_time = datetime.now()
            cleaned_count = 0
            
            for filename in os.listdir(self.backup_dir):
                if any(filename.startswith(f"{table}_backup_") for table in self.tables.keys()) and filename.endswith(".pkl"):
                    file_path = os.path.join(self.backup_dir, filename)
                    file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                    
                    if (current_time - file_time).days > keep_days:
                        os.remove(file_path)
                        cleaned_count += 1
            
            if cleaned_count > 0:
                logger.info(f"清理了 {cleaned_count} 个过期的财务数据备份文件")
                
        except Exception as e:
            logger.warning(f"清理财务数据备份文件失败: {e}")
    
    def get_update_info(self, table_name: str = None) -> Dict[str, Any]:
        """
        获取更新信息
        
        Args:
            table_name: 指定表名，None表示获取所有表信息
            
        Returns:
            包含更新状态信息的字典
        """
        try:
            tables_to_check = [table_name] if table_name else list(self.tables.keys())
            update_info = {
                'data_type': 'financial_data',
                'timestamp': datetime.now().isoformat(),
                'tables': {}
            }
            
            for table in tables_to_check:
                if table not in self.tables:
                    continue
                    
                local_latest = self.get_latest_local_date_from_file(table)
                db_latest = self.get_latest_local_date_from_db(table)
                file_path = os.path.join(self.data_root, self.tables[table]['local_file'])
                
                file_exists = os.path.exists(file_path)
                file_size_mb = 0
                if file_exists:
                    file_size_mb = os.path.getsize(file_path) / 1024 / 1024
                
                table_info = {
                    'description': self.tables[table]['description'],
                    'file_exists': file_exists,
                    'file_size_mb': round(file_size_mb, 1),
                    'local_latest_date': local_latest,
                    'db_latest_date': db_latest,
                    'need_update': local_latest is None or (db_latest is not None and local_latest < db_latest)
                }
                
                update_info['tables'][table] = table_info
            
            return update_info
            
        except Exception as e:
            return {
                'data_type': 'financial_data',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }


def main():
    """主函数，用于独立运行测试"""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    updater = IncrementalFinancialUpdater()
    
    print("财务数据增量更新器")
    print("=" * 50)
    
    # 获取状态信息
    info = updater.get_update_info()
    print("当前状态:")
    
    if 'tables' in info:
        for table_name, table_info in info['tables'].items():
            print(f"\n{table_info['description']} ({table_name}):")
            print(f"  文件存在: {table_info['file_exists']}")
            print(f"  文件大小: {table_info['file_size_mb']} MB")
            print(f"  本地最新: {table_info['local_latest_date']}")
            print(f"  数据库最新: {table_info['db_latest_date']}")
            print(f"  需要更新: {table_info['need_update']}")
    
    # 检查是否需要更新
    update_status = updater.needs_update()
    need_update = any(update_status.values())
    
    if need_update:
        print(f"\n发现需要更新的表: {[k for k, v in update_status.items() if v]}")
        
        # 询问是否执行更新
        try:
            choice = input("\n是否执行更新? (y/N): ").strip().lower()
            if choice == 'y':
                print("\n开始更新财务数据...")
                results = updater.update_all_tables()
                
                success_count = sum(results.values())
                total_count = len(results)
                
                if success_count == total_count:
                    print("✅ 所有财务数据更新成功")
                else:
                    print(f"⚠️  部分财务数据更新失败: {success_count}/{total_count}")
                    
                # 清理旧备份
                updater.clean_old_backups()
            else:
                print("取消更新")
        except KeyboardInterrupt:
            print("\n用户取消操作")
    else:
        print("\n✅ 所有财务数据已是最新")


if __name__ == "__main__":
    main()