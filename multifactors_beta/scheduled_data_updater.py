#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
定时数据更新脚本
支持多种数据类型的定时更新，包括价格数据、财务数据、行业数据等
用于自动化每日数据更新，可配置到Windows任务计划程序或cron
  # 健康检查
  python scheduled_data_updater.py --data-type financial --health-check

  # 正常更新
  python scheduled_data_updater.py --data-type financial

  # 强制更新
  python scheduled_data_updater.py --data-type financial --force

  # 更新所有数据类型
  python scheduled_data_updater.py --data-type all
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

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import logging
from datetime import datetime, time
from typing import Dict, List, Optional
from pathlib import Path
import pandas as pd
from data.fetcher.incremental_price_updater import IncrementalPriceUpdater
from data.fetcher.incremental_stop_price_updater import IncrementalStopPriceUpdater
from data.fetcher.incremental_financial_updater import IncrementalFinancialUpdater
from data.fetcher.data_fetcher import StockDataFetcher, MarketDataFetcher
from config import get_config
from core.data_registry import get_data_registry

# 配置日志到文件
log_dir = os.path.join(get_config('main.paths.data_root'), 'logs')
os.makedirs(log_dir, exist_ok=True)

log_file = os.path.join(log_dir, f"data_update_{datetime.now().strftime('%Y%m%d')}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


class DataUpdateResult:
    """数据更新结果"""
    
    def __init__(self, data_type: str, success: bool, message: str = "", 
                 duration: Optional[float] = None, details: Optional[Dict] = None):
        self.data_type = data_type
        self.success = success
        self.message = message
        self.duration = duration
        self.details = details or {}
        self.timestamp = datetime.now()


class BaseDataUpdater:
    """数据更新器基类"""
    
    def __init__(self, data_type: str):
        self.data_type = data_type
    
    def get_update_info(self) -> Dict:
        """获取更新信息"""
        raise NotImplementedError("子类必须实现get_update_info方法")
    
    def needs_update(self) -> bool:
        """检查是否需要更新"""
        raise NotImplementedError("子类必须实现needs_update方法")
    
    def update_data(self) -> bool:
        """执行数据更新"""
        raise NotImplementedError("子类必须实现update_data方法")
    
    def get_health_status(self) -> Dict:
        """获取健康状态"""
        raise NotImplementedError("子类必须实现get_health_status方法")


class PriceDataUpdater(BaseDataUpdater):
    """价格数据更新器"""
    
    def __init__(self):
        super().__init__("price_data")
        self.updater = IncrementalPriceUpdater()
    
    def get_update_info(self) -> Dict:
        """获取更新信息"""
        return self.updater.get_update_info()
    
    def needs_update(self) -> bool:
        """检查是否需要更新"""
        return self.updater.needs_update()
    
    def update_data(self) -> bool:
        """执行数据更新"""
        return self.updater.update_price_file()
    
    def get_health_status(self) -> Dict:
        """获取健康状态"""
        try:
            info = self.get_update_info()
            
            health_status = {
                'data_type': self.data_type,
                'timestamp': datetime.now().isoformat(),
                'file_exists': info['price_file_exists'],
                'file_size_mb': info['file_size_mb'],
                'local_latest_date': info['local_latest_date'],
                'db_latest_date': info['db_latest_date'],
                'need_update': info['need_update'],
                'status': 'healthy' if info['price_file_exists'] else 'warning'
            }
            
            # 检查数据是否过期
            if info['local_latest_date'] and info['db_latest_date']:
                local_date = datetime.strptime(info['local_latest_date'], '%Y-%m-%d')
                db_date = datetime.strptime(info['db_latest_date'], '%Y-%m-%d')
                gap_days = (db_date - local_date).days
                
                health_status['data_gap_days'] = gap_days
                
                if gap_days > 5:
                    health_status['status'] = 'error'
                    health_status['message'] = f'价格数据过期 {gap_days} 天'
                elif gap_days > 2:
                    health_status['status'] = 'warning'
                    health_status['message'] = f'价格数据落后 {gap_days} 天'
                else:
                    health_status['message'] = '价格数据正常'
            
            return health_status
            
        except Exception as e:
            return {
                'data_type': self.data_type,
                'timestamp': datetime.now().isoformat(),
                'status': 'error',
                'message': f'价格数据健康检查失败: {e}'
            }
    
    def clean_old_backups(self, keep_days: int = 3):
        """清理旧备份"""
        self.updater.clean_old_backups(keep_days)


class StopPriceDataUpdater(BaseDataUpdater):
    """涨跌停价格数据更新器"""
    
    def __init__(self):
        super().__init__("stop_price_data")
        self.updater = IncrementalStopPriceUpdater()
    
    def get_update_info(self) -> Dict:
        """获取更新信息"""
        return self.updater.get_update_info()
    
    def needs_update(self) -> bool:
        """检查是否需要更新"""
        return self.updater.needs_update()
    
    def update_data(self) -> bool:
        """执行数据更新"""
        return self.updater.update_stop_price_file()
    
    def get_health_status(self) -> Dict:
        """获取健康状态"""
        try:
            info = self.get_update_info()
            
            health_status = {
                'data_type': self.data_type,
                'timestamp': datetime.now().isoformat(),
                'file_exists': info.get('stop_price_file_exists', False),
                'file_size_mb': info.get('file_size_mb', 0),
                'local_latest_date': info.get('local_latest_date'),
                'db_latest_date': info.get('db_latest_date'),
                'need_update': info.get('need_update', False),
                'status': 'healthy' if info.get('stop_price_file_exists', False) else 'warning'
            }
            
            # 检查数据是否过期
            if info.get('local_latest_date') and info.get('db_latest_date'):
                local_date = datetime.strptime(info['local_latest_date'], '%Y-%m-%d')
                db_date = datetime.strptime(info['db_latest_date'], '%Y-%m-%d')
                gap_days = (db_date - local_date).days
                
                health_status['data_gap_days'] = gap_days
                
                if gap_days > 5:
                    health_status['status'] = 'error'
                    health_status['message'] = f'涨跌停数据过期 {gap_days} 天'
                elif gap_days > 2:
                    health_status['status'] = 'warning'
                    health_status['message'] = f'涨跌停数据落后 {gap_days} 天'
                else:
                    health_status['message'] = '涨跌停数据正常'
            
            return health_status
            
        except Exception as e:
            return {
                'data_type': self.data_type,
                'timestamp': datetime.now().isoformat(),
                'status': 'error',
                'message': f'涨跌停数据健康检查失败: {e}'
            }
    
    def clean_old_backups(self, keep_days: int = 3):
        """清理旧备份"""
        self.updater.clean_old_backups(keep_days)


class FinancialDataUpdater(BaseDataUpdater):
    """财务数据更新器"""
    
    def __init__(self):
        super().__init__("financial_data")
        self.updater = IncrementalFinancialUpdater()
    
    def get_update_info(self) -> Dict:
        """获取更新信息"""
        return self.updater.get_update_info()
    
    def needs_update(self) -> bool:
        """检查是否需要更新"""
        update_status = self.updater.needs_update()
        return any(update_status.values())
    
    def update_data(self) -> bool:
        """执行数据更新"""
        results = self.updater.update_all_tables()
        return all(results.values())
    
    def get_health_status(self) -> Dict:
        """获取健康状态"""
        try:
            info = self.get_update_info()
            
            if 'error' in info:
                return {
                    'data_type': self.data_type,
                    'timestamp': datetime.now().isoformat(),
                    'status': 'error',
                    'message': f'财务数据健康检查失败: {info["error"]}'
                }
            
            # 检查各个表的状态
            tables_info = info.get('tables', {})
            all_healthy = True
            messages = []
            
            for table_name, table_info in tables_info.items():
                if not table_info.get('file_exists', False):
                    all_healthy = False
                    messages.append(f"{table_info['description']}文件不存在")
                elif table_info.get('need_update', False):
                    messages.append(f"{table_info['description']}有更新")
                else:
                    messages.append(f"{table_info['description']}正常")
            
            # 汇总状态
            if all_healthy:
                status = 'healthy'
                message = '所有财务数据正常'
            elif any('文件不存在' in msg for msg in messages):
                status = 'warning'
                message = '部分财务数据文件缺失'
            else:
                status = 'healthy'
                message = '财务数据可用'
            
            # 统计信息
            total_tables = len(tables_info)
            existing_tables = sum(1 for info in tables_info.values() if info.get('file_exists', False))
            total_size_mb = sum(info.get('file_size_mb', 0) for info in tables_info.values())
            
            health_status = {
                'data_type': self.data_type,
                'timestamp': datetime.now().isoformat(),
                'status': status,
                'message': message,
                'total_tables': total_tables,
                'existing_tables': existing_tables,
                'total_size_mb': round(total_size_mb, 1),
                'tables_detail': '; '.join(messages)
            }
            
            return health_status
            
        except Exception as e:
            return {
                'data_type': self.data_type,
                'timestamp': datetime.now().isoformat(),
                'status': 'error',
                'message': f'财务数据健康检查失败: {e}'
            }
    
    def clean_old_backups(self, keep_days: int = 7):
        """清理旧备份"""
        self.updater.clean_old_backups(keep_days)


class SectorChangesDataUpdater(BaseDataUpdater):
    """板块进出数据更新器"""
    
    def __init__(self):
        super().__init__("sector_changes_data")
        self.fetcher = StockDataFetcher()
        
    def update_data(self) -> bool:
        """更新板块进出数据"""
        try:
            logger.info("开始更新板块进出数据...")
            
            # 获取配置
            data_root = get_config('main.paths.data_root')
            classification_data_path = Path(data_root)
            classification_data_path.mkdir(parents=True, exist_ok=True)
            
            # 定义文件路径
            sector_changes_file = classification_data_path / 'SectorChanges_data.pkl'
            
            # 确定更新起始日期
            if sector_changes_file.exists():
                try:
                    existing_data = pd.read_pickle(sector_changes_file)
                    last_date = existing_data['sel_day'].max()
                    begin_date = last_date + 1  # 从最后日期的下一天开始更新
                    logger.info(f"增量更新，从 {begin_date} 开始")
                except Exception as e:
                    logger.warning(f"读取现有数据失败: {e}，执行全量更新")
                    begin_date = 20200101
            else:
                begin_date = 20200101
                logger.info(f"文件不存在，执行全量更新，从 {begin_date} 开始")
            
            # 获取新数据
            logger.info("获取板块进出调整数据...")
            new_data = self.fetcher.fetch_data('sector_changes', begin_date=begin_date)
            
            if new_data.empty:
                logger.info("没有新的板块进出数据")
                return True
            
            # 处理现有数据
            if sector_changes_file.exists():
                try:
                    existing_data = pd.read_pickle(sector_changes_file)
                    # 合并数据，去重
                    combined_data = pd.concat([existing_data, new_data], ignore_index=True)
                    combined_data = combined_data.drop_duplicates(
                        subset=['sel_day', 'code', 'concept_code'], keep='last'
                    )
                    combined_data = combined_data.sort_values(['sel_day', 'concept_code', 'code'])
                    logger.info(f"数据合并完成，原有 {len(existing_data)} 条，新增 {len(new_data)} 条，合并后 {len(combined_data)} 条")
                except Exception as e:
                    logger.warning(f"合并数据失败: {e}，使用新数据覆盖")
                    combined_data = new_data
            else:
                combined_data = new_data
            
            # 保存数据
            combined_data.to_pickle(sector_changes_file)
            logger.info(f"板块进出数据保存完成: {sector_changes_file}")
            logger.info(f"数据形状: {combined_data.shape}")
            logger.info(f"日期范围: {combined_data['sel_day'].min()} - {combined_data['sel_day'].max()}")
            
            return True
            
        except Exception as e:
            logger.error(f"板块进出数据更新失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def get_update_info(self) -> Dict:
        """获取更新信息"""
        try:
            data_root = get_config('main.paths.data_root')
            sector_changes_file = Path(data_root) / 'auxiliary' / 'SectorChanges_data.pkl'
            
            info = {
                'data_type': 'sector_changes_data',
                'file_exists': sector_changes_file.exists(),
                'need_update': True
            }
            
            if sector_changes_file.exists():
                try:
                    data = pd.read_pickle(sector_changes_file)
                    latest_date = data['sel_day'].max()
                    info['latest_date'] = str(latest_date)
                    info['record_count'] = len(data)
                    
                    # 检查是否需要更新（如果最新数据超过1天则需要更新）
                    today = int(datetime.now().strftime('%Y%m%d'))
                    days_diff = (datetime.strptime(str(today), '%Y%m%d') - 
                               datetime.strptime(str(latest_date), '%Y%m%d')).days
                    info['days_since_update'] = days_diff
                    info['need_update'] = days_diff > 0
                    
                except Exception as e:
                    logger.warning(f"读取板块进出数据文件失败: {e}")
                    info['need_update'] = True
            
            return info
            
        except Exception as e:
            logger.error(f"获取板块进出数据更新信息失败: {e}")
            return {'data_type': 'sector_changes_data', 'error': str(e)}
    
    def get_health_status(self) -> Dict:
        """获取健康状态"""
        try:
            data_root = get_config('main.paths.data_root')
            sector_changes_file = Path(data_root) / 'SectorChanges_data.pkl'
            
            if not sector_changes_file.exists():
                return {
                    'data_type': self.data_type,
                    'timestamp': datetime.now().isoformat(),
                    'status': 'error',
                    'message': '板块进出数据文件不存在'
                }
            
            # 检查文件完整性
            data = pd.read_pickle(sector_changes_file)
            
            if data.empty:
                return {
                    'data_type': self.data_type,
                    'timestamp': datetime.now().isoformat(),
                    'status': 'warning',
                    'message': '板块进出数据文件为空'
                }
            
            # 检查数据时效性
            latest_date = data['sel_day'].max()
            today = int(datetime.now().strftime('%Y%m%d'))
            days_diff = (datetime.strptime(str(today), '%Y%m%d') - 
                        datetime.strptime(str(latest_date), '%Y%m%d')).days
            
            if days_diff > 7:
                return {
                    'data_type': self.data_type,
                    'timestamp': datetime.now().isoformat(),
                    'status': 'warning',
                    'message': f'板块进出数据已过期 {days_diff} 天'
                }
            
            return {
                'data_type': self.data_type,
                'timestamp': datetime.now().isoformat(),
                'status': 'healthy',
                'message': f'板块进出数据正常，最新日期: {latest_date}，共 {len(data)} 条记录'
            }
            
        except Exception as e:
            return {
                'data_type': self.data_type,
                'timestamp': datetime.now().isoformat(),
                'status': 'error',
                'message': f'健康检查失败: {e}'
            }
    
    def needs_update(self, force: bool = False) -> bool:
        """检查是否需要更新"""
        if force:
            return True
        
        try:
            info = self.get_update_info()
            return info.get('need_update', True)
        except Exception as e:
            logger.warning(f"检查更新需求失败: {e}")
            return True



class STDataUpdater(BaseDataUpdater):
    """ST股票数据更新器"""
    
    def __init__(self):
        super().__init__("st_data")
        from data.fetcher.data_fetcher import MarketDataFetcher
        self.fetcher = MarketDataFetcher()
        self.st_file_path = os.path.join(get_config('main.paths.data_root'), 'ST_stocks.pkl')
    
    def get_update_info(self) -> Dict:
        """获取更新信息"""
        try:
            info = {
                'data_type': self.data_type,
                'st_file_exists': os.path.exists(self.st_file_path),
            }
            
            if info['st_file_exists']:
                st_size = os.path.getsize(self.st_file_path) / 1024 / 1024
                info['st_file_size_mb'] = round(st_size, 2)
                
                # 检查ST数据时间
                import pandas as pd
                try:
                    st_data = pd.read_pickle(self.st_file_path)
                    if 'tradingday' in st_data.columns:
                        latest_date = st_data['tradingday'].max()
                        info['st_latest_date'] = latest_date.strftime('%Y-%m-%d') if pd.notna(latest_date) else None
                except:
                    info['st_latest_date'] = None
            else:
                info['st_file_size_mb'] = 0
                info['st_latest_date'] = None
            
            # ST股票数据通常更新频率较低，检查是否超过1个月未更新
            if info['st_latest_date']:
                from datetime import datetime, timedelta
                latest_date = datetime.strptime(info['st_latest_date'], '%Y-%m-%d')
                days_old = (datetime.now() - latest_date).days
                info['need_update'] = days_old > 30  # 30天未更新则需要更新
                info['days_since_update'] = days_old
            else:
                info['need_update'] = True
                info['days_since_update'] = None
            
            return info
            
        except Exception as e:
            return {
                'data_type': self.data_type,
                'error': str(e),
                'need_update': True
            }
    
    def needs_update(self) -> bool:
        """检查是否需要更新"""
        info = self.get_update_info()
        return info.get('need_update', True)
    
    def update_data(self) -> bool:
        """执行数据更新"""
        try:
            logger.info("开始更新ST股票数据...")
            
            # 获取ST股票数据
            logger.info("获取ST股票数据...")
            st_data = self.fetcher.fetch_data('st_stocks')
            
            if not st_data.empty:
                # 保存ST数据
                os.makedirs(os.path.dirname(self.st_file_path), exist_ok=True)
                st_data.to_pickle(self.st_file_path)
                logger.info(f"ST股票数据已保存: {self.st_file_path} ({st_data.shape})")
                
                # 记录数据统计信息
                unique_stocks = st_data['code'].nunique()
                if len(st_data) > 0:
                    min_date = str(st_data['tradingday'].min())
                    max_date = str(st_data['tradingday'].max())
                    date_range = f"{min_date[:4]}-{min_date[4:6]}-{min_date[6:8]} 到 {max_date[:4]}-{max_date[4:6]}-{max_date[6:8]}"
                else:
                    date_range = "无数据"
                logger.info(f"ST股票数据统计: {unique_stocks}只股票, 时间范围: {date_range}")
                
                return True
            else:
                logger.warning("ST股票数据为空")
                return False
                
        except Exception as e:
            logger.error(f"ST股票数据更新失败: {e}")
            return False
    
    def get_health_status(self) -> Dict:
        """获取健康状态"""
        try:
            info = self.get_update_info()
            
            if 'error' in info:
                return {
                    'data_type': self.data_type,
                    'timestamp': datetime.now().isoformat(),
                    'status': 'error',
                    'message': f'ST数据健康检查失败: {info["error"]}'
                }
            
            # 分析状态
            st_exists = info.get('st_file_exists', False)
            days_old = info.get('days_since_update')
            
            if not st_exists:
                status = 'error'
                message = 'ST股票数据文件不存在'
            elif days_old is None:
                status = 'warning'
                message = '无法获取ST数据更新时间'
            elif days_old > 90:  # 3个月
                status = 'warning'
                message = f'ST股票数据过期 {days_old} 天'
            elif days_old > 30:  # 1个月
                status = 'warning' 
                message = f'ST股票数据需要更新（{days_old} 天前）'
            else:
                status = 'healthy'
                message = 'ST股票数据正常'
            
            return {
                'data_type': self.data_type,
                'timestamp': datetime.now().isoformat(),
                'status': status,
                'message': message,
                'st_file_exists': st_exists,
                'st_latest_date': info.get('st_latest_date'),
                'days_since_update': days_old,
                'st_file_size_mb': info.get('st_file_size_mb', 0)
            }
            
        except Exception as e:
            return {
                'data_type': self.data_type,
                'timestamp': datetime.now().isoformat(),
                'status': 'error',
                'message': f'ST数据健康检查失败: {e}'
            }


class ScheduledDataUpdater:
    """定时数据更新管理器"""
    
    def __init__(self, data_types: Optional[List[str]] = None):
        """
        初始化定时数据更新器
        
        Args:
            data_types: 要更新的数据类型列表，None表示使用默认配置
        """
        # 注册所有可用的数据更新器
        self.updaters = {
            'price': PriceDataUpdater(),
            'stop_price': StopPriceDataUpdater(),
            'financial': FinancialDataUpdater(),
            'sector_changes': SectorChangesDataUpdater(),
            'st': STDataUpdater()
        }
        
        # 配置要更新的数据类型
        if data_types is None:
            # 默认更新价格数据和涨跌停数据
            self.active_updaters = ['price', 'stop_price']
        else:
            self.active_updaters = [dt for dt in data_types if dt in self.updaters]
        
        logger.info(f"初始化数据更新器，活跃类型: {self.active_updaters}")
    
    def should_update_now(self) -> bool:
        """
        判断是否应该在当前时间更新
        股票市场通常在交易日15:30收盘，数据一般在16:00后可用
        """
        now = datetime.now()
        current_time = now.time()
        
        # 工作日的16:00-23:59之间可以更新
        if now.weekday() < 5:  # 周一到周五
            update_start = time(16, 0)  # 16:00
            update_end = time(23, 59)   # 23:59
            
            if update_start <= current_time <= update_end:
                return True
        
        # 周末可以更新（补充可能遗漏的数据）
        elif now.weekday() >= 5:  # 周六周日
            return True
        
        return False
    
    def run_price_data_update(self, force: bool = False) -> DataUpdateResult:
        """
        运行价格数据更新（保持向后兼容）
        
        Args:
            force: 是否强制更新，忽略时间检查
            
        Returns:
            更新结果
        """
        return self.run_single_update('price', force=force)
    
    def run_single_update(self, data_type: str, force: bool = False) -> DataUpdateResult:
        """
        运行单个数据类型的更新
        
        Args:
            data_type: 数据类型
            force: 是否强制更新
            
        Returns:
            更新结果
        """
        start_time = datetime.now()
        
        if data_type not in self.updaters:
            return DataUpdateResult(
                data_type=data_type,
                success=False,
                message=f"未知的数据类型: {data_type}"
            )
        
        updater = self.updaters[data_type]
        
        try:
            logger.info(f"开始更新 {data_type} 数据...")
            
            # 检查更新时间
            if not force and not self.should_update_now():
                logger.info("当前时间不适合更新数据，跳过更新")
                return DataUpdateResult(
                    data_type=data_type,
                    success=True,
                    message="当前时间不适合更新，已跳过"
                )
            
            # 检查是否需要更新
            if not updater.needs_update():
                logger.info(f"{data_type} 数据已是最新，无需更新")
                return DataUpdateResult(
                    data_type=data_type,
                    success=True,
                    message="数据已是最新，无需更新"
                )
            
            # 执行更新
            logger.info(f"执行 {data_type} 数据更新...")
            success = updater.update_data()
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            if success:
                message = f"{data_type} 数据更新成功"
                logger.info(f"[OK] {message}，耗时: {duration:.1f}秒")
                
                # 清理备份文件
                if data_type in ['price', 'stop_price', 'financial']:
                    keep_days = 3 if data_type == 'price' else 7
                    updater.clean_old_backups(keep_days=keep_days)
                
                return DataUpdateResult(
                    data_type=data_type,
                    success=True,
                    message=message,
                    duration=duration,
                    details=updater.get_update_info()
                )
            else:
                message = f"{data_type} 数据更新失败"
                logger.error(f"[FAIL] {message}")
                return DataUpdateResult(
                    data_type=data_type,
                    success=False,
                    message=message,
                    duration=duration
                )
                
        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            message = f"{data_type} 数据更新异常: {e}"
            logger.error(message)
            
            return DataUpdateResult(
                data_type=data_type,
                success=False,
                message=message,
                duration=duration
            )
    
    def run_all_updates(self, force: bool = False) -> List[DataUpdateResult]:
        """
        运行所有活跃数据类型的更新
        
        Args:
            force: 是否强制更新
            
        Returns:
            所有更新结果的列表
        """
        start_time = datetime.now()
        
        logger.info("=" * 60)
        logger.info("开始定时数据更新任务")
        logger.info(f"当前时间: {start_time}")
        logger.info(f"更新数据类型: {self.active_updaters}")
        logger.info("=" * 60)
        
        results = []
        
        try:
            for data_type in self.active_updaters:
                result = self.run_single_update(data_type, force=force)
                results.append(result)
                
                # 如果更新失败，记录但继续其他更新
                if not result.success:
                    logger.error(f"{data_type} 更新失败: {result.message}")
            
            # 汇总结果
            total_success = sum(1 for r in results if r.success)
            total_count = len(results)
            
            end_time = datetime.now()
            total_duration = end_time - start_time
            
            logger.info("=" * 60)
            logger.info(f"数据更新任务完成: {total_success}/{total_count} 成功")
            logger.info(f"总耗时: {total_duration}")
            logger.info("=" * 60)
            
            return results
            
        except Exception as e:
            logger.error(f"批量数据更新失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            # 如果没有任何结果，创建一个失败结果
            if not results:
                results.append(DataUpdateResult(
                    data_type="batch_update",
                    success=False,
                    message=f"批量更新失败: {e}"
                ))
            
            return results
        finally:
            logger.info("定时数据更新任务结束")
    
    def run_health_check(self) -> Dict[str, Dict]:
        """
        运行所有数据类型的健康检查
        
        Returns:
            健康状态字典
        """
        health_results = {}
        
        for data_type in self.active_updaters:
            if data_type in self.updaters:
                updater = self.updaters[data_type]
                health_results[data_type] = updater.get_health_status()
        
        return health_results


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='定时数据更新器')
    parser.add_argument('--data-type', choices=['price', 'stop_price', 'financial', 'sector_changes', 'st', 'all'], 
                       default='price', help='要更新的数据类型')
    parser.add_argument('--force', action='store_true', help='强制更新，忽略时间检查')
    parser.add_argument('--health-check', action='store_true', help='只执行健康检查')
    parser.add_argument('--quiet', action='store_true', help='静默模式，减少输出')
    parser.add_argument('--list-data', action='store_true', help='列出所有可用数据集')
    parser.add_argument('--data-summary', action='store_true', help='显示数据摘要')
    
    args = parser.parse_args()
    
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    
    # 确定要更新的数据类型
    if args.data_type == 'all':
        data_types = ['price', 'stop_price', 'financial', 'industry', 'st']
    else:
        data_types = [args.data_type]
    
    # 处理数据注册器相关命令
    if args.list_data or args.data_summary:
        registry = get_data_registry()
        
        if args.list_data:
            print("\n=== 所有可用数据集 ===")
            df = registry.list_all_datasets()
            print(df.to_string(index=False))
            return
            
        if args.data_summary:
            registry.print_data_summary()
            return
    
    updater = ScheduledDataUpdater(data_types=data_types)
    
    try:
        if args.health_check:
            # 只执行健康检查
            health_results = updater.run_health_check()
            
            print("健康检查结果:")
            all_healthy = True
            
            for data_type, health_status in health_results.items():
                print(f"\n{data_type.upper()} 数据:")
                print(f"  状态: {health_status['status']}")
                print(f"  消息: {health_status.get('message', 'N/A')}")
                
                if 'file_exists' in health_status:
                    print(f"  文件存在: {health_status['file_exists']}")
                if 'local_latest_date' in health_status:
                    print(f"  本地最新: {health_status['local_latest_date']}")
                if 'db_latest_date' in health_status:
                    print(f"  数据库最新: {health_status['db_latest_date']}")
                
                if health_status['status'] not in ['healthy', 'not_implemented']:
                    all_healthy = False
            
            # 健康检查的退出码
            sys.exit(0 if all_healthy else 1)
        else:
            # 执行更新
            if args.data_type == 'all':
                results = updater.run_all_updates(force=args.force)
                success = all(r.success for r in results)
            else:
                result = updater.run_single_update(args.data_type, force=args.force)
                success = result.success
            
            sys.exit(0 if success else 1)
            
    except KeyboardInterrupt:
        logger.info("用户中断了程序")
        sys.exit(130)
    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()