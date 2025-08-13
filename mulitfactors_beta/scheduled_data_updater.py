#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
定时数据更新脚本
支持多种数据类型的定时更新，包括价格数据、财务数据、行业数据等
用于自动化每日数据更新，可配置到Windows任务计划程序或cron
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
from data.fetcher.incremental_price_updater import IncrementalPriceUpdater
from core.config_manager import get_path

# 配置日志到文件
log_dir = os.path.join(get_path('data_root'), 'logs')
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


# 未来扩展的数据更新器示例
class FinancialDataUpdater(BaseDataUpdater):
    """财务数据更新器（预留接口）"""
    
    def __init__(self):
        super().__init__("financial_data")
        # TODO: 实现财务数据更新器
    
    def get_update_info(self) -> Dict:
        return {"status": "not_implemented", "message": "财务数据更新器尚未实现"}
    
    def needs_update(self) -> bool:
        return False
    
    def update_data(self) -> bool:
        logger.info("财务数据更新器尚未实现")
        return True
    
    def get_health_status(self) -> Dict:
        return {
            'data_type': self.data_type,
            'timestamp': datetime.now().isoformat(),
            'status': 'not_implemented',
            'message': '财务数据更新器尚未实现'
        }


class IndustryDataUpdater(BaseDataUpdater):
    """行业数据更新器（预留接口）"""
    
    def __init__(self):
        super().__init__("industry_data")
        # TODO: 实现行业数据更新器
    
    def get_update_info(self) -> Dict:
        return {"status": "not_implemented", "message": "行业数据更新器尚未实现"}
    
    def needs_update(self) -> bool:
        return False
    
    def update_data(self) -> bool:
        logger.info("行业数据更新器尚未实现")
        return True
    
    def get_health_status(self) -> Dict:
        return {
            'data_type': self.data_type,
            'timestamp': datetime.now().isoformat(),
            'status': 'not_implemented',
            'message': '行业数据更新器尚未实现'
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
            'financial': FinancialDataUpdater(),
            'industry': IndustryDataUpdater()
        }
        
        # 配置要更新的数据类型
        if data_types is None:
            # 默认只更新价格数据（其他还未实现）
            self.active_updaters = ['price']
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
                
                # 价格数据需要清理备份
                if data_type == 'price':
                    updater.clean_old_backups(keep_days=3)
                
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
    parser.add_argument('--data-type', choices=['price', 'financial', 'industry', 'all'], 
                       default='price', help='要更新的数据类型')
    parser.add_argument('--force', action='store_true', help='强制更新，忽略时间检查')
    parser.add_argument('--health-check', action='store_true', help='只执行健康检查')
    parser.add_argument('--quiet', action='store_true', help='静默模式，减少输出')
    
    args = parser.parse_args()
    
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    
    # 确定要更新的数据类型
    if args.data_type == 'all':
        data_types = ['price', 'financial', 'industry']
    else:
        data_types = [args.data_type]
    
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