#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
定时价格数据更新脚本
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
from data.fetcher.incremental_price_updater import IncrementalPriceUpdater
from core.config_manager import get_path

# 配置日志到文件
log_dir = os.path.join(get_path('data_root'), 'logs')
os.makedirs(log_dir, exist_ok=True)

log_file = os.path.join(log_dir, f"price_update_{datetime.now().strftime('%Y%m%d')}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


class ScheduledUpdater:
    """定时更新器"""
    
    def __init__(self):
        self.updater = IncrementalPriceUpdater()
        
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
    
    def run_update(self, force: bool = False) -> bool:
        """
        运行更新
        
        Args:
            force: 是否强制更新，忽略时间检查
            
        Returns:
            是否更新成功
        """
        start_time = datetime.now()
        
        try:
            logger.info("=" * 60)
            logger.info("开始定时更新任务")
            logger.info(f"当前时间: {start_time}")
            logger.info("=" * 60)
            
            # 检查更新时间
            if not force and not self.should_update_now():
                logger.info("当前时间不适合更新数据，跳过更新")
                return True
            
            # 检查是否需要更新
            info = self.updater.get_update_info()
            
            logger.info("当前数据状态:")
            logger.info(f"  Price.pkl存在: {info['price_file_exists']}")
            logger.info(f"  本地最新日期: {info['local_latest_date']}")
            logger.info(f"  数据库最新日期: {info['db_latest_date']}")
            logger.info(f"  需要更新: {info['need_update']}")
            logger.info(f"  文件大小: {info['file_size_mb']:.1f} MB")
            
            if not info['need_update']:
                logger.info("数据已是最新，无需更新")
                return True
            
            # 执行更新
            logger.info("开始执行增量更新...")
            success = self.updater.update_price_file()
            
            end_time = datetime.now()
            duration = end_time - start_time
            
            if success:
                logger.info(f"[OK] 更新成功！耗时: {duration}")
                
                # 获取更新后的状态
                new_info = self.updater.get_update_info()
                logger.info("更新后状态:")
                logger.info(f"  本地最新日期: {new_info['local_latest_date']}")
                logger.info(f"  文件大小: {new_info['file_size_mb']:.1f} MB")
                
                # 清理旧备份（保留3天）
                logger.info("清理旧备份文件...")
                self.updater.clean_old_backups(keep_days=3)
                
                return True
            else:
                logger.error("[FAIL] 更新失败")
                return False
                
        except Exception as e:
            logger.error(f"定时更新任务失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
        finally:
            logger.info("=" * 60)
            logger.info("定时更新任务结束")
            logger.info("=" * 60)
    
    def run_health_check(self) -> dict:
        """
        运行健康检查
        
        Returns:
            健康状态字典
        """
        try:
            info = self.updater.get_update_info()
            
            health_status = {
                'timestamp': datetime.now().isoformat(),
                'file_exists': info['price_file_exists'],
                'file_size_mb': info['file_size_mb'],
                'local_latest_date': info['local_latest_date'],
                'db_latest_date': info['db_latest_date'],
                'need_update': info['need_update'],
                'status': 'healthy' if info['price_file_exists'] else 'warning'
            }
            
            # 检查数据是否过期（超过5天认为有问题）
            if info['local_latest_date'] and info['db_latest_date']:
                from datetime import datetime
                local_date = datetime.strptime(info['local_latest_date'], '%Y-%m-%d')
                db_date = datetime.strptime(info['db_latest_date'], '%Y-%m-%d')
                gap_days = (db_date - local_date).days
                
                health_status['data_gap_days'] = gap_days
                
                if gap_days > 5:
                    health_status['status'] = 'error'
                    health_status['message'] = f'数据过期 {gap_days} 天'
                elif gap_days > 2:
                    health_status['status'] = 'warning'
                    health_status['message'] = f'数据落后 {gap_days} 天'
                else:
                    health_status['message'] = '数据正常'
            
            return health_status
            
        except Exception as e:
            return {
                'timestamp': datetime.now().isoformat(),
                'status': 'error',
                'message': f'健康检查失败: {e}'
            }


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='定时价格数据更新')
    parser.add_argument('--force', action='store_true', help='强制更新，忽略时间检查')
    parser.add_argument('--health-check', action='store_true', help='只执行健康检查')
    parser.add_argument('--quiet', action='store_true', help='静默模式，减少输出')
    
    args = parser.parse_args()
    
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    
    updater = ScheduledUpdater()
    
    try:
        if args.health_check:
            # 只执行健康检查
            health_status = updater.run_health_check()
            
            print("健康检查结果:")
            print(f"状态: {health_status['status']}")
            print(f"消息: {health_status.get('message', 'N/A')}")
            print(f"文件存在: {health_status['file_exists']}")
            print(f"本地最新: {health_status['local_latest_date']}")
            print(f"数据库最新: {health_status['db_latest_date']}")
            
            # 健康检查的退出码
            if health_status['status'] == 'healthy':
                sys.exit(0)
            elif health_status['status'] == 'warning':
                sys.exit(1)
            else:
                sys.exit(2)
        else:
            # 执行更新
            success = updater.run_update(force=args.force)
            sys.exit(0 if success else 1)
            
    except KeyboardInterrupt:
        logger.info("用户中断了程序")
        sys.exit(130)
    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()