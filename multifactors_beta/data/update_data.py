"""
数据更新主脚本
提供命令行接口和自动化更新功能

使用方法：
    # 运行完整更新
    python update_data.py --full

    # 只更新价格和板块估值
    python update_data.py --price --sector

    # 每日更新
    python update_data.py --daily

    # 每周更新
    python update_data.py --weekly

    # 自定义板块估值日期范围
    python update_data.py --sector --sector-days 30
"""

import argparse
import sys
from pathlib import Path
import logging
from datetime import datetime
import time

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.processor import IntegratedDataPipeline, DataUpdateScheduler


def setup_logging():
    """设置日志配置"""
    log_dir = project_root / "logs"
    log_dir.mkdir(exist_ok=True)

    # 日志文件名包含日期
    log_file = log_dir / f"data_update_{datetime.now().strftime('%Y%m%d')}.log"

    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger(__name__)


def main():
    """主函数"""
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='多因子数据更新系统')

    # 更新模式
    parser.add_argument('--full', action='store_true',
                       help='运行完整数据更新流程')
    parser.add_argument('--daily', action='store_true',
                       help='运行每日更新任务')
    parser.add_argument('--weekly', action='store_true',
                       help='运行每周更新任务')
    parser.add_argument('--monthly', action='store_true',
                       help='运行每月更新任务')

    # 单独更新选项
    parser.add_argument('--price', action='store_true',
                       help='更新价格数据')
    parser.add_argument('--financial', action='store_true',
                       help='更新财务数据')
    parser.add_argument('--sector', action='store_true',
                       help='更新板块估值')

    # 板块估值参数
    parser.add_argument('--sector-days', type=int, default=None,
                       help='板块估值计算天数（默认252）')
    parser.add_argument('--no-sector', action='store_true',
                       help='跳过板块估值计算')

    # 其他选项
    parser.add_argument('--force', action='store_true',
                       help='强制更新（忽略缓存）')
    parser.add_argument('--quiet', action='store_true',
                       help='静默模式（只显示错误）')

    args = parser.parse_args()

    # 设置日志
    logger = setup_logging()

    if args.quiet:
        logging.getLogger().setLevel(logging.ERROR)

    # 记录开始时间
    start_time = time.time()

    logger.info("="*60)
    logger.info("数据更新任务开始")
    logger.info(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*60)

    try:
        # 创建数据更新调度器
        pipeline = IntegratedDataPipeline()
        scheduler = DataUpdateScheduler(pipeline)

        # 执行更新任务
        if args.full:
            logger.info("执行完整数据更新...")
            pipeline.run_full_pipeline(
                save_intermediate=True,
                include_sector_valuation=not args.no_sector,
                sector_date_range=args.sector_days
            )

        elif args.daily:
            logger.info("执行每日更新任务...")
            scheduler.run_daily_update()

        elif args.weekly:
            logger.info("执行每周更新任务...")
            scheduler.run_weekly_update()

        elif args.monthly:
            logger.info("执行每月更新任务...")
            scheduler.run_monthly_update()

        else:
            # 自定义更新
            if args.price or args.financial or args.sector:
                logger.info("执行自定义更新任务...")
                scheduler.run_custom_update(
                    update_price=args.price,
                    update_financial=args.financial,
                    update_sector_valuation=args.sector,
                    sector_date_range=args.sector_days
                )
            else:
                # 如果没有指定任何选项，显示帮助
                parser.print_help()
                return

        # 计算耗时
        elapsed = time.time() - start_time
        logger.info("="*60)
        logger.info(f"数据更新完成！总耗时: {elapsed:.2f}秒")
        logger.info("="*60)

    except Exception as e:
        logger.error(f"数据更新失败: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()