"""
运行增强的数据处理管道

展示如何使用并行处理和增量更新功能
"""
import sys
import argparse
from pathlib import Path
from datetime import datetime
import logging

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from data.processor import EnhancedDataProcessingPipeline


def setup_logging(level=logging.INFO):
    """设置日志"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'enhanced_processing_{datetime.now().strftime("%Y%m%d")}.log')
        ]
    )


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='运行增强的数据处理管道')
    
    # 处理选项
    parser.add_argument('--parallel', action='store_true', 
                       help='使用并行处理（默认开启）', default=True)
    parser.add_argument('--no-parallel', dest='parallel', action='store_false',
                       help='禁用并行处理')
    parser.add_argument('--incremental', action='store_true',
                       help='使用增量处理（默认开启）', default=True)
    parser.add_argument('--no-incremental', dest='incremental', action='store_false',
                       help='禁用增量处理')
    parser.add_argument('--force', action='store_true',
                       help='强制完整更新，忽略增量处理')
    parser.add_argument('--workers', type=int, default=None,
                       help='并行工作进程数（默认为CPU核心数-1）')
    parser.add_argument('--no-progress', action='store_true',
                       help='禁用进度条显示')
    parser.add_argument('--debug', action='store_true',
                       help='启用调试日志')
    
    args = parser.parse_args()
    
    # 设置日志级别
    setup_logging(logging.DEBUG if args.debug else logging.INFO)
    logger = logging.getLogger(__name__)
    
    print("="*60)
    print("增强数据处理管道")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    print(f"\n配置:")
    print(f"  并行处理: {'启用' if args.parallel else '禁用'}")
    print(f"  增量处理: {'启用' if args.incremental else '禁用'}")
    print(f"  强制更新: {'是' if args.force else '否'}")
    if args.parallel and args.workers:
        print(f"  工作进程: {args.workers}")
    print()
    
    try:
        # 创建增强管道
        logger.info("初始化增强数据处理管道...")
        pipeline = EnhancedDataProcessingPipeline(
            use_parallel=args.parallel,
            use_incremental=args.incremental,
            n_workers=args.workers
        )
        
        # 运行处理
        logger.info("开始数据处理...")
        results = pipeline.run_enhanced_pipeline(
            force_full_update=args.force,
            skip_unchanged=True,
            monitor_progress=not args.no_progress
        )
        
        # 显示结果摘要
        print("\n" + "="*60)
        print("处理完成！")
        print("="*60)
        
        if 'price_df' in results:
            print(f"\n价格数据:")
            print(f"  形状: {results['price_df'].shape}")
            print(f"  日期范围: {results['price_df'].index.get_level_values(0).min()} 至 {results['price_df'].index.get_level_values(0).max()}")
            
        if 'stock_3d' in results:
            print(f"\n3D矩阵:")
            print(f"  日期数: {len(results['stock_3d']['TradingDates'])}")
            print(f"  股票数: {len(results['stock_3d']['StockCodes'])}")
            
        # 统计生成的文件
        data_root = Path(pipeline.data_save_path)
        generated_files = list(data_root.glob("*.pkl"))
        print(f"\n生成文件数: {len(generated_files)}")
        
        # 显示性能提升信息
        if args.incremental and pipeline.incremental_processor:
            metadata = pipeline.incremental_processor.metadata
            if not metadata.empty:
                print("\n增量处理信息:")
                for data_type, info in metadata.iterrows():
                    print(f"  {data_type}: 最后处理 {info['last_processed_date']}")
                    
        return True
        
    except KeyboardInterrupt:
        logger.info("用户中断处理")
        return False
        
    except Exception as e:
        logger.error(f"处理失败: {e}", exc_info=True)
        return False
        

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)