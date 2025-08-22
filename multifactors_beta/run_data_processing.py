"""
运行数据处理管道

使用新的模块化数据处理系统处理股票数据
"""
import sys
import time
import logging
from pathlib import Path
from datetime import datetime

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from data.processor.data_processing_pipeline import DataProcessingPipeline

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """主函数"""
    print("="*60)
    print("数据处理管道")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    try:
        # 创建数据处理管道
        logger.info("初始化数据处理管道...")
        pipeline = DataProcessingPipeline()
        
        # 记录开始时间
        start_time = time.time()
        
        # 运行完整的处理流程
        logger.info("开始执行数据处理...")
        results = pipeline.run_full_pipeline(save_intermediate=True)
        
        # 计算耗时
        elapsed_time = time.time() - start_time
        
        print("\n" + "="*60)
        print("处理完成！")
        print(f"总耗时: {elapsed_time/60:.2f} 分钟")
        print("="*60)
        
        # 显示处理结果摘要
        if 'price_df' in results:
            print(f"\n价格数据处理完成:")
            print(f"  - 数据形状: {results['price_df'].shape}")
            
        if 'stock_3d' in results:
            print(f"\n3D矩阵生成完成:")
            print(f"  - 日期数: {len(results['stock_3d']['TradingDates'])}")
            print(f"  - 股票数: {len(results['stock_3d']['StockCodes'])}")
            print(f"  - 特征数: {len(results['stock_3d']['datacolumns'])}")
            
        print("\n生成的数据文件:")
        data_root = Path(pipeline.data_save_path)
        output_files = [
            "Stock3d.pkl",
            "LogReturn_daily_o2o.pkl",
            "LogReturn_daily_vwap.pkl",
            "LogReturn_weekly_o2o.pkl",
            "LogReturn_weekly_vwap.pkl",
            "LogReturn_monthly_o2o.pkl",
            "LogReturn_monthly_vwap.pkl",
            "LogReturn_5days_o2o.pkl",
            "LogReturn_20days_o2o.pkl",
            "released_dates_df.pkl",
            "released_dates_count_df.pkl",
            "lag1_released_logreturn.pkl",
            "lag5_released_logreturn.pkl",
            "lag20_released_logreturn.pkl",
            "lag1_released_alfa_logreturn.pkl",
            "lag5_released_alfa_logreturn.pkl",
            "lag20_released_alfa_logreturn.pkl"
        ]
        
        for filename in output_files:
            filepath = data_root / filename
            if filepath.exists():
                size_mb = filepath.stat().st_size / (1024 * 1024)
                print(f"  [v] {filename} ({size_mb:.1f} MB)")
            else:
                print(f"  [x] {filename} (未生成)")
                
        return True
        
    except Exception as e:
        logger.error(f"数据处理失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)