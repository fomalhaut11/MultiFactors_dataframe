#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础数据增量更新脚本
专门用于增量更新基础价格数据，避免复杂的并行处理
"""

import sys
import os
import logging
from datetime import datetime
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from data.fetcher.incremental_price_updater import IncrementalPriceUpdater
from core.config_manager import get_config

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'basic_data_update_{datetime.now().strftime("%Y%m%d")}.log')
    ]
)

logger = logging.getLogger(__name__)


def main():
    """主函数 - 增量更新基础价格数据"""
    
    print("="*60)
    print("基础数据增量更新")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    try:
        # 创建增量更新器
        logger.info("初始化价格数据更新器...")
        updater = IncrementalPriceUpdater()
        
        # 检查更新状态
        logger.info("检查数据状态...")
        info = updater.get_update_info()
        
        print(f"\n当前数据状态:")
        print(f"  数据文件: {info.get('price_file', 'N/A')}")
        print(f"  文件大小: {info.get('file_size_mb', 0):.1f} MB")
        print(f"  最新日期: {info.get('latest_date', 'N/A')}")
        print(f"  落后天数: {info.get('days_behind', 0)} 天")
        print(f"  需要更新: {'是' if info.get('days_behind', 0) > 0 else '否'}")
        
        # 执行更新
        if info.get('days_behind', 0) > 0:
            print(f"\n开始增量更新...")
            logger.info(f"开始更新 {info['days_behind']} 天的数据")
            
            # 执行更新
            success = updater.update_price_file()
            
            if success:
                print("\n✅ 基础数据更新完成！")
                logger.info("基础数据更新成功")
                
                # 更新后状态
                new_info = updater.get_update_info()
                print(f"  更新后最新日期: {new_info.get('latest_date', 'N/A')}")
                print(f"  文件大小: {new_info.get('file_size_mb', 0):.1f} MB")
            else:
                print("\n❌ 基础数据更新失败！")
                logger.error("基础数据更新失败")
                return False
        else:
            print(f"\n✅ 基础数据已是最新，无需更新")
            logger.info("基础数据已是最新")
        
        print(f"\n完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        return True
        
    except Exception as e:
        logger.error(f"更新过程出错: {e}")
        print(f"\n❌ 更新失败: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)