#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
获取2014年至今的完整历史价格数据
使用小块分批获取，确保数据库稳定性
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import logging
import time
import json
from datetime import datetime, timedelta
from data.fetcher.chunked_price_fetcher import ChunkedPriceFetcher
from config import get_config

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


class HistoricalPriceFetcher:
    """历史价格数据获取器（2014年至今）"""
    
    def __init__(self):
        # 使用更小的分块以确保稳定性
        self.fetcher = ChunkedPriceFetcher(chunk_days=30, chunk_stocks=1000)
        self.data_root = get_config('main.paths.data_root')
        self.progress_file = os.path.join(self.data_root, "price_fetch_progress.json")
        self.temp_dir = os.path.join(self.data_root, "temp_chunks")
        
        # 确保临时目录存在
        os.makedirs(self.temp_dir, exist_ok=True)
        
        logger.info("初始化历史价格数据获取器")
        logger.info(f"分块大小: 30天")
        logger.info(f"临时目录: {self.temp_dir}")
    
    def get_year_chunks(self, start_year: int = 2014, end_year: int = None):
        """
        按年份生成分块列表
        
        Args:
            start_year: 开始年份
            end_year: 结束年份，None表示当前年份
            
        Returns:
            年份分块列表
        """
        if end_year is None:
            end_year = datetime.now().year
        
        years = []
        for year in range(start_year, end_year + 1):
            years.append(year)
        
        logger.info(f"将获取 {len(years)} 年的数据: {start_year}-{end_year}")
        return years
    
    def save_progress(self, completed_years: list, current_year: int = None, current_chunk: int = 0):
        """保存进度信息"""
        progress = {
            'completed_years': completed_years,
            'current_year': current_year,
            'current_chunk': current_chunk,
            'last_update': datetime.now().isoformat()
        }
        
        try:
            with open(self.progress_file, 'w', encoding='utf-8') as f:
                json.dump(progress, f, indent=2, ensure_ascii=False)
            logger.debug(f"进度已保存: {progress}")
        except Exception as e:
            logger.warning(f"保存进度失败: {e}")
    
    def load_progress(self):
        """加载进度信息"""
        if not os.path.exists(self.progress_file):
            return {'completed_years': [], 'current_year': None, 'current_chunk': 0}
        
        try:
            with open(self.progress_file, 'r', encoding='utf-8') as f:
                progress = json.load(f)
            logger.info(f"已加载进度: 完成年份 {len(progress.get('completed_years', []))} 个")
            return progress
        except Exception as e:
            logger.warning(f"加载进度失败: {e}")
            return {'completed_years': [], 'current_year': None, 'current_chunk': 0}
    
    def fetch_year_data(self, year: int, retry_count: int = 3):
        """
        获取指定年份的数据
        
        Args:
            year: 年份
            retry_count: 重试次数
            
        Returns:
            是否成功
        """
        logger.info(f"开始获取 {year} 年数据")
        
        # 设置年份的开始和结束日期
        begin_date = int(f"{year}0101")
        end_date = int(f"{year}1231")
        
        # 当前年份的结束日期设为今天
        if year == datetime.now().year:
            end_date = int(datetime.now().strftime('%Y%m%d'))
        
        for attempt in range(retry_count):
            try:
                logger.info(f"第 {attempt + 1} 次尝试获取 {year} 年数据")
                
                # 获取年份数据
                year_data = self.fetcher.fetch_price_data_chunked(
                    begin_date=begin_date,
                    end_date=end_date,
                    by_date=True,
                    save_intermediate=False  # 年份级别不保存中间文件
                )
                
                if not year_data.empty:
                    # 保存年份数据到临时文件
                    temp_file = os.path.join(self.temp_dir, f"price_{year}.pkl")
                    year_data.to_pickle(temp_file)
                    
                    logger.info(f"{year} 年数据获取成功: {year_data.shape}, 保存到 {temp_file}")
                    return True
                else:
                    logger.warning(f"{year} 年没有数据")
                    return True  # 空数据也算成功
                    
            except Exception as e:
                logger.error(f"获取 {year} 年数据失败 (尝试 {attempt + 1}/{retry_count}): {e}")
                if attempt < retry_count - 1:
                    wait_time = (attempt + 1) * 10  # 递增等待时间
                    logger.info(f"等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"{year} 年数据获取最终失败")
                    return False
        
        return False
    
    def combine_yearly_data(self):
        """合并所有年份数据"""
        logger.info("开始合并年份数据...")
        
        # 查找所有年份临时文件
        temp_files = []
        for file in os.listdir(self.temp_dir):
            if file.startswith('price_') and file.endswith('.pkl'):
                temp_files.append(os.path.join(self.temp_dir, file))
        
        if not temp_files:
            logger.error("没有找到年份数据文件")
            return False
        
        logger.info(f"找到 {len(temp_files)} 个年份数据文件")
        temp_files.sort()  # 按文件名排序
        
        try:
            import pandas as pd
            
            all_data = []
            total_records = 0
            
            # 逐个加载和合并
            for i, temp_file in enumerate(temp_files):
                logger.info(f"加载文件 {i+1}/{len(temp_files)}: {os.path.basename(temp_file)}")
                
                year_data = pd.read_pickle(temp_file)
                if not year_data.empty:
                    all_data.append(year_data)
                    total_records += len(year_data)
                    logger.info(f"  数据量: {len(year_data):,} 条")
            
            if not all_data:
                logger.error("所有年份数据都为空")
                return False
            
            # 合并所有数据
            logger.info(f"合并 {len(all_data)} 个数据集，总记录数: {total_records:,}")
            combined_data = pd.concat(all_data, ignore_index=True)
            
            # 去重和排序
            logger.info("去重和排序...")
            original_count = len(combined_data)
            combined_data = combined_data.drop_duplicates(['code', 'tradingday'])
            combined_data = combined_data.sort_values(['tradingday', 'code'])
            
            logger.info(f"去重完成: {original_count:,} -> {len(combined_data):,}")
            
            # 保存最终的Price.pkl
            logger.info("保存最终的Price.pkl文件...")
            file_path = self.fetcher.save_price_data(combined_data, "Price.pkl")
            
            # 显示统计信息
            logger.info("=" * 60)
            logger.info("数据获取完成统计:")
            logger.info(f"总记录数: {len(combined_data):,}")
            logger.info(f"股票数量: {combined_data['code'].nunique():,}")
            logger.info(f"日期范围: {combined_data['tradingday'].min()} - {combined_data['tradingday'].max()}")
            logger.info(f"文件大小: {os.path.getsize(file_path) / 1024 / 1024:.1f} MB")
            logger.info(f"保存位置: {file_path}")
            logger.info("=" * 60)
            
            return True
            
        except Exception as e:
            logger.error(f"合并数据失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def cleanup_temp_files(self):
        """清理临时文件"""
        try:
            temp_files = os.listdir(self.temp_dir)
            for temp_file in temp_files:
                if temp_file.endswith('.pkl'):
                    os.remove(os.path.join(self.temp_dir, temp_file))
            
            # 删除进度文件
            if os.path.exists(self.progress_file):
                os.remove(self.progress_file)
            
            logger.info(f"清理了 {len(temp_files)} 个临时文件")
        except Exception as e:
            logger.warning(f"清理临时文件失败: {e}")
    
    def fetch_historical_data(self, start_year: int = 2014, resume: bool = True):
        """
        获取历史数据主函数
        
        Args:
            start_year: 开始年份
            resume: 是否从断点续传
        """
        logger.info("开始获取2014年至今的完整历史价格数据")
        
        # 加载进度
        progress = self.load_progress() if resume else {'completed_years': [], 'current_year': None}
        completed_years = progress.get('completed_years', [])
        
        # 获取年份列表
        all_years = self.get_year_chunks(start_year)
        remaining_years = [year for year in all_years if year not in completed_years]
        
        if not remaining_years:
            logger.info("所有年份数据已获取完成，开始合并...")
        else:
            logger.info(f"还需要获取 {len(remaining_years)} 年数据: {remaining_years}")
        
        # 逐年获取数据
        for i, year in enumerate(remaining_years):
            try:
                logger.info(f"进度: {i+1}/{len(remaining_years)} - 正在获取 {year} 年数据")
                
                success = self.fetch_year_data(year)
                
                if success:
                    completed_years.append(year)
                    self.save_progress(completed_years, current_year=year)
                    logger.info(f"{year} 年数据获取成功")
                    
                    # 每年之间稍作休息，减轻数据库压力
                    time.sleep(2)
                else:
                    logger.error(f"{year} 年数据获取失败，停止处理")
                    return False
                    
            except KeyboardInterrupt:
                logger.info("用户中断了数据获取过程")
                logger.info(f"当前进度已保存，已完成年份: {completed_years}")
                return False
            except Exception as e:
                logger.error(f"获取 {year} 年数据时发生错误: {e}")
                return False
        
        # 合并所有年份数据
        logger.info("所有年份数据获取完成，开始合并...")
        success = self.combine_yearly_data()
        
        if success:
            # 清理临时文件
            self.cleanup_temp_files()
            logger.info("历史数据获取和合并完成！")
            return True
        else:
            logger.error("数据合并失败")
            return False


def main():
    """主函数"""
    print("=" * 70)
    print("获取2014年至今的完整历史价格数据")
    print("=" * 70)
    print("数据范围: 2014年1月1日 - 今天")
    print("预计数据量: 1000万+ 条记录")
    print("预计耗时: 1-3小时")
    print("分块策略: 按年份 + 30天小块")
    print("=" * 70)
    
    # 检查是否有进度文件
    progress_file = os.path.join(get_config('main.paths.data_root'), "price_fetch_progress.json")
    has_progress = os.path.exists(progress_file)
    
    if has_progress:
        print("发现之前的进度文件")
        resume = input("是否从断点继续？(Y/n): ").strip().lower()
        resume = resume != 'n'
    else:
        resume = False
        confirm = input("确认开始获取数据？(y/N): ").strip().lower()
        if confirm != 'y':
            print("操作已取消")
            return
    
    # 开始获取数据
    fetcher = HistoricalPriceFetcher()
    
    try:
        start_time = datetime.now()
        logger.info(f"开始时间: {start_time}")
        
        success = fetcher.fetch_historical_data(start_year=2014, resume=resume)
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        if success:
            print("\n" + "=" * 70)
            print("[SUCCESS] 历史价格数据获取完成！")
            print(f"耗时: {duration}")
            print("Price.pkl文件已创建，可以开始因子分析了。")
            print("=" * 70)
        else:
            print("\n" + "=" * 70)
            print("[FAILED] 历史价格数据获取失败")
            print("可以稍后重新运行此脚本继续获取")
            print("=" * 70)
            
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
        print("进度已保存，可以稍后继续")


if __name__ == "__main__":
    main()