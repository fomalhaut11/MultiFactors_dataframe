"""
增强的数据处理管道

集成并行处理、增量处理和进度监控功能
"""
import os
import gc
import time
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable, Tuple
from datetime import datetime, timedelta
import logging
from tqdm import tqdm
import json

from .data_processing_pipeline import DataProcessingPipeline
from .price_processor import PriceDataProcessor
from .optimized_return_calculator import OptimizedReturnCalculator
from .financial_processor import FinancialDataProcessor
from .parallel_optimizer import ParallelOptimizer, IncrementalProcessor
from core.config_manager import get_path


class ProgressMonitor:
    """处理进度监控器"""
    
    def __init__(self, total_tasks: int, task_name: str = "数据处理"):
        """
        初始化进度监控器
        
        Args:
            total_tasks: 总任务数
            task_name: 任务名称
        """
        self.total_tasks = total_tasks
        self.task_name = task_name
        self.completed_tasks = 0
        self.start_time = time.time()
        self.task_times = []
        self.current_task = None
        self.pbar = tqdm(total=total_tasks, desc=task_name)
        
    def start_task(self, task_description: str):
        """开始新任务"""
        self.current_task = {
            'description': task_description,
            'start_time': time.time()
        }
        self.pbar.set_description(f"{self.task_name} - {task_description}")
        
    def complete_task(self, result_info: Optional[Dict] = None):
        """完成当前任务"""
        if self.current_task:
            elapsed = time.time() - self.current_task['start_time']
            self.current_task['elapsed_time'] = elapsed
            self.current_task['result'] = result_info
            self.task_times.append(self.current_task)
            
            self.completed_tasks += 1
            self.pbar.update(1)
            
            # 更新预计剩余时间
            avg_time = np.mean([t['elapsed_time'] for t in self.task_times])
            remaining_tasks = self.total_tasks - self.completed_tasks
            eta = avg_time * remaining_tasks
            
            self.pbar.set_postfix({
                '完成': f'{self.completed_tasks}/{self.total_tasks}',
                '预计剩余': f'{eta/60:.1f}分钟'
            })
            
    def get_summary(self) -> Dict:
        """获取处理摘要"""
        total_time = time.time() - self.start_time
        return {
            'total_tasks': self.total_tasks,
            'completed_tasks': self.completed_tasks,
            'total_time': total_time,
            'average_time_per_task': total_time / max(1, self.completed_tasks),
            'task_details': self.task_times
        }
        
    def close(self):
        """关闭进度条"""
        self.pbar.close()
        

class EnhancedDataProcessingPipeline(DataProcessingPipeline):
    """增强的数据处理管道"""
    
    def __init__(self, config_path: Optional[str] = None,
                 use_parallel: bool = True,
                 use_incremental: bool = True,
                 n_workers: Optional[int] = None):
        """
        初始化增强管道
        
        Args:
            config_path: 配置文件路径
            use_parallel: 是否使用并行处理
            use_incremental: 是否使用增量处理
            n_workers: 并行工作数
        """
        super().__init__(config_path)
        
        # 使用优化的收益率计算器
        self.return_calculator = OptimizedReturnCalculator(
            config_path, use_parallel=use_parallel, n_workers=n_workers
        )
        
        # 并行优化器
        self.parallel_optimizer = ParallelOptimizer(n_workers=n_workers) if use_parallel else None
        
        # 增量处理器
        self.incremental_processor = IncrementalProcessor(
            cache_dir=self.cache_dir / "incremental"
        ) if use_incremental else None
        
        self.use_parallel = use_parallel
        self.use_incremental = use_incremental
        
    def run_enhanced_pipeline(self, 
                            force_full_update: bool = False,
                            skip_unchanged: bool = True,
                            monitor_progress: bool = True) -> Dict[str, Any]:
        """
        运行增强的数据处理流程
        
        Args:
            force_full_update: 强制完整更新
            skip_unchanged: 跳过未变化的数据
            monitor_progress: 监控处理进度
            
        Returns:
            处理结果
        """
        self.logger.info("开始运行增强数据处理管道...")
        
        # 估算任务数
        total_tasks = self._estimate_tasks()
        progress_monitor = ProgressMonitor(total_tasks) if monitor_progress else None
        
        results = {}
        
        try:
            # 1. 处理价格数据
            if progress_monitor:
                progress_monitor.start_task("处理价格数据")
                
            price_df, stock_3d = self._process_price_data(
                force_full_update, skip_unchanged
            )
            results['price_df'] = price_df
            results['stock_3d'] = stock_3d
            
            if progress_monitor:
                progress_monitor.complete_task({
                    'shape': price_df.shape,
                    'stocks': len(stock_3d['StockCodes'])
                })
                
            # 2. 生成日期序列
            if progress_monitor:
                progress_monitor.start_task("生成日期序列")
                
            date_series = self._generate_date_series(price_df)
            
            if progress_monitor:
                progress_monitor.complete_task({
                    'daily': len(date_series['daily']),
                    'weekly': len(date_series['weekly']),
                    'monthly': len(date_series['monthly'])
                })
                
            # 3. 计算收益率（使用增量处理）
            if progress_monitor:
                progress_monitor.start_task("计算收益率")
                
            return_results = self._calculate_returns_incremental(
                price_df, date_series, force_full_update
            )
            results.update(return_results)
            
            if progress_monitor:
                progress_monitor.complete_task({
                    'return_types': len(return_results)
                })
                
            # 释放内存
            price_df = None
            gc.collect()
            
            # 4. 处理财报数据
            if progress_monitor:
                progress_monitor.start_task("处理财报数据")
                
            financial_results = self._process_financial_data_incremental(
                date_series['daily'], force_full_update
            )
            results.update(financial_results)
            
            if progress_monitor:
                progress_monitor.complete_task()
                
            # 5. 清理旧缓存
            if self.incremental_processor:
                self.incremental_processor.clean_old_cache(days=7)
                
            self.logger.info("增强数据处理管道执行完成！")
            
            # 保存处理摘要
            if progress_monitor:
                summary = progress_monitor.get_summary()
                summary_file = self.results_dir / f"processing_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(summary_file, 'w') as f:
                    json.dump(summary, f, indent=2, default=str)
                    
                self.logger.info(f"处理摘要已保存: {summary_file}")
                progress_monitor.close()
                
            return results
            
        except Exception as e:
            self.logger.error(f"处理失败: {e}")
            if progress_monitor:
                progress_monitor.close()
            raise
            
    def _estimate_tasks(self) -> int:
        """估算总任务数"""
        # 基础任务：价格处理(1) + 日期序列(1) + 财报数据(1)
        base_tasks = 3
        
        # 收益率计算任务：3个周期 x 2种类型 + 2个滚动窗口
        return_tasks = 8
        
        return base_tasks + return_tasks
        
    def _process_price_data(self, force_full: bool, skip_unchanged: bool) -> Tuple[pd.DataFrame, Dict]:
        """处理价格数据（支持增量）"""
        
        # 检查是否需要更新
        if self.incremental_processor and not force_full:
            # 获取价格文件的最新日期
            price_file = Path(get_path('data_root')) / "Price.pkl"
            if price_file.exists():
                file_mtime = datetime.fromtimestamp(price_file.stat().st_mtime)
                
                if not self.incremental_processor.need_update(
                    'price_data', pd.Timestamp(file_mtime)
                ):
                    self.logger.info("价格数据未更新，使用缓存")
                    price_df = pd.read_pickle(price_file)
                    stock_3d_file = Path(get_path('data_root')) / "Stock3d.pkl"
                    
                    if stock_3d_file.exists():
                        stock_3d = pd.read_pickle(stock_3d_file)
                        return price_df, stock_3d
                        
        # 执行完整处理
        return self.price_processor.process(save_to_file=True)
        
    def _generate_date_series(self, price_df: pd.DataFrame) -> Dict[str, pd.DatetimeIndex]:
        """生成日期序列"""
        return {
            'daily': self.price_processor.get_date_series(price_df, "daily"),
            'weekly': self.price_processor.get_date_series(price_df, "weekly"),
            'monthly': self.price_processor.get_date_series(price_df, "monthly")
        }
        
    def _calculate_returns_incremental(self, 
                                     price_df: pd.DataFrame,
                                     date_series: Dict[str, pd.DatetimeIndex],
                                     force_full: bool) -> Dict[str, pd.DataFrame]:
        """增量计算收益率"""
        results = {}
        
        # 检查哪些需要更新
        if self.incremental_processor and not force_full:
            # 获取最新数据日期
            latest_date = price_df.index.get_level_values(0).max()
            
            # 检查每种收益率是否需要更新
            return_configs = [
                ('daily_o2o', 'daily', 'o2o'),
                ('daily_vwap', 'daily', 'vwap'),
                ('weekly_o2o', 'weekly', 'o2o'),
                ('weekly_vwap', 'weekly', 'vwap'),
                ('monthly_o2o', 'monthly', 'o2o'),
                ('monthly_vwap', 'monthly', 'vwap')
            ]
            
            need_update = {}
            for key, period, return_type in return_configs:
                need_update[key] = self.incremental_processor.need_update(
                    f'return_{key}', latest_date
                )
                
        else:
            # 强制更新所有
            need_update = {key: True for key in [
                'daily_o2o', 'daily_vwap', 'weekly_o2o', 
                'weekly_vwap', 'monthly_o2o', 'monthly_vwap'
            ]}
            
        # 使用批量计算
        if self.use_parallel:
            # 准备需要计算的任务
            periods_to_calc = {}
            types_to_calc = set()
            
            for key, need in need_update.items():
                if need:
                    period, return_type = key.rsplit('_', 1)
                    if period not in periods_to_calc:
                        periods_to_calc[period] = date_series[period]
                    types_to_calc.add(return_type)
                    
            if periods_to_calc:
                # 批量计算
                calc_results = self.return_calculator.batch_calculate_returns(
                    price_df, periods_to_calc, list(types_to_calc),
                    rolling_windows=[5, 20] if 'daily' in periods_to_calc else None
                )
                
                # 保存结果
                for key, data in calc_results.items():
                    save_path = self.data_save_path / f"LogReturn_{key}.pkl"
                    pd.to_pickle(data, save_path)
                    
                    # 更新增量处理元数据
                    if self.incremental_processor:
                        latest_date = price_df.index.get_level_values(0).max()
                        self.incremental_processor.update_metadata(
                            f'return_{key}', latest_date
                        )
                        
                results.update(calc_results)
                
        else:
            # 串行计算（仅计算需要更新的）
            for key, need in need_update.items():
                if need:
                    period, return_type = key.rsplit('_', 1)
                    self.logger.info(f"计算 {key}...")
                    
                    result = self.return_calculator.calculate_log_return_vectorized(
                        price_df, date_series[period], return_type
                    )
                    
                    save_path = self.data_save_path / f"LogReturn_{key}.pkl"
                    pd.to_pickle(result, save_path)
                    results[key] = result
                    
                    # 更新元数据
                    if self.incremental_processor:
                        latest_date = price_df.index.get_level_values(0).max()
                        self.incremental_processor.update_metadata(
                            f'return_{key}', latest_date
                        )
                        
        return results
        
    def _process_financial_data_incremental(self,
                                          trading_dates: pd.DatetimeIndex,
                                          force_full: bool) -> Dict[str, Any]:
        """增量处理财报数据"""
        results = {}
        
        # 检查是否需要更新
        if self.incremental_processor and not force_full:
            latest_date = trading_dates.max()
            
            if not self.incremental_processor.need_update(
                'financial_data', latest_date
            ):
                self.logger.info("财报数据未更新，跳过处理")
                return results
                
        # 处理财报数据
        released_dates_df = self.financial_processor.get_released_dates_from_h5()
        save_path = self.data_save_path / "released_dates_df.pkl"
        released_dates_df.to_pickle(save_path)
        results['released_dates_df'] = released_dates_df
        
        # 计算时间差
        trading_dates_df = pd.DataFrame(trading_dates.tolist(), columns=["date"])
        released_dates_count_df = self.financial_processor.calculate_released_dates_count(
            released_dates_df, trading_dates_df
        )
        save_path = self.data_save_path / "released_dates_count_df.pkl"
        released_dates_count_df.to_pickle(save_path)
        results['released_dates_count_df'] = released_dates_count_df
        
        # 更新元数据
        if self.incremental_processor:
            self.incremental_processor.update_metadata(
                'financial_data', trading_dates.max()
            )
            
        return results