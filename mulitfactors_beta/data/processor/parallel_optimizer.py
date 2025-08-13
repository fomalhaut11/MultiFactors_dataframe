"""
并行处理优化器

提供数据处理的并行优化功能，加速大规模数据计算
"""
import os
import numpy as np
import pandas as pd
from typing import Callable, List, Any, Dict, Optional, Union, Tuple
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count, Manager
import time
import logging
from functools import partial
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


def _execute_single_task(task):
    """
    执行单个任务的全局函数（可被pickle序列化）
    
    Args:
        task: 包含name, func, args, kwargs的任务字典
        
    Returns:
        tuple: (任务名称, 结果)
    """
    name = task['name']
    func = task['func']
    args = task.get('args', ())
    kwargs = task.get('kwargs', {})
    
    start_time = time.time()
    try:
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        logger.info(f"任务 {name} 完成，耗时 {elapsed:.2f}秒")
        return name, result
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"任务 {name} 失败，耗时 {elapsed:.2f}秒，错误: {e}")
        return name, None


class ParallelOptimizer:
    """并行处理优化器"""
    
    def __init__(self, n_workers: Optional[int] = None, 
                 use_process: bool = True,
                 chunk_size: Optional[int] = None):
        """
        初始化并行优化器
        
        Args:
            n_workers: 工作进程/线程数，None表示使用CPU核心数
            use_process: 是否使用进程池，False则使用线程池
            chunk_size: 数据分块大小
        """
        self.n_workers = n_workers or max(1, cpu_count() - 1)
        self.use_process = use_process
        self.chunk_size = chunk_size
        self._progress_manager = None
        
        logger.info(f"初始化并行优化器: {self.n_workers} 个{'进程' if use_process else '线程'}")
        
    def parallel_apply(self, func: Callable, data: pd.DataFrame, 
                      axis: int = 0, **kwargs) -> pd.DataFrame:
        """
        并行应用函数到DataFrame
        
        Args:
            func: 要应用的函数
            data: 输入数据
            axis: 应用轴向 (0=行, 1=列)
            **kwargs: 传递给func的额外参数
            
        Returns:
            处理后的DataFrame
        """
        if data.empty:
            return data
            
        # 根据轴向分割数据
        if axis == 0:
            chunks = self._split_dataframe_rows(data)
        else:
            chunks = self._split_dataframe_cols(data)
            
        # 创建部分函数
        partial_func = partial(self._apply_func_to_chunk, func=func, 
                              axis=axis, **kwargs)
        
        # 并行处理
        results = self._execute_parallel(partial_func, chunks, 
                                       desc=f"并行处理DataFrame")
        
        # 合并结果
        if axis == 0:
            return pd.concat(results, axis=0)
        else:
            return pd.concat(results, axis=1)
            
    def parallel_groupby_apply(self, data: pd.DataFrame, 
                             groupby_cols: Union[str, List[str]],
                             func: Callable, **kwargs) -> pd.DataFrame:
        """
        并行处理groupby操作
        
        Args:
            data: 输入数据
            groupby_cols: 分组列
            func: 应用到每个组的函数
            **kwargs: 传递给func的额外参数
            
        Returns:
            处理后的DataFrame
        """
        # 获取分组
        groups = data.groupby(groupby_cols)
        group_keys = list(groups.groups.keys())
        
        # 创建处理函数
        def process_group(key):
            group_data = groups.get_group(key)
            result = func(group_data, **kwargs)
            return key, result
            
        # 并行处理
        results = self._execute_parallel(
            process_group, group_keys, 
            desc=f"并行处理{len(group_keys)}个分组"
        )
        
        # 重新组装结果
        processed_groups = []
        for key, result in results:
            if isinstance(result, pd.DataFrame):
                result[groupby_cols] = key
            processed_groups.append(result)
            
        return pd.concat(processed_groups, axis=0)
        
    def parallel_calculate_returns(self, price_df: pd.DataFrame,
                                 date_series: pd.DatetimeIndex,
                                 return_types: List[str],
                                 window_sizes: Optional[List[int]] = None) -> Dict[str, pd.DataFrame]:
        """
        并行计算多种收益率
        
        Args:
            price_df: 价格数据
            date_series: 日期序列
            return_types: 收益率类型列表
            window_sizes: 滚动窗口大小列表
            
        Returns:
            收益率字典
        """
        from .return_calculator import ReturnCalculator
        calculator = ReturnCalculator()
        
        tasks = []
        
        # 准备收益率计算任务
        for return_type in return_types:
            task = {
                'name': f'return_{return_type}',
                'func': calculator.calculate_log_return,
                'args': (price_df, date_series),
                'kwargs': {'return_type': return_type}
            }
            tasks.append(task)
            
        # 执行并行计算
        results = self._execute_tasks_parallel(tasks)
        
        # 如果需要计算滚动收益率
        if window_sizes and 'o2o' in return_types:
            base_return = results.get('return_o2o')
            if base_return is not None:
                rolling_tasks = []
                for window in window_sizes:
                    task = {
                        'name': f'return_{window}days',
                        'func': calculator.calculate_n_days_return,
                        'args': (base_return,),
                        'kwargs': {'lag': window}
                    }
                    rolling_tasks.append(task)
                    
                rolling_results = self._execute_tasks_parallel(rolling_tasks)
                results.update(rolling_results)
                
        return results
        
    def optimize_large_matrix_operation(self, matrix_func: Callable,
                                      data: np.ndarray,
                                      block_size: Optional[int] = None) -> np.ndarray:
        """
        优化大矩阵运算
        
        Args:
            matrix_func: 矩阵运算函数
            data: 输入矩阵
            block_size: 分块大小
            
        Returns:
            运算结果
        """
        if data.size < 1000000:  # 小于100万元素直接计算
            return matrix_func(data)
            
        # 分块处理
        block_size = block_size or int(np.sqrt(data.size / self.n_workers))
        blocks = self._split_matrix_blocks(data, block_size)
        
        # 并行处理
        results = self._execute_parallel(
            matrix_func, blocks,
            desc=f"并行处理{len(blocks)}个矩阵块"
        )
        
        # 重组结果
        return self._reassemble_matrix_blocks(results, data.shape, block_size)
        
    def _split_dataframe_rows(self, df: pd.DataFrame) -> List[pd.DataFrame]:
        """按行分割DataFrame"""
        n_chunks = self.n_workers if self.chunk_size is None else len(df) // self.chunk_size
        n_chunks = max(1, min(n_chunks, len(df)))
        return np.array_split(df, n_chunks)
        
    def _split_dataframe_cols(self, df: pd.DataFrame) -> List[pd.DataFrame]:
        """按列分割DataFrame"""
        n_chunks = self.n_workers if self.chunk_size is None else len(df.columns) // self.chunk_size
        n_chunks = max(1, min(n_chunks, len(df.columns)))
        
        col_chunks = np.array_split(df.columns, n_chunks)
        return [df[cols] for cols in col_chunks]
        
    def _split_matrix_blocks(self, matrix: np.ndarray, 
                           block_size: int) -> List[np.ndarray]:
        """将矩阵分割成块"""
        blocks = []
        for i in range(0, matrix.shape[0], block_size):
            for j in range(0, matrix.shape[1], block_size):
                block = matrix[i:i+block_size, j:j+block_size]
                blocks.append(block)
        return blocks
        
    def _reassemble_matrix_blocks(self, blocks: List[np.ndarray],
                                original_shape: Tuple[int, int],
                                block_size: int) -> np.ndarray:
        """重组矩阵块"""
        result = np.zeros(original_shape)
        idx = 0
        for i in range(0, original_shape[0], block_size):
            for j in range(0, original_shape[1], block_size):
                block = blocks[idx]
                result[i:i+block.shape[0], j:j+block.shape[1]] = block
                idx += 1
        return result
        
    def _apply_func_to_chunk(self, chunk: pd.DataFrame, func: Callable,
                           axis: int, **kwargs) -> pd.DataFrame:
        """应用函数到数据块"""
        if axis == 0:
            return chunk.apply(func, axis=1, **kwargs)
        else:
            return chunk.apply(func, axis=0, **kwargs)
            
    def _execute_parallel(self, func: Callable, items: List[Any],
                        desc: str = "并行处理") -> List[Any]:
        """执行并行任务"""
        results = []
        
        # 选择执行器
        Executor = ProcessPoolExecutor if self.use_process else ThreadPoolExecutor
        
        with Executor(max_workers=self.n_workers) as executor:
            # 提交任务
            future_to_item = {
                executor.submit(func, item): i 
                for i, item in enumerate(items)
            }
            
            # 收集结果
            for future in as_completed(future_to_item):
                idx = future_to_item[future]
                try:
                    result = future.result()
                    results.append((idx, result))
                except Exception as e:
                    logger.error(f"任务 {idx} 执行失败: {e}")
                    raise
                    
        # 按原始顺序排序
        results.sort(key=lambda x: x[0])
        return [r[1] for r in results]
        
    def _execute_tasks_parallel(self, tasks: List[Dict]) -> Dict[str, Any]:
        """并行执行多个任务"""
        results = {}
        
        # 并行执行
        task_results = self._execute_parallel(
            _execute_single_task, tasks,
            desc=f"并行执行{len(tasks)}个任务"
        )
        
        # 整理结果
        for name, result in task_results:
            results[name] = result
            
        return results
        

class IncrementalProcessor:
    """增量处理管理器"""
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        初始化增量处理器
        
        Args:
            cache_dir: 缓存目录
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path("./cache/incremental")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.cache_dir / "incremental_metadata.pkl"
        self._load_metadata()
        
    def _load_metadata(self):
        """加载元数据"""
        if self.metadata_file.exists():
            self.metadata = pd.read_pickle(self.metadata_file)
        else:
            self.metadata = pd.DataFrame(columns=[
                'data_type', 'last_processed_date', 'last_processed_time',
                'data_checksum', 'processing_version', 'parameters'
            ])
            
    def _save_metadata(self):
        """保存元数据"""
        self.metadata.to_pickle(self.metadata_file)
        
    def need_update(self, data_type: str, current_date: pd.Timestamp,
                   data_checksum: Optional[str] = None,
                   parameters: Optional[Dict] = None) -> bool:
        """
        检查是否需要更新
        
        Args:
            data_type: 数据类型
            current_date: 当前数据日期
            data_checksum: 数据校验和
            parameters: 处理参数
            
        Returns:
            是否需要更新
        """
        if data_type not in self.metadata.index:
            return True
            
        last_info = self.metadata.loc[data_type]
        
        # 检查日期
        if pd.isna(last_info['last_processed_date']):
            return True
            
        if current_date > last_info['last_processed_date']:
            return True
            
        # 检查数据完整性
        if data_checksum and data_checksum != last_info.get('data_checksum'):
            return True
            
        # 检查参数变化
        if parameters and parameters != last_info.get('parameters'):
            return True
            
        return False
        
    def get_incremental_dates(self, data_type: str,
                            all_dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
        """
        获取需要增量处理的日期
        
        Args:
            data_type: 数据类型
            all_dates: 所有日期
            
        Returns:
            需要处理的日期
        """
        if data_type not in self.metadata.index:
            return all_dates
            
        last_date = self.metadata.loc[data_type, 'last_processed_date']
        if pd.isna(last_date):
            return all_dates
            
        # 返回大于最后处理日期的日期
        return all_dates[all_dates > last_date]
        
    def update_metadata(self, data_type: str, 
                       processed_date: pd.Timestamp,
                       data_checksum: Optional[str] = None,
                       parameters: Optional[Dict] = None):
        """更新处理元数据"""
        self.metadata.loc[data_type] = {
            'last_processed_date': processed_date,
            'last_processed_time': pd.Timestamp.now(),
            'data_checksum': data_checksum,
            'processing_version': '1.0',
            'parameters': parameters
        }
        self._save_metadata()
        
    def get_cached_result(self, data_type: str, 
                         cache_key: str) -> Optional[Any]:
        """获取缓存结果"""
        cache_file = self.cache_dir / f"{data_type}_{cache_key}.pkl"
        if cache_file.exists():
            return pd.read_pickle(cache_file)
        return None
        
    def save_cached_result(self, data_type: str,
                         cache_key: str,
                         data: Any):
        """保存缓存结果"""
        cache_file = self.cache_dir / f"{data_type}_{cache_key}.pkl"
        pd.to_pickle(data, cache_file)
        
    def clean_old_cache(self, days: int = 30):
        """清理旧缓存"""
        cutoff_time = time.time() - (days * 24 * 3600)
        
        for cache_file in self.cache_dir.glob("*.pkl"):
            if cache_file.stat().st_mtime < cutoff_time:
                cache_file.unlink()
                logger.info(f"删除旧缓存: {cache_file}")