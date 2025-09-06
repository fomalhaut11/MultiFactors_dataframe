"""
Alpha191 批量计算接口

提供高效的批量因子计算功能
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Union, Any, Callable
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import warnings

from .alpha191_base import Alpha191Base, Alpha191DataAdapter
from .alpha191_calculator import Alpha191Calculator
from .alpha191_factors import (
    ALPHA_FACTOR_REGISTRY, 
    create_alpha_factors, 
    get_implemented_alphas,
    ALPHA_GROUPS
)

logger = logging.getLogger(__name__)

# 忽略计算过程中的警告
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)


class Alpha191BatchCalculator:
    """
    Alpha191 批量计算器
    
    支持高效的批量因子计算和管理
    """
    
    def __init__(self, 
                 n_jobs: int = 1,
                 enable_cache: bool = True,
                 cache_dir: Optional[str] = None):
        """
        Parameters
        ----------
        n_jobs : int
            并行作业数，1表示串行
        enable_cache : bool
            是否启用缓存
        cache_dir : str, optional
            缓存目录
        """
        self.n_jobs = n_jobs
        self.enable_cache = enable_cache
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        # 计算统计
        self.calculation_stats = {
            'total_requested': 0,
            'successful': 0,
            'failed': 0,
            'cached_hits': 0,
            'total_time': 0.0
        }
        
        logger.info(f"Alpha191 批量计算器初始化: n_jobs={n_jobs}, cache={enable_cache}")
    
    def calculate_factors(self, 
                         price_data: pd.DataFrame,
                         alpha_nums: Optional[List[int]] = None,
                         benchmark_data: Optional[pd.DataFrame] = None,
                         save_results: bool = True,
                         output_dir: Optional[str] = None,
                         **kwargs) -> Dict[int, pd.Series]:
        """
        批量计算 Alpha 因子
        
        Parameters
        ----------
        price_data : pd.DataFrame
            价格数据，MultiIndex [TradingDates, StockCodes]
        alpha_nums : list, optional
            要计算的因子编号列表，默认计算所有已实现因子
        benchmark_data : pd.DataFrame, optional
            基准数据
        save_results : bool
            是否保存结果
        output_dir : str, optional
            输出目录
        **kwargs
            其他参数
            
        Returns
        -------
        dict
            因子编号到计算结果的映射
        """
        start_time = time.time()
        
        # 确定要计算的因子
        if alpha_nums is None:
            alpha_nums = get_implemented_alphas()
        
        # 验证输入
        alpha_nums = self._validate_alpha_nums(alpha_nums)
        
        self.calculation_stats['total_requested'] = len(alpha_nums)
        
        logger.info(f"开始批量计算 {len(alpha_nums)} 个 Alpha 因子")
        
        # 数据预处理
        if not self._validate_data(price_data):
            logger.error("输入数据验证失败")
            return {}
        
        # 执行计算
        if self.n_jobs == 1 or len(alpha_nums) == 1:
            results = self._calculate_serial(price_data, alpha_nums, benchmark_data, **kwargs)
        else:
            results = self._calculate_parallel(price_data, alpha_nums, benchmark_data, **kwargs)
        
        # 保存结果
        if save_results and results:
            self._save_results(results, output_dir)
        
        # 更新统计
        total_time = time.time() - start_time
        self.calculation_stats['total_time'] = total_time
        self.calculation_stats['successful'] = len(results)
        self.calculation_stats['failed'] = len(alpha_nums) - len(results)
        
        # 输出统计信息
        self._log_statistics()
        
        return results
    
    def _validate_alpha_nums(self, alpha_nums: List[int]) -> List[int]:
        """验证因子编号列表"""
        valid_nums = []
        implemented = get_implemented_alphas()
        
        for num in alpha_nums:
            if not isinstance(num, int) or num < 1 or num > 191:
                logger.warning(f"无效的因子编号: {num}，跳过")
                continue
            
            if num not in implemented:
                logger.warning(f"Alpha{num:03d} 尚未实现，跳过")
                continue
            
            valid_nums.append(num)
        
        return valid_nums
    
    def _validate_data(self, price_data: pd.DataFrame) -> bool:
        """验证输入数据"""
        try:
            if not isinstance(price_data, pd.DataFrame):
                logger.error("price_data 必须是 DataFrame")
                return False
            
            if not isinstance(price_data.index, pd.MultiIndex):
                logger.error("price_data 必须使用 MultiIndex")
                return False
            
            required_columns = ['o', 'h', 'l', 'c', 'v', 'vwap']
            missing_columns = [col for col in required_columns if col not in price_data.columns]
            
            if missing_columns:
                logger.error(f"缺少必需列: {missing_columns}")
                return False
            
            # 检查数据量
            date_count = len(price_data.index.get_level_values('TradingDates').unique())
            if date_count < 50:
                logger.warning(f"数据量较少({date_count}个交易日)，可能影响部分因子计算")
            
            return True
            
        except Exception as e:
            logger.error(f"数据验证失败: {e}")
            return False
    
    def _calculate_serial(self, 
                         price_data: pd.DataFrame, 
                         alpha_nums: List[int],
                         benchmark_data: Optional[pd.DataFrame] = None,
                         **kwargs) -> Dict[int, pd.Series]:
        """串行计算因子"""
        logger.info("使用串行模式计算因子")
        
        results = {}
        failed_factors = []
        
        for i, alpha_num in enumerate(alpha_nums, 1):
            logger.info(f"[{i}/{len(alpha_nums)}] 计算 Alpha{alpha_num:03d}")
            
            try:
                # 检查缓存
                if self.enable_cache:
                    cached_result = self._load_from_cache(alpha_num, price_data)
                    if cached_result is not None:
                        results[alpha_num] = cached_result
                        self.calculation_stats['cached_hits'] += 1
                        logger.info(f"  ✅ 从缓存加载 Alpha{alpha_num:03d}")
                        continue
                
                # 计算因子
                factor = ALPHA_FACTOR_REGISTRY[alpha_num]()
                result = factor.calculate(price_data, benchmark_data, **kwargs)
                
                if result is not None and not result.empty:
                    results[alpha_num] = result
                    
                    # 保存到缓存
                    if self.enable_cache:
                        self._save_to_cache(alpha_num, result, price_data)
                    
                    logger.info(f"  ✅ Alpha{alpha_num:03d} 计算成功: {result.shape}")
                else:
                    failed_factors.append(alpha_num)
                    logger.warning(f"  ❌ Alpha{alpha_num:03d} 计算结果为空")
                    
            except Exception as e:
                failed_factors.append(alpha_num)
                logger.error(f"  ❌ Alpha{alpha_num:03d} 计算失败: {e}")
        
        if failed_factors:
            logger.warning(f"计算失败的因子: {failed_factors}")
        
        return results
    
    def _calculate_parallel(self, 
                           price_data: pd.DataFrame, 
                           alpha_nums: List[int],
                           benchmark_data: Optional[pd.DataFrame] = None,
                           **kwargs) -> Dict[int, pd.Series]:
        """并行计算因子"""
        logger.info(f"使用并行模式计算因子 (n_jobs={self.n_jobs})")
        
        results = {}
        
        def calculate_single_factor(alpha_num: int) -> tuple:
            """计算单个因子的函数"""
            try:
                # 检查缓存
                if self.enable_cache:
                    cached_result = self._load_from_cache(alpha_num, price_data)
                    if cached_result is not None:
                        return alpha_num, cached_result, 'cached'
                
                # 计算因子
                factor = ALPHA_FACTOR_REGISTRY[alpha_num]()
                result = factor.calculate(price_data, benchmark_data, **kwargs)
                
                if result is not None and not result.empty:
                    # 保存到缓存
                    if self.enable_cache:
                        self._save_to_cache(alpha_num, result, price_data)
                    
                    return alpha_num, result, 'success'
                else:
                    return alpha_num, None, 'empty'
                    
            except Exception as e:
                return alpha_num, None, f'error: {e}'
        
        # 提交任务
        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            future_to_alpha = {
                executor.submit(calculate_single_factor, alpha_num): alpha_num
                for alpha_num in alpha_nums
            }
            
            # 收集结果
            for future in as_completed(future_to_alpha):
                alpha_num, result, status = future.result()
                
                if status == 'success':
                    results[alpha_num] = result
                    logger.info(f"✅ Alpha{alpha_num:03d} 计算成功")
                elif status == 'cached':
                    results[alpha_num] = result
                    self.calculation_stats['cached_hits'] += 1
                    logger.info(f"✅ Alpha{alpha_num:03d} 从缓存加载")
                else:
                    logger.warning(f"❌ Alpha{alpha_num:03d} 失败: {status}")
        
        return results
    
    def _load_from_cache(self, alpha_num: int, price_data: pd.DataFrame) -> Optional[pd.Series]:
        """从缓存加载结果"""
        if not self.cache_dir or not self.cache_dir.exists():
            return None
        
        try:
            # 生成缓存文件路径
            data_hash = self._get_data_hash(price_data)
            cache_file = self.cache_dir / f"Alpha{alpha_num:03d}_{data_hash}.pkl"
            
            if cache_file.exists():
                return pd.read_pickle(cache_file)
                
        except Exception as e:
            logger.debug(f"缓存加载失败 Alpha{alpha_num:03d}: {e}")
        
        return None
    
    def _save_to_cache(self, alpha_num: int, result: pd.Series, price_data: pd.DataFrame):
        """保存结果到缓存"""
        if not self.cache_dir:
            return
        
        try:
            self.cache_dir.mkdir(exist_ok=True, parents=True)
            
            # 生成缓存文件路径
            data_hash = self._get_data_hash(price_data)
            cache_file = self.cache_dir / f"Alpha{alpha_num:03d}_{data_hash}.pkl"
            
            result.to_pickle(cache_file)
            
        except Exception as e:
            logger.debug(f"缓存保存失败 Alpha{alpha_num:03d}: {e}")
    
    def _get_data_hash(self, price_data: pd.DataFrame) -> str:
        """生成数据哈希值用于缓存"""
        try:
            # 使用数据的基本信息生成简单哈希
            info = (
                len(price_data),
                len(price_data.columns),
                price_data.index.min(),
                price_data.index.max()
            )
            return str(hash(info))[-8:]  # 取后8位
        except:
            return "default"
    
    def _save_results(self, results: Dict[int, pd.Series], output_dir: Optional[str]):
        """保存计算结果"""
        if not output_dir:
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        
        for alpha_num, result in results.items():
            try:
                filename = f"Alpha{alpha_num:03d}_{timestamp}.pkl"
                filepath = output_path / filename
                result.to_pickle(filepath)
                
                logger.info(f"保存 Alpha{alpha_num:03d} -> {filepath}")
                
            except Exception as e:
                logger.error(f"保存 Alpha{alpha_num:03d} 失败: {e}")
    
    def _log_statistics(self):
        """输出统计信息"""
        stats = self.calculation_stats
        
        logger.info("=" * 60)
        logger.info("Alpha191 批量计算统计")
        logger.info("=" * 60)
        logger.info(f"请求计算: {stats['total_requested']} 个因子")
        logger.info(f"计算成功: {stats['successful']} 个因子")
        logger.info(f"计算失败: {stats['failed']} 个因子")
        logger.info(f"缓存命中: {stats['cached_hits']} 个因子")
        logger.info(f"成功率: {stats['successful']/stats['total_requested']*100:.1f}%")
        logger.info(f"总耗时: {stats['total_time']:.2f} 秒")
        
        if stats['successful'] > 0:
            avg_time = stats['total_time'] / stats['successful']
            logger.info(f"平均每个因子: {avg_time:.2f} 秒")
        
        logger.info("=" * 60)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取计算统计信息"""
        return self.calculation_stats.copy()
    
    def clear_cache(self):
        """清理缓存"""
        if self.cache_dir and self.cache_dir.exists():
            for cache_file in self.cache_dir.glob("Alpha*.pkl"):
                try:
                    cache_file.unlink()
                except Exception as e:
                    logger.warning(f"清理缓存文件失败 {cache_file}: {e}")
            
            logger.info(f"缓存已清理: {self.cache_dir}")


# ==================== 便捷函数 ====================

def calculate_alpha_group(group_name: str, 
                         price_data: pd.DataFrame,
                         **kwargs) -> Dict[int, pd.Series]:
    """
    计算指定分组的 Alpha 因子
    
    Parameters
    ----------
    group_name : str
        分组名称
    price_data : pd.DataFrame
        价格数据
    **kwargs
        其他参数
        
    Returns
    -------
    dict
        计算结果
    """
    if group_name not in ALPHA_GROUPS:
        raise ValueError(f"未知分组: {group_name}")
    
    alpha_nums = ALPHA_GROUPS[group_name]
    
    calculator = Alpha191BatchCalculator()
    return calculator.calculate_factors(
        price_data=price_data,
        alpha_nums=alpha_nums,
        **kwargs
    )


def calculate_all_alphas(price_data: pd.DataFrame, **kwargs) -> Dict[int, pd.Series]:
    """
    计算所有已实现的 Alpha 因子
    
    Parameters
    ----------
    price_data : pd.DataFrame
        价格数据
    **kwargs
        其他参数
        
    Returns
    -------
    dict
        计算结果
    """
    calculator = Alpha191BatchCalculator()
    return calculator.calculate_factors(price_data=price_data, **kwargs)