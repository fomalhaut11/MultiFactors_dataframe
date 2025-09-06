"""
动量因子性能基准测试模块
对比新旧实现的性能差异，验证优化效果
"""
import pandas as pd
import numpy as np
import time
import logging
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

# 导入新的优化版实现
from factors.generator.technical.price_factors import (
    MomentumFactor as OptimizedMomentumFactor,
    MultiPeriodMomentumFactory,
    MomentumFactorBase
)

# 导入基类用于创建原始版本对比
from factors.base.factor_base import FactorBase

logger = logging.getLogger(__name__)


class OriginalMomentumFactor(FactorBase):
    """原始版本动量因子（用于性能对比）"""
    
    def __init__(self, window: int = 20):
        super().__init__(name=f'Original_Momentum_{window}', category='technical')
        self.window = window
        self.description = f"Original momentum implementation over {window} days"
        
    def calculate(self, price_data: pd.DataFrame, **kwargs) -> pd.Series:
        """原始的低效实现方式"""
        close_price = price_data['close']
        
        # 低效的实现：使用pct_change而不是优化的累积方式
        momentum = close_price.groupby(level='StockCodes').pct_change(periods=self.window)
        
        # 简单预处理
        momentum = self.preprocess(momentum)
        
        return momentum


class MomentumPerformanceBenchmark:
    """动量因子性能基准测试器"""
    
    def __init__(self):
        self.benchmark_results: Dict[str, Dict] = {}
        
    def generate_test_data(self, 
                          n_stocks: int = 1000,
                          n_days: int = 1000,
                          seed: int = 42) -> pd.DataFrame:
        """
        生成测试用价格数据
        
        Parameters:
        -----------
        n_stocks : 股票数量
        n_days : 交易日数量
        seed : 随机种子
        
        Returns:
        --------
        pd.DataFrame : 模拟的价格数据
        """
        np.random.seed(seed)
        
        # 生成日期和股票代码
        dates = pd.date_range('2020-01-01', periods=n_days, freq='D')
        stocks = [f'stock_{i:04d}' for i in range(n_stocks)]
        
        # 创建MultiIndex
        index = pd.MultiIndex.from_product([dates, stocks], 
                                         names=['TradingDates', 'StockCodes'])
        
        logger.info(f"生成测试数据: {n_stocks} 只股票 × {n_days} 个交易日 = {len(index):,} 条记录")
        
        # 生成价格数据（模拟几何布朗运动）
        n_obs = len(index)
        
        # 生成相关的股票收益率（考虑市场效应）
        market_returns = np.random.normal(0.0005, 0.015, n_days)  # 市场日收益率
        
        # 为每只股票生成beta和特异性收益
        betas = np.random.normal(1.0, 0.3, n_stocks)  # 股票beta
        betas = np.clip(betas, 0.2, 2.0)  # 限制beta范围
        
        # 生成股票收益率
        stock_returns = []
        for i, stock in enumerate(stocks):
            # 系统性收益 + 特异性收益
            systematic_ret = betas[i] * market_returns
            idiosyncratic_ret = np.random.normal(0, 0.02, n_days)
            total_ret = systematic_ret + idiosyncratic_ret
            stock_returns.extend(total_ret)
        
        # 计算累积价格
        returns_array = np.array(stock_returns)
        initial_prices = np.random.uniform(10, 100, n_stocks)
        
        prices = []
        for i, stock in enumerate(stocks):
            stock_returns_series = returns_array[i*n_days:(i+1)*n_days]
            stock_prices = initial_prices[i] * np.cumprod(1 + stock_returns_series)
            prices.extend(stock_prices)
        
        # 创建DataFrame
        price_data = pd.DataFrame({
            'close': prices,
            'open': np.array(prices) * (1 + np.random.normal(0, 0.001, len(prices))),
            'high': np.array(prices) * (1 + np.abs(np.random.normal(0, 0.005, len(prices)))),
            'low': np.array(prices) * (1 - np.abs(np.random.normal(0, 0.005, len(prices)))),
            'volume': np.random.randint(1000, 100000, len(prices)),
            'adjfactor': np.ones(len(prices))  # 简化的复权因子
        }, index=index)
        
        # 确保价格数据的合理性
        price_data['high'] = np.maximum(price_data['high'], price_data['close'])
        price_data['low'] = np.minimum(price_data['low'], price_data['close'])
        
        # 添加一些随机的停牌数据（5%的概率）
        suspension_mask = np.random.random(len(price_data)) < 0.05
        price_data.loc[suspension_mask, 'volume'] = 0
        
        logger.info(f"测试数据生成完成，数据大小: {price_data.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        return price_data
    
    def benchmark_single_factor_performance(self, 
                                          price_data: pd.DataFrame,
                                          window: int = 20) -> Dict[str, Dict]:
        """
        对比单个动量因子的性能
        
        Parameters:
        -----------
        price_data : 价格数据
        window : 动量窗口
        
        Returns:
        --------
        Dict[str, Dict] : 性能对比结果
        """
        logger.info(f"开始单因子性能基准测试 (window={window})")
        
        results = {}
        
        # 1. 测试原始版本
        logger.info("测试原始版本...")
        original_factor = OriginalMomentumFactor(window=window)
        
        start_time = time.time()
        original_result = original_factor.calculate(price_data)
        original_time = time.time() - start_time
        
        results['original'] = {
            'calculation_time': original_time,
            'memory_usage': original_result.memory_usage(deep=True),
            'valid_observations': original_result.count(),
            'result_mean': original_result.mean(),
            'result_std': original_result.std()
        }
        
        # 2. 测试优化版本
        logger.info("测试优化版本...")
        optimized_factor = OptimizedMomentumFactor(window=window)
        
        start_time = time.time()
        optimized_result = optimized_factor.calculate(price_data)
        optimized_time = time.time() - start_time
        
        results['optimized'] = {
            'calculation_time': optimized_time,
            'memory_usage': optimized_result.memory_usage(deep=True),
            'valid_observations': optimized_result.count(),
            'result_mean': optimized_result.mean(),
            'result_std': optimized_result.std()
        }
        
        # 3. 计算性能提升
        speedup = original_time / optimized_time
        memory_reduction = (results['original']['memory_usage'] - 
                           results['optimized']['memory_usage']) / results['original']['memory_usage']
        
        results['comparison'] = {
            'speedup_ratio': speedup,
            'memory_reduction_ratio': memory_reduction,
            'time_saved_seconds': original_time - optimized_time,
            'results_correlation': np.corrcoef(
                original_result.dropna(), 
                optimized_result.dropna()
            )[0, 1] if len(original_result.dropna()) > 0 and len(optimized_result.dropna()) > 0 else np.nan
        }
        
        logger.info(f"单因子基准测试完成:")
        logger.info(f"  - 速度提升: {speedup:.2f}x")
        logger.info(f"  - 内存减少: {memory_reduction:.1%}")
        logger.info(f"  - 结果相关性: {results['comparison']['results_correlation']:.4f}")
        
        return results
    
    def benchmark_multi_period_performance(self, 
                                         price_data: pd.DataFrame,
                                         periods: List[int] = [5, 10, 20, 60, 120]) -> Dict[str, Dict]:
        """
        测试多周期批量生成的性能
        
        Parameters:
        -----------
        price_data : 价格数据
        periods : 动量周期列表
        
        Returns:
        --------
        Dict[str, Dict] : 批量生成性能结果
        """
        logger.info(f"开始多周期批量性能基准测试 (periods={periods})")
        
        results = {}
        
        # 1. 测试逐个生成（模拟原始方式）
        logger.info("测试逐个生成方式...")
        start_time = time.time()
        
        individual_factors = {}
        for period in periods:
            factor = OptimizedMomentumFactor(window=period)
            individual_factors[f'Momentum_{period}d'] = factor.calculate(price_data)
        
        individual_time = time.time() - start_time
        
        results['individual'] = {
            'total_time': individual_time,
            'avg_time_per_factor': individual_time / len(periods),
            'factors_generated': len(individual_factors)
        }
        
        # 2. 测试批量生成（优化方式）
        logger.info("测试批量生成方式...")
        factory = MultiPeriodMomentumFactory(periods=periods)
        
        start_time = time.time()
        batch_factors = factory.generate_momentum_factors(price_data, factor_type='standard')
        batch_time = time.time() - start_time
        
        results['batch'] = {
            'total_time': batch_time,
            'avg_time_per_factor': batch_time / len(periods),
            'factors_generated': len(batch_factors)
        }
        
        # 3. 计算批量优化效果
        batch_speedup = individual_time / batch_time
        time_saved = individual_time - batch_time
        
        results['batch_comparison'] = {
            'speedup_ratio': batch_speedup,
            'time_saved_seconds': time_saved,
            'efficiency_improvement': (individual_time - batch_time) / individual_time
        }
        
        logger.info(f"多周期基准测试完成:")
        logger.info(f"  - 批量生成速度提升: {batch_speedup:.2f}x")
        logger.info(f"  - 节省时间: {time_saved:.2f} 秒")
        logger.info(f"  - 效率提升: {results['batch_comparison']['efficiency_improvement']:.1%}")
        
        return results
    
    def benchmark_memory_efficiency(self, 
                                   data_sizes: List[Tuple[int, int]] = [(100, 500), (500, 1000), (1000, 2000)],
                                   window: int = 20) -> pd.DataFrame:
        """
        测试不同数据规模下的内存效率
        
        Parameters:
        -----------
        data_sizes : 数据规模列表 [(n_stocks, n_days), ...]
        window : 动量窗口
        
        Returns:
        --------
        pd.DataFrame : 内存效率测试结果
        """
        logger.info("开始内存效率基准测试...")
        
        memory_results = []
        
        for n_stocks, n_days in data_sizes:
            logger.info(f"测试规模: {n_stocks} 股票 × {n_days} 天")
            
            # 生成测试数据
            test_data = self.generate_test_data(n_stocks=n_stocks, n_days=n_days)
            data_size_mb = test_data.memory_usage(deep=True).sum() / 1024**2
            
            # 测试优化版本
            factor = OptimizedMomentumFactor(window=window)
            
            # 监控内存使用
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss / 1024**2  # MB
            
            start_time = time.time()
            result = factor.calculate(test_data)
            calculation_time = time.time() - start_time
            
            memory_after = process.memory_info().rss / 1024**2  # MB
            memory_used = memory_after - memory_before
            
            memory_results.append({
                'n_stocks': n_stocks,
                'n_days': n_days,
                'data_size_mb': data_size_mb,
                'calculation_time': calculation_time,
                'memory_used_mb': memory_used,
                'valid_observations': result.count(),
                'memory_efficiency': data_size_mb / memory_used if memory_used > 0 else np.inf,
                'time_per_observation': calculation_time / len(test_data) * 1000  # ms per observation
            })
            
            logger.info(f"  - 计算时间: {calculation_time:.2f}s")
            logger.info(f"  - 内存使用: {memory_used:.1f}MB")
            logger.info(f"  - 内存效率: {memory_results[-1]['memory_efficiency']:.2f}")
        
        df = pd.DataFrame(memory_results)
        logger.info("内存效率基准测试完成")
        
        return df
    
    def generate_performance_report(self, 
                                  price_data: pd.DataFrame,
                                  save_plots: bool = True) -> Dict:
        """
        生成完整的性能基准报告
        
        Parameters:
        -----------
        price_data : 价格数据
        save_plots : 是否保存图表
        
        Returns:
        --------
        Dict : 完整的基准测试报告
        """
        logger.info("生成完整性能基准报告...")
        
        report = {}
        
        # 1. 单因子性能测试
        report['single_factor'] = self.benchmark_single_factor_performance(price_data, window=20)
        
        # 2. 多周期批量测试
        report['multi_period'] = self.benchmark_multi_period_performance(price_data)
        
        # 3. 内存效率测试
        report['memory_efficiency'] = self.benchmark_memory_efficiency()
        
        # 4. 生成可视化图表
        if save_plots:
            self._create_performance_plots(report)
        
        # 5. 生成汇总统计
        report['summary'] = self._generate_summary_stats(report)
        
        logger.info("性能基准报告生成完成")
        
        return report
    
    def _create_performance_plots(self, report: Dict):
        """创建性能对比图表"""
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 单因子性能对比
        ax1 = axes[0, 0]
        methods = ['Original', 'Optimized']
        times = [report['single_factor']['original']['calculation_time'],
                report['single_factor']['optimized']['calculation_time']]
        
        bars = ax1.bar(methods, times, color=['red', 'green'])
        ax1.set_title('单因子计算时间对比')
        ax1.set_ylabel('计算时间 (秒)')
        
        # 添加数值标签
        for bar, time in zip(bars, times):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{time:.3f}s', ha='center', va='bottom')
        
        # 2. 批量生成效率对比
        ax2 = axes[0, 1]
        batch_methods = ['Individual', 'Batch']
        batch_times = [report['multi_period']['individual']['total_time'],
                      report['multi_period']['batch']['total_time']]
        
        bars2 = ax2.bar(batch_methods, batch_times, color=['orange', 'blue'])
        ax2.set_title('多周期生成时间对比')
        ax2.set_ylabel('总计算时间 (秒)')
        
        for bar, time in zip(bars2, batch_times):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{time:.2f}s', ha='center', va='bottom')
        
        # 3. 内存效率曲线
        ax3 = axes[1, 0]
        memory_df = report['memory_efficiency']
        ax3.plot(memory_df['data_size_mb'], memory_df['calculation_time'], 
                'o-', color='purple', linewidth=2, markersize=6)
        ax3.set_title('内存使用 vs 计算时间')
        ax3.set_xlabel('数据大小 (MB)')
        ax3.set_ylabel('计算时间 (秒)')
        ax3.grid(True, alpha=0.3)
        
        # 4. 性能提升汇总
        ax4 = axes[1, 1]
        improvements = {
            'Speed\n(Single Factor)': report['single_factor']['comparison']['speedup_ratio'],
            'Speed\n(Multi-Period)': report['multi_period']['batch_comparison']['speedup_ratio'],
            'Memory\nEfficiency': 1 + report['single_factor']['comparison']['memory_reduction_ratio']
        }
        
        bars4 = ax4.bar(improvements.keys(), improvements.values(), 
                       color=['lightgreen', 'lightblue', 'lightcoral'])
        ax4.set_title('性能提升汇总')
        ax4.set_ylabel('提升倍数')
        ax4.axhline(y=1, color='black', linestyle='--', alpha=0.5)
        
        for bar, value in zip(bars4, improvements.values()):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.2f}x', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('momentum_factor_performance_benchmark.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info("性能对比图表已生成并保存")
    
    def _generate_summary_stats(self, report: Dict) -> Dict:
        """生成汇总统计信息"""
        summary = {
            'single_factor_speedup': report['single_factor']['comparison']['speedup_ratio'],
            'multi_period_speedup': report['multi_period']['batch_comparison']['speedup_ratio'],
            'memory_reduction': report['single_factor']['comparison']['memory_reduction_ratio'],
            'avg_memory_efficiency': report['memory_efficiency']['memory_efficiency'].mean(),
            'correlation_with_original': report['single_factor']['comparison']['results_correlation'],
            'total_time_saved': (report['single_factor']['comparison']['time_saved_seconds'] +
                               report['multi_period']['batch_comparison']['time_saved_seconds'])
        }
        
        return summary


def run_complete_benchmark():
    """运行完整的性能基准测试"""
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    logger.info("开始完整性能基准测试...")
    
    # 创建基准测试器
    benchmark = MomentumPerformanceBenchmark()
    
    # 生成大规模测试数据
    logger.info("生成大规模测试数据...")
    price_data = benchmark.generate_test_data(n_stocks=500, n_days=1000)
    
    # 运行完整基准测试
    report = benchmark.generate_performance_report(price_data)
    
    # 打印汇总结果
    print("\n" + "="*60)
    print("动量因子性能基准测试汇总报告")
    print("="*60)
    
    summary = report['summary']
    print(f"单因子速度提升:     {summary['single_factor_speedup']:.2f}x")
    print(f"批量生成速度提升:   {summary['multi_period_speedup']:.2f}x")
    print(f"内存使用减少:       {summary['memory_reduction']:.1%}")
    print(f"平均内存效率:       {summary['avg_memory_efficiency']:.2f}")
    print(f"与原始实现相关性:   {summary['correlation_with_original']:.4f}")
    print(f"总节省时间:         {summary['total_time_saved']:.2f} 秒")
    
    print("\n详细性能数据:")
    print("-" * 40)
    memory_df = report['memory_efficiency']
    print(memory_df.round(3))
    
    logger.info("完整性能基准测试完成！")
    
    return report


if __name__ == '__main__':
    # 运行基准测试
    benchmark_report = run_complete_benchmark()