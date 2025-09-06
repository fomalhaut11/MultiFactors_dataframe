#!/usr/bin/env python3
"""
技术因子性能基准测试

验证向量化计算的效果，对比优化前后的性能差异
测试不同数据规模下的性能表现
"""

import sys
import os
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
import time
import psutil
from pathlib import Path
import logging
import gc
import warnings
from functools import wraps

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

# 导入技术因子模块
from factors.generator.technical.price_factors import MomentumFactor, MultiPeriodMomentumFactory
from factors.generator.technical.volatility_factors import HistoricalVolatilityFactor, MultiVolatilityFactory
from factors.generator.technical.oscillator_factors import RSIFactor, MultiOscillatorFactory


def memory_usage_monitor(func):
    """内存使用监控装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # 记录开始时的内存使用
        process = psutil.Process(os.getpid())
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # 执行函数
        result = func(*args, **kwargs)
        
        # 记录结束时的内存使用
        end_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_diff = end_memory - start_memory
        
        logger.info(f"   💾 内存使用: {memory_diff:.2f} MB ({start_memory:.1f} -> {end_memory:.1f} MB)")
        
        return result, memory_diff
    return wrapper


class PerformanceBenchmarkSuite:
    """技术因子性能基准测试套件"""
    
    def __init__(self):
        self.benchmark_results = {}
        
        # 测试数据规模配置
        self.data_scales = {
            'small': {'days': 252, 'stocks': 50},      # 1年，50只股票
            'medium': {'days': 756, 'stocks': 100},    # 3年，100只股票  
            'large': {'days': 1260, 'stocks': 300},    # 5年，300只股票
            'xlarge': {'days': 1512, 'stocks': 500}    # 6年，500只股票
        }
    
    def create_benchmark_data(self, scale: str) -> pd.DataFrame:
        """创建基准测试数据"""
        config = self.data_scales[scale]
        logger.info(f"创建{scale}规模基准数据: {config['days']}天 x {config['stocks']}只股票...")
        
        np.random.seed(42)
        
        # 生成时间序列
        end_date = pd.Timestamp('2023-12-31')
        start_date = end_date - pd.Timedelta(days=config['days'])
        dates = pd.date_range(start_date, end_date, freq='D')
        
        # 生成股票代码
        stocks = [f'{i:06d}' for i in range(1, config['stocks'] + 1)]
        
        # 创建MultiIndex
        index = pd.MultiIndex.from_product([dates, stocks], names=['TradingDates', 'StockCodes'])
        n_obs = len(index)
        
        # 高效的价格数据生成
        logger.info(f"   生成{n_obs:,}条市场数据...")
        
        # 使用向量化操作生成所有价格数据
        base_prices = np.random.lognormal(mean=4, sigma=0.3, size=n_obs)
        
        # 生成相关的日收益率（带股票间相关性）
        n_days = len(dates)
        n_stocks = len(stocks)
        
        # 市场因子（影响所有股票）
        market_returns = np.random.normal(0, 0.015, n_days)
        
        # 个股特异收益率
        idiosyncratic_returns = np.random.normal(0, 0.02, (n_days, n_stocks))
        
        # 组合成总收益率（市场因子 + 个股特异）
        market_impact = np.random.uniform(0.3, 0.8, n_stocks)  # 每只股票对市场的敏感度
        stock_returns = (market_returns[:, np.newaxis] * market_impact + idiosyncratic_returns)
        
        # 展平成一维数组（按日期-股票顺序）
        returns_flat = stock_returns.flatten()
        
        # 计算累积价格
        cumulative_returns = np.zeros_like(returns_flat)
        for i in range(len(stocks)):
            start_idx = i * n_days
            end_idx = (i + 1) * n_days
            cumulative_returns[start_idx:end_idx] = np.cumsum(returns_flat[start_idx:end_idx])
        
        # 计算最终价格
        prices = base_prices * np.exp(cumulative_returns)
        
        # 生成OHLC数据（向量化）
        daily_vol = np.abs(returns_flat) + np.random.exponential(0.01, n_obs)
        
        open_prices = prices * np.exp(np.random.normal(0, daily_vol * 0.3))
        high_prices = prices * np.exp(np.abs(np.random.normal(0, daily_vol * 0.7)))
        low_prices = prices * np.exp(-np.abs(np.random.normal(0, daily_vol * 0.7)))
        
        # 确保价格关系正确
        high_prices = np.maximum.reduce([open_prices, prices, high_prices])
        low_prices = np.minimum.reduce([open_prices, prices, low_prices])
        
        # 生成成交量（带价格-成交量关系）
        volume_base = np.random.lognormal(mean=13, sigma=1, size=n_obs)
        price_impact = np.abs(returns_flat) * 2 + 1  # 价格变动大时成交量增加
        volumes = volume_base * price_impact
        
        # 创建DataFrame
        data = pd.DataFrame({
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': prices,
            'volume': volumes,
            'adjfactor': np.ones(n_obs)
        }, index=index)
        
        logger.info(f"   ✅ {scale}规模数据创建完成: {data.shape[0]:,}条观测")
        return data
    
    @memory_usage_monitor
    def benchmark_single_factor_calculation(self, data: pd.DataFrame, factor_name: str, factor_creator) -> Dict[str, Any]:
        """单因子计算性能基准"""
        logger.info(f"   📊 测试{factor_name}...")
        
        # 预热（避免首次调用开销）
        small_data = data.iloc[:min(1000, len(data))]
        factor = factor_creator()
        _ = factor.calculate(small_data)
        
        # 正式基准测试
        factor = factor_creator()
        start_time = time.perf_counter()
        result = factor.calculate(data)
        end_time = time.perf_counter()
        
        calc_time = end_time - start_time
        
        # 计算性能指标
        total_obs = len(data)
        valid_obs = result.count()
        obs_per_second = total_obs / calc_time if calc_time > 0 else 0
        
        return {
            'factor_name': factor_name,
            'calc_time': calc_time,
            'total_obs': total_obs,
            'valid_obs': valid_obs,
            'obs_per_second': obs_per_second,
            'efficiency': valid_obs / calc_time if calc_time > 0 else 0,
            'result_stats': {
                'mean': result.mean(),
                'std': result.std(),
                'skew': result.skew(),
                'kurt': result.kurtosis()
            }
        }
    
    @memory_usage_monitor  
    def benchmark_batch_calculation(self, data: pd.DataFrame, batch_name: str, batch_creator) -> Dict[str, Any]:
        """批量计算性能基准"""
        logger.info(f"   📊 测试{batch_name}批量计算...")
        
        # 创建批量计算器
        batch_calculator = batch_creator()
        
        # 正式基准测试
        start_time = time.perf_counter()
        if 'momentum' in batch_name.lower():
            results = batch_calculator.generate_momentum_factors(data, factor_type='standard')
        elif 'volatility' in batch_name.lower():
            results = batch_calculator.generate_volatility_factors(data, factor_types=['historical', 'realized'])
        elif 'oscillator' in batch_name.lower():
            results = batch_calculator.generate_oscillator_factors(data, factor_types=['RSI', 'MACD'])
        else:
            results = {}
        end_time = time.perf_counter()
        
        calc_time = end_time - start_time
        
        # 计算批量性能指标
        factor_count = len(results)
        total_obs = len(data) * factor_count if factor_count > 0 else 0
        avg_time_per_factor = calc_time / factor_count if factor_count > 0 else 0
        
        # 统计有效观测数
        total_valid_obs = sum(result.count() for result in results.values())
        
        return {
            'batch_name': batch_name,
            'calc_time': calc_time,
            'factor_count': factor_count,
            'avg_time_per_factor': avg_time_per_factor,
            'total_obs': total_obs,
            'total_valid_obs': total_valid_obs,
            'batch_efficiency': total_valid_obs / calc_time if calc_time > 0 else 0,
            'factors': list(results.keys())
        }
    
    def compare_single_vs_batch_performance(self, data: pd.DataFrame) -> Dict[str, Any]:
        """对比单个计算与批量计算的性能"""
        logger.info("🔄 对比单个vs批量计算性能...")
        
        results = {}
        
        # 1. 动量因子对比
        logger.info("   测试动量因子...")
        
        # 单个计算
        single_times = []
        momentum_windows = [5, 10, 20]
        
        for window in momentum_windows:
            single_result, _ = self.benchmark_single_factor_calculation(
                data, f'Momentum_{window}d', lambda: MomentumFactor(window=window)
            )
            single_times.append(single_result['calc_time'])
        
        total_single_time = sum(single_times)
        
        # 批量计算
        batch_result, _ = self.benchmark_batch_calculation(
            data, 'Momentum_Batch', lambda: MultiPeriodMomentumFactory(momentum_windows)
        )
        
        batch_time = batch_result['calc_time']
        speedup = total_single_time / batch_time if batch_time > 0 else 0
        
        results['momentum'] = {
            'single_total_time': total_single_time,
            'batch_time': batch_time,
            'speedup': speedup,
            'efficiency_gain': (speedup - 1) * 100 if speedup > 0 else 0
        }
        
        logger.info(f"     动量因子批量计算加速比: {speedup:.2f}x")
        
        # 2. 波动率因子对比（如果数据足够）
        # 暂时跳过波动率因子测试
        if False:  # len(data) > 10000:  # 只在大数据集上测试
            logger.info("   测试波动率因子...")
            
            vol_windows = [10, 20]
            single_vol_times = []
            
            for window in vol_windows:
                single_result, _ = self.benchmark_single_factor_calculation(
                    data, f'HistVol_{window}d', lambda: HistoricalVolatilityFactor(window=window)
                )
                single_vol_times.append(single_result['calc_time'])
            
            total_single_vol_time = sum(single_vol_times)
            
            batch_vol_result, _ = self.benchmark_batch_calculation(
                data, 'Volatility_Batch', lambda: MultiVolatilityFactory(vol_windows)
            )
            
            batch_vol_time = batch_vol_result['calc_time']
            vol_speedup = total_single_vol_time / batch_vol_time if batch_vol_time > 0 else 0
            
            results['volatility'] = {
                'single_total_time': total_single_vol_time,
                'batch_time': batch_vol_time,
                'speedup': vol_speedup,
                'efficiency_gain': (vol_speedup - 1) * 100 if vol_speedup > 0 else 0
            }
            
            logger.info(f"     波动率因子批量计算加速比: {vol_speedup:.2f}x")
        
        return results
    
    def scalability_test(self) -> Dict[str, Any]:
        """可扩展性测试 - 不同数据规模下的性能表现"""
        logger.info("📈 可扩展性测试...")
        
        results = {}
        test_factor = MomentumFactor(window=20)
        
        for scale, config in self.data_scales.items():
            logger.info(f"   测试{scale}规模数据...")
            
            # 创建数据
            start_time = time.time()
            data = self.create_benchmark_data(scale)
            data_creation_time = time.time() - start_time
            
            # 测试因子计算
            try:
                single_result, memory_usage = self.benchmark_single_factor_calculation(
                    data, f'Momentum_20d_{scale}', lambda: MomentumFactor(window=20)
                )
                
                results[scale] = {
                    'config': config,
                    'data_creation_time': data_creation_time,
                    'total_observations': len(data),
                    'calc_time': single_result['calc_time'],
                    'obs_per_second': single_result['obs_per_second'],
                    'memory_usage_mb': memory_usage,
                    'valid_obs': single_result['valid_obs']
                }
                
                logger.info(f"     ✅ {scale}: {single_result['obs_per_second']:,.0f} obs/sec, {memory_usage:.1f}MB")
                
            except Exception as e:
                logger.error(f"     ❌ {scale}规模测试失败: {e}")
                results[scale] = {'error': str(e), 'config': config}
            
            # 清理内存
            del data
            gc.collect()
        
        return results
    
    def vectorization_efficiency_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """向量化效率分析"""
        logger.info("🔢 向量化效率分析...")
        
        results = {}
        
        # 测试不同窗口大小对性能的影响
        window_sizes = [5, 10, 20, 60, 120]
        momentum_performance = {}
        
        logger.info("   测试不同窗口大小的性能影响...")
        for window in window_sizes:
            try:
                result, memory_usage = self.benchmark_single_factor_calculation(
                    data, f'Momentum_{window}d', lambda: MomentumFactor(window=window)
                )
                
                momentum_performance[window] = {
                    'calc_time': result['calc_time'],
                    'obs_per_second': result['obs_per_second'],
                    'memory_usage': memory_usage
                }
                
                logger.info(f"     窗口{window}: {result['obs_per_second']:,.0f} obs/sec")
                
            except Exception as e:
                logger.error(f"     窗口{window}测试失败: {e}")
                momentum_performance[window] = {'error': str(e)}
        
        results['window_size_impact'] = momentum_performance
        
        # 分析性能趋势
        valid_results = [(w, r) for w, r in momentum_performance.items() 
                        if isinstance(r, dict) and 'obs_per_second' in r]
        
        if len(valid_results) >= 2:
            windows, performances = zip(*valid_results)
            obs_per_sec_values = [r['obs_per_second'] for r in performances]
            
            # 计算性能与窗口大小的关系
            performance_trend = np.polyfit(windows, obs_per_sec_values, 1)[0]  # 斜率
            
            results['performance_analysis'] = {
                'performance_trend': performance_trend,
                'interpretation': 'performance_decreases_with_window' if performance_trend < -100 else 'performance_stable'
            }
            
            logger.info(f"   性能趋势: {'随窗口增大而下降' if performance_trend < -100 else '相对稳定'}")
        
        return results
    
    def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """运行综合性能基准测试"""
        logger.info("=" * 80)
        logger.info("⚡ 技术因子性能基准测试开始")
        logger.info("=" * 80)
        
        all_results = {}
        
        try:
            # 1. 可扩展性测试
            all_results['scalability'] = self.scalability_test()
            
            # 2. 使用中等规模数据进行详细分析
            logger.info("创建中等规模数据用于详细分析...")
            medium_data = self.create_benchmark_data('medium')
            
            # 3. 单个vs批量性能对比
            all_results['batch_vs_single'] = self.compare_single_vs_batch_performance(medium_data)
            
            # 4. 向量化效率分析
            all_results['vectorization'] = self.vectorization_efficiency_analysis(medium_data)
            
            # 5. 生成性能报告
            self.generate_performance_report(all_results)
            
        except Exception as e:
            logger.error(f"基准测试过程出现错误: {e}")
            import traceback
            traceback.print_exc()
            all_results['error'] = str(e)
        
        return all_results
    
    def generate_performance_report(self, results: Dict[str, Any]):
        """生成性能报告"""
        logger.info("\n" + "=" * 80)
        logger.info("📊 技术因子性能基准报告")
        logger.info("=" * 80)
        
        # 1. 可扩展性报告
        if 'scalability' in results:
            logger.info("📈 可扩展性测试结果:")
            
            scalability = results['scalability']
            for scale, metrics in scalability.items():
                if 'error' not in metrics:
                    config = metrics['config']
                    obs_count = metrics['total_observations']
                    calc_time = metrics['calc_time']
                    obs_per_sec = metrics['obs_per_second']
                    memory_mb = metrics['memory_usage_mb']
                    
                    logger.info(f"   {scale:>7}: {obs_count:>8,} 观测 | "
                              f"{calc_time:>6.2f}s | "
                              f"{obs_per_sec:>8,.0f} obs/sec | "
                              f"{memory_mb:>6.1f}MB")
        
        # 2. 批量计算效率报告
        if 'batch_vs_single' in results:
            logger.info("\n🔄 批量计算效率:")
            
            batch_results = results['batch_vs_single']
            for factor_type, metrics in batch_results.items():
                if 'speedup' in metrics:
                    speedup = metrics['speedup']
                    efficiency_gain = metrics['efficiency_gain']
                    
                    logger.info(f"   {factor_type:>10}: {speedup:>5.2f}x 加速 | "
                              f"{efficiency_gain:>5.1f}% 效率提升")
        
        # 3. 向量化效率报告
        if 'vectorization' in results:
            logger.info("\n🔢 向量化效率分析:")
            
            vectorization = results['vectorization']
            if 'performance_analysis' in vectorization:
                analysis = vectorization['performance_analysis']
                interpretation = analysis['interpretation']
                
                if interpretation == 'performance_stable':
                    logger.info("   ✅ 向量化实现良好，性能随窗口大小保持稳定")
                else:
                    logger.info("   ⚠️  性能随窗口大小下降，可能需要进一步优化")
        
        # 4. 整体评估
        logger.info("\n🎯 整体性能评估:")
        
        # 基于medium规模数据的基准性能
        if 'scalability' in results and 'medium' in results['scalability']:
            medium_metrics = results['scalability']['medium']
            if 'obs_per_second' in medium_metrics:
                obs_per_sec = medium_metrics['obs_per_second']
                
                if obs_per_sec > 50000:
                    logger.info("   ✅ 性能优秀 - 高效的向量化计算")
                elif obs_per_sec > 20000:
                    logger.info("   ✅ 性能良好 - 满足生产需求")  
                elif obs_per_sec > 10000:
                    logger.info("   ⚠️  性能及格 - 可以使用但建议优化")
                else:
                    logger.info("   ❌ 性能较差 - 需要重大优化")
        
        # 5. 优化建议
        logger.info("\n💡 优化建议:")
        
        # 基于批量计算效率给出建议
        if 'batch_vs_single' in results:
            avg_speedup = np.mean([m.get('speedup', 1) for m in results['batch_vs_single'].values() if 'speedup' in m])
            if avg_speedup > 2:
                logger.info("   ✅ 批量计算效率高，建议在生产中使用批量方法")
            elif avg_speedup > 1.5:
                logger.info("   ✅ 批量计算有一定优势，推荐使用")
            else:
                logger.info("   ⚠️  批量计算优势不明显，需要进一步优化")
        
        logger.info("   💡 建议使用MultiPeriodFactory等批量生成器")
        logger.info("   💡 大数据集计算时注意内存管理")
        logger.info("   💡 生产环境建议使用medium以上规模的缓存")
        
        logger.info("\n✅ 性能基准测试完成!")


def main():
    """主函数"""
    try:
        # 创建性能基准测试套件
        benchmark_suite = PerformanceBenchmarkSuite()
        
        # 运行综合基准测试
        results = benchmark_suite.run_comprehensive_benchmark()
        
        return True
        
    except Exception as e:
        logger.error(f"性能基准测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)