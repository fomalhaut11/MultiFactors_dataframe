"""
性能测试基准

比较优化前后的性能差异
"""
import sys
import os
import time
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Callable
import json

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.processor import PriceDataProcessor, ReturnCalculator
from data.processor.optimized_return_calculator import OptimizedReturnCalculator
from data.processor.data_processing_pipeline import DataProcessingPipeline
from data.processor.enhanced_pipeline import EnhancedDataProcessingPipeline


class PerformanceBenchmark:
    """性能测试基准"""
    
    def __init__(self, sample_size: str = "small"):
        """
        初始化性能测试
        
        Args:
            sample_size: 样本大小 ("small", "medium", "large")
        """
        self.sample_size = sample_size
        self.results = {}
        self.test_data = None
        
    def prepare_test_data(self) -> pd.DataFrame:
        """准备测试数据"""
        print(f"准备{self.sample_size}规模测试数据...")
        
        # 加载价格数据
        processor = PriceDataProcessor()
        price_df = pd.read_pickle(processor.price_file_path)
        
        # 根据样本大小选择数据
        if self.sample_size == "small":
            # 最近30天，前100只股票
            dates = price_df.index.get_level_values(0).unique()[-30:]
            stocks = price_df.index.get_level_values(1).unique()[:100]
        elif self.sample_size == "medium":
            # 最近180天，前500只股票
            dates = price_df.index.get_level_values(0).unique()[-180:]
            stocks = price_df.index.get_level_values(1).unique()[:500]
        else:  # large
            # 最近365天，前1000只股票
            dates = price_df.index.get_level_values(0).unique()[-365:]
            stocks = price_df.index.get_level_values(1).unique()[:1000]
            
        # 筛选数据
        mask = (price_df.index.get_level_values(0).isin(dates) & 
                price_df.index.get_level_values(1).isin(stocks))
        self.test_data = price_df[mask]
        
        print(f"测试数据准备完成: {self.test_data.shape}")
        print(f"日期数: {len(dates)}, 股票数: {len(stocks)}")
        
        return self.test_data
        
    def benchmark_return_calculation(self):
        """基准测试收益率计算"""
        if self.test_data is None:
            self.prepare_test_data()
            
        # 获取日期序列
        dates = self.test_data.index.get_level_values(0).unique()
        
        print("\n=== 收益率计算性能测试 ===")
        
        # 1. 测试原始实现
        print("\n1. 原始实现...")
        calculator_original = ReturnCalculator()
        
        start_time = time.time()
        result_original = calculator_original.calculate_log_return(
            self.test_data, dates, return_type="o2o"
        )
        time_original = time.time() - start_time
        
        print(f"   耗时: {time_original:.2f}秒")
        print(f"   结果形状: {result_original.shape}")
        
        # 2. 测试向量化实现
        print("\n2. 向量化实现...")
        calculator_optimized = OptimizedReturnCalculator(use_parallel=False)
        
        start_time = time.time()
        result_vectorized = calculator_optimized.calculate_log_return_vectorized(
            self.test_data, dates, return_type="o2o"
        )
        time_vectorized = time.time() - start_time
        
        print(f"   耗时: {time_vectorized:.2f}秒")
        print(f"   加速比: {time_original/time_vectorized:.2f}x")
        
        # 3. 测试并行实现
        print("\n3. 并行实现...")
        calculator_parallel = OptimizedReturnCalculator(use_parallel=True, n_workers=4)
        
        start_time = time.time()
        results = calculator_parallel.batch_calculate_returns(
            self.test_data,
            {'daily': dates},
            ['o2o'],
            rolling_windows=[5, 20]
        )
        time_parallel = time.time() - start_time
        
        print(f"   耗时: {time_parallel:.2f}秒")
        print(f"   加速比: {time_original/time_parallel:.2f}x")
        
        # 保存结果
        self.results['return_calculation'] = {
            'original': time_original,
            'vectorized': time_vectorized,
            'parallel': time_parallel,
            'speedup_vectorized': time_original / time_vectorized,
            'speedup_parallel': time_original / time_parallel
        }
        
        # 验证结果一致性
        print("\n4. 验证结果一致性...")
        self._verify_consistency(result_original, result_vectorized)
        
    def benchmark_pipeline(self):
        """基准测试完整管道"""
        print("\n=== 完整管道性能测试 ===")
        
        # 1. 原始管道
        print("\n1. 原始管道...")
        pipeline_original = DataProcessingPipeline()
        
        start_time = time.time()
        # 这里应该使用小数据集测试，避免太耗时
        # results_original = pipeline_original.run_full_pipeline()
        time_original = time.time() - start_time
        
        print(f"   耗时: {time_original:.2f}秒")
        
        # 2. 增强管道（不使用并行）
        print("\n2. 增强管道（串行）...")
        pipeline_enhanced = EnhancedDataProcessingPipeline(
            use_parallel=False, use_incremental=False
        )
        
        start_time = time.time()
        # results_enhanced = pipeline_enhanced.run_enhanced_pipeline()
        time_enhanced = time.time() - start_time
        
        print(f"   耗时: {time_enhanced:.2f}秒")
        
        # 3. 增强管道（使用并行）
        print("\n3. 增强管道（并行）...")
        pipeline_parallel = EnhancedDataProcessingPipeline(
            use_parallel=True, use_incremental=False
        )
        
        start_time = time.time()
        # results_parallel = pipeline_parallel.run_enhanced_pipeline()
        time_parallel = time.time() - start_time
        
        print(f"   耗时: {time_parallel:.2f}秒")
        
        # 保存结果
        self.results['pipeline'] = {
            'original': time_original,
            'enhanced_serial': time_enhanced,
            'enhanced_parallel': time_parallel
        }
        
    def benchmark_incremental_processing(self):
        """基准测试增量处理"""
        print("\n=== 增量处理性能测试 ===")
        
        # 模拟增量场景
        pipeline = EnhancedDataProcessingPipeline(
            use_parallel=True, use_incremental=True
        )
        
        # 1. 首次完整处理
        print("\n1. 首次完整处理...")
        start_time = time.time()
        # 执行处理...
        time_full = time.time() - start_time
        
        # 2. 增量更新（无变化）
        print("\n2. 增量更新（无变化）...")
        start_time = time.time()
        # 执行增量处理...
        time_no_change = time.time() - start_time
        
        # 3. 增量更新（小量变化）
        print("\n3. 增量更新（小量变化）...")
        start_time = time.time()
        # 执行增量处理...
        time_small_change = time.time() - start_time
        
        self.results['incremental'] = {
            'full_processing': time_full,
            'no_change': time_no_change,
            'small_change': time_small_change,
            'speedup_no_change': time_full / max(0.001, time_no_change),
            'speedup_small_change': time_full / max(0.001, time_small_change)
        }
        
    def _verify_consistency(self, df1: pd.DataFrame, df2: pd.DataFrame):
        """验证结果一致性"""
        # 对齐索引
        common_index = df1.index.intersection(df2.index)
        df1_aligned = df1.loc[common_index]
        df2_aligned = df2.loc[common_index]
        
        # 计算差异
        diff = np.abs(df1_aligned.values - df2_aligned.values)
        max_diff = np.nanmax(diff)
        mean_diff = np.nanmean(diff)
        
        print(f"   最大差异: {max_diff:.2e}")
        print(f"   平均差异: {mean_diff:.2e}")
        print(f"   一致性: {'[v] 通过' if max_diff < 1e-10 else '[x] 失败'}")
        
    def generate_report(self):
        """生成性能报告"""
        print("\n=== 性能测试报告 ===")
        
        # 生成文本报告
        report_lines = [
            f"性能测试报告",
            f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"测试规模: {self.sample_size}",
            "",
            "收益率计算性能:",
        ]
        
        if 'return_calculation' in self.results:
            rc = self.results['return_calculation']
            report_lines.extend([
                f"  原始实现: {rc['original']:.2f}秒",
                f"  向量化实现: {rc['vectorized']:.2f}秒 (加速 {rc['speedup_vectorized']:.1f}x)",
                f"  并行实现: {rc['parallel']:.2f}秒 (加速 {rc['speedup_parallel']:.1f}x)",
                ""
            ])
            
        # 保存报告
        report_file = project_root / "tests" / f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
            
        print(f"\n报告已保存: {report_file}")
        
        # 保存详细结果
        results_file = report_file.with_suffix('.json')
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
            
        # 生成性能图表
        self._plot_performance()
        
    def _plot_performance(self):
        """绘制性能对比图"""
        if not self.results:
            return
            
        # 准备数据
        if 'return_calculation' in self.results:
            rc = self.results['return_calculation']
            
            # 性能对比条形图
            plt.figure(figsize=(10, 6))
            
            methods = ['原始实现', '向量化', '并行处理']
            times = [rc['original'], rc['vectorized'], rc['parallel']]
            speedups = [1, rc['speedup_vectorized'], rc['speedup_parallel']]
            
            # 创建子图
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # 执行时间对比
            bars1 = ax1.bar(methods, times, color=['red', 'yellow', 'green'])
            ax1.set_ylabel('执行时间 (秒)')
            ax1.set_title('收益率计算执行时间对比')
            
            # 添加数值标签
            for bar, time in zip(bars1, times):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{time:.2f}s', ha='center', va='bottom')
                        
            # 加速比对比
            bars2 = ax2.bar(methods, speedups, color=['red', 'yellow', 'green'])
            ax2.set_ylabel('加速比')
            ax2.set_title('相对于原始实现的加速比')
            ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
            
            # 添加数值标签
            for bar, speedup in zip(bars2, speedups):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{speedup:.1f}x', ha='center', va='bottom')
                        
            plt.tight_layout()
            
            # 保存图表
            chart_file = project_root / "tests" / f"performance_chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(chart_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"性能图表已保存: {chart_file}")
            

def main():
    """主函数"""
    print("开始性能基准测试...")
    
    # 选择测试规模
    benchmark = PerformanceBenchmark(sample_size="small")
    
    # 运行测试
    benchmark.benchmark_return_calculation()
    # benchmark.benchmark_pipeline()  # 完整管道测试比较耗时
    # benchmark.benchmark_incremental_processing()
    
    # 生成报告
    benchmark.generate_report()
    
    print("\n性能测试完成！")
    

if __name__ == "__main__":
    main()