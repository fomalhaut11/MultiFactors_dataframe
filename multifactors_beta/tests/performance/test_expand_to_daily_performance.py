#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日频扩展方法性能对比测试

对比三种实现方式：
1. 当前实现（iterrows循环）
2. merge_asof方案
3. reindex + ffill方案
"""

import pandas as pd
import numpy as np
import time
import logging
from pathlib import Path
from typing import Dict, Callable
import tracemalloc

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ExpandToDailyBenchmark:
    """日频扩展方法性能基准测试"""

    def __init__(self):
        """初始化测试环境"""
        self.data_root = Path("E:/Documents/PythonProject/StockProject/StockData")
        self.results = {}

    def load_test_data(self, sample_size: int = 100):
        """
        加载测试数据

        Parameters:
        -----------
        sample_size : int
            采样股票数量，用于控制测试规模
        """
        logger.info(f"加载测试数据，采样{sample_size}只股票...")

        # 加载财务数据
        financial_data = pd.read_pickle(self.data_root / "auxiliary" / "FinancialData_unified.pkl")

        # 采样股票
        all_stocks = financial_data.index.get_level_values('StockCodes').unique()
        sampled_stocks = np.random.choice(all_stocks, min(sample_size, len(all_stocks)), replace=False)
        financial_data = financial_data[financial_data.index.get_level_values('StockCodes').isin(sampled_stocks)]

        # 加载辅助数据
        release_dates = pd.read_pickle(self.data_root / "auxiliary" / "ReleaseDates.pkl")
        trading_dates = pd.read_pickle(self.data_root / "auxiliary" / "TradingDates.pkl")

        # 创建测试因子数据（选择一个列）
        if 'TOT_OPER_REV' in financial_data.columns:
            factor_data = financial_data[['TOT_OPER_REV']].copy()
        else:
            factor_data = financial_data.iloc[:, :1].copy()

        logger.info(f"测试数据准备完成:")
        logger.info(f"  - 股票数: {len(sampled_stocks)}")
        logger.info(f"  - 财报记录: {len(factor_data)}")
        logger.info(f"  - 交易日数: {len(trading_dates)}")

        return factor_data, release_dates, trading_dates

    def method_current_implementation(self,
                                     factor_data: pd.DataFrame,
                                     release_dates: pd.DataFrame,
                                     trading_dates: pd.DatetimeIndex) -> pd.DataFrame:
        """
        方法1：当前实现（使用iterrows循环）
        这是项目中现有的实现方式
        """
        logger.info("执行方法1: 当前实现（iterrows循环）")

        # 合并财报数据和发布日期
        factor_with_release = factor_data.join(release_dates[['ReleasedDates']], how='inner')
        factor_reset = factor_with_release.reset_index()
        factor_sorted = factor_reset.sort_values(['StockCodes', 'ReleasedDates', 'ReportDates'])

        stock_codes = factor_sorted['StockCodes'].unique()
        results = []

        # 预计算交易日期的索引映射
        trading_dates_index = pd.Series(range(len(trading_dates)), index=trading_dates)

        # 逐股票处理
        for stock_code in stock_codes:
            stock_data = factor_sorted[factor_sorted['StockCodes'] == stock_code].copy()

            # 创建日频数据框架
            daily_result = pd.DataFrame(
                index=trading_dates,
                columns=factor_data.columns,
                dtype=float
            )

            # iterrows循环填充
            for _, row in stock_data.iterrows():
                release_date = row['ReleasedDates']

                # 找到生效交易日
                if release_date in trading_dates:
                    effective_date = release_date
                else:
                    future_dates = trading_dates[trading_dates > release_date]
                    effective_date = future_dates.min() if len(future_dates) > 0 else None

                if effective_date is None:
                    continue

                effective_idx = trading_dates_index.get(effective_date)
                if effective_idx is None:
                    continue

                mask = trading_dates_index >= effective_idx
                valid_dates = trading_dates_index.index[mask]

                # 填充数据
                for col in daily_result.columns:
                    if col in row and pd.notna(row[col]):
                        daily_result.loc[valid_dates, col] = row[col]

            daily_result['StockCodes'] = stock_code
            results.append(daily_result)

        # 合并结果
        expanded = pd.concat(results, ignore_index=False)
        expanded = expanded.reset_index().rename(columns={'index': 'TradingDates'})
        expanded = expanded.set_index(['TradingDates', 'StockCodes'])

        return expanded[factor_data.columns]

    def method_merge_asof(self,
                         factor_data: pd.DataFrame,
                         release_dates: pd.DataFrame,
                         trading_dates: pd.DatetimeIndex) -> pd.DataFrame:
        """
        方法2：使用merge_asof进行时间对齐
        完全向量化，避免循环
        """
        logger.info("执行方法2: merge_asof时间对齐")

        # 合并财报数据和发布日期
        data_with_release = factor_data.join(release_dates[['ReleasedDates']], how='inner')
        data_reset = data_with_release.reset_index()

        all_stocks = data_reset['StockCodes'].unique()
        result_pieces = []

        # 按股票处理（但内部完全向量化）
        for stock in all_stocks:
            stock_financial = data_reset[data_reset['StockCodes'] == stock].sort_values('ReleasedDates')
            stock_dates = pd.DataFrame({
                'TradingDates': trading_dates,
                'StockCodes': stock
            })

            # merge_asof: 对每个交易日找到最近的已发布财报
            merged = pd.merge_asof(
                stock_dates.sort_values('TradingDates'),
                stock_financial,
                left_on='TradingDates',
                right_on='ReleasedDates',
                direction='backward',
                suffixes=('', '_financial')
            )
            # 保留正确的StockCodes列
            if 'StockCodes_financial' in merged.columns:
                merged = merged.drop(columns=['StockCodes_financial'])
            result_pieces.append(merged)

        # 合并所有股票
        result = pd.concat(result_pieces, ignore_index=True)

        # 确保StockCodes列存在
        if 'StockCodes' not in result.columns:
            raise ValueError("StockCodes列缺失")
        result = result.set_index(['TradingDates', 'StockCodes'])

        return result[factor_data.columns]

    def method_reindex_ffill(self,
                            factor_data: pd.DataFrame,
                            release_dates: pd.DataFrame,
                            trading_dates: pd.DatetimeIndex) -> pd.DataFrame:
        """
        方法3：使用reindex + ffill
        利用pandas内置的向量化方法
        """
        logger.info("执行方法3: reindex + ffill")

        # 合并财报数据和发布日期
        data_with_release = factor_data.join(release_dates[['ReleasedDates']], how='inner')
        data_reset = data_with_release.reset_index()

        result_list = []

        # 按股票分组
        for stock, group in data_reset.groupby('StockCodes'):
            # 为该股票的每个发布日期设置财报值
            # 创建一个Series，索引为发布日期
            stock_series_dict = {}

            for col in factor_data.columns:
                # 提取该列的值，以ReleasedDates为索引
                col_data = group.set_index('ReleasedDates')[col]

                # 如果同一天有多个财报，取最后一个（最新的）
                col_data = col_data[~col_data.index.duplicated(keep='last')]

                # 排序确保单调性
                col_data = col_data.sort_index()

                # reindex到完整交易日序列，使用ffill前向填充
                col_daily = col_data.reindex(trading_dates, method='ffill')

                stock_series_dict[col] = col_daily

            # 组合为DataFrame
            stock_df = pd.DataFrame(stock_series_dict)
            stock_df['StockCodes'] = stock
            stock_df['TradingDates'] = trading_dates

            result_list.append(stock_df)

        # 合并所有股票
        result = pd.concat(result_list, ignore_index=True)
        result = result.set_index(['TradingDates', 'StockCodes'])

        return result

    def benchmark_method(self,
                        method_func: Callable,
                        method_name: str,
                        factor_data: pd.DataFrame,
                        release_dates: pd.DataFrame,
                        trading_dates: pd.DatetimeIndex) -> Dict:
        """
        基准测试单个方法

        Returns:
        --------
        Dict: 包含时间、内存等性能指标
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"测试方法: {method_name}")
        logger.info(f"{'='*60}")

        # 开始内存追踪
        tracemalloc.start()

        # 执行方法并计时
        start_time = time.time()

        try:
            result = method_func(factor_data, release_dates, trading_dates)

            end_time = time.time()
            elapsed_time = end_time - start_time

            # 获取内存使用
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            # 验证结果
            valid_ratio = result.notna().sum() / len(result) if len(result) > 0 else 0

            metrics = {
                'method': method_name,
                'elapsed_time': elapsed_time,
                'memory_current_mb': current / 1024 / 1024,
                'memory_peak_mb': peak / 1024 / 1024,
                'result_shape': result.shape,
                'valid_ratio': valid_ratio,
                'success': True
            }

            logger.info(f"✅ 执行成功:")
            logger.info(f"  - 耗时: {elapsed_time:.2f}秒")
            logger.info(f"  - 峰值内存: {peak/1024/1024:.2f} MB")
            logger.info(f"  - 结果形状: {result.shape}")
            logger.info(f"  - 有效数据比例: {valid_ratio:.2%}")

            return metrics

        except Exception as e:
            tracemalloc.stop()
            logger.error(f"❌ 执行失败: {e}")
            import traceback
            traceback.print_exc()

            return {
                'method': method_name,
                'success': False,
                'error': str(e)
            }

    def run_benchmark(self, sample_size: int = 100):
        """
        运行完整的性能基准测试

        Parameters:
        -----------
        sample_size : int
            测试使用的股票数量
        """
        logger.info(f"\n{'#'*60}")
        logger.info(f"日频扩展方法性能基准测试")
        logger.info(f"测试规模: {sample_size}只股票")
        logger.info(f"{'#'*60}\n")

        # 加载测试数据
        factor_data, release_dates, trading_dates = self.load_test_data(sample_size)

        # 定义测试方法
        methods = [
            (self.method_current_implementation, "方法1: 当前实现（iterrows）"),
            (self.method_merge_asof, "方法2: merge_asof"),
            (self.method_reindex_ffill, "方法3: reindex + ffill"),
        ]

        results = []

        # 逐个测试
        for method_func, method_name in methods:
            metrics = self.benchmark_method(
                method_func, method_name,
                factor_data, release_dates, trading_dates
            )
            results.append(metrics)

        # 汇总结果
        self._print_summary(results)

        return results

    def _print_summary(self, results: list):
        """打印性能对比汇总"""
        logger.info(f"\n{'='*60}")
        logger.info("性能对比汇总")
        logger.info(f"{'='*60}\n")

        # 创建对比表格
        df_results = pd.DataFrame(results)

        if 'success' in df_results.columns:
            successful = df_results[df_results['success'] == True]

            if len(successful) > 0:
                # 按执行时间排序
                successful = successful.sort_values('elapsed_time')

                print("\n性能排名（按耗时）:")
                print("-" * 80)
                for idx, row in successful.iterrows():
                    print(f"{row['method']:30s} | "
                          f"耗时: {row['elapsed_time']:6.2f}秒 | "
                          f"内存: {row['memory_peak_mb']:7.2f}MB | "
                          f"有效率: {row['valid_ratio']:6.2%}")
                print("-" * 80)

                # 计算加速比
                if len(successful) > 1:
                    baseline_time = successful.iloc[0]['elapsed_time']
                    print("\n加速比（相对于最快方法）:")
                    print("-" * 60)
                    for idx, row in successful.iterrows():
                        speedup = row['elapsed_time'] / baseline_time
                        print(f"{row['method']:30s} | {speedup:5.2f}x")
                    print("-" * 60)

        # 显示失败的方法
        if 'success' in df_results.columns:
            failed = df_results[df_results['success'] == False]
            if len(failed) > 0:
                print("\n失败的方法:")
                for idx, row in failed.iterrows():
                    print(f"  - {row['method']}: {row.get('error', 'Unknown error')}")


def main():
    """主测试函数"""
    benchmark = ExpandToDailyBenchmark()

    # 测试不同规模
    test_sizes = [50, 100, 200]  # 可以根据需要调整

    for size in test_sizes:
        logger.info(f"\n\n{'#'*70}")
        logger.info(f"测试规模: {size}只股票")
        logger.info(f"{'#'*70}\n")

        results = benchmark.run_benchmark(sample_size=size)

        # 稍作暂停
        time.sleep(2)


if __name__ == "__main__":
    main()
