"""
数据验证模块

提供权重数据格式验证和预处理功能
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple, List, Optional
import warnings

logger = logging.getLogger(__name__)

class ValidationError(Exception):
    """数据验证异常"""
    pass

class WeightsValidator:
    """
    权重数据验证器
    
    验证和预处理回测所需的权重数据格式
    """
    
    def __init__(self, tolerance: float = 1e-6):
        """
        初始化验证器
        
        Parameters
        ----------
        tolerance : float
            权重和的容差范围
        """
        self.tolerance = tolerance
        
    def validate_weights_format(self, weights: pd.DataFrame) -> pd.DataFrame:
        """
        验证权重数据格式并进行预处理
        
        Parameters
        ----------
        weights : pd.DataFrame
            权重数据，期望格式：
            - index: 日期 (DatetimeIndex)
            - columns: 股票代码
            - values: 权重值
            
        Returns
        -------
        pd.DataFrame
            验证和预处理后的权重数据
            
        Raises
        ------
        ValidationError
            当数据格式不符合要求时
        """
        logger.info(f"开始验证权重数据，形状: {weights.shape}")
        
        # 1. 基础格式检查
        self._check_basic_format(weights)
        
        # 2. 索引格式检查
        self._check_index_format(weights)
        
        # 3. 数值检查
        self._check_values(weights)
        
        # 4. 权重和检查
        cleaned_weights = self._check_and_fix_weight_sums(weights)
        
        # 5. 缺失值处理
        cleaned_weights = self._handle_missing_values(cleaned_weights)
        
        logger.info("权重数据验证完成")
        return cleaned_weights
    
    def _check_basic_format(self, weights: pd.DataFrame) -> None:
        """检查基础数据格式"""
        if not isinstance(weights, pd.DataFrame):
            raise ValidationError(f"权重数据必须是pandas.DataFrame，当前类型: {type(weights)}")
            
        if weights.empty:
            raise ValidationError("权重数据不能为空")
            
        if weights.shape[0] == 0:
            raise ValidationError("权重数据必须包含至少一个交易日")
            
        if weights.shape[1] == 0:
            raise ValidationError("权重数据必须包含至少一只股票")
            
        logger.debug(f"基础格式检查通过: {weights.shape[0]}天, {weights.shape[1]}只股票")
    
    def _check_index_format(self, weights: pd.DataFrame) -> None:
        """检查日期索引格式"""
        if not isinstance(weights.index, pd.DatetimeIndex):
            try:
                # 尝试转换为日期索引
                weights.index = pd.to_datetime(weights.index)
                logger.warning("索引已自动转换为DatetimeIndex")
            except Exception as e:
                raise ValidationError(f"无法将索引转换为日期格式: {e}")
        
        # 检查是否有重复日期
        if weights.index.duplicated().any():
            duplicated_dates = weights.index[weights.index.duplicated()].tolist()
            raise ValidationError(f"发现重复的交易日期: {duplicated_dates}")
        
        # 检查日期排序
        if not weights.index.is_monotonic_increasing:
            logger.warning("日期索引未按时间顺序排列，将自动排序")
            weights.sort_index(inplace=True)
            
        logger.debug(f"日期索引检查通过: {weights.index[0]} 到 {weights.index[-1]}")
    
    def _check_values(self, weights: pd.DataFrame) -> None:
        """检查权重数值"""
        # 检查数据类型
        non_numeric_cols = []
        for col in weights.columns:
            if not pd.api.types.is_numeric_dtype(weights[col]):
                non_numeric_cols.append(col)
        
        if non_numeric_cols:
            raise ValidationError(f"以下列包含非数值数据: {non_numeric_cols}")
        
        # 检查负权重（如果不允许做空）
        negative_weights = (weights < 0).sum().sum()
        if negative_weights > 0:
            logger.warning(f"发现 {negative_weights} 个负权重值（可能表示做空）")
        
        # 检查异常大的权重
        max_weight = weights.max().max()
        if max_weight > 1.0:
            logger.warning(f"发现单只股票权重超过100%: {max_weight:.2%}")
        
        # 检查无穷大和NaN的初步统计
        inf_count = np.isinf(weights.values).sum()
        if inf_count > 0:
            raise ValidationError(f"权重数据包含 {inf_count} 个无穷大值")
            
        logger.debug("权重数值检查通过")
    
    def _check_and_fix_weight_sums(self, weights: pd.DataFrame) -> pd.DataFrame:
        """检查并修正权重和"""
        daily_sums = weights.sum(axis=1)
        
        # 找出权重和偏离1.0的日期
        deviation_mask = np.abs(daily_sums - 1.0) > self.tolerance
        problematic_dates = daily_sums[deviation_mask]
        
        if len(problematic_dates) > 0:
            logger.warning(f"发现 {len(problematic_dates)} 个交易日的权重和偏离1.0:")
            for date, sum_val in problematic_dates.head(5).items():
                logger.warning(f"  {date.date()}: {sum_val:.6f}")
            
            if len(problematic_dates) > 5:
                logger.warning(f"  ... 还有 {len(problematic_dates) - 5} 个日期")
        
        # 决定是否自动修正
        max_deviation = np.abs(daily_sums - 1.0).max()
        
        if max_deviation > 0.01:  # 偏差超过1%，报错
            raise ValidationError(
                f"权重和偏差过大 (最大偏差: {max_deviation:.2%})，"
                f"请检查数据。偏差最大的日期: {daily_sums.idxmax()}"
            )
        elif max_deviation > self.tolerance:  # 小偏差，自动归一化
            logger.info(f"自动归一化权重 (最大偏差: {max_deviation:.4%})")
            normalized_weights = weights.div(daily_sums, axis=0)
            
            # 验证归一化效果
            new_sums = normalized_weights.sum(axis=1)
            assert np.allclose(new_sums, 1.0, atol=self.tolerance), "归一化失败"
            
            return normalized_weights
        
        logger.debug("权重和检查通过")
        return weights.copy()
    
    def _handle_missing_values(self, weights: pd.DataFrame) -> pd.DataFrame:
        """处理缺失值"""
        missing_count = weights.isnull().sum().sum()
        
        if missing_count == 0:
            logger.debug("无缺失值")
            return weights
        
        logger.warning(f"发现 {missing_count} 个缺失值")
        
        # 按列统计缺失值
        missing_by_stock = weights.isnull().sum()
        stocks_with_missing = missing_by_stock[missing_by_stock > 0]
        
        if len(stocks_with_missing) > 0:
            logger.warning("各股票缺失值统计:")
            for stock, count in stocks_with_missing.items():
                logger.warning(f"  {stock}: {count} 个缺失值")
        
        # 填充策略：用0填充缺失值（表示不持有）
        filled_weights = weights.fillna(0.0)
        
        # 重新归一化
        daily_sums = filled_weights.sum(axis=1)
        zero_sum_days = (daily_sums == 0).sum()
        
        if zero_sum_days > 0:
            logger.error(f"有 {zero_sum_days} 个交易日的权重和为0（所有股票都缺失）")
            zero_dates = daily_sums[daily_sums == 0].index.tolist()
            raise ValidationError(f"以下日期所有股票权重都缺失: {zero_dates[:5]}")
        
        # 归一化非零权重
        normalized_weights = filled_weights.div(daily_sums, axis=0)
        
        logger.info("缺失值处理完成")
        return normalized_weights
    
    def generate_validation_report(self, weights: pd.DataFrame) -> dict:
        """
        生成权重数据验证报告
        
        Parameters
        ----------
        weights : pd.DataFrame
            权重数据
            
        Returns
        -------
        dict
            验证报告
        """
        report = {
            'basic_info': {
                'shape': weights.shape,
                'date_range': (weights.index.min(), weights.index.max()),
                'trading_days': len(weights),
                'stocks_count': len(weights.columns),
                'total_observations': weights.size
            },
            'data_quality': {
                'missing_values': weights.isnull().sum().sum(),
                'missing_ratio': weights.isnull().sum().sum() / weights.size,
                'negative_weights_count': (weights < 0).sum().sum(),
                'zero_weights_count': (weights == 0).sum().sum()
            },
            'weight_statistics': {
                'daily_weight_sums': weights.sum(axis=1).describe(),
                'max_single_weight': weights.max().max(),
                'min_single_weight': weights.min().min(),
                'avg_single_weight': weights.mean().mean()
            },
            'stocks_info': {
                'stock_codes': weights.columns.tolist(),
                'avg_weight_by_stock': weights.mean().sort_values(ascending=False),
                'participation_rate': (weights > 0).mean()  # 每只股票被持有的比例
            }
        }
        
        return report
    
    def print_validation_report(self, weights: pd.DataFrame) -> None:
        """打印验证报告"""
        report = self.generate_validation_report(weights)
        
        print("=" * 60)
        print("权重数据验证报告")
        print("=" * 60)
        
        print("\n📊 基础信息:")
        basic = report['basic_info']
        print(f"  数据形状: {basic['shape']} ({basic['trading_days']}天 × {basic['stocks_count']}股票)")
        print(f"  时间范围: {basic['date_range'][0].date()} 到 {basic['date_range'][1].date()}")
        print(f"  总观测数: {basic['total_observations']:,}")
        
        print("\n🔍 数据质量:")
        quality = report['data_quality']
        print(f"  缺失值: {quality['missing_values']} ({quality['missing_ratio']:.2%})")
        print(f"  负权重: {quality['negative_weights_count']} 个")
        print(f"  零权重: {quality['zero_weights_count']} 个")
        
        print("\n📈 权重统计:")
        stats = report['weight_statistics']
        print(f"  最大单股权重: {stats['max_single_weight']:.2%}")
        print(f"  最小单股权重: {stats['min_single_weight']:.2%}")
        print(f"  平均单股权重: {stats['avg_single_weight']:.2%}")
        
        daily_sums = stats['daily_weight_sums']
        print(f"  每日权重和统计:")
        print(f"    均值: {daily_sums['mean']:.6f}")
        print(f"    标准差: {daily_sums['std']:.6f}")
        print(f"    最小值: {daily_sums['min']:.6f}")
        print(f"    最大值: {daily_sums['max']:.6f}")
        
        print("\n🏢 股票信息:")
        stocks = report['stocks_info']
        print(f"  股票数量: {len(stocks['stock_codes'])}")
        
        print("  平均权重前5名:")
        top_weights = stocks['avg_weight_by_stock'].head()
        for stock, weight in top_weights.items():
            print(f"    {stock}: {weight:.2%}")
        
        print("  参与率前5名:")
        top_participation = stocks['participation_rate'].sort_values(ascending=False).head()
        for stock, rate in top_participation.items():
            print(f"    {stock}: {rate:.1%}")

# 便捷函数
def validate_weights(weights: pd.DataFrame, tolerance: float = 1e-6) -> pd.DataFrame:
    """
    快速验证权重数据的便捷函数
    
    Parameters
    ----------
    weights : pd.DataFrame
        权重数据
    tolerance : float
        权重和的容差
        
    Returns
    -------
    pd.DataFrame
        验证后的权重数据
    """
    validator = WeightsValidator(tolerance=tolerance)
    return validator.validate_weights_format(weights)