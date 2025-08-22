#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据格式约定和验证模块

定义从数据库到factors模块的标准数据格式和验证机制
确保数据传递的一致性和可靠性

Author: MultiFactors Team
Date: 2025-08-21
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class DataSchema:
    """数据格式规范定义"""
    name: str                    # 数据类型名称
    required_columns: List[str]  # 必需字段
    optional_columns: List[str]  # 可选字段
    index_columns: List[str]     # 索引字段
    data_types: Dict[str, str]   # 字段数据类型
    constraints: Dict[str, Any]  # 约束条件
    description: str             # 数据描述


class DataSchemas:
    """标准数据格式定义"""
    
    # 价格数据格式
    PRICE_DATA = DataSchema(
        name="price_data",
        required_columns=['code', 'tradingday', 'c', 'adjfactor'],
        optional_columns=['o', 'h', 'l', 'v', 'amt', 'total_shares', 'free_float_shares', 'exchange_id'],
        index_columns=['code', 'tradingday'],
        data_types={
            'code': 'string',
            'tradingday': 'int64',
            'o': 'float64', 'h': 'float64', 'l': 'float64', 'c': 'float64',
            'v': 'float64', 'amt': 'float64', 'adjfactor': 'float64',
            'total_shares': 'float64', 'free_float_shares': 'float64',
            'exchange_id': 'int64'
        },
        constraints={
            'tradingday': {'min': 20100101, 'max': 29991231},
            'c': {'min': 0, 'max': np.inf},
            'adjfactor': {'min': 0, 'max': np.inf},
            'v': {'min': 0, 'max': np.inf}
        },
        description="股票日频价格数据"
    )
    
    # 财务数据格式
    FINANCIAL_DATA = DataSchema(
        name="financial_data",
        required_columns=['code', 'reportday', 'd_year', 'd_quarter'],
        optional_columns=[],  # 财务字段太多，按实际表结构动态验证
        index_columns=['code', 'reportday', 'd_year', 'd_quarter'],
        data_types={
            'code': 'string',
            'reportday': 'datetime64[ns]',
            'd_year': 'int64',
            'd_quarter': 'int64'
        },
        constraints={
            'd_year': {'min': 2000, 'max': 2050},
            'd_quarter': {'min': 1, 'max': 4}
        },
        description="财务报表数据"
    )
    
    # 发布日期数据格式
    RELEASE_DATES = DataSchema(
        name="release_dates",
        required_columns=['StockCodes', 'ReportPeriod', 'ReleasedDates'],
        optional_columns=[],
        index_columns=['StockCodes', 'ReportPeriod'],
        data_types={
            'StockCodes': 'string',
            'ReportPeriod': 'datetime64[ns]',
            'ReleasedDates': 'datetime64[ns]'
        },
        constraints={},
        description="财报发布日期数据"
    )
    
    # 交易日期数据格式
    TRADING_DATES = DataSchema(
        name="trading_dates",
        required_columns=['TradingDates'],
        optional_columns=[],
        index_columns=['TradingDates'],
        data_types={
            'TradingDates': 'datetime64[ns]'
        },
        constraints={},
        description="交易日期列表"
    )
    
    # 因子标准格式（传递给factors模块）
    FACTOR_FORMAT = DataSchema(
        name="factor_format",
        required_columns=[],  # Series数据无列名
        optional_columns=[],
        index_columns=['TradingDates', 'StockCodes'],
        data_types={'values': 'float64'},
        constraints={},
        description="标准因子格式 - MultiIndex Series[TradingDates, StockCodes]"
    )


class DataValidator:
    """数据格式验证器"""
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame, schema: DataSchema, 
                          strict: bool = True) -> Tuple[bool, List[str]]:
        """
        验证DataFrame是否符合指定格式
        
        Parameters
        ----------
        df : pd.DataFrame
            待验证的数据
        schema : DataSchema
            数据格式规范
        strict : bool
            是否严格模式（缺少可选字段也报错）
            
        Returns
        -------
        Tuple[bool, List[str]]
            (是否通过验证, 错误信息列表)
        """
        errors = []
        
        # 检查数据类型
        if not isinstance(df, pd.DataFrame):
            errors.append(f"数据必须是DataFrame类型，当前: {type(df)}")
            return False, errors
        
        # 检查是否为空
        if df.empty:
            errors.append("数据不能为空")
            return False, errors
        
        # 检查必需字段
        missing_required = set(schema.required_columns) - set(df.columns)
        if missing_required:
            errors.append(f"缺少必需字段: {missing_required}")
        
        # 检查可选字段（严格模式）
        if strict:
            missing_optional = set(schema.optional_columns) - set(df.columns)
            if missing_optional:
                errors.append(f"缺少可选字段: {missing_optional}")
        
        # 检查字段数据类型
        for col, expected_dtype in schema.data_types.items():
            if col in df.columns:
                actual_dtype = str(df[col].dtype)
                if not DataValidator._is_compatible_dtype(actual_dtype, expected_dtype):
                    errors.append(f"字段 {col} 数据类型不匹配: 期望 {expected_dtype}, 实际 {actual_dtype}")
        
        # 检查约束条件
        for col, constraint in schema.constraints.items():
            if col in df.columns:
                errors.extend(DataValidator._validate_constraints(df[col], col, constraint))
        
        # 检查重复值
        if schema.index_columns:
            available_index_cols = [col for col in schema.index_columns if col in df.columns]
            if available_index_cols:
                duplicates = df.duplicated(subset=available_index_cols).sum()
                if duplicates > 0:
                    errors.append(f"发现 {duplicates} 行重复数据")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_series(series: pd.Series, schema: DataSchema) -> Tuple[bool, List[str]]:
        """
        验证Series是否符合因子格式
        
        Parameters
        ----------
        series : pd.Series
            待验证的因子数据
        schema : DataSchema
            数据格式规范
            
        Returns
        -------
        Tuple[bool, List[str]]
            (是否通过验证, 错误信息列表)
        """
        errors = []
        
        # 检查数据类型
        if not isinstance(series, pd.Series):
            errors.append(f"数据必须是Series类型，当前: {type(series)}")
            return False, errors
        
        # 检查MultiIndex格式
        if not isinstance(series.index, pd.MultiIndex):
            errors.append("因子数据必须使用MultiIndex格式")
            return False, errors
        
        # 检查索引级别
        if series.index.nlevels != 2:
            errors.append(f"MultiIndex必须有2个级别，当前: {series.index.nlevels}")
        
        # 检查索引名称
        expected_names = schema.index_columns
        if list(series.index.names) != expected_names:
            errors.append(f"索引名称必须为 {expected_names}，当前: {series.index.names}")
        
        # 检查数据类型
        if not np.issubdtype(series.dtype, np.number):
            errors.append(f"因子值必须是数值类型，当前: {series.dtype}")
        
        # 检查缺失值比例
        null_ratio = series.isnull().sum() / len(series)
        if null_ratio > 0.5:
            errors.append(f"缺失值比例过高: {null_ratio:.1%}")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def _is_compatible_dtype(actual: str, expected: str) -> bool:
        """检查数据类型兼容性"""
        # 数值类型兼容性
        numeric_types = ['int64', 'int32', 'float64', 'float32']
        if expected in numeric_types and actual in numeric_types:
            return True
        
        # 字符串类型兼容性
        string_types = ['object', 'string']
        if expected in string_types and actual in string_types:
            return True
        
        # 日期类型兼容性
        datetime_types = ['datetime64[ns]', 'datetime64']
        if expected in datetime_types and actual in datetime_types:
            return True
        
        return actual == expected
    
    @staticmethod
    def _validate_constraints(series: pd.Series, col_name: str, constraints: Dict) -> List[str]:
        """验证约束条件"""
        errors = []
        
        # 数值范围约束
        if 'min' in constraints:
            min_val = series.min()
            if min_val < constraints['min']:
                errors.append(f"字段 {col_name} 存在小于最小值的数据: {min_val} < {constraints['min']}")
        
        if 'max' in constraints:
            max_val = series.max()
            if max_val > constraints['max']:
                errors.append(f"字段 {col_name} 存在大于最大值的数据: {max_val} > {constraints['max']}")
        
        # 非空约束
        if constraints.get('not_null', False):
            null_count = series.isnull().sum()
            if null_count > 0:
                errors.append(f"字段 {col_name} 不能包含空值，发现 {null_count} 个空值")
        
        return errors


class DataConverter:
    """数据格式转换器"""
    
    @staticmethod
    def price_to_factor_format(price_df: pd.DataFrame, 
                              value_column: str = 'c',
                              date_column: str = 'tradingday',
                              stock_column: str = 'code') -> pd.Series:
        """
        将价格DataFrame转换为标准因子格式
        
        Parameters
        ----------
        price_df : pd.DataFrame
            价格数据
        value_column : str
            数值列名
        date_column : str
            日期列名
        stock_column : str
            股票代码列名
            
        Returns
        -------
        pd.Series
            标准因子格式的Series
        """
        # 数据预处理
        df = price_df.copy()
        
        # 转换日期格式
        if df[date_column].dtype != 'datetime64[ns]':
            if df[date_column].dtype in ['int64', 'int32']:
                # 处理20241201格式的日期
                df[date_column] = pd.to_datetime(df[date_column].astype(str), format='%Y%m%d')
            else:
                df[date_column] = pd.to_datetime(df[date_column])
        
        # 设置MultiIndex
        df = df.set_index([date_column, stock_column])
        df.index.names = ['TradingDates', 'StockCodes']
        
        # 提取目标列作为Series
        factor_series = df[value_column]
        
        # 排序索引
        factor_series = factor_series.sort_index()
        
        return factor_series
    
    @staticmethod
    def financial_to_factor_format(financial_df: pd.DataFrame,
                                  value_column: str,
                                  date_column: str = 'reportday',
                                  stock_column: str = 'code') -> pd.Series:
        """
        将财务DataFrame转换为标准因子格式
        
        Parameters
        ----------
        financial_df : pd.DataFrame
            财务数据
        value_column : str
            数值列名
        date_column : str
            日期列名  
        stock_column : str
            股票代码列名
            
        Returns
        -------
        pd.Series
            标准因子格式的Series
        """
        df = financial_df.copy()
        
        # 确保日期格式
        if df[date_column].dtype != 'datetime64[ns]':
            df[date_column] = pd.to_datetime(df[date_column])
        
        # 设置MultiIndex
        df = df.set_index([date_column, stock_column])
        df.index.names = ['TradingDates', 'StockCodes']
        
        # 提取目标列
        factor_series = df[value_column]
        
        # 排序索引
        factor_series = factor_series.sort_index()
        
        return factor_series


class DataQualityChecker:
    """数据质量检查器"""
    
    @staticmethod
    def generate_quality_report(df: pd.DataFrame, 
                               schema: DataSchema) -> Dict[str, Any]:
        """
        生成数据质量报告
        
        Parameters
        ----------
        df : pd.DataFrame
            待检查的数据
        schema : DataSchema
            数据格式规范
            
        Returns
        -------
        Dict[str, Any]
            数据质量报告
        """
        report = {
            'data_name': schema.name,
            'check_time': datetime.now(),
            'basic_info': {},
            'quality_metrics': {},
            'issues': []
        }
        
        # 基本信息
        report['basic_info'] = {
            'shape': df.shape,
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
            'dtypes': df.dtypes.to_dict()
        }
        
        # 质量指标
        report['quality_metrics'] = {
            'completeness': {},
            'consistency': {},
            'validity': {}
        }
        
        # 完整性检查
        for col in df.columns:
            null_count = df[col].isnull().sum()
            total_count = len(df)
            completeness = 1 - (null_count / total_count)
            report['quality_metrics']['completeness'][col] = {
                'completeness_ratio': completeness,
                'null_count': null_count,
                'total_count': total_count
            }
            
            # 记录完整性问题
            if completeness < 0.9:  # 90%完整性阈值
                report['issues'].append({
                    'type': 'completeness',
                    'column': col,
                    'description': f'字段 {col} 完整性较低: {completeness:.1%}'
                })
        
        # 一致性检查
        if schema.index_columns:
            available_index_cols = [col for col in schema.index_columns if col in df.columns]
            if available_index_cols:
                duplicate_count = df.duplicated(subset=available_index_cols).sum()
                report['quality_metrics']['consistency']['duplicate_count'] = duplicate_count
                
                if duplicate_count > 0:
                    report['issues'].append({
                        'type': 'consistency',
                        'column': available_index_cols,
                        'description': f'发现 {duplicate_count} 行重复数据'
                    })
        
        # 有效性检查
        for col, constraint in schema.constraints.items():
            if col in df.columns:
                validity_issues = DataValidator._validate_constraints(df[col], col, constraint)
                if validity_issues:
                    for issue in validity_issues:
                        report['issues'].append({
                            'type': 'validity',
                            'column': col,
                            'description': issue
                        })
        
        return report
    
    @staticmethod
    def print_quality_report(report: Dict[str, Any]):
        """打印数据质量报告"""
        print(f"\n📊 数据质量报告 - {report['data_name']}")
        print("=" * 60)
        
        # 基本信息
        basic = report['basic_info']
        print(f"📋 基本信息:")
        print(f"  数据形状: {basic['shape']}")
        print(f"  内存使用: {basic['memory_usage_mb']:.1f}MB")
        
        # 完整性
        print(f"\n✅ 完整性检查:")
        completeness = report['quality_metrics']['completeness']
        for col, metrics in list(completeness.items())[:5]:  # 只显示前5个字段
            ratio = metrics['completeness_ratio']
            status = "✓" if ratio >= 0.9 else "⚠️"
            print(f"  {status} {col}: {ratio:.1%}")
        
        if len(completeness) > 5:
            print(f"  ... 还有 {len(completeness) - 5} 个字段")
        
        # 问题汇总
        issues = report['issues']
        if issues:
            print(f"\n⚠️ 发现 {len(issues)} 个问题:")
            for issue in issues[:3]:  # 只显示前3个问题
                print(f"  • {issue['description']}")
            if len(issues) > 3:
                print(f"  ... 还有 {len(issues) - 3} 个问题")
        else:
            print(f"\n✅ 未发现数据质量问题")
        
        print("=" * 60)


# 预定义验证函数
def validate_price_data(df: pd.DataFrame, strict: bool = False) -> Tuple[bool, List[str]]:
    """验证价格数据格式"""
    return DataValidator.validate_dataframe(df, DataSchemas.PRICE_DATA, strict)

def validate_financial_data(df: pd.DataFrame, strict: bool = False) -> Tuple[bool, List[str]]:
    """验证财务数据格式"""
    return DataValidator.validate_dataframe(df, DataSchemas.FINANCIAL_DATA, strict)

def validate_factor_format(series: pd.Series) -> Tuple[bool, List[str]]:
    """验证因子格式"""
    return DataValidator.validate_series(series, DataSchemas.FACTOR_FORMAT)

def convert_to_factor_format(df: pd.DataFrame, value_col: str, 
                           date_col: str = 'tradingday', 
                           stock_col: str = 'code') -> pd.Series:
    """转换为标准因子格式"""
    return DataConverter.price_to_factor_format(df, value_col, date_col, stock_col)