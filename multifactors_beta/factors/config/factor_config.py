"""
因子计算相关配置
"""
from typing import Dict, List, Any
from dataclasses import dataclass
import pandas as pd


@dataclass
class TimeSeriesConfig:
    """时间序列处理配置"""
    
    # 季度映射配置
    QUARTER_END_DATES = {
        1: "{year}-03-31",  # Q1
        2: "{year}-06-30",  # Q2  
        3: "{year}-09-30",  # Q3
        4: "{year}-12-31"   # Q4/年报
    }
    
    # TTM计算配置
    TTM_CALCULATION_RULES = {
        1: {  # Q1: 当前Q1 + 去年Q4 - 去年Q1
            'current_quarter': 0,
            'add_quarters': [-1],
            'subtract_quarters': [-4]
        },
        2: {  # Q2: 当前Q2 + 去年Q4 - 去年Q2
            'current_quarter': 0,
            'add_quarters': [-2],
            'subtract_quarters': [-4]
        },
        3: {  # Q3: 当前Q3 + 去年Q4 - 去年Q3
            'current_quarter': 0,
            'add_quarters': [-3],
            'subtract_quarters': [-4]
        },
        4: {  # Q4: 直接使用当年数据
            'current_quarter': 0,
            'add_quarters': [],
            'subtract_quarters': []
        }
    }
    
    # 单季度计算配置
    SINGLE_QUARTER_RULES = {
        1: {'use_current': True},  # Q1直接使用
        2: {'subtract_previous': 1},  # Q2 - Q1
        3: {'subtract_previous': 1},  # Q3 - Q2 
        4: {'subtract_previous': 1}   # Q4 - Q3
    }
    
    # 环比计算配置
    QOQ_CALCULATION_RULES = {
        1: {  # Q1 vs Q4（上年Q4单季）
            'current_single': True,
            'previous_calculation': 'q4_minus_q3'
        },
        2: {  # Q2 单季 vs Q1 单季
            'current_single': True,
            'previous_single': True
        },
        3: {  # Q3 单季 vs Q2 单季
            'current_single': True,
            'previous_single': True
        },
        4: {  # Q4 单季 vs Q3 单季
            'current_single': True,
            'previous_single': True
        }
    }


@dataclass 
class FactorConfig:
    """因子计算配置"""
    
    # 列名映射配置
    COLUMN_MAPPINGS = {
        'earnings': 'DEDUCTEDPROFIT',           # 扣非净利润
        'equity': 'EQY_BELONGTO_PARCOMSH',      # 归属母公司股东权益
        'quarter': 'd_quarter',                  # 季度标识
        'year': 'd_year',                       # 年份标识
        'release_date': 'ReleasedDates',        # 财报发布日期
        'revenue': 'OPREVENUE',                 # 营业收入
        'total_assets': 'TOT_ASSETS',           # 总资产
        'cash_flow': 'NETCASHFLOW_OPERATE',     # 经营现金流
        'financial_expense': 'FIN_EXP_IS',      # 财务费用
        'income_tax': 'TAX',                    # 所得税
    }
    
    # 因子默认参数
    FACTOR_DEFAULTS = {
        'EP': {
            'method': 'ttm',
            'valid_methods': ['ttm', 'single_quarter', 'lyr']
        },
        'BP': {
            'equity_method': 'avg',
            'valid_equity_methods': ['current', 'avg', 'lag1']
        },
        'ROE': {
            'earnings_method': 'ttm',
            'equity_method': 'avg',
            'valid_earnings_methods': ['ttm', 'single_quarter'],
            'valid_equity_methods': ['current', 'avg', 'lag1']
        },
        'SUE': {
            'method': 'ttm',
            'window': 4,
            'valid_methods': ['ttm', 'single_quarter']
        },
        'PEG': {
            'earnings_method': 'ttm',
            'growth_periods': 4  # 计算增长率的期数
        },
        'ProfitCost': {
            'method': 'ttm',
            'valid_methods': ['ttm', 'single_quarter', 'lyr']
        }
    }
    
    # 数据验证配置
    VALIDATION_CONFIG = {
        'min_data_points': 1,
        'max_null_ratio': 0.9,  # 最大空值比例
        'outlier_detection': {
            'method': 'iqr',
            'threshold': 3.0
        },
        'data_quality_checks': {
            'check_negative_equity': True,
            'check_zero_market_cap': True,
            'check_missing_quarters': True
        }
    }
    
    # 性能优化配置
    PERFORMANCE_CONFIG = {
        'use_optimized_processor': True,
        'memory_efficient_threshold': 1000,  # 股票数量超过此值时使用内存高效模式
        'chunk_size': 100,                   # 批处理大小
        'enable_parallel': False,            # 是否启用并行处理
        'max_workers': 4                     # 最大并行工作线程数
    }
    
    # 数据预处理配置
    PREPROCESSING_CONFIG = {
        'winsorize': {
            'enabled': True,
            'quantiles': (0.01, 0.99)  # 缩尾处理的分位数
        },
        'standardize': {
            'enabled': True,
            'method': 'zscore',  # zscore, minmax, robust
            'by_date': True      # 是否按日期标准化
        },
        'neutralize': {
            'enabled': False,
            'industry_neutral': True,
            'market_cap_neutral': True
        }
    }


@dataclass
class DatabaseConfig:
    """数据库相关配置"""
    
    # 表名配置
    TABLE_NAMES = {
        'financial': 'financial_data',
        'price': 'price_data', 
        'market_cap': 'market_cap_data',
        'release_dates': 'release_dates',
        'trading_dates': 'trading_dates',
        'stock_info': 'stock_basic_info'
    }
    
    # 字段映射
    FIELD_MAPPINGS = {
        'stock_code': 'StockCodes',
        'trade_date': 'TradingDates',
        'report_date': 'ReportDates',
        'release_date': 'ReleasedDates'
    }


# 全局配置实例
time_series_config = TimeSeriesConfig()
factor_config = FactorConfig()
database_config = DatabaseConfig()


def get_column_name(logical_name: str) -> str:
    """
    根据逻辑名称获取实际列名
    
    Parameters:
    -----------
    logical_name : 逻辑列名（如'earnings', 'equity'等）
    
    Returns:
    --------
    实际的列名
    """
    return factor_config.COLUMN_MAPPINGS.get(logical_name, logical_name)


def get_factor_defaults(factor_name: str) -> Dict[str, Any]:
    """
    获取因子的默认配置
    
    Parameters:
    -----------
    factor_name : 因子名称
    
    Returns:
    --------
    默认配置字典
    """
    return factor_config.FACTOR_DEFAULTS.get(factor_name, {})


def get_quarter_end_date(year: int, quarter: int) -> pd.Timestamp:
    """
    获取季度结束日期
    
    Parameters:
    -----------
    year : 年份
    quarter : 季度（1-4）
    
    Returns:
    --------
    季度结束日期
    """
    date_template = time_series_config.QUARTER_END_DATES.get(quarter)
    if date_template is None:
        raise ValueError(f"Invalid quarter: {quarter}")
    
    date_str = date_template.format(year=year)
    return pd.Timestamp(date_str)


def validate_factor_parameters(factor_name: str, **params) -> None:
    """
    验证因子参数
    
    Parameters:
    -----------
    factor_name : 因子名称
    **params : 参数字典
    
    Raises:
    -------
    ValueError : 如果参数无效
    """
    defaults = get_factor_defaults(factor_name)
    
    for param, value in params.items():
        # 检查有效值
        valid_key = f'valid_{param}s'
        if valid_key in defaults:
            valid_values = defaults[valid_key]
            if value not in valid_values:
                raise ValueError(f"Invalid {param} '{value}' for {factor_name}. Valid options: {valid_values}")


# 配置更新函数
def update_config(config_dict: Dict[str, Any]) -> None:
    """
    更新配置
    
    Parameters:
    -----------
    config_dict : 配置字典
    """
    global factor_config, time_series_config, database_config
    
    if 'factor' in config_dict:
        for key, value in config_dict['factor'].items():
            if hasattr(factor_config, key):
                setattr(factor_config, key, value)
    
    if 'time_series' in config_dict:
        for key, value in config_dict['time_series'].items():
            if hasattr(time_series_config, key):
                setattr(time_series_config, key, value)
    
    if 'database' in config_dict:
        for key, value in config_dict['database'].items():
            if hasattr(database_config, key):
                setattr(database_config, key, value)