"""
因子分析器配置模块
提供运行时配置和内部常量
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
from config import get_config
import logging

logger = logging.getLogger(__name__)


@dataclass
class AnalyzerConfig:
    """
    因子分析器配置（运行时配置）
    
    这里只放运行时参数和内部常量，不重复主配置的内容
    """
    
    # ============================================================================
    # 内部常量（不需要用户修改的）
    # ============================================================================
    
    # 指标名称映射
    METRICS_MAPPING = {
        'ic': 'ic_mean',
        'icir': 'icir',
        'sharpe': 'long_short_sharpe',
        'monotonicity': 'monotonicity_score',
        't_value': 't_value_mean',
        'rank_ic': 'rank_ic_mean',
        'rank_icir': 'rank_icir',
        'annual_return': 'long_short_annual_return',
        'max_drawdown': 'max_drawdown',
        'win_rate': 'win_rate'
    }
    
    # 分析方法映射
    ANALYSIS_METHODS = {
        'stability': ['rolling_ic', 'regime_change', 'parameter_sensitivity'],
        'correlation': ['pearson', 'spearman', 'dynamic', 'clustering'],
        'performance': ['cumulative_return', 'drawdown', 'turnover'],
        'robustness': ['bootstrap', 'subsample', 'time_decay']
    }
    
    # 可视化配置（内部使用）
    PLOT_SETTINGS = {
        'figure_size': (12, 8),
        'dpi': 100,
        'style': 'seaborn',
        'color_palette': 'husl',
        'grid': True,
        'alpha': 0.7
    }
    
    # 筛选器评分权重
    SCORING_WEIGHTS = {
        'ic_score': 0.25,
        'stability_score': 0.20,
        'monotonicity_score': 0.20,
        'sharpe_score': 0.20,
        'robustness_score': 0.15
    }
    
    # 市场环境定义
    MARKET_REGIMES = {
        'bull': {'return_threshold': 0.15, 'volatility_max': 0.25},
        'bear': {'return_threshold': -0.15, 'volatility_min': 0.20},
        'volatile': {'volatility_min': 0.30},
        'range_bound': {'return_range': (-0.10, 0.10), 'volatility_max': 0.20}
    }
    
    # 因子分类
    FACTOR_CATEGORIES = {
        'value': ['BP', 'EP', 'SP', 'NCFP', 'FCFP'],
        'growth': ['ROE', 'ROA', 'ROIC', 'SUE', 'SalesGrowth'],
        'quality': ['AssetTurnover', 'GrossMargin', 'DebtToEquity'],
        'momentum': ['PriceMomentum', 'EarningsMomentum', 'AnalystRevision'],
        'volatility': ['Vol', 'Beta', 'ResidualVol', 'Skewness'],
        'liquidity': ['Turnover', 'AmountAvg', 'Illiquidity'],
        'technical': ['RSI', 'MACD', 'Bollinger', 'MA']
    }
    
    @classmethod
    def from_config(cls, override: Optional[Dict[str, Any]] = None) -> 'AnalyzerConfig':
        """
        从主配置创建实例
        
        Parameters:
        -----------
        override : Dict, optional
            覆盖参数
            
        Returns:
        --------
        AnalyzerConfig
            配置实例
        """
        # 从主配置获取
        base_config = get_config('factor_analyzer')
        
        # 如果没有配置，使用默认值
        if base_config is None:
            base_config = {
                'screening': {
                    'ic_mean_min': 0.02,
                    'icir_min': 0.5,
                    'monotonicity_min': 0.6,
                    'sharpe_min': 1.0,
                    't_value_min': 2.0
                },
                'stability': {
                    'lookback_window': 30,
                    'rolling_window': 252
                }
            }
        
        # 合并覆盖参数
        if override:
            for key, value in override.items():
                if isinstance(value, dict) and key in base_config:
                    # 递归合并字典
                    base_config[key] = {**base_config[key], **value}
                else:
                    base_config[key] = value
        
        # 创建实例
        instance = cls()
        
        # 动态添加配置属性
        for key, value in base_config.items():
            setattr(instance, key, value)
        
        logger.info("AnalyzerConfig initialized with config keys: %s", list(base_config.keys()))
        
        return instance
    
    def get_screening_criteria(self, preset: Optional[str] = None) -> Dict[str, float]:
        """
        获取筛选标准
        
        Parameters:
        -----------
        preset : str, optional
            预设名称 ('strict', 'normal', 'loose')
            
        Returns:
        --------
        Dict
            筛选标准字典
        """
        if preset == 'strict':
            return {
                'ic_mean_min': 0.03,
                'icir_min': 0.7,
                'monotonicity_min': 0.7,
                'sharpe_min': 1.5,
                't_value_min': 2.5
            }
        elif preset == 'loose':
            return {
                'ic_mean_min': 0.01,
                'icir_min': 0.3,
                'monotonicity_min': 0.4,
                'sharpe_min': 0.5,
                't_value_min': 1.5
            }
        else:
            # 默认使用配置文件中的标准
            return self.screening if hasattr(self, 'screening') else {}
    
    def get_factor_category(self, factor_name: str) -> Optional[str]:
        """
        获取因子类别
        
        Parameters:
        -----------
        factor_name : str
            因子名称
            
        Returns:
        --------
        str or None
            因子类别
        """
        for category, factors in self.FACTOR_CATEGORIES.items():
            if factor_name in factors:
                return category
        return None
    
    def validate(self) -> bool:
        """
        验证配置有效性
        
        Returns:
        --------
        bool
            配置是否有效
        """
        required_sections = ['screening', 'stability', 'correlation']
        
        for section in required_sections:
            if not hasattr(self, section):
                logger.warning(f"Missing required config section: {section}")
                return False
        
        # 验证筛选标准
        if hasattr(self, 'screening'):
            screening = self.screening
            if screening.get('ic_mean_min', 0) > screening.get('ic_std_max', 1):
                logger.warning("Invalid screening criteria: ic_mean_min > ic_std_max")
                return False
        
        return True


# 全局配置实例（懒加载）
_global_config = None


def get_analyzer_config(override: Optional[Dict[str, Any]] = None) -> AnalyzerConfig:
    """
    获取分析器配置（单例模式）
    
    Parameters:
    -----------
    override : Dict, optional
        覆盖参数
        
    Returns:
    --------
    AnalyzerConfig
        配置实例
    """
    global _global_config
    
    if _global_config is None or override:
        _global_config = AnalyzerConfig.from_config(override)
    
    return _global_config