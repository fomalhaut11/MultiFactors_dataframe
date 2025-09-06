"""
因子生成器工具模块

提供各类因子的纯计算工具函数，不包含Factor类定义。
配合 generators/ 模块中的数据处理工具使用。

模块结构:
- financial: 财务因子计算工具 (ROE、PE、质量指标等)
- technical: 技术因子计算工具 (价格、成交量指标等)  
- mixed: 混合因子计算工具 (财务+市场数据结合)
- alpha191: Alpha191因子计算工具

新的使用方式:
```python
# 直接使用工具函数
from factors.generator.financial.tools_init import calculate_roe_ttm
result = calculate_roe_ttm(financial_data)

# 或通过注册表
from factors.generator.financial.tools_init import get_financial_calculator
calc_func = get_financial_calculator('ROE_ttm')
result = calc_func(financial_data)
```
"""

# 导入财务工具集成
from .financial.tools_init import (
    FINANCIAL_TOOLS_REGISTRY,
    get_financial_calculator,
    list_financial_factors
)

# 保留原有模块导入，但标记为deprecated
from . import financial
from . import technical  
from . import mixed
from . import alpha191

__all__ = [
    # 子模块
    'financial',
    'technical', 
    'mixed',
    'alpha191',
    
    # 财务工具集成
    'FINANCIAL_TOOLS_REGISTRY',
    'get_financial_calculator', 
    'list_financial_factors',
    
    # 便捷函数
    'get_calculator',
    'list_factors',
]

# 版本信息
__version__ = '3.0.0'  # 工具函数版本

# 便捷函数：快速获取财务计算器
def get_calculator(factor_name: str):
    """
    便捷函数：获取因子计算器
    
    Parameters
    ----------
    factor_name : str
        因子名称
        
    Returns
    -------
    callable
        计算函数
        
    Examples
    --------
    >>> from factors.generator import get_calculator
    >>> calc = get_calculator('ROE_ttm')
    >>> result = calc(financial_data)
    """
    return get_financial_calculator(factor_name)

# 便捷函数：列出可用因子
def list_factors():
    """
    便捷函数：列出所有可用因子
    
    Returns
    -------
    dict
        因子列表
        
    Examples
    --------
    >>> from factors.generator import list_factors
    >>> factors = list_factors()
    >>> print(factors['profitability'])
    """
    return list_financial_factors()