"""
标准因子文件模板

复制此模板创建新的因子文件，按照标准格式填写所有必需信息。
"""

# 因子元数据 - 必需
FACTOR_META = {
    # 必需字段
    "name": "YourFactorName",              # 因子名称，建议使用字母数字下划线
    "category": "profitability",           # 因子类别：profitability/value/quality/technical/risk/momentum/experimental
    "description": "因子的详细描述",        # 因子功能和意义的说明
    "dependencies": [                       # 计算所需的数据字段列表
        "DEDUCTEDPROFIT",
        "EQY_BELONGTO_PARCOMSH", 
        "d_year", 
        "d_quarter"
    ],
    
    # 可选字段
    "formula": "计算公式的数学表达式",       # 如：TTM扣非净利润 / 股东权益
    "data_frequency": "季报",              # 数据频率：季报/年报/日频/分钟等
    "calculation_method": "TTM",           # 计算方法：TTM/单季/同比/环比等
    "version": "1.0.0",                    # 版本号，建议使用语义版本
    "author": "Research Team",             # 作者信息
    "created": "2025-01-01",              # 创建日期
    "last_modified": "2025-01-01",        # 最后修改日期
    "requires_market_data": False,         # 是否需要市值等市场数据
    "tags": ["基础因子", "财务指标"],       # 标签，便于搜索和分类
    "references": []                       # 参考文献或资料
}

# 导入必需的库
import pandas as pd
import numpy as np
import logging

# 导入基础工具（根据需要选择）
from factors.generators.financial import calculate_ttm, calculate_yoy, calculate_qoq
# from factors.generators.technical import MovingAverageCalculator
# from factors.generators.alpha191 import ts_rank, delta

logger = logging.getLogger(__name__)


def calculate(financial_data: pd.DataFrame, **kwargs) -> pd.Series:
    """
    因子计算函数 - 必需
    
    这是因子的核心计算逻辑，必须实现此函数。
    
    Parameters
    ----------
    financial_data : pd.DataFrame
        输入的财务数据，MultiIndex格式 [ReportDates, StockCodes]
        包含FACTOR_META中dependencies列出的所有必需字段
    **kwargs : dict
        其他参数，如market_cap等市场数据
        
    Returns
    -------
    pd.Series
        计算结果，MultiIndex格式 [ReportDates, StockCodes]
        Series的name应该与FACTOR_META['name']一致
        
    Examples
    --------
    >>> factor_result = calculate(financial_data)
    >>> print(factor_result.name)  # 'YourFactorName'
    >>> print(type(factor_result.index))  # MultiIndex
    """
    try:
        # 1. 数据验证 - 检查必需字段
        required_cols = FACTOR_META['dependencies']
        missing_cols = [col for col in required_cols if col not in financial_data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # 2. 数据预处理（根据需要）
        # 示例：计算TTM数据
        ttm_results = calculate_ttm(financial_data)
        
        # 3. 因子计算逻辑
        # 示例：ROE计算
        earnings_col = 'DEDUCTEDPROFIT'
        equity_col = 'EQY_BELONGTO_PARCOMSH'
        
        earnings_ttm = ttm_results[f'{earnings_col}_ttm']
        equity_data = financial_data[equity_col]
        
        # 对齐数据
        earnings_aligned, equity_aligned = earnings_ttm.align(equity_data, join='inner')
        
        # 计算因子值
        factor_result = earnings_aligned / equity_aligned.replace(0, np.nan)
        
        # 4. 数据清洗 - 处理异常值
        factor_result = factor_result.replace([np.inf, -np.inf], np.nan)
        
        # 5. 设置因子名称
        factor_result.name = FACTOR_META['name']
        
        return factor_result
        
    except Exception as e:
        # 标准错误处理
        logger.error(f"计算{FACTOR_META['name']}失败: {e}")
        # 返回空Series而不抛出异常
        return pd.Series(dtype=float, name=FACTOR_META['name'])


def test_calculate():
    """
    单元测试函数 - 强烈建议
    
    为因子计算提供简单的单元测试，验证计算逻辑的正确性。
    """
    # 创建测试数据
    import numpy as np
    np.random.seed(42)
    
    dates = pd.to_datetime(['2020-12-31', '2021-12-31'])
    stocks = ['TEST1', 'TEST2']
    
    index_data = []
    data_rows = []
    
    for stock in stocks:
        for date in dates:
            index_data.append((date, stock))
            data_rows.append({
                'DEDUCTEDPROFIT': np.random.normal(5000000, 1000000),
                'EQY_BELONGTO_PARCOMSH': np.random.normal(50000000, 10000000),
                'd_year': date.year,
                'd_quarter': 4
            })
    
    index = pd.MultiIndex.from_tuples(index_data, names=['ReportDates', 'StockCodes'])
    test_data = pd.DataFrame(data_rows, index=index)
    
    # 执行计算
    result = calculate(test_data)
    
    # 验证结果
    assert isinstance(result, pd.Series), "结果应该是Series"
    assert result.name == FACTOR_META['name'], f"Series名称应该是{FACTOR_META['name']}"
    assert isinstance(result.index, pd.MultiIndex), "索引应该是MultiIndex"
    assert len(result) > 0, "结果不应为空"
    
    print(f"✓ {FACTOR_META['name']} 单元测试通过")
    return True


def get_sample_data() -> pd.DataFrame:
    """
    获取示例数据 - 可选
    
    返回适用于此因子的示例数据，便于测试和演示。
    
    Returns
    -------
    pd.DataFrame
        示例数据
    """
    # 实现示例数据生成逻辑
    pass


def validate_inputs(financial_data: pd.DataFrame) -> bool:
    """
    输入验证函数 - 可选
    
    在计算前验证输入数据的完整性和格式。
    
    Parameters
    ----------
    financial_data : pd.DataFrame
        输入数据
        
    Returns
    -------
    bool
        数据是否有效
    """
    # 实现输入验证逻辑
    required_cols = FACTOR_META['dependencies']
    return all(col in financial_data.columns for col in required_cols)


# 使用示例和文档
if __name__ == "__main__":
    """
    直接运行此文件时的示例代码
    
    展示如何使用这个因子进行计算
    """
    print(f"因子名称: {FACTOR_META['name']}")
    print(f"因子类别: {FACTOR_META['category']}")
    print(f"因子描述: {FACTOR_META['description']}")
    print(f"数据依赖: {FACTOR_META['dependencies']}")
    
    # 运行单元测试
    try:
        test_calculate()
        print("单元测试通过")
    except Exception as e:
        print(f"单元测试失败: {e}")
    
    # 展示计算示例
    print("\n示例用法:")
    print("```python")
    print("import factors.repository.category.your_factor as factor")
    print("result = factor.calculate(your_financial_data)")
    print("print(result.head())")
    print("```")