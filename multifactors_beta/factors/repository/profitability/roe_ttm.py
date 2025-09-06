"""
ROE_ttm 因子实现
净资产收益率（TTM）- 试点迁移因子
"""

# 因子元数据
FACTOR_META = {
    "name": "ROE_ttm",
    "category": "profitability",
    "description": "TTM净资产收益率，衡量企业股东权益的盈利能力",
    "dependencies": ["DEDUCTEDPROFIT", "EQY_BELONGTO_PARCOMSH", "d_year", "d_quarter"],
    "formula": "TTM扣非净利润 / 股东权益",
    "data_frequency": "季报",
    "calculation_method": "TTM",
    "version": "1.0.0",
    "author": "Factors Team", 
    "created": "2025-01-15",
    "last_modified": "2025-01-15",
    "requires_market_data": False,
    "tags": ["盈利能力", "ROE", "基础因子"],
    "references": ["传统财务分析指标"]
}

# 导入依赖
import pandas as pd
import numpy as np
import logging

from factors.generators.financial import calculate_ttm

logger = logging.getLogger(__name__)


def calculate(financial_data: pd.DataFrame, **kwargs) -> pd.Series:
    """
    计算ROE_ttm因子
    
    Parameters
    ----------
    financial_data : pd.DataFrame
        财务数据，MultiIndex格式 [ReportDates, StockCodes]
        必须包含：DEDUCTEDPROFIT, EQY_BELONGTO_PARCOMSH, d_year, d_quarter
    
    Returns
    -------
    pd.Series
        ROE_ttm因子值，MultiIndex格式
    """
    try:
        earnings_col = 'DEDUCTEDPROFIT'  # 扣非净利润
        equity_col = 'EQY_BELONGTO_PARCOMSH'  # 归属母公司股东权益
        
        # 检查必要字段
        required_cols = [earnings_col, equity_col, 'd_year', 'd_quarter']
        missing_cols = [col for col in required_cols if col not in financial_data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # 使用generators计算TTM
        ttm_results = calculate_ttm(financial_data)
        earnings_ttm = ttm_results[f'{earnings_col}_ttm']
        
        # 股东权益取最新值（季末值）
        equity_data = financial_data[equity_col]
        
        # 对齐数据并计算ROE
        earnings_aligned, equity_aligned = earnings_ttm.align(equity_data, join='inner')
        
        # 安全除法，处理异常值
        roe = earnings_aligned / equity_aligned.replace(0, np.nan)
        roe = roe.replace([np.inf, -np.inf], np.nan)
        
        # 设置因子名称
        roe.name = FACTOR_META['name']
        
        return roe
        
    except Exception as e:
        logger.error(f"计算ROE_ttm失败: {e}")
        return pd.Series(dtype=float, name=FACTOR_META['name'])


def test_calculate():
    """ROE_ttm因子的单元测试"""
    # 创建测试数据
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
    assert result.name == "ROE_ttm", "Series名称应该是ROE_ttm"
    assert isinstance(result.index, pd.MultiIndex), "索引应该是MultiIndex"
    assert len(result) == 4, "应该有4个结果"
    assert result.notna().sum() > 0, "应该有有效值"
    
    # 验证ROE值的合理性（通常在-1到1之间）
    valid_values = result.dropna()
    if len(valid_values) > 0:
        assert (valid_values.abs() < 10).all(), "ROE值应该在合理范围内"
    
    print(f"✓ ROE_ttm 单元测试通过，有效值数量: {result.notna().sum()}")
    return True


def validate_inputs(financial_data: pd.DataFrame) -> bool:
    """验证输入数据的完整性"""
    required_cols = FACTOR_META['dependencies']
    return all(col in financial_data.columns for col in required_cols)


# 使用示例
if __name__ == "__main__":
    print(f"因子名称: {FACTOR_META['name']}")
    print(f"因子描述: {FACTOR_META['description']}")
    print(f"计算公式: {FACTOR_META['formula']}")
    print(f"数据依赖: {FACTOR_META['dependencies']}")
    
    # 运行单元测试
    try:
        test_calculate()
        print("ROE_ttm 因子文件验证通过")
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n使用方法:")
    print("```python")
    print("# 方式1: 直接导入")
    print("import factors.repository.profitability.roe_ttm as roe_factor")
    print("result = roe_factor.calculate(financial_data)")
    print("")
    print("# 方式2: 通过加载器")
    print("from factors.library.loader import FactorLoader")
    print("loader = FactorLoader()")
    print("loader.load_all_factors()")
    print("roe_func = loader.get_factor_function('ROE_ttm')")
    print("result = roe_func(financial_data)")
    print("```")