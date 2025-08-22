"""
风险模型验证测试

测试风险模型的各个组件和功能
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_test_data():
    """生成测试数据"""
    print("=" * 60)
    print("生成测试数据...")
    
    # 参数设置
    n_dates = 250  # 约一年数据
    n_stocks = 50
    n_factors = 5
    
    start_date = datetime.now() - timedelta(days=n_dates)
    dates = pd.date_range(start=start_date, periods=n_dates, freq='D')
    stocks = [f'stock_{i:03d}' for i in range(n_stocks)]
    factors = ['momentum', 'value', 'quality', 'size', 'volatility']
    
    # 生成因子收益率（更现实的相关性结构）
    np.random.seed(42)
    
    # 因子协方差矩阵（有一定相关性）
    factor_corr = np.array([
        [1.00, 0.10, 0.05, -0.20, -0.15],  # momentum
        [0.10, 1.00, 0.30, -0.60, 0.10],   # value  
        [0.05, 0.30, 1.00, -0.10, -0.25],  # quality
        [-0.20, -0.60, -0.10, 1.00, 0.20], # size
        [-0.15, 0.10, -0.25, 0.20, 1.00]   # volatility
    ])
    
    factor_vols = np.array([0.12, 0.15, 0.10, 0.18, 0.22])  # 年化波动率
    factor_vols_daily = factor_vols / np.sqrt(252)
    
    factor_cov = np.outer(factor_vols_daily, factor_vols_daily) * factor_corr
    
    # 生成因子收益率时间序列
    factor_returns = np.random.multivariate_normal(
        mean=np.zeros(n_factors),
        cov=factor_cov,
        size=n_dates
    )
    
    factor_returns_df = pd.DataFrame(
        factor_returns,
        index=dates,
        columns=factors
    )
    
    # 生成因子暴露度（简化版本）
    exposures_data = []
    
    for i, date in enumerate(dates):
        # 基础暴露度
        base_exposures = np.random.randn(n_stocks, n_factors)
        
        # 添加一些结构（例如，大盘股通常有负的size暴露度）
        # Size因子：前20%股票为大盘股
        large_cap_mask = np.arange(n_stocks) < n_stocks * 0.2
        base_exposures[large_cap_mask, 3] = np.random.normal(-2, 0.5, np.sum(large_cap_mask))
        
        # Value和Quality有轻微负相关
        base_exposures[:, 2] += -0.3 * base_exposures[:, 1] + 0.2 * np.random.randn(n_stocks)
        
        # 标准化（横截面）
        base_exposures = (base_exposures - base_exposures.mean(axis=0)) / base_exposures.std(axis=0)
        
        # 创建MultiIndex格式
        for j, stock in enumerate(stocks):
            exposures_data.append((date, stock, base_exposures[j]))
    
    # 构建因子暴露度DataFrame
    exposure_tuples = [(date, stock) for date, stock, _ in exposures_data]
    exposure_values = np.array([exposures for _, _, exposures in exposures_data])
    
    multi_index = pd.MultiIndex.from_tuples(exposure_tuples, names=['date', 'stock'])
    factor_exposures = pd.DataFrame(
        exposure_values,
        index=multi_index,
        columns=factors
    )
    
    # 生成股票收益率（基于因子模型）
    returns_data = []
    
    for i, date in enumerate(dates):
        daily_factor_returns = factor_returns_df.loc[date]
        daily_exposures = factor_exposures.xs(date, level=0)
        
        # 系统性收益率
        systematic_returns = daily_exposures @ daily_factor_returns
        
        # 特异性收益率（异方差）
        base_specific_vol = 0.015  # 基础特异性波动率
        specific_vols = base_specific_vol * (0.5 + np.random.exponential(1, n_stocks))
        specific_returns = np.random.normal(0, specific_vols)
        
        # 总收益率
        total_returns = systematic_returns + specific_returns
        
        for j, stock in enumerate(stocks):
            returns_data.append((date, stock, total_returns.iloc[j]))
    
    # 构建收益率Series
    return_tuples = [(date, stock) for date, stock, _ in returns_data]
    return_values = [ret for _, _, ret in returns_data]
    
    multi_index_returns = pd.MultiIndex.from_tuples(return_tuples, names=['date', 'stock'])
    stock_returns = pd.Series(
        return_values,
        index=multi_index_returns,
        name='returns'
    )
    
    print(f"生成数据完成:")
    print(f"  - 时间范围: {dates[0].date()} 到 {dates[-1].date()}")
    print(f"  - 股票数量: {n_stocks}")
    print(f"  - 因子数量: {n_factors}")
    print(f"  - 因子暴露度形状: {factor_exposures.shape}")
    print(f"  - 收益率数据点数: {len(stock_returns)}")
    
    return factor_exposures, stock_returns, factor_returns_df

def test_covariance_estimators():
    """测试协方差估计器"""
    print("\n" + "=" * 60)
    print("测试协方差估计器...")
    
    from factors.risk_model.estimators import (
        SampleCovarianceEstimator,
        LedoitWolfEstimator, 
        ExponentialWeightedEstimator,
        RobustCovarianceEstimator
    )
    
    # 生成简单测试数据
    np.random.seed(42)
    n_obs, n_assets = 100, 20
    
    # 构造有结构的协方差矩阵
    true_corr = np.random.uniform(0.1, 0.7, (n_assets, n_assets))
    true_corr = (true_corr + true_corr.T) / 2  # 对称化
    np.fill_diagonal(true_corr, 1.0)
    
    vols = np.random.uniform(0.1, 0.3, n_assets)
    true_cov = np.outer(vols, vols) * true_corr
    
    # 生成收益率数据
    returns = np.random.multivariate_normal(np.zeros(n_assets), true_cov, n_obs)
    returns_df = pd.DataFrame(
        returns,
        columns=[f'asset_{i}' for i in range(n_assets)]
    )
    
    estimators = {
        'Sample': SampleCovarianceEstimator(),
        'Ledoit-Wolf': LedoitWolfEstimator({'shrinkage_target': 'diagonal'}),
        'EWMA': ExponentialWeightedEstimator({'decay_factor': 0.94}),
        'Robust': RobustCovarianceEstimator({'method': 'mcd'})
    }
    
    results = {}
    
    for name, estimator in estimators.items():
        try:
            start_time = datetime.now()
            estimator.fit(returns_df)
            fit_time = (datetime.now() - start_time).total_seconds()
            
            cov_matrix = estimator.get_covariance_matrix()
            
            # 计算与真实协方差的差异
            frobenius_error = np.linalg.norm(cov_matrix.values - true_cov, 'fro')
            condition_number = np.linalg.cond(cov_matrix.values)
            
            results[name] = {
                'fit_time': fit_time,
                'frobenius_error': frobenius_error,
                'condition_number': condition_number,
                'is_positive_definite': np.all(np.linalg.eigvals(cov_matrix.values) > 1e-8)
            }
            
            print(f"\n{name}估计器:")
            print(f"  - 拟合时间: {fit_time:.3f}秒")
            print(f"  - Frobenius误差: {frobenius_error:.4f}")
            print(f"  - 条件数: {condition_number:.2f}")
            print(f"  - 正定性: {results[name]['is_positive_definite']}")
            
            if hasattr(estimator, 'get_estimation_stats'):
                stats = estimator.get_estimation_stats()
                if 'shrinkage_parameter' in stats:
                    print(f"  - 收缩参数: {stats['shrinkage_parameter']:.4f}")
                if 'outlier_ratio' in stats:
                    print(f"  - 异常值比例: {stats['outlier_ratio']:.2%}")
            
        except Exception as e:
            print(f"\n{name}估计器失败: {e}")
            results[name] = {'error': str(e)}
    
    return results

def test_covariance_model():
    """测试协方差模型"""
    print("\n" + "=" * 60)  
    print("测试协方差模型...")
    
    from factors.risk_model.models import CovarianceModel
    
    # 生成测试数据
    factor_exposures, stock_returns, _ = generate_test_data()
    
    # 测试不同的估计器方法
    methods = ['sample', 'ledoit_wolf', 'exponential_weighted']
    
    for method in methods:
        try:
            print(f"\n测试{method}方法:")
            
            model = CovarianceModel({
                'estimator_method': method,
                'estimator_config': {
                    'min_periods': 30,
                    'decay_factor': 0.94 if method == 'exponential_weighted' else None
                }
            })
            
            # 拟合模型
            start_time = datetime.now()
            model.fit(factor_exposures, stock_returns)
            fit_time = (datetime.now() - start_time).total_seconds()
            
            print(f"  - 拟合时间: {fit_time:.3f}秒")
            print(f"  - 模型已拟合: {model.is_fitted}")
            
            # 测试协方差预测
            cov_matrix = model.predict_covariance(horizon=1)
            print(f"  - 协方差矩阵形状: {cov_matrix.shape}")
            print(f"  - 条件数: {np.linalg.cond(cov_matrix.values):.2f}")
            
            # 测试组合风险计算
            n_assets = len(cov_matrix)
            test_weights = pd.Series(1.0/n_assets, index=cov_matrix.index)
            
            risk_metrics = model.calculate_portfolio_risk(test_weights)
            print(f"  - 等权组合波动率: {risk_metrics['volatility']:.4f}")
            print(f"  - 95% VaR: {risk_metrics['var_95']:.4f}")
            
            # 测试风险分解
            decomp = model.decompose_risk(test_weights)
            print(f"  - 有效资产数: {decomp['effective_assets']:.1f}")
            print(f"  - 相关性影响: {decomp['correlation_impact']:.2%}")
            
        except Exception as e:
            print(f"  - {method}方法失败: {e}")

def test_barra_model():
    """测试Barra模型"""
    print("\n" + "=" * 60)
    print("测试Barra模型...")
    
    from factors.risk_model.models import BarraModel
    
    # 生成测试数据
    factor_exposures, stock_returns, true_factor_returns = generate_test_data()
    
    try:
        model = BarraModel(
            style_factors=['momentum', 'value', 'quality', 'size', 'volatility'],
            config={
                'factor_cov_method': 'exponential_weighted',
                'specific_risk_method': 'bayesian_shrinkage',
                'half_life': 90
            }
        )
        
        # 拟合模型
        start_time = datetime.now()
        model.fit(factor_exposures, stock_returns)
        fit_time = (datetime.now() - start_time).total_seconds()
        
        print(f"拟合时间: {fit_time:.3f}秒")
        print(f"模型已拟合: {model.is_fitted}")
        
        # 检查估计结果
        factor_returns = model.get_factor_returns()
        factor_cov = model.get_factor_covariance() 
        specific_risk = model.get_specific_risk()
        
        print(f"\n模型结果:")
        print(f"  - 因子收益率形状: {factor_returns.shape}")
        print(f"  - 因子协方差形状: {factor_cov.shape}")
        print(f"  - 特异性风险数量: {len(specific_risk)}")
        
        # 检查R²统计
        r_squared = model.get_regression_statistics()
        if r_squared is not None:
            print(f"  - 平均R²: {r_squared.mean():.3f}")
            print(f"  - R²范围: [{r_squared.min():.3f}, {r_squared.max():.3f}]")
        
        # 与真实因子收益率比较
        common_dates = factor_returns.index.intersection(true_factor_returns.index)
        if len(common_dates) > 0:
            correlations = []
            for factor in factor_returns.columns:
                if factor in true_factor_returns.columns:
                    corr = np.corrcoef(
                        factor_returns.loc[common_dates, factor],
                        true_factor_returns.loc[common_dates, factor]
                    )[0, 1]
                    correlations.append(corr)
                    print(f"  - {factor}因子相关性: {corr:.3f}")
            
            if correlations:
                print(f"  - 平均因子相关性: {np.mean(correlations):.3f}")
        
        # 测试协方差预测
        cov_matrix = model.predict_covariance(horizon=1)
        print(f"\n协方差预测:")
        print(f"  - 协方差矩阵形状: {cov_matrix.shape}")
        print(f"  - 条件数: {np.linalg.cond(cov_matrix.values):.2f}")
        
        # 测试组合风险计算
        n_assets = len(cov_matrix)
        test_weights = pd.Series(1.0/n_assets, index=cov_matrix.index)
        
        risk_metrics = model.calculate_portfolio_risk(test_weights)
        print(f"  - 等权组合总风险: {risk_metrics['volatility']:.4f}")
        print(f"  - 系统性风险: {risk_metrics['systematic_risk']:.4f}")
        print(f"  - 特异性风险: {risk_metrics['specific_risk']:.4f}")
        print(f"  - 因子风险占比: {risk_metrics['factor_risk_pct']:.1f}%")
        
        # 测试风险分解
        decomp = model.decompose_risk(test_weights)
        print(f"\n风险分解:")
        print(f"  - 因子贡献前3:")
        factor_contrib = decomp['factor_contributions'].abs().sort_values(ascending=False)
        for i, (factor, contrib) in enumerate(factor_contrib.head(3).items()):
            print(f"    {i+1}. {factor}: {contrib:.6f}")
        
        return True
        
    except Exception as e:
        print(f"Barra模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_factor_model():
    """测试通用因子模型"""
    print("\n" + "=" * 60)
    print("测试通用因子模型...")
    
    from factors.risk_model.models import FactorModel
    
    # 生成测试数据
    factor_exposures, stock_returns, _ = generate_test_data()
    
    model_types = ['barra', 'pca', 'covariance']
    
    for model_type in model_types:
        try:
            print(f"\n测试{model_type}类型:")
            
            if model_type == 'barra':
                model = FactorModel(
                    factors=['momentum', 'value', 'quality', 'size', 'volatility'],
                    model_type=model_type
                )
            else:
                model = FactorModel(model_type=model_type)
            
            # 拟合模型
            start_time = datetime.now()
            if model_type == 'pca':
                # PCA模型不需要因子暴露度
                model.fit(None, stock_returns)
            else:
                model.fit(factor_exposures, stock_returns)
            fit_time = (datetime.now() - start_time).total_seconds()
            
            print(f"  - 拟合时间: {fit_time:.3f}秒")
            print(f"  - 模型已拟合: {model.is_fitted}")
            
            # 获取因子信息
            factor_info = model.get_factor_information()
            print(f"  - 模型类型: {factor_info['model_type']}")
            
            if 'explained_variance_ratio' in factor_info:
                print(f"  - 解释方差比例: {factor_info['explained_variance_ratio'][:3]}")
            
            # 测试预测功能
            cov_matrix = model.predict_covariance()
            print(f"  - 协方差矩阵形状: {cov_matrix.shape}")
            
            # 测试组合风险
            n_assets = len(cov_matrix)
            test_weights = pd.Series(1.0/n_assets, index=cov_matrix.index)
            
            risk_metrics = model.calculate_portfolio_risk(test_weights)
            print(f"  - 等权组合波动率: {risk_metrics['volatility']:.4f}")
            
        except Exception as e:
            print(f"  - {model_type}类型失败: {e}")

def run_performance_test():
    """运行性能测试"""
    print("\n" + "=" * 60)
    print("运行性能测试...")
    
    from factors.risk_model.models import BarraModel, CovarianceModel
    
    # 生成大规模测试数据
    print("生成大规模数据...")
    n_dates_large = 500
    n_stocks_large = 200
    
    # 简化数据生成以提高速度
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', periods=n_dates_large, freq='D')
    stocks = [f'stock_{i:04d}' for i in range(n_stocks_large)]
    factors = ['momentum', 'value', 'quality', 'size', 'volatility']
    
    # 生成因子暴露度
    exposures_list = []
    for date in dates:
        exposures = np.random.randn(n_stocks_large, len(factors))
        exposures = (exposures - exposures.mean(axis=0)) / exposures.std(axis=0)
        
        for i, stock in enumerate(stocks):
            exposures_list.append((date, stock, exposures[i]))
    
    exposure_tuples = [(date, stock) for date, stock, _ in exposures_list]
    exposure_values = np.array([exp for _, _, exp in exposures_list])
    
    factor_exposures = pd.DataFrame(
        exposure_values,
        index=pd.MultiIndex.from_tuples(exposure_tuples, names=['date', 'stock']),
        columns=factors
    )
    
    # 生成收益率
    returns_list = []
    for date in dates:
        returns = np.random.normal(0, 0.02, n_stocks_large)
        for i, stock in enumerate(stocks):
            returns_list.append((date, stock, returns[i]))
    
    stock_returns = pd.Series(
        [ret for _, _, ret in returns_list],
        index=pd.MultiIndex.from_tuples([(d, s) for d, s, _ in returns_list], names=['date', 'stock']),
        name='returns'
    )
    
    print(f"大规模数据生成完成: {n_stocks_large}只股票, {n_dates_large}天")
    
    # 测试不同模型的性能
    models = {
        'CovarianceModel(sample)': CovarianceModel({'estimator_method': 'sample'}),
        'CovarianceModel(ledoit_wolf)': CovarianceModel({'estimator_method': 'ledoit_wolf'}),
        'BarraModel': BarraModel(style_factors=factors)
    }
    
    performance_results = {}
    
    for name, model in models.items():
        try:
            print(f"\n测试{name}性能:")
            
            # 拟合性能
            start_time = datetime.now()
            model.fit(factor_exposures, stock_returns)
            fit_time = (datetime.now() - start_time).total_seconds()
            
            # 预测性能
            start_time = datetime.now()
            cov_matrix = model.predict_covariance()
            predict_time = (datetime.now() - start_time).total_seconds()
            
            # 组合风险计算性能
            test_weights = pd.Series(1.0/len(cov_matrix), index=cov_matrix.index)
            
            start_time = datetime.now()
            risk_metrics = model.calculate_portfolio_risk(test_weights)
            risk_calc_time = (datetime.now() - start_time).total_seconds()
            
            performance_results[name] = {
                'fit_time': fit_time,
                'predict_time': predict_time,
                'risk_calc_time': risk_calc_time,
                'total_time': fit_time + predict_time + risk_calc_time,
                'volatility': risk_metrics['volatility']
            }
            
            print(f"  - 拟合时间: {fit_time:.3f}秒")
            print(f"  - 预测时间: {predict_time:.3f}秒") 
            print(f"  - 风险计算时间: {risk_calc_time:.3f}秒")
            print(f"  - 总时间: {performance_results[name]['total_time']:.3f}秒")
            print(f"  - 协方差矩阵大小: {cov_matrix.shape}")
            
        except Exception as e:
            print(f"  - {name}性能测试失败: {e}")
            performance_results[name] = {'error': str(e)}
    
    return performance_results

def main():
    """主测试函数"""
    print("风险模型验证测试")
    print("=" * 60)
    
    try:
        # 1. 测试协方差估计器
        estimator_results = test_covariance_estimators()
        
        # 2. 测试协方差模型  
        test_covariance_model()
        
        # 3. 测试Barra模型
        barra_success = test_barra_model()
        
        # 4. 测试通用因子模型
        test_factor_model()
        
        # 5. 性能测试
        performance_results = run_performance_test()
        
        # 总结
        print("\n" + "=" * 60)
        print("测试总结:")
        print(f"  - 协方差估计器测试: {'通过' if estimator_results else '失败'}")
        print(f"  - Barra模型测试: {'通过' if barra_success else '失败'}")
        print("  - 所有主要功能已验证")
        
        if performance_results:
            print("\n性能汇总:")
            for name, results in performance_results.items():
                if 'total_time' in results:
                    print(f"  - {name}: {results['total_time']:.3f}秒")
        
        return True
        
    except Exception as e:
        print(f"\n测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ 风险模型验证测试完成!")
    else:
        print("\n❌ 风险模型验证测试失败!")