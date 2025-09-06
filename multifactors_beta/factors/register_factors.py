#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子注册脚本
将所有已实现的因子注册到项目系统中
"""

import logging
from typing import Dict, List, Any
from datetime import datetime

# 导入因子注册表
from .meta.factor_registry import get_factor_registry, FactorType, NeutralizationCategory

# 导入所有已实现的因子类
from .generator.financial.profitability_factors import (
    ROE_ttm_Factor,
    ROA_ttm_Factor, 
    GrossProfitMargin_ttm_Factor,
    ProfitCost_ttm_Factor
)

from .generator.financial.solvency_factors import (
    CurrentRatio_Factor,
    DebtToAssets_Factor
)

from .generator.financial.legacy_financial_factors import (
    SUE_ttm_120d_Factor
)

from .generator.financial.value_factors import (
    EPRatioFactor,
    BPRatioFactor,
    SPRatioFactor,
    CFPRatioFactor,
    EarningsYieldFactor
)

from .generator.financial.quality_factors import (
    ROEQualityFactor,
    EarningsQualityFactor,
    DebtQualityFactor,
    ProfitabilityStabilityFactor,
    AssetQualityFactor
)

from .generator.technical.momentum_factors import (
    MomentumFactor,
    ShortTermReversalFactor,
    LongTermReversalFactor,
    TrendStrengthFactor,
    PriceMomentumFactor,
    VolatilityAdjustedMomentumFactor
)

logger = logging.getLogger(__name__)


def register_financial_factors():
    """注册所有财务因子"""
    registry = get_factor_registry()
    
    # 盈利能力因子
    profitability_factors = [
        {
            'factor_class': ROE_ttm_Factor,
            'type': FactorType.FUNDAMENTAL,
            'description': 'TTM净资产收益率，衡量企业使用股东资本的效率',
            'formula': 'ROE_ttm = 扣非净利润TTM / 股东权益均值',
            'neutralization_category': NeutralizationCategory.OPTIONAL_NEUTRALIZE,
            'category': 'profitability',
            'tags': ['profitability', 'roe', 'ttm', 'fundamental']
        },
        {
            'factor_class': ROA_ttm_Factor,
            'type': FactorType.FUNDAMENTAL,
            'description': 'TTM总资产收益率，衡量企业资产使用效率',
            'formula': 'ROA_ttm = 扣非净利润TTM / 总资产均值',
            'neutralization_category': NeutralizationCategory.OPTIONAL_NEUTRALIZE,
            'category': 'profitability',
            'tags': ['profitability', 'roa', 'ttm', 'fundamental']
        },
        {
            'factor_class': GrossProfitMargin_ttm_Factor,
            'type': FactorType.FUNDAMENTAL,
            'description': 'TTM毛利率，衡量企业毛利润水平',
            'formula': 'GrossProfitMargin_ttm = (营业收入TTM - 营业成本TTM) / 营业收入TTM',
            'neutralization_category': NeutralizationCategory.OPTIONAL_NEUTRALIZE,
            'category': 'profitability',
            'tags': ['profitability', 'margin', 'ttm', 'fundamental']
        },
        {
            'factor_class': ProfitCost_ttm_Factor,
            'type': FactorType.FUNDAMENTAL,
            'description': '利润成本比率，衡量企业盈利相对成本效率',
            'formula': 'ProfitCost_ttm = 扣非净利润TTM / (财务费用TTM + 所得税TTM)',
            'neutralization_category': NeutralizationCategory.OPTIONAL_NEUTRALIZE,
            'category': 'profitability',
            'tags': ['profitability', 'cost', 'ttm', 'efficiency']
        }
    ]
    
    # 偿债能力因子
    solvency_factors = [
        {
            'factor_class': CurrentRatio_Factor,
            'type': FactorType.FUNDAMENTAL,
            'description': '流动比率，衡量企业短期偿债能力',
            'formula': 'CurrentRatio = 流动资产 / 流动负债',
            'neutralization_category': NeutralizationCategory.OPTIONAL_NEUTRALIZE,
            'category': 'solvency',
            'tags': ['solvency', 'liquidity', 'fundamental']
        },
        {
            'factor_class': DebtToAssets_Factor,
            'type': FactorType.FUNDAMENTAL,
            'description': '资产负债率，衡量企业财务杠杆水平',
            'formula': 'DebtToAssets = 负债总额 / 资产总额',
            'neutralization_category': NeutralizationCategory.OPTIONAL_NEUTRALIZE,
            'category': 'solvency',
            'tags': ['solvency', 'leverage', 'fundamental']
        }
    ]
    
    # 盈余惊喜因子
    earnings_surprise_factors = [
        {
            'factor_class': SUE_ttm_120d_Factor,
            'type': FactorType.FUNDAMENTAL,
            'description': '标准化未预期盈余除以120日收益率',
            'formula': 'SUE_ttm_120d = SUE_ttm / |120日收益率|',
            'neutralization_category': NeutralizationCategory.MUST_NEUTRALIZE,
            'category': 'earnings_surprise',
            'tags': ['earnings', 'surprise', 'ttm', 'momentum']
        }
    ]
    
    # 注册所有财务因子
    all_financial_factors = profitability_factors + solvency_factors + earnings_surprise_factors
    
    for factor_info in all_financial_factors:
        factor_class = factor_info['factor_class']
        instance = factor_class()
        
        try:
            registry.register_factor(
                name=instance.name,
                factor_type=factor_info['type'],
                description=factor_info['description'],
                formula=factor_info.get('formula'),
                neutralization_category=factor_info['neutralization_category'],
                category=factor_info['category'],
                tags=factor_info['tags'],
                generator=factor_class.__name__,
                generator_params={'module': factor_class.__module__}
            )
            logger.info(f"已注册财务因子: {instance.name}")
            
        except Exception as e:
            logger.error(f"注册财务因子 {instance.name} 失败: {e}")


def register_value_factors():
    """注册价值因子"""
    registry = get_factor_registry()
    
    value_factors = [
        {
            'factor_class': EPRatioFactor,
            'description': 'EP比率，盈利收益率，PE的倒数',
            'formula': 'EP = 净利润TTM / 市值',
            'tags': ['value', 'ep', 'earnings', 'mixed']
        },
        {
            'factor_class': BPRatioFactor,
            'description': 'BP比率，账面价值比率，PB的倒数',
            'formula': 'BP = 净资产 / 市值',
            'tags': ['value', 'bp', 'book_value', 'mixed']
        },
        {
            'factor_class': SPRatioFactor,
            'description': 'SP比率，销售收益率，PS的倒数',
            'formula': 'SP = 营业收入TTM / 市值',
            'tags': ['value', 'sp', 'sales', 'mixed']
        },
        {
            'factor_class': CFPRatioFactor,
            'description': 'CFP比率，现金流收益率，PCF的倒数',
            'formula': 'CFP = 经营现金流TTM / 市值',
            'tags': ['value', 'cfp', 'cashflow', 'mixed']
        },
        {
            'factor_class': EarningsYieldFactor,
            'description': '盈利收益率，基于企业价值的估值指标',
            'formula': 'EarningsYield = EBIT / 企业价值',
            'tags': ['value', 'earnings_yield', 'ebit', 'enterprise_value']
        }
    ]
    
    for factor_info in value_factors:
        factor_class = factor_info['factor_class']
        instance = factor_class()
        
        try:
            registry.register_factor(
                name=instance.name,
                factor_type=FactorType.VALUE,
                description=factor_info['description'],
                formula=factor_info.get('formula'),
                neutralization_category=NeutralizationCategory.MUST_NEUTRALIZE,
                category='value',
                tags=factor_info['tags'],
                generator=factor_class.__name__,
                generator_params={'module': factor_class.__module__}
            )
            logger.info(f"已注册价值因子: {instance.name}")
            
        except Exception as e:
            logger.error(f"注册价值因子 {instance.name} 失败: {e}")


def register_quality_factors():
    """注册质量因子"""
    registry = get_factor_registry()
    
    quality_factors = [
        {
            'factor_class': ROEQualityFactor,
            'description': 'ROE质量因子，衡量ROE的稳定性和持续性',
            'tags': ['quality', 'roe', 'stability']
        },
        {
            'factor_class': EarningsQualityFactor,
            'description': '盈利质量因子，比较净利润与现金流一致性',
            'formula': 'EarningsQuality = 经营现金流TTM / 净利润TTM',
            'tags': ['quality', 'earnings', 'cashflow']
        },
        {
            'factor_class': DebtQualityFactor,
            'description': '债务质量因子，衡量财务杠杆健康程度',
            'tags': ['quality', 'debt', 'leverage', 'solvency']
        },
        {
            'factor_class': ProfitabilityStabilityFactor,
            'description': '盈利稳定性因子，衡量盈利能力的稳定程度',
            'tags': ['quality', 'profitability', 'stability']
        },
        {
            'factor_class': AssetQualityFactor,
            'description': '资产质量因子，评估资产质量和效率',
            'tags': ['quality', 'assets', 'efficiency']
        }
    ]
    
    for factor_info in quality_factors:
        factor_class = factor_info['factor_class']
        instance = factor_class()
        
        try:
            registry.register_factor(
                name=instance.name,
                factor_type=FactorType.QUALITY,
                description=factor_info['description'],
                formula=factor_info.get('formula'),
                neutralization_category=NeutralizationCategory.OPTIONAL_NEUTRALIZE,
                category='quality',
                tags=factor_info['tags'],
                generator=factor_class.__name__,
                generator_params={'module': factor_class.__module__}
            )
            logger.info(f"已注册质量因子: {instance.name}")
            
        except Exception as e:
            logger.error(f"注册质量因子 {instance.name} 失败: {e}")


def register_momentum_factors():
    """注册动量因子"""
    registry = get_factor_registry()
    
    momentum_factors = [
        {
            'factor_class': MomentumFactor,
            'type': FactorType.MOMENTUM,
            'description': '经典动量因子，基于历史价格表现',
            'tags': ['momentum', 'price', 'technical']
        },
        {
            'factor_class': ShortTermReversalFactor,
            'type': FactorType.REVERSAL,
            'description': '短期反转因子，基于短期价格反转效应',
            'tags': ['reversal', 'short_term', 'technical']
        },
        {
            'factor_class': LongTermReversalFactor,
            'type': FactorType.REVERSAL,
            'description': '长期均值回归因子，基于长期均值回归效应',
            'tags': ['reversal', 'long_term', 'mean_reversion']
        },
        {
            'factor_class': TrendStrengthFactor,
            'type': FactorType.MOMENTUM,
            'description': '趋势强度因子，衡量价格趋势的持续性',
            'tags': ['momentum', 'trend', 'strength']
        },
        {
            'factor_class': PriceMomentumFactor,
            'type': FactorType.MOMENTUM,
            'description': '经典12-1价格动量因子',
            'formula': 'PriceMomentum_12_1 = log(P_t-1m / P_t-12m)',
            'tags': ['momentum', 'price', 'jegadeesh_titman']
        },
        {
            'factor_class': VolatilityAdjustedMomentumFactor,
            'type': FactorType.MOMENTUM,
            'description': '波动率调整的动量因子，风险调整后的动量',
            'formula': 'VolAdjMom = 累积收益率 / 波动率',
            'tags': ['momentum', 'volatility_adjusted', 'risk_adjusted']
        }
    ]
    
    for factor_info in momentum_factors:
        factor_class = factor_info['factor_class']
        instance = factor_class()
        
        try:
            registry.register_factor(
                name=instance.name,
                factor_type=factor_info['type'],
                description=factor_info['description'],
                formula=factor_info.get('formula'),
                neutralization_category=NeutralizationCategory.MUST_NEUTRALIZE,
                category='technical',
                tags=factor_info['tags'],
                generator=factor_class.__name__,
                generator_params={'module': factor_class.__module__}
            )
            logger.info(f"已注册技术因子: {instance.name}")
            
        except Exception as e:
            logger.error(f"注册技术因子 {instance.name} 失败: {e}")


def register_all_factors():
    """注册所有因子"""
    logger.info("开始注册所有因子到系统中...")
    
    try:
        # 注册各类因子
        register_financial_factors()
        register_value_factors()
        register_quality_factors()
        register_momentum_factors()
        
        # 获取统计信息
        registry = get_factor_registry()
        stats = registry.get_factor_statistics()
        
        logger.info("因子注册完成！")
        logger.info(f"总计注册因子数量: {stats['total_factors']}")
        logger.info(f"激活因子数量: {stats['active_factors']}")
        logger.info(f"因子类型分布: {stats['factor_types']}")
        
        return True
        
    except Exception as e:
        logger.error(f"因子注册过程出错: {e}")
        return False


def export_factor_catalog():
    """导出因子目录到CSV文件"""
    registry = get_factor_registry()
    
    try:
        import os
        catalog_path = os.path.join(
            os.path.dirname(__file__), 'meta', 'factor_catalog.csv'
        )
        registry.export_to_csv(catalog_path)
        logger.info(f"因子目录已导出到: {catalog_path}")
        
    except Exception as e:
        logger.error(f"导出因子目录失败: {e}")


def show_registered_factors():
    """显示所有已注册的因子"""
    registry = get_factor_registry()
    
    print("\n=== 已注册因子列表 ===")
    
    # 按类型分组显示
    for factor_type in FactorType:
        factors = registry.list_factors(factor_type=factor_type)
        if factors:
            print(f"\n{factor_type.value.upper()} 因子 ({len(factors)}个):")
            for factor in factors:
                print(f"  - {factor.name}: {factor.description}")
                if factor.formula:
                    print(f"    公式: {factor.formula}")
                print(f"    标签: {', '.join(factor.tags)}")
    
    # 显示统计信息
    stats = registry.get_factor_statistics()
    print(f"\n=== 统计信息 ===")
    print(f"总因子数: {stats['total_factors']}")
    print(f"激活因子数: {stats['active_factors']}")
    print(f"正交化因子数: {stats['orthogonalized_factors']}")
    print(f"正交化率: {stats['orthogonalization_rate']:.1%}")


if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 注册所有因子
    success = register_all_factors()
    
    if success:
        # 导出因子目录
        export_factor_catalog()
        
        # 显示注册结果
        show_registered_factors()
    else:
        print("因子注册失败，请查看日志")