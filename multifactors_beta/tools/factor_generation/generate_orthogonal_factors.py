#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
标准中性化因子生成脚本
将原始因子对基准因子和行业进行正交化处理，生成标准中性化因子

特点：
- 🎯 使用标准控制变量进行正交化
- 📊 支持批量处理多个因子  
- 💾 保存到OrthogonalizationFactors目录
- 🔍 生成详细的处理报告
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import pickle
import json
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 导入因子注册表
from factors.meta import get_factor_registry, FactorType, NeutralizationCategory

from config import get_config, get_config
from core.utils.data_cleaning import OutlierHandler, Normalizer
from factors.tester.core.data_manager import DataManager

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OrthogonalFactorGenerator:
    """正交化因子生成器"""
    
    # 标准控制变量配置
    STANDARD_CONTROL_CONFIG = {
        'base_factors': [
            'LogMarketCap',     # 对数市值（最重要）
            'BP',               # 净资产市值比
            'LogTurnover_20',   # 换手率（如果有）
        ],
        'use_industry': True,   # 使用行业中性化
        'classification_name': 'classification_one_hot'
    }
    
    # 因子类别配置（决定是否需要中性化）
    FACTOR_CATEGORIES = {
        'must_neutralize': [
            'ROE_ttm', 'ROA_ttm', 'ROIC_ttm',           # 盈利能力
            'CurrentRatio', 'QuickRatio',               # 偿债能力  
            'AssetTurnover_ttm', 'EquityTurnover_ttm',  # 营运效率
            'RevenueGrowth_yoy', 'NetIncomeGrowth_yoy', # 成长能力
            'OperatingCashFlowRatio_ttm',               # 现金流
            'SUE_ss_4', 'SUE_ttm_4',                    # 盈余惊喜
        ],
        'optional_neutralize': [
            'EP_ttm', 'SP_ttm', 'CFP_ttm',              # 估值因子（可选）
            'Vol_120', 'Vol_20',                         # 波动率因子
            'ma_120', 'ma_20', 'ma_5',                   # 技术因子
        ],
        'skip_neutralize': [
            'LogMarketCap', 'MarketCap', 'Size',        # 规模因子（作为控制变量）
            'BP', 'LogFreeMarketCap',                   # 基准因子自身
        ]
    }
    
    def __init__(self):
        """初始化生成器"""
        self.raw_factors_path = Path(get_config('main.paths.raw_factors'))
        self.orth_factors_path = Path(get_config('main.paths.orthogonalization_factors'))
        self.output_path = Path(get_config('main.paths.factors'))
        
        # 确保输出目录存在
        self.orth_factors_path.mkdir(parents=True, exist_ok=True)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # 初始化数据管理器
        config = get_config('main.factor_test') or {}
        self.data_manager = DataManager(config)
        
        # 初始化因子注册表
        self.factor_registry = get_factor_registry()
        
        logger.info("正交化因子生成器初始化完成")
        logger.info(f"原始因子路径: {self.raw_factors_path}")
        logger.info(f"正交化因子路径: {self.orth_factors_path}")
    
    def get_available_factors(self) -> List[str]:
        """获取可用的原始因子列表"""
        factors = []
        
        if self.raw_factors_path.exists():
            for file in self.raw_factors_path.glob("*.pkl"):
                factor_name = file.stem
                factors.append(factor_name)
        
        logger.info(f"发现 {len(factors)} 个原始因子")
        return sorted(factors)
    
    def _update_factor_registry(
        self, 
        factor_name: str, 
        orthogonal_path: str,
        control_factors: List[str],
        stats: Dict
    ):
        """更新因子注册表，记录正交化信息"""
        try:
            # 获取或创建因子元数据
            metadata = self.factor_registry.get_factor(factor_name)
            
            if metadata is None:
                # 自动推断因子类型
                factor_type = self._infer_factor_type(factor_name)
                neutralization_category = self._get_neutralization_category(factor_name)
                
                # 注册新因子
                metadata = self.factor_registry.register_factor(
                    name=factor_name,
                    factor_type=factor_type,
                    description=f"自动注册的因子: {factor_name}",
                    neutralization_category=neutralization_category,
                    generator="OrthogonalFactorGenerator",
                    tags=["auto_registered"]
                )
            
            # 更新正交化信息
            self.factor_registry.mark_orthogonalized(
                name=factor_name,
                orthogonal_path=orthogonal_path,
                control_factors=control_factors,
                method='OLS'
            )
            
            # 更新性能统计
            if stats:
                performance_metrics = {
                    'orthogonalization_stats': stats,
                    'last_orthogonalization': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                self.factor_registry.update_factor(
                    factor_name, 
                    performance_metrics=performance_metrics
                )
            
            logger.info(f"已更新因子注册表: {factor_name}")
            
        except Exception as e:
            logger.error(f"更新因子注册表失败 {factor_name}: {e}")
            raise
    
    def _infer_factor_type(self, factor_name: str) -> FactorType:
        """根据因子名称推断因子类型"""
        name_lower = factor_name.lower()
        
        # 基本面因子
        if any(keyword in name_lower for keyword in ['roe', 'roa', 'roic', 'ep', 'bp', 'sp']):
            return FactorType.FUNDAMENTAL
        # 技术因子
        elif any(keyword in name_lower for keyword in ['ma_', 'vol_', 'rsi', 'macd']):
            return FactorType.TECHNICAL
        # 动量因子
        elif any(keyword in name_lower for keyword in ['momentum', 'ret_', 'return']):
            return FactorType.MOMENTUM
        # 波动率因子
        elif any(keyword in name_lower for keyword in ['vol', 'volatility', 'std']):
            return FactorType.VOLATILITY
        # Alpha191因子
        elif name_lower.startswith('alpha'):
            return FactorType.ALPHA191
        # 成长因子
        elif any(keyword in name_lower for keyword in ['growth', 'yoy']):
            return FactorType.GROWTH
        # 价值因子
        elif any(keyword in name_lower for keyword in ['value', 'book', 'price']):
            return FactorType.VALUE
        # 质量因子
        elif any(keyword in name_lower for keyword in ['quality', 'margin', 'turnover']):
            return FactorType.QUALITY
        else:
            return FactorType.DERIVED
    
    def _get_neutralization_category(self, factor_name: str) -> NeutralizationCategory:
        """根据因子名称确定中性化类别"""
        if factor_name in self.FACTOR_CATEGORIES['must_neutralize']:
            return NeutralizationCategory.MUST_NEUTRALIZE
        elif factor_name in self.FACTOR_CATEGORIES['optional_neutralize']:
            return NeutralizationCategory.OPTIONAL_NEUTRALIZE
        elif factor_name in self.FACTOR_CATEGORIES['skip_neutralize']:
            return NeutralizationCategory.SKIP_NEUTRALIZE
        else:
            # 默认为可选中性化
            return NeutralizationCategory.OPTIONAL_NEUTRALIZE
    
    def classify_factor(self, factor_name: str) -> str:
        """
        分类因子，决定处理策略
        
        Returns:
        - 'must': 必须中性化
        - 'optional': 可选中性化  
        - 'skip': 跳过中性化
        """
        if factor_name in self.FACTOR_CATEGORIES['must_neutralize']:
            return 'must'
        elif factor_name in self.FACTOR_CATEGORIES['optional_neutralize']:
            return 'optional'
        elif factor_name in self.FACTOR_CATEGORIES['skip_neutralize']:
            return 'skip'
        else:
            # 默认策略：其他因子建议中性化
            return 'must'
    
    def load_control_variables(self) -> Optional[pd.DataFrame]:
        """加载控制变量（基准因子 + 行业）"""
        try:
            # 加载基准因子
            base_factors_list = []
            available_base_factors = []
            
            for factor_name in self.STANDARD_CONTROL_CONFIG['base_factors']:
                try:
                    factor_data = self.data_manager.load_factor_data(factor_name)
                    if not factor_data.empty:
                        base_factors_list.append(factor_data)
                        available_base_factors.append(factor_name)
                        logger.info(f"加载基准因子: {factor_name}")
                    else:
                        logger.warning(f"基准因子为空: {factor_name}")
                except Exception as e:
                    logger.warning(f"无法加载基准因子 {factor_name}: {e}")
            
            # 合并基准因子
            if base_factors_list:
                base_factors_df = pd.concat(base_factors_list, axis=1, join='inner')
                base_factors_df.columns = available_base_factors
            else:
                logger.error("没有可用的基准因子")
                return None
            
            # 加载行业数据
            if self.STANDARD_CONTROL_CONFIG['use_industry']:
                try:
                    industry_data = self.data_manager.load_industry_data(
                        self.STANDARD_CONTROL_CONFIG['classification_name']
                    )
                    
                    if not industry_data.empty:
                        # 合并基准因子和行业数据
                        control_vars = self.data_manager._merge_base_and_industry(
                            base_factors_df, industry_data
                        )
                        logger.info(f"加载行业数据: {industry_data.shape[1]} 个行业")
                    else:
                        logger.warning("行业数据为空，仅使用基准因子")
                        control_vars = base_factors_df
                except Exception as e:
                    logger.warning(f"无法加载行业数据: {e}，仅使用基准因子")
                    control_vars = base_factors_df
            else:
                control_vars = base_factors_df
            
            logger.info(f"控制变量准备完成: {control_vars.shape}")
            return control_vars
            
        except Exception as e:
            logger.error(f"加载控制变量失败: {e}")
            return None
    
    def orthogonalize_factor(
        self, 
        factor_data: pd.Series, 
        control_vars: pd.DataFrame
    ) -> Tuple[Optional[pd.Series], Dict]:
        """
        对单个因子进行正交化处理
        
        Returns:
        - orthogonal_factor: 正交化后的因子
        - stats: 处理统计信息
        """
        import statsmodels.api as sm
        
        stats = {
            'original_count': len(factor_data),
            'valid_count': 0,
            'orthogonal_count': 0,
            'correlation_before': 0,
            'correlation_after': 0,
            'method_used': 'none'
        }
        
        try:
            # 数据对齐
            common_index = factor_data.index.intersection(control_vars.index)
            if len(common_index) < 100:
                logger.warning(f"数据对齐后样本过少: {len(common_index)}")
                return None, stats
            
            aligned_factor = factor_data.loc[common_index]
            aligned_controls = control_vars.loc[common_index]
            
            # 按日期处理
            orthogonal_results = []
            daily_stats = []
            
            for date, daily_factor in aligned_factor.groupby(level=0):
                if len(daily_factor) < 10:
                    continue
                
                daily_controls = aligned_controls.loc[date]
                
                # 数据预处理
                y = daily_factor.fillna(0)
                X = daily_controls.fillna(0)
                
                # 去除全零列
                valid_cols_mask = (X != 0).any(axis=0)
                X_valid = X.loc[:, valid_cols_mask]
                
                if X_valid.empty or len(X_valid.columns) == 0:
                    # 没有有效控制变量，使用原始因子
                    orthogonal_results.append(y)
                    daily_stats.append('no_controls')
                    continue
                
                try:
                    # 检查矩阵秩
                    X_with_const = sm.add_constant(X_valid)
                    rank = np.linalg.matrix_rank(X_with_const)
                    
                    if rank < X_with_const.shape[1]:
                        # 使用岭回归处理不满秩矩阵
                        try:
                            from sklearn.linear_model import Ridge
                            ridge = Ridge(alpha=1e-4)
                            ridge.fit(X_valid, y)
                            residuals = y - ridge.predict(X_valid)
                            daily_stats.append('ridge')
                        except ImportError:
                            # sklearn不可用，跳过该日期
                            continue
                    else:
                        # 使用OLS回归
                        model = sm.OLS(y, X_with_const)
                        result = model.fit()
                        residuals = result.resid
                        daily_stats.append('ols')
                    
                    # 标准化残差
                    orth_factor = Normalizer.normalize(residuals, method='zscore')
                    orthogonal_results.append(orth_factor)
                    
                except Exception as e:
                    logger.warning(f"日期 {date} 正交化失败: {e}")
                    orthogonal_results.append(y)
                    daily_stats.append('failed')
            
            # 合并结果
            if orthogonal_results:
                orthogonal_factor = pd.concat(orthogonal_results)
                orthogonal_factor.name = f"{factor_data.name}_orth"
                
                # 计算统计信息
                stats['valid_count'] = len(common_index)
                stats['orthogonal_count'] = len(orthogonal_factor)
                stats['method_used'] = max(set(daily_stats), key=daily_stats.count)
                
                # 计算相关性变化（如果有LogMarketCap）
                if 'LogMarketCap' in aligned_controls.columns:
                    try:
                        size_factor = aligned_controls['LogMarketCap']
                        common_for_corr = factor_data.index.intersection(size_factor.index)
                        if len(common_for_corr) > 100:
                            stats['correlation_before'] = factor_data.loc[common_for_corr].corr(
                                size_factor.loc[common_for_corr]
                            )
                            stats['correlation_after'] = orthogonal_factor.corr(
                                size_factor.loc[orthogonal_factor.index.intersection(size_factor.index)]
                            )
                    except:
                        pass
                
                return orthogonal_factor, stats
            else:
                return None, stats
                
        except Exception as e:
            logger.error(f"正交化处理失败: {e}")
            return None, stats
    
    def generate_single_factor(
        self, 
        factor_name: str, 
        control_vars: pd.DataFrame,
        force: bool = False
    ) -> Dict:
        """
        生成单个因子的正交化版本
        
        Returns:
        - result dict with status and details
        """
        result = {
            'factor_name': factor_name,
            'status': 'failed',
            'message': '',
            'output_file': None,
            'stats': {}
        }
        
        try:
            # 检查因子分类
            category = self.classify_factor(factor_name)
            if category == 'skip' and not force:
                result['status'] = 'skipped'
                result['message'] = f"因子类别为 {category}，跳过中性化"
                return result
            
            # 检查输出文件是否已存在
            output_file = self.orth_factors_path / f"{factor_name}_orth.pkl"
            if output_file.exists() and not force:
                result['status'] = 'existed'
                result['message'] = "正交化因子已存在"
                result['output_file'] = str(output_file)
                return result
            
            # 加载原始因子
            factor_data = self.data_manager.load_factor_data(factor_name)
            if factor_data.empty:
                result['message'] = "原始因子数据为空"
                return result
            
            # 执行正交化
            logger.info(f"正在处理因子: {factor_name} (类别: {category})")
            orthogonal_factor, stats = self.orthogonalize_factor(factor_data, control_vars)
            
            if orthogonal_factor is not None:
                # 保存正交化因子
                orthogonal_factor.to_pickle(output_file)
                
                # 更新因子注册表
                try:
                    self._update_factor_registry(
                        factor_name=factor_name,
                        orthogonal_path=str(output_file),
                        control_factors=self.STANDARD_CONTROL_CONFIG['base_factors'],
                        stats=stats
                    )
                except Exception as e:
                    logger.warning(f"更新因子注册表失败 {factor_name}: {e}")
                
                result['status'] = 'success'
                result['message'] = f"成功生成正交化因子，样本数: {len(orthogonal_factor)}"
                result['output_file'] = str(output_file)
                result['stats'] = stats
                
                logger.info(f"✅ {factor_name}: {result['message']}")
            else:
                result['message'] = "正交化处理失败"
                logger.warning(f"❌ {factor_name}: {result['message']}")
                
        except Exception as e:
            result['message'] = f"处理异常: {str(e)}"
            logger.error(f"❌ {factor_name}: {result['message']}")
        
        return result
    
    def generate_batch(
        self, 
        factor_names: Optional[List[str]] = None,
        force: bool = False,
        max_factors: Optional[int] = None
    ) -> Dict:
        """
        批量生成正交化因子
        
        Parameters:
        -----------
        factor_names : List[str], optional
            指定要处理的因子名称，如果为None则处理所有可用因子
        force : bool
            是否强制重新生成（覆盖已存在的文件）
        max_factors : int, optional
            最大处理因子数量（用于测试）
            
        Returns:
        --------
        Dict: 批量处理结果
        """
        start_time = datetime.now()
        logger.info("=" * 70)
        logger.info("🚀 批量正交化因子生成开始")
        logger.info(f"📅 开始时间: {start_time}")
        logger.info("=" * 70)
        
        # 准备因子列表
        if factor_names is None:
            available_factors = self.get_available_factors()
        else:
            available_factors = factor_names
        
        if max_factors:
            available_factors = available_factors[:max_factors]
            
        logger.info(f"📋 待处理因子: {len(available_factors)} 个")
        
        # 加载控制变量
        logger.info("📊 加载控制变量...")
        control_vars = self.load_control_variables()
        if control_vars is None:
            return {
                'status': 'failed',
                'message': '无法加载控制变量',
                'results': []
            }
        
        # 批量处理
        results = []
        success_count = 0
        skip_count = 0
        exist_count = 0
        
        for i, factor_name in enumerate(available_factors, 1):
            logger.info(f"[{i}/{len(available_factors)}] 处理因子: {factor_name}")
            
            result = self.generate_single_factor(factor_name, control_vars, force)
            results.append(result)
            
            if result['status'] == 'success':
                success_count += 1
            elif result['status'] == 'skipped':
                skip_count += 1
            elif result['status'] == 'existed':
                exist_count += 1
        
        # 生成处理报告
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        summary = {
            'status': 'completed',
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_seconds': duration,
            'total_factors': len(available_factors),
            'success_count': success_count,
            'skip_count': skip_count,
            'exist_count': exist_count,
            'fail_count': len(available_factors) - success_count - skip_count - exist_count,
            'results': results
        }
        
        # 保存处理报告
        report_file = self.output_path / f"orthogonal_generation_report_{start_time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        
        # 打印汇总信息
        logger.info("\n" + "=" * 70)
        logger.info("🎉 批量正交化因子生成完成")
        logger.info(f"⏱️  总耗时: {duration:.1f} 秒")
        logger.info(f"📊 处理结果:")
        logger.info(f"   ✅ 成功生成: {success_count}")
        logger.info(f"   ⏭️  跳过处理: {skip_count}")
        logger.info(f"   📁 已经存在: {exist_count}")
        logger.info(f"   ❌ 处理失败: {summary['fail_count']}")
        logger.info(f"📋 详细报告: {report_file}")
        logger.info("=" * 70)
        
        return summary

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='正交化因子生成工具')
    parser.add_argument('--factors', nargs='+', help='指定要处理的因子名称')
    parser.add_argument('--force', action='store_true', help='强制重新生成（覆盖已存在文件）')
    parser.add_argument('--max', type=int, help='最大处理因子数量（测试用）')
    parser.add_argument('--list', action='store_true', help='列出所有可用因子')
    
    args = parser.parse_args()
    
    generator = OrthogonalFactorGenerator()
    
    if args.list:
        factors = generator.get_available_factors()
        print("\n📋 可用原始因子:")
        print("=" * 50)
        
        # 按类别分组显示
        for category, desc in [
            ('must', '必须中性化'),
            ('optional', '可选中性化'), 
            ('skip', '跳过中性化')
        ]:
            category_factors = [f for f in factors if generator.classify_factor(f) == category]
            if category_factors:
                print(f"\n🎯 {desc} ({len(category_factors)}个):")
                for i, factor in enumerate(category_factors[:10]):  # 显示前10个
                    print(f"  {i+1:2d}. {factor}")
                if len(category_factors) > 10:
                    print(f"     ... 还有 {len(category_factors)-10} 个")
        
        return
    
    # 执行批量生成
    summary = generator.generate_batch(
        factor_names=args.factors,
        force=args.force,
        max_factors=args.max
    )
    
    if summary['success_count'] > 0:
        print(f"\n✨ 成功生成的正交化因子:")
        for result in summary['results']:
            if result['status'] == 'success':
                print(f"  📈 {result['factor_name']}_orth.pkl")

if __name__ == "__main__":
    main()