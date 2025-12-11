#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高级批量因子生成器 🚀
基于YAML配置文件的智能因子生成系统

特性：
- 📋 基于配置文件的因子管理
- 🎯 预设模式和自定义因子集
- 📊 智能数据依赖分析
- 🔍 自动质量检查和修复
- 📈 详细的生成报告
- ⚡ 优化的并行计算
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
import logging
import time
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp
import argparse
from dataclasses import dataclass

# 配置路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 导入项目模块
from config import get_config, get_config
from factors.generator.financial.pure_financial_factors import PureFinancialFactorCalculator
from factors.generator.financial.earnings_surprise_factors import SUEFactorCalculator

# 设置日志
def setup_logging(level=logging.INFO, log_file=None):
    """设置日志配置"""
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding='utf-8'))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    return logging.getLogger(__name__)

logger = setup_logging()


@dataclass
class FactorConfig:
    """因子配置数据类"""
    name: str
    description: str
    calculator: str
    method: str
    data_requirements: List[str]
    parameters: Dict[str, Any] = None
    enabled: bool = True
    priority: int = 1


@dataclass 
class GenerationResult:
    """因子生成结果数据类"""
    factor_name: str
    success: bool
    data: Optional[pd.Series] = None
    error: Optional[str] = None
    duration: float = 0.0
    quality_score: float = 0.0
    data_shape: Tuple[int] = None
    null_ratio: float = 0.0


class AdvancedFactorGenerator:
    """高级批量因子生成器"""
    
    def __init__(self, config_file: str = "factor_config.yaml"):
        """
        初始化高级因子生成器
        
        Parameters:
        -----------
        config_file : str
            配置文件路径
        """
        self.config_file = Path(config_file)
        self.config = self._load_config()
        self.factor_configs = self._parse_factor_configs()
        self.calculators = {}
        self.data_cache = {}
        self.generation_results = []
        
        # 初始化输出目录
        self.output_dir = Path(self.config['settings']['output_dir'])
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # 设置日志
        if self.config['logging']['save_log']:
            log_file = self.output_dir / f"factor_generation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            global logger
            logger = setup_logging(
                level=getattr(logging, self.config['logging']['level']),
                log_file=log_file
            )
        
        logger.info(f"高级因子生成器初始化完成")
        logger.info(f"配置文件: {self.config_file}")
        logger.info(f"输出目录: {self.output_dir}")
        logger.info(f"可用因子: {len(self.factor_configs)} 个")
        
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        if not self.config_file.exists():
            raise FileNotFoundError(f"配置文件不存在: {self.config_file}")
            
        with open(self.config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            
        logger.info(f"成功加载配置文件: {self.config_file}")
        return config
    
    def _parse_factor_configs(self) -> Dict[str, FactorConfig]:
        """解析因子配置"""
        factor_configs = {}
        
        for group_name, group_config in self.config['factor_groups'].items():
            if not group_config.get('enabled', True):
                continue
                
            # 遍历子分组
            for subgroup_name, subgroup_factors in group_config.items():
                if subgroup_name in ['description', 'enabled', 'priority']:
                    continue
                    
                if isinstance(subgroup_factors, list):
                    for factor_config in subgroup_factors:
                        factor_name = factor_config['name']
                        config_obj = FactorConfig(
                            name=factor_name,
                            description=factor_config['description'],
                            calculator=factor_config['calculator'],
                            method=factor_config['method'],
                            data_requirements=factor_config['data_requirements'],
                            parameters=factor_config.get('parameters', {}),
                            enabled=factor_config.get('enabled', True),
                            priority=group_config.get('priority', 1)
                        )
                        factor_configs[factor_name] = config_obj
        
        return factor_configs
    
    def _create_calculators(self) -> Dict[str, Any]:
        """创建因子计算器"""
        calculators = {}
        
        # 创建各种计算器
        calculator_classes = {
            'PureFinancialFactorCalculator': PureFinancialFactorCalculator,
            'SUEFactorCalculator': SUEFactorCalculator,
        }
        
        for calc_name, calc_class in calculator_classes.items():
            try:
                calculators[calc_name] = calc_class()
                logger.info(f"✅ 创建计算器: {calc_name}")
            except Exception as e:
                logger.error(f"❌ 创建计算器失败 {calc_name}: {e}")
        
        return calculators
    
    def _load_data(self, data_requirements: List[str]) -> Dict[str, Any]:
        """根据需求加载数据"""
        data = {}
        
        # 从配置中获取数据路径
        data_req_config = self.config['data_requirements']
        base_path = Path(get_config('main.paths.data_root'))
        
        for req in data_requirements:
            if req in self.data_cache:
                data[req] = self.data_cache[req]
                continue
                
            if req not in data_req_config:
                logger.warning(f"未知的数据需求: {req}")
                continue
                
            req_config = data_req_config[req]
            
            # 尝试主路径
            file_path = base_path / req_config['file_path']
            loaded = False
            
            if file_path.exists():
                try:
                    data[req] = pd.read_pickle(file_path)
                    self.data_cache[req] = data[req]
                    logger.info(f"✅ 加载数据 {req}: {data[req].shape}")
                    loaded = True
                except Exception as e:
                    logger.error(f"❌ 加载数据失败 {req}: {e}")
            
            # 尝试备用路径
            if not loaded and 'alt_paths' in req_config:
                for alt_path in req_config['alt_paths']:
                    alt_file_path = base_path / alt_path
                    if alt_file_path.exists():
                        try:
                            data[req] = pd.read_pickle(alt_file_path)
                            self.data_cache[req] = data[req]
                            logger.info(f"✅ 加载数据 {req} (备用路径): {data[req].shape}")
                            loaded = True
                            break
                        except Exception as e:
                            logger.error(f"❌ 加载数据失败 {req} (备用路径): {e}")
            
            if not loaded:
                if req_config.get('required', False):
                    logger.error(f"❌ 必需数据缺失: {req}")
                else:
                    logger.warning(f"⚠️  可选数据缺失: {req}")
                data[req] = None
        
        return data
    
    def _generate_single_factor(self, factor_config: FactorConfig, data: Dict[str, Any]) -> GenerationResult:
        """生成单个因子"""
        start_time = time.time()
        
        try:
            # 检查数据依赖
            missing_data = []
            for req in factor_config.data_requirements:
                if req not in data or data[req] is None:
                    missing_data.append(req)
            
            if missing_data:
                error_msg = f"缺少必要数据: {missing_data}"
                return GenerationResult(
                    factor_name=factor_config.name,
                    success=False,
                    error=error_msg,
                    duration=time.time() - start_time
                )
            
            # 获取计算器
            calculator = self.calculators.get(factor_config.calculator)
            if not calculator:
                error_msg = f"计算器不存在: {factor_config.calculator}"
                return GenerationResult(
                    factor_name=factor_config.name,
                    success=False,
                    error=error_msg,
                    duration=time.time() - start_time
                )
            
            # 调用计算方法
            factor_data = None
            
            if factor_config.calculator == 'custom':
                # 处理自定义计算逻辑
                factor_data = self._handle_custom_calculation(factor_config, data)
            else:
                # 调用计算器方法
                if hasattr(calculator, factor_config.method):
                    method = getattr(calculator, factor_config.method)
                    
                    # 准备参数
                    method_args = []
                    for req in factor_config.data_requirements:
                        method_args.append(data[req])
                    
                    # 调用方法
                    if factor_config.parameters:
                        factor_data = method(*method_args, **factor_config.parameters)
                    else:
                        factor_data = method(*method_args)
                else:
                    error_msg = f"计算方法不存在: {factor_config.calculator}.{factor_config.method}"
                    return GenerationResult(
                        factor_name=factor_config.name,
                        success=False,
                        error=error_msg,
                        duration=time.time() - start_time
                    )
            
            if factor_data is None or (hasattr(factor_data, 'empty') and factor_data.empty):
                return GenerationResult(
                    factor_name=factor_config.name,
                    success=False,
                    error="计算结果为空",
                    duration=time.time() - start_time
                )
            
            # 计算质量评分
            quality_score, null_ratio = self._calculate_quality_score(factor_data)
            
            return GenerationResult(
                factor_name=factor_config.name,
                success=True,
                data=factor_data,
                duration=time.time() - start_time,
                quality_score=quality_score,
                data_shape=factor_data.shape,
                null_ratio=null_ratio
            )
            
        except Exception as e:
            return GenerationResult(
                factor_name=factor_config.name,
                success=False,
                error=str(e),
                duration=time.time() - start_time
            )
    
    def _handle_custom_calculation(self, factor_config: FactorConfig, data: Dict[str, Any]) -> pd.Series:
        """处理自定义计算逻辑"""
        if factor_config.method == 'direct_market_cap':
            return data['market_cap']
        elif factor_config.method == 'log_market_cap':
            market_cap = data['market_cap']
            return np.log(market_cap)
        else:
            raise ValueError(f"未知的自定义计算方法: {factor_config.method}")
    
    def _calculate_quality_score(self, factor_data: pd.Series) -> Tuple[float, float]:
        """计算因子数据质量评分"""
        try:
            null_ratio = factor_data.isnull().mean()
            inf_count = np.isinf(factor_data.values).sum()
            unique_count = factor_data.nunique()
            
            # 基础分数
            quality_score = 100.0
            
            # 空值惩罚
            if null_ratio > 0.5:
                quality_score -= 40
            elif null_ratio > 0.3:
                quality_score -= 20
            elif null_ratio > 0.1:
                quality_score -= 10
            
            # 无穷值惩罚
            if inf_count > 100:
                quality_score -= 20
            elif inf_count > 0:
                quality_score -= 10
            
            # 唯一值惩罚
            if unique_count < 10:
                quality_score -= 30
            elif unique_count < 50:
                quality_score -= 10
            
            return max(0, quality_score), null_ratio
            
        except Exception:
            return 0.0, 1.0
    
    def _apply_quality_fixes(self, factor_data: pd.Series, factor_name: str) -> pd.Series:
        """应用数据质量修复"""
        qc_config = self.config['quality_control']
        
        # 修复无穷值
        if qc_config.get('auto_fix_inf', False):
            inf_mask = np.isinf(factor_data)
            if inf_mask.any():
                logger.info(f"修复 {factor_name} 中的 {inf_mask.sum()} 个无穷值")
                factor_data = factor_data.replace([np.inf, -np.inf], np.nan)
        
        return factor_data
    
    def generate_factors(self, factor_names: List[str], parallel: bool = True) -> List[GenerationResult]:
        """批量生成因子"""
        logger.info(f"🚀 开始生成 {len(factor_names)} 个因子")
        
        # 创建计算器
        if not self.calculators:
            self.calculators = self._create_calculators()
        
        # 分析数据依赖
        all_data_requirements = set()
        valid_factors = []
        
        for factor_name in factor_names:
            if factor_name in self.factor_configs:
                factor_config = self.factor_configs[factor_name]
                if factor_config.enabled:
                    all_data_requirements.update(factor_config.data_requirements)
                    valid_factors.append(factor_config)
                else:
                    logger.warning(f"因子已禁用: {factor_name}")
            else:
                logger.error(f"未知因子: {factor_name}")
        
        logger.info(f"有效因子: {len(valid_factors)}")
        logger.info(f"数据依赖: {list(all_data_requirements)}")
        
        # 加载数据
        data = self._load_data(list(all_data_requirements))
        
        # 生成因子
        results = []
        
        if parallel and len(valid_factors) > 1:
            logger.info("使用并行计算模式")
            # 由于序列化复杂性，暂时使用串行模式
            parallel = False
        
        if not parallel:
            logger.info("使用串行计算模式") 
            for i, factor_config in enumerate(valid_factors, 1):
                logger.info(f"[{i}/{len(valid_factors)}] 生成因子: {factor_config.name}")
                result = self._generate_single_factor(factor_config, data)
                results.append(result)
                
                # 输出结果
                if result.success:
                    logger.info(f"  ✅ {result.factor_name}: 形状={result.data_shape}, "
                              f"质量={result.quality_score:.1f}, 耗时={result.duration:.1f}s")
                    
                    # 应用质量修复
                    if result.data is not None:
                        result.data = self._apply_quality_fixes(result.data, result.factor_name)
                else:
                    logger.error(f"  ❌ {result.factor_name}: {result.error}")
        
        self.generation_results.extend(results)
        
        successful_results = [r for r in results if r.success]
        logger.info(f"🎯 生成完成: {len(successful_results)}/{len(valid_factors)} 成功")
        
        return results
    
    def save_factors(self, results: List[GenerationResult], 
                    mode: str = None) -> Dict[str, str]:
        """保存因子数据"""
        successful_results = [r for r in results if r.success and r.data is not None]
        
        if not successful_results:
            logger.warning("没有成功的因子数据需要保存")
            return {}
        
        logger.info(f"💾 保存 {len(successful_results)} 个因子")
        
        saved_files = {}
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        for result in successful_results:
            try:
                # 确定文件名
                if mode and mode != 'custom':
                    filename = f"{result.factor_name}_{mode}_{timestamp}.pkl"
                else:
                    filename = f"{result.factor_name}.pkl"
                
                file_path = self.output_dir / filename
                
                # 保存数据
                if self.config['output']['compression']:
                    result.data.to_pickle(file_path, compression='gzip')
                else:
                    result.data.to_pickle(file_path)
                
                saved_files[result.factor_name] = str(file_path)
                
                file_size = file_path.stat().st_size / 1024 / 1024  # MB
                logger.info(f"  💾 {result.factor_name}: {filename} ({file_size:.1f}MB)")
                
            except Exception as e:
                logger.error(f"❌ 保存失败 {result.factor_name}: {e}")
        
        # 保存生成报告
        if self.config['output']['generate_report']:
            self._generate_report(results, saved_files, mode)
        
        logger.info(f"保存完成: {len(saved_files)}/{len(successful_results)} 个文件")
        return saved_files
    
    def _generate_report(self, results: List[GenerationResult], 
                        saved_files: Dict[str, str], mode: str):
        """生成详细报告"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 统计信息
        total_factors = len(results)
        successful_factors = len([r for r in results if r.success])
        avg_quality = np.mean([r.quality_score for r in results if r.success])
        total_time = sum([r.duration for r in results])
        
        # 生成JSON报告
        report = {
            'generation_summary': {
                'timestamp': timestamp,
                'mode': mode,
                'total_factors': total_factors,
                'successful_factors': successful_factors,
                'success_rate': successful_factors / total_factors if total_factors > 0 else 0,
                'average_quality_score': avg_quality,
                'total_generation_time': total_time,
                'output_directory': str(self.output_dir)
            },
            'factor_results': [
                {
                    'name': r.factor_name,
                    'success': r.success,
                    'error': r.error,
                    'duration': r.duration,
                    'quality_score': r.quality_score,
                    'data_shape': list(r.data_shape) if r.data_shape else None,
                    'null_ratio': r.null_ratio,
                    'saved_file': saved_files.get(r.factor_name)
                }
                for r in results
            ],
            'quality_distribution': {
                'high_quality (>=80)': len([r for r in results if r.success and r.quality_score >= 80]),
                'medium_quality (60-80)': len([r for r in results if r.success and 60 <= r.quality_score < 80]),
                'low_quality (<60)': len([r for r in results if r.success and r.quality_score < 60])
            }
        }
        
        # 保存JSON报告
        report_file = self.output_dir / f'factor_generation_report_{timestamp}.json'
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📊 生成报告已保存: {report_file}")
    
    def list_available_factors(self) -> None:
        """列出所有可用因子"""
        print("\n📋 可用因子列表:")
        print("=" * 80)
        
        # 按分组显示
        for group_name, group_config in self.config['factor_groups'].items():
            if not group_config.get('enabled', True):
                continue
                
            factors_in_group = [f for f in self.factor_configs.values() 
                              if f.name in [factor['name'] for subgroup in group_config.values() 
                                          if isinstance(subgroup, list) 
                                          for factor in subgroup]]
            
            if factors_in_group:
                print(f"\n📦 {group_name.upper()} ({len(factors_in_group)}个):")
                print(f"   {group_config['description']}")
                
                for i, factor in enumerate(factors_in_group, 1):
                    status = "✅" if factor.enabled else "❌"
                    print(f"  {i:2d}. {status} {factor.name:<25} - {factor.description}")
        
        # 显示预设模式
        print(f"\n🎯 预设生成模式:")
        for mode_name, mode_config in self.config['generation_modes'].items():
            print(f"  📋 {mode_name:<10} ({len(mode_config['factors']):2d}个) - {mode_config['description']}")
    
    def run(self, mode: str = 'all', factor_list: List[str] = None) -> List[GenerationResult]:
        """运行因子生成"""
        print("=" * 80)
        print("🚀 高级批量因子生成器")
        print(f"📅 开始时间: {datetime.now()}")
        print(f"⚙️  生成模式: {mode}")
        print("=" * 80)
        
        start_time = time.time()
        
        # 确定要生成的因子列表
        if factor_list:
            factors_to_generate = factor_list
            logger.info(f"🎯 自定义因子模式: {len(factors_to_generate)} 个因子")
        elif mode in self.config['generation_modes']:
            factors_to_generate = self.config['generation_modes'][mode]['factors']
            logger.info(f"📋 预设模式 '{mode}': {len(factors_to_generate)} 个因子")
        elif mode == 'all':
            factors_to_generate = list(self.factor_configs.keys())
            logger.info(f"🌟 全量模式: {len(factors_to_generate)} 个因子")
        else:
            # 尝试按分组生成
            factors_to_generate = []
            for group_name, group_config in self.config['factor_groups'].items():
                if group_name == mode and group_config.get('enabled', True):
                    for subgroup in group_config.values():
                        if isinstance(subgroup, list):
                            factors_to_generate.extend([f['name'] for f in subgroup])
                    break
            
            if factors_to_generate:
                logger.info(f"📦 分组模式 '{mode}': {len(factors_to_generate)} 个因子")
            else:
                logger.error(f"❌ 未知模式: {mode}")
                return []
        
        # 生成因子
        results = self.generate_factors(factors_to_generate)
        
        # 保存结果
        saved_files = self.save_factors(results, mode)
        
        total_time = time.time() - start_time
        successful_count = len([r for r in results if r.success])
        
        print("\n" + "=" * 80)
        print("🎉 因子生成完成")
        print(f"⏱️  总耗时: {total_time:.1f} 秒")
        print(f"✅ 成功生成: {successful_count}/{len(results)} 个因子")
        print(f"💾 输出目录: {self.output_dir}")
        if successful_count > 0:
            avg_quality = np.mean([r.quality_score for r in results if r.success])
            print(f"📊 平均质量: {avg_quality:.1f} 分")
        print("=" * 80)
        
        return results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='高级批量因子生成工具')
    parser.add_argument('--config', default='factor_config.yaml', help='配置文件路径')
    parser.add_argument('--mode', default='core', help='生成模式或分组名称')
    parser.add_argument('--factors', type=str, help='指定因子列表，逗号分隔')
    parser.add_argument('--list', action='store_true', help='列出所有可用因子')
    parser.add_argument('--list-modes', action='store_true', help='列出所有可用模式')
    
    args = parser.parse_args()
    
    # 创建生成器
    try:
        generator = AdvancedFactorGenerator(args.config)
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        return
    
    # 列出因子或模式
    if args.list:
        generator.list_available_factors()
        return
    
    if args.list_modes:
        print("\n🎯 可用生成模式:")
        print("=" * 50)
        for mode_name, mode_config in generator.config['generation_modes'].items():
            print(f"📋 {mode_name:<12} - {mode_config['description']}")
            print(f"   包含因子: {len(mode_config['factors'])} 个")
        return
    
    # 解析因子列表
    factor_list = None
    if args.factors:
        factor_list = [f.strip() for f in args.factors.split(',')]
        print(f"🎯 指定因子: {factor_list}")
    
    # 运行生成
    results = generator.run(mode=args.mode, factor_list=factor_list)
    
    # 输出结果摘要
    if results:
        successful_factors = [r.factor_name for r in results if r.success]
        failed_factors = [r.factor_name for r in results if not r.success]
        
        if successful_factors:
            print(f"\n✨ 成功生成的因子:")
            for i, factor_name in enumerate(successful_factors, 1):
                print(f"  {i:2d}. {factor_name}")
        
        if failed_factors:
            print(f"\n❌ 生成失败的因子:")
            for i, factor_name in enumerate(failed_factors, 1):
                print(f"  {i:2d}. {factor_name}")


if __name__ == "__main__":
    main()