#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速因子生成脚本 ⚡
用于快速生成常用的核心因子，适合新手用户

特点：
- 🎯 预设核心因子集合
- ⚡ 简化的操作流程  
- 📊 自动数据检查
- 💾 标准化输出格式
- 🚀 一键运行

使用方式：
python quick_generate_factors.py              # 生成核心因子
python quick_generate_factors.py --basic      # 生成基础因子
python quick_generate_factors.py --test       # 生成测试因子
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import argparse

# 配置路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 导入项目模块
from config import get_config
from factors.generator.financial.pure_financial_factors import PureFinancialFactorCalculator
from factors.generator.mixed import get_mixed_factor_manager

# 设置简单日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class QuickFactorGenerator:
    """快速因子生成器"""
    
    # 预定义因子集合
    FACTOR_SETS = {
        'core': {
            'description': '核心因子集合 - 最重要的15个因子',
            'factors': [
                'ROE_ttm', 'ROA_ttm', 'BP', 'EP_ttm', 'Size', 
                'CurrentRatio', 'DebtToAssets', 'AssetTurnover_ttm',
                'GrossProfitMargin_ttm', 'NetProfitMargin_ttm',
                'RevenueGrowth_yoy', 'NetIncomeGrowth_yoy', 
                'OperatingCashFlowRatio_ttm', 'EarningsQuality_ttm',
                'ROIC_ttm'
            ]
        },
        'basic': {
            'description': '基础因子集合 - 涵盖主要因子类别',
            'factors': [
                'ROE_ttm', 'BP', 'Size', 'CurrentRatio', 
                'AssetTurnover_ttm', 'GrossProfitMargin_ttm',
                'RevenueGrowth_yoy', 'OperatingCashFlowRatio_ttm'
            ]
        },
        'test': {
            'description': '测试因子集合 - 用于快速测试',
            'factors': [
                'ROE_ttm', 'BP', 'Size', 'CurrentRatio'
            ]
        }
    }
    
    def __init__(self):
        """初始化快速因子生成器"""
        self.data_root = Path(get_config('main.paths.data_root'))
        self.auxiliary_path = project_root / 'data' / 'auxiliary'
        self.output_dir = self.data_root / 'factors'
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        self.data = {}
        self.calculator = None
        
        logger.info("快速因子生成器初始化完成")
        logger.info(f"输出目录: {self.output_dir}")
    
    def check_data_availability(self) -> bool:
        """检查数据可用性"""
        logger.info("🔍 检查数据可用性...")
        
        required_files = {
            'financial_data': self.auxiliary_path / 'FinancialData_unified.pkl',
            'market_cap': [
                self.data_root / 'MarketCap.pkl',
                self.data_root / 'LogMarketCap.pkl'
            ]
        }
        
        missing_files = []
        available_files = {}
        
        for key, file_paths in required_files.items():
            if isinstance(file_paths, list):
                found = False
                for file_path in file_paths:
                    if file_path.exists():
                        available_files[key] = file_path
                        found = True
                        break
                if not found:
                    missing_files.append(f"{key} (尝试: {[str(p) for p in file_paths]})")
            else:
                if file_paths.exists():
                    available_files[key] = file_paths
                else:
                    missing_files.append(str(file_paths))
        
        if missing_files:
            logger.error("❌ 缺少必要数据文件:")
            for file in missing_files:
                logger.error(f"   - {file}")
            logger.error("\n请先运行以下命令准备数据:")
            logger.error("1. python data/prepare_auxiliary_data.py")
            logger.error("2. python scheduled_data_updater.py --data-type financial")
            return False
        
        logger.info("✅ 所有必要数据文件都可用")
        self.available_files = available_files
        return True
    
    def load_data(self) -> bool:
        """加载数据"""
        logger.info("📊 加载数据...")
        
        try:
            # 加载财务数据
            financial_file = self.available_files['financial_data']
            self.data['financial_data'] = pd.read_pickle(financial_file)
            logger.info(f"✅ 财务数据: {self.data['financial_data'].shape}")
            
            # 加载市值数据
            market_cap_file = self.available_files['market_cap']
            market_cap = pd.read_pickle(market_cap_file)
            
            # 处理市值数据格式
            if isinstance(market_cap, pd.DataFrame):
                market_cap = market_cap.iloc[:, 0]
            
            # 如果是对数市值，转换为原始值
            if market_cap.median() < 100:
                logger.info("转换对数市值为原始市值")
                market_cap = np.exp(market_cap)
            
            self.data['market_cap'] = market_cap
            logger.info(f"✅ 市值数据: {market_cap.shape}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 数据加载失败: {e}")
            return False
    
    def create_calculator(self) -> bool:
        """创建因子计算器"""
        try:
            self.calculator = PureFinancialFactorCalculator()
            self.mixed_manager = get_mixed_factor_manager()
            logger.info("✅ 因子计算器创建成功")
            return True
        except Exception as e:
            logger.error(f"❌ 因子计算器创建失败: {e}")
            return False
    
    def generate_factor(self, factor_name: str) -> Optional[pd.Series]:
        """生成单个因子"""
        try:
            start_time = time.time()
            
            # 根据因子名称调用相应方法
            if factor_name == 'Size':
                factor_data = self.data['market_cap']
            elif factor_name == 'LogSize':
                factor_data = np.log(self.data['market_cap'])
            elif factor_name in ['BP', 'EP_ttm', 'SP_ttm', 'CFP_ttm']:
                # 混合因子（估值因子）
                factor_data = self.mixed_manager.calculate_factor(
                    factor_name, self.data
                )
            else:
                # 纯财务因子
                method_name = f'calculate_{factor_name}'
                if hasattr(self.calculator, method_name):
                    method = getattr(self.calculator, method_name)
                    factor_data = method(self.data['financial_data'])
                else:
                    logger.warning(f"因子计算方法不存在: {method_name}")
                    return None
            
            duration = time.time() - start_time
            
            if factor_data is not None and not factor_data.empty:
                logger.info(f"  ✅ {factor_name}: {factor_data.shape} ({duration:.1f}s)")
                return factor_data
            else:
                logger.error(f"  ❌ {factor_name}: 生成失败或结果为空")
                return None
                
        except Exception as e:
            logger.error(f"  ❌ {factor_name}: {str(e)}")
            return None
    
    def validate_factor(self, factor_name: str, factor_data: pd.Series) -> Dict[str, float]:
        """验证因子数据质量"""
        stats = {
            'null_ratio': factor_data.isnull().mean(),
            'inf_count': np.isinf(factor_data.values).sum(),
            'unique_count': factor_data.nunique(),
            'mean': factor_data.mean() if factor_data.dtype in ['float64', 'int64'] else np.nan,
            'std': factor_data.std() if factor_data.dtype in ['float64', 'int64'] else np.nan,
        }
        
        # 质量评分
        quality_score = 100
        if stats['null_ratio'] > 0.5:
            quality_score -= 40
        elif stats['null_ratio'] > 0.2:
            quality_score -= 20
        
        if stats['inf_count'] > 0:
            quality_score -= 20
        
        if stats['unique_count'] < 10:
            quality_score -= 20
        
        stats['quality_score'] = max(0, quality_score)
        return stats
    
    def save_factor(self, factor_name: str, factor_data: pd.Series) -> str:
        """保存因子数据"""
        try:
            filename = f"{factor_name}.pkl"
            file_path = self.output_dir / filename
            
            factor_data.to_pickle(file_path)
            
            file_size = file_path.stat().st_size / 1024 / 1024  # MB
            logger.info(f"  💾 {factor_name}: {filename} ({file_size:.1f}MB)")
            
            return str(file_path)
            
        except Exception as e:
            logger.error(f"❌ 保存失败 {factor_name}: {e}")
            return None
    
    def run(self, factor_set: str = 'core') -> Dict[str, str]:
        """运行快速因子生成"""
        print("=" * 70)
        print("⚡ 快速因子生成器")
        print(f"📅 开始时间: {datetime.now()}")
        print(f"🎯 因子集合: {factor_set}")
        print("=" * 70)
        
        start_time = time.time()
        
        # 检查因子集合是否有效
        if factor_set not in self.FACTOR_SETS:
            logger.error(f"❌ 未知因子集合: {factor_set}")
            return {}
        
        factor_config = self.FACTOR_SETS[factor_set]
        factors_to_generate = factor_config['factors']
        
        print(f"📋 {factor_config['description']}")
        print(f"🔢 包含因子: {len(factors_to_generate)} 个")
        print()
        
        # 1. 检查数据可用性
        if not self.check_data_availability():
            return {}
        
        # 2. 加载数据
        if not self.load_data():
            return {}
        
        # 3. 创建计算器
        if not self.create_calculator():
            return {}
        
        # 4. 生成因子
        logger.info(f"🚀 开始生成 {len(factors_to_generate)} 个因子...")
        
        results = {}
        saved_files = {}
        
        for i, factor_name in enumerate(factors_to_generate, 1):
            logger.info(f"[{i}/{len(factors_to_generate)}] 生成因子: {factor_name}")
            
            factor_data = self.generate_factor(factor_name)
            
            if factor_data is not None:
                # 验证质量
                stats = self.validate_factor(factor_name, factor_data)
                logger.info(f"    📊 质量评分: {stats['quality_score']:.1f}, "
                          f"空值率: {stats['null_ratio']:.1%}, "
                          f"唯一值: {stats['unique_count']}")
                
                # 保存因子
                file_path = self.save_factor(factor_name, factor_data)
                if file_path:
                    results[factor_name] = factor_data
                    saved_files[factor_name] = file_path
        
        # 5. 生成摘要
        total_time = time.time() - start_time
        success_count = len(results)
        
        summary = {
            'generation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'factor_set': factor_set,
            'total_factors': len(factors_to_generate),
            'successful_factors': success_count,
            'success_rate': success_count / len(factors_to_generate),
            'total_time': total_time,
            'saved_files': saved_files
        }
        
        # 保存摘要文件
        import json
        summary_file = self.output_dir / f'quick_generation_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print("\n" + "=" * 70)
        print("🎉 快速因子生成完成")
        print(f"⏱️  总耗时: {total_time:.1f} 秒")
        print(f"✅ 成功生成: {success_count}/{len(factors_to_generate)} 个因子")
        print(f"💾 输出目录: {self.output_dir}")
        print(f"📋 生成摘要: {summary_file}")
        print("=" * 70)
        
        if success_count > 0:
            print(f"\n✨ 成功生成的因子:")
            for i, factor_name in enumerate(results.keys(), 1):
                print(f"  {i:2d}. {factor_name}")
        
        if success_count < len(factors_to_generate):
            failed_factors = [f for f in factors_to_generate if f not in results]
            print(f"\n❌ 生成失败的因子:")
            for i, factor_name in enumerate(failed_factors, 1):
                print(f"  {i:2d}. {factor_name}")
        
        return saved_files


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='快速因子生成工具')
    parser.add_argument('--set', choices=['core', 'basic', 'test'], 
                       default='core', help='因子集合选择')
    parser.add_argument('--list', action='store_true', help='列出所有可用因子集合')
    
    args = parser.parse_args()
    
    generator = QuickFactorGenerator()
    
    if args.list:
        print("\n📋 可用因子集合:")
        print("=" * 50)
        for set_name, set_config in generator.FACTOR_SETS.items():
            print(f"\n🎯 {set_name.upper()}:")
            print(f"   描述: {set_config['description']}")
            print(f"   因子数量: {len(set_config['factors'])}")
            print(f"   包含因子: {', '.join(set_config['factors'][:5])}{'...' if len(set_config['factors']) > 5 else ''}")
        return
    
    # 运行快速生成
    saved_files = generator.run(args.set)
    
    if saved_files:
        print(f"\n🚀 快速开始使用生成的因子:")
        print("```python")
        print("import pandas as pd")
        print("from pathlib import Path")
        print("")
        print("# 加载生成的因子")
        print(f"factor_dir = Path('{generator.output_dir}')")
        for factor_name in list(saved_files.keys())[:3]:  # 显示前3个
            print(f"{factor_name.lower()} = pd.read_pickle(factor_dir / '{factor_name}.pkl')")
        print("```")


if __name__ == "__main__":
    main()