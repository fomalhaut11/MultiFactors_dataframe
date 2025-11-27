#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
制备并存储5日收益率因子

1. 计算5日收益率因子
2. 保存到因子数据存储目录
3. 注册到因子系统中
4. 验证存储和加载功能

Author: AI Assistant
Date: 2025-09-07
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import logging
from datetime import datetime

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from factors.repository.technical.returns_5d import Returns5DFactor
from factors.library.factor_registry import factor_registry
from config import get_config

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def prepare_factor_storage_directory():
    """准备因子数据存储目录"""
    try:
        # 从配置获取数据根目录
        config = get_config('main')
        if config and config.get('paths', {}).get('data_root'):
            data_root = Path(config['paths']['data_root'])
        else:
            # 使用默认相对路径
            data_root = project_root.parent / 'StockData'
        
        # 创建因子数据子目录
        factors_dir = data_root / 'factors'
        technical_dir = factors_dir / 'technical'
        
        # 确保目录存在
        technical_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"因子存储目录: {technical_dir}")
        return technical_dir
        
    except Exception as e:
        logger.error(f"准备因子存储目录失败: {e}")
        raise


def calculate_and_save_returns_5d_factor():
    """计算并保存5日收益率因子"""
    try:
        logger.info("=" * 60)
        logger.info("开始制备5日收益率因子")
        logger.info("=" * 60)
        
        # 1. 准备存储目录
        storage_dir = prepare_factor_storage_directory()
        
        # 2. 创建因子实例并计算
        logger.info("步骤1: 创建因子实例...")
        factor = Returns5DFactor()
        
        logger.info("步骤2: 计算5日收益率因子...")
        logger.info("注意: 这个过程可能需要1-3分钟，请耐心等待...")
        
        factor_data = factor.calculate()
        
        # 3. 数据质量检查
        logger.info("步骤3: 数据质量检查...")
        logger.info(f"因子数据形状: {factor_data.shape}")
        logger.info(f"因子数据类型: {type(factor_data)}")
        logger.info(f"因子名称: {factor_data.name}")
        logger.info(f"数据范围: [{factor_data.min():.4f}, {factor_data.max():.4f}]")
        logger.info(f"NaN数量: {factor_data.isna().sum():,}")
        
        # 显示时间范围
        if isinstance(factor_data.index, pd.MultiIndex):
            dates = factor_data.index.get_level_values(0)
            stocks = factor_data.index.get_level_values(1)
            logger.info(f"日期范围: {dates.min()} ~ {dates.max()}")
            logger.info(f"股票数量: {stocks.nunique()}")
            logger.info(f"交易日数量: {dates.nunique()}")
        
        # 4. 保存因子数据
        logger.info("步骤4: 保存因子数据...")
        factor_file = storage_dir / "Returns_5D_C2C.pkl"
        
        factor_data.to_pickle(factor_file)
        logger.info(f"因子数据已保存: {factor_file}")
        
        # 验证保存文件
        file_size_mb = factor_file.stat().st_size / (1024 * 1024)
        logger.info(f"文件大小: {file_size_mb:.1f} MB")
        
        # 5. 保存因子元数据
        logger.info("步骤5: 保存因子元数据...")
        metadata = factor.get_factor_info()
        metadata['calculation_date'] = datetime.now().isoformat()
        metadata['data_shape'] = factor_data.shape
        metadata['file_path'] = str(factor_file)
        metadata['file_size_mb'] = round(file_size_mb, 2)
        
        metadata_file = storage_dir / "Returns_5D_C2C_metadata.json"
        import json
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        logger.info(f"元数据已保存: {metadata_file}")
        
        # 6. 注册因子到系统
        logger.info("步骤6: 注册因子到系统...")
        
        # 过滤掉会冲突的元数据项
        filtered_metadata = {
            k: v for k, v in metadata.items() 
            if k not in ['name', 'category', 'description', 'data_requirements', 'file_path']
        }
        
        factor_registry.register_from_file(
            name=metadata['name'],
            category=metadata['category'],
            description=metadata['description'],
            dependencies=metadata['data_requirements'],
            calculate_func=factor.calculate,
            file_path=str(Path(__file__).parent / 'factors' / 'repository' / 'technical' / 'returns_5d.py'),
            **filtered_metadata
        )
        
        logger.info("因子注册成功！")
        
        # 7. 测试加载
        logger.info("步骤7: 验证数据加载...")
        loaded_data = pd.read_pickle(factor_file)
        
        if loaded_data.equals(factor_data):
            logger.info("数据完整性验证通过")
        else:
            logger.warning("数据完整性验证失败")
        
        logger.info("=" * 60)
        logger.info("5日收益率因子制备完成！")
        logger.info("=" * 60)
        logger.info(f"因子文件: {factor_file}")
        logger.info(f"元数据文件: {metadata_file}")
        logger.info(f"注册名称: {metadata['name']}")
        logger.info("=" * 60)
        
        return factor_file, metadata_file, factor_data
        
    except Exception as e:
        logger.error(f"制备5日收益率因子失败: {e}")
        import traceback
        traceback.print_exc()
        raise


def test_factor_loading_and_usage():
    """测试因子加载和使用"""
    try:
        logger.info("测试因子加载和使用...")
        
        # 1. 从注册表获取因子
        factor_func = factor_registry.get('Returns_5D_C2C')
        if not factor_func:
            raise ValueError("无法从注册表获取因子")
        
        # 2. 测试直接文件加载
        storage_dir = prepare_factor_storage_directory()
        factor_file = storage_dir / "Returns_5D_C2C.pkl"
        
        if factor_file.exists():
            loaded_data = pd.read_pickle(factor_file)
            logger.info(f"直接加载因子数据成功: {loaded_data.shape}")
            
            # 显示样本数据
            logger.info("样本数据:")
            logger.info(loaded_data.head(10))
            
            return True
        else:
            logger.error(f"因子文件不存在: {factor_file}")
            return False
            
    except Exception as e:
        logger.error(f"测试因子加载失败: {e}")
        return False


if __name__ == "__main__":
    print("5日收益率因子制备脚本")
    print("=" * 60)
    print("此脚本将:")
    print("1. 计算5日close-to-close滚动收益率")
    print("2. 保存因子数据到存储目录")
    print("3. 注册因子到系统中")
    print("4. 验证存储和加载功能")
    print("=" * 60)
    print("预计耗时: 1-3分钟")
    print("=" * 60)
    
    try:
        # 主要制备流程
        factor_file, metadata_file, factor_data = calculate_and_save_returns_5d_factor()
        
        # 测试加载
        if test_factor_loading_and_usage():
            print("\n" + "=" * 60)
            print("成功: 5日收益率因子制备和注册完成!")
            print("=" * 60)
            print(f"数据文件: {factor_file}")
            print(f"元数据文件: {metadata_file}")
            print(f"数据点数: {len(factor_data):,}")
            print("=" * 60)
            print("现在可以使用以下方式调用:")
            print("from factors.library.factor_registry import get_factor")
            print("factor_func = get_factor('Returns_5D_C2C')")
            print("result = factor_func()")
            print("=" * 60)
        else:
            print("警告: 因子制备成功但加载测试失败")
            
    except Exception as e:
        print(f"\n失败: 制备过程出现错误: {e}")
        sys.exit(1)