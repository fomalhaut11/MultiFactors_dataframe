#!/usr/bin/env python3
"""
配置文件验证脚本

验证所有配置文件的格式和内容是否正确。

Usage:
    python config/validate.py
    python config/validate.py --config main
"""

import sys
import argparse
import logging
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.manager import ConfigManager

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def validate_all_configs():
    """验证所有配置文件"""
    logger = logging.getLogger(__name__)
    
    try:
        # 创建配置管理器实例
        config = ConfigManager()
        
        # 获取配置信息
        config_info = config.get_config_info()
        logger.info(f"配置目录: {config_info['config_dir']}")
        logger.info(f"已加载配置: {config_info['loaded_configs']}")
        
        # 验证每个配置文件
        validation_results = {}
        
        for config_name in config_info['loaded_configs']:
            try:
                is_valid = config.validate_config(config_name)
                validation_results[config_name] = is_valid
                
                if is_valid:
                    logger.info(f"✅ 配置 {config_name} 验证通过")
                else:
                    logger.error(f"❌ 配置 {config_name} 验证失败")
                    
            except Exception as e:
                logger.error(f"❌ 配置 {config_name} 验证异常: {e}")
                validation_results[config_name] = False
        
        # 汇总结果
        total_configs = len(validation_results)
        valid_configs = sum(validation_results.values())
        
        logger.info(f"\n验证汇总: {valid_configs}/{total_configs} 个配置文件验证通过")
        
        if valid_configs == total_configs:
            logger.info("🎉 所有配置文件验证通过！")
            return True
        else:
            logger.error(f"⚠️  {total_configs - valid_configs} 个配置文件验证失败")
            return False
            
    except Exception as e:
        logger.error(f"配置验证过程异常: {e}")
        return False

def validate_single_config(config_name: str):
    """验证单个配置文件"""
    logger = logging.getLogger(__name__)
    
    try:
        config = ConfigManager()
        is_valid = config.validate_config(config_name)
        
        if is_valid:
            logger.info(f"✅ 配置 {config_name} 验证通过")
            
            # 显示配置内容概要
            config_data = config.get(config_name)
            if isinstance(config_data, dict):
                logger.info(f"配置 {config_name} 包含以下部分:")
                for key in config_data.keys():
                    logger.info(f"  - {key}")
        else:
            logger.error(f"❌ 配置 {config_name} 验证失败")
            
        return is_valid
        
    except Exception as e:
        logger.error(f"配置 {config_name} 验证异常: {e}")
        return False

def test_config_access():
    """测试配置访问功能"""
    logger = logging.getLogger(__name__)
    
    try:
        config = ConfigManager()
        
        # 测试各种配置访问
        test_cases = [
            ('main.database.host', '数据库主机'),
            ('main.paths.data_root', '数据根目录'),
            ('factors.settings.output_dir', '因子输出目录'),
            ('field_mappings.common_fields.revenue.chinese_name', '营业收入中文名'),
            ('agents.factor_expert.description', '因子专家描述')
        ]
        
        logger.info("\n🧪 测试配置访问功能:")
        
        for key_path, description in test_cases:
            try:
                value = config.get(key_path)
                if value is not None:
                    logger.info(f"✅ {description}: {value}")
                else:
                    logger.warning(f"⚠️  {description}: 配置不存在")
            except Exception as e:
                logger.error(f"❌ {description}: 访问异常 - {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"配置访问测试异常: {e}")
        return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='配置文件验证工具')
    parser.add_argument('--config', help='指定要验证的配置文件名称')
    parser.add_argument('--test', action='store_true', help='测试配置访问功能')
    
    args = parser.parse_args()
    
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("🔧 配置文件验证工具启动")
    
    success = True
    
    if args.config:
        # 验证指定配置
        success = validate_single_config(args.config)
    else:
        # 验证所有配置
        success = validate_all_configs()
    
    if args.test:
        # 测试配置访问
        success = success and test_config_access()
    
    if success:
        logger.info("✅ 验证完成，所有检查通过")
        sys.exit(0)
    else:
        logger.error("❌ 验证失败，请检查配置文件")
        sys.exit(1)

if __name__ == '__main__':
    main()