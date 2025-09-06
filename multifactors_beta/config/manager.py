#!/usr/bin/env python3
"""
统一配置管理器

提供统一的配置文件访问接口，支持多种配置格式，
实现配置热更新、验证和环境变量替换功能。

Author: AI Assistant
Date: 2025-08-28
"""

import os
import yaml
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union
from threading import Lock
import re

logger = logging.getLogger(__name__)


class ConfigManager:
    """统一配置管理器"""
    
    _instance = None
    _lock = Lock()
    
    def __new__(cls):
        """单例模式实现"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """初始化配置管理器"""
        if hasattr(self, '_initialized'):
            return
        
        self._initialized = True
        self._configs = {}
        self._config_dir = Path(__file__).parent
        self._auto_reload = False
        
        # 配置文件映射
        self._config_files = {
            'main': 'main.yaml',
            'factors': 'factors.yaml', 
            'field_mappings': 'field_mappings.yaml',
            'agents': 'agents.yaml'
        }
        
        # 首先加载环境变量
        self._load_environment_variables()
        
        # 加载所有配置
        self._load_all_configs()
        
        logger.info(f"配置管理器初始化完成，配置目录: {self._config_dir}")
    
    def _load_environment_variables(self):
        """加载环境变量文件"""
        try:
            # 获取项目根目录（config目录的父目录）
            project_root = self._config_dir.parent
            env_file = project_root / '.env'
            
            if env_file.exists():
                # 尝试使用python-dotenv加载
                try:
                    from dotenv import load_dotenv
                    load_dotenv(env_file)
                    logger.info(f"已加载环境变量文件: {env_file}")
                except ImportError:
                    # 如果没有安装python-dotenv，手动解析.env文件
                    self._manual_load_env(env_file)
                    logger.info(f"手动解析环境变量文件: {env_file}")
            else:
                logger.info(f"环境变量文件不存在: {env_file}")
                
        except Exception as e:
            logger.warning(f"加载环境变量失败: {e}")
    
    def _manual_load_env(self, env_file: Path):
        """手动解析.env文件"""
        try:
            with open(env_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        if '=' in line:
                            key, value = line.split('=', 1)
                            key = key.strip()
                            value = value.strip().strip('"').strip("'")
                            if key and not key in os.environ:
                                os.environ[key] = value
        except Exception as e:
            logger.warning(f"手动解析.env文件失败: {e}")
    
    def _load_all_configs(self):
        """加载所有配置文件"""
        for config_name, filename in self._config_files.items():
            try:
                self._load_config(config_name, filename)
                logger.info(f"配置文件 {filename} 加载成功")
            except Exception as e:
                logger.error(f"配置文件 {filename} 加载失败: {e}")
                # 对于关键配置文件，抛出异常
                if config_name in ['main']:
                    raise
    
    def _load_config(self, config_name: str, filename: str):
        """加载单个配置文件"""
        file_path = self._config_dir / filename
        
        if not file_path.exists():
            logger.warning(f"配置文件不存在: {file_path}")
            return
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if filename.endswith('.yaml') or filename.endswith('.yml'):
                    config_data = yaml.safe_load(f)
                elif filename.endswith('.json'):
                    config_data = json.load(f)
                else:
                    raise ValueError(f"不支持的配置文件格式: {filename}")
            
            # 环境变量替换
            config_data = self._replace_env_variables(config_data)
            
            self._configs[config_name] = config_data
            
        except Exception as e:
            logger.error(f"解析配置文件失败 {filename}: {e}")
            raise
    
    def _replace_env_variables(self, data: Any) -> Any:
        """递归替换配置中的环境变量"""
        if isinstance(data, dict):
            return {key: self._replace_env_variables(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._replace_env_variables(item) for item in data]
        elif isinstance(data, str):
            return self._substitute_env_vars(data)
        else:
            return data
    
    def _substitute_env_vars(self, text: str) -> str:
        """替换字符串中的环境变量"""
        # 支持 ${VAR_NAME} 和 $VAR_NAME 两种格式
        def replace_var(match):
            var_name = match.group(1) or match.group(2)
            return os.getenv(var_name, match.group(0))
        
        pattern = r'\$\{([^}]+)\}|\$([A-Za-z_][A-Za-z0-9_]*)'
        return re.sub(pattern, replace_var, text)
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        获取配置值
        
        Args:
            key_path: 配置路径，如 'main.database.host' 或 'factors.financial'
            default: 默认值
            
        Returns:
            配置值
            
        Examples:
            >>> config = ConfigManager()
            >>> db_host = config.get('main.database.host')
            >>> factors = config.get('factors.financial.profitability', [])
        """
        try:
            keys = key_path.split('.')
            config_name = keys[0]
            
            if config_name not in self._configs:
                logger.warning(f"配置文件不存在: {config_name}")
                return default
            
            current = self._configs[config_name]
            
            # 遍历路径
            for key in keys[1:]:
                if isinstance(current, dict) and key in current:
                    current = current[key]
                else:
                    logger.debug(f"配置路径不存在: {key_path}")
                    return default
            
            return current
            
        except Exception as e:
            logger.error(f"获取配置失败 {key_path}: {e}")
            return default
    
    def set(self, key_path: str, value: Any):
        """
        设置配置值（仅在内存中，不写入文件）
        
        Args:
            key_path: 配置路径
            value: 配置值
        """
        try:
            keys = key_path.split('.')
            config_name = keys[0]
            
            if config_name not in self._configs:
                self._configs[config_name] = {}
            
            current = self._configs[config_name]
            
            # 创建嵌套结构
            for key in keys[1:-1]:
                if key not in current or not isinstance(current[key], dict):
                    current[key] = {}
                current = current[key]
            
            # 设置最终值
            if len(keys) > 1:
                current[keys[-1]] = value
            else:
                self._configs[config_name] = value
            
            logger.debug(f"配置已更新: {key_path} = {value}")
            
        except Exception as e:
            logger.error(f"设置配置失败 {key_path}: {e}")
    
    def reload(self, config_name: Optional[str] = None):
        """
        重新加载配置文件
        
        Args:
            config_name: 指定重新加载的配置名称，None为全部重新加载
        """
        try:
            if config_name:
                if config_name in self._config_files:
                    self._load_config(config_name, self._config_files[config_name])
                    logger.info(f"配置 {config_name} 重新加载成功")
                else:
                    logger.warning(f"配置 {config_name} 不存在")
            else:
                self._load_all_configs()
                logger.info("所有配置重新加载成功")
                
        except Exception as e:
            logger.error(f"配置重新加载失败: {e}")
    
    def validate_config(self, config_name: str) -> bool:
        """
        验证配置文件格式
        
        Args:
            config_name: 配置名称
            
        Returns:
            验证结果
        """
        # 基本的配置验证
        if config_name not in self._configs:
            return False
        
        config_data = self._configs[config_name]
        
        # 根据配置类型进行特定验证
        if config_name == 'main':
            return self._validate_main_config(config_data)
        elif config_name == 'factors':
            return self._validate_factors_config(config_data)
        elif config_name == 'field_mappings':
            return self._validate_field_mappings_config(config_data)
        
        return True
    
    def _validate_main_config(self, config: Dict) -> bool:
        """验证主配置文件"""
        required_sections = ['database', 'paths']
        
        for section in required_sections:
            if section not in config:
                logger.error(f"主配置缺少必需部分: {section}")
                return False
        
        # 验证数据库配置
        db_config = config['database']
        required_db_fields = ['host', 'user', 'password', 'database']
        
        for field in required_db_fields:
            if field not in db_config:
                logger.error(f"数据库配置缺少必需字段: {field}")
                return False
        
        return True
    
    def _validate_factors_config(self, config: Dict) -> bool:
        """验证因子配置文件"""
        if 'factor_groups' not in config:
            logger.error("因子配置缺少 factor_groups 部分")
            return False
        
        return True
    
    def _validate_field_mappings_config(self, config: Dict) -> bool:
        """验证字段映射配置文件"""
        if 'common_fields' not in config:
            logger.error("字段映射配置缺少 common_fields 部分")
            return False
        
        return True
    
    def get_all_configs(self) -> Dict[str, Any]:
        """获取所有配置"""
        return self._configs.copy()
    
    def get_config_info(self) -> Dict[str, Any]:
        """获取配置管理器信息"""
        return {
            'config_dir': str(self._config_dir),
            'loaded_configs': list(self._configs.keys()),
            'config_files': self._config_files,
            'auto_reload': self._auto_reload
        }
    
    def __getitem__(self, key: str) -> Any:
        """支持字典式访问"""
        return self.get(key)
    
    def __setitem__(self, key: str, value: Any):
        """支持字典式设置"""
        self.set(key, value)


# 全局配置管理器实例
config_manager = ConfigManager()


# 便捷函数
def get_config(key_path: str, default: Any = None) -> Any:
    """
    便捷函数：获取配置值
    
    Args:
        key_path: 配置路径
        default: 默认值
        
    Returns:
        配置值
    """
    return config_manager.get(key_path, default)


def reload_config(config_name: Optional[str] = None):
    """
    便捷函数：重新加载配置
    
    Args:
        config_name: 配置名称，None为全部重新加载
    """
    config_manager.reload(config_name)


if __name__ == '__main__':
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    config = ConfigManager()
    
    # 测试获取配置
    print("配置管理器信息:", config.get_config_info())
    print("数据库主机:", config.get('main.database.host', 'localhost'))
    print("因子输出目录:", config.get('factors.settings.output_dir'))
    
    # 测试配置验证
    print("主配置验证:", config.validate_config('main'))
    print("因子配置验证:", config.validate_config('factors'))