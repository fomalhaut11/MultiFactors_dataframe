#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一配置管理模块
提供全局配置管理功能，支持YAML配置文件、环境变量和默认值
"""

import os
import sys
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
from datetime import datetime
try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

class ConfigManager:
    """统一配置管理器"""
    
    _instance = None
    _config = None
    
    def __new__(cls):
        """单例模式"""
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """初始化配置管理器"""
        if self._config is None:
            self._config = {}
            self._load_config()
            self._setup_logging()
    
    def _load_config(self):
        """加载配置文件"""
        # 获取项目根目录
        project_root = Path(__file__).parent.parent.parent 
        
        # 首先加载环境变量
        env_file = project_root / ".env"
        if env_file.exists() and load_dotenv is not None:
            load_dotenv(env_file)
        elif env_file.exists():
            # .env file found but python-dotenv not installed - silently skip
            pass
        
        # 按优先级查找配置文件
        config_files = [
            project_root / "config.yaml",           # 主配置文件
            project_root / "config.yml",            # 备用配置文件
            project_root / "default_config.yaml"    # 默认配置文件
        ]
        
        config_loaded = False
        
        for config_file in config_files:
            if config_file.exists():
                try:
                    with open(config_file, 'r', encoding='utf-8') as f:
                        file_config = yaml.safe_load(f)
                        if file_config:
                            # 替换环境变量
                            file_config = self._replace_env_variables(file_config)
                            self._config.update(file_config)
                            config_loaded = True
                            # Config file loaded successfully
                            break
                except Exception as e:
                    # Failed to load config file, try next one
                    continue
        
        if not config_loaded:
            # Config file not found, using default config
            self._load_default_config()
        
        # 加载环境变量覆盖
        self._load_env_variables()
        
        # 验证和处理路径
        self._process_paths()
    
    def _replace_env_variables(self, config_dict):
        """递归替换配置中的环境变量"""
        import re
        
        if isinstance(config_dict, dict):
            return {key: self._replace_env_variables(value) for key, value in config_dict.items()}
        elif isinstance(config_dict, list):
            return [self._replace_env_variables(item) for item in config_dict]
        elif isinstance(config_dict, str):
            # 匹配 ${VAR_NAME} 格式的环境变量
            pattern = r'\$\{([^}]+)\}'
            matches = re.findall(pattern, config_dict)
            result = config_dict
            for match in matches:
                env_value = os.getenv(match, f'${{{match}}}')  # 如果环境变量不存在，保持原样
                result = result.replace(f'${{{match}}}', env_value)
            return result
        else:
            return config_dict
    
    def _load_default_config(self):
        """加载默认配置"""
        project_root = Path(__file__).parent
        
        self._config = {
            # 数据库配置
            'database': {
                'host': '198.16.102.88',
                'user': 'sa', 
                'password': 'Sy88sjk',
                'database': 'stock_data',
                'min_database': 'stock_min1',
                'jq_database': 'jqdata',
                'wind_database': 'Wind'
            },
            
            # 数据路径配置
            'paths': {
                'project_root': str(project_root),
                'data_root': str(project_root.parent.parent.parent / 'StockData'),
                'raw_factors': str(project_root.parent.parent.parent / 'StockData' / 'RawFactors'),
                'raw_factors_alpha191': str(project_root.parent.parent.parent / 'StockData' / 'RawFactors_alpha191'),
                'orthogonalization_factors': str(project_root.parent.parent.parent / 'StockData' / 'OrthogonalizationFactors'),
                'classification_data': str(project_root.parent.parent.parent / 'StockData' / 'Classificationdata'),
                'single_factor_test': str(project_root.parent.parent.parent / 'StockData' / 'SingleFactorTestData'),
                'temp_data': str(project_root.parent.parent.parent / 'StockData' / 'TempData'),
                'factors': str(project_root.parent.parent.parent / 'StockData' / 'factors'),
                'results': str(project_root / 'results'),
                'logs': str(project_root / 'logs'),
                'cache': str(project_root / 'cache')
            },
            
            # 因子测试配置
            'factor_test': {
                'begin_date': '2018-01-01',
                'end_date': '2025-12-31',
                'backtest_type': 'daily',  # daily, weekly, monthly
                'group_nums': 10,
                'netral_base': True,
                'back_test_trading_price': 'o2o',  # o2o, vwap
                'base_factors': ['LogMarketCap'],
                'classification_name': 'classification_one_hot',
                'matrix_name': 'similarity_matrix',
                'nearest_nums': 50
            },
            
            # 数据处理配置
            'data_processing': {
                'outlier_method': 'IQR',  # IQR, median, mean
                'outlier_param': 5,
                'normalization_method': 'zscore',  # zscore, minmax, robust
                'fill_method': 'ffill',  # ffill, bfill, median, mean
                'min_periods': 252,  # 最少交易日数
                'max_missing_ratio': 0.5  # 最大缺失率
            },
            
            # 机器学习配置
            'ml_models': {
                'lightgbm': {
                    'objective': 'regression',
                    'metric': 'rmse',
                    'boosting_type': 'gbdt',
                    'num_leaves': 31,
                    'learning_rate': 0.05,
                    'feature_fraction': 0.9,
                    'bagging_fraction': 0.8,
                    'bagging_freq': 5,
                    'verbose': -1,
                    'random_state': 42,
                    'n_estimators': 100,
                    'early_stopping_rounds': 20
                },
                'cross_validation': {
                    'n_splits': 5,
                    'test_size': 0.2,
                    'random_state': 42
                }
            },
            
            # 股票池配置
            'stock_pools': {
                'min_price': 2.0,  # 最低股价
                'min_market_cap': 50e8,  # 最小市值(50亿)
                'max_suspend_days': 30,  # 最大停牌天数
                'exclude_st': True,  # 排除ST股票
                'exclude_new_stocks': True,  # 排除新股(上市不足N天)
                'new_stock_days': 252,  # 新股定义天数
                'liquidity_threshold': 1e6,  # 流动性阈值(成交额)
                'exclude_exchanges': ['BJ']  # 排除交易所
            },
            
            # 因子工程配置
            'factor_engineering': {
                'technical_periods': [5, 10, 20, 60, 120],
                'ma_periods': [5, 10, 20, 60, 120],
                'volatility_periods': [20, 60, 120],
                'momentum_periods': [5, 10, 20, 60],
                'rsi_period': 14,
                'bollinger_period': 20,
                'bollinger_std': 2.0
            },
            
            # 风险管理配置
            'risk_management': {
                'max_position': 0.05,  # 单股最大仓位
                'max_sector_exposure': 0.3,  # 单行业最大暴露
                'max_drawdown': 0.15,  # 最大回撤
                'stop_loss': 0.08,  # 止损线
                'rebalance_frequency': 'monthly'  # 调仓频率
            },
            
            # 系统配置
            'system': {
                'log_level': 'INFO',
                'log_format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'log_file': 'multifactor.log',
                'cache_enabled': True,
                'parallel_jobs': -1,  # 并行任务数量
                'memory_limit': '8GB',
                'chunk_size': 10000
            }
        }
    
    def _load_env_variables(self):
        """加载环境变量覆盖配置"""
        env_mappings = {
            'DB_HOST': ['database', 'host'],
            'DB_USER': ['database', 'user'],
            'DB_PASSWORD': ['database', 'password'],
            'DB_DATABASE': ['database', 'database'],
            'DATA_ROOT': ['paths', 'data_root'],
            'LOG_LEVEL': ['system', 'log_level'],
            'PARALLEL_JOBS': ['system', 'parallel_jobs']
        }
        
        for env_var, config_path in env_mappings.items():
            env_value = os.environ.get(env_var)
            if env_value:
                # 设置嵌套配置
                current = self._config
                for key in config_path[:-1]:
                    current = current.setdefault(key, {})
                
                # 类型转换
                if config_path[-1] in ['parallel_jobs']:
                    env_value = int(env_value)
                elif config_path[-1] in ['cache_enabled']:
                    env_value = env_value.lower() in ['true', '1', 'yes']
                
                current[config_path[-1]] = env_value
                # Environment variable override applied
    
    def _process_paths(self):
        """处理和验证路径"""
        paths_config = self._config.get('paths', {})
        
        # 确保路径存在
        for key, path_str in paths_config.items():
            if path_str:
                path = Path(path_str)
                try:
                    # 创建不存在的目录
                    if key in ['results', 'logs', 'cache', 'temp_data']:
                        path.mkdir(parents=True, exist_ok=True)
                    
                    # 转换为绝对路径
                    paths_config[key] = str(path.resolve())
                except Exception as e:
                    # Path processing failed, using original value
                    pass
    
    def _setup_logging(self):
        """设置日志系统"""
        log_config = self.get('system') or {}
        log_level = log_config.get('log_level', 'INFO')
        log_format = log_config.get('log_format', '%(asctime)s - %(levelname)s - %(message)s')
        log_file = log_config.get('log_file', 'multifactor.log')
        
        # 设置日志目录
        log_dir = Path(self.get('paths', 'logs') or './logs')
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / log_file
        
        # 配置日志
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format=log_format,
            handlers=[
                logging.FileHandler(log_path, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger('ConfigManager')
        self.logger.info("配置管理器初始化完成")
    
    def get(self, *keys, default=None) -> Any:
        """
        获取配置值
        
        Args:
            *keys: 配置键路径
            default: 默认值
            
        Returns:
            配置值
            
        Examples:
            config.get('database', 'host')
            config.get('paths', 'data_root')
        """
        current = self._config
        
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        
        return current
    
    def get_config(self) -> dict:
        """
        获取完整配置字典
        
        Returns:
            完整的配置字典
        """
        return self._config
    
    def set(self, *keys_and_value) -> None:
        """
        设置配置值
        
        Args:
            *keys_and_value: 配置键路径和值，最后一个参数为值
            
        Examples:
            config.set('database', 'host', 'localhost')
            config.set('system', 'log_level', 'DEBUG')
        """
        if len(keys_and_value) < 2:
            raise ValueError("至少需要一个键和一个值")
        
        keys = keys_and_value[:-1]
        value = keys_and_value[-1]
        
        current = self._config
        for key in keys[:-1]:
            current = current.setdefault(key, {})
        
        current[keys[-1]] = value
        
        if hasattr(self, 'logger'):
            self.logger.info(f"配置更新: {'.'.join(keys)} = {value}")
    
    def get_path(self, path_key: str, default: str = None) -> str:
        """
        获取路径配置
        
        Args:
            path_key: 路径键名
            default: 默认路径
            
        Returns:
            路径字符串
        """
        path = self.get('paths', path_key, default=default)
        if path:
            return str(Path(path).resolve())
        return default or ""
    
    def get_database_config(self, db_name: str = 'database') -> Dict[str, str]:
        """
        获取数据库配置
        
        Args:
            db_name: 数据库配置名称
            
        Returns:
            数据库配置字典
        """
        db_config = self.get('database', default={})
        
        if db_name != 'database':
            # 获取特定数据库配置，使用主数据库配置作为基础
            specific_db = db_config.get(db_name, db_config.get('database', 'stock_data'))
            result = db_config.copy()
            result['database'] = specific_db
            return result
        
        return db_config
    
    def get_factor_test_config(self) -> Dict[str, Any]:
        """获取因子测试配置"""
        return self.get('factor_test', default={})
    
    def get_ml_config(self, model_name: str = 'lightgbm') -> Dict[str, Any]:
        """
        获取机器学习模型配置
        
        Args:
            model_name: 模型名称
            
        Returns:
            模型配置字典
        """
        return self.get('ml_models', model_name, default={})
    
    def get_stock_pool_config(self) -> Dict[str, Any]:
        """获取股票池配置"""
        return self.get('stock_pools', default={})
    
    def update_from_dict(self, config_dict: Dict[str, Any]) -> None:
        """
        从字典更新配置
        
        Args:
            config_dict: 配置字典
        """
        def deep_update(base_dict, update_dict):
            for key, value in update_dict.items():
                if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
                    deep_update(base_dict[key], value)
                else:
                    base_dict[key] = value
        
        deep_update(self._config, config_dict)
        
        if hasattr(self, 'logger'):
            self.logger.info("配置已从字典更新")
    
    def save_config(self, file_path: Optional[str] = None) -> bool:
        """
        保存配置到文件
        
        Args:
            file_path: 保存路径，默认为config.yaml
            
        Returns:
            是否保存成功
        """
        if file_path is None:
            file_path = Path(__file__).parent / "config.yaml"
        else:
            file_path = Path(file_path)
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(self._config, f, default_flow_style=False, 
                         allow_unicode=True, indent=2)
            
            if hasattr(self, 'logger'):
                self.logger.info(f"配置已保存到: {file_path}")
            return True
            
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"保存配置失败: {e}")
            else:
                # Failed to save config - logger not available
                pass
            return False
    
    def reload_config(self) -> None:
        """重新加载配置"""
        self._config = {}
        self._load_config()
        
        if hasattr(self, 'logger'):
            self.logger.info("配置已重新加载")
    
    def validate_config(self) -> bool:
        """
        验证配置完整性
        
        Returns:
            配置是否有效
        """
        required_sections = ['database', 'paths', 'factor_test', 'system']
        
        for section in required_sections:
            if section not in self._config:
                if hasattr(self, 'logger'):
                    self.logger.error(f"缺少必需的配置段: {section}")
                return False
        
        # 验证数据库配置
        db_config = self.get('database', default={})
        required_db_keys = ['host', 'user', 'password', 'database']
        for key in required_db_keys:
            if key not in db_config:
                if hasattr(self, 'logger'):
                    self.logger.error(f"缺少数据库配置: {key}")
                return False
        
        # 验证路径配置
        data_root = self.get('paths', 'data_root')
        if not data_root or not Path(data_root).exists():
            if hasattr(self, 'logger'):
                self.logger.warning(f"数据根目录不存在: {data_root}")
        
        return True
    
    def get_all_config(self) -> Dict[str, Any]:
        """获取所有配置"""
        return self._config.copy()
    
    def print_config(self) -> None:
        """打印当前配置"""
        print("=" * 60)
        print("Current configuration:")
        print("=" * 60)
        print(yaml.dump(self._config, default_flow_style=False, 
                       allow_unicode=True, indent=2))
        print("=" * 60)

# 全局配置实例
config = ConfigManager()

# 便捷函数
def get_config(*keys, default=None):
    """获取配置的便捷函数"""
    return config.get(*keys, default=default)

def get_path(path_key: str, default: str = None) -> str:
    """获取路径的便捷函数"""
    return config.get_path(path_key, default)

def get_database_config(db_name: str = 'database') -> Dict[str, str]:
    """获取数据库配置的便捷函数"""
    return config.get_database_config(db_name)

if __name__ == "__main__":
    # 测试配置管理器
    print("Testing config manager...")
    
    # 验证配置
    if config.validate_config():
        print("[OK] Config validation passed")
    else:
        print("[FAIL] Config validation failed")
    
    # 打印配置
    config.print_config()
    
    # 测试获取配置
    print(f"\nDatabase host: {config.get('database', 'host')}")
    print(f"Data root: {config.get('paths', 'data_root')}")
    print(f"Factor test config: {config.get_factor_test_config()}")