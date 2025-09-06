"""
数据处理基类

定义数据处理器的通用接口和基础功能
"""
import os
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import hashlib
import json
from datetime import datetime

# 添加项目根目录到路径
import sys
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config import get_config
from core.utils.exit_handler import ExitHandler


class BaseDataProcessor(ABC):
    """数据处理器基类"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化数据处理器
        
        Args:
            config_path: 配置文件路径
        """
        self.logger = self._setup_logger()
        self._setup_paths()
        self._register_exit_handler()
        
        # 处理状态跟踪
        self._processing_history = []
        self._cache = {}
        
    def _setup_logger(self) -> logging.Logger:
        """设置日志器"""
        logger_name = f"{self.__class__.__name__}"
        logger = logging.getLogger(logger_name)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
            
        return logger
        
    def _setup_paths(self):
        """设置数据路径"""
        self.data_root = Path(get_config('main.paths.data_root'))
        self.cache_dir = self.data_root / 'cache'
        self.results_dir = self.data_root / 'processed'
        
        # 确保目录存在
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def _register_exit_handler(self):
        """注册退出处理器"""
        from core.utils.exit_handler import exit_handler
        
        def cleanup():
            self.logger.info(f"{self.__class__.__name__} 正在清理资源...")
            self._save_processing_history()
            
        exit_handler.register_cleanup(cleanup, f"{self.__class__.__name__}_cleanup")
        
    def _save_processing_history(self):
        """保存处理历史"""
        if self._processing_history:
            history_file = self.cache_dir / f"{self.__class__.__name__}_history.json"
            with open(history_file, 'w') as f:
                json.dump(self._processing_history, f, indent=2, default=str)
                
    def _record_processing(self, operation: str, params: Dict, result_info: Dict):
        """记录处理操作"""
        record = {
            'timestamp': datetime.now().isoformat(),
            'operation': operation,
            'params': params,
            'result': result_info
        }
        self._processing_history.append(record)
        
    def compute_checksum(self, data: Any) -> str:
        """
        计算数据校验和（用于验证数据一致性）
        
        Args:
            data: 要计算校验和的数据
            
        Returns:
            数据的MD5校验和
        """
        if isinstance(data, pd.DataFrame):
            buffer = data.to_json(orient='split', date_format='iso').encode()
        elif isinstance(data, pd.Series):
            buffer = data.to_json(orient='split', date_format='iso').encode()
        elif isinstance(data, np.ndarray):
            buffer = data.tobytes()
        elif isinstance(data, dict):
            buffer = json.dumps(data, sort_keys=True, default=str).encode()
        else:
            buffer = pickle.dumps(data)
            
        return hashlib.md5(buffer).hexdigest()
        
    def save_processed_data(self, data: Any, filename: str, 
                          compute_checksum: bool = True) -> Dict[str, str]:
        """
        保存处理后的数据
        
        Args:
            data: 要保存的数据
            filename: 文件名（不含路径）
            compute_checksum: 是否计算校验和
            
        Returns:
            保存信息字典
        """
        filepath = self.results_dir / filename
        
        # 保存数据
        if filename.endswith('.pkl'):
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
        elif filename.endswith('.csv'):
            if isinstance(data, pd.DataFrame):
                data.to_csv(filepath)
            else:
                raise ValueError("CSV格式仅支持DataFrame")
        else:
            raise ValueError(f"不支持的文件格式: {filename}")
            
        # 计算校验和
        save_info = {
            'filepath': str(filepath),
            'timestamp': datetime.now().isoformat(),
            'size': os.path.getsize(filepath)
        }
        
        if compute_checksum:
            save_info['checksum'] = self.compute_checksum(data)
            
        self.logger.info(f"数据已保存至: {filepath}")
        return save_info
        
    def load_processed_data(self, filename: str, verify_checksum: Optional[str] = None) -> Any:
        """
        加载处理后的数据
        
        Args:
            filename: 文件名
            verify_checksum: 要验证的校验和（可选）
            
        Returns:
            加载的数据
        """
        filepath = self.results_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"文件不存在: {filepath}")
            
        # 加载数据
        if filename.endswith('.pkl'):
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
        elif filename.endswith('.csv'):
            data = pd.read_csv(filepath)
        else:
            raise ValueError(f"不支持的文件格式: {filename}")
            
        # 验证校验和
        if verify_checksum:
            actual_checksum = self.compute_checksum(data)
            if actual_checksum != verify_checksum:
                raise ValueError(
                    f"数据校验和不匹配: 期望 {verify_checksum}, 实际 {actual_checksum}"
                )
                
        return data
        
    def get_cache_key(self, operation: str, **params) -> str:
        """生成缓存键"""
        params_str = json.dumps(params, sort_keys=True, default=str)
        return f"{operation}_{hashlib.md5(params_str.encode()).hexdigest()}"
        
    def get_from_cache(self, cache_key: str) -> Optional[Any]:
        """从缓存获取数据"""
        return self._cache.get(cache_key)
        
    def save_to_cache(self, cache_key: str, data: Any):
        """保存数据到缓存"""
        self._cache[cache_key] = data
        
    @abstractmethod
    def process(self, **kwargs) -> Any:
        """
        执行数据处理（子类必须实现）
        
        Returns:
            处理后的数据
        """
        raise NotImplementedError("子类必须实现process方法")
        
    @abstractmethod
    def validate_input(self, **kwargs) -> bool:
        """
        验证输入数据（子类必须实现）
        
        Returns:
            验证是否通过
        """
        raise NotImplementedError("子类必须实现validate_input方法")
        
    def run(self, **kwargs) -> Any:
        """
        运行数据处理流程
        
        Returns:
            处理结果
        """
        try:
            # 验证输入
            if not self.validate_input(**kwargs):
                raise ValueError("输入数据验证失败")
                
            # 检查缓存
            cache_key = self.get_cache_key(self.__class__.__name__, **kwargs)
            cached_result = self.get_from_cache(cache_key)
            if cached_result is not None:
                self.logger.info("使用缓存结果")
                return cached_result
                
            # 执行处理
            self.logger.info(f"开始执行 {self.__class__.__name__} 处理...")
            result = self.process(**kwargs)
            
            # 保存到缓存
            self.save_to_cache(cache_key, result)
            
            # 记录处理历史
            self._record_processing(
                self.__class__.__name__,
                kwargs,
                {'status': 'success', 'checksum': self.compute_checksum(result)}
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"处理失败: {e}")
            self._record_processing(
                self.__class__.__name__,
                kwargs,
                {'status': 'failed', 'error': str(e)}
            )
            raise