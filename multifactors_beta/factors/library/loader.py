"""
因子文件动态加载器

负责扫描、加载和验证因子仓库中的Python文件，支持与现有注册系统并存。
"""

import os
import sys
import importlib.util
import inspect
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
import pandas as pd

logger = logging.getLogger(__name__)


class FactorLoader:
    """
    因子文件动态加载器
    
    Features:
    - 扫描repository目录下的因子文件
    - 动态加载和验证因子文件
    - 与现有注册系统无缝集成
    - 支持热加载和重载
    """
    
    def __init__(self, repository_path: Optional[Union[str, Path]] = None):
        """
        初始化加载器
        
        Parameters
        ----------
        repository_path : str or Path, optional
            因子仓库路径，默认为factors/repository
        """
        if repository_path is None:
            # 默认路径：当前factors模块下的repository目录
            current_dir = Path(__file__).parent.parent
            repository_path = current_dir / "repository"
        
        self.repository_path = Path(repository_path)
        self.loaded_modules = {}  # 记录已加载的模块
        self.factor_files = {}    # 记录因子文件路径
        
        logger.info(f"FactorLoader初始化，仓库路径: {self.repository_path}")
    
    def scan_repository(self) -> List[Path]:
        """
        扫描因子仓库，发现所有因子文件
        
        Returns
        -------
        List[Path]
            发现的因子文件路径列表
        """
        factor_files = []
        
        if not self.repository_path.exists():
            logger.warning(f"因子仓库路径不存在: {self.repository_path}")
            return factor_files
        
        # 递归扫描.py文件（排除__init__.py）
        for py_file in self.repository_path.rglob("*.py"):
            if py_file.name != "__init__.py":
                factor_files.append(py_file)
                logger.debug(f"发现因子文件: {py_file}")
        
        logger.info(f"扫描完成，发现 {len(factor_files)} 个因子文件")
        return factor_files
    
    def load_factor_file(self, factor_file: Path, auto_register: bool = True) -> Optional[object]:
        """
        加载单个因子文件
        
        Parameters
        ----------
        factor_file : Path
            因子文件路径
        auto_register : bool
            是否自动注册到全局注册表
            
        Returns
        -------
        module or None
            加载的模块对象，失败返回None
        """
        try:
            # 生成唯一的模块名称
            relative_path = factor_file.relative_to(self.repository_path)
            module_name = f"factor_repo_{str(relative_path).replace('/', '_').replace('.py', '')}"
            
            # 动态加载模块
            spec = importlib.util.spec_from_file_location(module_name, factor_file)
            if spec is None or spec.loader is None:
                raise ImportError(f"无法创建模块规格: {factor_file}")
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # 验证模块格式
            if not self._validate_factor_module(module, factor_file):
                return None
            
            # 记录加载信息
            self.loaded_modules[module_name] = module
            self.factor_files[module.FACTOR_META['name']] = factor_file
            
            # 自动注册到全局注册表
            if auto_register:
                self._register_to_global_registry(module, factor_file)
            
            logger.info(f"成功加载因子文件: {factor_file} -> {module.FACTOR_META['name']}")
            return module
            
        except Exception as e:
            logger.error(f"加载因子文件失败 {factor_file}: {e}")
            return None
    
    def _validate_factor_module(self, module: object, file_path: Path) -> bool:
        """
        验证因子模块的格式和必需组件
        
        Parameters
        ----------
        module : object
            加载的模块对象
        file_path : Path
            文件路径（用于错误报告）
            
        Returns
        -------
        bool
            验证是否通过
        """
        # 检查必需的属性
        required_attrs = ['FACTOR_META', 'calculate']
        
        for attr in required_attrs:
            if not hasattr(module, attr):
                logger.error(f"因子文件缺少必需属性 '{attr}': {file_path}")
                return False
        
        # 验证FACTOR_META格式
        meta = module.FACTOR_META
        required_meta_keys = ['name', 'category', 'description']
        
        if not isinstance(meta, dict):
            logger.error(f"FACTOR_META必须是字典类型: {file_path}")
            return False
        
        for key in required_meta_keys:
            if key not in meta:
                logger.error(f"FACTOR_META缺少必需键 '{key}': {file_path}")
                return False
        
        # 验证calculate函数
        if not callable(module.calculate):
            logger.error(f"calculate必须是可调用函数: {file_path}")
            return False
        
        # 检查函数签名
        try:
            sig = inspect.signature(module.calculate)
            if len(sig.parameters) == 0:
                logger.warning(f"calculate函数没有参数: {file_path}")
        except Exception as e:
            logger.warning(f"无法检查函数签名 {file_path}: {e}")
        
        return True
    
    def _register_to_global_registry(self, module: object, file_path: Path):
        """
        将加载的因子注册到全局注册表
        
        Parameters
        ----------
        module : object
            因子模块
        file_path : Path
            文件路径
        """
        try:
            from .factor_registry import factor_registry
            
            meta = module.FACTOR_META
            
            # 检查是否已注册（避免重复注册）
            if meta['name'] in factor_registry:
                logger.warning(f"因子 {meta['name']} 已存在，跳过注册")
                return
            
            # 注册到全局注册表
            factor_registry.register_from_file(
                name=meta['name'],
                category=meta['category'],
                description=meta['description'],
                dependencies=meta.get('dependencies', []),
                calculate_func=module.calculate,
                file_path=str(file_path),
                **{k: v for k, v in meta.items() 
                   if k not in ['name', 'category', 'description', 'dependencies']}
            )
            
            logger.info(f"因子 {meta['name']} 已注册到全局注册表")
            
        except Exception as e:
            logger.error(f"注册因子到全局注册表失败: {e}")
    
    def get_factor_metadata(self, factor_name: str) -> Optional[Dict[str, Any]]:
        """
        获取因子的元数据
        
        Parameters
        ----------
        factor_name : str
            因子名称
            
        Returns
        -------
        dict or None
            因子元数据，不存在返回None
        """
        for module in self.loaded_modules.values():
            if hasattr(module, 'FACTOR_META') and module.FACTOR_META.get('name') == factor_name:
                return module.FACTOR_META.copy()
        return None
    
    def get_factor_function(self, factor_name: str) -> Optional[callable]:
        """
        获取因子的计算函数
        
        Parameters
        ----------
        factor_name : str
            因子名称
            
        Returns
        -------
        callable or None
            因子计算函数，不存在返回None
        """
        for module in self.loaded_modules.values():
            if hasattr(module, 'FACTOR_META') and module.FACTOR_META.get('name') == factor_name:
                return module.calculate
        return None
    
    def list_loaded_factors(self) -> Dict[str, Dict[str, Any]]:
        """
        列出所有已加载的因子及其元数据
        
        Returns
        -------
        dict
            因子名称到元数据的映射
        """
        result = {}
        for module in self.loaded_modules.values():
            if hasattr(module, 'FACTOR_META'):
                meta = module.FACTOR_META
                result[meta['name']] = meta.copy()
        return result
    
    def reload_factor(self, factor_name: str) -> bool:
        """
        重新加载指定因子（用于开发调试）
        
        Parameters
        ----------
        factor_name : str
            因子名称
            
        Returns
        -------
        bool
            重新加载是否成功
        """
        if factor_name not in self.factor_files:
            logger.error(f"因子 {factor_name} 未找到对应文件")
            return False
        
        factor_file = self.factor_files[factor_name]
        
        # 重新加载文件
        new_module = self.load_factor_file(factor_file)
        
        if new_module is not None:
            logger.info(f"成功重新加载因子: {factor_name}")
            return True
        else:
            logger.error(f"重新加载因子失败: {factor_name}")
            return False
    
    def load_all_factors(self) -> Tuple[int, int]:
        """
        加载仓库中的所有因子文件
        
        Returns
        -------
        tuple
            (成功加载数量, 总文件数量)
        """
        factor_files = self.scan_repository()
        success_count = 0
        
        for factor_file in factor_files:
            module = self.load_factor_file(factor_file)
            if module is not None:
                success_count += 1
        
        logger.info(f"批量加载完成: {success_count}/{len(factor_files)} 个因子文件加载成功")
        return success_count, len(factor_files)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取加载器统计信息
        
        Returns
        -------
        dict
            统计信息
        """
        factors_by_category = {}
        total_factors = 0
        
        for module in self.loaded_modules.values():
            if hasattr(module, 'FACTOR_META'):
                meta = module.FACTOR_META
                category = meta.get('category', 'unknown')
                
                if category not in factors_by_category:
                    factors_by_category[category] = []
                factors_by_category[category].append(meta['name'])
                total_factors += 1
        
        return {
            'repository_path': str(self.repository_path),
            'total_factors': total_factors,
            'loaded_modules': len(self.loaded_modules),
            'factors_by_category': factors_by_category,
            'factor_files': len(self.factor_files)
        }


# 全局加载器实例（可选）
_global_loader: Optional[FactorLoader] = None


def get_global_loader() -> FactorLoader:
    """
    获取全局加载器实例
    
    Returns
    -------
    FactorLoader
        全局加载器实例
    """
    global _global_loader
    if _global_loader is None:
        _global_loader = FactorLoader()
    return _global_loader


def load_repository_factors() -> Tuple[int, int]:
    """
    便捷函数：加载仓库中的所有因子
    
    Returns
    -------
    tuple
        (成功加载数量, 总文件数量)
    """
    loader = get_global_loader()
    return loader.load_all_factors()


# 导出
__all__ = [
    'FactorLoader',
    'get_global_loader', 
    'load_repository_factors'
]