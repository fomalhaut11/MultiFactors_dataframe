"""
因子文件验证器

提供因子文件格式验证、代码质量检查和测试功能。
"""

import ast
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import inspect
import pandas as pd

logger = logging.getLogger(__name__)


class FactorValidator:
    """
    因子文件验证器
    
    Features:
    - 验证因子文件格式和结构
    - 检查代码质量和潜在问题
    - 验证依赖关系和数据兼容性
    - 运行单元测试
    """
    
    def __init__(self):
        self.validation_rules = self._init_validation_rules()
        
    def _init_validation_rules(self) -> Dict[str, Any]:
        """初始化验证规则"""
        return {
            'required_meta_keys': [
                'name', 'category', 'description', 'dependencies'
            ],
            'optional_meta_keys': [
                'formula', 'data_frequency', 'calculation_method', 
                'version', 'author', 'created', 'last_modified',
                'requires_market_data'
            ],
            'valid_categories': [
                'profitability', 'value', 'quality', 'technical', 
                'risk', 'momentum', 'experimental'
            ],
            'naming_pattern': r'^[A-Za-z][A-Za-z0-9_]*$',
            'max_dependencies': 20,
            'required_functions': ['calculate']
        }
    
    def validate_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        验证因子文件
        
        Parameters
        ----------
        file_path : str or Path
            因子文件路径
            
        Returns
        -------
        dict
            验证结果，包含is_valid和详细信息
        """
        file_path = Path(file_path)
        result = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'file_path': str(file_path),
            'metadata': None
        }
        
        try:
            # 1. 基本文件检查
            self._validate_file_basic(file_path, result)
            
            # 2. 语法检查
            if result['is_valid']:
                self._validate_syntax(file_path, result)
            
            # 3. 结构检查
            if result['is_valid']:
                module = self._load_module_for_validation(file_path)
                if module:
                    self._validate_structure(module, result)
                    self._validate_metadata(module, result)
                    self._validate_functions(module, result)
                else:
                    result['is_valid'] = False
                    result['errors'].append("无法加载模块进行验证")
            
        except Exception as e:
            result['is_valid'] = False
            result['errors'].append(f"验证过程异常: {e}")
            logger.error(f"验证文件时发生异常 {file_path}: {e}")
        
        return result
    
    def _validate_file_basic(self, file_path: Path, result: Dict[str, Any]):
        """基本文件检查"""
        if not file_path.exists():
            result['is_valid'] = False
            result['errors'].append("文件不存在")
            return
        
        if not file_path.suffix == '.py':
            result['is_valid'] = False  
            result['errors'].append("文件扩展名必须为.py")
            return
        
        if file_path.stat().st_size == 0:
            result['is_valid'] = False
            result['errors'].append("文件为空")
            return
        
        # 检查文件名格式
        filename = file_path.stem
        if not re.match(self.validation_rules['naming_pattern'], filename):
            result['warnings'].append(f"文件名格式建议使用字母数字和下划线: {filename}")
    
    def _validate_syntax(self, file_path: Path, result: Dict[str, Any]):
        """语法检查"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 编译检查语法
            ast.parse(content, filename=str(file_path))
            
        except SyntaxError as e:
            result['is_valid'] = False
            result['errors'].append(f"语法错误 (行 {e.lineno}): {e.msg}")
        except UnicodeDecodeError as e:
            result['is_valid'] = False
            result['errors'].append(f"文件编码错误: {e}")
        except Exception as e:
            result['warnings'].append(f"语法检查异常: {e}")
    
    def _load_module_for_validation(self, file_path: Path) -> Optional[object]:
        """为验证目的加载模块"""
        try:
            import importlib.util
            
            spec = importlib.util.spec_from_file_location("validation_module", file_path)
            if spec is None or spec.loader is None:
                return None
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
            
        except Exception as e:
            logger.debug(f"验证时加载模块失败 {file_path}: {e}")
            return None
    
    def _validate_structure(self, module: object, result: Dict[str, Any]):
        """结构验证"""
        # 检查必需属性
        required_attrs = self.validation_rules['required_functions'] + ['FACTOR_META']
        
        for attr in required_attrs:
            if not hasattr(module, attr):
                result['is_valid'] = False
                result['errors'].append(f"缺少必需属性: {attr}")
        
        # 检查可选组件
        if hasattr(module, 'test_calculate'):
            if not callable(getattr(module, 'test_calculate')):
                result['warnings'].append("test_calculate存在但不可调用")
        else:
            result['warnings'].append("建议添加test_calculate函数进行单元测试")
    
    def _validate_metadata(self, module: object, result: Dict[str, Any]):
        """元数据验证"""
        if not hasattr(module, 'FACTOR_META'):
            return
        
        meta = module.FACTOR_META
        result['metadata'] = meta
        
        # 检查类型
        if not isinstance(meta, dict):
            result['is_valid'] = False
            result['errors'].append("FACTOR_META必须是字典类型")
            return
        
        # 检查必需键
        for key in self.validation_rules['required_meta_keys']:
            if key not in meta:
                result['is_valid'] = False
                result['errors'].append(f"FACTOR_META缺少必需键: {key}")
            elif not meta[key]:  # 检查空值
                result['errors'].append(f"FACTOR_META['{key}']不能为空")
        
        # 验证具体字段
        if 'name' in meta:
            if not re.match(self.validation_rules['naming_pattern'], meta['name']):
                result['warnings'].append("因子名称建议使用字母数字和下划线")
        
        if 'category' in meta:
            if meta['category'] not in self.validation_rules['valid_categories']:
                result['warnings'].append(f"未知的因子类别: {meta['category']}")
        
        if 'dependencies' in meta:
            deps = meta['dependencies']
            if not isinstance(deps, list):
                result['errors'].append("dependencies必须是列表类型")
            elif len(deps) > self.validation_rules['max_dependencies']:
                result['warnings'].append(f"依赖字段过多 ({len(deps)})，可能影响性能")
        
        # 检查版本格式
        if 'version' in meta:
            version = meta['version']
            if not re.match(r'^\d+\.\d+\.\d+$', str(version)):
                result['warnings'].append("建议使用语义版本格式 (x.y.z)")
    
    def _validate_functions(self, module: object, result: Dict[str, Any]):
        """函数验证"""
        if not hasattr(module, 'calculate'):
            return
        
        calculate_func = module.calculate
        
        if not callable(calculate_func):
            result['is_valid'] = False
            result['errors'].append("calculate必须是可调用函数")
            return
        
        # 检查函数签名
        try:
            sig = inspect.signature(calculate_func)
            params = list(sig.parameters.keys())
            
            if len(params) == 0:
                result['warnings'].append("calculate函数没有参数")
            else:
                # 检查第一个参数（应该是数据参数）
                first_param = list(sig.parameters.values())[0]
                if first_param.annotation == inspect.Parameter.empty:
                    result['warnings'].append("建议为第一个参数添加类型注解")
            
            # 检查返回值注解
            if sig.return_annotation == inspect.Parameter.empty:
                result['warnings'].append("建议添加返回值类型注解")
                
        except Exception as e:
            result['warnings'].append(f"无法检查函数签名: {e}")
        
        # 检查文档字符串
        if not calculate_func.__doc__:
            result['warnings'].append("calculate函数缺少文档字符串")
    
    def validate_factor_calculation(self, module: object, test_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        验证因子计算功能
        
        Parameters
        ----------
        module : object
            因子模块
        test_data : pd.DataFrame, optional
            测试数据
            
        Returns
        -------
        dict
            计算验证结果
        """
        result = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'test_result': None
        }
        
        if not hasattr(module, 'calculate'):
            result['is_valid'] = False
            result['errors'].append("缺少calculate函数")
            return result
        
        # 如果没有提供测试数据，生成简单测试数据
        if test_data is None:
            test_data = self._generate_test_data(module)
        
        try:
            # 执行计算
            calc_result = module.calculate(test_data)
            result['test_result'] = calc_result
            
            # 验证结果格式
            if calc_result is not None:
                if isinstance(calc_result, pd.Series):
                    # 检查是否有有效值
                    if calc_result.empty:
                        result['warnings'].append("计算结果为空Series")
                    elif calc_result.notna().sum() == 0:
                        result['warnings'].append("计算结果全为NaN")
                    
                    # 检查是否有无穷值
                    if (calc_result == float('inf')).any() or (calc_result == float('-inf')).any():
                        result['warnings'].append("计算结果包含无穷值")
                
                elif isinstance(calc_result, pd.DataFrame):
                    result['warnings'].append("返回DataFrame而非Series，可能不符合规范")
                else:
                    result['warnings'].append(f"返回类型 {type(calc_result)} 可能不合适")
            else:
                result['warnings'].append("计算返回None")
                
        except Exception as e:
            result['is_valid'] = False
            result['errors'].append(f"计算执行失败: {e}")
        
        return result
    
    def _generate_test_data(self, module: object) -> pd.DataFrame:
        """为测试生成简单数据"""
        import numpy as np
        
        # 从元数据获取依赖字段
        if hasattr(module, 'FACTOR_META') and 'dependencies' in module.FACTOR_META:
            deps = module.FACTOR_META['dependencies']
        else:
            deps = ['test_field']
        
        # 生成测试数据
        np.random.seed(42)
        dates = pd.to_datetime(['2020-12-31', '2021-12-31'])
        stocks = ['TEST1', 'TEST2']
        
        index_data = []
        data_rows = []
        
        for stock in stocks:
            for date in dates:
                index_data.append((date, stock))
                row_data = {}
                
                # 为每个依赖字段生成随机数据
                for dep in deps:
                    if dep in ['d_year']:
                        row_data[dep] = date.year
                    elif dep in ['d_quarter']:
                        row_data[dep] = date.quarter
                    else:
                        row_data[dep] = np.random.randn() * 1000000 + 5000000
                
                data_rows.append(row_data)
        
        index = pd.MultiIndex.from_tuples(index_data, names=['ReportDates', 'StockCodes'])
        return pd.DataFrame(data_rows, index=index)
    
    def run_unit_tests(self, module: object) -> Dict[str, Any]:
        """
        运行因子模块的单元测试
        
        Parameters
        ----------
        module : object
            因子模块
            
        Returns
        -------
        dict
            测试结果
        """
        result = {
            'has_tests': False,
            'tests_passed': False,
            'errors': [],
            'warnings': []
        }
        
        if hasattr(module, 'test_calculate'):
            result['has_tests'] = True
            
            try:
                # 执行测试函数
                test_func = module.test_calculate
                test_result = test_func()
                
                # 如果没有异常，认为测试通过
                result['tests_passed'] = True
                
            except Exception as e:
                result['errors'].append(f"单元测试失败: {e}")
        else:
            result['warnings'].append("未找到test_calculate函数")
        
        return result
    
    def validate_directory(self, directory: Union[str, Path]) -> Dict[str, Any]:
        """
        验证整个目录的因子文件
        
        Parameters
        ----------
        directory : str or Path
            目录路径
            
        Returns
        -------
        dict
            目录验证结果
        """
        directory = Path(directory)
        result = {
            'directory': str(directory),
            'total_files': 0,
            'valid_files': 0,
            'files': {},
            'summary': {
                'errors': 0,
                'warnings': 0
            }
        }
        
        if not directory.exists():
            result['error'] = "目录不存在"
            return result
        
        # 扫描.py文件
        for py_file in directory.rglob("*.py"):
            if py_file.name != "__init__.py":
                result['total_files'] += 1
                
                file_result = self.validate_file(py_file)
                result['files'][str(py_file)] = file_result
                
                if file_result['is_valid']:
                    result['valid_files'] += 1
                
                result['summary']['errors'] += len(file_result['errors'])
                result['summary']['warnings'] += len(file_result['warnings'])
        
        return result


# 导出
__all__ = [
    'FactorValidator'
]