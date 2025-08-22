"""
数据处理重构一致性测试框架

此测试模块用于确保重构后的数据处理结果与原始实现完全一致
"""
import os
import sys
import numpy as np
import pandas as pd
import pickle
import hashlib
from pathlib import Path
from typing import Dict, Any, Tuple
import json
from datetime import datetime

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root.parent / "MultiFactors_1.0"))

# 导入原始实现（仅用于测试）
import DataProcessingFromLocal as original_module


class DataProcessorConsistencyTester:
    """数据处理一致性测试器"""
    
    def __init__(self, test_data_path: str = None):
        """
        初始化测试器
        
        Args:
            test_data_path: 测试数据保存路径
        """
        self.test_data_path = test_data_path or os.path.join(project_root, "tests", "test_data", "processor")
        os.makedirs(self.test_data_path, exist_ok=True)
        self.results = {}
        
    def compute_data_fingerprint(self, data: Any) -> str:
        """
        计算数据指纹（用于比较数据一致性）
        
        Args:
            data: 要计算指纹的数据
            
        Returns:
            数据的MD5指纹
        """
        if isinstance(data, pd.DataFrame):
            # DataFrame转换为字节串
            buffer = data.to_json(orient='split', date_format='iso').encode()
        elif isinstance(data, pd.Series):
            # Series转换为字节串
            buffer = data.to_json(orient='split', date_format='iso').encode()
        elif isinstance(data, np.ndarray):
            # NumPy数组转换为字节串
            buffer = data.tobytes()
        elif isinstance(data, dict):
            # 字典转换为JSON字节串
            buffer = json.dumps(data, sort_keys=True, default=str).encode()
        else:
            # 其他类型使用pickle
            buffer = pickle.dumps(data)
            
        return hashlib.md5(buffer).hexdigest()
    
    def save_reference_output(self, func_name: str, output_data: Any, params: Dict = None):
        """
        保存原始函数的输出作为参考
        
        Args:
            func_name: 函数名称
            output_data: 函数输出数据
            params: 函数参数
        """
        fingerprint = self.compute_data_fingerprint(output_data)
        
        reference_data = {
            'func_name': func_name,
            'params': params or {},
            'fingerprint': fingerprint,
            'timestamp': datetime.now().isoformat(),
            'data_type': type(output_data).__name__,
            'data_shape': self._get_data_shape(output_data)
        }
        
        # 保存参考数据
        ref_file = os.path.join(self.test_data_path, f"{func_name}_reference.json")
        with open(ref_file, 'w') as f:
            json.dump(reference_data, f, indent=2)
            
        # 保存实际数据样本（用于调试）
        if isinstance(output_data, (pd.DataFrame, pd.Series)):
            sample_file = os.path.join(self.test_data_path, f"{func_name}_sample.pkl")
            output_data.head(100).to_pickle(sample_file)
            
    def _get_data_shape(self, data: Any) -> Any:
        """获取数据形状信息"""
        if hasattr(data, 'shape'):
            return data.shape
        elif isinstance(data, dict):
            return {k: self._get_data_shape(v) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            return len(data)
        else:
            return None
            
    def compare_outputs(self, func_name: str, new_output: Any, params: Dict = None) -> Dict[str, Any]:
        """
        比较新实现的输出与参考输出
        
        Args:
            func_name: 函数名称
            new_output: 新实现的输出
            params: 函数参数
            
        Returns:
            比较结果
        """
        # 计算新输出的指纹
        new_fingerprint = self.compute_data_fingerprint(new_output)
        
        # 加载参考数据
        ref_file = os.path.join(self.test_data_path, f"{func_name}_reference.json")
        if not os.path.exists(ref_file):
            return {
                'status': 'no_reference',
                'message': f'No reference data found for {func_name}'
            }
            
        with open(ref_file, 'r') as f:
            reference_data = json.load(f)
            
        # 比较结果
        is_consistent = new_fingerprint == reference_data['fingerprint']
        
        result = {
            'func_name': func_name,
            'is_consistent': is_consistent,
            'reference_fingerprint': reference_data['fingerprint'],
            'new_fingerprint': new_fingerprint,
            'data_type_match': type(new_output).__name__ == reference_data['data_type'],
            'shape_match': self._get_data_shape(new_output) == reference_data['data_shape']
        }
        
        # 如果不一致，进行详细比较
        if not is_consistent and isinstance(new_output, (pd.DataFrame, pd.Series)):
            result['detailed_comparison'] = self._detailed_dataframe_comparison(
                func_name, new_output
            )
            
        return result
        
    def _detailed_dataframe_comparison(self, func_name: str, new_data: pd.DataFrame) -> Dict:
        """对DataFrame进行详细比较"""
        sample_file = os.path.join(self.test_data_path, f"{func_name}_sample.pkl")
        if not os.path.exists(sample_file):
            return {'error': 'No sample data available'}
            
        ref_data = pd.read_pickle(sample_file)
        
        # 确保索引和列对齐
        if isinstance(new_data, pd.DataFrame) and isinstance(ref_data, pd.DataFrame):
            common_index = new_data.index.intersection(ref_data.index)
            common_cols = new_data.columns.intersection(ref_data.columns)
            
            new_aligned = new_data.loc[common_index, common_cols]
            ref_aligned = ref_data.loc[common_index, common_cols]
            
            # 数值比较
            diff = np.abs(new_aligned - ref_aligned)
            max_diff = diff.max().max()
            
            return {
                'max_absolute_difference': float(max_diff) if not pd.isna(max_diff) else None,
                'shape_difference': {
                    'reference': ref_data.shape,
                    'new': new_data.shape
                },
                'index_difference': {
                    'only_in_reference': len(ref_data.index.difference(new_data.index)),
                    'only_in_new': len(new_data.index.difference(ref_data.index))
                }
            }
            
        return {'error': 'Cannot perform detailed comparison'}
        
    def generate_test_report(self) -> str:
        """生成测试报告"""
        report = ["# 数据处理重构一致性测试报告\n"]
        report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        all_consistent = True
        for func_name, result in self.results.items():
            report.append(f"\n## 函数: {func_name}")
            if result.get('is_consistent'):
                report.append("[OK] **测试通过** - 输出完全一致")
            else:
                report.append("[FAIL] **测试失败** - 输出不一致")
                all_consistent = False
                
            report.append(f"- 数据类型匹配: {result.get('data_type_match', 'N/A')}")
            report.append(f"- 数据形状匹配: {result.get('shape_match', 'N/A')}")
            
            if 'detailed_comparison' in result:
                report.append("\n### 详细比较:")
                for key, value in result['detailed_comparison'].items():
                    report.append(f"- {key}: {value}")
                    
        report.append(f"\n## 总体结果: {'[OK] 所有测试通过' if all_consistent else '[FAIL] 存在不一致'}")
        
        return '\n'.join(report)
        

def create_reference_outputs():
    """创建原始实现的参考输出"""
    print("正在生成原始实现的参考输出...")
    
    tester = DataProcessorConsistencyTester()
    
    # 1. 测试get_price_data
    print("测试 get_price_data...")
    try:
        price_df, stock3d = original_module.get_price_data()
        tester.save_reference_output('get_price_data_pricedf', price_df)
        tester.save_reference_output('get_price_data_stock3d', stock3d)
        print("[v] get_price_data 参考数据已保存")
    except Exception as e:
        print(f"[x] get_price_data 失败: {e}")
    
    # 2. 测试日期序列生成
    print("测试 date_serries...")
    try:
        if 'price_df' in locals():
            daily_dates = original_module.date_serries(price_df, type="daily")
            weekly_dates = original_module.date_serries(price_df, type="weekly")
            monthly_dates = original_module.date_serries(price_df, type="monthly")
            
            tester.save_reference_output('date_serries_daily', daily_dates)
            tester.save_reference_output('date_serries_weekly', weekly_dates)
            tester.save_reference_output('date_serries_monthly', monthly_dates)
            print("[v] date_serries 参考数据已保存")
    except Exception as e:
        print(f"[x] date_serries 失败: {e}")
        
    # 3. 测试收益率计算（使用小样本）
    print("测试 logreturndf_dateserries...")
    try:
        if 'price_df' in locals() and 'daily_dates' in locals():
            # 使用前100个日期的小样本进行测试
            sample_dates = daily_dates[:100]
            sample_return = original_module.logreturndf_dateserries(
                price_df, sample_dates, ReturnType="o2o"
            )
            tester.save_reference_output(
                'logreturndf_dateserries_o2o', 
                sample_return,
                {'dates_count': 100, 'return_type': 'o2o'}
            )
            print("[v] logreturndf_dateserries 参考数据已保存")
    except Exception as e:
        print(f"[x] logreturndf_dateserries 失败: {e}")
        
    print("\n参考数据生成完成！")
    

if __name__ == "__main__":
    # 生成参考输出
    create_reference_outputs()