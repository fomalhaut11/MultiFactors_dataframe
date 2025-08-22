#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
字段映射工具 - 提供中英文字段映射和字段验证功能
"""
import json
import yaml
from pathlib import Path
from typing import Dict, Optional, List, Union
import logging

logger = logging.getLogger(__name__)


class FieldMapper:
    """字段映射工具类"""
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        初始化字段映射器
        
        Parameters:
        -----------
        config_path : str or Path, optional
            配置文件路径，默认使用内置配置
        """
        if config_path is None:
            config_path = Path(__file__).parent / "field_mapping.json"
        
        self.config_path = Path(config_path)
        self.field_mapping = self._load_config()
        
        logger.info(f"字段映射器已初始化，配置文件: {self.config_path}")
    
    def _load_config(self) -> Dict:
        """加载配置文件"""
        try:
            if self.config_path.suffix == '.json':
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            elif self.config_path.suffix in ['.yaml', '.yml']:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f)
            else:
                raise ValueError(f"不支持的配置文件格式: {self.config_path.suffix}")
        except Exception as e:
            logger.error(f"加载字段映射配置失败: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """获取默认配置（兜底）"""
        return {
            'field_descriptions': {
                'balance_sheet': {'name': '资产负债表', 'fields': {}},
                'income_statement': {'name': '利润表', 'fields': {}},
                'cash_flow': {'name': '现金流量表', 'fields': {}}
            },
            'common_fields': {
                'earnings': {'field_name': 'DEDUCTEDPROFIT', 'chinese_name': '扣非净利润'},
                'revenue': {'field_name': 'TOT_OPER_REV', 'chinese_name': '营业收入'},
                'operating_cash_flow': {'field_name': 'NETCASH_OPER', 'chinese_name': '经营现金流'}
            },
            'table_attribution': {}
        }
    
    def get_chinese_name(self, field_name: str) -> Optional[str]:
        """
        获取字段的中文名称
        
        Parameters:
        -----------
        field_name : str
            英文字段名
            
        Returns:
        --------
        str or None
            中文字段名，如果未找到返回None
        """
        # 首先在各表中查找
        for table_info in self.field_mapping['field_descriptions'].values():
            if field_name in table_info['fields']:
                return table_info['fields'][field_name]
        
        # 然后在常用字段中查找
        for common_field in self.field_mapping['common_fields'].values():
            if common_field['field_name'] == field_name:
                return common_field['chinese_name']
        
        return None
    
    def get_table_name(self, field_name: str) -> Optional[str]:
        """
        获取字段所属的表名
        
        Parameters:
        -----------
        field_name : str
            字段名
            
        Returns:
        --------
        str or None
            表名，如果未找到返回None
        """
        # 首先查看表归属映射
        if field_name in self.field_mapping['table_attribution']:
            return self.field_mapping['table_attribution'][field_name]
        
        # 然后在各表中查找
        for table_name, table_info in self.field_mapping['field_descriptions'].items():
            if field_name in table_info['fields']:
                return table_name
        
        # 最后在常用字段中查找
        for common_field in self.field_mapping['common_fields'].values():
            if common_field['field_name'] == field_name:
                return common_field.get('table')
        
        return None
    
    def get_common_field(self, logical_name: str) -> Optional[Dict]:
        """
        根据逻辑名获取常用字段信息
        
        Parameters:
        -----------
        logical_name : str
            逻辑字段名（如 'earnings', 'revenue'）
            
        Returns:
        --------
        dict or None
            字段信息字典，包含field_name, chinese_name, table等
        """
        return self.field_mapping['common_fields'].get(logical_name)
    
    def get_field_name(self, logical_name: str) -> Optional[str]:
        """
        根据逻辑名获取实际字段名
        
        Parameters:
        -----------
        logical_name : str
            逻辑字段名
            
        Returns:
        --------
        str or None
            实际字段名
        """
        common_field = self.get_common_field(logical_name)
        return common_field['field_name'] if common_field else None
    
    def validate_fields(self, field_names: List[str]) -> Dict[str, bool]:
        """
        验证字段名是否存在
        
        Parameters:
        -----------
        field_names : list
            字段名列表
            
        Returns:
        --------
        dict
            验证结果，{field_name: bool}
        """
        results = {}
        for field_name in field_names:
            results[field_name] = self.get_chinese_name(field_name) is not None
        return results
    
    def get_fields_by_table(self, table_name: str) -> Dict[str, str]:
        """
        获取指定表的所有字段
        
        Parameters:
        -----------
        table_name : str
            表名
            
        Returns:
        --------
        dict
            {field_name: chinese_name}
        """
        if table_name in self.field_mapping['field_descriptions']:
            return self.field_mapping['field_descriptions'][table_name]['fields'].copy()
        return {}
    
    def search_fields(self, keyword: str, search_chinese: bool = True) -> List[Dict]:
        """
        搜索包含关键词的字段
        
        Parameters:
        -----------
        keyword : str
            搜索关键词
        search_chinese : bool
            是否搜索中文名称
            
        Returns:
        --------
        list
            匹配的字段信息列表
        """
        results = []
        
        # 搜索各表字段
        for table_name, table_info in self.field_mapping['field_descriptions'].items():
            for field_name, chinese_name in table_info['fields'].items():
                match = False
                if keyword.upper() in field_name.upper():
                    match = True
                if search_chinese and keyword in chinese_name:
                    match = True
                
                if match:
                    results.append({
                        'field_name': field_name,
                        'chinese_name': chinese_name,
                        'table': table_name,
                        'table_chinese': table_info['name']
                    })
        
        return results
    
    def get_field_info(self, field_name: str) -> Optional[Dict]:
        """
        获取字段的完整信息
        
        Parameters:
        -----------
        field_name : str
            字段名
            
        Returns:
        --------
        dict or None
            字段信息
        """
        chinese_name = self.get_chinese_name(field_name)
        table_name = self.get_table_name(field_name)
        
        if chinese_name:
            return {
                'field_name': field_name,
                'chinese_name': chinese_name,
                'table': table_name,
                'table_chinese': self._get_table_chinese_name(table_name)
            }
        
        return None
    
    def _get_table_chinese_name(self, table_name: str) -> Optional[str]:
        """获取表的中文名称"""
        if table_name in self.field_mapping['field_descriptions']:
            return self.field_mapping['field_descriptions'][table_name]['name']
        return None
    
    def print_field_summary(self):
        """打印字段映射概要"""
        print("=" * 60)
        print("字段映射概要")
        print("=" * 60)
        
        total_fields = 0
        for table_name, table_info in self.field_mapping['field_descriptions'].items():
            field_count = len(table_info['fields'])
            total_fields += field_count
            print(f"{table_info['name']}: {field_count} 个字段")
        
        print(f"常用字段映射: {len(self.field_mapping['common_fields'])} 个")
        print(f"表归属映射: {len(self.field_mapping['table_attribution'])} 个")
        print(f"总字段数: {total_fields}")
        print("=" * 60)
    
    def export_field_list(self, output_path: Union[str, Path], 
                         format: str = 'excel') -> bool:
        """
        导出字段列表
        
        Parameters:
        -----------
        output_path : str or Path
            输出路径
        format : str
            导出格式 ('excel', 'csv', 'json')
            
        Returns:
        --------
        bool
            是否成功
        """
        try:
            import pandas as pd
            
            # 收集所有字段信息
            all_fields = []
            for table_name, table_info in self.field_mapping['field_descriptions'].items():
                for field_name, chinese_name in table_info['fields'].items():
                    all_fields.append({
                        '英文字段名': field_name,
                        '中文字段名': chinese_name,
                        '所属表': table_info['name'],
                        '表英文名': table_name
                    })
            
            df = pd.DataFrame(all_fields)
            
            output_path = Path(output_path)
            if format == 'excel':
                df.to_excel(output_path, index=False)
            elif format == 'csv':
                df.to_csv(output_path, index=False, encoding='utf-8-sig')
            elif format == 'json':
                df.to_json(output_path, orient='records', force_ascii=False, indent=2)
            else:
                raise ValueError(f"不支持的导出格式: {format}")
            
            print(f"字段列表已导出到: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"导出字段列表失败: {e}")
            return False


# 全局字段映射器实例
_field_mapper = None

def get_field_mapper() -> FieldMapper:
    """获取全局字段映射器实例"""
    global _field_mapper
    if _field_mapper is None:
        _field_mapper = FieldMapper()
    return _field_mapper


# 便捷函数
def get_chinese_name(field_name: str) -> Optional[str]:
    """获取字段中文名称"""
    return get_field_mapper().get_chinese_name(field_name)

def get_field_info(field_name: str) -> Optional[Dict]:
    """获取字段完整信息"""
    return get_field_mapper().get_field_info(field_name)

def search_fields(keyword: str) -> List[Dict]:
    """搜索字段"""
    return get_field_mapper().search_fields(keyword)


if __name__ == '__main__':
    # 测试字段映射器
    mapper = FieldMapper()
    mapper.print_field_summary()
    
    # 测试查找功能
    print("\n测试字段查找:")
    test_fields = ['DEDUCTEDPROFIT', 'TOT_OPER_REV', 'NETCASH_OPER']
    for field in test_fields:
        info = mapper.get_field_info(field)
        if info:
            print(f"{field} -> {info['chinese_name']} ({info['table_chinese']})")
        else:
            print(f"{field} -> 未找到")
    
    # 测试搜索功能
    print("\n搜索包含'利润'的字段:")
    results = mapper.search_fields('利润')
    for result in results[:3]:  # 只显示前3个结果
        print(f"  {result['field_name']} -> {result['chinese_name']}")