#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
创建字段映射配置文件
从Excel文件中读取中英文字段对照，生成配置文件
"""
import pandas as pd
import yaml
from pathlib import Path
import json

def create_field_mapping():
    """创建字段映射配置"""
    
    # 读取Excel文件
    file_path = r'E:\Documents\PythonProject\StockProject\StockData\3张表.xlsx'
    xlsx_file = pd.ExcelFile(file_path)
    
    print('Available sheets:', xlsx_file.sheet_names)
    
    # 使用实际的sheet名称
    sheet_names = xlsx_file.sheet_names
    
    # 读取各个表
    balance_sheet = pd.read_excel(file_path, sheet_name=sheet_names[0])  # 资产负债表
    income_statement = pd.read_excel(file_path, sheet_name=sheet_names[1])  # 利润表
    cash_flow = pd.read_excel(file_path, sheet_name=sheet_names[2])  # 现金流量表
    table_mapping = pd.read_excel(file_path, sheet_name=sheet_names[3])  # 波动指标
    
    # 创建字段映射配置
    field_mapping = {
        'field_descriptions': {
            'balance_sheet': {
                'name': '资产负债表',
                'name_en': 'Balance Sheet',
                'fields': {}
            },
            'income_statement': {
                'name': '利润表', 
                'name_en': 'Income Statement',
                'fields': {}
            },
            'cash_flow': {
                'name': '现金流量表',
                'name_en': 'Cash Flow Statement',
                'fields': {}
            }
        },
        'common_fields': {
            # 常用字段快速查找
            'earnings': {
                'field_name': 'DEDUCTEDPROFIT',
                'chinese_name': '扣除非经常性损益后的净利润',
                'table': 'income_statement'
            },
            'revenue': {
                'field_name': 'TOT_OPER_REV',
                'chinese_name': '营业总收入',
                'table': 'income_statement'
            },
            'operating_cash_flow': {
                'field_name': 'NETCASH_OPER',
                'chinese_name': '经营活动产生的现金流量净额',
                'table': 'cash_flow'
            },
            'financial_expense': {
                'field_name': 'FIN_EXP_IS',
                'chinese_name': '财务费用',
                'table': 'income_statement'
            },
            'equity': {
                'field_name': 'EQY_BELONGTO_PARCOMSH',
                'chinese_name': '归属于母公司股东权益合计',
                'table': 'balance_sheet'
            },
            'cash_equivalents': {
                'field_name': 'CASH',
                'chinese_name': '货币资金',
                'table': 'balance_sheet'
            },
            'short_term_debt': {
                'field_name': 'ST_BORROW',
                'chinese_name': '短期借款',
                'table': 'balance_sheet'
            },
            'quarter': {
                'field_name': 'd_quarter',
                'chinese_name': '季度',
                'table': 'auxiliary'
            }
        }
    }
    
    # 填充字段映射
    for _, row in balance_sheet.iterrows():
        if pd.notna(row['bt']) and pd.notna(row['bt_cn']):
            field_mapping['field_descriptions']['balance_sheet']['fields'][row['bt']] = row['bt_cn']
    
    for _, row in income_statement.iterrows():
        if pd.notna(row['bt']) and pd.notna(row['bt_cn']):
            field_mapping['field_descriptions']['income_statement']['fields'][row['bt']] = row['bt_cn']
    
    for _, row in cash_flow.iterrows():
        if pd.notna(row['bt']) and pd.notna(row['bt_cn']):
            field_mapping['field_descriptions']['cash_flow']['fields'][row['bt']] = row['bt_cn']
    
    # 添加表归属信息
    table_attribution = {}
    for _, row in table_mapping.iterrows():
        if pd.notna(row['bt']) and pd.notna(row['table']):
            table_attribution[row['bt']] = row['table']
    
    field_mapping['table_attribution'] = table_attribution
    
    # 创建配置目录
    config_path = Path('./factors/config/field_mapping.yaml')
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 保存为YAML配置文件（使用safe_dump避免参数问题）
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(field_mapping, f, indent=2, default_flow_style=False, 
                      allow_unicode=True)
    
    print(f'字段映射配置已保存到: {config_path}')
    print('各表字段数量:')
    for table_name, table_info in field_mapping['field_descriptions'].items():
        print(f'  {table_info["name"]}: {len(table_info["fields"])} 个字段')
    print(f'表归属映射数: {len(field_mapping["table_attribution"])}')
    print(f'常用字段映射数: {len(field_mapping["common_fields"])}')
    
    # 也保存一个JSON版本便于程序读取
    json_path = config_path.with_suffix('.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(field_mapping, f, ensure_ascii=False, indent=2)
    
    print(f'JSON版本已保存到: {json_path}')
    
    return field_mapping

if __name__ == '__main__':
    mapping = create_field_mapping()