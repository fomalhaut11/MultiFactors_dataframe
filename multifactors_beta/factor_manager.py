#!/usr/bin/env python3
"""
因子管理工具
用于注册、查询、更新因子元数据
"""

import argparse
import sys
import os
import json
from datetime import datetime
from typing import Dict, List, Optional

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from factors.meta import (
    get_factor_registry,
    FactorType,
    NeutralizationCategory
)
from config import get_config

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


class FactorManager:
    """因子管理器CLI接口"""
    
    def __init__(self):
        self.registry = get_factor_registry()
    
    def register_factor(self, args):
        """注册新因子"""
        try:
            factor_type = FactorType(args.type)
            neutralization_cat = NeutralizationCategory(args.neutralization) if args.neutralization else None
            
            # 构建其他参数
            kwargs = {
                'formula': args.formula,
                'generator': args.generator,
                'tags': args.tags.split(',') if args.tags else [],
                'category': args.category,
                'priority': args.priority,
            }
            
            if neutralization_cat:
                kwargs['neutralization_category'] = neutralization_cat
            
            # 移除None值
            kwargs = {k: v for k, v in kwargs.items() if v is not None}
            
            metadata = self.registry.register_factor(
                name=args.name,
                factor_type=factor_type,
                description=args.description,
                **kwargs
            )
            
            print(f"✅ 成功注册因子: {args.name}")
            print(f"   类型: {factor_type.value}")
            print(f"   描述: {args.description}")
            
        except Exception as e:
            print(f"❌ 注册因子失败: {e}")
            return False
        
        return True
    
    def list_factors(self, args):
        """列出因子"""
        try:
            factor_type = FactorType(args.type) if args.type else None
            has_orthogonal = args.orthogonal if args.orthogonal != 'any' else None
            
            factors = self.registry.list_factors(
                factor_type=factor_type,
                active_only=args.active_only,
                has_orthogonal=has_orthogonal
            )
            
            if not factors:
                print("没有找到符合条件的因子")
                return
            
            print(f"找到 {len(factors)} 个因子:")
            print("-" * 80)
            
            for factor in factors:
                status = "✅" if factor.is_active else "❌"
                orth_status = "🔀" if factor.is_orthogonalized else "🔸"
                
                print(f"{status} {orth_status} {factor.name}")
                print(f"    类型: {factor.type.value if factor.type else 'N/A'}")
                print(f"    描述: {factor.description}")
                print(f"    中性化: {factor.neutralization_category.value if factor.neutralization_category else 'N/A'}")
                if factor.tags:
                    print(f"    标签: {', '.join(factor.tags)}")
                if factor.created_date:
                    print(f"    创建: {factor.created_date}")
                print()
            
        except Exception as e:
            print(f"❌ 列出因子失败: {e}")
    
    def show_factor(self, args):
        """显示因子详情"""
        metadata = self.registry.get_factor(args.name)
        
        if not metadata:
            print(f"❌ 因子不存在: {args.name}")
            return
        
        print(f"因子详情: {args.name}")
        print("=" * 60)
        
        # 基本信息
        print(f"名称: {metadata.name}")
        print(f"类型: {metadata.type.value if metadata.type else 'N/A'}")
        print(f"描述: {metadata.description}")
        print(f"公式: {metadata.formula or 'N/A'}")
        print()
        
        # 状态信息
        status = "激活" if metadata.is_active else "停用"
        orth_status = "已正交化" if metadata.is_orthogonalized else "未正交化"
        print(f"状态: {status}")
        print(f"正交化: {orth_status}")
        print(f"中性化类别: {metadata.neutralization_category.value if metadata.neutralization_category else 'N/A'}")
        print()
        
        # 时间信息
        if metadata.created_date:
            print(f"创建时间: {metadata.created_date}")
        if metadata.updated_date:
            print(f"更新时间: {metadata.updated_date}")
        if metadata.orthogonalization_date:
            print(f"正交化时间: {metadata.orthogonalization_date}")
        print()
        
        # 文件路径
        if metadata.raw_version:
            print(f"原始版本: {metadata.raw_version}")
        if metadata.orthogonal_version:
            print(f"正交化版本: {metadata.orthogonal_version}")
        print()
        
        # 正交化信息
        if metadata.control_factors:
            print(f"控制因子: {', '.join(metadata.control_factors)}")
        if metadata.orthogonalization_method:
            print(f"正交化方法: {metadata.orthogonalization_method}")
        print()
        
        # 其他信息
        if metadata.tags:
            print(f"标签: {', '.join(metadata.tags)}")
        if metadata.category:
            print(f"分类: {metadata.category}")
        if metadata.priority:
            print(f"优先级: {metadata.priority}")
        if metadata.quality_score:
            print(f"质量评分: {metadata.quality_score}")
        
        # 性能指标
        if metadata.performance_metrics:
            print(f"\\n性能指标:")
            for key, value in metadata.performance_metrics.items():
                print(f"  {key}: {value}")
    
    def update_factor(self, args):
        """更新因子"""
        updates = {}
        
        if args.description:
            updates['description'] = args.description
        if args.formula:
            updates['formula'] = args.formula
        if args.tags:
            updates['tags'] = args.tags.split(',')
        if args.category:
            updates['category'] = args.category
        if args.priority is not None:
            updates['priority'] = args.priority
        if args.active is not None:
            updates['is_active'] = args.active
        
        if not updates:
            print("❌ 没有要更新的字段")
            return
        
        result = self.registry.update_factor(args.name, **updates)
        
        if result:
            print(f"✅ 成功更新因子: {args.name}")
            for key, value in updates.items():
                print(f"   {key}: {value}")
        else:
            print(f"❌ 更新因子失败: {args.name}")
    
    def statistics(self, args):
        """显示统计信息"""
        stats = self.registry.get_factor_statistics()
        
        print("因子注册表统计信息")
        print("=" * 50)
        print(f"总因子数: {stats['total_factors']}")
        print(f"激活因子数: {stats['active_factors']}")
        print(f"已正交化因子数: {stats['orthogonalized_factors']}")
        print(f"正交化比例: {stats['orthogonalization_rate']:.2%}")
        print()
        
        print("按类型分布:")
        for factor_type, count in stats['factor_types'].items():
            print(f"  {factor_type}: {count}")
        print()
        
        print("按中性化类别分布:")
        for category, count in stats['neutralization_categories'].items():
            print(f"  {category}: {count}")
        print()
        
        print(f"注册表路径: {stats['registry_path']}")
        print(f"最后更新: {stats['last_updated']}")
    
    def neutralization_candidates(self, args):
        """显示需要中性化的因子"""
        candidates = self.registry.get_neutralization_candidates()
        
        if not candidates:
            print("没有需要中性化的因子")
            return
        
        print(f"需要中性化的因子 ({len(candidates)} 个):")
        print("-" * 60)
        
        for factor in candidates:
            priority = "🔴" if factor.neutralization_category == NeutralizationCategory.MUST_NEUTRALIZE else "🟡"
            print(f"{priority} {factor.name}")
            print(f"    类型: {factor.type.value if factor.type else 'N/A'}")
            print(f"    类别: {factor.neutralization_category.value}")
            print(f"    描述: {factor.description}")
            print()
    
    def export(self, args):
        """导出注册表"""
        try:
            self.registry.export_to_csv(args.output)
            print(f"✅ 成功导出注册表到: {args.output}")
        except Exception as e:
            print(f"❌ 导出失败: {e}")
    
    def import_csv(self, args):
        """从CSV导入"""
        try:
            self.registry.import_from_csv(args.input)
            print(f"✅ 成功从CSV导入注册表: {args.input}")
        except Exception as e:
            print(f"❌ 导入失败: {e}")
    
    def validate_factors(self, args):
        """验证因子完整性"""
        print("🔍 开始验证因子完整性...")
        
        factors = self.registry.list_factors(active_only=args.active_only)
        
        issues_found = 0
        valid_factors = 0
        
        for factor in factors:
            print(f"\n检查因子: {factor.name}")
            
            # 检查必需字段
            issues = []
            
            if not factor.description or len(factor.description.strip()) < 10:
                issues.append("描述过短或缺失")
            
            if not factor.type:
                issues.append("缺少因子类型")
                
            # 检查文件存在性
            if factor.raw_version and not os.path.exists(factor.raw_version):
                issues.append(f"原始版本文件不存在: {factor.raw_version}")
                
            if factor.orthogonal_version and not os.path.exists(factor.orthogonal_version):
                issues.append(f"正交化版本文件不存在: {factor.orthogonal_version}")
            
            # 检查正交化逻辑
            if factor.is_orthogonalized and not factor.control_factors:
                issues.append("已标记为正交化但缺少控制因子信息")
            
            if issues:
                print(f"  ❌ 发现问题:")
                for issue in issues:
                    print(f"    - {issue}")
                issues_found += len(issues)
            else:
                print(f"  ✅ 验证通过")
                valid_factors += 1
        
        print(f"\n验证完成:")
        print(f"  总因子数: {len(factors)}")
        print(f"  有效因子: {valid_factors}")
        print(f"  发现问题: {issues_found}")
        
        if issues_found > 0:
            print(f"  建议运行: python factor_manager.py cleanup 清理问题")
    
    def cleanup(self, args):
        """清理无效因子"""
        print("🧹 开始清理无效因子...")
        
        factors = self.registry.list_factors(active_only=False)
        cleaned_count = 0
        
        for factor in factors:
            should_clean = False
            reasons = []
            
            # 检查文件存在性
            if factor.raw_version and not os.path.exists(factor.raw_version):
                should_clean = True
                reasons.append(f"原始文件不存在: {factor.raw_version}")
            
            if factor.orthogonal_version and not os.path.exists(factor.orthogonal_version):
                should_clean = True
                reasons.append(f"正交化文件不存在: {factor.orthogonal_version}")
            
            # 检查是否长期未更新且无效
            if not factor.is_active and not factor.updated_date:
                should_clean = True
                reasons.append("长期未激活且无更新记录")
            
            if should_clean and not args.dry_run:
                try:
                    self.registry.deactivate_factor(factor.name)
                    print(f"🗑️  清理因子: {factor.name}")
                    for reason in reasons:
                        print(f"    理由: {reason}")
                    cleaned_count += 1
                except Exception as e:
                    print(f"❌ 清理失败 {factor.name}: {e}")
            elif should_clean and args.dry_run:
                print(f"[DRY RUN] 将清理因子: {factor.name}")
                for reason in reasons:
                    print(f"    理由: {reason}")
                cleaned_count += 1
        
        if args.dry_run:
            print(f"\n[DRY RUN] 将清理 {cleaned_count} 个因子")
            print("使用 --execute 参数执行实际清理")
        else:
            print(f"\n清理完成，共处理 {cleaned_count} 个因子")
    
    def backup(self, args):
        """备份因子注册表"""
        try:
            backup_path = args.output or f"factor_registry_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            # 获取所有因子数据
            factors = self.registry.list_factors(active_only=False)
            
            backup_data = {
                "backup_date": datetime.now().isoformat(),
                "total_factors": len(factors),
                "factors": []
            }
            
            for factor in factors:
                factor_data = {
                    "name": factor.name,
                    "type": factor.type.value if factor.type else None,
                    "description": factor.description,
                    "formula": factor.formula,
                    "is_active": factor.is_active,
                    "is_orthogonalized": factor.is_orthogonalized,
                    "neutralization_category": factor.neutralization_category.value if factor.neutralization_category else None,
                    "tags": factor.tags,
                    "category": factor.category,
                    "priority": factor.priority,
                    "created_date": factor.created_date.isoformat() if factor.created_date else None,
                    "updated_date": factor.updated_date.isoformat() if factor.updated_date else None,
                    "raw_version": factor.raw_version,
                    "orthogonal_version": factor.orthogonal_version,
                    "control_factors": factor.control_factors,
                    "orthogonalization_method": factor.orthogonalization_method,
                    "performance_metrics": factor.performance_metrics,
                    "quality_score": factor.quality_score
                }
                backup_data["factors"].append(factor_data)
            
            with open(backup_path, 'w', encoding='utf-8') as f:
                json.dump(backup_data, f, indent=2, ensure_ascii=False)
            
            print(f"✅ 备份完成: {backup_path}")
            print(f"   包含 {len(factors)} 个因子")
            
        except Exception as e:
            print(f"❌ 备份失败: {e}")
    
    def restore(self, args):
        """恢复因子注册表"""
        try:
            if not os.path.exists(args.input):
                print(f"❌ 备份文件不存在: {args.input}")
                return
            
            with open(args.input, 'r', encoding='utf-8') as f:
                backup_data = json.load(f)
            
            factors_data = backup_data.get("factors", [])
            
            print(f"准备恢复 {len(factors_data)} 个因子...")
            
            if not args.force:
                confirm = input("这将覆盖现有因子数据，确认继续? (y/N): ")
                if confirm.lower() != 'y':
                    print("操作取消")
                    return
            
            restored_count = 0
            failed_count = 0
            
            for factor_data in factors_data:
                try:
                    # 这里需要根据实际的registry API来实现恢复逻辑
                    # 暂时使用简单的重新注册方式
                    factor_type = FactorType(factor_data["type"]) if factor_data["type"] else None
                    
                    if factor_type:
                        self.registry.register_factor(
                            name=factor_data["name"],
                            factor_type=factor_type,
                            description=factor_data["description"],
                            formula=factor_data.get("formula"),
                            tags=factor_data.get("tags", []),
                            category=factor_data.get("category"),
                            priority=factor_data.get("priority", 0)
                        )
                        restored_count += 1
                    
                except Exception as e:
                    print(f"❌ 恢复因子失败 {factor_data['name']}: {e}")
                    failed_count += 1
            
            print(f"恢复完成:")
            print(f"  成功: {restored_count}")
            print(f"  失败: {failed_count}")
            
        except Exception as e:
            print(f"❌ 恢复失败: {e}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="因子管理工具")
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 注册因子
    register_parser = subparsers.add_parser('register', help='注册新因子')
    register_parser.add_argument('name', help='因子名称')
    register_parser.add_argument('type', choices=[t.value for t in FactorType], help='因子类型')
    register_parser.add_argument('description', help='因子描述')
    register_parser.add_argument('--formula', help='计算公式')
    register_parser.add_argument('--neutralization', choices=[c.value for c in NeutralizationCategory], help='中性化类别')
    register_parser.add_argument('--generator', help='生成器名称')
    register_parser.add_argument('--tags', help='标签(逗号分隔)')
    register_parser.add_argument('--category', help='分类')
    register_parser.add_argument('--priority', type=int, default=0, help='优先级')
    
    # 列出因子
    list_parser = subparsers.add_parser('list', help='列出因子')
    list_parser.add_argument('--type', choices=[t.value for t in FactorType], help='按类型筛选')
    list_parser.add_argument('--active-only', action='store_true', default=True, help='只显示激活的因子')
    list_parser.add_argument('--orthogonal', choices=['true', 'false', 'any'], default='any', help='按正交化状态筛选')
    
    # 显示因子详情
    show_parser = subparsers.add_parser('show', help='显示因子详情')
    show_parser.add_argument('name', help='因子名称')
    
    # 更新因子
    update_parser = subparsers.add_parser('update', help='更新因子')
    update_parser.add_argument('name', help='因子名称')
    update_parser.add_argument('--description', help='更新描述')
    update_parser.add_argument('--formula', help='更新公式')
    update_parser.add_argument('--tags', help='更新标签(逗号分隔)')
    update_parser.add_argument('--category', help='更新分类')
    update_parser.add_argument('--priority', type=int, help='更新优先级')
    update_parser.add_argument('--active', type=bool, help='更新激活状态')
    
    # 统计信息
    stats_parser = subparsers.add_parser('stats', help='显示统计信息')
    
    # 中性化候选
    neutralization_parser = subparsers.add_parser('neutralization', help='显示需要中性化的因子')
    
    # 导出
    export_parser = subparsers.add_parser('export', help='导出注册表到CSV')
    export_parser.add_argument('output', help='输出文件路径')
    
    # 导入
    import_parser = subparsers.add_parser('import', help='从CSV导入注册表')
    import_parser.add_argument('input', help='输入文件路径')
    
    # 验证因子
    validate_parser = subparsers.add_parser('validate', help='验证因子完整性')
    validate_parser.add_argument('--active-only', action='store_true', default=True, help='只验证激活的因子')
    
    # 清理因子
    cleanup_parser = subparsers.add_parser('cleanup', help='清理无效因子')
    cleanup_parser.add_argument('--dry-run', action='store_true', help='预览清理操作，不执行实际清理')
    
    # 备份
    backup_parser = subparsers.add_parser('backup', help='备份因子注册表')
    backup_parser.add_argument('--output', help='备份文件路径（可选）')
    
    # 恢复
    restore_parser = subparsers.add_parser('restore', help='恢复因子注册表')
    restore_parser.add_argument('input', help='备份文件路径')
    restore_parser.add_argument('--force', action='store_true', help='强制恢复，不询问确认')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    manager = FactorManager()
    
    # 执行对应命令
    if args.command == 'register':
        manager.register_factor(args)
    elif args.command == 'list':
        manager.list_factors(args)
    elif args.command == 'show':
        manager.show_factor(args)
    elif args.command == 'update':
        manager.update_factor(args)
    elif args.command == 'stats':
        manager.statistics(args)
    elif args.command == 'neutralization':
        manager.neutralization_candidates(args)
    elif args.command == 'export':
        manager.export(args)
    elif args.command == 'import':
        manager.import_csv(args)
    elif args.command == 'validate':
        manager.validate_factors(args)
    elif args.command == 'cleanup':
        manager.cleanup(args)
    elif args.command == 'backup':
        manager.backup(args)
    elif args.command == 'restore':
        manager.restore(args)


if __name__ == "__main__":
    main()