#!/usr/bin/env python3
"""
Agent助手 - 实际的SubAgent调用工具

这个工具提供了一个简单的接口来调用不同的专业Agent。
注意: 这个工具需要在Claude Code环境中运行，因为它依赖Task工具。

Author: AI Assistant
Date: 2025-08-26
"""

import yaml
import os
from typing import Dict, Any

class AgentHelper:
    """Agent调用助手"""
    
    def __init__(self):
        """初始化Agent助手"""
        config_path = os.path.join(os.path.dirname(__file__), 'subagent_config.yaml')
        self.agents = self._load_agent_configs(config_path)
    
    def _load_agent_configs(self, config_path: str) -> Dict[str, Any]:
        """加载agent配置"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config.get('agents', {})
        except Exception as e:
            print(f"警告: 无法加载配置文件 {config_path}: {e}")
            return {}
    
    def get_factor_expert(self, task: str = ""):
        """调用因子专家"""
        agent_config = self.agents.get('factor_expert', {})
        prompt = agent_config.get('prompt', '你是一个因子专家。')
        
        full_prompt = f"""
{prompt}

当前任务: {task}

请基于你的专业知识提供详细的分析和建议。
"""
        
        print(f"🔬 正在调用因子专家...")
        print(f"📋 任务: {task}")
        print("⏳ 请等待响应...")
        
        # 这里返回prompt，实际使用时需要在Claude环境中手动调用Task工具
        return {
            'agent_type': 'factor_expert',
            'task_prompt': full_prompt,
            'instruction': "请使用Task工具调用general-purpose agent，并使用上述prompt"
        }
    
    def get_system_architect(self, task: str = ""):
        """调用系统架构师"""
        agent_config = self.agents.get('system_architect', {})
        prompt = agent_config.get('prompt', '你是一个系统架构师。')
        
        full_prompt = f"""
{prompt}

当前任务: {task}

请基于你的专业知识提供详细的分析和建议。
"""
        
        print(f"🏗️ 正在调用系统架构师...")
        print(f"📋 任务: {task}")
        print("⏳ 请等待响应...")
        
        return {
            'agent_type': 'system_architect', 
            'task_prompt': full_prompt,
            'instruction': "请使用Task工具调用general-purpose agent，并使用上述task_prompt"
        }
    
    def get_portfolio_optimizer(self, task: str = ""):
        """调用投资组合优化专家"""
        agent_config = self.agents.get('portfolio_optimizer', {})
        prompt = agent_config.get('prompt', '你是一个投资组合优化专家。')
        
        full_prompt = f"""
{prompt}

当前任务: {task}

请基于你的专业知识提供详细的分析和建议。
"""
        
        print(f"📊 正在调用投资组合优化专家...")
        print(f"📋 任务: {task}")
        print("⏳ 请等待响应...")
        
        return {
            'agent_type': 'portfolio_optimizer',
            'task_prompt': full_prompt, 
            'instruction': "请使用Task工具调用general-purpose agent，并使用上述task_prompt"
        }
    
    def get_factor_developer(self, task: str = ""):
        """调用因子开发工程师"""
        agent_config = self.agents.get('factor_developer', {})
        prompt = agent_config.get('prompt', '你是一个因子开发工程师。')
        
        full_prompt = f"""
{prompt}

当前任务: {task}

请基于你的专业知识提供详细的分析和建议。
"""
        
        print(f"⚙️ 正在调用因子开发工程师...")
        print(f"📋 任务: {task}")
        print("⏳ 请等待响应...")
        
        return {
            'agent_type': 'factor_developer',
            'task_prompt': full_prompt,
            'instruction': "请使用Task工具调用general-purpose agent，并使用上述task_prompt"
        }
    
    def get_research_analyst(self, task: str = ""):
        """调用因子研究分析师"""
        agent_config = self.agents.get('research_analyst', {})
        prompt = agent_config.get('prompt', '你是一个因子研究分析师。')
        
        full_prompt = f"""
{prompt}

当前任务: {task}

请基于你的专业知识提供详细的分析和建议。
"""
        
        print(f"📈 正在调用因子研究分析师...")
        print(f"📋 任务: {task}")
        print("⏳ 请等待响应...")
        
        return {
            'agent_type': 'research_analyst',
            'task_prompt': full_prompt,
            'instruction': "请使用Task工具调用general-purpose agent，并使用上述task_prompt"
        }
    
    def get_ml_specialist(self, task: str = ""):
        """调用机器学习因子挖掘专家"""
        agent_config = self.agents.get('ml_specialist', {})
        prompt = agent_config.get('prompt', '你是一个机器学习因子挖掘专家。')
        
        full_prompt = f"""
{prompt}

当前任务: {task}

请基于你的专业知识提供详细的分析和建议。
"""
        
        print(f"🤖 正在调用机器学习专家...")
        print(f"📋 任务: {task}")
        print("⏳ 请等待响应...")
        
        return {
            'agent_type': 'ml_specialist',
            'task_prompt': full_prompt,
            'instruction': "请使用Task工具调用general-purpose agent，并使用上述task_prompt"
        }
    
    def list_agents(self):
        """列出所有可用的agents"""
        print("🤖 可用的专业Agent:")
        print("=" * 30)
        
        agents = [
            ("factor_expert", "🔬 因子专家", "因子设计、测试、分析"),
            ("system_architect", "🏗️ 系统架构师", "架构设计、技术选型"),
            ("portfolio_optimizer", "📊 投资组合优化专家", "组合优化、风险模型"),
            ("factor_developer", "⚙️ 因子开发工程师", "因子实现、代码优化"),
            ("research_analyst", "📈 因子研究分析师", "深度研究、报告分析"),
            ("ml_specialist", "🤖 机器学习专家", "AI因子挖掘、预测模型")
        ]
        
        for agent_id, name, desc in agents:
            print(f"{name}")
            print(f"   描述: {desc}")
            print(f"   调用: helper.get_{agent_id}('你的任务描述')")
            print()
    
    def help(self):
        """显示帮助信息"""
        print("📚 Agent助手使用指南")
        print("=" * 30)
        print()
        print("1. 创建助手实例:")
        print("   from agent_helper import AgentHelper")
        print("   helper = AgentHelper()")
        print()
        print("2. 调用专业Agent:")
        print("   result = helper.get_factor_expert('分析ROE因子有效性')")
        print("   # 会返回包含prompt的字典，然后使用Task工具调用")
        print()
        print("3. 查看所有可用Agent:")
        print("   helper.list_agents()")
        print()
        self.list_agents()


# 便捷实例
helper = AgentHelper()

if __name__ == "__main__":
    helper.help()