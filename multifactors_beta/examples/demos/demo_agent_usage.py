#!/usr/bin/env python3
"""
演示如何使用SubAgent配置

Author: AI Assistant
Date: 2025-08-26
"""

import yaml
import os

def demo_agent_usage():
    """演示SubAgent的使用方法"""
    print("SubAgent配置演示")
    print("=" * 30)
    
    # 读取配置
    config_path = os.path.join(os.path.dirname(__file__), 'subagent_config.yaml')
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        agents = config.get('agents', {})
        scenarios = config.get('scenarios', {})
        
        print(f"已加载配置，包含 {len(agents)} 个Agent和 {len(scenarios)} 个场景")
        print()
        
        # 显示可用的Agents
        print("可用的专业Agent:")
        print("-" * 20)
        for agent_name, agent_config in agents.items():
            print(f"- {agent_name}: {agent_config['description']}")
        
        print()
        print("可用的协作场景:")
        print("-" * 20)
        for scenario_name, scenario_config in scenarios.items():
            print(f"- {scenario_name}: {scenario_config['description']}")
        
        print()
        print("使用方法:")
        print("-" * 10)
        print("1. 通过Task工具调用:")
        print("   - 复制subagent_config.yaml中对应Agent的prompt")
        print("   - 使用Task工具，subagent_type='general-purpose'")
        print("   - 将prompt和你的具体任务组合")
        
        print()
        print("2. 使用agent_helper.py:")
        print("   from agent_helper import AgentHelper")
        print("   helper = AgentHelper()")
        print("   result = helper.get_factor_expert('你的任务')")
        
        print()
        print("示例 - 调用因子专家:")
        print("-" * 20)
        
        # 显示因子专家的配置示例
        factor_expert = agents.get('factor_expert', {})
        if factor_expert:
            print("Agent: factor_expert")
            print("描述:", factor_expert['description'])
            print("Prompt预览:")
            prompt = factor_expert.get('prompt', '')[:200] + '...'
            print(prompt)
        
        return True
        
    except Exception as e:
        print(f"加载配置失败: {e}")
        return False

if __name__ == "__main__":
    demo_agent_usage()