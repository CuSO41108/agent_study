from __future__ import annotations

from agent_app.agent.definition import AgentDefinition


def render_system_prompt(agent: AgentDefinition) -> str:
    rules_block = "\n".join(f"- {rule}" for rule in agent.rules)
    return (
        f"{agent.system_prompt_template}\n\n"
        f"Goal:\n{agent.goal}\n\n"
        f"Rules:\n{rules_block}\n"
    )

