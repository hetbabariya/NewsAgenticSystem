import logging
import sys
from datetime import datetime

class AgentLogger:
    """CrewAI-style formatted logger for agents and tools."""

    COLORS = {
        "HEADER": "\033[95m",
        "BLUE": "\033[94m",
        "CYAN": "\033[96m",
        "GREEN": "\033[92m",
        "WARNING": "\033[93m",
        "FAIL": "\033[91m",
        "ENDC": "\033[0m",
        "BOLD": "\033[1m",
        "UNDERLINE": "\033[4m",
    }

    @staticmethod
    def _format_message(role: str, message: str, color: str) -> str:
        timestamp = datetime.now().strftime("%H:%M:%S")
        return f"{AgentLogger.COLORS[color]}[{timestamp}][{role.upper()}]{AgentLogger.COLORS['ENDC']} {message}"

    def log_agent_start(self, agent_name: str, task: str):
        print("\n" + "="*80)
        print(self._format_message("SYSTEM", f"Entering Agent: {AgentLogger.COLORS['BOLD']}{agent_name}{AgentLogger.COLORS['ENDC']}", "HEADER"))
        print(self._format_message("TASK", task, "CYAN"))
        print("-" * 80)

    def log_tool_call(self, tool_name: str, args: dict):
        print(self._format_message("TOOL", f"Calling: {AgentLogger.COLORS['BOLD']}{tool_name}{AgentLogger.COLORS['ENDC']} with args: {args}", "BLUE"))

    # Truncate long results for readability
        if isinstance(result, str):
            display_result = (result[:500] + '...') if len(result) > 500 else result
        else:
            display_result = str(result)[:500] + '...' if len(str(result)) > 500 else str(result)
        print(self._format_message("RESULT", f"Tool {tool_name} returned: {display_result}", "GREEN"))

    def log_thought(self, agent_name: str, thought: str):
        print(self._format_message(agent_name, f"Thought: {thought}", "WARNING"))

    def log_final_answer(self, agent_name: str, answer: str):
        print("-" * 80)
        print(self._format_message(agent_name, f"Final Answer: {AgentLogger.COLORS['BOLD']}{answer}{AgentLogger.COLORS['ENDC']}", "GREEN"))
        print("="*80 + "\n")

    def log_error(self, role: str, error: str):
        print(self._format_message(role, f"ERROR: {error}", "FAIL"))

agent_logger = AgentLogger()
