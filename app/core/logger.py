import logging
import sys
from datetime import datetime
from typing import Any
from rich.console import Console, Group
from rich.panel import Panel
from rich.text import Text as RichText
from rich.table import Table

class AgentLogger:
    """CrewAI-style formatted logger for agents and tools using Rich for beautiful output."""

    def __init__(self):
        self.console = Console()

    COLORS = {
        "HEADER": "magenta",
        "BLUE": "blue",
        "CYAN": "cyan",
        "GREEN": "green",
        "WARNING": "yellow",
        "FAIL": "red",
        "BOLD": "bold",
    }

    def _get_timestamp(self) -> str:
        return datetime.now().strftime("%H:%M:%S")

    def log_agent_start(self, agent_name: str, task: str):
        self.console.print("\n" + "="*80, style="bold grey50")
        timestamp = self._get_timestamp()
        header = RichText()
        header.append(f"[{timestamp}]", style="grey50")
        header.append("[SYSTEM]", style="bold magenta")
        header.append(f" Entering Agent: ", style="default")
        header.append(agent_name, style="bold magenta")
        self.console.print(header)

        task_text = RichText()
        task_text.append(f"[{timestamp}]", style="grey50")
        task_text.append("[TASK]", style="bold cyan")
        task_text.append(f" {task}", style="default")
        self.console.print(task_text)
        self.console.print("-" * 80, style="grey50")

    def log_tool_call(self, tool_name: str, args: dict):
        timestamp = self._get_timestamp()
        msg = RichText()
        msg.append(f"[{timestamp}]", style="grey50")
        msg.append("[TOOL]", style="bold blue")
        msg.append(f" Calling: ", style="default")
        msg.append(tool_name, style="bold blue")
        msg.append(f" with args: {args}", style="default")
        self.console.print(msg)

    def log_tool_result(self, tool_name: str, result: Any):
        timestamp = self._get_timestamp()
        if isinstance(result, str):
            display_result = (result[:500] + '...') if len(result) > 500 else result
        else:
            display_result = str(result)[:500] + '...' if len(str(result)) > 500 else str(result)

        msg = RichText()
        msg.append(f"[{timestamp}]", style="grey50")
        msg.append("[RESULT]", style="bold green")
        msg.append(f" Tool {tool_name} returned: {display_result}", style="default")
        self.console.print(msg)

    def log_thought(self, agent_name: str, thought: str):
        timestamp = self._get_timestamp()
        msg = RichText()
        msg.append(f"[{timestamp}]", style="grey50")
        msg.append(f"[{agent_name.upper()}]", style="bold yellow")
        msg.append(f" Thought: {thought}", style="default")
        self.console.print(msg)

    def log_final_answer(self, agent_name: str, answer: str):
        self.console.print("-" * 80, style="grey50")
        timestamp = self._get_timestamp()
        msg = RichText()
        msg.append(f"[{timestamp}]", style="grey50")
        msg.append(f"[{agent_name.upper()}]", style="bold green")
        msg.append(f" Final Answer: ", style="default")
        msg.append(answer, style="bold green")
        self.console.print(msg)
        self.console.print("="*80 + "\n", style="bold grey50")

    def log_error(self, role: str, error: str):
        timestamp = self._get_timestamp()
        msg = RichText()
        msg.append(f"[{timestamp}]", style="grey50")
        msg.append(f"[{role.upper()}]", style="bold red")
        msg.append(f" ERROR: {error}", style="default")
        self.console.print(msg)

    def log_llm_io(self, agent_name: str, prompt: Any, response: Any):
        """Log the raw LLM input (prompt) and output (response) in a beautiful Rich panel."""

        # Input Table
        input_table = Table(show_header=True, header_style="bold blue", box=None, padding=(0, 1))
        input_table.add_column("Role", width=10)
        input_table.add_column("Content")

        if isinstance(prompt, list):
            for msg in prompt:
                role = getattr(msg, "type", "message").upper()
                content = str(getattr(msg, "content", msg))
                if len(content) > 1000:
                    content = content[:1000] + "\n... (truncated)"
                input_table.add_row(role, content)
        else:
            input_table.add_row("PROMPT", str(prompt)[:1000])

        # Output Section
        output_text = RichText()
        content = str(getattr(response, "content", response))
        output_text.append(content)

        if hasattr(response, "tool_calls") and response.tool_calls:
            output_text.append("\n\n")
            output_text.append("TOOL CALLS:", style="bold cyan")
            for tc in response.tool_calls:
                output_text.append(f"\n- {tc['name']}({tc['args']})")

        # Create combined panel
        panel = Panel(
            Group(
                RichText("INPUT (PROMPT)", style="bold blue"),
                input_table,
                RichText("\nOUTPUT (RESPONSE)", style="bold green"),
                output_text
            ),
            title=f"[bold]LLM INTERACTION - {agent_name}[/bold]",
            border_style="bright_blue",
            padding=(1, 2),
            expand=False
        )
        self.console.print(panel)

agent_logger = AgentLogger()
