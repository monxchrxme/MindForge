from typing import Dict, Any
from agents.tools.base import BaseTool
import logging

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Реестр инструментов агента"""

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """Регистрирует инструмент"""
        self._tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")

    def execute_tool(self, name: str, **kwargs) -> Dict[str, Any]:
        """Выполняет инструмент по имени"""
        logger.info(f"[TOOL] Executing: {name}")

        try:
            if name not in self._tools:
                raise ValueError(f"Tool '{name}' not found")

            tool = self._tools[name]
            result = tool.execute(**kwargs)

            logger.info(f"[TOOL] {name} completed: success={result.get('success', False)}")
            return result

        except Exception as e:
            logger.error(f"[TOOL] {name} failed: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    def get_tools_description(self) -> str:
        """Возвращает описание tools для промпта"""
        if not self._tools:
            return ""

        lines = ["\n=== ДОСТУПНЫЕ ИНСТРУМЕНТЫ ===\n"]

        for idx, tool in enumerate(self._tools.values(), 1):
            lines.append(f"{idx}. {tool.name}")
            lines.append(f"{tool.description}\n")

        lines.append("⚠️ Инструменты ОПЦИОНАЛЬНЫ - вызывай только при необходимости\n")

        return "\n".join(lines)