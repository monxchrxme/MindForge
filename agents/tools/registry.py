from typing import Dict, List, Any
from agents.tools.base import BaseTool
import logging

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Реестр всех доступных инструментов для агента"""

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """Регистрирует новый инструмент"""
        name = tool.name
        if name in self._tools:
            logger.warning(f"Tool '{name}' already registered, overwriting")

        self._tools[name] = tool
        logger.info(f"Registered tool: {name}")

    def execute_tool(self, name: str, **kwargs) -> Dict[str, Any]:
        """Выполняет инструмент по имени"""
        logger.info(f"[REGISTRY] Executing tool: {name}")

        try:
            if name not in self._tools:
                raise ValueError(f"Tool '{name}' not found")

            tool = self._tools[name]
            result = tool.execute(**kwargs)

            logger.info(f"[REGISTRY] Tool '{name}' completed: success={result.get('success', False)}")
            return result

        except Exception as e:
            logger.error(f"[REGISTRY] Tool '{name}' failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": f"Tool execution failed: {str(e)}"
            }

    def get_tools_description_for_llm(self) -> str:
        """Возвращает описание всех инструментов для промпта LLM"""
        if not self._tools:
            return ""

        descriptions = ["\n=== ДОСТУПНЫЕ ИНСТРУМЕНТЫ (опциональные) ===\n"]

        for idx, (name, tool) in enumerate(self._tools.items(), 1):
            descriptions.append(f"{idx}. **{name}**")
            descriptions.append(f"{tool.description}\n")

        descriptions.append("⚠️ ВАЖНО:")
        descriptions.append("- Все инструменты ОПЦИОНАЛЬНЫ — вызывай только при необходимости")
        descriptions.append("- Можешь создать вопрос без использования tools")
        descriptions.append("- Можешь вызвать несколько tools для одного вопроса\n")

        return "\n".join(descriptions)

    def list_tools(self) -> List[str]:
        """Возвращает список имён всех зарегистрированных инструментов"""
        return list(self._tools.keys())
