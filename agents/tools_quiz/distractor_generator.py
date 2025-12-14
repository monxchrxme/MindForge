from typing import Dict, Any
from agents.tools.base import BaseTool
from services.gigachat_client import GigaChatClient
import logging


logger = logging.getLogger(__name__)


class DistractorGeneratorTool(BaseTool):
    """Генерирует правдоподобные неправильные варианты ответа"""

    def __init__(self, client: GigaChatClient):
        self.client = client

    @property
    def name(self) -> str:
        return "generate_plausible_distractor"

    @property
    def description(self) -> str:
        return """
Генерирует правдоподобные неправильные варианты ответа.

КОГДА ИСПОЛЬЗОВАТЬ:
- Твои дистракторы слишком очевидны ("banana", "error", "123")
- Дистракторы не связаны с темой

ПРИМЕР ВЫЗОВА:
{
  "tool_calls": [{
    "tool": "generate_plausible_distractor",
    "args": {
      "question": "Что делает метод __init__?",
      "correct_answer": "Инициализирует объект",
      "concept_definition": "Конструктор класса...",
      "num_needed": 3
    }
  }]
}
""".strip()

    def execute(self, **kwargs) -> Dict[str, Any]:
        """Генерирует дистракторы"""
        question = kwargs.get("question", "")
        correct = kwargs.get("correct_answer", "")
        concept_def = kwargs.get("concept_definition", "")
        num = kwargs.get("num_needed", 3)

        if not all([question, correct, concept_def]):
            return {"success": False, "error": "Missing arguments"}

        prompt = f"""
Сгенерируй {num} ПРАВДОПОДОБНЫХ неправильных вариантов ответа.

Вопрос: {question}
Правильный ответ: {correct}
Концепт: {concept_def}

ТРЕБОВАНИЯ:
- Похожи на правильный ответ по формату
- Отражают типичные ошибки студентов
- НЕ абсурдны

ФОРМАТ (только JSON):
{{
    "distractors": ["Вариант 1", "Вариант 2", "Вариант 3"]
}}
""".strip()

        try:
            response = self.client.generate_json(prompt)
            distractors = response.get("distractors", [])

            return {
                "success": True,
                "distractors": distractors[:num]
            }
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return {"success": False, "error": str(e)}
