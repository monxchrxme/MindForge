from typing import Dict, Any, List
from agents.tools.base import BaseTool
from services.gigachat_client import GigaChatClient
import logging

logger = logging.getLogger(__name__)


class DistractorGeneratorTool(BaseTool):
    """
    Генерирует правдоподобные неправильные варианты ответа.
    Агент вызывает, когда чувствует, что его дистракторы слишком слабые.
    """

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
- Дистракторы не связаны с темой вопроса

КОГДА НЕ ИСПОЛЬЗОВАТЬ:
- Твои дистракторы уже правдоподобны
- Вопрос типа true_false

ВАЖНО: Это опциональный инструмент.
""".strip()

    def validate_args(self, **kwargs) -> bool:
        required = ["question", "correct_answer", "concept_definition"]
        for field in required:
            if not kwargs.get(field):
                logger.error(f"Missing required argument: {field}")
                return False
        return True

    def execute(self, **kwargs) -> Dict[str, Any]:
        """Генерирует правдоподобные дистракторы"""
        logger.info(f"[TOOL] {self.name} called")

        if not self.validate_args(**kwargs):
            return {
                "success": False,
                "error": "Invalid arguments",
                "distractors": []
            }

        question = kwargs["question"]
        correct = kwargs["correct_answer"]
        concept_def = kwargs["concept_definition"]
        num_needed = kwargs.get("num_needed", 3)

        prompt = f"""
Сгенерируй {num_needed} ПРАВДОПОДОБНЫХ неправильных вариантов ответа.

КОНТЕКСТ:
Вопрос: {question}
Правильный ответ: {correct}
Определение: {concept_def}

ТРЕБОВАНИЯ:
- Правдоподобные (похожи на правильный ответ)
- Отражают типичные ошибки студентов
- НЕ абсурдны ("banana", "123")

ФОРМАТ (JSON):
{{
    "distractors": ["Вариант 1", "Вариант 2", "Вариант 3"]
}}

НЕ используй markdown, только JSON.
""".strip()

        try:
            response = self.client.generate_json(prompt)
            distractors = response.get("distractors", [])

            logger.info(f"[TOOL] Generated {len(distractors)} distractors")

            return {
                "success": True,
                "distractors": distractors[:num_needed]
            }

        except Exception as e:
            logger.error(f"[TOOL] Error: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "distractors": []
            }
