# agents/factcheck.py

import json
import logging
from typing import List, Dict, Any
from services.gigachat_client import GigaChatClient

logger = logging.getLogger(__name__)


class FactCheckAgent:
    def __init__(self, client: GigaChatClient):
        self.client = client

    def verify_concepts(self, concepts: list) -> list:
        """
        Получает список концептов. Для каждого вызывает LLM на проверку и исправление дефиниции.
        Возвращает новый список (с исправлениями).
        """
        if not concepts:
            return []

        try:
            # Строим промпт для проверки
            prompt = self._build_prompt(concepts)

            # Отправляем запрос к API с автоматическим парсингом JSON
            # ✅ ИСПРАВЛЕНО: Используем generate_json() вместо несуществующего send_request()
            response_data = self.client.generate_json(prompt)
            
            # Извлекаем проверенные концепты
            verified_concepts = response_data.get("concepts", [])
            
            # Базовая валидация структуры
            for concept in verified_concepts:
                if not isinstance(concept, dict) or "term" not in concept or "definition" not in concept:
                    raise ValueError("Неверная структура концепта")
            
            logger.info(f"Successfully verified {len(verified_concepts)} concepts")
            return verified_concepts

        except Exception as e:
            logger.error(f"Ошибка в FactCheckAgent: {e}")
            # При ошибке возвращаем оригинальные концепты
            return concepts

    def _build_prompt(self, concepts: List[Dict[str, str]]) -> str:
        """Строит промпт для проверки концептов."""

        concepts_list = ""
        for i, concept in enumerate(concepts, 1):
            term = concept.get('term', '')
            definition = concept.get('definition', '')
            concepts_list += f"{i}. Термин: {term}\n   Определение: {definition}\n\n"

        prompt = f"""
Проверь следующие образовательные концепты на фактические ошибки и неточности:

{concepts_list}

ИНСТРУКЦИИ:
1. Проверь каждый концепт на соответствие научным знаниям
2. Если найдешь ошибку - исправь определение
3. Если концепт корректен - оставь без изменений
4. Сохрани оригинальную структуру терминов
5. Не добавляй новые концепты

Верни ответ в формате JSON:
{{
    "concepts": [
        {{
            "term": "оригинальный термин",
            "definition": "проверенное определение"
        }}
    ]
}}

Только JSON, без дополнительного текста.
"""
        return prompt
