#TODO Добавить factcheck для кода
import logging
from typing import List, Dict
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
            prompt = self._build_prompt(concepts)
            response_data = self.client.generate_json(prompt)
            verified_raw = response_data.get("concepts", [])

            # MERGE: Объединяем проверенные определения с оригинальным кодом
            verified_concepts = []

            # Создаем словарь для быстрого поиска оригиналов по термину
            original_map = {c.get("term"): c for c in concepts}

            for v_concept in verified_raw:
                term = v_concept.get("term")
                original = original_map.get(term)

                if original:
                    # Берем проверенное определение
                    definition = v_concept.get("definition")

                    # Но код берем из ОРИГИНАЛА (чтобы LLM его не испортила или не потеряла)
                    code = original.get("code_snippet")

                    verified_concepts.append({
                        "term": term,
                        "definition": definition,
                        "code_snippet": code
                    })

            # Если LLM вернула меньше концептов, чем было (потеряла что-то),
            # можно добавить логику восстановления или вернуть как есть.
            if len(verified_concepts) == 0 and len(concepts) > 0:
                logger.warning("FactCheck returned empty list, using originals")
                return concepts

            logger.info(f"Successfully verified {len(verified_concepts)} concepts")
            return verified_concepts

        except Exception as e:
            logger.error(f"Ошибка в FactCheckAgent: {e}")
            return concepts

    def _build_prompt(self, concepts: List[Dict[str, str]]) -> str:
        """Строит промпт для проверки концептов."""
        concepts_list = ""
        for i, concept in enumerate(concepts, 1):
            term = concept.get('term', '')
            definition = concept.get('definition', '')
            # Добавляем код в контекст проверки, чтобы модель видела его
            code = concept.get('code_snippet')

            concepts_list += f"{i}. Термин: {term}\n   Определение: {definition}\n"
            if code:
                concepts_list += f"   Код: {code}\n"
            concepts_list += "\n"

        prompt = f"""
Проверь следующие образовательные концепты на фактические ошибки и неточности:

{concepts_list}

ИНСТРУКЦИИ:
1. Проверь каждый концепт на соответствие научным знаниям.
2. Если найдешь ошибку — исправь определение.
3. Если концепт корректен — оставь без изменений.
4. Сохрани оригинальную структуру терминов.
5. ВАЖНО: Если у концепта был приложен код (code_snippet), ОБЯЗАТЕЛЬНО верни его без изменений.

Верни ответ в формате JSON:
{{
  "concepts": [
    {{
      "term": "оригинальный термин",
      "definition": "проверенное определение",
      "code_snippet": "код или null (если кода не было)"
    }}
  ]
}}

Только JSON, без дополнительного текста.
"""
        return prompt

