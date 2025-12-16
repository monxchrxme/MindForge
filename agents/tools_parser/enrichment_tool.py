from typing import Dict, Any
from agents.tools_quiz.base import BaseTool
from services.gigachat_client import GigaChatClient
import logging
import re

logger = logging.getLogger(__name__)


class ConceptEnrichmentTool(BaseTool):
    """Обогащает определения концептов дополнительными свойствами и связями"""

    def __init__(self, client: GigaChatClient):
        self.client = client

    @property
    def name(self) -> str:
        return "enrich_concept_definition"

    @property
    def description(self) -> str:
        return """
Обогащает определение концепта дополнительными свойствами и связями.

КОГДА ИСПОЛЬЗОВАТЬ:
- Определение содержит только базовую формулировку (1 короткое предложение)
- Отсутствуют ключевые свойства или характеристики
- Не указаны связи с другими концептами
- Слишком краткое для учебных целей

НЕ ИСПОЛЬЗОВАТЬ ЕСЛИ:
- Определение содержит 3+ предложения с деталями
- Указаны свойства, характеристики И связи
- Есть примеры и практическое применение

ПРИМЕР ВЫЗОВА:
{
    "term": "Фотосинтез",
    "basic_definition": "Процесс преобразования света в энергию",
    "context": "Биология, растения..."
}
        """.strip()

    def execute(self, **kwargs) -> Dict[str, Any]:
        """Обогащает определение концепта"""
        term = kwargs.get("term", "")
        basic_definition = kwargs.get("basic_definition", "")
        context = kwargs.get("context", "")

        if not term or not basic_definition:
            return {
                "success": False,
                "error": "Missing required arguments: term and basic_definition",
                "enriched_definition": basic_definition
            }

        prompt = f"""
Вы — эксперт-методист. Обогатите учебное определение.

КОНЦЕПТ: {term}
ИСХОДНОЕ ОПРЕДЕЛЕНИЕ: {basic_definition}

ПРАВИЛА ОБОГАЩЕНИЯ:
1. ОБЯЗАТЕЛЬНО начните с исходного определения (сохраните его полностью!)
2. Добавьте 1-2 ключевых свойства в ОДНОМ предложении
3. Добавьте главную связь/применение в ОДНОМ предложении (опционально)
4. Итого: 2 предложения МАКСИМУМ

СТРОГИЕ ЗАПРЕТЫ:
НЕ используйте квадратные скобки []
НЕ используйте переносы строк внутри определения
НЕ переписывайте исходное определение
НЕ перечисляйте всё подряд через запятую
НЕ делайте списки и маркеры
НЕ добавляй примеры и огромные пояснения

ФОРМАТ ВЫВОДА:
[Исходное определение слово в слово]. [1-2 свойства]. [Применение при необходимости].

ПЛОХОЙ ПРИМЕР:
"[Класс определяет структуру, поведение, атрибуты, методы, инкапсуляцию, наследование, полиморфизм...]"

ХОРОШИЙ ПРИМЕР:
"Класс — шаблон для создания объектов. Определяет их структуру (атрибуты) и поведение (методы). Основа объектно-ориентированного программирования."

Обогащённое определение (2-3 предложения, без скобок, без переносов):
        """.strip()

        try:
            enriched = self.client.generate(prompt)

            # Агрессивная очистка
            enriched = enriched.strip()

            # Удаляем квадратные скобки
            enriched = enriched.replace('[', '').replace(']', '')

            # Заменяем переносы строк на пробелы
            enriched = enriched.replace('\n', ' ')

            # Схлопываем множественные пробелы
            enriched = re.sub(r'\s+', ' ', enriched)

            # Убираем markdown заголовки
            if enriched.startswith('#'):
                enriched = '\n'.join(enriched.split('\n')[1:]).strip()

            # Убираем возможные вводные фразы
            prefixes_to_remove = [
                "Обогащённое определение:",
                "Определение:",
                "Ответ:",
                "Результат:",
            ]
            for prefix in prefixes_to_remove:
                if enriched.startswith(prefix):
                    enriched = enriched[len(prefix):].strip()

            # Проверка на избыточную длину
            sentences = enriched.count('.') + enriched.count('!') + enriched.count('?')
            if sentences > 3:
                logger.warning(f"Enrichment for '{term}' too long ({sentences} sentences), truncating...")
                # Оставляем только первые 3 предложения
                parts = []
                current = ""
                for char in enriched:
                    current += char
                    if char in '.!?':
                        parts.append(current.strip())
                        current = ""
                        if len(parts) >= 3:
                            break
                enriched = ' '.join(parts)

            # Проверка на дубликаты (если определение повторяется)
            words = enriched.split()
            if len(words) != len(set(words)):
                # Есть повторяющиеся слова - возможно дублирование
                logger.warning(f"Possible duplication in '{term}', checking...")

            if not enriched or len(enriched) < 20:
                logger.warning(f"Enrichment returned empty/short result for '{term}'")
                return {
                    "success": False,
                    "error": "Empty or too short response",
                    "enriched_definition": basic_definition
                }

            # Финальная проверка: сохранено ли исходное определение?
            # Берем первые 20 символов исходного определения
            basic_start = basic_definition[:20].lower().strip()
            enriched_start = enriched[:20].lower().strip()

            if basic_start not in enriched[:50].lower():
                logger.warning(f"Original definition lost for '{term}', prepending it...")
                enriched = f"{basic_definition} {enriched}"

            logger.info(f"Successfully enriched '{term}': {len(enriched)} chars, ~{sentences} sentences")
            return {
                "success": True,
                "enriched_definition": enriched,
                "original_definition": basic_definition
            }

        except Exception as e:
            logger.error(f"Enrichment failed for '{term}': {e}")
            return {
                "success": False,
                "error": str(e),
                "enriched_definition": basic_definition
            }
