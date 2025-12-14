

from services.gigachat_client import GigaChatClient
from services.cache_manager import CacheManager
from utils.hashing import compute_hash
import logging
from typing import Any, Dict, List
from langchain.tools import tool

logger = logging.getLogger(__name__)


@tool
def enrich_concept_definition(term: str, basic_definition: str, context: str = "") -> str:
    """
    Инструмент для обогащения определения концепта дополнительными свойствами и связями.

    Args:
        term: Название концепта
        basic_definition: Базовое определение из текста
        context: Дополнительный контекст (если есть частичные свойства/связи)

    Returns:
        Обогащённое определение с ключевыми свойствами и связями
    """
    enrichment_prompt = (
        f"Вы — эксперт-методист. Дан концепт '{term}' с базовым определением: '{basic_definition}'.\n"
        f"Дополнительный контекст: {context if context else 'отсутствует'}.\n\n"
        "Ваша задача — обогатить это определение:\n"
        "1. Добавить 2-3 ключевых свойства или характеристики концепта (общеизвестные, проверенные факты).\n"
        "2. Указать важнейшие связи с другими релевантными понятиями (противопоставление, применение, примеры).\n"
        "3. Сохранить исходное определение, дополнив его, а не переписывая.\n\n"
        "Верните ТОЛЬКО обогащённое определение одним абзацем, без заголовков и пояснений."
    )
    # Здесь предполагается, что client доступен через глобальный контекст или передаётся
    # В реальной реализации нужно передать client через замыкание или другой механизм
    return enrichment_prompt  # Заглушка для демонстрации структуры


class ParserAgent:
    def __init__(self, client: GigaChatClient, cache_manager: CacheManager, cache_enabled: bool = True):
        """
        :param client: Экземпляр GigaChatClient
        :param cache_manager: Экземпляр CacheManager
        :param cache_enabled: Включать ли кэширование концептов (True/False)
        """
        self.client = client
        self.cache_manager = cache_manager
        self.cache_enabled = cache_enabled

    def parse_note(self, text: str) -> list:
        """
        Принимает сырой текст заметки.
        1. Вычисляет хеш.
        2. Проверяет и при необходимости читает кэш.
        3. Если нет кэша — вызывает LLM для извлечения концептов.
        4. Сохраняет результат в кэш при необходимости.
        5. Возвращает список концептов (list of dict).
        """
        note_hash = compute_hash(text)

        if self.cache_enabled:
            cached = self.cache_manager.get(note_hash)
            if cached is not None:
                return cached

        concepts = self._extract_concepts_from_llm(text)

        print("Извлечённые концепты из LLM:")
        for concept in concepts:
            print(f" - {concept['term']}: {concept['definition']}")

        logger.info("Извлечённые концепты (LLM): %s",
                    "; ".join(f"{c['term']}: {c['definition']}" for c in concepts))

        if self.cache_enabled:
            self.cache_manager.save(note_hash, concepts)

        return concepts

    def parse_code_note(self, text: str) -> List[Dict[str, Any]]:
        """
        Специализированный парсинг для заметок с кодом.
        Извлекает ключевые концепты И связанные с ними куски кода.
        При необходимости использует @tool для обогащения определений.
        """
        note_hash = compute_hash(text)

        # Проверка кэша
        if self.cache_enabled:
            cached = self.cache_manager.get(note_hash)
            if cached is not None:
                logger.info("ParserAgent: Using cached result for CODE extraction")
                return cached
        logger.info("ParserAgent: Running CODE extraction mode")


        prompt = (
            "Вы — интеллектуальный помощник-методист с глубокими знаниями в образовательных дисциплинах и технических областях. "
            "Ваша задача — извлечь из учебной или технической заметки ключевые концепты и составить для каждого максимально полное и полезное определение.\n\n"
            "ВАЖНО: Определение должно быть самодостаточным и образовательно ценным. Это означает:\n"
            "1. Если в тексте явно указаны свойства, характеристики или связи концепта — обязательно включите их в определение.\n"
            "2. Если в тексте указано только базовое определение БЕЗ свойств и связей — пометьте это в поле 'needs_enrichment': true. "
            "Такие концепты будут автоматически дополнены специальным инструментом.\n"
            "3. Если свойства и связи частично упомянуты в тексте — также пометьте 'needs_enrichment': true для дополнения недостающими деталями.\n\n"
            "ДОПОЛНИТЕЛЬНО ДЛЯ ТЕХНИЧЕСКИХ ЗАМЕТОК:\n"
            "4. Если в тексте встречаются фрагменты кода (функции, классы, алгоритмы, паттерны), которые иллюстрируют концепт:\n"
            "  • Выделите связь между теорией и кодом.\n"
            "  • Включите фрагмент кода в поле code_snippet (точная копия из текста).\n"
            "  • Если концепт чисто теоретический и не имеет кода — оставьте code_snippet как null.\n\n"
            "Структура определения (всё в поле definition):\n"
            "— Начните с чёткой формулировки понятия.\n"
            "— Далее перечислите ключевые свойства и характеристики (из текста).\n"
            "— Если информации достаточно, завершите описанием важнейших связей с другими концептами.\n"
            "— Если есть иллюстрирующий код — упомяните его связь с теорией в definition.\n\n"
            "Примеры правильного подхода:\n\n"
            "Пример 1 (теоретический концепт — только базовое определение, требует обогащения):\n"
            "Исходный текст: «Фотосинтез — процесс преобразования света в энергию».\n"
            "Ваш вывод:\n"
            "{\n"
            "  \"term\": \"Фотосинтез\",\n"
            "  \"definition\": \"Фотосинтез — процесс преобразования света в энергию.\",\n"
            "  \"code_snippet\": null,\n"
            "  \"needs_enrichment\": true\n"
            "}\n\n"
            "Пример 2 (технический концепт с кодом — полная информация):\n"
            "Исходный текст: «Декоратор @staticmethod в Python создаёт статические методы класса, не требующие self или cls. "
            "Применяется для утилитарных функций класса. Пример: @staticmethod def foo(): pass».\n"
            "Ваш вывод:\n"
            "{\n"
            "  \"term\": \"Декоратор @staticmethod\",\n"
            "  \"definition\": \"Декоратор @staticmethod в Python используется для создания статических методов класса — методов, которые не требуют доступа к экземпляру (self) или классу (cls). "
            "Применяется для вспомогательных функций, логически связанных с классом, но не зависящих от его состояния. Часто используется вместе с @classmethod для организации утилит.\",\n"
            "  \"code_snippet\": \"@staticmethod\\ndef foo():\\n    pass\",\n"
            "  \"needs_enrichment\": false\n"
            "}\n\n"
            "Пример 3 (алгоритм с кодом — частичная информация, требует обогащения):\n"
            "Исходный текст: «Бинарный поиск работает на отсортированных массивах. Код: while left <= right: mid = (left+right)//2...».\n"
            "Ваш вывод:\n"
            "{\n"
            "  \"term\": \"Бинарный поиск\",\n"
            "  \"definition\": \"Бинарный поиск работает на отсортированных массивах.\",\n"
            "  \"code_snippet\": \"while left <= right:\\n    mid = (left+right)//2\\n    ...\",\n"
            "  \"needs_enrichment\": true\n"
            "}\n\n"
            "Формат вывода:\n"
            "— JSON-список словарей с полями term, definition, code_snippet и needs_enrichment.\n"
            "— Не используйте Markdown-блоки, вводные комментарии или пояснения.\n"
            "— Строго следуйте формату для автоматической обработки.\n"
            "— Выделяйте только значимые концепты из текста, не добавляйте термины, которых там нет.\n"
            "— Если концепт чисто теоретический (без кода) — code_snippet = null.\n\n"
            "Текст заметки:\n"
            f"{text}"
        )

        try:
            result = self.client.generate_json(prompt)

            # >>> ВСТАВИТЬ ЛОГИРОВАНИЕ ЗДЕСЬ <<<
            import json
            logger.info("=" * 40)
            logger.info("DEBUG PARSER RAW RESULT:")
            logger.info(json.dumps(result, ensure_ascii=False, indent=2))
            logger.info("=" * 40)
            # >>> КОНЕЦ ВСТАВКИ <<<

            # Валидация и очистка + обогащение при необходимости
            valid_items = []
            if isinstance(result, list):
                for item in result:
                    # Нормализация ключей
                    term = item.get("term")
                    definition = item.get("definition")
                    code = item.get("code_snippet")
                    needs_enrichment = item.get("needs_enrichment", False)

                    if term and (definition or code):
                        # Если определение требует обогащения, вызываем @tool
                        if needs_enrichment and definition:
                            logger.info(f"Enriching concept: {term}")
                            try:
                                # Вызов декоратора @tool для обогащения
                                enriched_def = self._enrich_with_tool(term, definition, text)
                                definition = enriched_def
                                logger.info(f"Enriched definition for '{term}': {definition[:100]}...")
                            except Exception as enrich_error:
                                logger.warning(f"Failed to enrich '{term}': {enrich_error}. Using original definition.")

                        # Явно формируем чистый словарь
                        clean_item = {
                            "term": term,
                            "definition": definition or "",  # Пустая строка вместо None для текста
                            "code_snippet": code  # Здесь может быть None
                        }
                        valid_items.append(clean_item)

            logger.info(f"Extracted {len(valid_items)} code-concept pairs")
            if self.cache_enabled:
                self.cache_manager.save(note_hash, valid_items)
            return valid_items

        except Exception as e:
            logger.error(f"Code parsing failed: {e}")
            return []

    def _enrich_with_tool(self, term: str, basic_definition: str, full_text: str) -> str:
        """
        Внутренний метод для вызова инструмента обогащения определений.
        Использует декоратор @tool через LLM.
        """
        enrichment_prompt = (
            f"Вы — эксперт-методист. Дан концепт '{term}' с определением: '{basic_definition}'.\n"
            f"Контекст из исходного текста: {full_text[:500]}...\n\n"
            "Ваша задача — обогатить это определение:\n"
            "1. Сохраните исходное определение.\n"
            "2. Добавьте 2-3 ключевых свойства или характеристики (общеизвестные факты из вашей базы знаний).\n"
            "3. Укажите важнейшие связи с другими понятиями (противопоставление, применение, примеры).\n"
            "4. Используйте только проверенные, общепринятые в науке/технике факты.\n\n"
            "ФОРМАТ: Верните ТОЛЬКО обогащённое определение одним связным абзацем, без заголовков, маркеров и пояснений.\n"
        )

        try:
            # Используем обычный generate вместо generate_json для текстового ответа
            enriched = self.client.generate(enrichment_prompt)

            # Очистка от возможных артефактов форматирования
            enriched = enriched.strip().replace('\n\n', ' ').replace('  ', ' ')

            return enriched if enriched else basic_definition
        except Exception as e:
            logger.error(f"Enrichment tool failed: {e}")
            return basic_definition

    def _extract_concepts_from_llm(self, text: str) -> list:
        """
        Формирует промпт, отправляет в GigaChat, возвращает список концептов.
        При необходимости использует @tool для обогащения определений.
        """
        prompt = (
            "Вы — интеллектуальный помощник-методист с глубокими знаниями в образовательных дисциплинах. "
            "Ваша задача — извлечь из учебной заметки ключевые концепты и составить для каждого максимально полное и полезное определение.\n\n"
            "ВАЖНО: Определение должно быть самодостаточным и образовательно ценным. Это означает:\n"
            "1. Если в тексте явно указаны свойства, характеристики или связи концепта — обязательно включите их в определение.\n"
            "2. Если в тексте указано только базовое определение БЕЗ свойств и связей — пометьте это в поле 'needs_enrichment': true. "
            "Такие концепты будут автоматически дополнены специальным инструментом.\n"
            "3. Если свойства и связи частично упомянуты в тексте — также пометьте 'needs_enrichment': true для дополнения недостающими деталями.\n\n"
            "Структура определения (всё в одном поле definition):\n"
            "— Начните с чёткой формулировки понятия.\n"
            "— Далее перечислите ключевые свойства и характеристики (из текста).\n"
            "— Если информации достаточно, завершите описанием важнейших связей с другими концептами.\n\n"
            "Примеры правильного подхода:\n\n"
            "Пример 1 (в тексте только определение — требует обогащения):\n"
            "Исходный текст: «Фотосинтез — процесс преобразования света в энергию».\n"
            "Ваш вывод:\n"
            "{\n"
            "  \"term\": \"Фотосинтез\",\n"
            "  \"definition\": \"Фотосинтез — процесс преобразования света в энергию.\",\n"
            "  \"needs_enrichment\": true\n"
            "}\n\n"
            "Пример 2 (в тексте есть свойства, но нет связей — частичное обогащение):\n"
            "Исходный текст: «Хлорофилл — зелёный пигмент, поглощает свет».\n"
            "Ваш вывод:\n"
            "{\n"
            "  \"term\": \"Хлорофилл\",\n"
            "  \"definition\": \"Хлорофилл — зелёный пигмент, поглощает свет.\",\n"
            "  \"needs_enrichment\": true\n"
            "}\n\n"
            "Пример 3 (в тексте полная информация — обогащение не требуется):\n"
            "Исходный текст: «Митохондрии — органеллы клетки, производят АТФ, имеют двойную мембрану, содержат собственную ДНК, участвуют в дыхании».\n"
            "Ваш вывод:\n"
            "{\n"
            "  \"term\": \"Митохондрии\",\n"
            "  \"definition\": \"Митохондрии — органеллы эукариотических клеток, отвечающие за производство АТФ (энергетической валюты клетки). "
            "Имеют двойную мембрану, содержат собственную кольцевую ДНК (что указывает на симбиотическое происхождение), участвуют в процессе клеточного дыхания. "
            "Тесно связаны с процессом окисления глюкозы и цикла Кребса, противоположны хлоропластам по функции (митохондрии расходуют кислород, хлоропласты его производят).\",\n"
            "  \"needs_enrichment\": false\n"
            "}\n\n"
            "Формат вывода:\n"
            "— JSON-список словарей с полями term, definition и needs_enrichment.\n"
            "— Не используйте Markdown-блоки, вводные комментарии или пояснения.\n"
            "— Строго следуйте формату для автоматической обработки.\n"
            "— Выделяйте только значимые концепты из текста, не добавляйте термины, которых там нет.\n\n"
            "Текст заметки:\n"
            f"{text}"
        )

        result = self.client.generate_json(prompt)

        import json
        logger.info("=" * 40)
        logger.info("DEBUG PARSER RAW RESULT (parse_note):")
        logger.info(json.dumps(result, ensure_ascii=False, indent=2))
        logger.info("=" * 40)

        # Опционально: валидация структуры результата здесь
        if not isinstance(result, list):
            raise ValueError("GigaChat вернул неожиданный формат (ожидается список концептов)")

        # Обогащение концептов при необходимости
        enriched_result = []
        for concept in result:
            term = concept.get("term")
            definition = concept.get("definition")
            needs_enrichment = concept.get("needs_enrichment", False)

            if needs_enrichment and definition and term:
                logger.info(f"Enriching concept: {term}")
                try:
                    # Вызов декоратора @tool для обогащения
                    enriched_def = self._enrich_with_tool(term, definition, text)
                    concept["definition"] = enriched_def
                    logger.info(f"Enriched definition for '{term}': {enriched_def[:100]}...")
                except Exception as enrich_error:
                    logger.warning(f"Failed to enrich '{term}': {enrich_error}. Using original definition.")

            # Удаляем служебное поле перед возвратом
            concept.pop("needs_enrichment", None)
            enriched_result.append(concept)

        return enriched_result
