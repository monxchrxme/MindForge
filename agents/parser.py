from services.gigachat_client import GigaChatClient
from services.cache_manager import CacheManager
from agents.tools_parser.enrichment_tool import ConceptEnrichmentTool
from utils.hashing import compute_hash
import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


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

        # Инициализируем enrichment tool
        self.enrichment_tool = ConceptEnrichmentTool(client)
        logger.info(f"✅ Enrichment tool initialized: {self.enrichment_tool.name}")

    def parse_note(self, text: str) -> List[Dict[str, Any]]:
        """
        Парсинг обычных заметок (без кода).
        Извлекает ключевые концепты с полными определениями.
        """
        logger.info("ParserAgent: Running STANDARD extraction mode")

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
            "Тесно связаны с процессом окисления глюкозы и цикла Кребса, противоположны хлоропластам по функции (митохондрии расходуют кислород, хлоропласты их производят).\",\n"
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

        try:
            result = self.client.generate_json(prompt, temperature=0.2)

            # DEBUG: Логирование сырого результата от LLM
            import json
            logger.info("=" * 40)
            logger.info("DEBUG PARSER RAW RESULT (STANDARD MODE):")
            logger.info(json.dumps(result, ensure_ascii=False, indent=2))
            logger.info("=" * 40)

            # Валидация и нормализация + обогащение при необходимости
            valid_items = []
            if isinstance(result, list):
                for item in result:
                    # Нормализация ключей
                    term = item.get("term")
                    definition = item.get("definition")
                    needs_enrichment = item.get("needs_enrichment", False)

                    # Проверка обязательных полей
                    if term and definition:
                        # Если определение требует обогащения, вызываем tool
                        if needs_enrichment:
                            logger.info(f"🔧 Enriching concept: {term}")
                            enriched_def = self._enrich_with_tool(term, definition, text[:300])
                            if enriched_def:
                                definition = enriched_def
                                logger.info(f"✅ Enriched '{term}': {definition[:100]}...")

                        # Формируем чистый словарь
                        clean_item = {
                            "term": term,
                            "definition": definition
                        }
                        valid_items.append(clean_item)

            logger.info(f"Extracted {len(valid_items)} concepts")
            return valid_items

        except Exception as e:
            logger.error(f"Standard parsing failed: {e}")
            return []

    def parse_code_note(self, text: str) -> List[Dict[str, Any]]:
        """
        Специализированный парсинг для заметок с кодом.
        Извлекает ключевые концепты И связанные с ними куски кода.
        """
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
            "   • Выделите связь между теорией и кодом.\n"
            "   • Включите фрагмент кода в поле code_snippet (точная копия из текста).\n"
            "   • Если концепт чисто теоретический и не имеет кода — оставьте code_snippet как null.\n\n"
            "Структура определения (всё в поле definition):\n"
            "— Начните с чёткой формулировки понятия.\n"
            "— Далее перечислите ключевые свойства и характеристики (из текста).\n"
            "— Если информации достаточно, завершите описанием важнейших связей с другими концептами.\n"
            "— Если есть иллюстрирующий код — упомяните его связь с теорией в definition.\n\n"
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
            result = self.client.generate_json(prompt, temperature=0.2)

            # DEBUG: Логирование
            import json
            logger.info("=" * 40)
            logger.info("DEBUG PARSER RAW RESULT (CODE MODE):")
            logger.info(json.dumps(result, ensure_ascii=False, indent=2))
            logger.info("=" * 40)

            # Валидация и очистка + обогащение при необходимости
            valid_items = []
            if isinstance(result, list):
                for item in result:
                    # Нормализация ключей
                    term = item.get("term")
                    definition = item.get("definition")
                    code = item.get("code_snippet") or item.get("code") or item.get("snippet") or item.get("example")
                    needs_enrichment = item.get("needs_enrichment", False)

                    if term and (definition or code):
                        # Если определение требует обогащения, вызываем tool
                        if needs_enrichment and definition:
                            logger.info(f"🔧 Enriching concept: {term}")
                            enriched_def = self._enrich_with_tool(term, definition, text[:300])
                            if enriched_def:
                                definition = enriched_def
                                logger.info(f"✅ Enriched '{term}': {definition[:100]}...")

                        # Явно формируем чистый словарь
                        clean_item = {
                            "term": term,
                            "definition": definition or "",  # Пустая строка вместо None для текста
                            "code_snippet": code  # Здесь может быть None
                        }
                        valid_items.append(clean_item)

            logger.info(f"Extracted {len(valid_items)} code-concept pairs")
            return valid_items

        except Exception as e:
            logger.error(f"Code parsing failed: {e}")
            return []

    def _enrich_with_tool(self, term: str, basic_definition: str, context: str) -> str:
        """
        Внутренний метод для вызова инструмента обогащения определений.
        Использует ConceptEnrichmentTool.

        :param term: Название концепта
        :param basic_definition: Базовое определение
        :param context: Контекст из исходного текста (первые 300 символов)
        :return: Обогащённое определение или исходное (если обогащение не удалось)
        """
        try:
            result = self.enrichment_tool.execute(
                term=term,
                basic_definition=basic_definition,
                context=context
            )

            if result.get("success"):
                return result.get("enriched_definition", basic_definition)
            else:
                error = result.get("error", "Unknown error")
                logger.warning(f"⚠️ Enrichment failed for '{term}': {error}, using original")
                return basic_definition

        except Exception as e:
            logger.error(f"Enrichment tool error for '{term}': {e}")
            return basic_definition
