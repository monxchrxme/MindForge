from services.gigachat_client import GigaChatClient
from services.cache_manager import CacheManager
from agents.tools_parser.enrichment_tool import ConceptEnrichmentTool
from utils.hashing import compute_hash
import logging
from typing import Any, Dict, List
import json

logger = logging.getLogger(__name__)


class ParserAgent:
    """
    Агент для извлечения концептов из учебных заметок с интеллектуальным обогащением.
    Parser САМ решает через LLM, когда вызывать enrichment tool.
    """

    def __init__(self, client: GigaChatClient, cache_manager: CacheManager, cache_enabled: bool = True):
        """
        Инициализация агента парсинга.

        Args:
            client: Экземпляр GigaChatClient для LLM-запросов
            cache_manager: Менеджер кэширования результатов
            cache_enabled: Флаг включения кэширования
        """
        self.client = client
        self.cache_manager = cache_manager
        self.cache_enabled = cache_enabled

        # Инициализируем tool обогащения
        self.enrichment_tool = ConceptEnrichmentTool(client)
        logger.info(f"✅ Enrichment tool initialized: {self.enrichment_tool.name}")

    def parse_note(self, text: str) -> List[Dict[str, Any]]:
        """
        Парсинг обычных заметок (без кода).
        Извлекает концепты и автоматически обогащает определения.

        Args:
            text: Текст учебной заметки

        Returns:
            Список словарей [{term: str, definition: str}, ...]
        """
        logger.info("ParserAgent: Running STANDARD extraction mode")

        # Упрощенный промпт - только извлечение
        extraction_prompt = (
            "Ты — эксперт по извлечению ключевых концептов из учебных материалов.\n\n"
            "ЗАДАЧА: Извлеки из текста все значимые концепты (термины, понятия, определения).\n\n"
            "ДЛЯ КАЖДОГО КОНЦЕПТА:\n"
            "- term: точное название концепта\n"
            "- definition: определение из текста КАК ЕСТЬ (не додумывай, не расширяй)\n\n"
            "ПРАВИЛА:\n"
            "- Извлекай только то, что явно указано в тексте\n"
            "- Один концепт = одна запись\n"
            "- Если определения нет, используй описание из текста\n"
            "- Не добавляй концепты, которых нет в тексте\n\n"
            "ФОРМАТ ВЫВОДА:\n"
            "JSON-массив: [{\"term\": \"...\", \"definition\": \"...\"}, ...]\n"
            "Без Markdown-блоков, без комментариев.\n\n"
            f"ТЕКСТ ЗАМЕТКИ:\n{text}"
        )

        try:
            result = self.client.generate_json(extraction_prompt, temperature=0.2)

            # DEBUG: Логирование
            logger.info("=" * 50)
            logger.info("PARSER RAW EXTRACTION (STANDARD MODE):")
            logger.info(json.dumps(result, ensure_ascii=False, indent=2))
            logger.info("=" * 50)

            valid_items = []
            if isinstance(result, list):
                for idx, item in enumerate(result, 1):
                    term = item.get("term")
                    definition = item.get("definition")

                    if not term or not definition:
                        logger.warning(f"Skipping item {idx}: missing term or definition")
                        continue

                    logger.info(f"[{idx}/{len(result)}] Processing concept: {term}")

                    # Parser САМ решает через LLM: нужно ли обогащение
                    final_definition = self._decide_and_enrich(
                        term=term,
                        definition=definition,
                        context=text[:500]
                    )

                    valid_items.append({
                        "term": term,
                        "definition": final_definition
                    })

                    logger.info(f"✓ Processed '{term}': {len(final_definition)} chars")

            logger.info(f"Successfully extracted and processed {len(valid_items)} concepts")
            return valid_items

        except Exception as e:
            logger.error(f"Standard parsing failed: {e}", exc_info=True)
            return []

    def parse_code_note(self, text: str) -> List[Dict[str, Any]]:
        """
        Специализированный парсинг для заметок с кодом.
        Извлекает концепты + code snippets, обогащает через решение парсера.

        Args:
            text: Текст технической заметки с кодом

        Returns:
            Список словарей [{term: str, definition: str, code_snippet: str|None}, ...]
        """
        logger.info("ParserAgent: Running CODE extraction mode")

        extraction_prompt = (
            "Ты — эксперт по извлечению технических концептов из учебных материалов с кодом.\n\n"
            "ЗАДАЧА: Извлеки концепты И связанные с ними фрагменты кода.\n\n"
            "ДЛЯ КАЖДОГО КОНЦЕПТА:\n"
            "- term: название концепта (функция, класс, алгоритм, паттерн)\n"
            "- definition: определение из текста КАК ЕСТЬ\n"
            "- code_snippet: точная копия кода из текста (если есть) или null\n\n"
            "ПРАВИЛА:\n"
            "- Извлекай только явно указанные в тексте концепты\n"
            "- Если код иллюстрирует концепт — включи его в code_snippet\n"
            "- Если концепт теоретический (без кода) — code_snippet = null\n"
            "- Сохраняй форматирование кода (отступы, переносы)\n\n"
            "ФОРМАТ ВЫВОДА:\n"
            "JSON-массив: [{\"term\": \"...\", \"definition\": \"...\", \"code_snippet\": \"...\" или null}, ...]\n"
            "Без Markdown-блоков, без комментариев.\n\n"
            f"ТЕКСТ ЗАМЕТКИ:\n{text}"
        )

        try:
            result = self.client.generate_json(extraction_prompt, temperature=0.2)

            # DEBUG: Логирование
            logger.info("=" * 50)
            logger.info("PARSER RAW EXTRACTION (CODE MODE):")
            logger.info(json.dumps(result, ensure_ascii=False, indent=2))
            logger.info("=" * 50)

            valid_items = []
            if isinstance(result, list):
                for idx, item in enumerate(result, 1):
                    term = item.get("term")
                    definition = item.get("definition")
                    code = (item.get("code_snippet") or
                            item.get("code") or
                            item.get("snippet") or
                            item.get("example"))

                    if not term or not (definition or code):
                        logger.warning(f"Skipping item {idx}: missing required fields")
                        continue

                    logger.info(f"[{idx}/{len(result)}] Processing code concept: {term}")

                    # Parser решает через LLM: нужно ли обогащение
                    if definition:
                        final_definition = self._decide_and_enrich(
                            term=term,
                            definition=definition,
                            context=text[:500]
                        )
                    else:
                        final_definition = ""
                        logger.warning(f"No definition for '{term}', skipping enrichment")

                    valid_items.append({
                        "term": term,
                        "definition": final_definition,
                        "code_snippet": code
                    })

                    logger.info(
                        f"✓ Processed '{term}': def={len(final_definition)} chars, code={'Yes' if code else 'No'}"
                    )

            logger.info(f"Successfully extracted and processed {len(valid_items)} code concepts")
            return valid_items

        except Exception as e:
            logger.error(f"Code parsing failed: {e}", exc_info=True)
            return []

    def _decide_and_enrich(self, term: str, definition: str, context: str = "") -> str:
        """
        Parser САМ решает через LLM: нужно ли обогащение, и вызывает tool если нужно.

        Args:
            term: Название концепта
            definition: Определение из текста
            context: Контекст из документа

        Returns:
            Финальное определение (обогащённое или исходное)
        """
        # Промпт для принятия решения (смягченный)
        decision_prompt = f"""
    Ты — эксперт по оценке учебных определений.

    КОНЦЕПТ: "{term}"
    ОПРЕДЕЛЕНИЕ: "{definition}"

    ЗАДАЧА: Реши, нужно ли ОБОГАТИТЬ это определение для студентов.

    ОБОГАЩЕНИЕ НУЖНО (ответь ENRICH) если:
    • Определение короче 2 предложений
    • Только базовая формулировка без деталей
    • Нет свойств, характеристик или примеров
    • Студент не поймет КАК это работает или ГДЕ применяется

    ОБОГАЩЕНИЕ НЕ НУЖНО (ответь KEEP) если:
    • Определение содержит 3+ предложения с деталями
    • Уже есть свойства И связи с другими понятиями
    • Есть конкретные примеры или применение
    • Информация избыточна для базового понимания

    ПРИМЕРЫ:

    ПРИМЕР 1:
    Концепт: "Класс"
    Определение: "Класс — это шаблон для создания объектов."
    Решение: ENRICH (слишком кратко, нет деталей)

    ПРИМЕР 2:
    Концепт: "Полиморфизм"
    Определение: "Полиморфизм — способность объектов разных классов обрабатываться через единый интерфейс. В Python реализуется через утиную типизацию и переопределение методов. Позволяет писать гибкий код."
    Решение: KEEP (уже есть суть, свойства и применение)

    ПРИМЕР 3:
    Концепт: "Инкапсуляция"
    Определение: "Инкапсуляция — механизм сокрытия данных."
    Решение: ENRICH (базовое определение без объяснения КАК и ЗАЧЕМ)

    ТВОЕ РЕШЕНИЕ (одно слово):
    ENRICH или KEEP?
        """.strip()

        try:
            # LLM принимает решение
            decision = self.client.generate(decision_prompt, temperature=0.1).strip().upper()

            # Логируем полный ответ для отладки
            logger.info(f"Decision for '{term}': {decision}")

            # Парсим ответ (ищем ключевое слово)
            needs_enrichment = "ENRICH" in decision

            if needs_enrichment:
                logger.info(f"🔧 Enriching '{term}' via tool...")

                # Вызываем enrichment tool
                result = self.enrichment_tool.execute(
                    term=term,
                    basic_definition=definition,
                    context=context
                )

                if result.get("success"):
                    enriched_def = result.get("enriched_definition", definition)
                    logger.info(f"✅ Enriched '{term}': {len(enriched_def)} chars")
                    return enriched_def
                else:
                    error = result.get("error", "Unknown error")
                    logger.warning(f"⚠️ Enrichment failed for '{term}': {error}, using original")
                    return definition
            else:
                logger.info(f"✓ Definition for '{term}' is complete, no enrichment needed")
                return definition

        except Exception as e:
            logger.error(f"Decision/enrichment process failed for '{term}': {e}")
            return definition

