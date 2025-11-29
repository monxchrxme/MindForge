from services.gigachat_client import GigaChatClient
from services.cache_manager import CacheManager
from utils.hashing import compute_hash
import logging

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

    def _extract_concepts_from_llm(self, text: str) -> list:
        """
        Формирует промпт, отправляет в GigaChat, возвращает список концептов.
        """
        prompt = (
            "Вы — интеллектуальный помощник-методист, обладающий экспертными навыками в анализе и структурировании учебных текстов для студентов. Ваша задача — максимально точно и объективно выделить из переданного ниже фрагмента академической заметки ключевые термины/понятия, их определения и, если есть, важные причинно-следственные связи или основные факты.\n\n"
            "Инструкция:\n"
            "1. Проанализируйте текст как преподаватель, знакомый с методиками активного обучения и стандартами академического письма.\n"
            "2. Выделяйте только действительно значимые концепты: термины, определения, законы, правила, важные факты и отношения между ними. Не добавляйте выдуманных терминов.\n"
            "3. Для каждого понятия составьте пару \"term\": <ключевой термин/понятие>, \"definition\": <четкое определение или краткое описание>.\n"
            "4. Если находите причинно-следственную связь, оформите ее также как отдельный элемент: \"term\": <описание связи>, \"definition\": <пояснение сути этой связи>.\n\n"
            "Формат вывода:\n"
            "— Выводите результат строго в виде JSON-списка словарей без пояснений и лишнего текста, пример:\n"
            "[\n"
            "  {\"term\": \"Сумма углов треугольника\", \"definition\": \"В каждом треугольнике сумма углов равна 180 градусам.\"},\n"
            "  {\"term\": \"Теорема Пифагора\", \"definition\": \"В прямоугольном треугольнике квадрат гипотенузы равен сумме квадратов катетов.\"}\n"
            "]\n"
            "— Не используйте Markdown-блоки.\n"
            "— Не используй формулы Latex.\n"
            "— Не добавляйте вводные или заключительные комментарии.\n"
            "— Строго следуйте указанному формату, чтобы результат мог быть автоматически обработан.\n\n"
            "Ваша задача — повысить эффективность обучения студента, помогая ему выявить главные идеи, термины и взаимосвязи, заложенные в заметке. Все определения должны быть сформулированы однозначно и понятно для уровня бакалавра.\n\n"
            "Текст заметки:\n"
            f"{text}"
        )
        result = self.client.generate_json(prompt)
        # Опционально: валидация структуры результата здесь
        if not isinstance(result, list):
            raise ValueError("GigaChat вернул неожиданный формат (ожидается список концептов)")
        return result