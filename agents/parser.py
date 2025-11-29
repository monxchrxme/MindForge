from services.gigachat_client import GigaChatClient
from services.cache_manager import CacheManager
from utils.hashing import get_hash

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
        note_hash = get_hash(text)
        if self.cache_enabled:
            cached = self.cache_manager.get(note_hash)
            if cached is not None:
                return cached

        concepts = self._extract_concepts_from_llm(text)
        if self.cache_enabled:
            self.cache_manager.save(note_hash, concepts)
        return concepts

    def _extract_concepts_from_llm(self, text: str) -> list:
        """
        Формирует промпт, отправляет в GigaChat, возвращает список концептов.
        """
        prompt = (
            "Извлеки из следующей учебной заметки ключевые понятия, определения и причинно-следственные связи. "
            "Верни результат строго в формате JSON-списка словарей: "
            "[{'term': <термин>, 'definition': <определение>}]. Вот заметка: \n"
            f"{text}"
        )
        result = self.client.generate_json(prompt)
        # Опционально: валидация структуры результата здесь
        if not isinstance(result, list):
            raise ValueError("GigaChat вернул неожиданный формат (ожидается список концептов)")
        return result