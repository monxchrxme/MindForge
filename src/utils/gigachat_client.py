"""
Обёртка для работы с GigaChat через LangChain с подсчетом токенов
"""

from langchain_gigachat import GigaChat
from langchain_core.messages import HumanMessage
import os
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# ГЛОБАЛЬНЫЙ ТРЕКЕР ТОКЕНОВ
# ============================================================================

class TokenUsageTracker:
    """Класс для отслеживания использования токенов"""

    def __init__(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.embedding_tokens = 0
        self.request_count = 0
        self.embedding_request_count = 0

    def add_usage(self, prompt: int, completion: int, total: int):
        """Добавление статистики использования LLM"""
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.total_tokens += total
        self.request_count += 1

        logger.info(
            f"📊 LLM запрос #{self.request_count}: "
            f"prompt={prompt}, completion={completion}, total={total}"
        )

    def add_embedding_usage(self, tokens: int):
        """Добавление статистики использования эмбеддингов"""
        self.embedding_tokens += tokens
        self.embedding_request_count += 1

        logger.info(
            f"🔢 Embedding запрос #{self.embedding_request_count}: "
            f"tokens={tokens}"
        )

    def get_summary(self) -> dict:
        """Получение итоговой статистики"""
        return {
            'total_requests': self.request_count,
            'prompt_tokens': self.prompt_tokens,
            'completion_tokens': self.completion_tokens,
            'total_tokens': self.total_tokens,
            'embedding_tokens': self.embedding_tokens,
            'embedding_requests': self.embedding_request_count,
            'grand_total_tokens': self.total_tokens + self.embedding_tokens
        }

    def log_summary(self):
        """Логирование итоговой статистики"""
        logger.info("="*70)
        logger.info("📊 ИТОГОВАЯ СТАТИСТИКА ИСПОЛЬЗОВАНИЯ ТОКЕНОВ")
        logger.info("="*70)
        logger.info(f"LLM запросов: {self.request_count}")
        logger.info(f"  Токенов в промптах: {self.prompt_tokens:,}")
        logger.info(f"  Токенов в ответах: {self.completion_tokens:,}")
        logger.info(f"  Всего LLM токенов: {self.total_tokens:,}")
        logger.info("-"*70)
        logger.info(f"Embedding запросов: {self.embedding_request_count}")
        logger.info(f"  Embedding токенов: {self.embedding_tokens:,}")
        logger.info("-"*70)
        logger.info(f"ВСЕГО ТОКЕНОВ: {self.total_tokens + self.embedding_tokens:,}")
        logger.info("="*70)


_global_token_tracker = TokenUsageTracker()


def get_global_token_tracker() -> TokenUsageTracker:
    """Получение глобального трекера токенов"""
    return _global_token_tracker


# ============================================================================
# ПРОКСИ ДЛЯ ПОДСЧЕТА ТОКЕНОВ
# ============================================================================

class GigaChatProxy:
    """Прокси-обертка для GigaChat с автоматическим подсчетом токенов"""

    def __init__(self, llm: GigaChat):
        self._llm = llm
        self._tracker = get_global_token_tracker()

    def invoke(self, messages, **kwargs):
        response = self._llm.invoke(messages, **kwargs)
        self._extract_and_log_tokens(response)
        return response

    def chat(self, message, **kwargs):
        if not isinstance(message, list):
            message = [message]
        return self.invoke(message, **kwargs)

    def _extract_and_log_tokens(self, response):
        try:
            prompt_tokens = 0
            completion_tokens = 0
            total_tokens = 0

            if hasattr(response, 'response_metadata'):
                metadata = response.response_metadata

                if 'token_usage' in metadata:
                    usage = metadata['token_usage']
                    prompt_tokens = usage.get('prompt_tokens', 0)
                    completion_tokens = usage.get('completion_tokens', 0)
                    total_tokens = usage.get('total_tokens', 0)
                elif 'usage_metadata' in metadata:
                    usage = metadata['usage_metadata']
                    prompt_tokens = usage.get('input_tokens', 0)
                    completion_tokens = usage.get('output_tokens', 0)
                    total_tokens = usage.get('total_tokens', 0)

            if total_tokens == 0 and (prompt_tokens > 0 or completion_tokens > 0):
                total_tokens = prompt_tokens + completion_tokens

            if total_tokens > 0:
                self._tracker.add_usage(prompt_tokens, completion_tokens, total_tokens)
            else:
                estimated = len(response.content.split()) * 1.3
                self._tracker.add_usage(0, int(estimated), int(estimated))

        except Exception as e:
            logger.warning(f"⚠️  Ошибка извлечения токенов: {e}")

    def __getattr__(self, name):
        return getattr(self._llm, name)


# ============================================================================
# ПРОКСИ ДЛЯ ЭМБЕДДИНГОВ С ПОДСЧЕТОМ ТОКЕНОВ
# ============================================================================

class GigaChatEmbeddingsProxy:
    """Прокси для GigaChatEmbeddings с подсчетом токенов"""

    def __init__(self, embeddings):
        self._embeddings = embeddings
        self._tracker = get_global_token_tracker()

    def embed_documents(self, texts):
        """Прокси для embed_documents с подсчетом токенов"""
        result = self._embeddings.embed_documents(texts)

        total_text = ' '.join(texts)
        tokens = int(len(total_text.split()) * 1.3)
        self._tracker.add_embedding_usage(tokens)

        return result

    def embed_query(self, text):
        """Прокси для embed_query с подсчетом токенов"""
        result = self._embeddings.embed_query(text)

        tokens = int(len(text.split()) * 1.3)
        self._tracker.add_embedding_usage(tokens)

        return result

    def __call__(self, text):
        """
        ИСПРАВЛЕНО: Делаем прокси callable для совместимости с FAISS
        FAISS вызывает embedding_function(text) напрямую
        """
        return self.embed_query(text)

    def __getattr__(self, name):
        return getattr(self._embeddings, name)


# ============================================================================
# ФУНКЦИИ СОЗДАНИЯ КЛИЕНТОВ
# ============================================================================

def create_gigachat_parser_client():
    """Создание GigaChat клиента для Parser Agent"""
    llm = GigaChat(
        credentials=os.getenv("GIGACHAT_CREDENTIALS"),
        verify_ssl_certs=False,
        scope=os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS"),
    )
    return GigaChatProxy(llm)


def create_gigachat_quiz_client():
    """Создание GigaChat клиента для Quiz Agent"""
    llm = GigaChat(
        credentials=os.getenv("GIGACHAT_CREDENTIALS"),
        verify_ssl_certs=False,
        scope=os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS"),
    )
    return GigaChatProxy(llm)


def create_gigachat_embeddings():
    """Создание GigaChat эмбеддингов с подсчетом токенов"""
    from langchain_community.embeddings import GigaChatEmbeddings

    embeddings = GigaChatEmbeddings(
        credentials=os.getenv("GIGACHAT_CREDENTIALS"),
        verify_ssl_certs=False,
        scope=os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS")
    )

    return GigaChatEmbeddingsProxy(embeddings)
