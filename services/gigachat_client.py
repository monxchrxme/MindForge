# services/gigachat_client.py

from langchain_gigachat import GigaChat
from typing import Any, Dict, List, Union
import json
import logging
import re

logger = logging.getLogger(__name__)


class GigaChatClient:
    """
    Обертка-враппер над LangChain-GigaChat.
    Предоставляет унифицированный интерфейс для всех агентов системы.
    Ведет статистику использования токенов и количества запросов.
    """

    def __init__(
            self,
            credentials: dict,
            model: str = "GigaChat",
            temperature: float = 0.7,
            timeout: int = 30,
            verify_ssl_certs: bool = False
    ):
        """
        Инициализация клиента GigaChat.

        Args:
            credentials: Словарь с ключами:
                - 'client_id': идентификатор клиента
                - 'client_secret': секретный ключ авторизации
            model: Название модели (GigaChat, GigaChat-Pro, etc.)
            temperature: Параметр случайности генерации (0.0 - 1.0)
            timeout: Таймаут запроса в секундах
            verify_ssl_certs: Проверка SSL сертификатов
        """
        self.model_name = model
        self.temperature = temperature
        self.timeout = timeout

        # Валидация credentials
        if not credentials.get("client_id") or not credentials.get("client_secret"):
            raise ValueError("Credentials must contain 'client_id' and 'client_secret'")

        # Инициализация LangChain GigaChat
        try:
            self.gigachat = GigaChat(
                credentials=credentials.get("client_secret"),  # LangChain использует client_secret напрямую
                model=self.model_name,
                temperature=self.temperature,
                timeout=self.timeout,
                verify_ssl_certs=verify_ssl_certs
            )
            logger.info(f"GigaChat client initialized: model={model}, temperature={temperature}")
        except Exception as e:
            logger.error(f"Failed to initialize GigaChat: {str(e)}")
            raise

        # Статистика использования
        self.total_prompt_tokens: int = 0
        self.total_completion_tokens: int = 0
        self.total_requests: int = 0

    def generate(self, prompt: str) -> str:
        """
        Универсальный метод для получения текстового результата (RAW STRING).
        Используется для свободной генерации текста без структурированного формата.

        Args:
            prompt: Текст промпта для модели

        Returns:
            str: Сгенерированный текст от модели

        Raises:
            Exception: При ошибках сети или API
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        try:
            logger.debug(f"Generating text response (prompt length: {len(prompt)} chars)")

            # Вызов модели через LangChain
            response = self.gigachat.invoke(prompt)

            # Извлечение текста из ответа
            if hasattr(response, 'content'):
                result_text = response.content
            else:
                result_text = str(response)

            # Обновление статистики
            self._update_stats(prompt, result_text)

            logger.debug(f"Text generation successful (response length: {len(result_text)} chars)")

            return result_text

        except Exception as e:
            logger.error(f"Error in generate(): {str(e)}", exc_info=True)
            raise Exception(f"GigaChat API error: {str(e)}")

    def generate_json(
            self,
            prompt: str,
            retry_attempts: int = 3
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Запрос к GigaChat с ожиданием JSON-ответа.
        Автоматически парсит ответ и возвращает Python-объект (dict/list).

        Args:
            prompt: Текст промпта (должен содержать инструкцию возврата JSON)
            retry_attempts: Количество попыток при ошибке парсинга

        Returns:
            Union[Dict, List[Dict]]: Распарсенный JSON-объект

        Raises:
            ValueError: Если не удалось распарсить JSON после всех попыток
            Exception: При ошибках API
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        last_error = None

        for attempt in range(1, retry_attempts + 1):
            try:
                logger.debug(f"Generating JSON response (attempt {attempt}/{retry_attempts})")

                # Получение сырого текста
                raw_response = self.generate(prompt)

                # Парсинг JSON из ответа
                parsed_json = self._parse_json_from_text(raw_response)

                logger.debug(f"JSON parsing successful on attempt {attempt}")

                return parsed_json

            except json.JSONDecodeError as e:
                last_error = e
                logger.warning(
                    f"JSON parsing failed on attempt {attempt}: {str(e)}\n"
                    f"Raw response preview: {raw_response[:200]}..."
                )

                # Добавляем уточнение в промпт для следующей попытки
                if attempt < retry_attempts:
                    prompt = self._enhance_json_prompt(prompt)

            except Exception as e:
                logger.error(f"Unexpected error in generate_json(): {str(e)}")
                raise

        # Если все попытки провалились
        error_msg = f"Failed to parse JSON after {retry_attempts} attempts. Last error: {str(last_error)}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    def get_usage_stats(self) -> Dict[str, int]:
        """
        Получение текущей статистики использования модели.

        Returns:
            Dict с ключами:
                - prompt_tokens: количество токенов в промптах
                - completion_tokens: количество токенов в ответах
                - total_requests: общее количество запросов
        """
        return {
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "total_requests": self.total_requests
        }

    def reset_stats(self) -> None:
        """
        Сброс статистики использования.
        Полезно для измерения потребления на отдельных операциях.
        """
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_requests = 0
        logger.debug("Usage stats reset")

    def _update_stats(self, prompt: str, response: str) -> None:
        """
        Обновление статистики токенов.
        Приблизительная оценка на основе длины текста (1 токен ≈ 4 символа для русского).

        Args:
            prompt: Текст промпта
            response: Текст ответа модели
        """
        # Приблизительная оценка токенов
        # Для более точного подсчета нужен токенизатор модели
        prompt_tokens = len(prompt) // 4
        completion_tokens = len(response) // 4

        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_requests += 1

        logger.debug(
            f"Stats updated: +{prompt_tokens} prompt tokens, "
            f"+{completion_tokens} completion tokens"
        )

    def _parse_json_from_text(self, text: str) -> Union[Dict, List[Dict]]:
        """
        Извлечение и парсинг JSON из текста.
        Модель может вернуть JSON в Markdown блоках (``````) или с комментариями.

        Args:
            text: Сырой текст ответа от модели

        Returns:
            Распарсенный JSON объект

        Raises:
            json.JSONDecodeError: Если JSON невалиден
        """
        # Удаление возможных Markdown блоков
        cleaned_text = text.strip()

        # Паттерн для извлечения JSON из markdown блока
        json_block_pattern = r'``````'
        json_match = re.search(json_block_pattern, cleaned_text)

        if json_match:
            cleaned_text = json_match.group(1).strip()
            logger.debug("Extracted JSON from markdown block")

        # Удаление возможных комментариев (// ... или /* ... */)
        cleaned_text = re.sub(r'//.*$', '', cleaned_text, flags=re.MULTILINE)
        cleaned_text = re.sub(r'/\*.*?\*/', '', cleaned_text, flags=re.DOTALL)

        # Попытка парсинга
        try:
            parsed = json.loads(cleaned_text)
            return parsed
        except json.JSONDecodeError:
            # Попытка найти JSON в тексте по фигурным/квадратным скобкам
            json_pattern = r'(\{[\s\S]*\}|\[[\s\S]*\])'
            potential_json = re.search(json_pattern, cleaned_text)

            if potential_json:
                return json.loads(potential_json.group(1))
            else:
                raise

    def _enhance_json_prompt(self, original_prompt: str) -> str:
        """
        Улучшение промпта для повторной попытки генерации валидного JSON.

        Args:
            original_prompt: Оригинальный промпт

        Returns:
            Улучшенный промпт с дополнительными инструкциями
        """
        enhancement = (
            "\n\nКРИТИЧЕСКИ ВАЖНО: Верни ТОЛЬКО валидный JSON без комментариев, "
            "без дополнительного текста, без markdown разметки. "
            "Проверь все запятые и кавычки."
        )

        return original_prompt + enhancement


# Вспомогательная функция для создания клиента из конфига
def create_client_from_config(config: dict, credentials: dict) -> GigaChatClient:
    """
    Фабричная функция для создания GigaChatClient из конфигурации.

    Args:
        config: Словарь с настройками из config.json
        credentials: Словарь с секретными ключами

    Returns:
        Инициализированный GigaChatClient
    """
    llm_settings = config.get("llm_settings", {})

    return GigaChatClient(
        credentials=credentials,
        model=llm_settings.get("model", "GigaChat"),
        temperature=llm_settings.get("temperature", 0.7),
        timeout=llm_settings.get("timeout", 30),
        verify_ssl_certs=llm_settings.get("verify_ssl_certs", False)
    )
