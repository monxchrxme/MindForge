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
            verify_ssl_certs: bool = False,
            use_api_for_tokens=True
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
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        last_error = None

        for attempt in range(1, retry_attempts + 1):
            try:
                logger.debug(f"Generating JSON response (attempt {attempt}/{retry_attempts})")

                # 1. Получение сырого текста
                raw_response = self.generate(prompt)

                # ЛОГИРОВАНИЕ ДЛЯ ОТЛАДКИ
                logger.info(f"--- RAW RESPONSE (Attempt {attempt}) ---")
                logger.info(raw_response)
                logger.info("----------------------------------------")

                # 2. Парсинг
                parsed_json = self._parse_json_from_text(raw_response)

                logger.debug(f"JSON parsing successful on attempt {attempt}")
                return parsed_json

            except json.JSONDecodeError as e:
                last_error = e
                logger.warning(f"JSON parsing failed on attempt {attempt}: {e}")
                # Добавляем инструкцию для исправления
                if attempt < retry_attempts:
                    prompt = self._enhance_json_prompt(prompt)
            except Exception as e:
                logger.error(f"Unexpected error in generate_json: {e}")
                raise

        # Если все попытки исчерпаны
        error_msg = f"Failed to parse JSON after {retry_attempts} attempts. Last error: {last_error}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    def _parse_json_from_text(self, text: str) -> Union[Dict, List[Dict]]:
        """
        Устойчивый парсинг JSON из грязного ответа LLM.
        """
        if not text:
            raise json.JSONDecodeError("Empty string", "", 0)

        # 1. Очистка от Markdown (``````)
        cleaned = text.strip()
        # Ищем блок кода `````` (жадный поиск, берем содержимое)
        match = re.search(r"``````", cleaned, re.DOTALL | re.IGNORECASE)
        if match:
            cleaned = match.group(1).strip()

        # 2. Очистка от комментариев (// и /* */)
        # Осторожно: это может удалить ссылки http://, но для JSON структур это редкость
        # Используем более безопасную регулярку, которая не трогает строки внутри кавычек - сложно.
        # Проще: удаляем только комментарии в начале строк или явные блоки
        cleaned = re.sub(r'^\s*//.*$', '', cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r'/\*.*?\*/', '', cleaned, flags=re.DOTALL)

        # 3. Очистка от невидимых символов и BOM
        # Удаляем BOM (Byte Order Mark)
        cleaned = cleaned.replace('\ufeff', '')
        # Заменяем неразрывные пробелы
        cleaned = cleaned.replace('\u00a0', ' ')

        # 4. Экранирование управляющих символов, которые часто забывает LLM
        # В JSON внутри строк запрещены реальные табы и переносы.
        # Но мы не можем просто сделать replace('\t', '\\t') глобально, так как это может быть
        # форматирование самого JSON (отступы), а не значение внутри строки.
        # Однако, json.loads(strict=False) обычно справляется с табами-отступами.

        # Попытка 1: Стандартный парсинг
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Попытка 2: Разрешаем управляющие символы (strict=False)
        try:
            return json.loads(cleaned, strict=False)
        except json.JSONDecodeError:
            pass

        # Попытка 3: Поиск JSON-подобной структуры (если есть лишний текст вокруг)
        # Ищем от первой { или [ до последней } или ]
        try:
            match = re.search(r'(\{.*\}|\[.*\])', cleaned, re.DOTALL)
            if match:
                potential_json = match.group(0)
                return json.loads(potential_json, strict=False)
        except json.JSONDecodeError:
            pass

        # Попытка 4 (Крайняя мера): Замена Python-style значений на JSON
        # Иногда модель пишет None вместо null, True вместо true
        try:
            replacements = {
                'None': 'null',
                'True': 'true',
                'False': 'false',
                "'": '"',  # Замена одинарных кавычек на двойные (опасно, но иногда нужно)
            }
            fixed_text = cleaned
            # Аккуратно меняем только литералы (не внутри слов) - упрощенно
            for k, v in replacements.items():
                fixed_text = fixed_text.replace(k, v)

            return json.loads(fixed_text, strict=False)
        except json.JSONDecodeError as e:
            # Если ничего не помогло, выбрасываем оригинальную ошибку
            # чтобы видеть, что именно не так
            raise e

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

        Считаем токены через официальный токенайзер GigaChat
        (get_num_tokens). При use_api_for_tokens=True под капотом
        используется /tokens/count.
        """
        # 1. Считаем токены за текущий запрос
        try:
            prompt_tokens = self.gigachat.get_num_tokens(prompt)
        except Exception as e:
            logger.warning(f"Token count fallback (prompt): {e}")
            prompt_tokens = max(1, round(len(prompt) / 4.6))

        try:
            completion_tokens = self.gigachat.get_num_tokens(response)
        except Exception as e:
            logger.warning(f"Token count fallback (completion): {e}")
            completion_tokens = max(1, round(len(response) / 4.6))

        current_total = prompt_tokens + completion_tokens

        # 2. Обновляем общую статистику
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_requests += 1

        global_total = self.total_prompt_tokens + self.total_completion_tokens

        # 3. Выводим лог (динамика)
        #p - промт токены, c - сколько нейро выдало
        logger.info(
            f"💰 Token Usage [Req #{self.total_requests}]: "
            f"+{current_total} (P:{prompt_tokens}/C:{completion_tokens}) "
            f"| Total Session: {global_total}"
        )


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
