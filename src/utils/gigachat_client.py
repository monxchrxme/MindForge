"""
Обёртка для работы с GigaChat через LangChain
"""

from langchain_gigachat import GigaChat
from langchain_core.messages import HumanMessage, SystemMessage
from typing import Type, Any
import json
import logging
import re

logger = logging.getLogger(__name__)


def create_gigachat_parser_client():
    """Создание клиента GigaChat для Parser Agent"""
    import os
    return GigaChatClient(credentials=os.getenv("GIGACHAT_CREDENTIALS"))


def create_gigachat_quiz_client():
    """Создание клиента GigaChat для Quiz Agent"""
    import os
    return GigaChatClient(credentials=os.getenv("GIGACHAT_CREDENTIALS"))


class GigaChatClient:
    """
    Обёртка для GigaChat API с поддержкой structured output
    """

    def __init__(self, credentials: str, model: str = "GigaChat", temperature: float = 0.7):
        """
        Args:
            credentials: API credentials для GigaChat
            model: название модели
            temperature: температура генерации
        """
        self.credentials = credentials
        self.model = model
        self.default_temperature = temperature

        # Создаем клиент GigaChat через LangChain
        self.llm = GigaChat(
            credentials=credentials,
            model=model,
            verify_ssl_certs=False,
            temperature=temperature
        )

        logger.info(f"GigaChatClient инициализирован (model={model})")

    def generate(self, prompt: str, temperature: float = None) -> str:
        """
        Генерация текста через GigaChat

        Args:
            prompt: входной промпт
            temperature: температура генерации (опционально)

        Returns:
            сгенерированный текст
        """
        try:
            # Используем температуру из параметра или дефолтную
            temp = temperature if temperature is not None else self.default_temperature

            # Создаем временный клиент с нужной температурой если она отличается
            if temp != self.default_temperature:
                llm = GigaChat(
                    credentials=self.credentials,
                    model=self.model,
                    verify_ssl_certs=False,
                    temperature=temp
                )
            else:
                llm = self.llm

            # Вызов через LangChain invoke
            message = HumanMessage(content=prompt)
            response = llm.invoke([message])

            result = response.content
            logger.debug(f"Генерация завершена: {len(result)} символов")

            return result

        except Exception as e:
            logger.error(f"Ошибка генерации: {e}")
            raise

    def generate_structured(
        self,
        prompt: str,
        response_model: Type[Any],
        temperature: float = None
    ) -> Any:
        """
        Генерация со структурированным выводом (Pydantic model)

        Args:
            prompt: входной промпт
            response_model: Pydantic модель для парсинга
            temperature: температура генерации

        Returns:
            экземпляр response_model
        """
        temp = temperature if temperature is not None else self.default_temperature

        # Получаем JSON схему модели
        schema = response_model.schema()
        schema_str = json.dumps(schema, indent=2, ensure_ascii=False)

        # Дополняем промпт инструкцией для JSON
        json_prompt = f"""{prompt}

КРИТИЧЕСКИ ВАЖНО: Верни результат СТРОГО в формате JSON без дополнительного текста.

JSON Schema:
{schema_str}

Твой ответ (только валидный JSON):"""

        try:
            # Генерация текста
            response_text = self.generate(json_prompt, temperature=temp)
            logger.debug(f"Ответ LLM: {response_text[:300]}...")

            # Извлечение JSON из ответа
            json_str = self._extract_json(response_text)

            if not json_str:
                raise ValueError("JSON не найден в ответе LLM")

            logger.debug(f"Извлечен JSON: {json_str[:300]}...")

            # Парсинг в Pydantic модель
            result = response_model.parse_raw(json_str)
            logger.debug(f"✓ Structured output успешно распарсен: {type(result).__name__}")

            return result

        except json.JSONDecodeError as e:
            logger.error(f"Ошибка парсинга JSON: {e}")
            logger.error(f"Проблемный JSON: {json_str if 'json_str' in locals() else 'N/A'}")
            raise
        except Exception as e:
            logger.error(f"Ошибка structured generation: {e}")
            if 'response_text' in locals():
                logger.error(f"Ответ LLM: {response_text[:500]}")
            raise

    def _extract_json(self, text: str) -> str:
        """
        Извлечение JSON объекта из текста

        Args:
            text: текст с JSON

        Returns:
            JSON строка
        """
        # Вариант 1: JSON в блоке кода ``````
        json_code_pattern = r'``````'
        match = re.search(json_code_pattern, text, re.DOTALL)
        if match:
            logger.debug("JSON найден в блоке ``````")
            return match.group(1).strip()

        # Вариант 2: JSON в блоке кода ``````
        code_pattern = r'``````'
        match = re.search(code_pattern, text, re.DOTALL)
        if match:
            logger.debug("JSON найден в блоке ```")
            return match.group(1).strip()

        # Вариант 3: Поиск первого { до последнего }
        start = text.find('{')
        end = text.rfind('}')

        if start != -1 and end != -1 and end > start:
            json_str = text[start:end + 1]
            logger.debug("JSON найден между { и }")
            return json_str.strip()

        # Вариант 4: Весь текст может быть JSON
        text_stripped = text.strip()
        if text_stripped.startswith('{') and text_stripped.endswith('}'):
            logger.debug("Весь текст является JSON")
            return text_stripped

        logger.warning("JSON не найден, возвращаем исходный текст")
        return text

    def chat(self, message: Any, temperature: float = None) -> Any:
        """
        Совместимость с существующим кодом
        Обёртка над invoke для работы с сообщениями
        """
        temp = temperature if temperature is not None else self.default_temperature

        if temp != self.default_temperature:
            llm = GigaChat(
                credentials=self.credentials,
                model=self.model,
                verify_ssl_certs=False,
                temperature=temp
            )
        else:
            llm = self.llm

        return llm.invoke([message])
