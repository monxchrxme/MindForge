"""
ExplainAgent — Агент объяснения ошибок и создания мнемонических образов.

Входящая данные (от Orchestrator):
- user_answer: str — неправильный ответ, который дал пользователь
- correct_answer: str — правильный ответ
- question: str — текст вопроса для контекста
- concept: dict — словарь с информацией о концепте {"term": "...", "definition": "..."}

Выходящие данные:
- Dict с полями:
  - "status": "success" | "error"
  - "explanation": str — текстовое объяснение (2-3 предложения)
  - "mnemonic_image": str — описание визуального образа для запоминания
  - "message": str — (если status=="error") описание ошибки
"""

import json
import logging
from typing import Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class ExplainAgent:
    """
    Агент тьютора. Объясняет ошибки пользователя и помогает запомнить правильный ответ
    через мнемонические образы (метод дворца памяти).

    Не хранит состояние — это stateless компонент.
    """

    def __init__(self, gigachat_client, config: Optional[Dict] = None):
        """
        Инициализация ExplainAgent.

        Args:
            gigachat_client: Экземпляр GigaChatClient для обращения к LLM
            config: Dict с настройками (опционально):
                - "explanation_language": "russian" | "english" (по умолчанию "russian")
                - "mnemonic_style": "vivid" | "absurd" | "creative" (стиль образов)
                - "response_timeout": int (в секундах, по умолчанию 30)
        """
        self.gigachat_client = gigachat_client
        self.config = config or {}
        self.language = self.config.get("explanation_language", "russian")
        self.mnemonic_style = self.config.get("mnemonic_style", "absurd")
        self.timeout = self.config.get("response_timeout", 30)

    def explain_error(
            self,
            user_answer: str,
            correct_answer: str,
            question: str,
            concept: Optional[Dict] = None
    ) -> Dict[str, Union[str, bool]]:
        """
        Основной публичный метод. Генерирует объяснение ошибки и мнемонический образ.

        Args:
            user_answer: str — неправильный ответ пользователя (то, что он выбрал)
            correct_answer: str — правильный ответ
            question: str — текст вопроса для контекста
            concept: Optional[Dict] — информация о концепте:
                {
                    "term": "название_концепта",
                    "definition": "определение",
                    "importance": "high" | "medium" | "low"  # опционально
                }

        Returns:
            Dict с форматом ответа:
            {
                "status": "success" | "error",
                "explanation": "Текстовое объяснение...",  # только если status=="success"
                "mnemonic_image": "Описание образа...",     # только если status=="success"
                "message": "Описание ошибки"                # только если status=="error"
            }

        Примечания:
            - Метод НЕ хранит состояние — каждый вызов независим
            - Обращается к LLM для генерации объяснения
            - Возвращает структурированный JSON
        """
        try:
            # Валидация входных данных
            validation_result = self._validate_input(
                user_answer, correct_answer, question
            )
            if not validation_result["valid"]:
                return {
                    "status": "error",
                    "message": validation_result["error"]
                }

            # Построение промпта
            prompt = self._build_prompt(
                user_answer=user_answer,
                correct_answer=correct_answer,
                question=question,
                concept=concept
            )

            # Запрос к GigaChat
            llm_response = self.gigachat_client.generate(
                prompt=prompt,
                #timeout=self.timeout
            )

            # Парсинг ответа LLM
            parsed_response = self._parse_llm_response(llm_response)

            if not parsed_response["success"]:
                return {
                    "status": "error",
                    "message": parsed_response.get("error", "Ошибка парсинга ответа LLM")
                }

            return {
                "status": "success",
                "explanation": parsed_response["explanation"],
                "mnemonic_image": parsed_response["mnemonic_image"]
            }

        except Exception as e:
            logger.error(f"ExplainAgent.explain() ошибка: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "message": f"Внутренняя ошибка агента: {str(e)}"
            }

    def _validate_input(
            self,
            user_answer: str,
            correct_answer: str,
            question: str
    ) -> Dict[str, Union[bool, str]]:
        """
        Проверка входных параметров на корректность.

        Args:
            user_answer: str — ответ пользователя
            correct_answer: str — правильный ответ
            question: str — текст вопроса

        Returns:
            Dict с полями:
            - "valid": bool — валидны ли данные
            - "error": str — (если invalid) описание проблемы
        """
        # Проверка типов
        if not isinstance(user_answer, str) or not user_answer.strip():
            return {
                "valid": False,
                "error": "user_answer должен быть непустой строкой"
            }

        if not isinstance(correct_answer, str) or not correct_answer.strip():
            return {
                "valid": False,
                "error": "correct_answer должен быть непустой строкой"
            }

        if not isinstance(question, str) or not question.strip():
            return {
                "valid": False,
                "error": "question должен быть непустой строкой"
            }

        # Проверка, что ответы не совпадают (иначе это не ошибка)
        if user_answer.strip().lower() == correct_answer.strip().lower():
            return {
                "valid": False,
                "error": "Ответы совпадают — это не ошибка"
            }

        return {"valid": True}

    def _build_prompt(
            self,
            user_answer: str,
            correct_answer: str,
            question: str,
            concept: Optional[Dict] = None
    ) -> str:
        """
        Построение структурированного промпта для LLM.

        Args:
            user_answer: str — неправильный ответ
            correct_answer: str — правильный ответ
            question: str — вопрос
            concept: Optional[Dict] — контекст концепта

        Returns:
            str — готовый промпт для LLM
        """
        concept_info = ""
        if concept and isinstance(concept, dict):
            term = concept.get("term", "")
            definition = concept.get("definition", "")
            if term:
                concept_info = f"\nКонцепт: {term}"
            if definition:
                concept_info += f"\nОпределение: {definition}"

        mnemonic_instruction = {
            "vivid": "ярким, запоминающимся",
            "absurd": "абсурдным, сумасшедшим, веселым",
            "creative": "творческим и оригинальным"
        }.get(self.mnemonic_style, "ярким и запоминающимся")

        language_note = "на русском языке" if self.language == "russian" else "на английском языке"

        prompt = f"""Ты — опытный тьютор, который помогает студентам учиться на их ошибках.

ЗАДАЧА:
1. Объясни кратко (2-3 предложения), почему ответ студента неправильный и какой ответ правильный.
2. Придумай {mnemonic_instruction} образ или ассоциацию, которая поможет студенту запомнить правильный ответ на долго.

КОНТЕКСТ:
Вопрос: {question}
Ответ студента: {user_answer}
Правильный ответ: {correct_answer}{concept_info}

ТРЕБОВАНИЯ К ОТВЕТУ:
- Ответ ТОЛЬКО на {language_note}
- Объяснение: 2-3 предложения, доброжелательное, без критики
- Мнемонический образ: Опиши {mnemonic_instruction} визуальный образ, сюжет или ассоциацию (3-5 предложений)
- ОБЯЗАТЕЛЬНО верни ответ в следующем JSON-формате (без добавления текста до/после):

{{
  "explanation": "Объяснение ошибки здесь...",
  "mnemonic_image": "Описание визуального образа здесь..."
}}
"""
        return prompt

    def _parse_llm_response(self, llm_response: str) -> Dict[str, Union[bool, str]]:
        """
        Парсинг ответа от LLM.

        Ожидаемый формат: JSON с полями explanation и mnemonic_image.
        Если LLM вернет JSON обернутый в Markdown-блоки, метод должен их обработать.

        Args:
            llm_response: str — сырой ответ от LLM

        Returns:
            Dict с полями:
            - "success": bool — успешен ли парсинг
            - "explanation": str — (если success) текст объяснения
            - "mnemonic_image": str — (если success) текст образа
            - "error": str — (если not success) описание ошибки
        """
        try:
            # Очистка от Markdown-обертки (если есть)
            cleaned_response = self._extract_json_from_markdown(llm_response)

            # Парсинг JSON
            data = json.loads(cleaned_response)

            # Валидация полей
            if "explanation" not in data or "mnemonic_image" not in data:
                missing = []
                if "explanation" not in data:
                    missing.append("explanation")
                if "mnemonic_image" not in data:
                    missing.append("mnemonic_image")

                return {
                    "success": False,
                    "error": f"Недостающие поля в ответе LLM: {', '.join(missing)}"
                }

            # Проверка, что значения — это строки
            explanation = data["explanation"]
            mnemonic_image = data["mnemonic_image"]

            if not isinstance(explanation, str) or not explanation.strip():
                return {
                    "success": False,
                    "error": "Поле 'explanation' не содержит непустую строку"
                }

            if not isinstance(mnemonic_image, str) or not mnemonic_image.strip():
                return {
                    "success": False,
                    "error": "Поле 'mnemonic_image' не содержит непустую строку"
                }

            return {
                "success": True,
                "explanation": explanation.strip(),
                "mnemonic_image": mnemonic_image.strip()
            }

        except json.JSONDecodeError as e:
            logger.warning(f"JSON парсинг ошибка: {str(e)}. Ответ: {llm_response}")
            return {
                "success": False,
                "error": f"Невалидный JSON в ответе LLM: {str(e)}"
            }
        except Exception as e:
            logger.error(f"Парсинг ответа ошибка: {str(e)}")
            return {
                "success": False,
                "error": f"Ошибка при парсинге ответа: {str(e)}"
            }

    def _extract_json_from_markdown(self, text: str) -> str:
        """
        Извлечение JSON из Markdown-блоков.

        LLM может вернуть JSON обернутый в:
        ```json
        { ... }
        ```
        или
        ``` json
        { ... }
        ```

        Args:
            text: str — текст с возможной Markdown-обёрткой

        Returns:
            str — чистый JSON
        """
        text = text.strip()

        # Проверка на Markdown-блоки
        if text.startswith("```"):
            # Найти первый перевод строки
            lines = text.split("\n")

            # Найти начало JSON (после открывающей скобки или первой непустой строки после ```)
            start_idx = 0
            for i, line in enumerate(lines):
                if line.startswith("```"):
                    start_idx = i + 1
                    break

            # Найти конец JSON (перед закрывающей ```)
            end_idx = len(lines)
            for i in range(start_idx, len(lines)):
                if lines[i].startswith("```"):
                    end_idx = i
                    break

            # Собрать JSON
            json_lines = lines[start_idx:end_idx]
            text = "\n".join(json_lines).strip()

        return text

    def explain_batch(
            self,
            errors: List[Dict[str, str]]
    ) -> List[Dict[str, Union[str, bool]]]:
        """
        Пакетное объяснение нескольких ошибок.

        Используется, когда пользователь прошел квиз и получил несколько ошибок —
        система может объяснить их все подряд.

        Args:
            errors: List[Dict] — список ошибок, каждая с полями:
                {
                    "user_answer": "...",
                    "correct_answer": "...",
                    "question": "...",
                    "concept": {...}  # опционально
                }

        Returns:
            List[Dict] — список результатов (в том же порядке)
        """
        if not isinstance(errors, list):
            logger.warning("explain_batch: errors должен быть List, получен %s", type(errors))
            return [
                {
                    "status": "error",
                    "message": "errors должен быть List"
                }
            ]

        results = []
        for i, error_data in enumerate(errors):
            try:
                result = self.explain(
                    user_answer=error_data.get("user_answer", ""),
                    correct_answer=error_data.get("correct_answer", ""),
                    question=error_data.get("question", ""),
                    concept=error_data.get("concept")
                )
                results.append(result)
            except Exception as e:
                logger.error(
                    f"explain_batch: ошибка при обработке ошибки #{i}: {str(e)}",
                    exc_info=True
                )
                results.append({
                    "status": "error",
                    "message": f"Ошибка при обработке: {str(e)}"
                })

        return results

    def get_config(self) -> Dict:
        """
        Возвращает текущую конфигурацию агента.

        Returns:
            Dict с текущими настройками
        """
        return {
            "language": self.language,
            "mnemonic_style": self.mnemonic_style,
            "timeout": self.timeout
        }

    def set_config(self, **kwargs) -> None:
        """
        Обновление конфигурации агента.

        Args:
            **kwargs: Настройки для обновления:
                - language: "russian" | "english"
                - mnemonic_style: "vivid" | "absurd" | "creative"
                - response_timeout: int
        """
        valid_keys = {"language", "mnemonic_style", "response_timeout"}
        for key, value in kwargs.items():
            if key not in valid_keys:
                logger.warning(f"Неизвестный параметр конфигурации: {key}")
                continue

            if key == "language":
                if value in ["russian", "english"]:
                    self.language = value
                else:
                    logger.warning(f"Неподдерживаемый язык: {value}")

            elif key == "mnemonic_style":
                if value in ["vivid", "absurd", "creative"]:
                    self.mnemonic_style = value
                else:
                    logger.warning(f"Неподдерживаемый стиль мнемоники: {value}")

            elif key == "response_timeout":
                if isinstance(value, int) and value > 0:
                    self.timeout = value
                else:
                    logger.warning(f"Таймаут должен быть положительным числом: {value}")