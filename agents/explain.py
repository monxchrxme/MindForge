#TODO Убрать концепты
"""
ExplainAgent — Агент объяснения ошибок и создания мнемонических образов.

Входящие данные (от Orchestrator):
- question_text: str — текст вопроса
- user_ans: str — неправильный ответ пользователя
- correct_ans: str — правильный ответ
- concept_def: str — определение концепта для контекста

Выходящие данные:
- Dict с полями:
  - "explanation_text": str — текстовое объяснение
  - "memory_palace_image": str — описание визуального образа для запоминания
"""

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class ExplainAgent:
    """
    Агент тьютора. Объясняет ошибки пользователя и помогает запомнить правильный ответ
    через мнемонические образы (метод дворца памяти).

    Использует LangChain-GigaChat как основной API для генерации объяснений.
    Это stateless компонент — не хранит состояние между вызовами.
    """

    def __init__(self, client):
        """
        Инициализация ExplainAgent.

        Args:
            client: Экземпляр GigaChatClient для обращения к LLM через LangChain.
                   Ожидается, что client имеет методы:
                   - generate(prompt: str) -> str
                   - generate_json(prompt: str) -> dict
        """
        self.client = client
        logger.info("ExplainAgent initialized")

    def explain_error(
        self,
        question_text: str,
        user_ans: str,
        correct_ans: str,
    ) -> Dict[str, str]:
        """
        Основной публичный метод. Генерирует объяснение ошибки и мнемонический образ.

        ВАЖНО: Сигнатура совместима с вызовом из OrchestratorAgent.submit_answer():
            result = self.explainer.explain_error(
                question_text=question_text,
                user_ans=user_answer,
                correct_ans=correct_answer,
                concept_def=concept_definition
            )

        Args:
            question_text: str — текст вопроса для контекста
            user_ans: str — неправильный ответ пользователя
            correct_ans: str — правильный ответ

        Returns:
            Dict с полями:
                - "explanation_text": str — объяснение ошибки (2-3 предложения)
                - "memory_palace_image": str — описание визуального образа (3-5 предложений)

        Raises:
            Exception: При ошибках API или парсинга
        """
        try:
            # Валидация входных данных
            validation_error = self._validate_input(
                question_text, user_ans, correct_ans
            )
            if validation_error:
                logger.warning(f"Validation error: {validation_error}")
                raise ValueError(validation_error)

            # Построение промпта
            prompt = self._build_prompt(
                question_text=question_text,
                user_ans=user_ans,
                correct_ans=correct_ans,
            )

            logger.debug("Sending request to GigaChat...")

            # Запрос к GigaChat через LangChain (получаем JSON напрямую)
            response_data = self.client.generate_json(prompt)

            # Валидация структуры ответа
            if not isinstance(response_data, dict):
                raise ValueError(f"Expected dict response, got {type(response_data)}")

            # Извлечение полей (новая сигнатура для совместимости с Orchestrator)
            explanation_text = response_data.get("explanation", "")
            memory_palace_image = response_data.get("mnemonic_image", "")

            if not explanation_text or not memory_palace_image:
                logger.error(f"Missing fields in response: {response_data.keys()}")
                raise ValueError("Response missing 'explanation' or 'mnemonic_image'")

            result = {
                "explanation_text": explanation_text.strip(),
                "memory_palace_image": memory_palace_image.strip()
            }

            logger.info("Explanation generated successfully")
            return result

        except Exception as e:
            logger.error(f"Error in explain_error(): {str(e)}", exc_info=True)
            raise

    def _validate_input(
        self,
        question_text: str,
        user_ans: str,
        correct_ans: str,
    ) -> Optional[str]:
        """
        Проверка входных параметров на корректность.

        Args:
            question_text: str
            user_ans: str
            correct_ans: str

        Returns:
            str — сообщение об ошибке, или None если валидно
        """
        # Проверка типов и пустоты
        if not isinstance(question_text, str) or not question_text.strip():
            return "question_text должен быть непустой строкой"

        if not isinstance(user_ans, str) or not user_ans.strip():
            return "user_ans должен быть непустой строкой"

        if not isinstance(correct_ans, str) or not correct_ans.strip():
            return "correct_ans должен быть непустой строкой"

        # Проверка, что ответы не совпадают
        if user_ans.strip().lower() == correct_ans.strip().lower():
            return "Ответы совпадают — это не ошибка"

        return None

    def _build_prompt(
        self,
        question_text: str,
        user_ans: str,
        correct_ans: str,
    ) -> str:
        """
        Построение структурированного промпта для LLM.

        Args:
            question_text: str
            user_ans: str
            correct_ans: str

        Returns:
            str — готовый промпт для отправки в GigaChat через LangChain
        """
        prompt = f"""Ты — опытный тьютор, который помогает студентам учиться на их ошибках.  

        ЗАДАЧА:  
        1. Объясни кратко, но так, чтобы было понятно (2-3 предложения), почему ответ пользователя неправильный. Там, где надо, используй термины, чтобы они были уместны.  
        2. Придумай абсурдный, веселый и запоминающийся визуальный образ или ассоциацию для правильного ответа, но используй смешной и абсурдный только в мнемоническом образе.  

        КОНТЕКСТ:  
        - Вопрос: {question_text}  
        - Ответ студента (неправильно): {user_ans}  
        - Правильный ответ: {correct_ans}  

        ТРЕБОВАНИЯ К ОТВЕТУ:  
        1. Объяснение: 2-3 предложения; технически верное, объясни почему ответ к этой задаче именно такой, попытайся пользоваться сложными терминами там, где это надо, но объяснение должно быть понятно даже новичку. Делай ответ без критики.  
        2. Мнемонический образ: Опиши смешной, забавный и запоминающийся визуальный образ (3-5 предложений), который помогает запомнить правильный ответ.  
        3. Язык: русский.  
        4. ОБЯЗАТЕЛЬНО верни ответ ТОЛЬКО в следующем JSON-формате, без дополнительного текста:  

        {{  
          "explanation": "Объяснение здесь...",  
          "mnemonic_image": "Описание образа здесь..."}}"""

        return prompt


def explain_batch(
        self,
        errors: list
    ) -> list:
        """
        Пакетное объяснение нескольких ошибок.

        Используется, когда пользователь прошел квиз и получил несколько ошибок.

        Args:
            errors: List[Dict] — список ошибок, каждая с полями:
                {
                  "question_text": "...",
                  "user_ans": "...",
                  "correct_ans": "...",
                  "concept_def": "..."
                }

        Returns:
            List[Dict] — список результатов (в том же порядке)
        """
        if not isinstance(errors, list):
            logger.error(f"explain_batch: errors must be list, got {type(errors)}")
            raise TypeError("errors должен быть List")

        results = []
        for i, error_data in enumerate(errors):
            try:
                result = self.explain_error(
                    question_text=error_data.get("question_text", ""),
                    user_ans=error_data.get("user_ans", ""),
                    correct_ans=error_data.get("correct_ans", ""),
                )
                results.append(result)
            except Exception as e:
                logger.error(
                    f"explain_batch: error processing error #{i}: {str(e)}",
                    exc_info=True
                )
                results.append({
                    "explanation_text": f"Ошибка при обработке: {str(e)}",
                    "memory_palace_image": ""
                })

        return results