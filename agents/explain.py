"""
ExplainAgent - Полностью автономный агент объяснения ошибок.

АРХИТЕКТУРА АГЕНТА:
1. Анализирует content_type самостоятельно (DECISION MAKING)
2. Выбирает одну из 3 стратегий объяснения (STRATEGY SELECTION)
3. Генерирует адаптивный промпт (ADAPTIVE GENERATION)
4. Валидирует результат (QUALITY CHECK)

content_type определяется исключительно внутри самого агента.
"""

import logging
from typing import Dict, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class ContentTypeEnum(Enum):
    """Типы контента (определяются самим агентом)"""
    THEORY = "theory"   # Обычный текст, определения, факты
    CODE = "code"       # Программный код, сниппеты
    SHORT = "short"     # Короткие заметки (zettelkasten)


class ExplainAgent:
    """
    Полностью автономный агент тьютора с адаптивной стратегией объяснения.

    Ключевые особенности агента:
    1. Самостоятельно определяет тип контента (DECISION MAKING - сам, без помощи)
    2. Принимает решение о стратегии на основе анализа контента
    3. выбирает стратегию (3 разных подхода) в зависимости от типа
    4. Генерируют адаптивные промты для каждого типа
    5. Валидирует результаты перед возвратом

    Не имеет внешних зависимостей при определении типа контента.
    """

    def __init__(self, client):
        """
        Инициализация полностью автономного ExplainAgent.

        Args:
            client: Экземпляр GigaChatClient для обращения к LLM через LangChain.
                   Ожидается, что client имеет методы:
                   - generate(prompt: str) -> str
                   - generate_json(prompt: str) -> dict
        """
        self.client = client
        logger.info("ExplainAgent initialized (FULLY AUTONOMOUS MODE)")

    def explain_error(
        self,
        question_text: str,
        user_ans: str,
        correct_ans: str,
    ) -> Dict[str, str]:
        """
        ОСНОВНОЙ ПУБЛИЧНЫЙ МЕТОД: объяснение ошибки.

        АГЕНТСКИЙ ПРОЦЕСС (только 3 входных параметра - ничего более не требуется):
        1. Валидирует входные данные
        2. Анализирует content_type (DECISION MAKING)
        3. Выбирает стратегию объяснения (STRATEGY SELECTION)
        4. Генерирует адаптивный промпт
        5. Вызывает LLM с промптом
        6. Валидирует результат (QUALITY CHECK)

        Args:
            question_text: str - текст вопроса для контекста
            user_ans: str - неправильный ответ пользователя
            correct_ans: str - правильный ответ

        Returns:
            Dict с полями:
            - "explanation_text": str - объяснение ошибки
            - "memory_palace_image": str - описание визуального образа
            - "strategy_used": str - использованная стратегия

        Raises:
            ValueError: При ошибках валидации или парсинга
            Exception: При ошибках API
        """
        try:
            logger.info("=" * 70)
            logger.info("🤖 AUTONOMOUS AGENT DECISION PROCESS STARTED")
            logger.info("=" * 70)

            # ===== ЭТАП 1: ВАЛИДАЦИЯ =====
            validation_error = self._validate_input(
                question_text, user_ans, correct_ans
            )
            if validation_error:
                logger.warning(f"Validation error: {validation_error}")
                raise ValueError(validation_error)

            # ===== ЭТАП 2: САМОСТОЯТЕЛЬНЫЙ АНАЛИЗ КОНТЕНТА (DECISION MAKING) =====
            logger.info("\n📊 STEP 1: Self-analyzing content type...")
            analyzed_type = self._analyze_content_type(
                question_text, user_ans, correct_ans
            )
            logger.info(f"✓ Self-determined content type: {analyzed_type.value.upper()}")

            # ===== ЭТАП 3: ВЫБОР СТРАТЕГИИ (STRATEGY SELECTION) =====
            logger.info("\n🎯 STEP 2: Selecting explanation strategy...")
            strategy = self._select_strategy(analyzed_type)
            logger.info(f"✓ Selected strategy: {strategy['name']}")

            # ===== ЭТАП 4: ПОСТРОЕНИЕ АДАПТИВНОГО ПРОМПТА =====
            logger.info("\n📝 STEP 3: Building adaptive prompt...")
            prompt = self._build_adaptive_prompt(
                question_text=question_text,
                user_ans=user_ans,
                correct_ans=correct_ans,
                content_type=analyzed_type,
                strategy=strategy,
            )
            logger.debug(f"Prompt length: {len(prompt)} chars")

            # ===== ЭТАП 5: ГЕНЕРАЦИЯ (LLM CALL) =====
            logger.info("\n🔄 STEP 4: Calling LLM for explanation generation...")
            response_data = self.client.generate_json(prompt)

            # ===== ЭТАП 6: ВАЛИДАЦИЯ РЕЗУЛЬТАТА (QUALITY CHECK) =====
            logger.info("\n✓ STEP 5: Validating response structure...")
            if not isinstance(response_data, dict):
                raise ValueError(
                    f"Expected dict response, got {type(response_data)}"
                )

            explanation_text = response_data.get("explanation", "")
            memory_palace_image = response_data.get("mnemonic_image", "")

            if not explanation_text or not memory_palace_image:
                logger.error(f"Missing fields in response: {response_data.keys()}")
                raise ValueError(
                    "Response missing 'explanation' or 'mnemonic_image'"
                )

            # ===== ФОРМИРОВАНИЕ РЕЗУЛЬТАТА =====
            result = {
                "explanation_text": explanation_text.strip(),
                "memory_palace_image": memory_palace_image.strip(),
                "strategy_used": strategy["name"],
            }

            logger.info("=" * 70)
            logger.info("✅ AUTONOMOUS AGENT COMPLETED SUCCESSFULLY")
            logger.info("=" * 70)
            logger.info(
                f"Explanation: {len(result['explanation_text'])} chars | "
                f"Memory palace: {len(result['memory_palace_image'])} chars"
            )

            return result

        except Exception as e:
            logger.error(f"Error in explain_error(): {str(e)}", exc_info=True)
            raise

    # ============================================================================
    # АГЕНТСКИЕ МЕТОДЫ: DECISION MAKING, STRATEGY SELECTION, ADAPTATION
    # ============================================================================

    def _analyze_content_type(
        self,
        question_text: str,
        user_ans: str,
        correct_ans: str,
    ) -> ContentTypeEnum:
        """
        АГЕНТСКИЙ МЕТОД #1: Самостоятельный анализ типа контента.

        Агент анализирует вопрос для определения типа (это делает агента умным LLM агентом).

        Не зависит ни от Orchestrator, ни от каких-либо внешних параметров.

        Args:
            question_text: str - текст вопроса
            user_ans: str - неправильный ответ
            correct_ans: str - правильный ответ

        Returns:
            ContentTypeEnum: Определённый тип контента
        """
        logger.info(" Self-analyzing content type...")

        combined_text = (
            f"Вопрос: {question_text}\n"
            f"Ответ пользователя: {user_ans}\n"
            f"Правильный ответ: {correct_ans}"
        )

        # Расширенный список маркеров кода для точного определения
        code_markers = [
            "код", "function", "def ", "class ", "import", "return",
            "var ", "const ", "print(", "{", "}", "==", "!=",
            "=>", "async", "await", "try", "catch", "if ", "for ",
            "while ", "lambda", "[", "]", "method", "constructor",
            "super", "this", "self", "new ", "Array", "string", "int ",
            "float", "bool", "null", "undefined", "typeof", "instanceof"
        ]

        # Определение типа на основе анализа маркеров
        if any(keyword in combined_text.lower() for keyword in code_markers):
            logger.info(" ✓ CODE detected")
            return ContentTypeEnum.CODE

        # Для коротких текстов
        if len(combined_text.split()) < 20:
            logger.info(" ✓ SHORT text detected")
            return ContentTypeEnum.SHORT

        # Дефолт для всего остального
        logger.info(" ✓ THEORY (default)")
        return ContentTypeEnum.THEORY

    def _select_strategy(self, content_type: ContentTypeEnum) -> Dict:
        """
        АГЕНТСКИЙ МЕТОД #2: Выбор стратегии объяснения.

        На основе определённого типа контента агент выбирает
        одну из 3 адаптивных стратегий объяснения.

        Args:
            content_type: ContentTypeEnum - определённый тип контента

        Returns:
            Dict: Информация о выбранной стратегии
        """
        strategies = {
            ContentTypeEnum.CODE: {
                "name": "code_analysis_strategy",
                "description": "Технический анализ кода",
            },
            ContentTypeEnum.THEORY: {
                "name": "theory_deep_dive_strategy",
                "description": "Глубокий анализ теории",
            },
            ContentTypeEnum.SHORT: {
                "name": "short_concise_strategy",
                "description": "Краткое объяснение",
            },
        }
        selected = strategies.get(content_type, strategies[ContentTypeEnum.THEORY])
        logger.info(f" ✓ {selected['description']}")
        return selected

    def _build_adaptive_prompt(
        self,
        question_text: str,
        user_ans: str,
        correct_ans: str,
        content_type: ContentTypeEnum,
        strategy: Dict,
    ) -> str:
        """
        АГЕНТСКИЙ МЕТОД #3: Построение адаптивного промпта.

        Агент генерирует уникальный промпт для каждого типа контента,
        учитывая специфику выбранной стратегии.

        ВАЖНО:
        - "Пояснение" содержит ТОЛЬКО логическое объяснение
        - "Мнемонический образ" содержит визуальные образы и ассоциации

        КРИТИЧЕСКИ ВАЖНО: В JSON всегда используй точные значения ключей.
        Убедись, что выходной JSON имеет ПРАВИЛЬНО ОТФОРМАТИРОВАННЫЕ значения
        с ПРОБЕЛАМИ между словами.

        Args:
            question_text: str - текст вопроса
            user_ans: str - неправильный ответ
            correct_ans: str - правильный ответ
            content_type: ContentTypeEnum - тип контента
            strategy: Dict - информация о стратегии

        Returns:
            str: Адаптивный промпт для LLM
        """
        base = "Ты - опытный тьютор.\n\n"

        if content_type == ContentTypeEnum.THEORY:
            return (
                f"{base}"
                f"КОНТЕКСТ: Учебная заметка.\n"
                f"ЗАДАЧА: Объясни почему ответ неправильный и почему правильный.\n"
                f"Вопрос: {question_text}\n"
                f"Ответ пользователя: {user_ans}\n"
                f"Правильный ответ: {correct_ans}\n\n"
                f"ВАЖНО:\n"
                f"1. Пояснение: 4-5 предложений с логическим анализом и взаимосвязями теории, при надобности расписывай на большее количество предложений. "
                f"БЕЗ образов и ассоциаций для запоминания. "
                f"Текст должен быть чистым, с ПРОБЕЛАМИ между всеми словами.\n"
                f"2. Мнемонический образ: 3-4 предложения с визуальными образами, "
                f"метафорами и ассоциациями для запоминания. "
                f"Сделай образ ярким, запоминающимся и связанным с правильным ответом. "
                f"Убедись, что текст полностью РАЗБОРЧИВ с ПРОБЕЛАМИ между словами.\n\n"
                f"Возврати JSON:\n"
                f'{{"explanation": "...", "mnemonic_image": "..."}}'
            )
        elif content_type == ContentTypeEnum.CODE:
            return (
                f"{base}"
                f"КОНТЕКСТ: Кодовая заметка.\n"
                f"ЗАДАЧА: Разбери синтаксис и логику правильного ответа.\n"
                f"Вопрос: {question_text}\n"
                f"Ответ пользователя: {user_ans}\n"
                f"Правильный ответ: {correct_ans}\n\n"
                f"ВАЖНО:\n"
                f"1. Пояснение: 3-7 предложений с техническим анализом синтаксиса и логики кода, при надобности расписывай на большее количество предложений. "
                f"Объясни, почему неправильный ответ ошибочен и как работает правильный. "
                f"БЕЗ ассоциаций и образов. "
                f"ОБЯЗАТЕЛЬНО разделяй слова ПРОБЕЛАМИ, текст должен быть четким и понятным.\n"
                f"2. Мнемонический образ: 2-3 предложения с метафорами, аналогиями "
                f"или образными сравнениями программирования для лучшего запоминания. "
                f"Связь с реальными примерами приветствуется. "
                f"Проверь, что все слова разделены ПРОБЕЛАМИ - без слитного текста.\n\n"
                f"Возврати JSON:\n"
                f'{{"explanation": "...", "mnemonic_image": "..."}}'
            )
        else:  # SHORT
            return (
                f"{base}"
                f"КОНТЕКСТ: Короткая заметка.\n"
                f"ЗАДАЧА: Кратко объясни ошибку.\n"
                f"Вопрос: {question_text}\n"
                f"Ответ пользователя: {user_ans}\n"
                f"Правильный ответ: {correct_ans}\n\n"
                f"ВАЖНО:\n"
                f"1. Пояснение: РОВНО 2-3 предложения. Дай логическое объяснение без образов. "
                f"Четко и ясно. ГАРАНТИРУЙ, что между словами ВСЕГДА есть ПРОБЕЛЫ.\n"
                f"2. Мнемонический образ: 1-2 предложения. Простой, смешной или ироничный образ "
                f"для запоминания. Творческий подход приветствуется. "
                f"Проверь разборчивость - слова должны быть РАЗДЕЛЕНЫ ПРОБЕЛАМИ.\n\n"
                f"Возврати JSON:\n"
                f'{{"explanation": "...", "mnemonic_image": "..."}}'
            )

    def _validate_input(
        self, question_text: str, user_ans: str, correct_ans: str
    ) -> Optional[str]:
        """
        Вспомогательный метод: Валидация входных параметров.

        Args:
            question_text: str - текст вопроса
            user_ans: str - ответ пользователя
            correct_ans: str - правильный ответ

        Returns:
            Optional[str]: Сообщение об ошибке или None если всё хорошо
        """
        if not isinstance(question_text, str) or not question_text.strip():
            return "question_text должен быть непустой строкой"
        if not isinstance(user_ans, str) or not user_ans.strip():
            return "user_ans должен быть непустой строкой"
        if not isinstance(correct_ans, str) or not correct_ans.strip():
            return "correct_ans должен быть непустой строкой"
        if user_ans.strip().lower() == correct_ans.strip().lower():
            return "Ответы совпадают"
        return None

    def explain_batch(self, errors: list) -> list:
        """
        Пакетное объяснение нескольких ошибок.

        Args:
            errors: List[Dict] - список ошибок с полями:
                   - question_text: str
                   - user_ans: str
                   - correct_ans: str

        Returns:
            List[Dict]: Список результатов объяснений
        """
        if not isinstance(errors, list):
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
                logger.error(f"Error processing error #{i}: {str(e)}")
                results.append({
                    "explanation_text": f"Ошибка: {str(e)}",
                    "memory_palace_image": "",
                    "strategy_used": "error_handler",
                })
        return results