import logging
from typing import Dict, Optional, Any
from enum import Enum

# Настройка логирования
logger = logging.getLogger(__name__)

class ContentTypeEnum(Enum):
    """
    Типы контента (определяются самим агентом).
    Agent Perception: Категории, которые агент различает в мире.
    """
    THEORY = "theory"   # Обычный текст, определения, факты
    CODE = "code"       # Программный код, сниппеты
    SHORT = "short"     # Короткие заметки (zettelkasten)

class ExplainAgent:
    """
    🤖 ExplainAgent: Автономный агент обучения с обратной связью.

    АРХИТЕКТУРА АГЕНТА:
    1. PERCEPTION (Восприятие): Анализирует входной текст (_analyze_content_type)
    2. PLANNING (Планирование): Выбирает стратегию объяснения (_select_strategy)
    3. ACTION (Действие): Генерирует объяснение (_build_adaptive_prompt + client.generate)
    4. REFLECTION (Рефлексия): Оценивает качество собственной работы (validate_explanation_quality)
    5. CORRECTION (Исправление): Переделывает работу, если качество низкое (Control Loop)
    """

    def __init__(self, client):
        """
        Инициализация агента.

        Args:
            client: LLM-клиент (GigaChat/OpenAI), умеющий делать generate_json.
        """
        self.client = client
        logger.info("🤖 ExplainAgent initialized (AGENCY LEVEL: HIGH - Feedback Loop Enabled)")

    def validate_explanation_quality(
        self,
        explanation: str,
        mnemonic_image: str,
        content_type: ContentTypeEnum
    ) -> dict:
        """
        🧠 AGENT REFLECTION (РЕФЛЕКСИЯ АГЕНТА) - TOOL #1

        Роль: Внутренний критик.
        Что делает: Оценивает качество генерации по объективным метрикам.
        Зачем: Чтобы агент мог принять решение - "хорошо получилось" или "надо переделать".

        Метрики:
        1. Spacing: Проверка на "слипание" слов.
        2. Clarity: Проверка пунктуации и читаемости.
        3. Vividness: Проверка наличия ярких образов в мнемотехнике.
        4. Type Match: Соответствие текста типу контента (код, теория).

        Returns:
            dict: {score, requires_regeneration, details...}
        """
        # 1. Проверка пробелов (защита от технических сбоев LLM)
        words = explanation.split()
        has_spacing = len(words) > (len(explanation) / 5)

        # 2. Проверка ясности (пунктуация)
        clarity_indicators = ['.', '!', '?', ';', '\n']
        has_clarity = any(indicator in explanation for indicator in clarity_indicators)

        # 3. Проверка образности (для мнемотехники)
        vivid_words = [
            'представь', 'вообрази', 'видим', 'образ', 'цвет',
            'форма', 'метафора', 'аналогия', 'сравнение',
            'видеть', 'представляется', 'как', 'словно'
        ]
        is_vivid = any(word in mnemonic_image.lower() for word in vivid_words)

        # 4. Проверка соответствия типу
        if content_type == ContentTypeEnum.THEORY:
            type_match = len(explanation.split('.')) >= 3  # Теория должна быть развернутой
        elif content_type == ContentTypeEnum.CODE:
            code_keywords = ['код', 'функция', 'переменная', 'цикл', 'условие', 'значение', 'return']
            type_match = any(kw in explanation.lower() for kw in code_keywords)
        else:  # SHORT
            type_match = len(explanation.split()) <= 40  # Short должен быть кратким

        # Расчет итогового балла (0.0 - 1.0)
        quality_score = (
            float(has_spacing) +
            float(has_clarity) +
            float(is_vivid) +
            float(type_match)
        ) / 4

        # РЕШЕНИЕ АГЕНТА: Переделывать, если качество ниже 70%
        requires_regeneration = quality_score < 0.7

        logger.info(f"🔍 REFLECTION: Score={quality_score:.2f} | Regen={requires_regeneration}")

        return {
            'quality_score': quality_score,
            'requires_regeneration': requires_regeneration,
            'details': {
                'spacing': has_spacing,
                'clarity': has_clarity,
                'vivid': is_vivid,
                'type_match': type_match
            }
        }

    def explain_error(
        self,
        question_text: str,
        user_ans: str,
        correct_ans: str,
    ) -> Dict[str, str]:
        """
        ⚙️ AGENT ORCHESTRATION (ГЛАВНЫЙ ЦИКЛ АГЕНТА)

        Это "мозг" агента, который управляет процессом.

        Логика работы (Pipeline + Loop):
        1. Validate: Проверка входа.
        2. Perception: Понять, что за контент перед нами.
        3. Planning: Выбрать стратегию.
        4. Action (Attempt 1): Первая попытка генерации.
        5. Reflection (Tool #1): Оценка качества.
        6. Correction (Loop): Если оценка плохая -> ПЕРЕДЕЛАТЬ (Action Attempt 2).
        """
        try:
            logger.info("=" * 70)
            logger.info("🤖 AGENT STARTED: Processing new explanation request")

            # --- ЭТАП 1: ВАЛИДАЦИЯ ВХОДА ---
            validation_error = self._validate_input(question_text, user_ans, correct_ans)
            if validation_error:
                logger.warning(f"Validation failed: {validation_error}")
                raise ValueError(validation_error)

            # --- ЭТАП 2: ВОСПРИЯТИЕ (Perception) ---
            # Агент сам решает, с каким типом контента работает
            content_type = self._analyze_content_type(question_text, user_ans, correct_ans)
            logger.info(f"👁️ PERCEPTION: Identified content type as '{content_type.name}'")

            # --- ЭТАП 3: ПЛАНИРОВАНИЕ (Planning) ---
            # Агент выбирает инструмент (стратегию) под задачу
            strategy = self._select_strategy(content_type)
            logger.info(f"🧠 PLANNING: Selected strategy '{strategy['name']}'")

            # --- ЭТАП 4: ДЕЙСТВИЕ (Action - First Pass) ---
            prompt = self._build_adaptive_prompt(
                question_text, user_ans, correct_ans, content_type, strategy
            )
            logger.info("🎬 ACTION: Generating initial explanation...")
            response_data = self.client.generate_json(prompt)

            # Предварительная проверка структуры (техническая валидация)
            if not self._validate_response_structure(response_data):
                # Fallback если JSON битый, пробуем починить или вернуть заглушку
                response_data = {
                    "explanation": "Ошибка генерации ответа.",
                    "mnemonic_image": "Не удалось создать образ."
                }

            # --- ЭТАП 5 & 6: РЕФЛЕКСИЯ И КОРРЕКЦИЯ (Reflection & Correction Loop) ---
            # Интеграция TOOL #1
            logger.info("🤔 REFLECTION: Validating quality (Tool #1)...")
            quality = self.validate_explanation_quality(
                response_data.get("explanation", ""),
                response_data.get("mnemonic_image", ""),
                content_type
            )

            # ЦИКЛ ПРИНЯТИЯ РЕШЕНИЙ (DECISION MAKING LOOP)
            if quality['requires_regeneration']:
                logger.warning(f"❌ DECISION: Quality low ({quality['quality_score']:.2%}). REGENERATING...")

                # Формируем корректирующий промпт (Feedback Prompt)
                correction_instruction = "\n\nВАЖНО: Предыдущая версия была недостаточно качественной. "
                if not quality['details']['spacing']: correction_instruction += "Добавь пробелы между словами. "
                if not quality['details']['clarity']: correction_instruction += "Сделай предложения более связными. "
                if not quality['details']['vivid']: correction_instruction += "Сделай мнемонический образ ярче и визуальнее. "

                improved_prompt = prompt + correction_instruction

                # Повторное действие (Action - Second Pass)
                response_data = self.client.generate_json(improved_prompt)
                logger.info("✅ CORRECTION: Regeneration complete.")

                # (Опционально) Можно проверить качество еще раз, но обычно 1 цикла достаточно
            else:
                logger.info(f"✅ DECISION: Quality good ({quality['quality_score']:.2%}). Accepting result.")

            # --- ФИНАЛИЗАЦИЯ ---
            result = {
                "explanation_text": response_data.get("explanation", "").strip(),
                "memory_palace_image": response_data.get("mnemonic_image", "").strip(),
                "strategy_used": strategy["name"],
                "quality_score": quality['quality_score'] # Добавляем мета-информацию
            }

            logger.info("🏁 AGENT FINISHED successfully")
            return result

        except Exception as e:
            logger.error(f"💥 AGENT CRASH: {str(e)}", exc_info=True)
            raise

    # ============================================================================
    # ВСПОМОГАТЕЛЬНЫЕ КОГНИТИВНЫЕ ФУНКЦИИ АГЕНТА
    # ============================================================================

    def _analyze_content_type(self, question_text: str, user_ans: str, correct_ans: str) -> ContentTypeEnum:
        """
        👁️ PERCEPTION MODULE (МОДУЛЬ ВОСПРИЯТИЯ)

        Роль: Глаза и уши агента.
        Что делает: Сканирует текст на наличие паттернов (код, короткие фразы, длинный текст).
        Зачем: Чтобы выбрать правильный режим работы (Code Analysis vs Theory vs Short).
        """
        combined_text = (f"{question_text} {user_ans} {correct_ans}").lower()

        # Маркеры кода
        code_markers = [
            "def ", "class ", "return", "import ", "print(", "{", "}",
            "function", "var ", "const ", "if ", "for ", "while ", "=>"
        ]

        if any(marker in combined_text for marker in code_markers):
            return ContentTypeEnum.CODE

        # Эвристика для коротких ответов
        if len(combined_text.split()) < 25:
            return ContentTypeEnum.SHORT

        return ContentTypeEnum.THEORY

    def _select_strategy(self, content_type: ContentTypeEnum) -> Dict[str, str]:
        """
        🧠 PLANNING MODULE (МОДУЛЬ ПЛАНИРОВАНИЯ)

        Роль: Стратег.
        Что делает: Подбирает "инструмент" (шаблон промпта) под задачу.
        Зачем: Нельзя объяснять код так же, как исторический факт.
        """
        strategies = {
            ContentTypeEnum.THEORY: {
                "name": "narrative_breakdown",
                "instruction": "Используй аналогии и сторителлинг. Объясни концепцию через 'почему', а не просто 'что'."
            },
            ContentTypeEnum.CODE: {
                "name": "syntax_logic_analysis",
                "instruction": "Разбери код построчно. Укажи на логическую ошибку. Приведи исправленный сниппет."
            },
            ContentTypeEnum.SHORT: {
                "name": "flashcard_style",
                "instruction": "Будь предельно краток. Используй формат 'Факт -> Причина'. Максимум 2 предложения."
            }
        }
        return strategies.get(content_type, strategies[ContentTypeEnum.THEORY])

    def _build_adaptive_prompt(
        self,
        question_text: str,
        user_ans: str,
        correct_ans: str,
        content_type: ContentTypeEnum,
        strategy: Dict[str, str]
    ) -> str:
        """
        📝 ACTION FORMULATION (ФОРМИРОВАНИЕ ДЕЙСТВИЯ)

        Роль: Переводчик намерений в команды.
        Что делает: Собирает контекст, стратегию и данные в один запрос для LLM.
        """
        base_prompt = (
            f"Ты - опытный ментор по программированию.\n"
            f"Контекст: Пользователь допустил ошибку в вопросе.\n"
            f"Вопрос: {question_text}\n"
            f"Ответ студента: {user_ans}\n"
            f"Правильный ответ: {correct_ans}\n\n"
            f"Твоя задача: Объяснить ошибку, используя стратегию: {strategy['name']}.\n"
            f"Инструкция: {strategy['instruction']}\n\n"
            f"Верни ответ ТОЛЬКО в формате JSON:\n"
            f"{{\n"
            f"  'explanation': 'Текст объяснения',\n"
            f"  'mnemonic_image': 'Яркий визуальный образ для запоминания (Дворец Памяти)'\n"
            f"}}"
        )
        return base_prompt

    def _validate_input(self, q: str, u: str, c: str) -> Optional[str]:
        """
        🛡️ SAFETY FILTER (ФИЛЬТР БЕЗОПАСНОСТИ)

        Роль: Охранник.
        Что делает: Проверяет, что входные данные не пустые и безопасны для обработки.
        """
        if not q or not q.strip(): return "Question text is empty"
        if not c or not c.strip(): return "Correct answer is empty"
        return None

    def _validate_response_structure(self, response: Any) -> bool:
        """
        🔧 TECHNICAL CHECK (ТЕХНИЧЕСКАЯ ПРОВЕРКА)

        Роль: Парсер.
        Что делает: Проверяет, что LLM вернула валидный JSON с нужными ключами.
        """
        if not isinstance(response, dict): return False
        return "explanation" in response and "mnemonic_image" in response
