import logging
from typing import Dict, Optional, Any, List, Tuple
from enum import Enum

# Настройка логирования
logger = logging.getLogger(__name__)


class ContentTypeEnum(Enum):
    THEORY = "theory"  # Обычный текст, определения, факты
    CODE = "code"  # Программный код, сниппеты
    SHORT = "short"  # Короткие заметки (zettelkasten)


class ExplainAgent:
    """
    ExplainAgent: Автономный агент обучения с обратной связью.

    АРХИТЕКТУРА АГЕНТА:
    1. PERCEPTION (Восприятие): Анализирует входной текст (_analyze_content_type)
    2. PLANNING (Планирование): Выбирает стратегию объяснения (_select_strategy)
    3. ACTION (Действие): Генерирует объяснение (_build_adaptive_prompt + client.generate)
    4. REFLECTION (Рефлексия): Оценивает качество собственной работы (validate_explanation_quality)
    5. CORRECTION (Исправление): Переделывает работу (до 4 раз максимум) с сохранением лучшего
    """

    # Константа: максимум попыток регенерации
    MAX_REGENERATION_ATTEMPTS = 4

    # Константа: минимальный приемлемый score
    MIN_QUALITY_THRESHOLD = 0.7

    def __init__(self, client):
        """
        Инициализация агента.

        Args:
            client: LLM-клиент (GigaChat/OpenAI), умеющий делать generate_json.
        """
        self.client = client
        logger.info("🤖 ExplainAgent initialized")
        logger.info(f"   ├─ MAX_REGENERATION_ATTEMPTS: {self.MAX_REGENERATION_ATTEMPTS}")
        logger.info(f"   ├─ MIN_QUALITY_THRESHOLD: {self.MIN_QUALITY_THRESHOLD}")
        logger.info(f"   └─ INFINITE LOOP PROTECTION: ENABLED")

    def _verify_correct_answer_present(
            self,
            explanation: str,
            correct_ans: str
    ) -> bool:
        """
        Args:
            explanation: Текст пояснения
            correct_ans: Правильный ответ

        Returns:
            bool: True если правильный ответ присутствует в пояснении
        """

        if not explanation or not correct_ans:
            return False

        explanation_lower = explanation.lower()
        correct_ans_lower = correct_ans.lower()

        # МЕТОД 1: Точное совпадение (если есть - сразу OK)
        if correct_ans_lower in explanation_lower:
            return True

        # МЕТОД 2: Основные слова (игнорируем флексии)
        ans_words = [w for w in correct_ans_lower.split() if len(w) > 3]

        # Требует 70% совпадений основных слов
        matches = sum(1 for word in ans_words if word in explanation_lower)
        required_matches = max(1, len(ans_words) - 1)  # Допускаем 1 отклонение
        has_answer = matches >= required_matches

        logger.debug(
            f"📋 Answer verification: {matches}/{required_matches} keywords found. "
            f"Result: {'✅' if has_answer else '❌'}"
        )

        return has_answer


    def validate_explanation_quality(
            self,
            explanation: str,
            mnemonic_image: str,
            content_type: ContentTypeEnum,
            correct_ans: str = ""  # НОВОЕ: добавляем правильный ответ
    ) -> dict:
        """
        AGENT REFLECTION (РЕФЛЕКСИЯ АГЕНТА) - TOOL #1

        Роль: Внутренний критик с защитой качества.
        Что делает: Оценивает качество генерации по объективным метрикам (5 проверок).
        Зачем: Чтобы агент мог принять решение - "хорошо получилось" или "надо переделать".

        Метрики оценки (5 проверок):
        1. Spacing: Проверка на "слипание" слов (Защита от технических сбоев)
        2. Clarity: Проверка пунктуации и читаемости
        3. Vividness: Проверка наличия ярких образов в мнемотехнике
        4. Type Match: Соответствие текста типу контента (код, теория)
        5. Answer Presence: Проверка наличия правильного ответа в пояснении

        Returns:
            dict: {score, requires_regeneration, details, diagnostic_info...}
        """

        # ============ ТИПОБЕЗОПАСНОСТЬ ============

        # Защита от list/dict типов
        if not isinstance(explanation, str):
            logger.warning(
                f"⚠️  Type Error: explanation is {type(explanation).__name__}, "
                f"expected str. Converting..."
            )
            try:
                explanation = str(explanation) if explanation else ""
            except:
                explanation = ""

        if not isinstance(mnemonic_image, str):
            logger.warning(
                f"⚠️  Type Error: mnemonic_image is {type(mnemonic_image).__name__}, "
                f"expected str. Converting..."
            )
            try:
                mnemonic_image = str(mnemonic_image) if mnemonic_image else ""
            except:
                mnemonic_image = ""

        # Защита от пустых строк
        explanation = explanation.strip() if explanation else ""
        mnemonic_image = mnemonic_image.strip() if mnemonic_image else ""

        # ============ МЕТРИКА 1: SPACING (Пробелы) ============
        words = explanation.split()
        # Хорошее соотношение: слов должно быть 20-30% от всех символов
        has_spacing = len(words) > (len(explanation) / 5) if explanation else False
        logger.debug(f"  1️⃣  SPACING: {len(words)} words. Status: {'✅' if has_spacing else '❌'}")

        # ============ МЕТРИКА 2: CLARITY (Ясность) ============
        clarity_indicators = ['.', '!', '?', ';', '\n']
        has_clarity = any(indicator in explanation for indicator in clarity_indicators)
        logger.debug(f"  2️⃣  CLARITY: Punctuation found. Status: {'✅' if has_clarity else '❌'}")

        # ============ МЕТРИКА 3: VIVIDNESS (Образность) ============
        vivid_words = [
            'представь', 'вообрази', 'видим', 'образ', 'цвет', 'представляю',
            'форма', 'метафора', 'аналогия', 'сравнение', 'представляется',
            'видеть', 'как', 'словно', 'похоже', 'воображение', 'рисунок',
            'визуально', 'думай', 'вспомни', 'представляется'
        ]
        is_vivid = any(word in mnemonic_image.lower() for word in vivid_words)
        logger.debug(f"  3️⃣  VIVIDNESS: Vivid words in mnemonic. Status: {'✅' if is_vivid else '❌'}")

        # ============ МЕТРИКА 4: TYPE MATCH (Соответствие типу) ============
        if content_type == ContentTypeEnum.THEORY:
            sentences = explanation.split('.')
            # Фильтруем: минимум 5 слов в предложении = считаем "полным"
            full_sentences = len([s for s in sentences if len(s.split()) >= 5])

            # НОВОЕ: Требуем минимум 2 полных предложения вместо 3
            type_match = full_sentences >= 2
            logger.debug(f"  4️⃣  TYPE_MATCH (THEORY): {sentences} sentences. Status: {'✅' if type_match else '❌'}")


        elif content_type == ContentTypeEnum.CODE:
            # Код должен содержать ключевые слова программирования
            code_keywords = [
                'код', 'функция', 'переменная', 'цикл', 'условие', 'значение',
                'return', 'error', 'ошибка', 'логик', 'синтакси', 'типы данных'
            ]
            type_match = any(kw in explanation.lower() for kw in code_keywords)
            logger.debug(f"  4️⃣  TYPE_MATCH (CODE): Code keywords found. Status: {'✅' if type_match else '❌'}")

        else:  # SHORT
            # Короткий ответ должен быть действительно кратким
            word_count = len(explanation.split())
            sentence_count = len([s for s in explanation.split('.') if s.strip()])
            # Проходит если: <= 50 слов ИЛИ (2-3 предложения И четко разделены)
            type_match = (word_count <= 50) or (2 <= sentence_count <= 3)
            logger.debug(
                f"  4️⃣  TYPE_MATCH (SHORT): {len(explanation.split())} words. Status: {'✅' if type_match else '❌'}")

        # ============ МЕТРИКА 5: ANSWER PRESENCE ============
        has_correct_answer = self._verify_correct_answer_present(explanation, correct_ans)
        logger.debug(
            f"  5️⃣  ANSWER_PRESENCE: Correct answer in explanation. Status: {'✅' if has_correct_answer else '❌'}")

        # ============ РАСЧЕТ ИТОГОВОГО БАЛЛА ============
        quality_score = (
                                float(has_spacing) +  # 0.0 или 1.0
                                float(has_clarity) +  # 0.0 или 1.0
                                float(is_vivid) +  # 0.0 или 1.0
                                float(type_match) +  # 0.0 или 1.0
                                float(has_correct_answer)  # 0.0 или 1.0
                        ) / 5.0  # Делим на 5, поскольку метрик 5

        # РЕШЕНИЕ АГЕНТА: Переделывать, если качество ниже порога
        requires_regeneration = quality_score < self.MIN_QUALITY_THRESHOLD

        logger.info(
            f"🔍 REFLECTION: Score={quality_score:.2%} | "
            f"Threshold={self.MIN_QUALITY_THRESHOLD:.0%} | "
            f"Regenerate={'❌ YES' if requires_regeneration else '✅ NO'}"
        )

        return {
            'quality_score': quality_score,
            'requires_regeneration': requires_regeneration,
            'details': {
                'spacing': has_spacing,
                'clarity': has_clarity,
                'vivid': is_vivid,
                'type_match': type_match,
                'correct_answer': has_correct_answer  # НОВОЕ
            }
        }

    def explain_error(
            self,
            question_text: str,
            user_ans: str,
            correct_ans: str,
    ) -> Dict[str, str]:
        """
        AGENT ORCHESTRATION (ГЛАВНЫЙ ЦИКЛ АГЕНТА)

        Это "мозг" агента, который управляет процессом.

        Логика работы (Pipeline + Loop + Safety):
        1. Validate: Проверка входа.
        2. Perception: Понять, что за контент перед нами.
        3. Planning: Выбрать стратегию.
        4. Action (Attempt 1): Первая попытка генерации.
        5. Reflection (Tool #1): Оценка качества с проверкой ответа.
        6. Correction Loop (Макс 4 попытки):
           - Если качество плохое → переделать
           - Если качество хорошее → принять
           - Если исчерпали попытки → вернуть лучший результат
        7. Graceful Degradation: При ошибке → fallback пояснение
        """

        try:
            logger.info("=" * 70)
            logger.info("🤖 AGENT STARTED: Processing new explanation request")
            logger.info(f"   Question: {question_text[:50]}...")
            logger.info(f"   User Answer: {user_ans[:50]}...")
            logger.info(f"   Correct Answer: {correct_ans[:50]}...")

            # --- ЭТАП 1: ВАЛИДАЦИЯ ВХОДА ---
            validation_error = self._validate_input(question_text, user_ans, correct_ans)
            if validation_error:
                logger.warning(f"❌ Validation failed: {validation_error}")
                raise ValueError(validation_error)

            # --- ЭТАП 2: ВОСПРИЯТИЕ (Perception) ---
            content_type = self._analyze_content_type(question_text, user_ans, correct_ans)
            logger.info(f"👁️  PERCEPTION: Identified content type as '{content_type.name}'")

            # --- ЭТАП 3: ПЛАНИРОВАНИЕ (Planning) ---
            strategy = self._select_strategy(content_type)
            logger.info(f"🧠 PLANNING: Selected strategy '{strategy['name']}'")

            # --- ЭТАП 4: ДЕЙСТВИЕ (Action - First Pass) ---
            prompt = self._build_adaptive_prompt(
                question_text, user_ans, correct_ans, content_type, strategy
            )
            logger.info("🎬 ACTION: Generating initial explanation (Attempt 1/4)...")

            response_data = self.client.generate_json(prompt)

            # Техническая валидация структуры
            if not self._validate_response_structure(response_data):
                logger.warning("⚠️  Response structure invalid. Using fallback...")
                response_data = {
                    "explanation": f"Правильный ответ: {correct_ans}",
                    "mnemonic_image": "Обратите внимание на правильный ответ выше."
                }

            # --- ЭТАП 5-6: РЕФЛЕКСИЯ И КОРРЕКЦИЯ LOOP ---
            logger.info("🤔 REFLECTION LOOP: Validating quality with answer check...")

            # Инициализация истории генераций
            generation_history: List[Tuple[dict, float]] = []
            best_response = response_data
            best_quality_score = 0.0

            # Цикл регенерации (максимум MAX_REGENERATION_ATTEMPTS)
            for attempt in range(self.MAX_REGENERATION_ATTEMPTS):
                # Оцениваем текущий результат
                quality = self.validate_explanation_quality(
                    response_data.get("explanation", ""),
                    response_data.get("mnemonic_image", ""),
                    content_type,
                    correct_ans  # передаем правильный ответ
                )

                # Сохраняем в историю
                generation_history.append((response_data, quality['quality_score']))

                # Обновляем лучший результат
                if quality['quality_score'] > best_quality_score:
                    best_response = response_data
                    best_quality_score = quality['quality_score']
                    logger.info(f"🏆 New best result: {best_quality_score:.2%}")

                # Проверяем, нужна ли переделка
                if quality['requires_regeneration'] and attempt < self.MAX_REGENERATION_ATTEMPTS - 1:
                    attempt_num = attempt + 2
                    logger.warning(
                        f"❌ DECISION: Quality low ({quality['quality_score']:.2%}). "
                        f"REGENERATING (Attempt {attempt_num}/{self.MAX_REGENERATION_ATTEMPTS})..."
                    )

                    # Формируем корректирующий промпт
                    correction_instruction = (
                        "\n\n⚠️  ВАЖНО: Предыдущая версия была недостаточно качественной. "
                    )

                    if not quality['details']['spacing']:
                        correction_instruction += "Разделяй слова пробелами. "
                    if not quality['details']['clarity']:
                        correction_instruction += "Добавь пунктуацию. "
                    if not quality['details']['vivid']:
                        correction_instruction += "Сделай образ ярче и визуальнее. "
                    if not quality['details']['type_match']:
                        correction_instruction += "Лучше соотноси с типом контента. "
                    if not quality['details']['correct_answer']:
                        correction_instruction += (
                            f"КРИТИЧНО: Упомяни правильный ответ '{correct_ans}' в пояснении. "
                        )

                    improved_prompt = prompt + correction_instruction

                    # Повторная генерация
                    response_data = self.client.generate_json(improved_prompt)

                    if not self._validate_response_structure(response_data):
                        logger.warning("⚠️  Regenerated response structure invalid")
                        response_data = best_response
                        break

                else:
                    # Качество хорошее или исчерпали попытки
                    if quality['requires_regeneration']:
                        logger.warning(
                            f"⚠️  MAX ATTEMPTS REACHED. Using best result: "
                            f"{best_quality_score:.2%}"
                        )
                        response_data = best_response
                    else:
                        logger.info(
                            f"✅ DECISION: Quality good ({quality['quality_score']:.2%}). "
                            f"Accepting result."
                        )
                    break

            # --- ФИНАЛИЗАЦИЯ ---
            logger.info(
                f"📊 GENERATION HISTORY: {len(generation_history)} attempts. "
                f"Best: {best_quality_score:.2%}"
            )

            result = {
                "explanation_text": response_data.get("explanation", "").strip(),
                "memory_palace_image": response_data.get("mnemonic_image", "").strip(),
                "strategy_used": strategy["name"],
                "quality_score": best_quality_score,
                "attempts_used": len(generation_history)
            }

            logger.info("🏁 AGENT FINISHED SUCCESSFULLY")
            logger.info("=" * 70)

            return result

        # ============ GRACEFUL DEGRADATION ============
        except Exception as e:
            logger.error(f"💥 AGENT ERROR: {str(e)}", exc_info=True)
            logger.info("🚨 ACTIVATING GRACEFUL DEGRADATION...")

            # Возвращаем fallback объяснение вместо краша
            fallback_result = {
                "explanation_text": (
                    f"Вы выбрали: '{user_ans}'\n\n"
                    f"Правильный ответ: '{correct_ans}'\n\n"
                    f"Объяснение: Это ключевой момент, который стоит запомнить. "
                    f"Обратитесь к исходным материалам для полного понимания темы. "
                    f"Советуем перечитать определение и примеры, чтобы закрепить знание."
                ),
                "memory_palace_image": (
                    f"Представьте яркую картину: вы в аудитории, и преподаватель указывает на доску, "
                    f"где крупными буквами написано: '{correct_ans}'. Это ПРАВИЛЬНЫЙ ответ. "
                    f"Свяжите этот образ с тем, что вы знаете о теме. Запомните эту картину!"
                ),
                "strategy_used": "fallback_emergency",
                "quality_score": 0.5,
                "error": str(e),
                "attempts_used": 0
            }

            logger.info("✅ FALLBACK RESULT GENERATED")
            return fallback_result

    # ============================================================================
    # ВСПОМОГАТЕЛЬНЫЕ КОГНИТИВНЫЕ ФУНКЦИИ АГЕНТА
    # ============================================================================

    def _analyze_content_type(
            self,
            question_text: str,
            user_ans: str,
            correct_ans: str
    ) -> ContentTypeEnum:
        """
        👁️  PERCEPTION MODULE (МОДУЛЬ ВОСПРИЯТИЯ)

        Роль: Глаза и уши агента.
        Что делает: Сканирует текст на наличие паттернов (код, короткие фразы, длинный текст).
        Зачем: Чтобы выбрать правильный режим работы (Code Analysis vs Theory vs Short).
        """
        combined_text = (f"{question_text} {user_ans} {correct_ans}").lower()

        # Маркеры кода
        code_markers = [
            "def ", "class ", "return", "import ", "print(", "{", "}",
            "function", "var ", "const ", "if ", "for ", "while ", "=>",
            "async ", "await ", "try ", "except ", "finally ", "lambda",
            "=>", "==", "!=", "===", "!==", "//"
        ]

        if any(marker in combined_text for marker in code_markers):
            return ContentTypeEnum.CODE

        # Эвристика для коротких ответов
        if len(combined_text.split()) < 25:
            return ContentTypeEnum.SHORT

        return ContentTypeEnum.THEORY

    def _select_strategy(self, content_type: ContentTypeEnum) -> Dict[str, str]:
        """
        PLANNING MODULE (МОДУЛЬ ПЛАНИРОВАНИЯ)

        Что делает: Подбирает "инструмент" (шаблон промпта) под задачу.
        """
        strategies = {
            ContentTypeEnum.THEORY: {
                "name": "narrative_breakdown",
                "instruction": (
                    "Используй аналогии, термины и сторителлинг при надобности. " # было "Используй аналогии и сторителлинг"
                    "Объясни концепцию через 'почему', а не просто 'что'. "
                    "Сделай объяснение доступным и запоминающимся."
                    "Используй 3-5 предложений для объяснения ответа, но при явной надобности используй большее количество."
                )
            },
            ContentTypeEnum.CODE: {
                "name": "syntax_logic_analysis",
                "instruction": (
                    "Разбери код построчно. "
                    "Укажи на логическую ошибку четко и ясно. "
                    "Приведи исправленный сниппет кода с объяснением."
                    "Используй 3-5 предложений для объяснения ответа, но при явной надобности используй большее количество."
                )
            },
            ContentTypeEnum.SHORT: {
                "name": "flashcard_style",
                "instruction": (
                    "Будь предельно краток. "
                    "Используй формат 'Факт -> Причина'. "
                    "ТРЕБОВАНИЕ: максимум 35-40 слов. Если нужно больше - используй 3 предложения."
                )
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
        ACTION FORMULATION (ФОРМИРОВАНИЕ ДЕЙСТВИЯ)

        Что делает: Собирает контекст, стратегию и данные в один запрос для LLM.
        """

        if content_type == ContentTypeEnum.SHORT:
            sentence_instruction = "Используй 2-3 предложения максимум (35-40 слов). Будь лаконичен."
        elif content_type == ContentTypeEnum.THEORY:
            sentence_instruction = "Используй 3-4 предложения (80-120 слов). Развернуто и понятно."
        else:  # CODE
            sentence_instruction = "Используй 3-5 предложений для объяснения ответа."

        if content_type == ContentTypeEnum.CODE:
            mentor_title = "по программированию и анализу кода"
        elif content_type == ContentTypeEnum.THEORY:
            mentor_title = "по образованию и пониманию концепций"
        else:  # SHORT
            mentor_title = "для быстрого обучения и запоминания"

        base_prompt = (
            f"Ты - опытный ментор {mentor_title}.\n\n"
            f"Задача: Объяснить, почему ответ пользователя неправильный.\n"
            f"Вопрос: {question_text}\n"
            f"Ответ пользователя: {user_ans}\n"
            f"Правильный ответ: {correct_ans}\n\n"
            f"ТРЕБОВАНИЯ (ОБЯЗАТЕЛЬНЫЕ):\n"
            f"1. Твое пояснение ДОЛЖНО содержать правильный ответ '{correct_ans}'\n"
            f"2. Объясни, почему '{user_ans}' неправильный\n"
            f"3. Подробно объясни, почему '{correct_ans}' правильный\n"
            f"4. Используй стратегию: {strategy['name']}\n"
            f"5. {strategy['instruction']}\n\n"
            f"6. {sentence_instruction}\n"
            f"ФОРМАТ ОТВЕТА (ОБЯЗАТЕЛЬНЫЙ):\n"
            f"Верни ТОЛЬКО валидный JSON без кода и комментариев:\n"
            f"{{\n"
            f"  \"explanation\": \"Полное объяснение ошибки (включай '{correct_ans}')\",\n"
            f"  \"mnemonic_image\": \"Яркий визуальный образ для запоминания (Дворец Памяти, метафора или история)\"\n"
            f"}}\n\n"
            f"ПОМНИ: Пояснение должно быть четким, ясным и содержать правильный ответ!"
        )

        return base_prompt

    def _validate_input(self, q: str, u: str, c: str) -> Optional[str]:
        """
        SAFETY FILTER (ФИЛЬТР БЕЗОПАСНОСТИ)

        Что делает: Проверяет, что входные данные не пустые и безопасны для обработки.
        """
        if not q or not q.strip():
            return "Question text is empty"
        if not c or not c.strip():
            return "Correct answer is empty"
        return None

    def _validate_response_structure(self, response: Any) -> bool:
        """
        TECHNICAL CHECK (ТЕХНИЧЕСКАЯ ПРОВЕРКА)

        Что делает: Проверяет, что LLM вернула валидный JSON с нужными ключами.
        """
        # Защита от non-dict типов
        if not isinstance(response, dict):
            logger.warning(
                f"⚠️  Response is not dict: {type(response).__name__}. "
                f"Cannot parse."
            )
            return False

        # Проверка ключей
        if "explanation" not in response:
            logger.warning("⚠️  Missing 'explanation' key in response")
            return False

        if "mnemonic_image" not in response:
            logger.warning("⚠️  Missing 'mnemonic_image' key in response")
            return False

        # Проверка не пустоты
        explanation = response.get("explanation", "")
        mnemonic = response.get("mnemonic_image", "")

        if not explanation or not isinstance(explanation, str) or not explanation.strip():
            logger.warning("⚠️  'explanation' is empty or not a string")
            return False

        if not mnemonic or not isinstance(mnemonic, str) or not mnemonic.strip():
            logger.warning("⚠️  'mnemonic_image' is empty or not a string")
            return False

        return True

