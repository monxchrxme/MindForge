#TODO Написать нормальный промт для случаем: code и direct
from typing import List, Dict, Any
from services.gigachat_client import GigaChatClient
import uuid
import logging

logger = logging.getLogger(__name__)

class QuizAgent:
    """
    Агент-экзаменатор. Использует LLM для генерации уникальных вопросов по концептам.
    Формат входных и выходных данных строго соответствует архитектуре проекта.
    """

    def __init__(
            self,
            client: GigaChatClient,
            questions_count: int = 5,
            difficulty: str = "auto for each question, based on complexity of the related concept"
    ):
        """
        :param client: Экземпляр GigaChatClient (обязательный для всех агентов)
        :param questions_count: Сколько вопросов генерировать за один квиз
        :param difficulty: Уровень сложности вопросов (например, 'easy', 'medium', 'hard')
        """
        self.client = client
        self.questions_count = questions_count
        self.difficulty = difficulty
        logger.info(f"QuizAgent initialized: questions_count={questions_count}, difficulty={difficulty}")

    def generate_questions(
            self,
            concepts: List[Dict[str, Any]],
            avoid_history: List[str],
            raw_text: str = None,
            mode: str = "standard"
    ) -> List[Dict[str, Any]]:

        logger.info(f"[START] QuizAgent strategy dispatch: mode={mode}")

        if mode == "direct_quiz":
            return self._generate_direct_quiz(raw_text, avoid_history)
        elif mode == "code_practice":
            return self._generate_code_quiz(concepts, avoid_history)
        else:  # standard
            return self._generate_standard_quiz(concepts, avoid_history)

    def _generate_direct_quiz(self, text: str, history: List[str]) -> List[Dict]:
        logger.info("🚀 STRATEGY EXECUTION: Direct Quiz")
        # Лимит вопросов для direct режима (защита от галлюцинаций)
        count = min(self.questions_count, 3)
        prompt = self._direct_text_prompt(text, history, count)

        # Передаем пустой список концептов, т.к. в direct режиме их нет
        return self._execute_pipeline(prompt, [], history)

    def _generate_code_quiz(self, concepts: List[Dict], history: List[str]) -> List[Dict]:
        logger.info("💻 STRATEGY EXECUTION: Code Practice")
        prompt = self._code_prompt(concepts, history)
        return self._execute_pipeline(prompt, concepts, history)

    def _generate_standard_quiz(self, concepts: List[Dict], history: List[str]) -> List[Dict]:
        logger.info("📚 STRATEGY EXECUTION: Standard Quiz")
        prompt = self._standard_prompt(concepts, history)
        return self._execute_pipeline(prompt, concepts, history)

    def _execute_pipeline(
            self,
            prompt: str,
            concepts: List[Dict],
            history: List[str]
    ) -> List[Dict]:
        """
        Общий конвейер обработки: LLM -> JSON -> Validate -> Unique -> PostProcess
        """
        # 1. Вызов LLM
        try:
            raw_questions = self.client.generate_json(prompt)
        except Exception as e:
            logger.error(f"[ERROR] LLM generation failed: {e}")
            return []

        # 2. Валидация структуры (общая для всех)
        valid_questions = self._validate_and_filter_questions(raw_questions)

        # 3. Фильтрация дублей в текущей пачке
        unique_questions = self._validate_unique(valid_questions, history)

        # 4. Пост-процессинг (UUID, Definitions)
        final_questions = self._post_process_questions(unique_questions, concepts)

        logger.info(f"[FINISH] Pipeline completed. Generated {len(final_questions)} questions.")
        return final_questions

    def _direct_text_prompt(self, text: str, avoid_history: List[str], count: int) -> str:
        """
        Промпт для генерации вопросов напрямую по тексту (без выделения концептов).
        """
        # Формируем блок истории, которую нужно избегать
        avoid_part = ""
        if avoid_history:
            recent_history = list(avoid_history)[-15:]
            avoid_part = "НЕ создавай вопросы, похожие на эти:\n" + "\n".join([f"- {q}" for q in recent_history]) + "\n"

        return (
            f"""
            Ты — генератор учебных вопросов для системы квизов. Сгенерируй {self.questions_count} уникальных вопросов уровня сложности '{self.difficulty}' на основе текста заметки:
                    
            {text[:2000]}
            
            Типы вопросов: ~80% multiple_choice, ~20% true_false
            
            Требования:
            - Дистракторы должны быть правдоподобны
            - Избегай слов "всегда", "никогда" и другие универсальные утверждения
            - НЕ создавай вопросы, похожие на эти (сравнивай по смыслу, теме и структуре!):
            {avoid_part}
            
            {self._get_direct_quiz_format()}
            """
        )



    def _code_prompt(self, concepts: List[Dict], avoid_history: List[str]) -> str:
        """
        Промпт для генерации задач по коду.
        Concepts здесь — это список словарей с ключом 'code_snippet'.
        """
        avoid_part = ""
        if avoid_history:
            # Ограничиваем и обрезаем историю для экономии токенов
            recent_history = list(avoid_history)[-15:]
            shortened_history = [
                q[:100] + "..." if len(q) > 100 else q
                for q in recent_history
            ]
            avoid_part = (
                    "НЕ создавай вопросы, похожие на эти (сравнивай по смыслу, теме и структуре!):\n"
                    + "\n".join([f"- {q}" for q in shortened_history]) + "\n"
            )

        # Формируем контекст: Теория + Код
        context_part = ""
        for c in concepts:
            snippet = c.get('code_snippet')
            term = c.get('term')
            if snippet:
                context_part += f"КОНЦЕПТ: {term}\nКод:\n{snippet}\n\n"
            else:
                context_part += f"КОНЦЕПТ: {term}\n{c.get('definition')}\n\n"

        return (
            f"""
            Ты — Senior Developer, занимающийся разработкой квизов для обучающихся. Сгенерируй {self.questions_count} разноплановых (теоретических и практических) задач по этому материалу.
            
            МАТЕРИАЛ:
            {context_part}
            
            ТИПЫ ВОПРОСОВ (~80% multiple_choice, ~20% true_false):         

            - Дистракторы должны быть правдоподобны
            - Избегай слов "всегда", "никогда" и другие универсальные утверждения
            {avoid_part}

            ВАЖНО: Если вопрос требует анализа кода:
            1. Помести сам код в ТОЛЬКО в поле 'code_context'.
            2. В поле 'question' оставь ТОЛЬКО сам вопрос.
            {self._get_code_quiz_format()}
            """
        )

    def _standard_prompt(
            self,
            concepts: List[Dict[str, Any]],
            avoid_history: List[str]
    ) -> str:
        """
        Собирает системный промпт для LLM.
        :param concepts: Список концептов [{ "term":..., "definition":...}]
        :param avoid_history: Множество текстов/хешей ранее сгенерированных вопросов
        :return: Строка-промпт
        """

        logger.info("[STEP] Constructing questions prompt")

        avoid_part = ""
        if avoid_history:
            # Ограничиваем до 10 последних вопросов
            recent_history = list(avoid_history)[-10:]

            # Обрезаем слишком длинные вопросы в истории, чтобы не тратить токены
            # Нам важна суть, а не полный текст
            shortened_history = [
                q[:100] + "..." if len(q) > 100 else q
                for q in recent_history
            ]

            avoid_part = (
                    "НЕ создавай вопросы, похожие на эти (сравнивай по смыслу, теме и структуре!):\n"
                    + "\n".join([f"- {q}" for q in shortened_history])
            )

        concept_part = "\n".join([
            f"{c['term']}: {c['definition']}" for c in concepts
        ])

        prompt = (
            f"""Ты — генератор учебных вопросов для интеллектуальной системы квизов. Сгенерируй {self.questions_count} уникальных образовательных вопросов уровня сложности '{self.difficulty}' на основе концептов:
            {concept_part}

            Типы вопросов: ~80% multiple_choice, ~20% true_false

            Сложность:
            - в случае автоматической сложности для каждого вопроса постарайся, чтобы 50% - высокая сложность (hard), 30% - средняя сложность (medium), 20% - легкая сложность (easy)

            Требования:
            - Дистракторы должны быть правдоподобны
            - Избегай слов "всегда", "никогда" и другие универсальные утверждения
            {avoid_part}

            {self._get_standard_quiz_format()}
            """
            )

        logger.info(f"[STEP] Prompt ready")
        return prompt

    def _get_standard_quiz_format(self) -> str:
        """
        Возвращает строгие инструкции по формату JSON для промпта.
        Используется в генерации по концептам.
        """
        return (
            """
            СТРОГИЙ формат JSON (массив объектов):
            [
              {
                "question": "Текст вопроса (макс 200 символов)",
                "code_context": "(ОПЦИОНАЛЬНО) Кусок кода, к которому относится вопрос. Если кода нет - null или пустая строка.",
                "type": "multiple_choice",
                "options": ["В1", "В2", "В3", "В4"], 
                "correct_answer": "В2",
                "related_concept": "тема вопроса (термин или ключевая фраза)"
              }
            ]
            ВАЖНО:
            1. Возвращай ТОЛЬКО валидный JSON-массив.
            2. Не добавляй никаких комментариев, Markdown-разметки и блоков (```)
            3. Поле 'correct_answer' должно ТОЧНО совпадать с одним из элементов 'options'.
            4. При режиме 'multiple_choice' в поле 'options' ДОЛЖНО БЫТЬ СТРОГО 4 варианта ответа, при режиме 'true_false' должно быть два варианта ["True", "False"]
            5. Поле 'type' может быть ТОЛЬКО вариантами из списка: ["multiple_choice", "true_false"]
            """
        )

    def _get_code_quiz_format(self) -> str:
        """
        Формат JSON для Code Quiz, где code_context критически важен.
        """
        return (
            r"""СТРОГИЙ формат JSON (массив объектов):
    
            [
              {
                "question": "Что выведет этот код?",
                "code_context": "def func():\n    return 42",
                "type": "multiple_choice",
                "options": ["42", "Error", "None", "0"],
                "correct_answer": "42",
                "related_concept": "Функции",
                "concept_definition": "..."
              }
            ]
    
            КРИТИЧЕСКИ ВАЖНО ДЛЯ ПОЛЯ 'code_context':
            1. Код должен быть ОДНОЙ СТРОКОЙ в JSON
            2. Переносы строк заменяй на \n (обратный слеш + буква n)
            3. Табуляцию заменяй на \t или 4 пробела
            4. НЕ используй реальные переносы строк внутри строки!
            5. НЕ используй тройные бэктики (```) и HTML теги (<br>, de> и т.д.)
    
            ПРИМЕРЫ ПРАВИЛЬНОГО ФОРМАТИРОВАНИЯ code_context:
            ПРАВИЛЬНО: "code_context": "class A:\n    def method(self):\n        return 42"
            
            НЕПРАВИЛЬНО (программа упадет с ошибкой JSON!):
            "code_context": "class A:
                def method(self):
                    return 42"
    
            ОБЩИЕ ТРЕБОВАНИЯ:
            1. Возвращай ТОЛЬКО валидный JSON-массив.
            2. Не добавляй никаких комментариев, Markdown-разметки и блоков (```)
            3. Поле 'correct_answer' должно ТОЧНО совпадать с одним из элементов 'options'.
            4. При режиме 'multiple_choice' в поле 'options' ДОЛЖНО БЫТЬ СТРОГО 4 варианта ответа, при режиме 'true_false' должно быть два варианта ["True", "False"]
            5. Поле 'type' может быть ТОЛЬКО вариантами из списка: ["multiple_choice", "true_false"]
                """
        )

    def _get_direct_quiz_format(self) -> str:
        """
        Формат JSON для Direct Quiz с обязательным полем concept_definition.
        """
        return (
            """
            СТРОГИЙ формат JSON (массив объектов):
            [
              {
                "question": "Текст вопроса (макс 200 символов)",
                "code_context": "(ОПЦИОНАЛЬНО) Кусок кода, к которому относится вопрос. Если кода нет - null или пустая строка.",
                "type": "multiple_choice", 
                "options": ["В1", "В2", "В3", "В4"],
                "correct_answer": "В2",
                "related_concept": "тема вопроса (термин или ключевая фраза)",
                "concept_definition": "ОБЯЗАТЕЛЬНО: Краткое теоретическое объяснение ответа."
              }
            ]
               
            ВАЖНО: 
            1. Возвращай ТОЛЬКО валидный JSON-массив.
            2. Не добавляй никаких комментариев, Markdown-разметки и блоков (```)
            3. Поле 'correct_answer' должно ТОЧНО совпадать с одним из элементов 'options'.
            4. При режиме 'multiple_choice' в поле 'options' ДОЛЖНО БЫТЬ СТРОГО 4 варианта ответа, при режиме 'true_false' должно быть два варианта ["True", "False"]
            5. Поле 'type' может быть ТОЛЬКО вариантами из списка: ["multiple_choice", "true_false"]
            """
        )



    def _validate_and_filter_questions(self, raw_questions: Any) -> List[Dict[str, Any]]:
        """
        Фильтрует вопросы по структуре.

        :param raw_questions: Сырой ответ от LLM (должен быть list)
        :return: Список валидных вопросов
        """
        # Проверка что это список
        if not isinstance(raw_questions, list):
            logger.error(f"[ERROR] Expected list, got {type(raw_questions).__name__}")
            return []

        valid_questions = []
        for idx, q in enumerate(raw_questions):
            if not isinstance(q, dict):
                logger.warning(f"[SKIP] Question #{idx + 1} is not a dict")
                continue

            # Используем новый метод валидации
            if self._validate_question_structure(q):
                valid_questions.append(q)
                logger.debug(f"[VALID] Question #{idx + 1} passed validation")
            else:
                logger.warning(f"[SKIP] Question #{idx + 1} failed validation")

        logger.info(f"[STEP] Validated {len(valid_questions)}/{len(raw_questions)} questions")
        return valid_questions

    def _validate_question_structure(self, q: Dict[str, Any]) -> bool:
        """
        Проверяет и НОРМАЛИЗУЕТ структуру вопроса.
        Исправляет типичные ошибки LLM (типы, регистр, форматы).
        """

        # 1. ВАЛИДАЦИЯ И НОРМАЛИЗАЦИЯ ТЕКСТА ВОПРОСА
        if not q.get("question") or not str(q.get("question")).strip():
            logger.warning("[VALIDATION] Empty question text")
            return False

        # Нормализуем текст вопроса
        q["question"] = str(q["question"]).strip()

        # Опционально: ограничение длины
        if len(q["question"]) > 300:
            logger.warning(f"[VALIDATION] Question too long ({len(q['question'])} chars), truncating")
            q["question"] = q["question"][:297] + "..."

        # 2. НОРМАЛИЗАЦИЯ ТИПА ВОПРОСА
        raw_type = str(q.get("type", "")).lower().strip()

        # Нормализуем все возможные варианты написания типов
        if raw_type in ["single_choice", "multi_choice", "choice", "multiple_choice", "multiplechoice"]:
            q["type"] = "multiple_choice"
        elif raw_type in ["boolean", "bool", "yes_no", "true_false", "true-false", "truefalse", "tf"]:
            q["type"] = "true_false"
        else:
            # Неизвестный тип - отклоняем вопрос
            logger.warning(f"[VALIDATION] Unknown type: '{raw_type}' (original: {q.get('type')})")
            return False


        # 3. НОРМАЛИЗАЦИЯ RELATED_CONCEPT
        if not q.get("related_concept") or not str(q.get("related_concept")).strip():
            q["related_concept"] = "General"
        else:
            q["related_concept"] = str(q["related_concept"]).strip()

        # 4. ВАЛИДАЦИЯ MULTIPLE_CHOICE
        if q["type"] == "multiple_choice":
            options = q.get("options", [])

            # Проверка что options - это список
            if not isinstance(options, list):
                logger.warning(f"[VALIDATION] options must be a list, got {type(options).__name__}")
                return False

            # Проверка наличия correct_answer
            if "correct_answer" not in q or q["correct_answer"] is None:
                logger.warning("[VALIDATION] Missing 'correct_answer' field")
                return False

            # Нормализация опций: убираем пустые строки и None
            q["options"] = [
                str(opt).strip()
                for opt in options
                if opt is not None and str(opt).strip()
            ]

            # Проверка что после очистки осталось минимум 2 опции
            if len(q["options"]) < 2:
                logger.warning(f"[VALIDATION] Not enough options after cleanup: {q['options']}")
                return False

            # Дедупликация опций (регистронезависимая)
            seen_lower = {}
            unique_options = []
            for opt in q["options"]:
                opt_lower = opt.lower()
                if opt_lower not in seen_lower:
                    seen_lower[opt_lower] = opt
                    unique_options.append(opt)

            if len(unique_options) != len(q["options"]):
                logger.debug(
                    f"[VALIDATION] Removed {len(q['options']) - len(unique_options)} duplicate options"
                )
                q["options"] = unique_options

            # Проверка минимального количества уникальных опций
            if len(q["options"]) < 2:
                logger.warning(f"[VALIDATION] Not enough unique options: {q['options']}")
                return False

            # Нормализация correct_answer
            q["correct_answer"] = str(q["correct_answer"]).strip()

            if not q["correct_answer"]:
                logger.warning("[VALIDATION] Empty correct_answer after normalization")
                return False

            # РЕГИСТРОНЕЗАВИСИМОЕ сравнение correct_answer с options
            answer_lower = q["correct_answer"].lower()
            options_lower = [opt.lower() for opt in q["options"]]

            if answer_lower not in options_lower:
                logger.warning(
                    f"[VALIDATION] correct_answer '{q['correct_answer']}' "
                    f"not in options {q['options']}"
                )
                return False

            # Приводим correct_answer к точному написанию из options
            # (сохраняем регистр из списка вариантов)
            matching_index = options_lower.index(answer_lower)
            q["correct_answer"] = q["options"][matching_index]

        # 5. ВАЛИДАЦИЯ TRUE_FALSE
        elif q["type"] == "true_false":
            # Проверка наличия correct_answer
            if "correct_answer" not in q or q["correct_answer"] is None:
                logger.warning("[VALIDATION] Missing 'correct_answer' field")
                return False

            # Нормализация ответа (поддержка разных форматов)
            ans_str = str(q["correct_answer"]).lower().strip()

            if ans_str in ["true", "1", "yes", "верно", "да", "истина", "правда"]:
                q["correct_answer"] = "True"
            elif ans_str in ["false", "0", "no", "неверно", "нет", "ложь"]:
                q["correct_answer"] = "False"
            else:
                logger.warning(f"[VALIDATION] Invalid bool answer: '{ans_str}'")
                return False

            # Принудительно ставим стандартные опции
            q["options"] = ["True", "False"]

        return True


    def _validate_unique(
            self,
            questions: List[Dict[str, Any]],
            history: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Фильтрует самоповторы внутри текущей генерации.
        Глобальная уникальность проверяется в Orchestrator через VectorHistoryManager.
        """
        unique = []
        # Следим, чтобы внутри одной пачки из 5 вопросов не было одинаковых
        seen_in_batch = set()

        for idx, q in enumerate(questions):
            text = q.get("question", "").strip()
            if not text:
                continue

            text_lower = text.lower()

            # Проверка на дубликаты внутри ТЕКУЩЕЙ генерации
            if text_lower in seen_in_batch:
                logger.warning(f"[SKIP] Question #{idx + 1}: duplicate within current batch")
                continue

            unique.append(q)
            seen_in_batch.add(text_lower)

        return unique



    def _post_process_questions(
            self,
            questions: List[Dict[str, Any]],
            concepts: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Добавляет UUID и concept_definition к каждому вопросу.

        Для direct_quiz режима: LLM сам генерирует concept_definition.
        Для standard/code режимов: извлекается из списка концептов.

        Поиск концептов регистронезависимый.
        """

        # Создаем регистронезависимый lookup
        concept_lookup = {}
        if concepts:
            for c in concepts:
                term = c.get("term", "").strip()
                if not term:
                    continue

                term_lower = term.lower()

                # Предупреждение о дубликатах (редкий случай)
                if term_lower in concept_lookup:
                    logger.debug(
                        f"[POST-PROCESS] Duplicate concept '{term}', keeping first definition"
                    )
                else:
                    concept_lookup[term_lower] = c.get("definition", "")

        for idx, q in enumerate(questions, 1):
            # 1. Генерация уникального ID
            q["question_id"] = str(uuid.uuid4())

            # 2. Нормализация code_context (может быть None/null)
            q["code_context"] = q.get("code_context")

            # 3. Обработка concept_definition
            if q.get("concept_definition"):
                # Direct Mode: LLM уже вернул определение
                pass
            else:
                # Standard/Code Mode: ищем в концептах
                related = q.get("related_concept", "").strip()

                if not related:
                    q["concept_definition"] = ""
                    logger.warning(
                        f"[POST-PROCESS] Question #{idx} has empty 'related_concept'"
                    )
                else:
                    # Регистронезависимый поиск
                    definition = concept_lookup.get(related.lower(), "")
                    q["concept_definition"] = definition

                    # Логирование только если не нашли и есть концепты
                    if not definition and concepts:
                        available = list(concept_lookup.keys())[:5]  # Первые 5 для краткости
                        logger.warning(
                            f"[POST-PROCESS] Question #{idx}: concept '{related}' not found. "
                            f"Available: {available}..."
                        )

        return questions
