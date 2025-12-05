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
        prompt = self._code_questions_prompt(concepts, history)
        return self._execute_pipeline(prompt, concepts, history)

    def _generate_standard_quiz(self, concepts: List[Dict], history: List[str]) -> List[Dict]:
        logger.info("📚 STRATEGY EXECUTION: Standard Quiz")
        prompt = self._questions_prompt(concepts, history)
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
            f"Ты — генератор учебных квизов. Твоя задача — составить проверочные вопросы по тексту заметки.\n\n"
            f"Сгенерируй {count} уникальных вопросов..."
            
            f"ТЕКСТ ЗАМЕТКИ:\n{text[:2000]}\n\n"  # Ограничиваем, чтобы влезло в контекст
            f"ЗАДАЧА:\n"
            f"Сгенерируй {self.questions_count} уникальных вопросов уровня сложности '{self.difficulty}'.\n"
            f"Распределение типов: ~80% multiple_choice, ~20% true_false.\n\n"
            f"ТРЕБОВАНИЯ К КОНТЕНТУ:\n"
            f"- Вопросы должны проверять понимание сути текста, а не мелких деталей.\n"
            f"- Дистракторы (неверные ответы) должны быть правдоподобными.\n"
            f"{avoid_part}\n"
            f"{self._get_direct_quiz_format()}"
        )

    def _get_code_quiz_format(self) -> str:
        """
        Формат JSON для Code Quiz, где code_context критически важен.
        """
        return (
            "СТРОГИЙ формат JSON (массив объектов):\n"
            "[\n"
            " {\n"
            '  "question": "Что выведет этот код?",\n'
            '  "code_context": "def func():\\n    return 42",\n'
            '  "type": "multiple_choice",\n'
            '  "options": ["42", "Error", "None", "0"],\n'
            '  "correct_answer": "42",\n'
            '  "related_concept": "Функции",\n'
            '  "concept_definition": "..."\n'
            " }\n"
            "]\n"
            "ВАЖНО: Поле 'code_context' должно содержать форматированный код с переносами строк (\\n)."
        )


    def _code_questions_prompt(self, concepts: List[Dict], avoid_history: List[str]) -> str:
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
                    "НЕ создавай вопросы, похожие на эти (по смыслу и коду):\n"
                    + "\n".join([f"- {q}" for q in shortened_history]) + "\n"
            )

        # Формируем контекст: Теория + Код
        context_part = ""
        for c in concepts:
            snippet = c.get('code_snippet')
            term = c.get('term')
            if snippet:
                context_part += f"=== КОНЦЕПТ: {term} ===\nКод:\n{snippet}\n\n"
            else:
                context_part += f"=== КОНЦЕПТ: {term} ===\n{c.get('definition')}\n\n"

        return (
            f"Ты — Senior Developer, проводящий собеседование. Сгенерируй {self.questions_count} практических задач по этому материалу.\n\n"
            f"МАТЕРИАЛ:\n{context_part}\n\n"
            f"ТИПЫ ВОПРОСОВ:\n"
            f"1. Анализ кода: 'Что выведет этот код?', 'Какова сложность этого алгоритма?', 'Найди ошибку в строке 3'.\n"
            f"2. Теория: только если к концепту не приложен код.\n\n"

            # Вставляем блок избегания повторов
            f"{avoid_part}\n"

            f"ВАЖНО: Если вопрос требует анализа кода:\n"
            f"1. Помести сам код в поле 'code_context'.\n"
            f"2. В поле 'question' оставь только сам вопрос (например: 'Какова сложность этого алгоритма?').\n\n"
            f"{self._get_code_quiz_format()}"
        )

    def _get_format_instructions(self) -> str:
        """
        Возвращает строгие инструкции по формату JSON для промпта.
        Используется во всех типах генерации (по концептам и по тексту).
        """
        """СТРОГИЙ формат JSON (массив объектов):

                    [
                      {{
                        "question": "Текст вопроса (макс 180 символов)",
                        "type": "multiple_choice",
                        "options": ["Вариант1", "Вариант2", ...] для multiple_choice,
                        "related_concept": "конкретный концепт из списка концептов, на котором базируется вопрос",
                        "correct_answer": "Вариант1" 
                      }},
                      {{
                        "question": "Текст вопроса-утверждения",
                        "type": "true_false",
                        "options": ["True", "False"],
                        "related_concept": "конкретный концепт из списка концептов, на котором базируется вопрос"
                        "correct_answer": "True"
                      }}
                    ]

                    КРИТИЧЕСКИ ВАЖНО: 
                    - Возвращай ТОЛЬКО JSON-массив
                    - Без пояснений, комментариев, markdown разметки
                    - Проверь запятые и кавычки перед отправкой"""
        return (
            "СТРОГИЙ формат JSON (массив объектов):\n"
            "[\n"
            "  {\n"
            "    \"question\": \"Текст вопроса (макс 200 символов)\",\n"
            "    \"code_context\": \"(ОПЦИОНАЛЬНО) Кусок кода, к которому относится вопрос. Если кода нет - null или пустая строка.\",\n"
            "    \"type\": \"multiple_choice\",\n"
            "    \"options\": [\"вариант1\", \"вариант2\", \"вариант3\", \"вариант4\"],\n"
            "    \"correct_answer\": \"вариант1\",\n"
            "    \"related_concept\": \"тема вопроса (термин или ключевая фраза)\"\n"
            "  },\n"
            "  {\n"
            "    \"question\": \"Текст утверждения\",\n"
            "    \"type\": \"true_false\",\n"
            "    \"options\": [\"True\", \"False\"],\n"
            "    \"correct_answer\": \"True\",\n"
            "    \"related_concept\": \"тема вопроса\"\n"
            "  }\n"
            "]\n\n"
            "ВАЖНО:\n"
            "1. Возвращай ТОЛЬКО валидный JSON-массив.\n"
            "2. Не добавляй никаких комментариев, Markdown-блоков (```"
            "3. Поле 'correct_answer' должно ТОЧНО совпадать с одним из элементов 'options'.\n"
            "4. В multiple_choice должно быть 4 варианта ответа."
        )


    def _get_direct_quiz_format(self) -> str:
        """
        Формат JSON для Direct Quiz с обязательным полем concept_definition.
        """
        return (
            "СТРОГИЙ формат JSON (массив объектов):\n"
            "[\n"
            " {\n"
            '  "question": "Текст вопроса...",\n'
            '  "code_context": "Код или null",\n'
            '  "type": "multiple_choice",\n'
            '  "options": ["вариант1", ...],\n'
            '  "correct_answer": "вариант1",\n'
            '  "related_concept": "тема вопроса",\n'
            '  "concept_definition": "ОБЯЗАТЕЛЬНО: Краткое теоретическое объяснение ответа."\n'
            " }\n"
            "]\n"
            "ВАЖНО: Возвращай ТОЛЬКО валидный JSON-массив."
        )





    def _questions_prompt(
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
            # Ограничиваем до 15 последних вопросов
            recent_history = list(avoid_history)[-15:]

            # Обрезаем слишком длинные вопросы в истории, чтобы не тратить токены
            # Нам важна суть, а не полный текст
            shortened_history = [
                q[:100] + "..." if len(q) > 100 else q
                for q in recent_history
            ]

            avoid_part = (
                    "НЕ создавай вопросы, похожие на эти:\n"
                    + "\n".join([f"- {q}" for q in shortened_history])
            )

        concept_part = "\n".join([
            f"{c['term']}: {c['definition']}" for c in concepts
        ])

        prompt = ( f"""Ты — генератор учебных вопросов для интеллектуальной системы квизов. Сгенерируй {self.questions_count} уникальных образовательных вопросов уровня сложности '{self.difficulty}' на основе концептов:
            {concept_part}
            
            Типы вопросов (80% multiple_choice, 20% true_false):
            1. multiple_choice: 4-6 вариантов ответа
            2. true_false: вопрос с ответом True/False
            
            Сложность:
            - в случае автоматической сложности для каждого вопроса постарайся, чтобы 50% - высокая сложность (hard), 30% - средняя сложность (medium), 20% - легкая сложность (easy)
            Для каждого вопроса самостоятельно назначь уровень difficulty на основе:
            - Абстрактность концепта (факт = easy, принцип = medium, теория = hard)
            - Когнитивная нагрузка (вспомнить = easy, понять = medium, применить = hard)
            - Количество шагов рассуждения (один = easy, несколько = medium/hard)      
            
            Требования:
            - Каждый вопрос ОБЯЗАТЕЛЬНО должел быть связан с одним концептом из списка
            - Если концепт глубокий, содержащий много информации и позволяет на своей основе составить несколько нетривиальных уникальных вопросов, можно использовать его несколько раз
            - Вопросы проверяют понимание, а не запоминание
            - Дистракторы (неправильные варианты в multiple_choice) должны быть правдоподобны и не вызывать сомнений своей искусственностью
            - Избегай слов "всегда", "никогда" и другие универсальные утверждения
            - НЕ создавай вопросы, похожие на эти (сравнивай по смыслу, теме и структуре!):
            {avoid_part}\n
            f"{self._get_format_instructions()}"
            """
        )

        logger.info(f"[STEP] Prompt ready")
        return prompt

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
        # 1. Проверка текста вопроса
        if not q.get("question"):
            logger.warning("[VALIDATION] Empty question text")
            return False

        # 2. Авто-коррекция типа вопроса
        raw_type = q.get("type", "").lower().strip()
        if raw_type in ["single_choice", "multi_choice", "choice"]:
            q["type"] = "multiple_choice"
        elif raw_type in ["boolean", "bool", "yes_no"]:
            q["type"] = "true_false"

        # 3. Проверка поддерживаемых типов
        valid_types = ["multiple_choice", "true_false"]
        if q["type"] not in valid_types:
            logger.warning(f"[VALIDATION] Unknown type '{q.get('type')}' (raw: {raw_type})")
            return False

        # 4. Нормализация related_concept
        if not q.get("related_concept"):
            q["related_concept"] = "General"

        # 5. Валидация multiple_choice
        if q["type"] == "multiple_choice":
            options = q.get("options", [])
            if not isinstance(options, list) or len(options) < 2:
                logger.warning(f"[VALIDATION] multiple_choice needs list of 2+ options. Got: {options}")
                return False

            # Нормализация опций и ответа (все в строки)
            q["options"] = [str(opt).strip() for opt in options]
            q["correct_answer"] = str(q.get("correct_answer", "")).strip()

            if q["correct_answer"] not in q["options"]:
                logger.warning(f"[VALIDATION] correct_answer '{q['correct_answer']}' not in options {q['options']}")
                return False

        # 6. Валидация true_false
        if q["type"] == "true_false":
            # Нормализация ответа
            ans_str = str(q.get("correct_answer", "")).lower().strip()

            if ans_str in ["true", "1", "yes", "верно", "да"]:
                q["correct_answer"] = "True"
            elif ans_str in ["false", "0", "no", "неверно", "нет"]:
                q["correct_answer"] = "False"
            else:
                logger.warning(f"[VALIDATION] Invalid bool answer: {ans_str}")
                return False

            # Принудительно ставим красивые опции
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
        concept_lookup = {c["term"]: c["definition"] for c in concepts}

        for q in questions:
            q["question_id"] = str(uuid.uuid4())

            # Нормализация полей (чтобы не было KeyError)
            q["code_context"] = q.get("code_context")  # None если нет

            # Логика определений
            if q.get("concept_definition"):
                # Если LLM сама дала определение (Direct Mode) - оставляем
                pass
            else:
                # Иначе ищем в базе концептов (Standard/Code Mode)
                related = q.get("related_concept", "")
                q["concept_definition"] = concept_lookup.get(related, "")

        return questions
