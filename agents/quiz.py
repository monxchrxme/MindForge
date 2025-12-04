#TODO Написать нормальный промт для случаем: code и direct
from typing import List, Dict, Set, Any
from services.gigachat_client import GigaChatClient
import uuid
import json
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
            mode: str = None
    ) -> List[Dict[str, Any]]:
        """
        Генерирует уникальные вопросы на основе списка концептов.

        :param concepts: Список концептов [{ "term": str, "definition": str, ... }, ...]
        :param avoid_history: Список вопросов, которые нельзя повторять в этой сессии
        :param raw_text: #TODO дописать описание
        :param mode: #TODO Дописать описание mode
        :return: Список новых вопросов в формате:
        [
            {
                "question_id": str, # UUID для уникальности (генерируется здесь)
                "question": str,    # текст вопроса
                "type": str,        # "multiple_choice", "true_false", etc.
                "options": List[str],         # для multiple_choice
                "correct_answer": str,
                "related_concept": str,       # термин/ключ, на который ссылается вопрос
                "concept_definition": str     # определение (для ExplainAgent)
            },
            ...
        ]
        """
        logger.info("[START] QuizAgent.generate_questions called")
        logger.debug(f"[INPUT] concepts:\n{json.dumps(concepts, ensure_ascii=False, indent=2)}")
        logger.debug(f"[INPUT] avoid_history:\n{json.dumps(list(avoid_history), ensure_ascii=False, indent=2)}")

        if mode == "code_practice": #TODO mode
            prompt = self._code_questions_prompt(concepts, avoid_history)
        elif raw_text:
            logger.info("Using DIRECT TEXT strategy")
            prompt = self._direct_text_prompt(raw_text, avoid_history)
        else:
            logger.info("Using CONCEPT-BASED strategy")
            if not concepts:
                logger.warning("No concepts provided for concept-based strategy")
                return []
            prompt = self._questions_prompt(concepts, avoid_history)

        # logger.debug(f"[STEP] Prompt constructed:\n{prompt}")

        # raw_questions = self.client.generate_json(prompt)
        # logger.info(f"[STEP] raw_questions from LLM:\n{json.dumps(raw_questions, ensure_ascii=False, indent=2)}")

        # Шаг 1: Получение JSON от LLM (с обработкой ошибок)
        try:
            raw_questions = self.client.generate_json(prompt)
            logger.debug(
                f"[STEP] Received {len(raw_questions) if isinstance(raw_questions, list) else 'N/A'} raw questions from LLM")
        except ValueError as e:
            logger.error(f"[ERROR] JSON parsing failed after retries: {e}")
            return []
        except Exception as e:
            logger.error(f"[ERROR] Unexpected error in generate_json: {e}")
            return []

        # Шаг 2: Валидация структуры (НОВЫЙ МЕТОД)
        valid_and_filtered_questions = self._validate_and_filter_questions(raw_questions)

        # Шаг 3: Проверка уникальности
        valid_questions = self._validate_unique(valid_and_filtered_questions, avoid_history)
        logger.debug(f"[STEP] After validation, valid_questions:\n{json.dumps(valid_questions, ensure_ascii=False, indent=2)}")

        # Шаг 4: Постобработка (добавление UUID, concept_definition)
        processed_questions = self._post_process_questions(valid_questions, concepts)
        logger.info("[FINISH] QuizAgent.generate_questions finished")
        logger.info(f"[FINISH] Returning {len(processed_questions)} questions")
        logger.debug(f"[OUTPUT] processed_questions:\n{json.dumps(processed_questions, ensure_ascii=False, indent=2)}")

        # ДОБАВИТЬ логирование для диагностики
        if len(processed_questions) < self.questions_count:
            logger.warning(
                f"[WARNING] Generated {len(processed_questions)}/{self.questions_count} questions. "
                f"Some questions were filtered out during validation."
            )

        if not processed_questions:
            logger.error("[ERROR] No valid questions generated. Check prompt and LLM response.")

        logger.info(f"[FINISH] Returning {len(processed_questions)} questions")

        return processed_questions

    def _direct_text_prompt(self, text: str, avoid_history: List[str]) -> str:
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
            f"ТЕКСТ ЗАМЕТКИ:\n{text[:2000]}\n\n"  # Ограничиваем, чтобы влезло в контекст
            f"ЗАДАЧА:\n"
            f"Сгенерируй {self.questions_count} уникальных вопросов уровня сложности '{self.difficulty}'.\n"
            f"Распределение типов: ~80% multiple_choice, ~20% true_false.\n\n"
            f"ТРЕБОВАНИЯ К КОНТЕНТУ:\n"
            f"- Вопросы должны проверять понимание сути текста, а не мелких деталей.\n"
            f"- Дистракторы (неверные ответы) должны быть правдоподобными.\n"
            f"{avoid_part}\n"
            f"{self._get_format_instructions()}"  # <-- Используем наш новый метод
        )

    def _code_questions_prompt(self, concepts: List[Dict], avoid_history: List[str]) -> str:
        """
        Промпт для генерации задач по коду.
        Concepts здесь — это список словарей с ключом 'code_snippet'.
        """
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
            f"ВАЖНО: Если вопрос требует анализа кода:\n"
            f"1. Помести сам код в поле 'code_context'.\n"
            f"2. В поле 'question' оставь только сам вопрос (например: 'Какова сложность этого алгоритма?').\n\n"
            f"{self._get_format_instructions()}"
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
            avoid_part = (
                    "НЕ создавай вопросы, похожие на эти:\n"
                    + "\n".join([f"- {q}" for q in recent_history])
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
        Проверяет, что вопрос содержит все обязательные поля и корректные значения.

        :param q: Словарь с данными вопроса
        :return: True если структура корректна, False иначе
        """
        # Проверка обязательных полей
        required_fields = ["question", "type", "correct_answer", "related_concept"]
        for field in required_fields:
            if field not in q or not q[field]:
                logger.warning(f"[VALIDATION] Missing or empty required field '{field}': {q}")
                return False

        # Проверка допустимых типов вопросов
        valid_types = ["multiple_choice", "true_false"]
        if q["type"] not in valid_types:
            logger.warning(f"[VALIDATION] Invalid question type '{q['type']}'. Expected: {valid_types}")
            return False

        # Валидация для multiple_choice
        if q["type"] == "multiple_choice":
            options = q.get("options")

            # options должны быть списком
            if not isinstance(options, list) or len(options) < 2:
                logger.warning(f"[VALIDATION] multiple_choice must have list of options (min 2): {options}")
                return False

            # correct_answer должен быть в options
            if q["correct_answer"] not in options:
                logger.warning(f"[VALIDATION] correct_answer '{q['correct_answer']}' not in options: {options}")
                return False

        # Валидация для true_false
        if q["type"] == "true_false":
            valid_answers = ["True", "False", "true", "false"]
            if q["correct_answer"] not in valid_answers:
                logger.warning(
                    f"[VALIDATION] true_false correct_answer must be True/False, got: '{q['correct_answer']}'")
                return False

        # Проверка длины вопроса (опционально)
        if len(q["question"]) > 250:
            logger.warning(f"[VALIDATION] Question too long ({len(q['question'])} chars): {q['question'][:50]}...")
            return False

        return True

    def _validate_unique(
            self,
            questions: List[Dict[str, Any]],
            history: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Фильтрует вопросы по уникальности (точное и семантическое совпадение).

        :param questions: Список вопросов после структурной валидации
        :param history: Множество текстов ранее заданных вопросов
        :return: Список уникальных вопросов
        """
        logger.info("[STEP] Validating uniqueness (exact + semantic)")

        unique = []
        seen_exact = set(history)  # Для точного совпадения
        seen_texts = list(history)  # Для семантического сравнения

        for idx, q in enumerate(questions):
            text = q.get("question", "").strip()
            text_lower = text.lower()

            if not text:
                logger.warning(f"[SKIP] Question #{idx + 1}: empty text")
                continue

            # Проверка 1: Точное совпадение
            if text_lower in seen_exact:
                logger.info(f"[SKIP] Question #{idx + 1}: exact duplicate")
                continue

            # Проверка 2: Семантическое совпадение
            is_duplicate = False
            for seen_text in seen_texts:
                if self._is_semantically_similar(text, seen_text):
                    logger.info(f"[SKIP] Question #{idx + 1}: semantically similar to existing")
                    is_duplicate = True
                    break

            if is_duplicate:
                continue

            # Вопрос уникален
            unique.append(q)
            seen_exact.add(text_lower)
            seen_texts.append(text)
            logger.debug(f"[VALID] Question #{idx + 1} added as unique")

        logger.info(f"[STEP] {len(unique)}/{len(questions)} questions passed uniqueness check")
        return unique


    def _is_semantically_similar(
            self,
            q1: str,
            q2: str,
            threshold: float = 0.7
    ) -> bool:
        """
        Проверяет семантическую похожесть вопросов через коэффициент Жаккара.

        :param q1: Первый вопрос
        :param q2: Второй вопрос
        :param threshold: Порог похожести (0.7 = 70% совпадающих слов)
        :return: True если вопросы похожи
        """
        # Удаляем стоп-слова и знаки препинания
        import re
        stop_words = {"как", "что", "где", "когда", "почему", "какой", "в", "на", "из", "по", "для", "с", "к"}

        # Токенизация и очистка
        def tokenize(text: str) -> Set[str]:
            # Убираем знаки препинания, приводим к нижнему регистру
            words = re.findall(r'\b\w+\b', text.lower())
            # Фильтруем стоп-слова и короткие слова
            return set(w for w in words if w not in stop_words and len(w) > 2)

        words1 = tokenize(q1)
        words2 = tokenize(q2)

        if not words1 or not words2:
            return False

        # Коэффициент Жаккара (Jaccard similarity)
        intersection = len(words1 & words2)
        union = len(words1 | words2)

        if union == 0:
            return False

        similarity = intersection / union

        logger.debug(f"[SIMILARITY] {similarity:.2f} between:\n  '{q1[:50]}...'\n  '{q2[:50]}...'")

        return similarity >= threshold

    def _post_process_questions(
            self,
            questions: List[Dict[str, Any]],
            concepts: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Добавляет UUID, подтягивает определение концепта для ExplainAgent.
        :param questions: Список вопросов без дубликатов
        :param concepts: Исходные концепты
        :return: Финализированный список вопросов для Orchestrator
        """

        logger.info("[STEP] Post-processing questions (assigning UUIDs, concept_definition)")

        concept_lookup = {c["term"]: c["definition"] for c in concepts}

        for idx, q in enumerate(questions):
            original = q.copy()
            q["question_id"] = str(uuid.uuid4())
            related = q.get("related_concept") or ""
            q["concept_definition"] = concept_lookup.get(related, "")
            logger.debug(
                f"[UPDATE] Processed question #{idx + 1}:\n"
                f"[ORIGINAL] {json.dumps(original, ensure_ascii=False, indent=2)}\n"
                f"[UPDATED]  {json.dumps(q, ensure_ascii=False, indent=2)}"
            )

        logger.info(f"[STEP] Post-processing complete: {len(questions)} questions processed")
        return questions

