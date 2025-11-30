# agents/quiz.py
#TODO Сис промт логируется, чзх
#TODO убрать почти все в дебаг
#TODO много тру фолс
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
            difficulty: str = "medium"
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
            avoid_history: Set[str]
    ) -> List[Dict[str, Any]]:
        """
        Генерирует уникальные вопросы на основе списка концептов.

        :param concepts: Список концептов [{ "term": str, "definition": str, ... }, ...]
        :param avoid_history: Множество (set) хешей текстов вопросов, которые нельзя повторять в этой сессии

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
        logger.info(f"[INPUT] concepts:\n{json.dumps(concepts, ensure_ascii=False, indent=2)}")
        logger.info(f"[INPUT] avoid_history:\n{json.dumps(list(avoid_history), ensure_ascii=False, indent=2)}")

        prompt = self._questions_prompt(concepts, avoid_history)
        logger.info(f"[STEP] Prompt constructed:\n{prompt}")

        # raw_questions = self.client.generate_json(prompt)
        # logger.info(f"[STEP] raw_questions from LLM:\n{json.dumps(raw_questions, ensure_ascii=False, indent=2)}")

        # Шаг 1: Получение JSON от LLM (с обработкой ошибок)
        try:
            raw_questions = self.client.generate_json(prompt)
            logger.info(
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
        logger.info(f"[STEP] After validation, valid_questions:\n{json.dumps(valid_questions, ensure_ascii=False, indent=2)}")

        # Шаг 4: Постобработка (добавление UUID, concept_definition)
        processed_questions = self._post_process_questions(valid_questions, concepts)
        logger.info("[FINISH] QuizAgent.generate_questions finished")
        logger.info(f"[FINISH] Returning {len(processed_questions)} questions")
        logger.info(f"[OUTPUT] processed_questions:\n{json.dumps(processed_questions, ensure_ascii=False, indent=2)}")

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


    def _questions_prompt(
            self,
            concepts: List[Dict[str, Any]],
            avoid_history: Set[str]
    ) -> str:
        """
        Собирает системный промпт для LLM.
        :param concepts: Список концептов [{ "term":..., "definition":...}]
        :param avoid_history: Множество текстов/хешей ранее сгенерированных вопросов
        :return: Строка-промпт
        """

        logger.info("[STEP] Constructing questions prompt")

        # avoid_part = ""
        # if avoid_history:
        #     avoid_part = (
        #             "\n".join(list(avoid_history))
        #     )

        avoid_part = ""
        if avoid_history:
            # Ограничиваем до 15 последних вопросов
            recent_history = list(avoid_history)[15:]
            avoid_part = (
                    "НЕ создавай вопросы, похожие на эти:\n"
                    + "\n".join([f"- {q}" for q in recent_history])
            )

        concept_part = "\n".join([
            f"{c['term']}: {c['definition']}" for c in concepts
        ])

        '''
        старый промт 
        '''
        # prompt = (
        #     f"На основе следующих понятий и определений:\n{concept_part}\n\n"
        #     f"Сгенерируй {self.questions_count} уникальных вопросов уровня сложности '{self.difficulty}' "
        #     f"(разных типов: множественный выбор, верно/неверно и т.п.). "
        #     f"{avoid_part}\n"
        #     "Формат ответа — JSON list из словарей, в каждом:\n"
        #     "{'question': str, 'type': str, 'options': list|null, 'correct_answer': str, "
        #     "'related_concept': str}\nВсе ответы должны быть различны по сути и формулировке."
        # )

        # prompt = (
        #     f"Ты — генератор учебных вопросов для интеллектуальной системы квизов. "
        #     f"Твоя задача — на основе следующих понятий и их определений:\n"
        #     f"{concept_part}\n\n"
        #     f"Сгенерируй {self.questions_count} уникальных осмысленных образовательных вопросов уровня сложности '{self.difficulty}'.\n"
        #     f"Типы вопросов должны быть разнообразны и включать: множественный выбор (1 или больше вариантов ответа) (multiple_choice), верно/неверно (true_false)"
        #     f"Старайся, чтобы примерно 80% вопросов были с выбором ответа (multiple_choice), 20% — верно/неверно (true_false)\n\n"
        #
        #     "Требования к вопросам:\n"
        #     "— Каждый вопрос должен проверять понимание концепта, а не простое запоминание определения.\n"
        #     "— Вопросы должны максимально различаться по смыслу и формулировкам, не допускать перефразирования одного и того же.\n"
        #     "— Не используй слова 'всегда', 'никогда' и другие универсальные утверждения.\n"
        #     "— Дистракторы (неправильные варианты в multiple_choice) должны быть правдоподобны и не вызывать сомнений своей искусственностью.\n"
        #     "— Каждый вопрос обязательно должен быть связан с одним из переданных концептов.\n"
        #     "— Для каждого вопроса в поле 'related_concept' укажи, к какому именно термину он относится из списка выше.\n"
        #     "— НЕ создавай вопросы, похожие на эти (сравнивай по смыслу, теме и структуре!):\n"
        #     f"{avoid_part}\n\n"
        #
        #     "СТРОГИЙ формат ответа — JSON массив (list) из словарей, где каждый словарь (объект) обязательно содержит такие поля:\n"
        #     "  'question': (str) — текст вопроса (до 180 символов)\n"
        #     "  'type': (str) — тип вопроса: 'multiple_choice', 'true_false'\n"
        #     "  'options': (list или null) — Список строк-ответов если multiple_choice (4-6 вариантов), иначе null\n"
        #     "  'correct_answer': (str) — верный вариант ('a') — для multiple_choice; 'True'/'False' — для true_false;\n"
        #     "  'related_concept': (str) — термин из переданного списка\n\n"
        #
        #     "Пример одного объекта для multiple_choice:\n"
        #     "{"
        #     "  'question': 'Какой основной принцип способа loci?',"
        #     "  'type': 'multiple_choice',"
        #     "  'options': ['a) Использование пространства памяти', 'b) Цветовые ассоциации', 'c) Ассоциативные числа', 'd) Усиление эмоций'],"
        #     "  'correct_answer': 'a',"
        #     "  'related_concept': 'метод локи'"
        #     "}\n\n"
        #
        #     "Дополнительные указания:\n"
        #     "— НЕ добавляй пояснений, комментариев, текста до и после/после JSON — только сам JSON-массив!\n"
        #     "— Соблюдай формат строго, иначе твой ответ не пройдет автоматическую проверку.\n"
        #     "— Вопросы должны быть максимально информативны для учебного теста.\n"
        # )

        prompt = ( f"""Ты — генератор учебных вопросов для интеллектуальной системы квизов. Сгенерируй {self.questions_count} уникальных образовательных вопросов уровня сложности '{self.difficulty}' на основе концептов:
            {concept_part}
            
            Типы вопросов (80% multiple_choice, 20% true_false):
            1. multiple_choice: 4-6 вариантов ответа
            2. true_false: вопрос с ответом True/False
            
            Требования:
            - Каждый вопрос связан с одним концептом из списка
            - Вопросы проверяют понимание, а не запоминание
            - Дистракторы правдоподобны
            - Избегай слов "всегда", "никогда"
            - НЕ создавай вопросы, похожие на эти:
            {avoid_part}
            
            СТРОГИЙ формат JSON (массив объектов):
            
            [
              {{
                "question": "текст Вопроса (макс 180 символов)",
                "type": "multiple_choice",
                "options": ["вариант1", "вариант2", ...] для multiple_choice,
                "related_concept": "конкретный концепт из списка концептов, на котором базируется вопрос",
                "correct_answer": "вариант1" 
              }},
              {{
                "question": "Текст вопроса-утверждения",
                "type": "true_false",
                "options": ["True", "False"],
                "related_concept": "конкретный концепт из списка концептов, на котором базируется вопрос"
                "correct_answer": "True"
              }}
            ]
            
            ВАЖНО: Возвращай ТОЛЬКО JSON-массив, без комментариев!"""
        )

        logger.info(f"[STEP] Prompt ready:\n{prompt}")
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
                logger.info(f"[VALID] Question #{idx + 1} passed validation")
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
            history: Set[str]
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
            logger.info(f"[VALID] Question #{idx + 1} added as unique")

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
            logger.info(
                f"[UPDATE] Processed question #{idx + 1}:\n"
                f"[ORIGINAL] {json.dumps(original, ensure_ascii=False, indent=2)}\n"
                f"[UPDATED]  {json.dumps(q, ensure_ascii=False, indent=2)}"
            )

        logger.info(f"[STEP] Post-processing complete: {len(questions)} questions processed")
        return questions
