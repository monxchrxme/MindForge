"""
Quiz Agent - генерирует квизы на основе концепций из Parser Agent
"""

from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field, validator
import logging
import json
import random
import re

from .base_agent import BaseAgent
from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)


# ============================================================================
# PYDANTIC МОДЕЛИ
# ============================================================================

class Question(BaseModel):
    """Модель одного вопроса квиза"""
    question_text: str = Field(description="Текст вопроса", min_length=5)
    question_type: str = Field(description="Тип: multiple_choice, true_false, short_answer")
    options: List[str] = Field(default_factory=list, description="Варианты ответа")
    correct_answer: str = Field(description="Правильный ответ")
    explanation: str = Field(description="Объяснение ответа", min_length=10)
    difficulty: str = Field(default="medium", description="Сложность: easy, medium, hard")
    concept_reference: Optional[str] = Field(default=None, description="Ссылка на концепцию")

    @validator('question_type')
    def validate_question_type(cls, v):
        allowed = ['multiple_choice', 'true_false', 'short_answer']
        if v not in allowed:
            logger.warning(f"Неверный question_type '{v}', используем 'short_answer'")
            return 'short_answer'
        return v

    @validator('difficulty')
    def validate_difficulty(cls, v):
        allowed = ['easy', 'medium', 'hard']
        if v not in allowed:
            logger.warning(f"Неверный difficulty '{v}', используем 'medium'")
            return 'medium'
        return v

    @validator('options')
    def validate_options(cls, v, values):
        question_type = values.get('question_type')
        if question_type == 'multiple_choice' and len(v) < 2:
            logger.warning(f"multiple_choice должен содержать минимум 2 варианта")
            while len(v) < 4:
                v.append(f"Вариант {len(v) + 1}")
        return v

    class Config:
        extra = "ignore"


class Quiz(BaseModel):
    """Модель полного квиза"""
    questions: List[Question] = Field(description="Список вопросов")
    total_questions: int = Field(description="Общее количество")
    difficulty_distribution: Dict[str, int] = Field(default_factory=dict)
    type_distribution: Dict[str, int] = Field(default_factory=dict)

    def calculate_distributions(self):
        """Вычисление статистики"""
        self.difficulty_distribution = {
            'easy': sum(1 for q in self.questions if q.difficulty == 'easy'),
            'medium': sum(1 for q in self.questions if q.difficulty == 'medium'),
            'hard': sum(1 for q in self.questions if q.difficulty == 'hard')
        }
        self.type_distribution = {
            'multiple_choice': sum(1 for q in self.questions if q.question_type == 'multiple_choice'),
            'true_false': sum(1 for q in self.questions if q.question_type == 'true_false'),
            'short_answer': sum(1 for q in self.questions if q.question_type == 'short_answer')
        }


# ============================================================================
# QUIZ AGENT
# ============================================================================

class QuizAgent(BaseAgent):
    """Агент для генерации квизов"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__("QuizAgent")
        self.config = config

        from ..utils.gigachat_client import create_gigachat_quiz_client
        self.llm_client = create_gigachat_quiz_client()

        self.num_questions = config['quiz'].get('num_questions', 7)
        self.question_types = config['quiz'].get('question_types', [
            'multiple_choice', 'true_false', 'short_answer'
        ])
        self.temperature = config['gigachat'].get('temperature', 0.7)
        self.few_shot_examples = self._load_few_shot_examples()

        logger.info(f"QuizAgent инициализирован: {self.num_questions} вопросов")

    def generate_quiz(self, parsed_note: Any) -> Quiz:
        """Главный метод генерации"""
        self.log_input(parsed_note)
        logger.info("Начало генерации квиза")

        if hasattr(parsed_note, 'concepts'):
            concepts = parsed_note.concepts
        else:
            raise ValueError("parsed_note должен содержать concepts")

        if not concepts:
            raise ValueError("Не найдено концепций")

        logger.info(f"Обнаружено {len(concepts)} концепций")

        questions_distribution = self._distribute_questions(concepts)
        all_questions = []

        for concept, num_questions in questions_distribution:
            concept_title = getattr(concept, 'title', 'Unknown')
            logger.info(f"Генерация {num_questions} вопросов для: {concept_title}")

            try:
                concept_questions = self._generate_questions_for_concept(concept, num_questions)
                all_questions.extend(concept_questions)
                logger.info(f"Успешно сгенерировано {len(concept_questions)} вопросов")
            except Exception as e:
                logger.error(f"Ошибка генерации для '{concept_title}': {e}")
                continue

        if len(all_questions) == 0:
            logger.error("Не удалось сгенерировать ни одного вопроса!")
            return Quiz(questions=[], total_questions=0)

        random.shuffle(all_questions)
        all_questions = all_questions[:self.num_questions]

        quiz = Quiz(questions=all_questions, total_questions=len(all_questions))
        quiz.calculate_distributions()

        logger.info(f"Квиз сгенерирован: {quiz.total_questions} вопросов")
        return quiz

    def _distribute_questions(self, concepts: List[Any]) -> List[tuple]:
        """Распределение вопросов"""
        base = self.num_questions // len(concepts)
        remainder = self.num_questions % len(concepts)
        distribution = []

        importance_order = {'high': 3, 'medium': 2, 'low': 1}
        sorted_concepts = sorted(
            concepts,
            key=lambda c: importance_order.get(getattr(c, 'importance', 'medium'), 1),
            reverse=True
        )

        for i, concept in enumerate(sorted_concepts):
            extra = 1 if i < remainder else 0
            num = max(1, base + extra)
            distribution.append((concept, num))

        return distribution

    def _generate_questions_for_concept(self, concept: Any, num_questions: int) -> List[Question]:
        """Генерация вопросов для концепции"""
        questions = []
        types = self._select_question_types(num_questions)

        for i, q_type in enumerate(types):
            try:
                prompt = self._build_question_prompt(concept, q_type, i+1)
                question = self._generate_single_question(prompt, getattr(concept, 'title', 'Unknown'))
                questions.append(question)
            except Exception as e:
                logger.error(f"Ошибка генерации вопроса {i+1}: {e}")
                continue

        return questions

    def _generate_single_question(self, prompt: str, concept_title: str) -> Question:
        """Генерация одного вопроса"""
        max_retries = 2

        for attempt in range(max_retries):
            try:
                logger.debug(f"Попытка {attempt + 1}/{max_retries}: '{concept_title}'")

                message = HumanMessage(content=prompt)
                response = self.llm_client.chat(message, temperature=self.temperature)
                response_text = response.content

                logger.debug(f"Получен ответ ({len(response_text)} символов)")

                json_str = self._extract_json(response_text)
                json_str = self._validate_and_fix_json(json_str)

                question = Question.parse_raw(json_str)
                question.concept_reference = concept_title

                logger.info(f"✓ Вопрос сгенерирован для '{concept_title}'")
                return question

            except Exception as e:
                logger.error(f"✗ Попытка {attempt + 1} не удалась: {e}")

                if attempt < max_retries - 1:
                    logger.info("Повторная попытка с упрощенным промптом...")
                    prompt = self._build_simplified_prompt(concept_title)
                else:
                    logger.error(f"Все попытки исчерпаны для '{concept_title}'")
                    return self._create_fallback_question(concept_title)

        return self._create_fallback_question(concept_title)

    def _extract_json(self, text: str) -> str:
        """Извлечение JSON из текста"""
        # Вариант 1: ``````
        match = re.search(r'``````', text, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Вариант 2: ``````
        match = re.search(r'``````', text, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Вариант 3: { ... } с вложенностью
        start = text.find('{')
        if start == -1:
            return text

        brace_count = 0
        for i in range(start, len(text)):
            if text[i] == '{':
                brace_count += 1
            elif text[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    return text[start:i + 1].strip()

        # Вариант 4: первый { до последнего }
        end = text.rfind('}')
        if start != -1 and end != -1 and end > start:
            return text[start:end + 1].strip()

        return text.strip()

    def _validate_and_fix_json(self, json_str: str) -> str:
        """Валидация и исправление JSON"""
        try:
            json.loads(json_str)
            return json_str
        except json.JSONDecodeError:
            logger.warning("JSON невалиден, попытка исправления...")

            # Убираем trailing запятые
            json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
            # Одинарные кавычки -> двойные
            json_str = json_str.replace("'", '"')
            # Control characters
            json_str = re.sub(r'[\x00-\x1f\x7f]', '', json_str)

            try:
                json.loads(json_str)
                logger.info("JSON исправлен")
                return json_str
            except:
                logger.error("Не удалось исправить JSON")
                return json_str

    def _build_question_prompt(self, concept: Any, question_type: str, question_number: int) -> str:
        """Построение промпта"""
        examples = self.few_shot_examples.get(question_type, "")

        title = getattr(concept, 'title', 'Unknown')
        description = getattr(concept, 'description', '')

        prompt = f"""Ты — эксперт по созданию образовательных тестов.

    КОНЦЕПЦИЯ:
    {title}
    {description}

    ЗАДАЧА: Создай вопрос типа "{question_type}"

    КРИТИЧЕСКИ ВАЖНО:
    - НЕ используй LaTeX формулы ($ или \\)
    - НЕ используй escape-последовательности в тексте
    - Используй обычный текст для математических символов
    - Пример: вместо $\\Delta x$ пиши "Δx" или "дельта x"

    ПРИМЕРЫ:
    {examples}

    ТРЕБОВАНИЯ К JSON:
    1. "question_text" - строка вопроса БЕЗ LaTeX
    2. "question_type" - "{question_type}"
    3. "options" - массив {'4 варианта для multiple_choice' if question_type == 'multiple_choice' else 'пустой []'}
    4. "correct_answer" - правильный ответ
    5. "explanation" - объяснение
    6. "difficulty" - "easy", "medium" или "hard"

    Верни ТОЛЬКО чистый JSON без markdown блоков!"""

        return prompt

    def _load_few_shot_examples(self) -> Dict[str, str]:
        """Загрузка примеров"""
        return {
            'multiple_choice': """
Вопрос: Что такое Python?
A) Язык программирования
B) Змея
C) Операционная система  
D) База данных
Ответ: A) Язык программирования
""",
            'true_false': """
Вопрос: Python поддерживает ООП?
Ответ: True
""",
            'short_answer': """
Вопрос: Кто создал Python?
Ответ: Гвидо ван Россум
"""
        }

    def _select_question_types(self, num: int) -> List[str]:
        """Выбор типов вопросов"""
        types = []
        for i in range(num):
            types.append(self.question_types[i % len(self.question_types)])
        random.shuffle(types)
        return types

    def _build_simplified_prompt(self, concept_title: str) -> str:
        """Упрощенный промпт"""
        return f"""Создай простой вопрос по теме "{concept_title}".

JSON:
{{
    "question_text": "Что такое {concept_title}?",
    "question_type": "short_answer",
    "options": [],
    "correct_answer": "Краткий ответ",
    "explanation": "Объяснение",
    "difficulty": "easy"
}}

Только JSON!"""

    def _create_fallback_question(self, concept_title: str) -> Question:
        """Fallback вопрос"""
        return Question(
            question_text=f"Объясните: {concept_title}",
            question_type="short_answer",
            options=[],
            correct_answer="См. материалы лекции",
            explanation="Вопрос создан автоматически",
            difficulty="medium",
            concept_reference=concept_title
        )

    def process(self, input_data: Any) -> Quiz:
        """Интерфейс BaseAgent"""
        return self.generate_quiz(input_data)
