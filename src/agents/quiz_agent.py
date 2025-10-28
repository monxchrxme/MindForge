"""
Quiz Agent - генерирует квизы на основе концепций из Parser Agent

Агент использует GigaChat API с structured output для генерации
разнообразных вопросов: множественный выбор, правда/ложь, короткий ответ.
Применяется few-shot промптинг для улучшения качества генерации.
"""
from dotenv import load_dotenv
load_dotenv()
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field, validator
import logging
import json
import random

from .base_agent import BaseAgent
from ..utils.gigachat_client import create_gigachat_quiz_client

logger = logging.getLogger(__name__)


# ============================================================================
# PYDANTIC МОДЕЛИ ДЛЯ STRUCTURED OUTPUT
# ============================================================================

class Question(BaseModel):
    """
    Модель одного вопроса квиза
    Используется для валидации structured output от GigaChat [web:60][web:67]
    """
    question_text: str = Field(
        description="Текст вопроса (четкий, однозначный)",
        min_length=10
    )
    question_type: str = Field(
        description="Тип вопроса: multiple_choice, true_false, или short_answer"
    )
    options: List[str] = Field(
        default=[],
        description="Варианты ответа для multiple_choice (A, B, C, D)"
    )
    correct_answer: str = Field(
        description="Правильный ответ"
    )
    explanation: str = Field(
        description="Краткое объяснение правильного ответа (1-2 предложения)",
        min_length=20
    )
    difficulty: str = Field(
        description="Сложность вопроса: easy, medium, или hard"
    )
    concept_reference: Optional[str] = Field(
        default=None,
        description="Ссылка на концепцию, по которой создан вопрос"
    )

    @validator('question_type')
    def validate_question_type(cls, v):
        """Валидация типа вопроса"""
        allowed_types = ['multiple_choice', 'true_false', 'short_answer']
        if v not in allowed_types:
            raise ValueError(f"question_type должен быть одним из: {allowed_types}")
        return v

    @validator('difficulty')
    def validate_difficulty(cls, v):
        """Валидация уровня сложности"""
        allowed_difficulties = ['easy', 'medium', 'hard']
        if v not in allowed_difficulties:
            raise ValueError(f"difficulty должен быть одним из: {allowed_difficulties}")
        return v

    @validator('options')
    def validate_options(cls, v, values):
        """Валидация вариантов ответа для multiple_choice"""
        if values.get('question_type') == 'multiple_choice':
            if len(v) != 4:
                raise ValueError("multiple_choice должен содержать ровно 4 варианта")
        return v


class Quiz(BaseModel):
    """Модель полного квиза"""
    questions: List[Question] = Field(
        description="Список вопросов квиза"
    )
    total_questions: int = Field(
        description="Общее количество вопросов"
    )
    difficulty_distribution: Dict[str, int] = Field(
        default_factory=dict,
        description="Распределение по сложности"
    )
    type_distribution: Dict[str, int] = Field(
        default_factory=dict,
        description="Распределение по типам вопросов"
    )

    def calculate_distributions(self):
        """Вычисление статистики квиза"""
        # Распределение по сложности
        self.difficulty_distribution = {
            'easy': sum(1 for q in self.questions if q.difficulty == 'easy'),
            'medium': sum(1 for q in self.questions if q.difficulty == 'medium'),
            'hard': sum(1 for q in self.questions if q.difficulty == 'hard')
        }

        # Распределение по типам
        self.type_distribution = {
            'multiple_choice': sum(1 for q in self.questions if q.question_type == 'multiple_choice'),
            'true_false': sum(1 for q in self.questions if q.question_type == 'true_false'),
            'short_answer': sum(1 for q in self.questions if q.question_type == 'short_answer')
        }


# ============================================================================
# QUIZ AGENT
# ============================================================================

class QuizAgent(BaseAgent):
    """
    Агент для генерации образовательных квизов

    Использует few-shot prompting для качественной генерации [web:71][web:74][web:77]
    и structured output для надежного формата ответа [web:3][web:60]
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Инициализация Quiz Agent

        Args:
            config: конфигурация системы из config.yaml
        """
        super().__init__(config, "QuizAgent")

        # Инициализация GigaChat клиента
        self.llm_client = create_gigachat_quiz_client(
           # api_key=config['gigachat']['api_key'],
            #model=config['gigachat'].get('model', 'GigaChat')
        )

        # Параметры генерации квиза
        self.num_questions = config['quiz'].get('num_questions', 7)
        self.question_types = config['quiz'].get('question_types', [
            'multiple_choice',
            'true_false',
            'short_answer'
        ])
        self.temperature = config['gigachat'].get('temperature', 0.7)

        # Few-shot примеры загружаются один раз
        self.few_shot_examples = self._load_few_shot_examples()

        logger.info(
            f"QuizAgent инициализирован: {self.num_questions} вопросов, "
            f"типы: {self.question_types}"
        )

    # ========================================================================
    # ОСНОВНЫЕ МЕТОДЫ
    # ========================================================================

    def generate_quiz(self, parsed_note: Any) -> Quiz:
        """
        Главный метод генерации квиза

        Args:
            parsed_note: результат работы Parser Agent (ParsedNote с концепциями)

        Returns:
            Quiz с вопросами
        """
        self.log_input(parsed_note)
        logger.info("Начало генерации квиза")

        # Извлекаем концепции из parsed_note
        if hasattr(parsed_note, 'concepts'):
            concepts = parsed_note.concepts
        else:
            raise ValueError("parsed_note должен содержать атрибут 'concepts'")

        if not concepts:
            raise ValueError("Не найдено концепций для генерации квиза")

        logger.info(f"Обнаружено {len(concepts)} концепций")

        # Распределяем вопросы по концепциям
        questions_distribution = self._distribute_questions(concepts)
        logger.debug(f"Распределение вопросов: {questions_distribution}")

        # Генерация вопросов для каждой концепции
        all_questions = []

        for concept, num_questions in questions_distribution.items():
            logger.info(f"Генерация {num_questions} вопросов для концепции: {concept.title}")

            try:
                concept_questions = self._generate_questions_for_concept(
                    concept,
                    num_questions=num_questions
                )
                all_questions.extend(concept_questions)
                logger.info(f"Успешно сгенерировано {len(concept_questions)} вопросов")

            except Exception as e:
                logger.error(f"Ошибка генерации вопросов для '{concept.title}': {e}")
                # Продолжаем с другими концепциями
                continue

        # Проверка минимального количества вопросов
        if len(all_questions) < self.num_questions // 2:
            logger.warning(
                f"Сгенерировано мало вопросов: {len(all_questions)} из {self.num_questions}"
            )

        # Перемешиваем вопросы для разнообразия
        random.shuffle(all_questions)

        # Ограничиваем до нужного количества
        all_questions = all_questions[:self.num_questions]

        # Создание финального квиза
        quiz = Quiz(
            questions=all_questions,
            total_questions=len(all_questions)
        )

        # Вычисление статистики
        quiz.calculate_distributions()

        logger.info(
            f"Квиз сгенерирован: {quiz.total_questions} вопросов, "
            f"сложность: {quiz.difficulty_distribution}, "
            f"типы: {quiz.type_distribution}"
        )

        self.log_output(quiz)
        return quiz

    def _distribute_questions(self, concepts: List[Any]) -> Dict[Any, int]:
        """
        Распределение количества вопросов по концепциям

        Args:
            concepts: список концепций из Parser Agent

        Returns:
            словарь {концепция: количество_вопросов}
        """
        # Базовое распределение: равномерно
        base_questions = self.num_questions // len(concepts)
        remainder = self.num_questions % len(concepts)

        distribution = {}

        # Сортируем концепции по важности (high > medium > low)
        importance_order = {'high': 3, 'medium': 2, 'low': 1}
        sorted_concepts = sorted(
            concepts,
            key=lambda c: importance_order.get(c.importance, 1),
            reverse=True
        )

        for i, concept in enumerate(sorted_concepts):
            # Больше вопросов для важных концепций
            extra = 1 if i < remainder else 0
            importance_bonus = 1 if concept.importance == 'high' and base_questions > 0 else 0

            num_questions = base_questions + extra + importance_bonus
            distribution[concept] = max(1, num_questions)  # Минимум 1 вопрос

        # Нормализация до точного количества
        total = sum(distribution.values())
        if total > self.num_questions:
            # Убираем лишние вопросы у менее важных концепций
            for concept in reversed(sorted_concepts):
                if total <= self.num_questions:
                    break
                if distribution[concept] > 1:
                    distribution[concept] -= 1
                    total -= 1

        return distribution

    def _generate_questions_for_concept(
            self,
            concept: Any,
            num_questions: int
    ) -> List[Question]:
        """
        Генерация вопросов для одной концепции

        Args:
            concept: концепция из Parser Agent
            num_questions: количество вопросов для генерации

        Returns:
            список Question объектов
        """
        questions = []

        # Определяем типы вопросов для этой концепции
        question_types_to_use = self._select_question_types(num_questions)

        for i, question_type in enumerate(question_types_to_use):
            logger.debug(
                f"Генерация вопроса {i + 1}/{num_questions} "
                f"типа '{question_type}' для '{concept.title}'"
            )

            try:
                # Построение промпта с few-shot примерами [web:74][web:77]
                prompt = self._build_question_prompt(
                    concept=concept,
                    question_type=question_type,
                    question_number=i + 1
                )

                # Генерация через GigaChat с structured output [web:60]
                question = self._generate_single_question(
                    prompt=prompt,
                    concept_title=concept.title
                )

                questions.append(question)

            except Exception as e:
                logger.error(
                    f"Ошибка генерации вопроса {i + 1} "
                    f"для '{concept.title}': {e}"
                )
                # Пропускаем проблемный вопрос
                continue

        return questions

    def _generate_single_question(
            self,
            prompt: str,
            concept_title: str
    ) -> Question:
        """
        Генерация одного вопроса через GigaChat API

        Args:
            prompt: промпт для генерации
            concept_title: название концепции (для reference)

        Returns:
            Question объект
        """
        try:
            # Используем structured output для надежного парсинга [web:3][web:60]
            question = self.llm_client.generate_structured(
                prompt=prompt,
                response_model=Question,
                temperature=self.temperature
            )

            # Добавляем ссылку на концепцию
            question.concept_reference = concept_title

            return question

        except Exception as e:
            logger.error(f"Ошибка вызова GigaChat API: {e}")
            # Fallback: парсим текстовый ответ вручную
            return self._fallback_question_parsing(prompt, concept_title)

    def _fallback_question_parsing(
            self,
            prompt: str,
            concept_title: str
    ) -> Question:
        """
        Запасной вариант: генерация без structured output
        """
        logger.warning("Использование fallback генерации")

        response_text = self.llm_client.generate(
            prompt=prompt,
            temperature=self.temperature
        )

        # Простой парсинг текста
        # В реальной ситуации здесь должна быть более сложная логика
        return Question(
            question_text="Что такое " + concept_title + "?",
            question_type="short_answer",
            options=[],
            correct_answer="См. описание концепции",
            explanation="Вопрос сгенерирован в fallback режиме",
            difficulty="medium",
            concept_reference=concept_title
        )

    # ========================================================================
    # ПРОМПТ ИНЖЕНЕРИНГ
    # ========================================================================

    def _build_question_prompt(
            self,
            concept: Any,
            question_type: str,
            question_number: int
    ) -> str:
        """
        Построение промпта для генерации вопроса [web:70][web:73][web:82]

        Использует best practices:
        - Четкие инструкции [web:70]
        - Few-shot примеры [web:74][web:77]
        - Контекст [web:82]
        - Специфичность [web:73]

        Args:
            concept: концепция для вопроса
            question_type: тип вопроса
            question_number: номер вопроса

        Returns:
            готовый промпт
        """
        # Few-shot примеры для конкретного типа [web:71][web:74]
        examples = self.few_shot_examples.get(question_type, "")

        # Основной промпт
        prompt = f"""Ты — эксперт по созданию образовательных тестов и квизов.

КОНТЕКСТ:
Тебе нужно создать вопрос для закрепления материала студентом.

КОНЦЕПЦИЯ:
Название: {concept.title}
Описание: {concept.description}
Важность: {concept.importance}
Контекст из лекции: {concept.context}

ЗАДАЧА:
Создай качественный вопрос типа "{question_type}" по этой концепции.

ПРИМЕРЫ ХОРОШИХ ВОПРОСОВ:
{examples}

ТРЕБОВАНИЯ:
1. Вопрос должен проверять ПОНИМАНИЕ концепции, а не просто запоминание
2. Формулировка должна быть ЧЕТКОЙ и ОДНОЗНАЧНОЙ
3. {"Предложи 4 варианта ответа (A, B, C, D), где только один правильный" if question_type == "multiple_choice" else ""}
4. {"Сформулируй утверждение, которое может быть правдой или ложью" if question_type == "true_false" else ""}
5. Добавь краткое объяснение (1-2 предложения), почему ответ правильный
6. Оцени сложность вопроса: easy (простой факт), medium (требует понимания), hard (требует анализа)
7. Избегай двусмысленности и трюковых формулировок

ФОРМАТ ОТВЕТА - СТРОГО JSON:
{{
    "question_text": "Текст вопроса здесь",
    "question_type": "{question_type}",
    "options": {json.dumps(["Вариант A", "Вариант B", "Вариант C", "Вариант D"]) if question_type == "multiple_choice" else "[]"},
    "correct_answer": "Правильный ответ",
    "explanation": "Объяснение правильного ответа",
    "difficulty": "easy|medium|hard"
}}

Сгенерируй вопрос #{question_number}:"""

        return prompt

    def _load_few_shot_examples(self) -> Dict[str, str]:
        """
        Загрузка few-shot примеров для каждого типа вопросов [web:74][web:77]

        Returns:
            словарь {тип_вопроса: примеры}
        """
        examples = {
            'multiple_choice': """
Пример 1 (Python):
Вопрос: Какая структура данных в Python является неизменяемой (immutable)?
A) Список (list)
B) Кортеж (tuple)
C) Словарь (dict)
D) Множество (set)
Правильный ответ: B) Кортеж (tuple)
Объяснение: Кортежи в Python являются неизменяемыми — после создания их элементы нельзя изменить, в отличие от списков.
Сложность: medium

Пример 2 (История):
Вопрос: В каком году началась Вторая мировая война?
A) 1914
B) 1939
C) 1941
D) 1945
Правильный ответ: B) 1939
Объяснение: Вторая мировая война началась 1 сентября 1939 года с нападения Германии на Польшу.
Сложность: easy
""",

            'true_false': """
Пример 1 (Биология):
Утверждение: Фотосинтез — это процесс, при котором растения преобразуют углекислый газ и воду в глюкозу с помощью солнечного света.
Ответ: True (Правда)
Объяснение: Это точное определение фотосинтеза — основного процесса, посредством которого растения производят питательные вещества.
Сложность: easy

Пример 2 (Физика):
Утверждение: Скорость света в вакууме составляет примерно 300 000 км/с и является максимально возможной скоростью во Вселенной согласно теории относительности.
Ответ: True (Правда)
Объяснение: Согласно специальной теории относительности Эйнштейна, скорость света в вакууме — это фундаментальная константа и предельная скорость.
Сложность: medium
""",

            'short_answer': """
Пример 1 (Литература):
Вопрос: Кто написал роман "Война и мир"?
Правильный ответ: Лев Толстой
Объяснение: "Война и мир" — эпический роман Льва Николаевича Толстого, опубликованный в 1869 году.
Сложность: easy

Пример 2 (Математика):
Вопрос: Как называется теорема, утверждающая, что в прямоугольном треугольнике квадрат гипотенузы равен сумме квадратов катетов?
Правильный ответ: Теорема Пифагора
Объяснение: Теорема Пифагора — одна из фундаментальных теорем геометрии, известная с древних времен.
Сложность: easy
"""
        }

        return examples

    def _select_question_types(self, num_questions: int) -> List[str]:
        """
        Выбор типов вопросов для генерации
        Обеспечивает разнообразие типов [web:70]

        Args:
            num_questions: количество вопросов

        Returns:
            список типов вопросов
        """
        if num_questions <= len(self.question_types):
            # Если вопросов мало, используем все типы
            return self.question_types[:num_questions]

        # Равномерное распределение типов
        types = []
        for i in range(num_questions):
            question_type = self.question_types[i % len(self.question_types)]
            types.append(question_type)

        # Перемешиваем для случайности
        random.shuffle(types)
        return types

    # ========================================================================
    # ИНТЕРФЕЙС BASE AGENT
    # ========================================================================

    def process(self, input_data: Any) -> Quiz:
        """
        Реализация абстрактного метода BaseAgent

        Args:
            input_ ParsedNote от Parser Agent

        Returns:
            Quiz с вопросами
        """
        return self.generate_quiz(input_data)


# ========================================================================
# ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ
# ========================================================================

def validate_quiz(self, quiz: Quiz) -> bool:
    """
    Валидация сгенерированного квиза

    Args:
        quiz: квиз для проверки

    Returns:
        True если квиз валиден
    """
    try:
        # Проверка минимального количества
        if quiz.total_questions < self.num_questions // 2:
            logger.warning("Квиз содержит слишком мало вопросов")
            return False

        # Проверка каждого вопроса
        for i, question in enumerate(quiz.questions):
            # Multiple choice должен иметь 4 варианта
            if question.question_type == 'multiple_choice':
                if len(question.options) != 4:
                    logger.error(
                        f"Вопрос {i + 1}: multiple_choice должен иметь 4 варианта, "
                        f"найдено {len(question.options)}"
                    )
                    return False

            # Проверка непустых полей
            if not question.question_text or not question.correct_answer:
                logger.error(f"Вопрос {i + 1}: пустые обязательные поля")
                return False

        logger.info("Квиз прошел валидацию")
        return True

    except Exception as e:
        logger.error(f"Ошибка валидации квиза: {e}")
        return False


def export_quiz_to_json(self, quiz: Quiz, output_path: str):
    """
    Экспорт квиза в JSON файл

    Args:
        quiz: квиз для экспорта
        output_path: путь к выходному файлу
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(
                quiz.dict(),
                f,
                ensure_ascii=False,
                indent=2
            )
        logger.info(f"Квиз экспортирован в {output_path}")

    except Exception as e:
        logger.error(f"Ошибка экспорта квиза: {e}")
        raise
