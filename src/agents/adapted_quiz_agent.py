"""
Adapted Quiz Agent - адаптация под LangGraph и единый формат данных
"""

from typing import Dict, Any, List
import logging
from .state_schema import GraphState, ConceptSchema
# Импортируем оригинальные классы из quiz_agent.py
from .quiz_agent import QuizAgent as OriginalQuizAgent, Quiz, Question

logger = logging.getLogger(__name__)


class AdaptedQuizAgent:
    """
    Обертка над оригинальным Quiz Agent для работы с LangGraph
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: конфигурация (совместима с оригинальным агентом)
        """
        logger.info("Инициализация Adapted Quiz Agent...")

        # Инициализируем оригинальный агент
        self.original_agent = OriginalQuizAgent(config)

        logger.info("Adapted Quiz Agent инициализирован")

    def process(self, state: GraphState) -> GraphState:
        """
        Обработка состояния из LangGraph

        Args:
            state: состояние с концепциями от Parser Agent

        Returns:
            обновленное состояние с вопросами квиза
        """
        logger.info("=" * 50)
        logger.info("Quiz Agent: начало генерации квиза")
        logger.info("=" * 50)

        try:
            # Проверка наличия концепций
            if not state.get("concepts"):
                raise ValueError("Отсутствуют концепции от Parser Agent")

            # Конвертация dict -> ConceptSchema
            concepts = [
                ConceptSchema(**concept_dict)
                for concept_dict in state["concepts"]
            ]

            logger.info(f"Получено {len(concepts)} концепций от Parser Agent")

            # Создаем mock ParsedNote для оригинального агента
            class ParsedNote:
                def __init__(self, concepts):
                    self.concepts = concepts

            parsed_note = ParsedNote(concepts)

            # Вызываем оригинальный метод генерации
            quiz = self.original_agent.generate_quiz(parsed_note)

            # Конвертируем Quiz в dict для состояния
            quiz_dict = quiz.dict()

            # Обновляем состояние
            state["quiz_questions"] = quiz_dict["questions"]
            state["messages"].append(
                f"Quiz: сгенерировано {len(quiz.questions)} вопросов"
            )

            logger.info(f"✓ Сгенерировано {len(quiz.questions)} вопросов")
            logger.info(f"✓ Распределение по сложности: {quiz.difficulty_distribution}")
            logger.info(f"✓ Распределение по типам: {quiz.type_distribution}")

            return state

        except Exception as e:
            logger.error(f"✗ Ошибка в Quiz Agent: {e}")
            state["error"] = str(e)
            state["messages"].append(f"Quiz: ОШИБКА - {e}")
            return state
