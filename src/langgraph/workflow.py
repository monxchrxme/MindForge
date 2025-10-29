"""
LangGraph Workflow для координации Parser и Quiz агентов
"""

from langgraph.graph import StateGraph, END
from ..langgraph.state_schema import GraphState
from ..agents.parser_agent import ParserAgent  # ИЗМЕНЕНО: новый импорт
from ..agents.adapted_quiz_agent import AdaptedQuizAgent
import logging

logger = logging.getLogger(__name__)


class QuizGenerationWorkflow:
    """LangGraph Workflow для генерации квизов"""

    def __init__(
        self,
        gigachat_credentials: str,
        quiz_config: dict,
        use_rag: bool = True,
        enable_web_search: bool = False
    ):
        """
        Инициализация workflow

        Args:
            gigachat_credentials: GigaChat API credentials
            quiz_config: конфигурация для Quiz Agent
            use_rag: использовать ли RAG
            enable_web_search: проверять ли факты через веб-поиск
        """
        # ИЗМЕНЕНО: используем ParserAgent вместо EnhancedParserAgent
        self.parser_agent = ParserAgent(
            gigachat_credentials=gigachat_credentials,
            use_rag=use_rag,
            enable_web_search=enable_web_search
        )

        self.quiz_agent = AdaptedQuizAgent(config=quiz_config)

        # Построение графа
        self.workflow = self._build_workflow()

        logger.info(
            f"QuizGenerationWorkflow инициализирован "
            f"(RAG={use_rag}, WebSearch={enable_web_search})"
        )

    def _build_workflow(self) -> StateGraph:
        """Построение LangGraph workflow"""

        # Создание графа с GraphState
        workflow = StateGraph(GraphState)

        # Добавление узлов
        workflow.add_node("parser", self.parser_agent.process)
        workflow.add_node("quiz_generator", self.quiz_agent.process)

        # Определение рёбер
        workflow.set_entry_point("parser")
        workflow.add_edge("parser", "quiz_generator")
        workflow.add_edge("quiz_generator", END)

        # Компиляция
        compiled = workflow.compile()

        logger.info("✓ Workflow скомпилирован: parser → quiz_generator → END")
        return compiled

    def run(self, lecture_text: str) -> dict:
        """
        Запуск workflow

        Args:
            lecture_text: текст лекции

        Returns:
            финальное состояние с результатами
        """
        logger.info(f"Запуск workflow (текст: {len(lecture_text)} символов)")

        # Инициализация начального состояния
        initial_state = GraphState(
            lecture_text=lecture_text,
            key_facts=[],
            concepts=[],
            quiz_questions=[],
            messages=[],
            error=None
        )

        try:
            # Выполнение workflow
            final_state = self.workflow.invoke(initial_state)

            logger.info("="*70)
            logger.info("WORKFLOW ЗАВЕРШЁН УСПЕШНО")
            logger.info("="*70)
            logger.info(f"Фактов: {len(final_state.get('key_facts', []))}")
            logger.info(f"Концепций: {len(final_state.get('concepts', []))}")
            logger.info(f"Вопросов: {len(final_state.get('quiz_questions', []))}")
            logger.info("\nИстория выполнения:")
            for msg in final_state.get('messages', []):
                logger.info(f"  - {msg}")

            return final_state

        except Exception as e:
            logger.error(f"Ошибка выполнения workflow: {e}")
            import traceback
            traceback.print_exc()

            return {
                "error": str(e),
                "key_facts": [],
                "concepts": [],
                "quiz_questions": [],
                "messages": [f"Workflow error: {e}"]
            }
