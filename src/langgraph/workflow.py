"""
LangGraph Workflow - связывание агентов через граф состояний
"""

import logging
from typing import Dict, Any
from langgraph.graph import StateGraph, END
from .state_schema import GraphState
from ..agents.enhanced_parser_agent import EnhancedParserAgent
from ..agents.adapted_quiz_agent import AdaptedQuizAgent

logger = logging.getLogger(__name__)


class QuizGenerationWorkflow:
    """
    LangGraph workflow для генерации квизов

    Граф:
    START -> Parser Agent -> Quiz Agent -> END
    """

    def __init__(
        self,
        gigachat_credentials: str,
        quiz_config: Dict[str, Any],
        use_rag: bool = True,
        enable_web_search: bool = False
    ):
        """
        Args:
            gigachat_credentials: GigaChat API ключ
            quiz_config: конфигурация для Quiz Agent
            use_rag: использовать RAG в Parser Agent
            enable_web_search: включить веб-поиск
        """
        logger.info("Инициализация LangGraph Workflow...")

        # Инициализация агентов
        self.parser_agent = EnhancedParserAgent(
            credentials=gigachat_credentials,
            use_rag=use_rag,
            enable_web_search=enable_web_search
        )

        self.quiz_agent = AdaptedQuizAgent(config=quiz_config)

        # Построение графа
        self.graph = self._build_graph()

        logger.info("LangGraph Workflow готов к работе")

    def _build_graph(self) -> StateGraph:
        """Построение LangGraph графа"""
        logger.info("Построение LangGraph графа...")

        # Определяем граф состояний
        workflow = StateGraph(GraphState)

        # Добавляем узлы (агенты)
        workflow.add_node("parser", self.parser_agent.process)
        workflow.add_node("quiz_generator", self.quiz_agent.process)

        # Добавляем рёбра (связи между агентами)
        workflow.set_entry_point("parser")  # Начинаем с парсера
        workflow.add_edge("parser", "quiz_generator")  # Parser -> Quiz
        workflow.add_edge("quiz_generator", END)  # Quiz -> Конец

        # Компилируем граф
        compiled_graph = workflow.compile()

        logger.info("Граф успешно построен: parser -> quiz_generator -> END")
        return compiled_graph

    def run(self, lecture_text: str) -> Dict[str, Any]:
        """
        Запуск workflow

        Args:
            lecture_text: текст лекции для анализа

        Returns:
            финальное состояние с квизом
        """
        logger.info("\n" + "="*70)
        logger.info("ЗАПУСК WORKFLOW")
        logger.info("="*70)

        # Инициализация начального состояния
        initial_state: GraphState = {
            "lecture_text": lecture_text,
            "key_facts": [],
            "concepts": [],
            "rag_context": None,
            "quiz_questions": None,
            "messages": [],
            "error": None
        }

        try:
            # Запуск графа
            final_state = self.graph.invoke(initial_state)

            logger.info("\n" + "="*70)
            logger.info("WORKFLOW ЗАВЕРШЁН УСПЕШНО")
            logger.info("="*70)

            # Вывод статистики
            logger.info(f"Извлечено фактов: {len(final_state['key_facts'])}")
            logger.info(f"Создано концепций: {len(final_state['concepts'])}")
            if final_state.get('quiz_questions'):
                logger.info(f"Сгенерировано вопросов: {len(final_state['quiz_questions'])}")

            logger.info("\nИстория выполнения:")
            for msg in final_state["messages"]:
                logger.info(f"  - {msg}")

            return final_state

        except Exception as e:
            logger.error(f"\n✗ КРИТИЧЕСКАЯ ОШИБКА WORKFLOW: {e}")
            raise

    def visualize_graph(self, output_path: str = "workflow_graph.png"):
        """
        Визуализация графа (опционально)

        Args:
            output_path: путь для сохранения изображения
        """
        try:
            from IPython.display import Image, display

            # Генерируем Mermaid диаграмму
            mermaid_png = self.graph.get_graph().draw_mermaid_png()

            # Сохраняем
            with open(output_path, 'wb') as f:
                f.write(mermaid_png)

            logger.info(f"График сохранен в {output_path}")

            # Отображаем (если в Jupyter)
            display(Image(mermaid_png))

        except Exception as e:
            logger.warning(f"Визуализация недоступна: {e}")
