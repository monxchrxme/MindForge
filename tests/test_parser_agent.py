"""
Тесты для Parser Agent
"""

import pytest
from src.agents import ParserAgent


@pytest.fixture
def parser_agent():
    """Фикстура для создания Parser Agent"""
    return ParserAgent()


@pytest.fixture
def sample_state():
    """Фикстура с примером state"""
    return {
        "lecture_text": """
        Производная функции — это фундаментальное понятие математического анализа.
        Производная показывает скорость изменения функции в данной точке.
        Геометрический смысл производной — тангенс угла наклона касательной.
        """,
        "key_facts": [],
        "quiz_questions": [],
        "messages": [],
        "current_step": "start"
    }


def test_parser_agent_initialization(parser_agent):
    """Тест инициализации Parser Agent"""
    assert parser_agent.agent_name == "ParserAgent"
    assert parser_agent.llm is not None


def test_parser_agent_process(parser_agent, sample_state):
    """Тест обработки заметки Parser Agent"""
    result = parser_agent.process(sample_state)

    # Проверка, что факты извлечены
    assert "key_facts" in result
    assert len(result["key_facts"]) > 0

    # Проверка, что state обновлен
    assert result["current_step"] == "parsing_complete"

# TODO: Добавить тесты для различных типов заметок
# TODO: Добавить тесты для обработки ошибок API
# TODO: Добавить тесты для валидации фактов
