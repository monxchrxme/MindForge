"""
State Schema - единая схема состояний для LangGraph
Обеспечивает совместимость данных между агентами
"""

from typing import List, Dict, Any, Optional, TypedDict
from pydantic import BaseModel, Field


class ConceptSchema(BaseModel):
    """Схема концепции - единый формат для обоих агентов"""
    title: str = Field(description="Название концепции")
    description: str = Field(description="Подробное описание")
    importance: str = Field(description="Уровень важности: high, medium, low")
    context: str = Field(default="", description="Контекст из лекции")


class GraphState(TypedDict):
    """
    Состояние LangGraph - передается между агентами

    Fields:
        lecture_text: исходный текст лекции
        key_facts: извлеченные ключевые факты (список строк)
        concepts: концепции в едином формате
        quiz_questions: сгенерированные вопросы
        messages: история сообщений для логирования
        error: информация об ошибках
    """
    lecture_text: str
    key_facts: List[str]
    concepts: List[Dict[str, Any]]  # Конвертируем в dict для TypedDict
    rag_context: Optional[List[str]]
    quiz_questions: Optional[List[Dict[str, Any]]]
    messages: List[str]
    error: Optional[str]
