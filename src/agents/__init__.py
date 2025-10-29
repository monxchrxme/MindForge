"""
Agents package
"""

from .base_agent import BaseAgent
from .parser_agent import ParserAgent  # ИЗМЕНЕНО
from .quiz_agent import QuizAgent, Quiz, Question
from .adapted_quiz_agent import AdaptedQuizAgent

__all__ = [
    'BaseAgent',
    'ParserAgent',  # ИЗМЕНЕНО
    'QuizAgent',
    'Quiz',
    'Question',
    'AdaptedQuizAgent'
]
