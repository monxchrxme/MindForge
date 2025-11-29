"""
Слой бизнес-логики (Business Logic Layer).

Этот пакет содержит специализированные AI-агенты для интеллектуальной обработки
учебных заметок и генерации квизов. Каждый агент отвечает за конкретную задачу
в конвейере обработки данных.

Архитектура агентов:
    - OrchestratorAgent: Центральный координатор, управляет состоянием сессии
    - ParserAgent: Извлекает образовательные концепты из текста заметки
    - FactCheckAgent: Валидирует извлеченные концепты на фактическую точность
    - QuizAgent: Генерирует вопросы разных типов на основе концептов
    - ExplainAgent: Создает объяснения ошибок и мнемонические образы

Workflow:
    1. OrchestratorAgent получает заметку от пользователя
    2. ParserAgent анализирует текст и извлекает ключевые концепты
    3. FactCheckAgent проверяет корректность извлеченных данных
    4. QuizAgent генерирует вопросы для самопроверки
    5. ExplainAgent помогает при неправильных ответах

Примечание:
    Только OrchestratorAgent хранит состояние сессии. Остальные агенты являются
    функциональными исполнителями (stateless) - они получают данные, обрабатывают
    и возвращают результат без сохранения контекста между вызовами.
"""

from agents.orchestrator import OrchestratorAgent
from agents.parser import ParserAgent
from agents.factcheck import FactCheckAgent
from agents.quiz import QuizAgent
from agents.explain import ExplainAgent

# Публичный API пакета
__all__ = [
    # Главный координатор (stateful)
    "OrchestratorAgent",

    # Специализированные агенты (stateless)
    "ParserAgent",
    "FactCheckAgent",
    "QuizAgent",
    "ExplainAgent",
]

# Метаданные пакета
__version__ = "1.0.0"
__author__ = "Quiz Generator Team"

# Порядок вызова агентов в pipeline (для документации)
AGENT_PIPELINE = [
    "ParserAgent",  # Шаг 1: Анализ текста
    "FactCheckAgent",  # Шаг 2: Проверка фактов (опционально)
    "QuizAgent",  # Шаг 3: Генерация вопросов
    "ExplainAgent",  # Шаг 4: Объяснение при ошибках
]