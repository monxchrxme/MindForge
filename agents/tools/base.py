from abc import ABC, abstractmethod
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class BaseTool(ABC):
    """Базовый класс для всех инструментов агента"""

    @property
    @abstractmethod
    def name(self) -> str:
        """Имя инструмента для вызова агентом"""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Описание для LLM: что делает, когда использовать"""
        pass

    @abstractmethod
    def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Выполнение инструмента.
        Returns: Dict с результатом выполнения
        """
        pass

    def validate_args(self, **kwargs) -> bool:
        """Валидация аргументов перед выполнением"""
        return True