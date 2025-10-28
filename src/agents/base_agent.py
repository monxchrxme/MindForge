"""
Базовый класс агента
Определяет общий интерфейс для всех агентов
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Абстрактный базовый класс для всех агентов
    Определяет общий интерфейс
    """

    def __init__(self, agent_name: str):
        """
        Args:
            agent_name: имя агента для логирования
        """
        self.agent_name = agent_name
        self.logger = logging.getLogger(f"{__name__}.{agent_name}")
        self.logger.info(f"{agent_name} инициализирован")

    @abstractmethod
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Основная логика обработки данных агентом
        Должен быть реализован в подклассах

        Args:
            state: словарь с состоянием (lecture_text, key_facts и т.д.)

        Returns:
            обновленный state
        """
        pass

# TODO: Добавить методы log_input() и log_output() для удобства
