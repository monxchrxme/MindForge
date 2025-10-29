"""
Базовый класс агента
Определяет общий интерфейс для всех агентов
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Абстрактный базовый класс для всех агентов
    Определяет общий интерфейс
    """

    def __init__(self, agent_name: str, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            agent_name: имя агента для логирования
            config: опциональная конфигурация агента
        """
        self.agent_name = agent_name
        self.config = config or {}
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

    def log_input(self, data: Any):
        """Логирование входных данных"""
        self.logger.debug(f"Входные данные: {str(data)[:200]}...")

    def log_output(self, data: Any):
        """Логирование выходных данных"""
        self.logger.debug(f"Выходные данные: {str(data)[:200]}...")
