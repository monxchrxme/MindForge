"""
Слой инфраструктуры (Infrastructure Layer).

Этот пакет содержит технические компоненты для работы с внешними сервисами
и файловой системой. Модули этого слоя не содержат бизнес-логики и не знают
о квизах или агентах - они выполняют только технические задачи.

Компоненты:
    - GigaChatClient: Низкоуровневая работа с GigaChat API через LangChain
    - CacheManager: Управление JSON-кэшем на диске для экономии токенов
"""

from services.gigachat_client import GigaChatClient, create_client_from_config
from services.cache_manager import CacheManager, create_cache_manager

# Публичный API пакета
__all__ = [
    # Основные классы
    "GigaChatClient",
    "CacheManager",

    # Фабричные функции
    "create_client_from_config",
    "create_cache_manager",
]

# Метаданные пакета
__version__ = "1.0.0"
__author__ = "Quiz Generator Team"
