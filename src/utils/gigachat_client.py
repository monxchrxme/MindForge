"""
Клиент для работы с GigaChat API
Содержит функции создания клиентов для Parser и Quiz агентов
"""

import os
from dotenv import load_dotenv
from langchain_gigachat.chat_models import GigaChat

# Загрузка переменных окружения
load_dotenv()


def create_gigachat_parser_client():
    """
    Создание GigaChat клиента для Parser Agent
    Параметры: temperature=0.2 (более детерминированный вывод)

    Returns:
        GigaChat: клиент для парсинга
    """
    return GigaChat(
        credentials=os.getenv("GIGACHAT_CREDENTIALS"),
        verify_ssl_certs=False,
        scope=os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS"),
        temperature=0.2,
        timeout=60
    )


def create_gigachat_quiz_client():
    """
    Создание GigaChat клиента для Quiz Agent
    Параметры: temperature=0.6 (более креативный вывод для вопросов)

    Returns:
        GigaChat: клиент для генерации квизов
    """
    return GigaChat(
        credentials=os.getenv("GIGACHAT_CREDENTIALS"),
        verify_ssl_certs=False,
        scope=os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS"),
        temperature=0.6,
        timeout=60
    )

# TODO: Добавить обработку ошибок подключения
# TODO: Добавить retry логику для API вызовов
# TODO: Добавить логирование всех запросов к API
