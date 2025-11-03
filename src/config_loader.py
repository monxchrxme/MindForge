"""
Загрузка конфигурации из YAML и .env
"""

import yaml
import os
from dotenv import load_dotenv



def load_config(config_path: str = "config/config.yaml") -> dict:
    """
    Загрузка конфигурации из YAML файла и .env

    Args:
        config_path: путь к config.yaml

    Returns:
        dict: объединенная конфигурация
    """
    # Загрузка .env
    load_dotenv()

    # Загрузка YAML
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Добавление переменных окружения в config
    config['gigachat']['credentials'] = os.getenv('GIGACHAT_CREDENTIALS')
    config['gigachat']['scope'] = os.getenv('GIGACHAT_SCOPE', 'GIGACHAT_API_PERS')

    return config

# TODO: Добавить валидацию конфигурации
# TODO: Добавить значения по умолчанию для отсутствующих ключей
