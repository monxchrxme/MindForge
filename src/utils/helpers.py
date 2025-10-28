"""
Вспомогательные функции для работы с файлами и данными
"""

from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def load_note(file_path: str) -> str:
    """
    Загрузка заметки из файла (аналогично твоему open("sample_lecture.txt"))

    Args:
        file_path: путь к файлу заметки

    Returns:
        str: содержимое файла

    Raises:
        FileNotFoundError: если файл не найден
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {file_path}")

    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()

    logger.info(f"Загружена заметка: {file_path} ({len(content)} символов)")
    return content

# TODO: Добавить функцию calculate_hash() для кэширования
# TODO: Добавить функцию format_output() для красивого вывода
# TODO: Добавить функцию chunk_text() для RAG (на будущее)
