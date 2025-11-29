# utils/hashing.py

"""
Утилиты для хеширования данных (Pure Functions).

Основное назначение:
    - Генерация уникальных идентификаторов для заметок на основе их содержимого
    - Создание имён файлов для кэша (data/cache/)
    - Проверка целостности данных

Используется алгоритм SHA-256 для стабильного хеширования:
    - Одинаковый текст → одинаковый хеш (детерминированность)
    - Разные тексты → разные хеши (уникальность)
    - Невозможность восстановить текст из хеша (безопасность)

Примечание:
    Все функции являются pure functions - не имеют побочных эффектов.
"""

import hashlib
from typing import Any, Dict, List, Union


def compute_hash(text: str, algorithm: str = "sha256") -> str:
    """
    Вычисление криптографического хеша текста.

    Используется для создания уникальных идентификаторов заметок.
    Одинаковый текст всегда даёт одинаковый хеш, что позволяет
    избежать повторного анализа одной и той же заметки.

    Args:
        text: Текст для хеширования (обычно текст заметки)
        algorithm: Алгоритм хеширования ('sha256', 'md5', 'sha1')

    Returns:
        str: Шестнадцатеричное представление хеша (64 символа для SHA-256)

    Examples:
        >>> compute_hash("Hello World")
        'a591a6d40bf420404a011733cfb7b190d62c65bf0bcda32b57b277d9ad9f146e'
        >>> compute_hash("Hello World") == compute_hash("Hello World")
        True
        >>> compute_hash("Hello") != compute_hash("World")
        True

    Raises:
        ValueError: Если algorithm не поддерживается
    """
    if not text:
        text = ""

    # Нормализация текста: приведение к UTF-8 байтам
    text_bytes = text.encode('utf-8')

    # Выбор алгоритма хеширования
    try:
        if algorithm == "sha256":
            hash_obj = hashlib.sha256(text_bytes)
        elif algorithm == "md5":
            hash_obj = hashlib.md5(text_bytes)
        elif algorithm == "sha1":
            hash_obj = hashlib.sha1(text_bytes)
        else:
            raise ValueError(f"Unsupported hash algorithm: {algorithm}")
    except Exception as e:
        raise ValueError(f"Error creating hash: {str(e)}")

    # Возврат шестнадцатеричного представления
    return hash_obj.hexdigest()


def compute_short_hash(text: str, length: int = 16) -> str:
    """
    Вычисление укороченного хеша для более читаемых идентификаторов.

    Полезно для логирования или отладки, когда нужен компактный ID.

    Args:
        text: Текст для хеширования
        length: Длина результата (количество первых символов хеша)

    Returns:
        str: Укороченный хеш (первые N символов SHA-256)

    Examples:
        >>> compute_short_hash("Hello World", length=8)
        'a591a6d4'
        >>> len(compute_short_hash("Any text", length=16))
        16

    Raises:
        ValueError: Если length > 64 (максимум для SHA-256)
    """
    if length > 64:
        raise ValueError("Length cannot exceed 64 for SHA-256")

    if length <= 0:
        raise ValueError("Length must be positive")

    full_hash = compute_hash(text)
    return full_hash[:length]


def hash_dict(data: Dict[str, Any]) -> str:
    """
    Хеширование словаря (для кэширования объектов).

    Преобразует словарь в стабильное строковое представление,
    затем вычисляет хеш. Ключи сортируются для детерминированности.

    Args:
        data: Словарь для хеширования

    Returns:
        str: SHA-256 хеш словаря

    Examples:
        >>> hash_dict({"b": 2, "a": 1}) == hash_dict({"a": 1, "b": 2})
        True
        >>> hash_dict({"key": "value"})
        '...'  # 64-символьный хеш
    """
    import json

    # Сериализация в JSON с сортировкой ключей
    json_str = json.dumps(data, sort_keys=True, ensure_ascii=False)

    return compute_hash(json_str)


def hash_list(data: List[Any]) -> str:
    """
    Хеширование списка (для кэширования массивов).

    Преобразует список в строковое представление и вычисляет хеш.

    Args:
        data: Список для хеширования

    Returns:
        str: SHA-256 хеш списка

    Examples:
        >>> hash_list([1, 2, 3])
        '...'  # 64-символьный хеш
        >>> hash_list([1, 2]) != hash_list([2, 1])
        True
    """
    import json

    # Сериализация в JSON
    json_str = json.dumps(data, ensure_ascii=False)

    return compute_hash(json_str)


def verify_hash(text: str, expected_hash: str) -> bool:
    """
    Проверка соответствия текста ожидаемому хешу.

    Используется для проверки целостности данных:
    убедиться, что текст не изменился с момента хеширования.

    Args:
        text: Текст для проверки
        expected_hash: Ожидаемый хеш (результат предыдущего compute_hash)

    Returns:
        bool: True если хеш совпадает, False иначе

    Examples:
        >>> text = "Hello World"
        >>> hash_value = compute_hash(text)
        >>> verify_hash(text, hash_value)
        True
        >>> verify_hash("Different text", hash_value)
        False
    """
    actual_hash = compute_hash(text)
    return actual_hash == expected_hash


def generate_cache_filename(text: str, extension: str = "json") -> str:
    """
    Генерация имени файла кэша на основе хеша текста.

    Используется CacheManager для создания уникальных имён файлов.

    Args:
        text: Текст заметки (или любые данные для идентификации)
        extension: Расширение файла (по умолчанию 'json')

    Returns:
        str: Имя файла вида 'a1b2c3d4ef...json'

    Examples:
        >>> generate_cache_filename("My note content")
        '7d9a8f2b1c3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0.json'
        >>> generate_cache_filename("Data", extension="txt")
        '...txt'
    """
    hash_value = compute_hash(text)

    # Удаление точки в начале расширения, если есть
    extension = extension.lstrip('.')

    return f"{hash_value}.{extension}"


def compare_hashes(hash1: str, hash2: str) -> bool:
    """
    Безопасное сравнение двух хешей (защита от timing attacks).

    Использует constant-time сравнение для предотвращения утечки
    информации через время выполнения операции.

    Args:
        hash1: Первый хеш
        hash2: Второй хеш

    Returns:
        bool: True если хеши идентичны, False иначе

    Examples:
        >>> h1 = compute_hash("text")
        >>> h2 = compute_hash("text")
        >>> compare_hashes(h1, h2)
        True
    """
    import hmac

    # hmac.compare_digest обеспечивает constant-time сравнение
    return hmac.compare_digest(hash1, hash2)


def hash_with_salt(text: str, salt: str) -> str:
    """
    Хеширование текста с добавлением соли (salt).

    Используется когда нужно сделать хеш уникальным для конкретного
    пользователя или сессии, предотвращая rainbow table атаки.

    Args:
        text: Текст для хеширования
        salt: Соль (случайная строка для уникальности)

    Returns:
        str: SHA-256 хеш (text + salt)

    Examples:
        >>> hash_with_salt("password", "user123")
        '...'
        >>> hash_with_salt("password", "user123") != compute_hash("password")
        True
    """
    combined = f"{text}{salt}"
    return compute_hash(combined)


def batch_hash(texts: List[str]) -> List[str]:
    """
    Хеширование списка текстов (пакетная обработка).

    Эффективно обрабатывает множество текстов за один вызов.

    Args:
        texts: Список текстов для хеширования

    Returns:
        List[str]: Список хешей в том же порядке

    Examples:
        >>> batch_hash(["text1", "text2", "text3"])
        ['hash1...', 'hash2...', 'hash3...']
        >>> len(batch_hash(["a", "b", "c"]))
        3
    """
    return [compute_hash(text) for text in texts]


def hash_to_int(text: str, max_value: int = 2 ** 32) -> int:
    """
    Преобразование текста в детерминированное целое число.

    Полезно для разделения данных по партициям/шардам.

    Args:
        text: Текст для преобразования
        max_value: Максимальное значение результата (для ограничения диапазона)

    Returns:
        int: Целое число от 0 до max_value-1

    Examples:
        >>> hash_to_int("text", max_value=100)
        42  # Всегда одно и то же число для "text"
        >>> 0 <= hash_to_int("any", max_value=1000) < 1000
        True
    """
    hash_value = compute_hash(text)
    # Преобразование первых 8 байт хеша в int
    hash_int = int(hash_value[:16], 16)
    return hash_int % max_value


# Константы для удобства
DEFAULT_HASH_ALGORITHM = "sha256"
CACHE_FILENAME_EXTENSION = "json"
SHORT_HASH_LENGTH = 16
