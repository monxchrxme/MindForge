# utils/text_cleaner.py

"""
Набор чистых функций (Pure Functions) для постобработки ответов от LLM.

Основные задачи:
    - Извлечение JSON из текста с markdown разметкой
    - Удаление комментариев из JSON
    - Валидация и попытка "починки" невалидного JSON
    - Очистка текста от лишних пробелов и управляющих символов
    - Нормализация строк

Примечание:
    Все функции являются pure functions - не имеют побочных эффектов,
    всегда возвращают одинаковый результат для одинаковых входных данных.
"""

import json
import re
from typing import Any, Dict, List, Optional, Union


def extract_json_from_markdown(text: str) -> str:
    """
    Извлекает JSON из текста, обернутого в markdown блоки.

    LLM часто возвращает JSON в формате:
    ```
    { "key": "value" }
    ```
    или
    ```
    { "key": "value" }
    ```

    Args:
        text: Текст, потенциально содержащий JSON в markdown блоке

    Returns:
        str: Извлеченный JSON или оригинальный текст, если блок не найден

    Examples:
        >>> extract_json_from_markdown('``````')
        '{"a": 1}'
        >>> extract_json_from_markdown('Some text {"a": 1} more text')
        'Some text {"a": 1} more text'
    """
    text = text.strip()

    # Паттерн для markdown блока с опциональным указанием языка
    pattern = r'``````'
    match = re.search(pattern, text)

    if match:
        return match.group(1).strip()

    return text


def remove_comments(text: str) -> str:
    """
    Удаляет комментарии из текста (JSON с комментариями - невалиден).

    Удаляет:
        - Однострочные комментарии: // comment
        - Многострочные комментарии: /* comment */

    Args:
        text: Текст, потенциально содержащий комментарии

    Returns:
        str: Текст без комментариев

    Examples:
        >>> remove_comments('{"a": 1} // comment')
        '{"a": 1} '
        >>> remove_comments('{"a": /* inline */ 1}')
        '{"a":  1}'
    """
    # Удаление многострочных комментариев /* ... */
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)

    # Удаление однострочных комментариев // ...
    text = re.sub(r'//.*?$', '', text, flags=re.MULTILINE)

    return text


def extract_json_object(text: str) -> str:
    """
    Извлекает JSON объект или массив из текста.

    Ищет первое вхождение {} или [] и извлекает его с учетом вложенности.

    Args:
        text: Текст, содержащий JSON среди другого текста

    Returns:
        str: Извлеченный JSON или пустая строка, если не найден

    Examples:
        >>> extract_json_object('Some text {"a": 1} more text')
        '{"a": 1}'
        >>> extract_json_object('Text [1, 2, 3] end')
        '[1, 2, 3]'
    """
    # Попытка найти JSON объект {...}
    obj_match = re.search(r'\{[\s\S]*\}', text)
    if obj_match:
        return obj_match.group(0)

    # Попытка найти JSON массив [...]
    arr_match = re.search(r'\[[\s\S]*\]', text)
    if arr_match:
        return arr_match.group(0)

    return ""


def clean_json_text(text: str) -> str:
    """
    Полная очистка текста перед парсингом JSON.

    Выполняет последовательно:
        1. Извлечение из markdown блоков
        2. Удаление комментариев
        3. Нормализацию пробелов

    Args:
        text: Сырой текст от LLM

    Returns:
        str: Очищенный текст, готовый к парсингу

    Examples:
        >>> clean_json_text('``````')
        '{"a": 1}'
    """
    # Шаг 1: Извлечь из markdown
    text = extract_json_from_markdown(text)

    # Шаг 2: Удалить комментарии
    text = remove_comments(text)

    # Шаг 3: Нормализация пробелов
    text = text.strip()

    return text


def parse_llm_json(
        text: str,
        strict: bool = False
) -> Union[Dict[str, Any], List[Any], None]:
    """
    Парсинг JSON из ответа LLM с автоматической очисткой.

    Пытается распарсить JSON, применяя различные стратегии очистки.

    Args:
        text: Текст от LLM, потенциально содержащий JSON
        strict: Если True, возвращает None при ошибке вместо попыток починки

    Returns:
        Распарсенный dict/list или None при ошибке

    Examples:
        >>> parse_llm_json('``````')
        {'a': 1}
        >>> parse_llm_json('Invalid JSON')
        None
    """
    if not text or not text.strip():
        return None

    # Стратегия 1: Попытка прямого парсинга
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    if strict:
        return None

    # Стратегия 2: Очистка и повторная попытка
    cleaned = clean_json_text(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Стратегия 3: Извлечение JSON объекта из текста
    extracted = extract_json_object(cleaned)
    if extracted:
        try:
            return json.loads(extracted)
        except json.JSONDecodeError:
            pass

    # Стратегия 4: Попытка починить распространенные ошибки
    fixed = fix_common_json_errors(extracted or cleaned)
    if fixed != (extracted or cleaned):
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass

    return None


def fix_common_json_errors(text: str) -> str:
    """
    Попытка автоматического исправления распространенных ошибок в JSON.

    Исправляет:
        - Лишние запятые перед закрывающими скобками
        - Одинарные кавычки вместо двойных
        - Отсутствие кавычек у ключей
        - Trailing commas

    Args:
        text: Текст с потенциально невалидным JSON

    Returns:
        str: Текст с попыткой исправления

    Warning:
        Не гарантирует валидный JSON, только пытается исправить частые ошибки
    """
    if not text:
        return text

    # Удаление trailing commas перед ] и }
    text = re.sub(r',\s*]', ']', text)
    text = re.sub(r',\s*}', '}', text)

    # Замена одинарных кавычек на двойные (осторожно!)
    # Только если они явно выглядят как разделители строк
    text = re.sub(r"'([^']*)':", r'"\1":', text)
    text = re.sub(r":\s*'([^']*)'", r': "\1"', text)

    return text


def normalize_whitespace(text: str) -> str:
    """
    Нормализация пробелов в тексте.

    Заменяет множественные пробелы на один, удаляет пробелы в начале/конце.

    Args:
        text: Исходный текст

    Returns:
        str: Текст с нормализованными пробелами

    Examples:
        >>> normalize_whitespace('  text   with    spaces  ')
        'text with spaces'
    """
    # Замена всех whitespace символов (пробелы, табы, переносы) на один пробел
    text = re.sub(r'\s+', ' ', text)

    # Удаление пробелов в начале и конце
    return text.strip()


def truncate_text(text: str, max_length: int = 1000, suffix: str = "...") -> str:
    """
    Обрезает текст до указанной длины с добавлением суффикса.

    Полезно для логирования длинных ответов от LLM.

    Args:
        text: Исходный текст
        max_length: Максимальная длина (включая суффикс)
        suffix: Суффикс для добавления к обрезанному тексту

    Returns:
        str: Обрезанный текст

    Examples:
        >>> truncate_text('a' * 100, max_length=10)
        'aaaaaaa...'
    """
    if len(text) <= max_length:
        return text

    return text[:max_length - len(suffix)] + suffix


def sanitize_for_json(text: str) -> str:
    """
    Очистка текста для безопасного включения в JSON.

    Экранирует специальные символы, которые могут сломать JSON.

    Args:
        text: Исходный текст

    Returns:
        str: Текст с экранированными специальными символами

    Examples:
        >>> sanitize_for_json('Text with "quotes" and \\backslash')
        'Text with \\\\"quotes\\\\" and \\\\\\\\backslash'
    """
    # Экранирование обратного слеша
    text = text.replace('\\', '\\\\')

    # Экранирование двойных кавычек
    text = text.replace('"', '\\"')

    # Экранирование управляющих символов
    text = text.replace('\n', '\\n')
    text = text.replace('\r', '\\r')
    text = text.replace('\t', '\\t')

    return text


def validate_json_string(text: str) -> bool:
    """
    Проверка, является ли текст валидным JSON.

    Args:
        text: Текст для проверки

    Returns:
        bool: True если валидный JSON, False иначе

    Examples:
        >>> validate_json_string('{"a": 1}')
        True
        >>> validate_json_string('invalid')
        False
    """
    try:
        json.loads(text)
        return True
    except (json.JSONDecodeError, TypeError):
        return False


def extract_code_blocks(text: str, language: Optional[str] = None) -> List[str]:
    """
    Извлекает все блоки кода из markdown текста.

    Args:
        text: Текст с markdown разметкой
        language: Фильтр по языку (например, 'python', 'json')

    Returns:
        List[str]: Список извлеченных блоков кода

    Examples:
        >>> text = '``````\\n``````'
        >>> extract_code_blocks(text, 'python')
        ['print("hi")']
    """
    if language:
        pattern = rf'``````'
    else:
        pattern = r'``````'

    matches = re.findall(pattern, text)
    return [match.strip() for match in matches]


# Вспомогательная функция для быстрого доступа
def quick_parse_json(text: str) -> Optional[Union[Dict, List]]:
    """
    Быстрый парсинг JSON с автоматической очисткой (алиас для parse_llm_json).

    Args:
        text: Текст от LLM

    Returns:
        Распарсенный JSON или None
    """
    return parse_llm_json(text, strict=False)
