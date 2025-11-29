"""
Слой утилит (Utility Layer).

Этот пакет содержит вспомогательные Pure Functions, которые используются
по всему проекту. Утилиты не содержат бизнес-логики и не зависят от
внешних сервисов - это чистые математические преобразования данных.

Модули:
    - hashing: Криптографическое хеширование для кэша и идентификации
    - text_cleaner: Постобработка ответов от LLM (очистка JSON, markdown)

Принципы:
    - Все функции являются Pure Functions (без побочных эффектов)
    - Детерминированность: одинаковый вход → одинаковый выход
    - Независимость от состояния системы
    - Легко тестируемые и переиспользуемые
"""

# Импорты из модуля hashing
from utils.hashing import (
    compute_hash,
    compute_short_hash,
    hash_dict,
    hash_list,
    verify_hash,
    generate_cache_filename,
    compare_hashes,
    hash_with_salt,
    batch_hash,
    hash_to_int,
)

# Импорты из модуля text_cleaner
from utils.text_cleaner import (
    extract_json_from_markdown,
    remove_comments,
    extract_json_object,
    clean_json_text,
    parse_llm_json,
    fix_common_json_errors,
    normalize_whitespace,
    truncate_text,
    sanitize_for_json,
    validate_json_string,
    extract_code_blocks,
    quick_parse_json,
)

# Публичный API пакета
__all__ = [
    # Функции хеширования
    "compute_hash",
    "compute_short_hash",
    "hash_dict",
    "hash_list",
    "verify_hash",
    "generate_cache_filename",
    "compare_hashes",
    "hash_with_salt",
    "batch_hash",
    "hash_to_int",

    # Функции очистки текста
    "extract_json_from_markdown",
    "remove_comments",
    "extract_json_object",
    "clean_json_text",
    "parse_llm_json",
    "fix_common_json_errors",
    "normalize_whitespace",
    "truncate_text",
    "sanitize_for_json",
    "validate_json_string",
    "extract_code_blocks",
    "quick_parse_json",
]

# Метаданные пакета
__version__ = "1.0.0"
__author__ = "Quiz Generator Team"

# Группировка функций по категориям (для документации)
HASHING_FUNCTIONS = [
    "compute_hash",
    "compute_short_hash",
    "generate_cache_filename",
    "verify_hash",
]

TEXT_CLEANING_FUNCTIONS = [
    "parse_llm_json",
    "clean_json_text",
    "extract_json_from_markdown",
    "quick_parse_json",
]

DATA_STRUCTURE_HASHING = [
    "hash_dict",
    "hash_list",
    "batch_hash",
]

JSON_UTILITIES = [
    "validate_json_string",
    "fix_common_json_errors",
    "sanitize_for_json",
]