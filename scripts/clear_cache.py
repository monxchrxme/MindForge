#!/usr/bin/env python3
# scripts/clear_cache.py

"""
Скрипт очистки кэша системы генерации квизов.

Назначение:
    - Удаление всех или устаревших JSON-файлов из data/cache/
    - Освобождение дискового пространства
    - Принудительная перегенерация концептов после изменения промптов

Использование:
    python scripts/clear_cache.py              # Очистить всё
    python scripts/clear_cache.py --days 30    # Удалить файлы старше 30 дней
    python scripts/clear_cache.py --stats      # Показать статистику без удаления
    python scripts/clear_cache.py --confirm    # Запросить подтверждение

Примеры:
    # Полная очистка с подтверждением
    python scripts/clear_cache.py --confirm

    # Удалить файлы старше 7 дней
    python scripts/clear_cache.py --days 7

    # Просмотр статистики кэша
    python scripts/clear_cache.py --stats
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Добавление корневой директории проекта в PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))

from services import CacheManager


# ============================================================================
# НАСТРОЙКА ЛОГИРОВАНИЯ
# ============================================================================

def setup_logging(verbose: bool = False) -> None:
    """
    Настройка логирования для скрипта.

    Args:
        verbose: Если True, выводить DEBUG сообщения
    """
    level = logging.DEBUG if verbose else logging.INFO

    logging.basicConfig(
        level=level,
        format='%(levelname)s: %(message)s'
    )


# ============================================================================
# ФУНКЦИИ ОЧИСТКИ
# ============================================================================

def load_config() -> dict:
    """
    Загрузка конфигурации для получения пути к кэшу.

    Returns:
        dict: Конфигурация из config.json
    """
    config_path = Path(__file__).parent.parent / "config.json"

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logging.warning(f"Could not load config.json: {e}")
        return {}


def display_cache_stats(cache_manager: CacheManager) -> None:
    """
    Отображение статистики кэша.

    Args:
        cache_manager: Менеджер кэша
    """
    stats = cache_manager.get_stats()

    print("\n" + "=" * 70)
    print("📊 СТАТИСТИКА КЭША")
    print("=" * 70)
    print(f"📁 Директория: {cache_manager.cache_dir}")
    print(f"📄 Всего файлов: {stats['total_files']}")
    print(f"💾 Размер: {stats['total_size_mb']} MB ({stats['total_size_bytes']} bytes)")

    if stats['oldest_file']:
        print(f"📅 Самый старый файл: {stats['oldest_file']}")
    if stats['newest_file']:
        print(f"📅 Самый новый файл: {stats['newest_file']}")

    print("=" * 70 + "\n")


def confirm_deletion(file_count: int, total_size_mb: float) -> bool:
    """
    Запрос подтверждения удаления файлов.

    Args:
        file_count: Количество файлов для удаления
        total_size_mb: Общий размер в мегабайтах

    Returns:
        bool: True если пользователь подтвердил, False иначе
    """
    print(f"\n⚠️  ВНИМАНИЕ: Будет удалено {file_count} файлов ({total_size_mb} MB)")
    print("Это действие необратимо!")

    response = input("\nПродолжить? (yes/no): ").strip().lower()

    return response in ['yes', 'y', 'да', 'д']


def clear_cache(
        cache_manager: CacheManager,
        max_age_days: int = None,
        require_confirm: bool = False,
        verbose: bool = False
) -> int:
    """
    Очистка кэша с опциональным подтверждением.

    Args:
        cache_manager: Менеджер кэша
        max_age_days: Максимальный возраст файлов (None = все)
        require_confirm: Требовать подтверждение от пользователя
        verbose: Выводить подробную информацию

    Returns:
        int: Количество удаленных файлов
    """
    logger = logging.getLogger(__name__)

    # Получение статистики перед удалением
    stats = cache_manager.get_stats()

    if stats['total_files'] == 0:
        print("\n✅ Кэш пуст. Нечего удалять.")
        return 0

    # Запрос подтверждения
    if require_confirm:
        if not confirm_deletion(stats['total_files'], stats['total_size_mb']):
            print("\n❌ Операция отменена пользователем.")
            return 0

    # Выполнение очистки
    print("\n⏳ Очистка кэша...")

    deleted_count = cache_manager.clear(max_age_days=max_age_days)

    if deleted_count > 0:
        print(f"\n✅ Успешно удалено файлов: {deleted_count}")

        if verbose:
            freed_space_mb = stats['total_size_mb']
            print(f"💾 Освобождено места: {freed_space_mb} MB")

            # Статистика после очистки
            new_stats = cache_manager.get_stats()
            print(f"📄 Осталось файлов: {new_stats['total_files']}")
    else:
        print("\n⚠️  Файлы не были удалены.")

    return deleted_count


# ============================================================================
# ПАРСИНГ АРГУМЕНТОВ КОМАНДНОЙ СТРОКИ
# ============================================================================

def parse_arguments() -> argparse.Namespace:
    """
    Парсинг аргументов командной строки.

    Returns:
        argparse.Namespace: Распарсенные аргументы
    """
    parser = argparse.ArgumentParser(
        description='Очистка кэша системы генерации квизов',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  %(prog)s                    # Полная очистка кэша
  %(prog)s --days 30          # Удалить файлы старше 30 дней
  %(prog)s --stats            # Показать статистику без удаления
  %(prog)s --confirm          # Запросить подтверждение перед удалением
  %(prog)s -v                 # Подробный вывод
        """
    )

    parser.add_argument(
        '--days',
        type=int,
        metavar='N',
        help='Удалить только файлы старше N дней (по умолчанию: удалить все)'
    )

    parser.add_argument(
        '--stats',
        action='store_true',
        help='Показать статистику кэша без удаления файлов'
    )

    parser.add_argument(
        '--confirm',
        action='store_true',
        help='Запросить подтверждение перед удалением'
    )

    parser.add_argument(
        '--cache-dir',
        type=str,
        metavar='PATH',
        help='Путь к директории кэша (по умолчанию: из config.json)'
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Подробный вывод'
    )

    return parser.parse_args()


# ============================================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ============================================================================

def main() -> None:
    """
    Главная функция скрипта очистки кэша.
    """
    # Парсинг аргументов
    args = parse_arguments()

    # Настройка логирования
    setup_logging(verbose=args.verbose)

    logger = logging.getLogger(__name__)

    # Определение директории кэша
    if args.cache_dir:
        cache_dir = args.cache_dir
    else:
        config = load_config()
        cache_dir = config.get('cache_settings', {}).get('cache_dir', 'data/cache')

    logger.debug(f"Cache directory: {cache_dir}")

    # Создание CacheManager
    try:
        cache_manager = CacheManager(cache_dir=cache_dir)
    except Exception as e:
        print(f"\n❌ Ошибка инициализации CacheManager: {e}")
        sys.exit(1)

    # Режим работы: только статистика или очистка
    if args.stats:
        # Только отображение статистики
        display_cache_stats(cache_manager)
    else:
        # Очистка кэша
        print("\n" + "=" * 70)
        print("🗑️  ОЧИСТКА КЭША")
        print("=" * 70)

        if args.days:
            print(f"Режим: Удаление файлов старше {args.days} дней")
        else:
            print("Режим: Полная очистка кэша")

        # Отображение текущей статистики
        if args.verbose:
            display_cache_stats(cache_manager)

        # Выполнение очистки
        try:
            deleted_count = clear_cache(
                cache_manager=cache_manager,
                max_age_days=args.days,
                require_confirm=args.confirm,
                verbose=args.verbose
            )

            if deleted_count > 0:
                logger.info(f"Cache cleared successfully: {deleted_count} files deleted")

            print("=" * 70)

        except Exception as e:
            logger.error(f"Error during cache clearing: {e}", exc_info=args.verbose)
            print(f"\n❌ Ошибка при очистке кэша: {e}")
            sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Операция прервана пользователем.")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {e}")
        sys.exit(1)
