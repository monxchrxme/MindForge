import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class CacheManager:
    """
    Абстракция над файловой системой для сохранения «дорогих» данных.
    Управляет JSON-кэшем на диске для экономии токенов API.
    Работает с папкой data/cache/ и именами файлов на основе SHA-256 хешей.
    """

    def __init__(self, cache_dir: str = "data/cache"):
        """
        Инициализация менеджера кэша.

        Args:
            cache_dir: Путь к директории для хранения кэш-файлов
        """
        self.cache_dir = Path(cache_dir)
        self._ensure_cache_directory()

        logger.info(f"CacheManager initialized: cache_dir={self.cache_dir.absolute()}")

    def exists(self, filename: str) -> bool:
        """
        Проверка наличия файла в кэше.

        Args:
            filename: Имя файла (хеш или имя с расширением .json)

        Returns:
            bool: True если файл существует, False иначе
        """
        filepath = self._get_filepath(filename)
        exists = filepath.exists() and filepath.is_file()

        logger.debug(f"Cache check for '{filename}': {'HIT' if exists else 'MISS'}")

        return exists

    def get(self, filename: str) -> Optional[Union[Dict[str, Any], List[Any]]]:
        """
        Попытка прочитать данные из кэш-файла.
        Алиас для load() для совместимости с разными частями кода.

        Args:
            filename: Имя файла (хеш или имя с расширением .json)

        Returns:
            Dict/List если файл найден и валиден, None если файла нет или ошибка
        """
        return self.load(filename)

    def load(self, filename: str) -> Optional[Union[Dict[str, Any], List[Any]]]:
        """
        Чтение данных из JSON-файла в кэше.

        Args:
            filename: Имя файла (хеш или имя с расширением .json)

        Returns:
            Dict/List с данными из файла или None при ошибке
        """
        filepath = self._get_filepath(filename)

        if not filepath.exists():
            logger.debug(f"Cache file not found: {filename}")
            return None

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            logger.info(f"Cache loaded successfully: {filename}")
            return data

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in cache file {filename}: {str(e)}")
            return None

        except Exception as e:
            logger.error(f"Error reading cache file {filename}: {str(e)}", exc_info=True)
            return None

    def save(self, filename: str, data: Union[Dict[str, Any], List[Any]]) -> bool:
        """
        Сохранение Python-объекта в JSON-файл в папке data/cache/.

        Args:
            filename: Имя файла (хеш или имя с расширением .json)
            data: Python-словарь или список для сериализации

        Returns:
            bool: True если успешно сохранено, False при ошибке
        """
        if not isinstance(data, (dict, list)):
            logger.error(f"Invalid data type for caching: {type(data)}. Expected dict or list.")
            return False

        filepath = self._get_filepath(filename)

        try:
            # Создаем директорию если не существует
            self._ensure_cache_directory()

            # Сохраняем с красивым форматированием для отладки
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            logger.info(f"Cache saved successfully: {filename} ({self._get_file_size(filepath)})")
            return True

        except Exception as e:
            logger.error(f"Error saving cache file {filename}: {str(e)}", exc_info=True)
            return False

    def delete(self, filename: str) -> bool:
        """
        Удаление конкретного файла из кэша.

        Args:
            filename: Имя файла для удаления

        Returns:
            bool: True если успешно удалено, False если файла не было или ошибка
        """
        filepath = self._get_filepath(filename)

        if not filepath.exists():
            logger.debug(f"Cannot delete non-existent cache file: {filename}")
            return False

        try:
            filepath.unlink()
            logger.info(f"Cache file deleted: {filename}")
            return True

        except Exception as e:
            logger.error(f"Error deleting cache file {filename}: {str(e)}")
            return False

    def clear(self, max_age_days: Optional[int] = None) -> int:
        """
        Очистка кэш-директории.
        Удаляет все файлы или только файлы старше указанного возраста.

        Args:
            max_age_days: Максимальный возраст файлов в днях.
                         Если None - удаляются все файлы.
                         Если указано - удаляются только файлы старше этого срока.

        Returns:
            int: Количество удаленных файлов
        """
        if not self.cache_dir.exists():
            logger.debug("Cache directory doesn't exist, nothing to clear")
            return 0

        deleted_count = 0
        cutoff_time = None

        if max_age_days is not None:
            cutoff_time = datetime.now() - timedelta(days=max_age_days)
            logger.info(f"Clearing cache files older than {max_age_days} days")
        else:
            logger.info("Clearing all cache files")

        try:
            for filepath in self.cache_dir.glob("*.json"):
                should_delete = False

                if cutoff_time is None:
                    # Удаляем все файлы
                    should_delete = True
                else:
                    # Проверяем возраст файла
                    file_mtime = datetime.fromtimestamp(filepath.stat().st_mtime)
                    should_delete = file_mtime < cutoff_time

                if should_delete:
                    try:
                        filepath.unlink()
                        deleted_count += 1
                        logger.debug(f"Deleted cache file: {filepath.name}")
                    except Exception as e:
                        logger.error(f"Failed to delete {filepath.name}: {str(e)}")

            logger.info(f"Cache cleared: {deleted_count} files deleted")
            return deleted_count

        except Exception as e:
            logger.error(f"Error during cache clearing: {str(e)}", exc_info=True)
            return deleted_count

    def get_stats(self) -> Dict[str, Any]:
        """
        Получение статистики по кэшу.

        Returns:
            Dict с ключами:
                - total_files: количество файлов в кэше
                - total_size_bytes: общий размер в байтах
                - total_size_mb: общий размер в мегабайтах
                - oldest_file: дата создания самого старого файла
                - newest_file: дата создания самого нового файла
        """
        if not self.cache_dir.exists():
            return {
                "total_files": 0,
                "total_size_bytes": 0,
                "total_size_mb": 0.0,
                "oldest_file": None,
                "newest_file": None
            }

        total_files = 0
        total_size = 0
        oldest_time = None
        newest_time = None

        for filepath in self.cache_dir.glob("*.json"):
            total_files += 1
            total_size += filepath.stat().st_size

            file_mtime = datetime.fromtimestamp(filepath.stat().st_mtime)

            if oldest_time is None or file_mtime < oldest_time:
                oldest_time = file_mtime

            if newest_time is None or file_mtime > newest_time:
                newest_time = file_mtime

        return {
            "total_files": total_files,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "oldest_file": oldest_time.isoformat() if oldest_time else None,
            "newest_file": newest_time.isoformat() if newest_time else None
        }

    def _ensure_cache_directory(self) -> None:
        """
        Создание директории кэша, если она не существует.
        """
        if not self.cache_dir.exists():
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created cache directory: {self.cache_dir.absolute()}")

            # Создаем .gitkeep чтобы Git отслеживал пустую папку
            gitkeep_path = self.cache_dir / ".gitkeep"
            if not gitkeep_path.exists():
                gitkeep_path.touch()

    def _get_filepath(self, filename: str) -> Path:
        """
        Получение полного пути к файлу в кэш-директории.
        Автоматически добавляет расширение .json если его нет.

        Args:
            filename: Имя файла (хеш или имя файла)

        Returns:
            Path: Полный путь к файлу
        """
        # Добавляем .json если нет расширения
        if not filename.endswith('.json'):
            filename = f"{filename}.json"

        return self.cache_dir / filename

    def _get_file_size(self, filepath: Path) -> str:
        """
        Получение размера файла в читаемом формате.

        Args:
            filepath: Путь к файлу

        Returns:
            str: Размер файла (например, "1.5 KB", "234 B")
        """
        try:
            size_bytes = filepath.stat().st_size

            if size_bytes < 1024:
                return f"{size_bytes} B"
            elif size_bytes < 1024 * 1024:
                return f"{size_bytes / 1024:.1f} KB"
            else:
                return f"{size_bytes / (1024 * 1024):.1f} MB"
        except Exception:
            return "unknown"


# Вспомогательная функция для создания менеджера кэша
def create_cache_manager(cache_dir: str = "data/cache") -> CacheManager:
    """
    Фабричная функция для создания CacheManager.

    Args:
        cache_dir: Путь к директории кэша

    Returns:
        Инициализированный CacheManager
    """
    return CacheManager(cache_dir=cache_dir)
