import argparse
import shutil
from pathlib import Path
import sys


def clear_logs(require_confirm: bool = False):
    # Определяем путь к папке логов относительно скрипта
    # scripts/ -> parent -> data/logs
    log_dir = Path(__file__).parent.parent / "data" / "logs"

    if not log_dir.exists():
        print(f"❌ Папка логов не найдена: {log_dir}")
        return

    # Ищем все файлы логов (app.log, app.log.1, app.log.2 ...)
    log_files = list(log_dir.glob("app.log*"))

    if not log_files:
        print("✅ Логов нет (папка пуста).")
        return

    count = len(log_files)
    size_mb = sum(f.stat().st_size for f in log_files) / (1024 * 1024)

    print(f"\n🔍 Найдено файлов логов: {count}")
    print(f"💾 Общий размер: {size_mb:.2f} MB")

    if require_confirm:
        answer = input("⚠️ Удалить эти файлы? (y/n): ").strip().lower()
        if answer not in ['y', 'yes', 'д', 'да']:
            print("❌ Отмена.")
            return

    deleted = 0
    for log_file in log_files:
        try:
            log_file.unlink()
            deleted += 1
            # print(f"Удален: {log_file.name}") # Можно раскомментировать для детальности
        except Exception as e:
            print(f"❌ Ошибка при удалении {log_file.name}: {e}")

    print(f"✅ Готово! Удалено файлов: {deleted}")


def main():
    parser = argparse.ArgumentParser(description="Очистка логов приложения")
    parser.add_argument(
        "-y", "--yes",
        action="store_true",
        help="Удалить без подтверждения"
    )
    args = parser.parse_args()

    # Если флага -y нет, требуем подтверждение (require_confirm=True)
    # Если флаг -y есть, require_confirm=False
    clear_logs(require_confirm=not args.yes)


if __name__ == "__main__":
    main()
