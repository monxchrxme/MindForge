# main.py

"""
CLI Точка входа в приложение «Генератор Умных Квизов».
Запуск: python main.py <path_to_note> [options]

Аргументы:
  path_to_note          Путь к текстовому файлу с заметкой
  --difficulty, -d      Сложность (easy, medium, hard)
  --questions, -q       Количество вопросов (по умолчанию из config.json)
  --force, -f           Принудительный парсинг (игнорировать кэш)
  --debug               Включить подробное логирование
  --help, -h            Показать справку по флагам
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

from agents import OrchestratorAgent
from services import CacheManager



# ============================================================================
# НАСТРОЙКА ЛОГИРОВАНИЯ
# ============================================================================

def setup_logging(debug_mode: bool = False):
    """
    Настройка системы логирования.

    Args:
        debug_mode: Если True - уровень DEBUG, иначе INFO
    """
    level = logging.DEBUG if debug_mode else logging.INFO
    format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    # Создаем директорию для логов
    Path("data/logs").mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=level,
        format=format_str,
        handlers=[
            logging.FileHandler("data/logs/app.log", encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    # Убираем лишний шум от библиотек
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


# ============================================================================
# ЗАГРУЗКА КОНФИГУРАЦИИ
# ============================================================================

def load_config(config_path: str = "config.json") -> dict:
    """Загрузка конфигурации из JSON файла."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file '{config_path}' not found")
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_credentials() -> dict:
    """Загрузка секретных ключей из .env файла."""
    load_dotenv()
    client_id = os.getenv('GIGACHAT_CLIENT_ID')
    client_secret = os.getenv('GIGACHAT_CREDENTIALS')

    if not client_id or not client_secret:
        raise ValueError("Missing GIGACHAT_CLIENT_ID or GIGACHAT_CREDENTIALS in .env")

    return {'client_id': client_id, 'client_secret': client_secret}


# ============================================================================
# ИНТЕРАКТИВНАЯ СЕССИЯ КВИЗА
# ============================================================================

def run_cli_quiz_session(orchestrator: OrchestratorAgent, quiz_data: list):
    """
    Интерактивный режим прохождения квиза в консоли.

    Args:
        orchestrator: Оркестратор для проверки ответов
        quiz_data: Список вопросов
    """
    print("\n" + "=" * 60)
    print(f"🚀 КВИЗ ГОТОВ! Всего вопросов: {len(quiz_data)}")
    print("=" * 60)
    print("Введите номер правильного ответа или 'exit' для выхода.\n")

    for i, question in enumerate(quiz_data, 1):
        print(f"❓ ВОПРОС {i}/{len(quiz_data)}")
        print(f"   {question['question']}")
        print("-" * 40)

        options = question.get('options', [])
        if question['type'] == 'multiple_choice':
            for idx, opt in enumerate(options, 1):
                print(f"   {idx}. {opt}")
        elif question['type'] == 'true_false':
            print("   1. True")
            print("   2. False")

        # Цикл ввода ответа
        while True:
            user_input = input("\n👉 Ваш ответ: ").strip().lower()

            if user_input in ['exit', 'quit']:
                print("⚠️ Выход из квиза...")
                return

            # Валидация и приведение к внутреннему формату
            formatted_answer = None
            try:
                if question['type'] == 'multiple_choice':
                    idx = int(user_input) - 1
                    if 0 <= idx < len(options):
                        formatted_answer = options[idx]
                elif question['type'] == 'true_false':
                    if user_input in ['1', 'true']:
                        formatted_answer = 'true'
                    elif user_input in ['2', 'false']:
                        formatted_answer = 'false'

                if formatted_answer is not None:
                    break
                print("❌ Некорректный ввод. Введите номер варианта.")
            except ValueError:
                print("❌ Введите число.")

        # Проверка
        print("⏳ Проверка...")
        result = orchestrator.submit_answer(question['question_id'], formatted_answer)

        if result['is_correct']:
            print(f"✅ ВЕРНО! (Счет: {result['score']}/{result['total']})")
        else:
            print(f"❌ ОШИБКА. Правильный ответ: {result['correct_answer']}")
            if result.get('explanation'):
                print(f"\n💡 ПОЯСНЕНИЕ:\n{result['explanation']}")
            if result.get('memory_palace'):
                print(f"\n🏰 ДВОРЕЦ ПАМЯТИ (для запоминания):\n{result['memory_palace']}")

        print("\n" + "_" * 60 + "\n")

    # Итоги
    stats = orchestrator.get_session_stats()
    print("=" * 60)
    print("🎉 ТЕСТ ЗАВЕРШЕН!")
    print("=" * 60)
    print(f"📊 Итоговый счет: {stats['score']} из {stats['total_questions']} ({stats['accuracy']}%)")
    print("=" * 60)


# ============================================================================
# ПАРСИНГ АРГУМЕНТОВ КОМАНДНОЙ СТРОКИ
# ============================================================================

def parse_arguments():
    """
    Парсинг аргументов командной строки.

    Returns:
        argparse.Namespace: Объект с аргументами
    """
    parser = argparse.ArgumentParser(
        description="🎓 Генератор Умных Квизов - CLI версия",
        epilog="Примеры использования:\n"
               "  python main.py notes.txt\n"
               "  python main.py notes.txt -d hard -q 10\n"
               "  python main.py notes.txt --force --debug\n",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Позиционный аргумент
    parser.add_argument(
        "file",
        help="Путь к файлу заметки (.txt, .md)"
    )

    # Опциональные аргументы
    parser.add_argument(
        "-d", "--difficulty",
        choices=['easy', 'medium', 'hard'],
        help="Сложность вопросов (по умолчанию: medium)"
    )

    parser.add_argument(
        "-q", "--questions",
        type=int,
        help="Количество вопросов (по умолчанию: из config.json)"
    )

    parser.add_argument(
        "-f", "--force",
        action="store_true",
        help="Принудительный парсинг (игнорировать кэш, даже если файл не изменился)"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Режим отладки (подробное логирование)"
    )

    return parser.parse_args()


# ============================================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ============================================================================

def main():
    """
    Главная функция приложения.

    Workflow:
    1. Парсинг аргументов командной строки
    2. Загрузка конфигурации и credentials
    3. Инициализация системы
    4. Чтение файла заметки
    5. Запуск пайплайна обработки
    6. Интерактивная сессия квиза
    """
    # 1. Парсинг аргументов
    args = parse_arguments()

    # 2. Проверка существования файла
    file_path = Path(args.file)
    if not file_path.exists():
        print(f"❌ Ошибка: Файл '{args.file}' не найден.")
        sys.exit(1)

    try:
        # 3. Инициализация системы
        setup_logging(args.debug)
        logger = logging.getLogger(__name__)

        logger.info("=" * 70)
        logger.info("APPLICATION START")
        logger.info("=" * 70)

        config = load_config()
        credentials = load_credentials()

        # Явно инициализируем CacheManager
        cache_manager = CacheManager(
            cache_dir=config.get('cache_settings', {}).get('cache_dir', 'data/cache')
        )

        orchestrator = OrchestratorAgent(config, credentials, cache_manager)

        # 4. Чтение файла
        logger.info(f"Reading file: {args.file}")
        with open(file_path, 'r', encoding='utf-8') as f:
            note_text = f.read()

        if not note_text.strip():
            print("❌ Файл пуст.")
            sys.exit(1)

        # Вывод информации о режиме
        print(f"\n⚙️ Запуск анализа файла: {args.file}")
        if args.force:
            print("🔄 Режим принудительного парсинга (кэш игнорируется)")
        if args.difficulty:
            print(f"🎯 Сложность: {args.difficulty}")
        if args.questions:
            print(f"📝 Количество вопросов: {args.questions}")
        print()

        # 5. Запуск пайплайна
        logger.info(f"Starting pipeline (force_reparse={args.force})")
        result = orchestrator.process_note_pipeline(
            note_text=note_text,
            questions_count=args.questions,
            difficulty=args.difficulty,
            force_reparse=args.force  # ✅ ПЕРЕДАЕМ ФЛАГ
        )

        if result['status'] == 'error':
            print(f"❌ Ошибка генерации: {result['message']}")
            sys.exit(1)

        print(f"✅ {result['message']}")

        # 6. Запуск квиза
        run_cli_quiz_session(orchestrator, result['quiz'])

        logger.info("Application finished successfully")

    except FileNotFoundError as e:
        print(f"\n❌ Ошибка: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"\n❌ Ошибка конфигурации: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⚠️ Программа прервана пользователем. До свидания!")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Critical Error: {e}", exc_info=True)
        print(f"\n❌ Критическая ошибка: {e}")
        print("Проверьте логи для подробностей.")
        sys.exit(1)


if __name__ == "__main__":
    main()
