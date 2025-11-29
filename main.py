# main.py

"""
Точка входа в приложение «Генератор Умных Квизов» (MVP).

Основной workflow:
    1. Загрузка конфигурации и секретов
    2. Инициализация сервисов (GigaChatClient, CacheManager)
    3. Создание OrchestratorAgent
    4. Запуск интерактивного CLI интерфейса
    5. Обработка команд пользователя

Архитектура:
    - Stateless агенты выполняют специализированные задачи
    - OrchestratorAgent координирует работу и хранит состояние сессии
    - CacheManager экономит токены через файловый кэш
    - Все взаимодействие с LLM через GigaChatClient
"""
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Импорты из нашей архитектуры
from agents import OrchestratorAgent
from services import GigaChatClient, CacheManager
from utils import compute_short_hash


# ============================================================================
# НАСТРОЙКА ЛОГИРОВАНИЯ
# ============================================================================

def setup_logging(config: dict) -> None:
    """
    Настройка системы логирования согласно config.json.

    Args:
        config: Конфигурация из config.json
    """
    log_config = config.get('logging', {})
    log_level = log_config.get('level', 'INFO')
    log_file = log_config.get('file', 'data/logs/app.log')
    log_format = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Создание директории для логов
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Настройка корневого логгера
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )

    logger = logging.getLogger(__name__)
    logger.info("=" * 70)
    logger.info("Logging system initialized")
    logger.info(f"Log level: {log_level}")
    logger.info(f"Log file: {log_file}")
    logger.info("=" * 70)


# ============================================================================
# ЗАГРУЗКА КОНФИГУРАЦИИ
# ============================================================================

def load_config(config_path: str = "config.json") -> dict:
    """
    Загрузка конфигурации из JSON файла.

    Args:
        config_path: Путь к файлу конфигурации

    Returns:
        dict: Конфигурация приложения

    Raises:
        FileNotFoundError: Если config.json не найден
        json.JSONDecodeError: Если config.json невалиден
    """
    logger = logging.getLogger(__name__)

    if not os.path.exists(config_path):
        logger.error(f"Configuration file not found: {config_path}")
        raise FileNotFoundError(f"Config file '{config_path}' does not exist")

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        logger.info(f"Configuration loaded from {config_path}")
        return config

    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in config file: {str(e)}")
        raise


def load_credentials() -> dict:
    """
    Загрузка секретных ключей из .env файла.

    Returns:
        dict: Словарь с ключами 'client_id' и 'client_secret'

    Raises:
        ValueError: Если обязательные переменные окружения не найдены
    """
    logger = logging.getLogger(__name__)

    # Загрузка переменных из .env
    load_dotenv()

    client_id = os.getenv('GIGACHAT_CLIENT_ID')
    client_secret = os.getenv('GIGACHAT_CREDENTIALS') #

    if not client_id or not client_secret:
        logger.error("Missing required environment variables")
        raise ValueError(
            "GIGACHAT_CLIENT_ID and GIGACHAT_CREDENTIALS must be set in .env file"
        )

    logger.info("Credentials loaded successfully")

    return {
        'client_id': client_id,
        'client_secret': client_secret
    }


# ============================================================================
# ИНИЦИАЛИЗАЦИЯ СИСТЕМЫ
# ============================================================================

def initialize_system(config: dict, credentials: dict) -> OrchestratorAgent:
    """
    Инициализация всех компонентов системы.

    Args:
        config: Конфигурация из config.json
        credentials: Секретные ключи из .env

    Returns:
        OrchestratorAgent: Готовый к работе оркестратор
    """
    logger = logging.getLogger(__name__)

    logger.info("Initializing system components...")

    # 1. Создание CacheManager
    cache_dir = config.get('cache_settings', {}).get('cache_dir', 'data/cache')
    cache_manager = CacheManager(cache_dir=cache_dir)
    logger.info(f"✓ CacheManager initialized: {cache_dir}")

    # 2. Создание GigaChatClient
    llm_settings = config.get('llm_settings', {})
    client = GigaChatClient(
        credentials=credentials,
        model=llm_settings.get('model', 'GigaChat'),
        temperature=llm_settings.get('temperature', 0.7),
        timeout=llm_settings.get('timeout', 30),
        verify_ssl_certs=llm_settings.get('verify_ssl_certs', False)
    )
    logger.info(f"✓ GigaChatClient initialized: {llm_settings.get('model')}")

    # 3. Создание OrchestratorAgent
    orchestrator = OrchestratorAgent(
        config=config,
        credentials=credentials,
        cache_manager=cache_manager
    )
    logger.info("✓ OrchestratorAgent initialized")

    logger.info("System initialization complete!")

    return orchestrator


# ============================================================================
# ИНТЕРАКТИВНЫЙ CLI ИНТЕРФЕЙС
# ============================================================================

def print_welcome() -> None:
    """Вывод приветственного сообщения."""
    print("\n" + "=" * 70)
    print("🎓 ГЕНЕРАТОР УМНЫХ КВИЗОВ - MVP")
    print("=" * 70)
    print("Превратите ваши заметки в интерактивные квизы для самопроверки!")
    print("=" * 70 + "\n")


def print_menu() -> None:
    """Вывод главного меню."""
    print("\n📋 ДОСТУПНЫЕ КОМАНДЫ:")
    print("  1. new    - Создать новый квиз из заметки")
    print("  2. regen  - Регенерировать квиз (новые вопросы)")
    print("  3. stats  - Показать статистику сессии")
    print("  4. help   - Показать справку")
    print("  5. exit   - Выход из программы")
    print()


def read_note_from_file(file_path: str) -> Optional[str]:
    """
    Чтение текста заметки из файла.

    Args:
        file_path: Путь к файлу с заметкой

    Returns:
        str: Текст заметки или None при ошибке
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"❌ Ошибка чтения файла: {str(e)}")
        return None


def read_note_input() -> str:
    """
    Чтение многострочного ввода заметки от пользователя.

    Returns:
        str: Текст заметки
    """
    print("\n📝 Введите текст заметки (пустая строка для завершения):")
    print("-" * 70)

    lines = []
    while True:
        try:
            line = input()
            if not line.strip():
                break
            lines.append(line)
        except EOFError:
            break

    return '\n'.join(lines)


def display_quiz(quiz: list) -> None:
    """
    Отображение вопросов квиза.

    Args:
        quiz: Список вопросов из OrchestratorAgent
    """
    print("\n" + "=" * 70)
    print("📚 ВАШ КВИЗ ГОТОВ!")
    print("=" * 70)

    for i, question in enumerate(quiz, 1):
        print(f"\n❓ Вопрос {i}/{len(quiz)}")
        print(f"   {question.get('question', 'N/A')}")

        q_type = question.get('type', 'unknown')

        if q_type == 'multiple_choice':
            options = question.get('options', [])
            for idx, option in enumerate(options):
                print(f"   {idx + 1}. {option}")
        elif q_type == 'true_false':
            print("   1. True")
            print("   2. False")

        print()


def run_quiz(orchestrator: OrchestratorAgent, quiz: list) -> None:
    """
    Интерактивное прохождение квиза.

    Args:
        orchestrator: Оркестратор для проверки ответов
        quiz: Список вопросов
    """
    print("\n" + "=" * 70)
    print("🎯 НАЧИНАЕМ ТЕСТИРОВАНИЕ!")
    print("=" * 70)
    print("Введите номер ответа или 'skip' для пропуска вопроса\n")

    for i, question in enumerate(quiz, 1):
        print(f"\n📌 Вопрос {i}/{len(quiz)}")
        print(f"   {question.get('question', 'N/A')}")

        q_type = question.get('type', 'unknown')
        question_id = question.get('question_id', '')

        # Отображение вариантов
        if q_type == 'multiple_choice':
            options = question.get('options', [])
            for idx, option in enumerate(options):
                print(f"   {idx + 1}. {option}")
        elif q_type == 'true_false':
            print("   1. True")
            print("   2. False")

        # Ввод ответа
        while True:
            user_input = input("\n👉 Ваш ответ: ").strip().lower()

            if user_input == 'skip':
                print("⏭️  Вопрос пропущен")
                break

            # Валидация ввода
            try:
                if q_type == 'multiple_choice':
                    answer_idx = int(user_input) - 1
                    if 0 <= answer_idx < len(options):
                        user_answer = str(answer_idx)
                        break
                elif q_type == 'true_false':
                    if user_input in ['1', '2', 'true', 'false']:
                        user_answer = 'true' if user_input in ['1', 'true'] else 'false'
                        break
                elif q_type == 'open_ended':
                    user_answer = user_input
                    break

                print("❌ Некорректный ввод. Попробуйте снова.")
            except ValueError:
                print("❌ Введите число.")

        if user_input == 'skip':
            continue

        # Проверка ответа через оркестратор
        result = orchestrator.submit_answer(question_id, user_answer)

        if result.get('is_correct'):
            print(f"✅ Правильно! Счёт: {result.get('score')}/{result.get('progress').split('/')[1]}")
        else:
            print(f"❌ Неправильно. Правильный ответ: {result.get('correct_answer')}")

            # Вывод объяснения
            explanation = result.get('explanation', '')
            if explanation:
                print(f"\n💡 Объяснение:\n   {explanation}")

            # Вывод мнемонического образа
            memory_palace = result.get('memory_palace', '')
            if memory_palace:
                print(f"\n🏰 Дворец памяти:\n   {memory_palace}")

    # Финальная статистика
    stats = orchestrator.get_session_stats()
    print("\n" + "=" * 70)
    print("🎊 ТЕСТ ЗАВЕРШЁН!")
    print("=" * 70)
    print(f"📊 Результат: {stats['score']}/{stats['total']}")
    print(f"📈 Точность: {stats['accuracy']}%")
    print("=" * 70)


def display_statistics(orchestrator: OrchestratorAgent) -> None:
    """
    Отображение статистики текущей сессии.

    Args:
        orchestrator: Оркестратор с данными статистики
    """
    stats = orchestrator.get_session_stats()

    print("\n" + "=" * 70)
    print("📊 СТАТИСТИКА СЕССИИ")
    print("=" * 70)
    print(f"✅ Правильных ответов: {stats['score']}")
    print(f"📝 Всего отвечено: {stats['total']}")
    print(f"📈 Точность: {stats['accuracy']}%")
    print(f"🧠 Концептов извлечено: {stats['concepts_extracted']}")
    print(f"❓ Вопросов сгенерировано: {stats['questions_generated']}")
    print(f"📜 Вопросов в истории: {stats['questions_in_history']}")

    llm_stats = stats.get('llm_stats', {})
    print(f"\n🤖 LLM Статистика:")
    print(f"   Токенов в промптах: {llm_stats.get('prompt_tokens', 0)}")
    print(f"   Токенов в ответах: {llm_stats.get('completion_tokens', 0)}")
    print(f"   Всего запросов: {llm_stats.get('total_requests', 0)}")
    print("=" * 70)


def run_interactive_mode(orchestrator: OrchestratorAgent) -> None:
    """
    Запуск интерактивного режима CLI.

    Args:
        orchestrator: Инициализированный оркестратор
    """
    logger = logging.getLogger(__name__)

    print_welcome()

    while True:
        print_menu()
        command = input("👉 Введите команду: ").strip().lower()

        if command in ['1', 'new']:
            # Создание нового квиза
            print("\n📂 Выберите источник заметки:")
            print("  1. Ввести текст вручную")
            print("  2. Загрузить из файла")

            choice = input("👉 Ваш выбор: ").strip()

            note_text = None
            if choice == '1':
                note_text = read_note_input()
            elif choice == '2':
                file_path = input("📁 Путь к файлу: ").strip()
                note_text = read_note_from_file(file_path)
            else:
                print("❌ Некорректный выбор")
                continue

            if not note_text or not note_text.strip():
                print("❌ Текст заметки пуст")
                continue

            # Генерация квиза
            print("\n⏳ Анализирую заметку и генерирую квиз...")
            note_hash = compute_short_hash(note_text, length=8)
            logger.info(f"Processing note {note_hash}")

            result = orchestrator.start_new_session(note_text)

            if result['status'] == 'success':
                print(f"✅ {result['message']}")
                quiz = result['quiz']
                display_quiz(quiz)

                # Предложение пройти тест
                proceed = input("\n🎯 Пройти тест сейчас? (y/n): ").strip().lower()
                if proceed == 'y':
                    run_quiz(orchestrator, quiz)
            else:
                print(f"❌ {result['message']}")

        elif command in ['2', 'regen']:
            # Регенерация квиза
            print("\n⏳ Генерирую новые вопросы...")
            result = orchestrator.regenerate_quiz()

            if result['status'] == 'success':
                print(f"✅ {result['message']}")
                quiz = result['quiz']
                display_quiz(quiz)

                proceed = input("\n🎯 Пройти тест сейчас? (y/n): ").strip().lower()
                if proceed == 'y':
                    run_quiz(orchestrator, quiz)
            else:
                print(f"❌ {result['message']}")

        elif command in ['3', 'stats']:
            # Статистика
            display_statistics(orchestrator)

        elif command in ['4', 'help']:
            # Справка
            print("\n" + "=" * 70)
            print("📖 СПРАВКА")
            print("=" * 70)
            print("Эта система превращает ваши учебные заметки в интерактивные квизы.")
            print("\nОсновной workflow:")
            print("  1. Создайте новый квиз командой 'new'")
            print("  2. Введите или загрузите текст заметки")
            print("  3. Система извлечет ключевые концепты")
            print("  4. Сгенерирует вопросы для самопроверки")
            print("  5. Пройдите тест и получите объяснения ошибок")
            print("\nКоманда 'regen' создаст новые вопросы по тем же концептам.")
            print("=" * 70)

        elif command in ['5', 'exit', 'quit']:
            # Выход
            print("\n👋 До свидания! Удачи в учёбе!")
            logger.info("Application terminated by user")
            break

        else:
            print("❌ Неизвестная команда. Введите 'help' для справки.")


# ============================================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ============================================================================

def main() -> None:
    """
    Главная функция приложения.

    Последовательность:
        1. Загрузка config.json
        2. Настройка логирования
        3. Загрузка .env credentials
        4. Инициализация системы
        5. Запуск интерактивного режима
    """
    try:
        # 1. Загрузка конфигурации
        config = load_config()

        # 2. Настройка логирования
        setup_logging(config)

        logger = logging.getLogger(__name__)
        logger.info("Application started")

        # 3. Загрузка credentials
        credentials = load_credentials()

        # 4. Инициализация системы
        orchestrator = initialize_system(config, credentials)

        # 5. Запуск интерактивного режима
        run_interactive_mode(orchestrator)

    except FileNotFoundError as e:
        print(f"\n❌ Ошибка: {str(e)}")
        print("Убедитесь, что файлы config.json и .env существуют.")
        sys.exit(1)

    except ValueError as e:
        print(f"\n❌ Ошибка конфигурации: {str(e)}")
        sys.exit(1)

    except KeyboardInterrupt:
        print("\n\n👋 Программа прервана пользователем. До свидания!")
        sys.exit(0)

    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        print(f"\n❌ Критическая ошибка: {str(e)}")
        print("Проверьте логи для подробностей.")
        sys.exit(1)


if __name__ == "__main__":
    main()
