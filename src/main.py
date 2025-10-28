"""
Главная точка входа - запуск LangGraph Workflow
Поддержка загрузки лекций из внешних TXT файлов
"""

import sys
from pathlib import Path

# Добавляем корень проекта в sys.path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

import os
import logging
from dotenv import load_dotenv
import argparse

from src.config_loader import load_config
from src.langgraph.workflow import QuizGenerationWorkflow
from src.utils.helpers import load_lecture_from_file, format_quiz_results

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('quiz_generation.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# Загрузка переменных окружения
load_dotenv()


def parse_arguments():
    """Парсинг аргументов командной строки"""
    parser = argparse.ArgumentParser(
        description='Генерация квиза из лекции с помощью LangGraph и GigaChat'
    )

    parser.add_argument(
        'lecture_file',
        type=str,
        help='Путь к TXT/MD файлу с текстом лекции'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Путь к файлу конфигурации (по умолчанию: config/config.yaml)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='quiz_result.json',
        help='Путь для сохранения результата в JSON (по умолчанию: quiz_result.json)'
    )

    parser.add_argument(
        '--no-rag',
        action='store_true',
        help='Отключить RAG в Parser Agent'
    )

    parser.add_argument(
        '--web-search',
        action='store_true',
        help='Включить веб-поиск для обогащения контекста'
    )

    return parser.parse_args()


def main():
    """Основная функция"""

    # Парсинг аргументов
    args = parse_arguments()

    logger.info("="*70)
    logger.info("OBSIDIAN QUIZ PLUGIN - LANGGRAPH WORKFLOW")
    logger.info("="*70)

    try:
        # 1. Загрузка конфигурации
        logger.info(f"\n1️⃣  Загрузка конфигурации: {args.config}")
        config = load_config(args.config)

        # Проверка credentials
        gigachat_credentials = config['gigachat'].get('credentials')
        if not gigachat_credentials:
            raise ValueError(
                "GIGACHAT_CREDENTIALS не найден.\n"
                "Добавьте в .env файл:\n"
                "GIGACHAT_CREDENTIALS=your_api_key_here"
            )

        # 2. Загрузка лекции
        logger.info(f"\n2️⃣  Загрузка лекции: {args.lecture_file}")
        lecture_text = load_lecture_from_file(args.lecture_file)
        logger.info(f"   Длина текста: {len(lecture_text)} символов")
        logger.info(f"   Слов: ~{len(lecture_text.split())}")

        # 3. Создание workflow
        logger.info("\n3️⃣  Инициализация LangGraph Workflow")

        workflow = QuizGenerationWorkflow(
            gigachat_credentials=gigachat_credentials,
            quiz_config=config,
            use_rag=not args.no_rag,
            enable_web_search=args.web_search
        )

        logger.info(f"   RAG: {'✓ Включен' if not args.no_rag else '✗ Отключен'}")
        logger.info(f"   Веб-поиск: {'✓ Включен' if args.web_search else '✗ Отключен'}")

        # 4. Запуск workflow
        logger.info("\n4️⃣  Запуск генерации квиза...")
        logger.info("-"*70)

        result = workflow.run(lecture_text)

        # 5. Проверка на ошибки
        if result.get("error"):
            logger.error(f"\n✗ Workflow завершился с ошибкой: {result['error']}")
            sys.exit(1)

        # 6. Вывод результатов
        logger.info("\n5️⃣  Форматирование и сохранение результатов")

        formatted_output = format_quiz_results(result, output_file=args.output)
        print(formatted_output)

        # 7. Финальная статистика
        logger.info("\n" + "="*70)
        logger.info("✓ ГЕНЕРАЦИЯ ЗАВЕРШЕНА УСПЕШНО")
        logger.info("="*70)
        logger.info(f"Результаты сохранены в: {args.output}")
        logger.info(f"Логи записаны в: quiz_generation.log")

    except FileNotFoundError as e:
        logger.error(f"\n✗ Ошибка: {e}")
        logger.error("Проверьте путь к файлу лекции")
        sys.exit(1)

    except ValueError as e:
        logger.error(f"\n✗ Ошибка конфигурации: {e}")
        sys.exit(1)

    except Exception as e:
        logger.error(f"\n✗ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
