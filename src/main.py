"""
Главная точка входа - LangGraph Workflow с CLI
"""

import sys
import os
from pathlib import Path
import argparse

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import logging
from dotenv import load_dotenv

from .config_loader import load_config
from .langgraph.workflow import QuizGenerationWorkflow
from .utils.helpers import load_lecture_from_file, format_quiz_results
from .utils.gigachat_client import get_global_token_tracker

load_dotenv()

"""
Настройка логов
"""
logging.basicConfig(
    level=logging.INFO, # Наименьший уровень инфы для вывода в лог
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', # Формат вывода в логе
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('quiz_generation.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


def main():
    """Основная функция с CLI"""

    # Парсинг аргументов
    parser = argparse.ArgumentParser(description='Генерация квиза из лекции')
    parser.add_argument('lecture_file', type=str, help='Путь к файлу лекции (.txt/.md)')
    parser.add_argument('--web-search', action='store_true', help='Включить веб-поиск для проверки фактов')
    parser.add_argument('--no-rag', action='store_true', help='Отключить RAG (только для коротких текстов)') #rag
    parser.add_argument('--output', type=str, default='quiz_result.json', help='Файл для сохранения результата')
    args = parser.parse_args()

    logger.info("="*70)
    logger.info("MINDFORGE QUIZ GENERATOR - LANGGRAPH WORKFLOW")
    logger.info("="*70)

    try:
        # 1. Загрузка конфигурации
        logger.info("\n1️⃣  Загрузка конфигурации")
        config = load_config()

        gigachat_credentials = os.getenv("GIGACHAT_CREDENTIALS")
        if not gigachat_credentials:
            raise ValueError("GIGACHAT_CREDENTIALS не найден в .env")

        # 2. Загрузка лекции
        logger.info(f"\n2️⃣  Загрузка лекции: {args.lecture_file}")
        lecture_text = load_lecture_from_file(args.lecture_file)
        logger.info(f"   📄 Длина: {len(lecture_text)} символов")
        logger.info(f"   📝 Слов: ~{len(lecture_text.split())}")

        # 3. Создание workflow
        logger.info("\n3️⃣  Инициализация LangGraph Workflow")
        workflow = QuizGenerationWorkflow(
            gigachat_credentials=gigachat_credentials,
            quiz_config=config,
            enable_web_search=args.web_search
        )

        logger.info(f"   RAG: {'✓ Включен' if not args.no_rag else '✗ Отключен'}") #rag
        logger.info(f"   Веб-поиск: {'✓ Включен' if args.web_search else '✗ Отключен'}")

        # 4. Запуск workflow
        logger.info("\n4️⃣  Запуск генерации квиза...")
        logger.info("-"*70)

        result = workflow.run(lecture_text)

        if result.get("error"):
            logger.error(f"\n✗ Ошибка: {result['error']}")
            tracker = get_global_token_tracker()
            tracker.log_summary()
            sys.exit(1)

        # 5. Форматирование результатов
        logger.info("\n5️⃣  Форматирование результатов")
        formatted_output = format_quiz_results(result, output_file=args.output)
        print(formatted_output)

        # 6. Вывод статистики токенов
        logger.info("\n6️⃣  Статистика использования токенов")
        tracker = get_global_token_tracker()
        tracker.log_summary()

        usage = tracker.get_summary()
        print("\n" + "="*70)
        print("📊 ИСПОЛЬЗОВАНИЕ ТОКЕНОВ GIGACHAT API")
        print("="*70)
        print(f"LLM запросов:         {usage['total_requests']}")
        print(f"  Промпт токенов:     {usage['prompt_tokens']:,}")
        print(f"  Ответ токенов:      {usage['completion_tokens']:,}")
        print(f"  Итого LLM:          {usage['total_tokens']:,}")
        print("="*70)

        # Финал
        logger.info("\n" + "="*70)
        logger.info("✓ ГЕНЕРАЦИЯ ЗАВЕРШЕНА УСПЕШНО")
        logger.info("="*70)
        logger.info(f"📁 Результаты: {args.output}")
        logger.info(f"📋 Логи: quiz_generation.log")

    except FileNotFoundError as e:
        logger.error(f"\n✗ Файл не найден: {e}")
        sys.exit(1)

    except ValueError as e:
        logger.error(f"\n✗ Ошибка конфигурации: {e}")
        sys.exit(1)

    except Exception as e:
        logger.error(f"\n✗ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()

        try:
            tracker = get_global_token_tracker()
            tracker.log_summary()
        except:
            pass

        sys.exit(1)


if __name__ == "__main__":
    main()
