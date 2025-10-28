"""
Главная точка входа приложения
Демонстрирует работу Parser Agent (пока без Quiz Agent)
"""

import logging
import sys
from pathlib import Path

from .agents.parser_agent import ParserAgent
from .utils.helpers import load_note

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """
    Главная функция для демонстрации Parser Agent
    Аналогично твоему коду тестирования из ноутбука
    """
    print("\n" + "=" * 60)
    print("OBSIDIAN QUIZ PLUGIN - PARSER AGENT DEMO")
    print("=" * 60 + "\n")

    # Проверка аргументов командной строки
    if len(sys.argv) < 2:
        print("Использование: python -m src.main <путь_к_заметке>")
        print("\nПример:")
        print("  python -m src.main data/sample_notes/sample_lecture.txt")
        sys.exit(1)

    note_path = sys.argv[1]

    try:
        # Загрузка заметки (аналогично твоему open())
        logger.info(f"Загрузка заметки: {note_path}")
        lecture_text = load_note(note_path)

        # Инициализация state (как в твоем коде)
        state = {
            "lecture_text": lecture_text,
            "key_facts": [],
            "quiz_questions": [],
            "messages": [],
            "current_step": "start"
        }

        # Создание и запуск Parser Agent
        logger.info("Инициализация Parser Agent...")
        parser = ParserAgent()

        logger.info("Запуск парсинга...")
        result_state = parser.process(state)

        # Вывод результатов (как в твоем коде)
        print("\n" + "=" * 60)
        print("РЕЗУЛЬТАТЫ ПАРСИНГА")
        print("=" * 60 + "\n")

        print(f"Извлечено фактов: {len(result_state['key_facts'])}\n")

        for i, fact in enumerate(result_state['key_facts'], 1):
            print(f"{i}. ✓ {fact}")

        print("\n" + "=" * 60)
        print("СТАТУС: Parser Agent работает успешно!")
        print("=" * 60)

        # TODO: После реализации Quiz Agent добавить генерацию квиза
        print("\n⚠️  Quiz Agent пока не реализован (будет добавлен на следующем этапе)")

    except FileNotFoundError as e:
        logger.error(f"Ошибка: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Непредвиденная ошибка: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
