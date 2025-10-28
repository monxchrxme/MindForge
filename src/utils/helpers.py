"""
Вспомогательные функции
"""

from pathlib import Path
from typing import Dict, Any
import json
import logging

logger = logging.getLogger(__name__)


def load_lecture_from_file(file_path: str) -> str:
    """
    Загрузка текста лекции из файла

    Args:
        file_path: путь к TXT файлу с лекцией

    Returns:
        текст лекции
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Файл лекции не найден: {file_path}")

    if not path.suffix.lower() in ['.txt', '.md']:
        raise ValueError(f"Поддерживаются только .txt и .md файлы, получен: {path.suffix}")

    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()

    if not content.strip():
        raise ValueError(f"Файл пустой: {file_path}")

    logger.info(f"✓ Загружена лекция: {file_path} ({len(content)} символов)")
    return content


def format_quiz_results(result: Dict[str, Any], output_file: str = None) -> str:
    """
    Форматирование результатов квиза для вывода

    Args:
        result: результат работы workflow
        output_file: опционально - путь для сохранения JSON

    Returns:
        отформатированная строка
    """
    output = []

    # Факты
    output.append("\n" + "="*70)
    output.append("📝 ИЗВЛЕЧЕННЫЕ КЛЮЧЕВЫЕ ФАКТЫ")
    output.append("="*70)

    for i, fact in enumerate(result.get('key_facts', []), 1):
        output.append(f"\n{i}. {fact}")

    # Вопросы квиза
    if result.get('quiz_questions'):
        output.append("\n" + "="*70)
        output.append("❓ СГЕНЕРИРОВАННЫЙ КВИЗ")
        output.append("="*70)

        for i, q in enumerate(result['quiz_questions'], 1):
            output.append(f"\n{'='*40}")
            output.append(f"Вопрос {i}")
            output.append(f"{'='*40}")
            output.append(f"Тип: {q['question_type']}")
            output.append(f"Сложность: {q['difficulty']}")
            output.append(f"\n{q['question_text']}")

            if q.get('options'):
                output.append("\nВарианты ответа:")
                for opt in q['options']:
                    output.append(f"  {opt}")

            output.append(f"\n✓ Правильный ответ: {q['correct_answer']}")
            output.append(f"💡 Объяснение: {q['explanation']}")

    # Статистика
    output.append("\n" + "="*70)
    output.append("📊 СТАТИСТИКА")
    output.append("="*70)
    output.append(f"Фактов извлечено: {len(result.get('key_facts', []))}")
    output.append(f"Вопросов создано: {len(result.get('quiz_questions', []))}")

    formatted_text = "\n".join(output)

    # Сохранение в JSON
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'key_facts': result.get('key_facts', []),
                'quiz_questions': result.get('quiz_questions', []),
                'concepts': result.get('concepts', [])
            }, f, ensure_ascii=False, indent=2)
        logger.info(f"✓ Результаты сохранены: {output_file}")

    return formatted_text
