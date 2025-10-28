"""
Пример использования LangGraph Workflow
"""

import os
import logging
from dotenv import load_dotenv
import json

from .langgraph.workflow import QuizGenerationWorkflow

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Загрузка переменных окружения
load_dotenv()

def main():
    """Основная функция"""

    # Получаем credentials
    gigachat_credentials = os.getenv("GIGACHAT_CREDENTIALS")
    if not gigachat_credentials:
        raise ValueError("GIGACHAT_CREDENTIALS не найден в .env файле")

    # Конфигурация Quiz Agent
    quiz_config = {
        "gigachat": {
            "api_key": gigachat_credentials,
            "model": "GigaChat",
            "temperature": 0.7
        },
        "quiz": {
            "num_questions": 7,
            "question_types": [
                "multiple_choice",
                "true_false",
                "short_answer"
            ]
        }
    }

    # Создаем workflow
    workflow = QuizGenerationWorkflow(
        gigachat_credentials=gigachat_credentials,
        quiz_config=quiz_config,
        use_rag=True,  # Включаем RAG
        enable_web_search=False  # Веб-поиск выключен (опционально)
    )

    # Пример текста лекции
    lecture_text = """
    Производная функции - одно из фундаментальных понятий математического анализа.
    
    Производная функции f(x) в точке x₀ определяется как предел отношения приращения 
    функции к приращению аргумента при стремлении приращения аргумента к нулю:
    
    f'(x₀) = lim(Δx→0) [f(x₀ + Δx) - f(x₀)] / Δx
    
    Геометрический смысл производной: производная в точке равна угловому коэффициенту 
    касательной к графику функции в этой точке (тангенсу угла наклона).
    
    Основные правила дифференцирования:
    1. Производная константы: (C)' = 0
    2. Производная степенной функции: (x^n)' = n·x^(n-1)
    3. Производная суммы: (f + g)' = f' + g'
    4. Производная произведения: (f·g)' = f'·g + f·g'
    5. Производная частного: (f/g)' = (f'·g - f·g') / g²
    
    Физический смысл производной: если s(t) - путь, пройденный телом за время t,
    то s'(t) - это мгновенная скорость тела в момент времени t.
    """

    # Запуск workflow
    logger.info("Запуск генерации квиза...")
    result = workflow.run(lecture_text)

    # Проверка на ошибки
    if result.get("error"):
        logger.error(f"Workflow завершился с ошибкой: {result['error']}")
        return

    # Вывод результатов
    print("\n" + "="*70)
    print("РЕЗУЛЬТАТЫ")
    print("="*70)

    print(f"\n📝 ИЗВЛЕЧЕННЫЕ ФАКТЫ ({len(result['key_facts'])}):")
    for i, fact in enumerate(result['key_facts'], 1):
        print(f"{i}. {fact}")

    if result.get('quiz_questions'):
        print(f"\n❓ СГЕНЕРИРОВАННЫЕ ВОПРОСЫ ({len(result['quiz_questions'])}):")
        for i, q in enumerate(result['quiz_questions'], 1):
            print(f"\n--- Вопрос {i} ---")
            print(f"Тип: {q['question_type']}")
            print(f"Сложность: {q['difficulty']}")
            print(f"Вопрос: {q['question_text']}")
            if q.get('options'):
                print("Варианты:")
                for opt in q['options']:
                    print(f"  - {opt}")
            print(f"Правильный ответ: {q['correct_answer']}")
            print(f"Объяснение: {q['explanation']}")

    # Сохранение результатов
    with open("quiz_result.json", "w", encoding="utf-8") as f:
        json.dump({
            "key_facts": result['key_facts'],
            "quiz_questions": result['quiz_questions']
        }, f, ensure_ascii=False, indent=2)

    logger.info("\n✓ Результаты сохранены в quiz_result.json")


if __name__ == "__main__":
    main()
