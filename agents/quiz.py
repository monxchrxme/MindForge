# agents/quiz.py

from typing import List, Dict, Set, Any
from services.gigachat_client import GigaChatClient
import uuid


class QuizAgent:
    """
    Агент-экзаменатор. Использует LLM для генерации уникальных вопросов по концептам.
    Формат входных и выходных данных строго соответствует архитектуре проекта.
    """

    def __init__(
            self,
            client: GigaChatClient,
            questions_count: int = 5,
            difficulty: str = "medium"
    ):
        """
        :param client: Экземпляр GigaChatClient (обязательный для всех агентов)
        :param questions_count: Сколько вопросов генерировать за один квиз
        :param difficulty: Уровень сложности вопросов (например, 'easy', 'medium', 'hard')
        """
        self.client = client
        self.questions_count = questions_count
        self.difficulty = difficulty

    def generate_questions(
            self,
            concepts: List[Dict[str, Any]],
            avoid_history: Set[str]
    ) -> List[Dict[str, Any]]:
        """
        Генерирует уникальные вопросы на основе списка концептов.

        :param concepts: Список концептов [{ "term": str, "definition": str, ... }, ...]
        :param avoid_history: Множество (set) хешей текстов вопросов, которые нельзя повторять в этой сессии

        :return: Список новых вопросов в формате:
        [
            {
                "question_id": str, # UUID для уникальности (генерируется здесь)
                "question": str,    # текст вопроса
                "type": str,        # "multiple_choice", "true_false", etc.
                "options": List[str],         # для multiple_choice
                "correct_answer": str,
                "related_concept": str,       # термин/ключ, на который ссылается вопрос
                "concept_definition": str     # определение (для ExplainAgent)
            },
            ...
        ]
        """
        prompt = self._questions_prompt(concepts, avoid_history)
        raw_questions = self.client.generate_json(prompt)
        valid_questions = self._validate_unique(raw_questions, avoid_history)
        return self._post_process_questions(valid_questions, concepts)

    def _questions_prompt(
            self,
            concepts: List[Dict[str, Any]],
            avoid_history: Set[str]
    ) -> str:
        """
        Собирает системный промпт для LLM.
        :param concepts: Список концептов [{ "term":..., "definition":...}]
        :param avoid_history: Множество текстов/хешей ранее сгенерированных вопросов
        :return: Строка-промпт
        """
        avoid_part = ""
        if avoid_history:
            avoid_part = (
                    "\n".join(list(avoid_history))
            )
        concept_part = "\n".join([
            f"{c['term']}: {c['definition']}" for c in concepts
        ])

        '''
        старый промт 
        '''
        # prompt = (
        #     f"На основе следующих понятий и определений:\n{concept_part}\n\n"
        #     f"Сгенерируй {self.questions_count} уникальных вопросов уровня сложности '{self.difficulty}' "
        #     f"(разных типов: множественный выбор, верно/неверно и т.п.). "
        #     f"{avoid_part}\n"
        #     "Формат ответа — JSON list из словарей, в каждом:\n"
        #     "{'question': str, 'type': str, 'options': list|null, 'correct_answer': str, "
        #     "'related_concept': str}\nВсе ответы должны быть различны по сути и формулировке."
        # )

        prompt = (
            f"Ты — генератор учебных вопросов для интеллектуальной системы квизов. "
            f"Твоя задача — на основе следующих понятий и их определений:\n"
            f"{concept_part}\n\n"
            f"Сгенерируй {self.questions_count} уникальных осмысленных образовательных вопросов уровня сложности '{self.difficulty}'.\n"
            f"Типы вопросов должны быть разнообразны и включать: множественный выбор (1 или больше вариантов ответа) (multiple_choice), верно/неверно (true_false), сопоставление (comparison) "
            f"Старайся, чтобы примерно 60% вопросов были с выбором ответа, 20% — верно/неверно, 20% — на сопоставление.\n\n"

            "Требования к вопросам:\n"
            "— Каждый вопрос должен проверять понимание концепта, а не простое запоминание определения.\n"
            "— Вопросы должны максимально различаться по смыслу и формулировкам, не допускать перефразирования одного и того же.\n"
            "— Не используй слова 'всегда', 'никогда' и другие универсальные утверждения.\n"
            "— Дистракторы (неправильные варианты в multiple_choice) должны быть правдоподобны и не вызывать сомнений своей искусственностью.\n"
            "— Каждый вопрос обязательно должен быть связан с одним из переданных концептов.\n"
            "— Для каждого вопроса в поле 'related_concept' укажи, к какому именно термину он относится из списка выше.\n"
            "— НЕ создавай вопросы, похожие на эти (сравнивай по смыслу, теме и структуре!):\n"
            f"{avoid_part}\n\n"

            "СТРОГИЙ формат ответа — JSON массив (list) из словарей, где каждый словарь (объект) обязательно содержит такие поля:\n"
            "  'question': (str) — текст вопроса (до 180 символов)\n"
            "  'type': (str) — тип вопроса: 'multiple_choice', 'true_false' или 'comparison'\n"
            "  'options': (list или null) — Список строк-ответов если multiple_choice (4-6 вариантов), если comparison, то список сопоставлений вопрос-утверждение в формате (текст вопроса 1 - текст утверждения a, текст вопроса 2 - текст утверждения b, ..) (3-5 вариантов), иначе null\n"
            "  'correct_answer': (str) — верный вариант ('a', 'b', 'c', 'd' — для multiple_choice; 'True'/'False' — для true_false; список правильных сопоставлений в формате (1-a, 2-b) для comparison)\n"
            "  'related_concept': (str) — термин из переданного списка\n\n"

            "Пример одного объекта для multiple_choice:\n"
            "{"
            "  'question': 'Какой основной принцип способа loci?',"
            "  'type': 'multiple_choice',"
            "  'options': ['Использование пространства памяти', 'Цветовые ассоциации', 'Ассоциативные числа', 'Усиление эмоций'],"
            "  'correct_answer': 'Использование пространства памяти',"
            "  'related_concept': 'метод локи'"
            "}\n\n"

            "Дополнительные указания:\n"
            "— НЕ добавляй пояснений, комментариев, текста до и после/после JSON — только сам JSON-массив!\n"
            "— Соблюдай формат строго, иначе твой ответ не пройдет автоматическую проверку.\n"
            "— Вопросы должны быть максимально информативны для учебного теста.\n"
        )
        return prompt

    def _validate_unique(
            self,
            questions: List[Dict[str, Any]],
            history: Set[str]
    ) -> List[Dict[str, Any]]:
        """
        Фильтрует вопросы, чтобы убрать дубли из avoid_history (по тексту вопроса).
        :param questions: Сырые вопросы из LLM
        :param history: Сет текстов старых вопросов
        :return: Новый список
        """
        unique = []
        seen = set(history)
        for q in questions:
            text = q.get("question", "").strip().lower()
            if text and text not in seen:
                unique.append(q)
                seen.add(text)
        return unique

    def _post_process_questions(
            self,
            questions: List[Dict[str, Any]],
            concepts: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Добавляет UUID, подтягивает определение концепта для ExplainAgent.
        :param questions: Список вопросов без дубликатов
        :param concepts: Исходные концепты
        :return: Финализированный список вопросов для Orchestrator
        """
        concept_lookup = {c["term"]: c["definition"] for c in concepts}
        for q in questions:
            q["question_id"] = str(uuid.uuid4())
            related = q.get("related_concept") or ""
            q["concept_definition"] = concept_lookup.get(related, "")
        return questions
