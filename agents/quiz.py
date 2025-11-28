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
        prompt = (
            f"На основе следующих понятий и определений:\n{concept_part}\n\n"
            f"Сгенерируй {self.questions_count} уникальных вопросов уровня сложности '{self.difficulty}' "
            f"(разных типов: множественный выбор, верно/неверно и т.п.). "
            f"{avoid_part}\n"
            "Формат ответа — JSON list из словарей, в каждом:\n"
            "{'question': str, 'type': str, 'options': list|null, 'correct_answer': str, "
            "'related_concept': str}\nВсе ответы должны быть различны по сути и формулировке."
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
