# agents/orchestrator.py

from agents.parser import ParserAgent
from agents.factcheck import FactCheckAgent
from agents.quiz import QuizAgent
from agents.explain import ExplainAgent
from services.gigachat_client import GigaChatClient
from services.cache_manager import CacheManager
from utils.hashing import compute_hash
from typing import Any, Dict, List, Set, Optional
import logging

logger = logging.getLogger(__name__)


class OrchestratorAgent:
    """
    Центральный координатор мульти-агентной системы.
    Единственный класс, который хранит состояние («память») текущей сессии.
    Управляет потоком данных между специализированными агентами.
    """

    def __init__(
            self,
            config: dict,
            credentials: dict,
            cache_manager: CacheManager
    ):
        """
        Инициализация оркестратора и всех подчиненных агентов.

        Args:
            config: Настройки из config.json (llm_settings, quiz_settings, etc.)
            credentials: Словарь с ключами GigaChat (client_id, client_secret)
            cache_manager: Менеджер файлового кэша для экономии токенов
        """
        # Инициализация клиента GigaChat
        llm_settings = config.get("llm_settings", {})
        self.client = GigaChatClient(
            credentials=credentials,
            model=llm_settings.get("model", "GigaChat"),
            temperature=llm_settings.get("temperature", 0.7)
        )

        # Инициализация агентов
        cache_enabled = config.get("cache_enabled", True)
        self.parser = ParserAgent(
            client=self.client,
            cache_manager=cache_manager,
            cache_enabled=cache_enabled
        )

        self.fact_checker = FactCheckAgent(client=self.client)

        quiz_settings = config.get("quiz_settings", {})
        self.quiz_generator = QuizAgent(
            client=self.client,
            questions_count=quiz_settings.get("questions_count", 5),
            difficulty=quiz_settings.get("difficulty", "medium")
        )

        self.explainer = ExplainAgent(client=self.client)

        # Настройки
        self.factcheck_enabled = config.get("enable_fact_check", True)

        # Состояние сессии (оперативная память)
        self.current_note_hash: str = ""
        self.current_note_text: str = ""
        self.extracted_concepts: List[Dict] = []
        self.verified_concepts: List[Dict] = []
        self.current_quiz: List[Dict] = []
        self.quiz_history: Set[str] = set()

        # Статистика прохождения
        self.user_score: int = 0
        self.total_questions_answered: int = 0

        logger.info("OrchestratorAgent initialized successfully")

    def start_new_session(self, note_text: str) -> Dict[str, Any]:
        """
        Запуск нового сценария: анализ заметки → проверка фактов → генерация квиза.

        Workflow:
        1. Сброс предыдущего состояния
        2. Парсинг текста заметки (ParserAgent)
        3. Проверка фактов (FactCheckAgent, если включен)
        4. Генерация вопросов (QuizAgent)
        5. Обновление истории

        Args:
            note_text: Текст учебной заметки студента

        Returns:
            Dict с ключами:
                - status: "success" | "error"
                - quiz: List[Dict] - сгенерированные вопросы
                - concepts_count: int - количество извлеченных концептов
                - message: str - информационное сообщение
        """
        try:
            logger.info("Starting new session")

            # Шаг 1: Сброс состояния
            self._reset_session()
            self.current_note_text = note_text
            self.current_note_hash = compute_hash(note_text)

            logger.info(f"Note hash: {self.current_note_hash[:16]}...")

            # Шаг 2: Извлечение концептов (с кэшем)
            logger.info("Step 1/3: Parsing concepts from note")
            self.extracted_concepts = self.parser.parse_note(note_text)

            if not self.extracted_concepts:
                return {
                    "status": "error",
                    "message": "Не удалось извлечь концепты из заметки. Возможно, текст слишком короткий или содержит мало образовательной информации."
                }

            logger.info(f"Extracted {len(self.extracted_concepts)} concepts")

            # Шаг 3: Проверка фактов (опционально)
            if self.factcheck_enabled:
                logger.info("Step 2/3: Fact-checking concepts")
                self.verified_concepts = self.fact_checker.verify_concepts(
                    self.extracted_concepts
                )
                logger.info(f"Verified {len(self.verified_concepts)} concepts")
            else:
                logger.info("Fact-check disabled, skipping")
                self.verified_concepts = self.extracted_concepts

            # Шаг 4: Генерация вопросов
            logger.info("Step 3/3: Generating quiz questions")
            self.current_quiz = self.quiz_generator.generate_questions(
                concepts=self.verified_concepts,
                avoid_history=self.quiz_history
            )

            if not self.current_quiz:
                return {
                    "status": "error",
                    "message": "Не удалось сгенерировать вопросы. Попробуйте другую заметку."
                }

            # Шаг 5: Обновление истории
            self._update_history(self.current_quiz)

            logger.info(f"Quiz generated successfully: {len(self.current_quiz)} questions")

            return {
                "status": "success",
                "quiz": self.current_quiz,
                "concepts_count": len(self.verified_concepts),
                "message": f"Квиз успешно создан! Найдено концептов: {len(self.verified_concepts)}, вопросов: {len(self.current_quiz)}"
            }

        except Exception as e:
            logger.error(f"Error in start_new_session: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "message": f"Произошла ошибка: {str(e)}"
            }

    def regenerate_quiz(self) -> Dict[str, Any]:
        """
        Регенерация квиза по тем же концептам, но с новыми вопросами.
        Использует уже извлеченные и проверенные концепты (экономия токенов).

        Returns:
            Dict с тем же форматом, что и start_new_session
        """
        try:
            logger.info("Regenerating quiz with new questions")

            if not self.verified_concepts:
                return {
                    "status": "error",
                    "message": "Нет сохраненных концептов. Сначала вызовите start_new_session()."
                }

            # Генерация новых вопросов с учетом истории
            self.current_quiz = self.quiz_generator.generate_questions(
                concepts=self.verified_concepts,
                avoid_history=self.quiz_history
            )

            if not self.current_quiz:
                # История может быть переполнена - сбрасываем
                logger.warning("Cannot generate new questions, clearing history")
                self.quiz_history.clear()

                self.current_quiz = self.quiz_generator.generate_questions(
                    concepts=self.verified_concepts,
                    avoid_history=set()
                )

            # Обновление истории новыми вопросами
            self._update_history(self.current_quiz)

            # Сброс счетчика для нового прохождения
            self.user_score = 0
            self.total_questions_answered = 0

            logger.info(f"Quiz regenerated: {len(self.current_quiz)} new questions")

            return {
                "status": "success",
                "quiz": self.current_quiz,
                "concepts_count": len(self.verified_concepts),
                "message": f"Квиз обновлен! Новых вопросов: {len(self.current_quiz)}"
            }

        except Exception as e:
            logger.error(f"Error in regenerate_quiz: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "message": f"Ошибка при регенерации: {str(e)}"
            }

    def submit_answer(self, question_id: str, user_answer: str) -> Dict[str, Any]:
        """
        Проверка ответа пользователя на вопрос.
        При ошибке вызывает ExplainAgent для генерации подсказки.

        Args:
            question_id: Уникальный идентификатор вопроса
            user_answer: Ответ пользователя (индекс варианта или "true"/"false")

        Returns:
            Dict с ключами:
                - status: "correct" | "incorrect" | "error"
                - is_correct: bool
                - correct_answer: str
                - explanation: Optional[str] - объяснение ошибки
                - memory_palace: Optional[str] - мнемонический образ
                - score: int - текущий счет
                - progress: str - прогресс (например, "3/5")
        """
        try:
            # Поиск вопроса в текущем квизе
            question = self._find_question_by_id(question_id)

            if not question:
                return {
                    "status": "error",
                    "message": f"Вопрос с ID {question_id} не найден"
                }

            # Извлечение правильного ответа
            correct_answer = question.get("correct_answer")
            question_text = question.get("question", "")
            concept_definition = question.get("concept_definition", "")

            # Проверка правильности
            is_correct = str(user_answer).lower().strip() == str(correct_answer).lower().strip()

            # Обновление статистики
            self.total_questions_answered += 1
            if is_correct:
                self.user_score += 1

            result = {
                "status": "correct" if is_correct else "incorrect",
                "is_correct": is_correct,
                "correct_answer": correct_answer,
                "score": self.user_score,
                "progress": f"{self.total_questions_answered}/{len(self.current_quiz)}"
            }

            # Генерация объяснения при ошибке
            if not is_correct:
                logger.info(f"Wrong answer for question {question_id}, generating explanation")

                explanation_data = self.explainer.explain_error(
                    question_text=question_text,
                    user_ans=user_answer,
                    correct_ans=correct_answer,
                    concept_def=concept_definition
                )

                result["explanation"] = explanation_data.get("explanation_text", "")
                result["memory_palace"] = explanation_data.get("memory_palace_image", "")

            logger.info(f"Answer processed: {'correct' if is_correct else 'incorrect'}")

            return result

        except Exception as e:
            logger.error(f"Error in submit_answer: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "message": f"Ошибка при проверке ответа: {str(e)}"
            }

    def get_session_stats(self) -> Dict[str, Any]:
        """
        Получение статистики текущей сессии.

        Returns:
            Dict с ключами:
                - score: int - правильных ответов
                - total: int - всего вопросов отвечено
                - accuracy: float - процент правильных ответов
                - concepts_extracted: int
                - questions_generated: int
                - llm_stats: Dict - статистика использования токенов
        """
        accuracy = 0.0
        if self.total_questions_answered > 0:
            accuracy = round((self.user_score / self.total_questions_answered) * 100, 2)

        return {
            "score": self.user_score,
            "total": self.total_questions_answered,
            "accuracy": accuracy,
            "concepts_extracted": len(self.extracted_concepts),
            "questions_generated": len(self.current_quiz),
            "questions_in_history": len(self.quiz_history),
            "llm_stats": self.client.get_usage_stats()
        }

    def _update_history(self, new_questions: List[Dict]) -> None:
        """
        Добавление новых вопросов в историю сессии.
        Хеширует текст вопроса для предотвращения дубликатов.

        Args:
            new_questions: Список новых вопросов для добавления в историю
        """
        for question in new_questions:
            question_hash = compute_hash(question.get("question", ""))
            self.quiz_history.add(question_hash)

        logger.debug(f"History updated: {len(self.quiz_history)} unique questions")

    def _find_question_by_id(self, question_id: str) -> Optional[Dict]:
        """
        Поиск вопроса в текущем квизе по его ID.

        Args:
            question_id: Уникальный идентификатор вопроса

        Returns:
            Dict с данными вопроса или None
        """
        for question in self.current_quiz:
            if question.get("question_id") == question_id:
                return question
        return None

    def _reset_session(self) -> None:
        """
        Полный сброс состояния сессии.
        Вызывается при старте новой сессии.
        """
        self.current_note_hash = ""
        self.current_note_text = ""
        self.extracted_concepts = []
        self.verified_concepts = []
        self.current_quiz = []
        self.quiz_history.clear()
        self.user_score = 0
        self.total_questions_answered = 0

        logger.debug("Session state reset")