import logging
import json
from typing import Any, Dict, List, Optional

from agents.parser import ParserAgent
from agents.factcheck import FactCheckAgent
from agents.quiz import QuizAgent
from agents.explain import ExplainAgent
from services.gigachat_client import GigaChatClient
from services.cache_manager import CacheManager
from utils.hashing import compute_hash

from enum import Enum
from dataclasses import dataclass

# Типы контента, которые мы умеем различать
class ContentType(Enum):
    THEORY = "theory"       # Обычный текст, определения, факты
    CODE = "code"           # Программный код, сниппеты
    MATH = "math"           # Формулы, теоремы
    LIST = "list"           # Списки, перечисления
    SHORT = "short"    # Короткие заметки (zettelkasten)
    GARBAGE = "garbage"
    UNKNOWN = "unknown"

@dataclass
class NoteAnalysis:
    content_type: ContentType
    summary: str
    complexity: str  # easy, medium, hard
    recommended_strategy: str # "standard", "code_practice", "direct_quiz"


logger = logging.getLogger(__name__)


class OrchestratorAgent:
    """
    Центральный координатор мульти-агентной системы.
    Управляет потоком данных между специализированными агентами.
    Логирует все входящие и исходящие данные для отладки.
    """

    def __init__(
            self,
            config: dict,
            credentials: dict,
            cache_manager: CacheManager
    ):
        """Инициализация оркестратора и всех подчиненных агентов."""
        logger.info("=" * 70)
        logger.info("ORCHESTRATOR INITIALIZATION")
        logger.info("=" * 70)

        self.config = config
        self.cache_manager = cache_manager

        # Инициализация клиента GigaChat
        llm_settings = config.get("llm_settings", {})
        logger.info(f"LLM Settings: model={llm_settings.get('model')}, temp={llm_settings.get('temperature')}")

        self.client = GigaChatClient(
            credentials=credentials,
            model=llm_settings.get("model", "GigaChat"),
            temperature=llm_settings.get("temperature", 0.7)
        )

        # Инициализация агентов
        cache_enabled = config.get("cache_enabled", True)
        logger.info(f"Initializing agents (cache_enabled={cache_enabled})...")

        self.parser = ParserAgent(
            client=self.client,
            cache_manager=cache_manager,
            cache_enabled=cache_enabled
        )

        self.fact_checker = FactCheckAgent(client=self.client)

        self.default_quiz_settings = config.get("quiz_settings", {})
        self.quiz_generator = QuizAgent(
            client=self.client,
            questions_count=self.default_quiz_settings.get("questions_count", 5),
            difficulty=self.default_quiz_settings.get("difficulty", "medium")
        )

        self.explainer = ExplainAgent(client=self.client)

        # Настройки
        self.factcheck_enabled = config.get("enable_fact_check", True)
        logger.info(f"FactCheck enabled: {self.factcheck_enabled}")

        # Состояние сессии
        self.current_note_hash: str = ""
        self.verified_concepts: List[Dict] = []
        self.current_quiz: List[Dict] = []
        self.quiz_history: List[str] = []

        # загрузка глобальной истории вопросов
        self.global_history_key = "global_quiz_history"
        if self.cache_manager.exists(self.global_history_key):
            loaded_history = self.cache_manager.load(self.global_history_key)
            # Превращаем список обратно в множество
            self.quiz_history: List[str] = list(loaded_history) if loaded_history else []
            logger.info(f"Loaded global history: {len(self.quiz_history)} questions")
        else:
            self.quiz_history: List[str] = []

        # Статистика
        self.user_score: int = 0
        self.total_questions_answered: int = 0

        logger.info("✓ OrchestratorAgent initialized successfully")
        logger.info("=" * 70)

    def process_note_pipeline(
            self,
            note_text: str,
            questions_count: int = None,
            difficulty: str = None,
            force_reparse: bool = False,
            ignore_history: bool = False
    ) -> Dict[str, Any]:
        """
        Полный пайплайн обработки заметки с детальным логированием.

        Args:
            note_text: Текст учебной заметки
            questions_count: Количество вопросов (опционально)
            difficulty: Сложность вопросов (опционально)
            force_reparse: Игнорировать кэш и выполнить полный парсинг

        Returns:
            Dict с результатом генерации квиза
        """
        logger.info("\n" + "=" * 70)
        logger.info("ORCHESTRATOR: process_note_pipeline() STARTED")
        logger.info("=" * 70)
        logger.info(f"Input parameters:")
        logger.info(f"  - note_text length: {len(note_text)} chars")
        logger.info(f"  - questions_count: {questions_count}")
        logger.info(f"  - difficulty: {difficulty}")
        logger.info(f"  - force_reparse: {force_reparse}")
        logger.info(f"  - ignore_history: {ignore_history}")

        try:
            # 1. Инициализация
            self._reset_session()
            self.current_note_hash = compute_hash(note_text)
            logger.info(f"Note hash computed: {self.current_note_hash}")

            if questions_count or difficulty:
                logger.info(f"Updating quiz settings (count={questions_count}, difficulty={difficulty})")
                self._update_quiz_settings(questions_count, difficulty)

            # 2. Анализ контента
            analysis = self._analyze_content(note_text)

            # Логирование анализа (с фиксом Enum для JSON)
            analysis_log = analysis.__dict__.copy()
            analysis_log["content_type"] = str(analysis.content_type.value)
            self._log_data_transfer("Orchestrator", "Self", analysis_log, "analysis_result")

            # 3. Фильтр мусора
            if analysis.content_type == ContentType.UNKNOWN and len(note_text) < 50:
                return {"status": "error", "message": "Текст слишком короткий."}
            elif analysis.content_type == ContentType.GARBAGE:
                return {"status": "error", "message": "Текст неинформативный."}

            # 4. Работа с кэшем концептов
            verified_cache_key = f"verified_{self.current_note_hash}"
            cached_verified = None

            if not force_reparse and self.cache_manager.exists(verified_cache_key):
                logger.info("✓ Verified cache found, loading...")
                cached_verified = self.cache_manager.load(verified_cache_key)

            # 5. Определение стратегии и данных
            # Если есть кэш и не нужен репарсинг -> используем кэш
            if cached_verified and not force_reparse: #TODO
                logger.info(f"✓ HOT START: Loaded {len(cached_verified)} concepts from cache")
                self.verified_concepts = cached_verified
                # Стратегию оставляем ту, что предложил анализатор, или дефолтную
                current_strategy = analysis.recommended_strategy

            else:
                # === ХОЛОДНЫЙ СТАРТ (Генерация с нуля) ===
                logger.info("\n" + "-" * 70)
                logger.info("COLD START: Running full analysis pipeline")
                logger.info("-" * 70)

                current_strategy = analysis.recommended_strategy
                extracted = []

                # 5.1 Выполнение стратегии (Парсинг)
                try:
                    if current_strategy == "direct_quiz":
                        logger.info("🚀 STRATEGY: Direct Quiz (skipping parser)")
                        extracted = []  # Парсинг не нужен

                    elif current_strategy == "code_practice":
                        logger.info("💻 STRATEGY: Code Practice")
                        extracted = self.parser.parse_code_note(note_text)

                    else:  # standard
                        logger.info("📚 STRATEGY: Standard Pipeline")
                        self._log_data_transfer("Orchestrator", "ParserAgent", note_text, "note_text")
                        extracted = self.parser.parse_note(note_text)

                except Exception as e:
                    logger.error(f"Parsing failed: {e}")
                    extracted = []

                # 5.2 Логика Fallback (Самокоррекция)
                # Если парсер вернул пустоту, но мы хотели парсить -> переключаемся на Direct Quiz
                if not extracted and current_strategy != "direct_quiz":
                    logger.warning(
                        f"⚠️ Strategy '{current_strategy}' returned 0 concepts. Switching to Fallback: DIRECT_QUIZ.")
                    current_strategy = "direct_quiz"

                if not extracted and current_strategy != "direct_quiz":
                    return {"status": "error", "message": "Не удалось извлечь концепты. Попробуйте другой текст."}

                # 5.3 Фактчек (только если есть что проверять)
                if extracted and self.factcheck_enabled:
                    logger.info("\n>>> CALLING FactCheckAgent.verify_concepts()")
                    self.verified_concepts = self.fact_checker.verify_concepts(extracted)
                else:
                    self.verified_concepts = extracted

                # 5.4 Сохранение в кэш (если нашли концепты)
                if self.verified_concepts:
                    logger.info(f"Saving {len(self.verified_concepts)} concepts to cache...")
                    self.cache_manager.save(verified_cache_key, self.verified_concepts)

            # === ГЕНЕРАЦИЯ КВИЗА ===
            logger.info("\n" + "-" * 70)
            logger.info("QUIZ GENERATION")
            logger.info("-" * 70)
            logger.info(f"Concepts available: {len(self.verified_concepts)}")
            logger.info(f"Quiz history size: {len(self.quiz_history)}")

            history_to_use = [] if ignore_history else self.quiz_history
            if ignore_history:
                logger.info("⚠️ IGNORING HISTORY mode enabled")

            quiz_difficulty = difficulty if difficulty else analysis.complexity

            self.quiz_generator.difficulty = quiz_difficulty
            logger.info("\n>>> CALLING QuizAgent.generate_questions()")
            self._log_data_transfer("Orchestrator", "QuizAgent", {
                "concepts": self.verified_concepts,
                "avoid_history": list(self.quiz_history)
            }, "generation_params")

            self.current_quiz = self.quiz_generator.generate_questions(
                concepts=self.verified_concepts,
                avoid_history=history_to_use,
                mode=analysis.recommended_strategy
            )

            self._log_data_transfer("QuizAgent", "Orchestrator", self.current_quiz, "generated_quiz")

            if not self.current_quiz:
                logger.error("QuizAgent returned empty quiz")
                return {
                    "status": "error",
                    "message": "Не удалось сгенерировать вопросы."
                }

            logger.info(f"✓ Received {len(self.current_quiz)} questions from QuizAgent")
            self._update_history(self.current_quiz)

            cache_status = "из кэша" if (cached_verified and not force_reparse) else "новый анализ"

            result = {
                "status": "success",
                "quiz": self.current_quiz,
                "concepts_count": len(self.verified_concepts),
                "message": f"Квиз готов! Концептов: {len(self.verified_concepts)}, "
                           f"вопросов: {len(self.current_quiz)} ({cache_status})"
            }

            logger.info("\n" + "=" * 70)
            logger.info("ORCHESTRATOR: process_note_pipeline() COMPLETED")
            logger.info(f"Result: {result['status']}")
            logger.info("=" * 70 + "\n")

            return result

        except Exception as e:
            logger.error(f"Pipeline error: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "message": f"System Error: {str(e)}"
            }

    def submit_answer(self, question_id: str, user_answer: str) -> Dict[str, Any]:
        """
        Проверка ответа пользователя с детальным логированием.

        Args:
            question_id: ID вопроса
            user_answer: Ответ пользователя

        Returns:
            Dict с результатом проверки
        """
        logger.info("\n" + "=" * 60)
        logger.info("ORCHESTRATOR: submit_answer() called")
        logger.info(f"Input: question_id={question_id}, user_answer={user_answer}")

        try:
            # Поиск вопроса
            question = self._find_question_by_id(question_id)
            if not question:
                logger.error(f"Question {question_id} not found in current quiz")
                return {
                    "status": "error",
                    "message": f"Вопрос с ID {question_id} не найден"
                }

            logger.debug(f"Found question: {question.get('question', '')[:50]}...")

            correct_answer = question.get("correct_answer")
            is_correct = str(user_answer).lower().strip() == str(correct_answer).lower().strip()

            logger.info(f"Comparison: user='{user_answer}' vs correct='{correct_answer}' => {is_correct}")

            # Обновление статистики
            self.total_questions_answered += 1
            if is_correct:
                self.user_score += 1

            logger.info(f"Score updated: {self.user_score}/{self.total_questions_answered}")

            result = {
                "status": "correct" if is_correct else "incorrect",
                "is_correct": is_correct,
                "correct_answer": correct_answer,
                "score": self.user_score,
                "total": len(self.current_quiz)
            }

            # Генерация объяснения при ошибке
            if not is_correct:
                logger.info("\n>>> Wrong answer, calling ExplainAgent")

                logger.info(">>> CALLING ExplainAgent.explain_error()")
                self._log_data_transfer("Orchestrator", "ExplainAgent", {
                    "question": question.get("question"),
                    "user_answer": user_answer,
                    "correct_answer": correct_answer
                }, "explanation_request")

                try:
                    explanation_data = self.explainer.explain_error(
                        question_text=question.get("question"),
                        user_ans=user_answer,
                        correct_ans=correct_answer
                    )

                    self._log_data_transfer("ExplainAgent", "Orchestrator", explanation_data,
                                            "explanation_response")

                    # ✅ ИСПРАВЛЕНИЕ: используем правильные ключи из ExplainAgent
                    result["explanation"] = explanation_data.get("explanation_text", "")
                    result["memory_palace"] = explanation_data.get("memory_palace_image", "")

                    logger.info(f"✓ Explanation received: {len(result['explanation'])} chars")
                    logger.info(f"✓ Memory palace received: {len(result['memory_palace'])} chars")

                except Exception as explain_error:
                    logger.error(f"ExplainAgent error: {str(explain_error)}", exc_info=True)
                    result["explanation"] = "Не удалось сгенерировать объяснение."
                    result["memory_palace"] = ""

            logger.info(f"Result: {result['status']}")
            logger.info("=" * 60 + "\n")
            return result

        except Exception as e:
            logger.error(f"Error in submit_answer: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "message": f"Ошибка при проверке ответа: {str(e)}"
            }


    def get_session_stats(self) -> Dict[str, Any]:
        """Получение статистики с логированием."""
        logger.info("ORCHESTRATOR: get_session_stats() called")

        accuracy = 0.0
        if self.total_questions_answered > 0:
            accuracy = round((self.user_score / self.total_questions_answered) * 100, 2)

        stats = {
            "score": self.user_score,
            "total_questions": len(self.current_quiz),
            "answered": self.total_questions_answered,
            "accuracy": accuracy,
            "llm_stats": self.client.get_usage_stats()
        }

        logger.info(f"Stats: score={stats['score']}, accuracy={stats['accuracy']}%")
        return stats

    def _analyze_content(self, text: str) -> NoteAnalysis:
        """
        Анализирует тип и структуру заметки с помощью LLM,
        чтобы выбрать лучшую стратегию генерации.
        """
        logger.info("🧠 ORCHESTRATOR: Analyzing note structure...")

        # Берем начало текста, чтобы не тратить токены (обычно суть в начале)
        preview_text = text[:2000]

        prompt = (
            f"Проанализируй текст заметки и определи его тип.\n"
            f"Текст (начало): {preview_text}\n\n"
            f"Возможные типы:\n"
            f"- theory: лекции, статьи, определения (стандартный текст)\n"
            f"- code: программный код, функции, классы\n"
            f"- math: математические формулы, задачи, теоремы\n"
            f"- list: просто список фактов или слов\n"
            f"- short: очень короткий текст (1-2 абзаца)\n\n"
            f"Верни JSON: {{'type': '...', 'summary': 'кратко о чем', 'complexity': 'easy/medium/hard'}}"
        )

        try:
            # Используем self.client для вызова LLM
            # ВАЖНО: Тут предполагается, что ваш client умеет generate_json.
            # Если нет, используйте просто generate и парсите.
            response = self.client.generate_json(prompt)

            c_type_str = response.get("type", "unknown").lower()
            # Маппинг строки в Enum
            try:
                c_type = ContentType(c_type_str)
            except ValueError:
                c_type = ContentType.THEORY  # Фоллбек на стандарт

            # Определяем стратегию
            strategy = "standard"
            if c_type == ContentType.CODE:
                strategy = "code_practice"
            elif c_type == ContentType.SHORT or c_type == ContentType.LIST:
                strategy = "direct_quiz"  # Пропускаем парсер, генерим сразу


            c_complexity = response.get("complexity", "medium").lower()
            if "hard" in c_complexity or "сложн" in c_complexity:
                c_complexity = "hard"
            elif "easy" in c_complexity or "легк" in c_complexity:
                c_complexity = "easy"
            else:
                c_complexity = "medium"

            logger.info(f"🧠 Analysis Result: Type={c_type.value}, Strategy={strategy}")
            return NoteAnalysis(
                content_type=c_type,
                summary=response.get("summary", ""),
                complexity=c_complexity,
                recommended_strategy=strategy
            )

        except Exception as e:
            logger.error(f"Analysis failed: {e}. Falling back to STANDARD strategy.")
            # В случае ошибки возвращаем дефолт
            return NoteAnalysis(ContentType.THEORY, "", "medium", "standard")

    def _update_quiz_settings(self, count: int, difficulty: str):
        """Обновление настроек квиза."""
        logger.info("Updating quiz generator settings:")
        if count:
            logger.info(f"  - questions_count: {self.quiz_generator.questions_count} → {count}")
            self.quiz_generator.questions_count = count
        if difficulty:
            logger.info(f"  - difficulty: {self.quiz_generator.difficulty} → {difficulty}")
            self.quiz_generator.difficulty = difficulty

    def _update_history(self, new_questions: List[Dict]):
        """Обновление истории вопросов и сохранение на диск."""
        logger.info("Updating quiz history...")
        old_size = len(self.quiz_history)

        updated = False
        for q in new_questions:
            # Используем ваше исправление (сырой текст вопроса)
            question_text = q.get("question", "").strip()

            if question_text and question_text not in self.quiz_history:
                self.quiz_history.append(question_text)
                updated = True

        new_size = len(self.quiz_history)
        logger.info(f"History updated: {old_size} → {new_size} unique questions")

        # --- ДОБАВЛЕНО 1 версия
        if updated:
            logger.info("Saving updated history to disk...")
            # CacheManager принимает сериализуемые объекты, поэтому преобразуем set в list
            self.cache_manager.save(self.global_history_key, list(self.quiz_history))
        # -----------------

    def _find_question_by_id(self, q_id: str) -> Optional[Dict]:
        """Поиск вопроса по ID."""
        for q in self.current_quiz:
            if q.get("question_id") == q_id:
                return q
        return None

    def _reset_session(self):
        """Сброс состояния сессии."""
        logger.info("Resetting session state...")
        self.current_note_hash = ""
        self.verified_concepts = []
        self.current_quiz = []
        # self.quiz_history.clear()
        self.user_score = 0
        self.total_questions_answered = 0
        logger.info("✓ Session reset complete")

    def _log_data_transfer(self, source: str, destination: str, data: Any, data_name: str):
        """
        Логирование передачи данных между компонентами.

        Args:
            source: Источник данных
            destination: Получатель данных
            data: Передаваемые данные
            data_name: Название данных
        """
        logger.info(f"\n📤 DATA TRANSFER: {source} → {destination}")
        logger.info(f"   Data type: {data_name}")

        if isinstance(data, (list, tuple)):
            logger.info(f"   Data size: {len(data)} items")
            if len(data) > 0 and len(data) <= 5:
                # default=str заставит json вызывать str() для всех неизвестных типов (включая Enum)
                logger.debug(f" Data preview: {json.dumps(data, ensure_ascii=False, indent=2, default=str)[:200]}...")

        elif isinstance(data, dict):
            logger.info(f"   Data keys: {list(data.keys())}")
            # default=str заставит json вызывать str() для всех неизвестных типов (включая Enum)
            logger.debug(f" Data preview: {json.dumps(data, ensure_ascii=False, indent=2, default=str)[:200]}...")

        elif isinstance(data, str):
            logger.info(f"   Data length: {len(data)} chars")
            logger.debug(f"   Data preview: '{data[:100]}...'")
        else:
            logger.info(f"   Data type: {type(data)}")
