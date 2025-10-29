"""
Parser Agent с RAG и веб-поиском
"""

from typing import List, Dict, Any
import logging
from langchain_core.messages import HumanMessage

from .base_agent import BaseAgent
from ..langgraph.state_schema import GraphState, ConceptSchema
from ..utils.gigachat_client import create_gigachat_parser_client, create_gigachat_embeddings
from ..rag.chunker import TextChunker
from ..rag.vector_store import VectorStore
from ..search.web_search import WebSearchService

logger = logging.getLogger(__name__)


class ParserAgent(BaseAgent):
    """Parser Agent с RAG и веб-поиском для проверки фактов"""

    def __init__(
        self,
        gigachat_credentials: str,
        use_rag: bool = True,
        enable_web_search: bool = False
    ):
        """
        Инициализация Parser Agent

        Args:
            gigachat_credentials: GigaChat credentials
            use_rag: использовать RAG для длинных текстов
            enable_web_search: проверять факты через веб-поиск
        """
        super().__init__("ParserAgent")

        self.gigachat_credentials = gigachat_credentials
        self.use_rag = use_rag
        self.enable_web_search = enable_web_search

        # GigaChat клиент с подсчетом токенов
        self.llm = create_gigachat_parser_client()

        # RAG компоненты
        if self.use_rag:
            self.chunker = TextChunker(chunk_size=500, chunk_overlap=100)
            self.embeddings = create_gigachat_embeddings()
            self.vector_store = VectorStore(self.embeddings)
            logger.info("✓ RAG инициализирован")

        # Web Search - ИСПРАВЛЕНО: убран аргумент search_provider
        if self.enable_web_search:
            self.web_search = WebSearchService()  # БЕЗ аргументов!
            logger.info("✓ WebSearch инициализирован")

        logger.info(
            f"ParserAgent готов: "
            f"RAG={'ON' if use_rag else 'OFF'}, "
            f"WebSearch={'ON' if enable_web_search else 'OFF'}"
        )

    def process(self, state: GraphState) -> GraphState:
        """
        Обработка состояния через LangGraph

        Args:
            state: GraphState с lecture_text

        Returns:
            обновлённый GraphState с key_facts и concepts
        """
        logger.info("="*50)
        logger.info("Parser Agent: начало работы")
        logger.info("="*50)

        try:
            lecture_text = state.get("lecture_text", "")

            if not lecture_text:
                raise ValueError("lecture_text отсутствует в state")

            logger.info(f"📝 Текст лекции: {len(lecture_text)} символов")

            # Извлечение фактов (с RAG или без)
            if self.use_rag and len(lecture_text) > 1000:
                logger.info("🔍 Режим: RAG (текст > 1000 символов)")
                facts = self._extract_facts_with_rag(lecture_text)
            else:
                logger.info("📄 Режим: прямая обработка")
                facts = self._extract_facts_direct(lecture_text)

            logger.info(f"✓ Извлечено {len(facts)} фактов")

            # Проверка через веб-поиск
            if self.enable_web_search and facts:
                logger.info("🌐 Проверка фактов в интернете...")
                verification = self.web_search.verify_facts(facts, max_results=2)

                verified_facts = verification['verified_facts']
                logger.info(
                    f"✓ Проверка завершена: "
                    f"{len(verified_facts)}/{len(facts)} фактов подтверждены"
                )

                facts = verified_facts

            # Конвертация в концепции
            concepts = self._facts_to_concepts(facts)

            # Обновление состояния
            state["key_facts"] = facts
            state["concepts"] = [c.dict() for c in concepts]
            state["messages"].append(f"Parser: извлечено {len(facts)} фактов")

            logger.info(f"✓ Создано {len(concepts)} концепций для QuizAgent")

            return state

        except Exception as e:
            logger.error(f"✗ Ошибка в ParserAgent: {e}")
            import traceback
            traceback.print_exc()

            state["error"] = str(e)
            state["messages"].append(f"Parser: ОШИБКА - {e}")
            return state

    def _extract_facts_with_rag(self, lecture_text: str) -> List[str]:
        """Извлечение фактов с использованием RAG"""
        logger.info("🔍 RAG: Шаг 1/4 - Разбиение текста на чанки")
        chunks = self.chunker.split(lecture_text)
        logger.info(f"   Создано {len(chunks)} чанков")

        logger.info("🔍 RAG: Шаг 2/4 - Создание векторного индекса FAISS")
        self.vector_store.create_from_texts(chunks)

        logger.info("🔍 RAG: Шаг 3/4 - Семантический поиск релевантных фрагментов")
        query = "Основные концепции, определения, формулы и ключевые факты"
        relevant_docs = self.vector_store.similarity_search(query, k=min(5, len(chunks)))

        logger.info(f"   Найдено {len(relevant_docs)} релевантных фрагментов")

        # Объединяем релевантные чанки в единый контекст
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        logger.info(f"   Объединённый контекст: {len(context)} символов")

        logger.info("🔍 RAG: Шаг 4/4 - Извлечение фактов через LLM")
        facts = self._extract_facts_from_context(context)

        logger.info(f"✓ RAG завершён: извлечено {len(facts)} фактов")
        return facts

    def _extract_facts_direct(self, lecture_text: str) -> List[str]:
        """Прямое извлечение фактов без RAG"""
        logger.info("Прямая обработка: использование первых 3000 символов")
        return self._extract_facts_from_context(lecture_text[:3000])

    def _extract_facts_from_context(self, context: str) -> List[str]:
        """Извлечение фактов из контекста через LLM"""
        prompt = f"""Проанализируй текст лекции и извлеки 5-10 ключевых фактов.

Текст:
{context}

Требования:
1. Каждый факт должен быть самодостаточным
2. Четкая и краткая формулировка
3. Приоритет: определения, формулы, ключевые идеи
4. БЕЗ LaTeX формул - используй обычный текст

Формат: список, каждый факт с новой строки, начинается с "- "
"""

        message = HumanMessage(content=prompt)
        response = self.llm.chat(message)

        # Парсинг фактов из ответа
        facts = []
        for line in response.content.split('\n'):
            line = line.strip()
            if line.startswith('-') or line.startswith('•') or line.startswith('*'):
                fact = line.lstrip('-•* ').strip()
                if fact and len(fact) > 10:
                    facts.append(fact)

        # Если парсинг не сработал, берём непустые строки
        if not facts:
            facts = [
                line.strip()
                for line in response.content.split('\n')
                if line.strip() and len(line.strip()) > 10
            ][:10]

        logger.info(f"   Извлечено {len(facts)} фактов")
        return facts[:10]

    def _facts_to_concepts(self, facts: List[str]) -> List[ConceptSchema]:
        """
        Конвертация фактов в концепции для QuizAgent

        Args:
            facts: список извлечённых фактов

        Returns:
            список ConceptSchema объектов
        """
        concepts = []

        for i, fact in enumerate(facts):
            # Определяем важность: первые 3 - high, 4-6 - medium, остальные - low
            if i < 3:
                importance = "high"
            elif i < 6:
                importance = "medium"
            else:
                importance = "low"

            # Извлекаем заголовок (до первой точки или первые 50 символов)
            title = fact.split('.')[0][:50] if '.' in fact else fact[:50]

            concept = ConceptSchema(
                title=title,
                description=fact,
                importance=importance,
                context=fact
            )
            concepts.append(concept)

        return concepts
