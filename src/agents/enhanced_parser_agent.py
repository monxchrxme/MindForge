"""
Enhanced Parser Agent - с RAG через GigaChat Embeddings
Добавлен векторный поиск и обогащение контекста
"""

from typing import Dict, Any, List, Optional
import logging
from langchain_gigachat import GigaChat, GigaChatEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import VectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_core.documents import Document
import requests
from bs4 import BeautifulSoup

from ..langgraph.state_schema import GraphState, ConceptSchema

logger = logging.getLogger(__name__)


class EnhancedParserAgent:
    """
    Улучшенный Parser Agent с RAG и веб-поиском
    Сохраняет оригинальную логику, добавляет RAG для длинных текстов
    """

    def __init__(
            self,
            credentials: str,
            use_rag: bool = True,
            enable_web_search: bool = False
    ):
        """
        Args:
            credentials: GigaChat API ключ
            use_rag: использовать RAG для длинных текстов
            enable_web_search: включить веб-поиск для обогащения
        """
        logger.info("Инициализация Enhanced Parser Agent...")

        # LLM для парсинга (оригинальная логика)
        self.llm = GigaChat(
            credentials=credentials,
            verify_ssl_certs=False,
            temperature=0.2,  # детерминированный вывод
            model="GigaChat"
        )

        # Embeddings для RAG
        self.embeddings = GigaChatEmbeddings(
            credentials=credentials,
            verify_ssl_certs=False
        )

        self.use_rag = use_rag
        self.enable_web_search = enable_web_search
        self.vector_store: Optional[VectorStore] = None

        # Text splitter для RAG
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        # Оригинальный промпт (БЕЗ ИЗМЕНЕНИЙ!)
        self.system_prompt = ChatPromptTemplate.from_messages([
            ("system", """Ты — эксперт-аналитик образовательного контента с многолетним опытом работы в академической среде.
Твоя основная задача: тщательно проанализировать предоставленный текст лекции и извлечь наиболее значимые ключевые факты, которые студент должен запомнить для глубокого понимания темы.

КРИТЕРИИ ОТБОРА ФАКТОВ:
1. Полнота и самодостаточность — каждый факт представляет собой законченное, самостоятельное утверждение, которое можно понять вне контекста лекции
2. Проверяемость и конкретность — факты должны содержать конкретные определения, формулы, даты, названия, числовые данные или четкие логические связи
3. Приоритет важности — отбирай только концептуально значимую информацию: определения ключевых терминов, основные теоремы, законы, формулы, исторические вехи, причинно-следственные связи
4. Точность формулировок — сохраняй терминологию и научную строгость из оригинального текста
5. Избегай повторов — каждый факт должен нести уникальную информацию

КОЛИЧЕСТВО: извлеки от 5 до 8 ключевых фактов в зависимости от насыщенности материала.

ФОРМАТ ВЫВОДА:
- Каждый факт начинается с дефиса "-" и новой строки
- Факт формулируется как одно полное предложение (допустимо 2-3 предложения, если информация сложная)
- НЕ добавляй нумерацию, заголовки, комментарии или метаинформацию
- НЕ пиши вводных фраз типа "Вот факты:" или "Список фактов:"

{rag_context}

Анализируй лекцию внимательно, выделяя фундаментальные знания, которые формируют понимание темы."""),
            ("user", "Текст лекции:\n\n{lecture_text}\n\nИзвлеки ключевые факты согласно заданным критериям:")
        ])

        logger.info("Enhanced Parser Agent инициализирован")

    def _build_vector_store(self, text: str) -> VectorStore:
        """Построение векторного хранилища из текста"""
        try:
            logger.info("Создание векторного хранилища...")

            # Разбиваем текст на чанки
            chunks = self.text_splitter.split_text(text)
            documents = [Document(page_content=chunk) for chunk in chunks]

            logger.info(f"Создано {len(documents)} чанков")

            # Создаем FAISS векторное хранилище
            vector_store = FAISS.from_documents(
                documents=documents,
                embedding=self.embeddings
            )

            logger.info("Векторное хранилище создано успешно")
            return vector_store

        except Exception as e:
            logger.error(f"Ошибка создания векторного хранилища: {e}")
            return None

    def _retrieve_relevant_context(
            self,
            query: str,
            k: int = 3
    ) -> List[str]:
        """
        Поиск релевантного контекста через RAG

        Args:
            query: запрос для поиска
            k: количество релевантных чанков
        """
        if not self.vector_store:
            return []

        try:
            logger.info(f"Поиск релевантного контекста (k={k})...")

            # Векторный поиск
            docs_with_scores = self.vector_store.similarity_search_with_score(
                query=query,
                k=k
            )

            # Извлекаем содержимое
            contexts = [doc.page_content for doc, score in docs_with_scores]

            logger.info(f"Найдено {len(contexts)} релевантных чанков")
            return contexts

        except Exception as e:
            logger.error(f"Ошибка RAG поиска: {e}")
            return []

    def _web_search_enhancement(self, topic: str) -> List[str]:
        """
        Опциональный веб-поиск для обогащения контекста

        Args:
            topic: тема для поиска
        """
        if not self.enable_web_search:
            return []

        try:
            logger.info(f"Веб-поиск по теме: {topic}")

            # Простой поиск через DuckDuckGo HTML (без API ключа)
            search_url = f"https://html.duckduckgo.com/html/?q={topic}"
            headers = {"User-Agent": "Mozilla/5.0"}

            response = requests.get(search_url, headers=headers, timeout=5)
            soup = BeautifulSoup(response.text, 'html.parser')

            # Извлекаем сниппеты результатов
            results = []
            for result in soup.find_all('a', class_='result__snippet')[:3]:
                results.append(result.get_text(strip=True))

            logger.info(f"Найдено {len(results)} результатов из веба")
            return results

        except Exception as e:
            logger.warning(f"Веб-поиск не удался: {e}")
            return []

    def process(self, state: GraphState) -> GraphState:
        """
        Основной метод обработки (СОХРАНЕНА ОРИГИНАЛЬНАЯ ЛОГИКА)

        Args:
            state: текущее состояние графа

        Returns:
            обновленное состояние с извлеченными фактами
        """
        logger.info("=" * 50)
        logger.info("Parser Agent: начало обработки")
        logger.info("=" * 50)

        try:
            lecture_text = state["lecture_text"]

            # RAG для длинных текстов (> 2000 символов)
            rag_context_str = ""
            if self.use_rag and len(lecture_text) > 2000:
                logger.info("Текст длинный - активация RAG")

                # Строим векторное хранилище
                self.vector_store = self._build_vector_store(lecture_text)

                # Извлекаем главную тему для RAG
                topic_query = lecture_text[:500]  # первые 500 символов как запрос
                rag_contexts = self._retrieve_relevant_context(topic_query, k=3)

                if rag_contexts:
                    rag_context_str = "\n\nДОПОЛНИТЕЛЬНЫЙ КОНТЕКСТ ИЗ RAG:\n" + "\n---\n".join(rag_contexts)
                    state["rag_context"] = rag_contexts

            # Оригинальная chain (БЕЗ ИЗМЕНЕНИЙ)
            chain = self.system_prompt | self.llm

            # Вызов GigaChat API
            response = chain.invoke({
                "lecture_text": lecture_text,
                "rag_context": rag_context_str
            })

            facts_text = response.content

            # Оригинальный парсинг фактов (БЕЗ ИЗМЕНЕНИЙ)
            facts = [
                line.strip().lstrip("- ").lstrip("• ")
                for line in facts_text.split("\n")
                if line.strip() and (line.strip().startswith("-") or line.strip().startswith("•"))
            ]

            # Конвертация фактов в концепции (для Quiz Agent)
            concepts = self._facts_to_concepts(facts)

            # Обновление состояния
            state["key_facts"] = facts
            state["concepts"] = [c.dict() for c in concepts]  # конвертируем в dict
            state["messages"].append(f"Parser: извлечено {len(facts)} фактов")

            logger.info(f"✓ Извлечено {len(facts)} ключевых фактов")
            logger.info(f"✓ Создано {len(concepts)} концепций для Quiz Agent")

            return state

        except Exception as e:
            logger.error(f"✗ Ошибка в Parser Agent: {e}")
            state["error"] = str(e)
            state["messages"].append(f"Parser: ОШИБКА - {e}")
            return state

    def _facts_to_concepts(self, facts: List[str]) -> List[ConceptSchema]:
        """
        Конвертация фактов в концепции для Quiz Agent

        Args:
            facts: список извлеченных фактов

        Returns:
            список концепций в едином формате
        """
        concepts = []

        for i, fact in enumerate(facts):
            # Простая эвристика для определения важности
            importance = "high" if i < 3 else "medium" if i < 6 else "low"

            concept = ConceptSchema(
                title=f"Концепция {i + 1}",  # можно улучшить через LLM
                description=fact,
                importance=importance,
                context=fact  # используем сам факт как контекст
            )
            concepts.append(concept)

        return concepts
