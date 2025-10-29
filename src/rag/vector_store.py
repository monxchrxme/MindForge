"""
Векторное хранилище FAISS для RAG
"""

from langchain_community.vectorstores import FAISS
from typing import List
import logging

logger = logging.getLogger(__name__)


class VectorStore:
    """Обёртка для FAISS векторного хранилища"""

    def __init__(self, embeddings):
        """
        Args:
            embeddings: GigaChatEmbeddings instance
        """
        self.embeddings = embeddings
        self.vectorstore = None
        logger.info("VectorStore инициализирован")

    def create_from_texts(self, texts: List[str]):
        """
        Создание векторного хранилища из текстов

        Args:
            texts: список текстовых чанков
        """
        logger.info(f"Создание FAISS индекса для {len(texts)} чанков...")
        self.vectorstore = FAISS.from_texts(texts, self.embeddings)
        logger.info("✓ FAISS индекс создан")

    def similarity_search(self, query: str, k: int = 5) -> List[any]:
        """
        Семантический поиск релевантных документов

        Args:
            query: поисковый запрос
            k: количество результатов

        Returns:
            список релевантных документов
        """
        if not self.vectorstore:
            raise ValueError("Векторное хранилище не создано")

        logger.debug(f"Поиск {k} релевантных фрагментов...")
        results = self.vectorstore.similarity_search(query, k=k)
        logger.debug(f"✓ Найдено {len(results)} фрагментов")

        return results
