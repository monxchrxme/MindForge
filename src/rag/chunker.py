"""
Разбиение текста на чанки для RAG
"""

from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
import logging

logger = logging.getLogger(__name__)


class TextChunker:
    """Разбиение текста на семантические чанки"""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        """
        Args:
            chunk_size: размер чанка в символах
            chunk_overlap: перекрытие между чанками
        """
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        """
        Создание объекта, который будет рекурсивно разбивать текст, 
        пока не будет достигнут нужный размер 
        """
        logger.info(f"TextChunker инициализирован (size={chunk_size}, overlap={chunk_overlap})")

    def split(self, text: str) -> List[str]:
        """
        Разбиение текста на чанки

        Args:
            text: исходный текст

        Returns:
            список чанков
        """
        chunks = self.splitter.split_text(text)
        logger.info(f"Текст разбит на {len(chunks)} чанков")
        return chunks
