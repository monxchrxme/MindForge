import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import logging
from typing import List, Dict
import os
import json

logger = logging.getLogger(__name__)


class VectorHistoryManager:
    """
    Управление историей вопросов через векторное хранилище.
    Использует ChromaDB для семантического поиска дубликатов.
    """

    def __init__(self, persist_directory: str = "data/vector_db"):
        # БЫЛО (работает в памяти или старый синтаксис):
        # self.client = chromadb.Client(Settings(...))

        # СТАЛО (гарантированно сохраняет на диск):
        self.client = chromadb.PersistentClient(path=persist_directory)

        # Используем модель, которая понимает РУССКИЙ язык
        # Она скачается один раз при первом запуске (~500MB)
        sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="cointegrated/rubert-tiny2" #"intfloat/multilingual-e5-base" #cointegrated/rubert-tiny2 - легче
        )

        self.collection = self.client.get_or_create_collection(
            name="quiz_history",
            embedding_function=sentence_transformer_ef,  # <-- Передаем функцию
            metadata={"description": "История сгенерированных вопросов"}
        )

        self.seq_file = os.path.join(persist_directory, "sequence.json")
        self.current_seq_id = self._load_seq_id()

    def _load_seq_id(self) -> int:
        if os.path.exists(self.seq_file):
            try:
                with open(self.seq_file, 'r') as f:
                    return json.load(f).get('seq_id', 0)
            except:
                return 0
        return 0

    def _save_seq_id(self):
        # Создаем директорию если нет (для первого запуска)
        os.makedirs(os.path.dirname(self.seq_file), exist_ok=True)
        with open(self.seq_file, 'w') as f:
            json.dump({'seq_id': self.current_seq_id}, f)

    def add_questions(self, questions: List[Dict]) -> None:
        """
        Добавляет новые вопросы в векторное хранилище.

        Args:
            questions: Список вопросов с полями 'question_id', 'question'
        """
        if not questions:
            return

        ids = [q['question_id'] for q in questions]
        texts = [q['question'] for q in questions]

        # Метаданные для фильтрации (опционально)
        metadatas = []
        for q in questions:
            self.current_seq_id += 1  # Увеличиваем счетчик
            metadatas.append({
                "type": q.get('type', 'unknown'),
                "seq_id": self.current_seq_id  # <-- Сохраняем номер
            })

        self.collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas
        )
        self._save_seq_id()
        logger.info(f"Added {len(questions)} questions to vector history")

    def find_similar(self,
                     question_text: str,
                     threshold: float = 0.85,
                     limit: int = 5,
                     lookback_count: int = 100  # <-- Окно в "штуках" вопросов
                     ) -> List[str]:
        """
        Ищет семантически похожие вопросы.

        Args:
            question_text: Текст нового вопроса
            threshold: Порог схожести (0.0-1.0). Выше = строже
            limit: Максимум результатов
            lookback_count: Окно последних вопросов (штук)
        Returns:
            Список похожих вопросов

        """
        min_seq_id = max(0, self.current_seq_id - lookback_count)

        results = self.collection.query(
            query_texts=[question_text],
            n_results=limit,
            where={"seq_id": {"$gt": min_seq_id}}  # Ищем только в "свежих"
        )

        # ChromaDB возвращает distances (меньше = похожее)
        # Преобразуем в similarity score
        if results['documents']:
            similar = []
            for doc, distance in zip(results['documents'][0], results['distances'][0]):
                # Cosine distance to similarity: 1 - distance
                similarity = 1 - distance
                if similarity >= threshold:
                    similar.append(doc)
            return similar
        return []

    def get_recent_questions(self, limit: int = 15) -> List[str]:
        """
        Возвращает последние N вопросов для контекста в промпте.

        Args:
            limit: Количество последних вопросов

        Returns:
            Список текстов вопросов
        """
        # ChromaDB не гарантирует порядок, поэтому можно:
        # 1) Хранить timestamp в metadata и сортировать
        # 2) Просто вернуть случайную выборку (для "избегай этих тем")

        total = self.collection.count()
        if total == 0:
            return []

        results = self.collection.get(
            limit=min(limit, total)
        )

        return results['documents'] if results['documents'] else []
