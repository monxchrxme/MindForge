"""
Веб-поиск через Tavily API (LangChain)
"""

import os
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


class WebSearchService:
    """Сервис веб-поиска через Tavily API"""

    def __init__(self):
        """Инициализация с проверкой Tavily API ключа"""
        self.tavily_api_key = os.getenv("TAVILY_API_KEY")
        self.search_tool = None
        self.mode = "mock"

        if self.tavily_api_key:
            self._init_tavily()
        else:
            logger.info("WebSearchService: режим MOCK")
            logger.info("💡 Для реального поиска: export TAVILY_API_KEY=your_key")
            logger.info("💡 Получить ключ: https://tavily.com/ (бесплатно 1000 req/month)")

    def _init_tavily(self):
        """Инициализация Tavily Search"""
        try:
            from langchain_community.tools.tavily_search import TavilySearchResults

            self.search_tool = TavilySearchResults(
                max_results=2,
                search_depth="basic",
                include_answer=False,
                include_raw_content=False,
                include_images=False
            )
            self.mode = "tavily"
            logger.info("✓ WebSearchService: режим TAVILY (реальный веб-поиск)")

        except ImportError:
            logger.warning("⚠️  tavily-python не установлен: pip install tavily-python")
            self.mode = "mock"
        except Exception as e:
            logger.warning(f"⚠️  Ошибка инициализации Tavily: {e}")
            self.mode = "mock"

    def verify_facts(self, facts: List[str], max_results: int = 2) -> Dict[str, any]:
        """
        Проверка фактов через веб-поиск

        Args:
            facts: список фактов
            max_results: максимум результатов

        Returns:
            результаты проверки
        """
        verification_results = {
            'verified_facts': [],
            'unverified_facts': [],
            'search_results': []
        }

        logger.info(f"🔍 Проверка {len(facts)} фактов ({self.mode} режим)...")

        for i, fact in enumerate(facts, 1):
            logger.info(f"   Факт {i}/{len(facts)}: {fact[:60]}...")

            try:
                if self.mode == "tavily":
                    sources = self._search_tavily(fact)
                    is_verified = len(sources) > 0
                else:
                    sources = self._mock_search(fact)
                    is_verified = True

                if is_verified:
                    verification_results['verified_facts'].append(fact)
                    if self.mode == "mock":
                        logger.info(f"      ✓ Подтверждён (mock)")
                    else:
                        logger.info(f"      ✓ Подтверждён ({len(sources)} источников)")
                else:
                    verification_results['unverified_facts'].append(fact)
                    logger.warning(f"      ✗ Не найдено подтверждений")

                verification_results['search_results'].append({
                    'fact': fact,
                    'verified': is_verified,
                    'sources': sources
                })

            except Exception as e:
                logger.error(f"      Ошибка поиска: {e}")
                # При ошибке в mock режиме подтверждаем
                if self.mode == "mock":
                    verification_results['verified_facts'].append(fact)
                else:
                    verification_results['unverified_facts'].append(fact)

        verified_count = len(verification_results['verified_facts'])
        logger.info(
            f"✓ Проверка завершена: {verified_count}/{len(facts)} "
            f"фактов подтверждены ({self.mode})"
        )

        return verification_results

    def _search_tavily(self, query: str) -> List[Dict[str, str]]:
        """Поиск через Tavily API"""
        try:
            # Tavily возвращает список словарей
            raw_results = self.search_tool.invoke({"query": query})

            results = []
            for item in raw_results:
                if isinstance(item, dict):
                    results.append({
                        'title': item.get('title', item.get('name', 'No title'))[:100],
                        'url': item.get('url', ''),
                        'snippet': item.get('content', item.get('snippet', ''))[:200]
                    })

            return results

        except Exception as e:
            logger.error(f"Ошибка Tavily API: {e}")
            return []


