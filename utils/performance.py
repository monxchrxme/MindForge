import time
import logging
from typing import Dict, List, Callable
from functools import wraps
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class PerformanceMetrics:
    """Сборщик метрик производительности"""

    def __init__(self):
        self.agent_timings: Dict[str, List[float]] = {}
        self.total_start_time: float = None
        self.total_end_time: float = None

    def start_session(self):
        """Запуск сессии измерения"""
        self.total_start_time = time.time()
        logger.info("[METRICS] Session timing started")

    def end_session(self):
        """Завершение сессии измерения"""
        self.total_end_time = time.time()
        logger.info("[METRICS] Session timing ended")

    def add_agent_timing(self, agent_name: str, duration_ms: float):
        """Добавить время работы агента"""
        if agent_name not in self.agent_timings:
            self.agent_timings[agent_name] = []
        self.agent_timings[agent_name].append(duration_ms)

    def get_total_time_ms(self) -> float:
        """Получить общее время работы системы в мс"""
        if self.total_start_time and self.total_end_time:
            return (self.total_end_time - self.total_start_time) * 1000
        return 0.0

    def get_agent_stats(self, agent_name: str) -> Dict[str, float]:
        """Получить статистику по агенту"""
        if agent_name not in self.agent_timings:
            return {"count": 0, "total_ms": 0, "avg_ms": 0, "min_ms": 0, "max_ms": 0}

        timings = self.agent_timings[agent_name]
        return {
            "count": len(timings),
            "total_ms": sum(timings),
            "avg_ms": sum(timings) / len(timings),
            "min_ms": min(timings),
            "max_ms": max(timings)
        }

    def print_summary(self, token_count: int = None):
        """Вывести итоговую статистику"""
        print("\n" + "=" * 60)
        print("📊 МЕТРИКИ ПРОИЗВОДИТЕЛЬНОСТИ")
        print("=" * 60)

        # Общее время системы
        total_ms = self.get_total_time_ms()
        print(f"\n⏱️  Общее время работы системы: {total_ms:.2f} мс ({total_ms / 1000:.2f} сек)")

        # Статистика по агентам
        if self.agent_timings:
            print(f"\n🤖 Время работы агентов:\n")
            for agent_name in sorted(self.agent_timings.keys()):
                stats = self.get_agent_stats(agent_name)
                print(f"  {agent_name}:")
                print(f"    • Вызовов: {stats['count']}")
                print(f"    • Общее время: {stats['total_ms']:.2f} мс")
                print(f"    • Среднее: {stats['avg_ms']:.2f} мс")
                print(f"    • Мин/Макс: {stats['min_ms']:.2f} / {stats['max_ms']:.2f} мс")
                print()

        # Токены (если передано)
        if token_count is not None:
            print(f"🔢 Токены использовано: {token_count:,}")
            if total_ms > 0:
                tokens_per_sec = (token_count / (total_ms / 1000))
                print(f"   Скорость: {tokens_per_sec:.1f} токенов/сек")

        print("=" * 60 + "\n")

        # Также пишем в лог
        logger.info(f"[METRICS] Total system time: {total_ms:.2f} ms")
        for agent_name, stats in [(name, self.get_agent_stats(name))
                                  for name in self.agent_timings.keys()]:
            logger.info(f"[METRICS] {agent_name}: {stats['total_ms']:.2f} ms "
                        f"(avg: {stats['avg_ms']:.2f} ms, calls: {stats['count']})")


# Глобальный экземпляр
_metrics = PerformanceMetrics()


def get_metrics() -> PerformanceMetrics:
    """Получить глобальный экземпляр метрик"""
    return _metrics


def measure_time(agent_name: str):
    """
    Декоратор для замера времени выполнения метода агента

    Пример использования:
    @measure_time("ParserAgent")
    def parse_note(self, text):
        ...
    """

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration_ms = (time.time() - start_time) * 1000
                _metrics.add_agent_timing(agent_name, duration_ms)
                logger.info(f"[TIMING] {agent_name}.{func.__name__} took {duration_ms:.2f} ms")

        return wrapper

    return decorator


@contextmanager
def measure_block(block_name: str):
    """
    Контекстный менеджер для замера времени блока кода

    Пример использования:
    with measure_block("Quiz generation"):
        questions = agent.generate(...)
    """
    start_time = time.time()
    logger.info(f"[TIMING] {block_name} started")
    try:
        yield
    finally:
        duration_ms = (time.time() - start_time) * 1000
        logger.info(f"[TIMING] {block_name} completed in {duration_ms:.2f} ms")
