# tests/experiment_tracker.py
"""
Отслеживание времени экспериментов.
"""

from datetime import datetime
from typing import Optional


def format_duration(seconds: float) -> str:
    """
    Форматирует длительность в секундах в человекочитаемый формат.
    
    Параметры
    ----------
    seconds : float
        Длительность в секундах
    
    Возвращает
    ----------
    str
        Форматированная строка вида "1ч 2м 3с"
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours}ч {minutes}м {secs}с"


class ExperimentTracker:
    """Трекер времени эксперимента."""
    
    def __init__(self, experiment_id: str):
        self.experiment_id = experiment_id
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.duration_seconds: float = 0.0
        self.duration_formatted: str = ""
    
    def start(self):
        """Записывает время начала эксперимента."""
        self.start_time = datetime.now()
    
    def stop(self):
        """Записывает время окончания и вычисляет длительность."""
        self.end_time = datetime.now()
        self.duration_seconds = (self.end_time - self.start_time).total_seconds()
        self.duration_formatted = format_duration(self.duration_seconds)
    
    def get_report(self) -> dict:
        """Возвращает отчёт о времени эксперимента."""
        return {
            "experiment_id": self.experiment_id,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.duration_seconds,
            "duration_formatted": self.duration_formatted,
        }