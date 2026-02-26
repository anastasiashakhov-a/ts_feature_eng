# tests/results_manager.py
"""
Управление сохранением результатов экспериментов.
"""

import os
import json
import pandas as pd
from datetime import datetime
from typing import Optional
from experiment_tracker import format_duration


class ResultsManager:
    """Менеджер для сохранения результатов экспериментов."""
    
    def __init__(self, results_dir: str = "results"):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
    
    def save_metrics(
        self,
        experiment_id: str,
        horizon: int,
        metrics: dict,
        duration_seconds: Optional[float] = None
    ):
        """Сохраняет метрики в CSV."""
        metrics_file = os.path.join(self.results_dir, "metrics_history.csv")
        
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        # Форматируем длительность
        if duration_seconds is not None:
            duration_formatted = format_duration(duration_seconds)
        else:
            duration_seconds = float('nan')
            duration_formatted = ""
        
        record = {
            "timestamp": timestamp,
            "experiment_id": experiment_id,
            "horizon_hours": horizon,
            "duration_seconds": duration_seconds,
            "duration_formatted": duration_formatted,
            **{k.lower().replace(" ", "_"): v for k, v in metrics.items()}
        }
        
        file_exists = os.path.exists(metrics_file)
        
        with open(metrics_file, "a", encoding="utf-8") as f:
            if not file_exists:
                f.write(",".join(record.keys()) + "\n")
            f.write(",".join(str(v) for v in record.values()) + "\n")
    
    def save_summary(
        self,
        experiment_id: str,
        start_time: datetime,
        end_time: datetime,
        duration_seconds: float,
        horizons_tested: list,
        config: dict,
        results: list
    ):
        """Сохраняет итоговый JSON отчёт."""
        summary_file = os.path.join(
            self.results_dir,
            f"experiment_{experiment_id}_summary.json"
        )
        
        summary = {
            "experiment_id": experiment_id,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": duration_seconds,
            "duration_formatted": format_duration(duration_seconds),
            "horizons_tested": horizons_tested,
            "config": config,
            "results": results
        }
        
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
    
    def update_experiment_duration(
        self,
        experiment_id: str,
        duration_seconds: float
    ):
        """Обновляет длительность в существующем CSV."""
        metrics_file = os.path.join(self.results_dir, "metrics_history.csv")
        
        if not os.path.exists(metrics_file):
            return
        
        df = pd.read_csv(metrics_file)
        mask = df["experiment_id"] == experiment_id
        
        if mask.any():
            df.loc[mask, "duration_seconds"] = duration_seconds
            df.loc[mask, "duration_formatted"] = format_duration(duration_seconds)
            df.to_csv(metrics_file, index=False, encoding="utf-8")