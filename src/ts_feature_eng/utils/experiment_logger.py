# src/ts_feature_eng/utils/experiment_logger.py
import os
import json
import hashlib
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple, Union
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch

class ExperimentManager:
    """
    Централизованный менеджер для сохранения результатов экспериментов.
    Унифицирует хранение метрик, конфигов, JSON-отчетов и графиков.
    """
    
    # Маппинг типов экспериментов к именам папок
    EXPERIMENT_TYPES = {
        "energy": "energy",
        "morocco": "energy",
        "temperature": "temperature",
        "temp": "temperature",
        "acn": "acn",
        "ev_charging": "acn",
        "quick": "quick"
    }
    
    # Единицы измерения для разных типов экспериментов
    UNITS = {
        "energy": "МВт",
        "temperature": "°C",
        "acn": "kWh"
    }
    
    def __init__(self, experiment_type: str, results_base_dir: str = "results"):
        self.experiment_type = experiment_type.lower()
        self.results_dir = self.EXPERIMENT_TYPES.get(experiment_type.lower(), experiment_type.lower())
        self.base_dir = results_base_dir
        self.full_dir = os.path.join(results_base_dir, self.results_dir)
        
        # Определяем единицу измерения
        self.unit = self.UNITS.get(self.experiment_type, "")
        
        # Создаем директории
        os.makedirs(self.full_dir, exist_ok=True)
        os.makedirs(os.path.join(self.full_dir, "json_reports"), exist_ok=True)
        os.makedirs(os.path.join(self.full_dir, "plots"), exist_ok=True)
        
        # Пути к глобальным CSV (лежат в корне results/)
        self.global_metrics_file = os.path.join(results_base_dir, "global_metrics_history.csv")
        self.global_config_file = os.path.join(results_base_dir, "global_experiments_config.csv")
        
    def get_experiment_id(self, config: Dict) -> str:
        """Генерирует уникальный ID на основе конфига."""
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    def save_metrics(self, metrics: Dict, config: Dict, experiment_id: str, 
                     horizon: int, duration_seconds: float = 0.0):
        """Сохраняет метрики в локальный и глобальный CSV."""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        record = {
            "timestamp": timestamp,
            "experiment_type": self.experiment_type,
            "experiment_id": experiment_id,
            "horizon_days": horizon,
            "duration_seconds": round(duration_seconds, 2),
            "model_type": metrics.get("model_type", "unknown"),
            "mae": metrics.get("MAE (МВт)", metrics.get("MAE (kWh)", metrics.get("MAE (°C)", 0))),
            "rmse": metrics.get("RMSE (МВт)", metrics.get("RMSE (kWh)", metrics.get("RMSE (°C)", 0))),
            "mape_pct": metrics.get("MAPE (%)", 0),
            "r2": metrics.get("R²", 0),
            "n_features": metrics.get("n_features", 0),
            "n_train_samples": metrics.get("n_train_samples", 0),
            "n_test_samples": metrics.get("n_test_samples", 0),
        }
        
        # 1. Сохраняем в глобальный CSV (для сводной статистики по всем типам)
        self._append_to_csv(self.global_metrics_file, record)
        
        # 2. Сохраняем в локальный CSV (для детального анализа конкретного типа)
        local_metrics_file = os.path.join(self.full_dir, "metrics_history.csv")
        self._append_to_csv(local_metrics_file, record)
        
        return record
    
    def save_config(self, config: Dict, experiment_id: str, horizon: int, 
                    duration_seconds: float = 0.0):
        """Сохраняет параметры эксперимента."""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        record = {
            "timestamp": timestamp,
            "experiment_type": self.experiment_type,
            "experiment_id": experiment_id,
            "horizon_days": horizon,
            "duration_seconds": round(duration_seconds, 2),
            "n_calls": config.get("n_calls", 0),
            "n_estimators": config.get("n_estimators", 0),
            "max_depth": config.get("max_depth", 0),
            "learning_rate": config.get("learning_rate", 0),
            "train_test_split": config.get("train_test_split", 0),
            "use_lstm": config.get("use_lstm", False),
        }
        
        self._append_to_csv(self.global_config_file, record)
        local_config_file = os.path.join(self.full_dir, "experiments_config.csv")
        self._append_to_csv(local_config_file, record)
        
    def save_json_report(self, metrics: Dict, config: Dict, experiment_id: str, 
                         horizon: int, duration_seconds: float = 0.0) -> str:
        """Сохраняет полный JSON-отчет."""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        report = {
            "metadata": {
                "timestamp": timestamp,
                "experiment_type": self.experiment_type,
                "experiment_id": experiment_id,
                "horizon_days": horizon,
                "duration_seconds": duration_seconds,
                "duration_formatted": self._format_duration(duration_seconds)
            },
            "config": config,
            "metrics": metrics,
        }
        
        filename = f"report_{experiment_id}_h{horizon}d_{timestamp}.json"
        filepath = os.path.join(self.full_dir, "json_reports", filename)
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
            
        return filepath
    
    def save_forecast_comparison(self, 
                                  y_true: Union[pd.Series, np.ndarray, List],
                                  y_pred: Union[pd.Series, np.ndarray, List],
                                  timestamps: Optional[Union[pd.Series, np.ndarray, List]] = None,
                                  horizon: int = 1,
                                  model_name: str = "Model",
                                  mae: Optional[float] = None,
                                  rmse: Optional[float] = None,
                                  r2: Optional[float] = None,
                                  max_points: int = 300,
                                  figsize: Tuple[int, int] = (14, 10),
                                  save_name: Optional[str] = None) -> str:
        """
        Создаёт и сохраняет детальный график сравнения прогноза с фактическими значениями.
        
        Parameters:
        -----------
        y_true : array-like
            Фактические значения
        y_pred : array-like
            Прогнозируемые значения
        timestamps : array-like, optional
            Временные метки (если есть)
        horizon : int
            Горизонт прогнозирования в днях
        model_name : str
            Название модели
        mae, rmse, r2 : float, optional
            Метрики качества (если известны)
        max_points : int
            Максимальное количество точек для отображения (для читаемости)
        figsize : tuple
            Размер фигуры (ширина, высота)
        save_name : str, optional
            Имя файла для сохранения (без расширения)
            
        Returns:
        --------
        str : Путь к сохранённому файлу
        """
        # Конвертируем в numpy массивы
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        # Обрезаем до max_points если нужно
        if len(y_true) > max_points:
            y_true = y_true[-max_points:]
            y_pred = y_pred[-max_points:]
            if timestamps is not None:
                timestamps = timestamps[-max_points:]
        
        # Вычисляем ошибку
        error = y_pred - y_true
        
        # Если метрики не переданы, вычисляем их
        if mae is None:
            mae = np.mean(np.abs(error))
        if rmse is None:
            rmse = np.sqrt(np.mean(error ** 2))
        if r2 is None:
            ss_res = np.sum(error ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Создаём фигуру с двумя подграфиками
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, 
                                        gridspec_kw={'height_ratios': [2, 1]})
        
        # ===== ВЕРХНИЙ ГРАФИК: Факт vs Прогноз =====
        if timestamps is not None:
            # Если есть временные метки
            ax1.plot(timestamps, y_true, 'b-', linewidth=1.5, label='Фактическое значение', alpha=0.8)
            ax1.plot(timestamps, y_pred, 'r--', linewidth=1.5, label='Прогноз', alpha=0.8)
            
            # Заполняем область между линиями
            ax1.fill_between(timestamps, y_true, y_pred, 
                            where=(y_pred >= y_true), 
                            interpolate=True, alpha=0.3, color='gray',
                            label='Ошибка')
            ax1.fill_between(timestamps, y_true, y_pred, 
                            where=(y_pred < y_true), 
                            interpolate=True, alpha=0.3, color='gray')
            
            # Форматируем ось времени
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax1.xaxis.set_major_locator(mdates.MonthLocator())
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        else:
            # Если временных меток нет, используем индексы
            indices = np.arange(len(y_true))
            ax1.plot(indices, y_true, 'b-', linewidth=1.5, label='Фактическое значение', alpha=0.8)
            ax1.plot(indices, y_pred, 'r--', linewidth=1.5, label='Прогноз', alpha=0.8)
            ax1.fill_between(indices, y_true, y_pred, alpha=0.3, color='gray', label='Ошибка')
        
        ax1.set_ylabel(f'Значение ({self.unit})', fontsize=11)
        ax1.set_title(f'Сравнение прогноза с фактическими значениями (горизонт: {horizon} дн.)', 
                     fontsize=13, fontweight='bold', pad=10)
        ax1.legend(loc='upper left', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Добавляем информацию о метриках в легенду
        metrics_text = f'MAE: {mae:.2f} {self.unit}\nRMSE: {rmse:.2f} {self.unit}\nR²: {r2:.4f}'
        ax1.text(0.98, 0.02, metrics_text, transform=ax1.transAxes, 
                fontsize=10, verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # ===== НИЖНИЙ ГРАФИК: Ошибка прогноза =====
        if timestamps is not None:
            ax2.plot(timestamps, error, 'purple', linewidth=1, alpha=0.7)
            
            # Заполняем положительную ошибку (переоценка) красным
            ax2.fill_between(timestamps, 0, error, 
                            where=(error > 0), 
                            interpolate=True, alpha=0.5, color='red',
                            label='Переоценка')
            
            # Заполняем отрицательную ошибку (недооценка) зелёным
            ax2.fill_between(timestamps, 0, error, 
                            where=(error <= 0), 
                            interpolate=True, alpha=0.5, color='green',
                            label='Недооценка')
            
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax2.xaxis.set_major_locator(mdates.MonthLocator())
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        else:
            indices = np.arange(len(error))
            ax2.plot(indices, error, 'purple', linewidth=1, alpha=0.7)
            ax2.fill_between(indices, 0, error, 
                            where=(error > 0), 
                            interpolate=True, alpha=0.5, color='red',
                            label='Переоценка')
            ax2.fill_between(indices, 0, error, 
                            where=(error <= 0), 
                            interpolate=True, alpha=0.5, color='green',
                            label='Недооценка')
        
        ax2.axhline(y=0, color='black', linewidth=1.5, linestyle='-')
        ax2.set_ylabel(f'Ошибка ({self.unit})', fontsize=11)
        ax2.set_xlabel('Время', fontsize=11)
        ax2.set_title('Ошибка прогноза', fontsize=12, fontweight='bold', pad=10)
        ax2.legend(loc='upper right', fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # Добавляем статистику ошибки
        error_stats = (f'Средняя ошибка: {np.mean(error):.2f}\n'
                      f'Стд. отклонение: {np.std(error):.2f}\n'
                      f'Мин: {np.min(error):.2f}\n'
                      f'Макс: {np.max(error):.2f}')
        ax2.text(0.02, 0.98, error_stats, transform=ax2.transAxes, 
                fontsize=9, verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        plt.suptitle(f'{model_name} - Прогнозирование {self.experiment_type}', 
                    fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        # Сохраняем график
        if save_name is None:
            save_name = f"forecast_comparison_h{horizon}d"
        
        filepath = os.path.join(self.full_dir, "plots", f"{save_name}.png")
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"  ✓ График сравнения сохранён: {filepath}")
        
        return filepath
    
    def save_plot(self, fig: plt.Figure, name: str, dpi: int = 150):
        """Сохраняет график в папку эксперимента."""
        if not name.endswith(".png"):
            name += ".png"
        filepath = os.path.join(self.full_dir, name)
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        return filepath
    
    def save_summary(self, all_results: List[Dict], experiment_id: str, 
                     start_time: datetime, end_time: datetime):
        """Сохраняет итоговый summary по всем горизонтам."""
        duration = (end_time - start_time).total_seconds()
        
        summary = {
            "metadata": {
                "experiment_type": self.experiment_type,
                "experiment_id": experiment_id,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_seconds": duration,
                "duration_formatted": self._format_duration(duration)
            },
            "horizons": [
                {
                    "horizon_days": r["horizon"],
                    "best_model": r["metrics"].get("model_type", "unknown"),
                    "mae": r["metrics"].get("MAE (МВт)", r["metrics"].get("MAE (kWh)", r["metrics"].get("MAE (°C)", 0))),
                    "r2": r["metrics"]["R²"],
                    "improvement_pct": ((r["naive_mae"] - r["best_mae"]) / r["naive_mae"] * 100) if r["naive_mae"] > 0 else 0
                }
                for r in all_results
            ]
        }
        
        timestamp = end_time.strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"summary_{experiment_id}_{timestamp}.json"
        filepath = os.path.join(self.full_dir, "json_reports", filename)
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
            
        return filepath
    
    def _append_to_csv(self, filepath: str, record: Dict):
        """Добавляет запись в CSV, создавая файл при необходимости."""
        file_exists = os.path.exists(filepath)
        
        # Проверяем структуру, если файл существует
        if file_exists:
            try:
                existing_df = pd.read_csv(filepath, nrows=0)
                if set(existing_df.columns) != set(record.keys()):
                    backup = filepath + ".backup"
                    os.rename(filepath, backup)
                    file_exists = False
            except:
                backup = filepath + ".corrupt"
                try: os.rename(filepath, backup)
                except: pass
                file_exists = False
        
        df = pd.DataFrame([record])
        df.to_csv(filepath, mode='a', header=not file_exists, index=False, encoding='utf-8')
    
    def _format_duration(self, seconds: float) -> str:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours}ч {minutes}м {secs}с"