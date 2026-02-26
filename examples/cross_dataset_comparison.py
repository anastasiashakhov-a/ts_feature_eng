# examples/cross_dataset_comparison.py

"""
Сравнительный анализ работы моделей прогнозирования на разных датасетах.

Демонстрирует:
1. Загрузку и нормализацию трёх различных датасетов
2. Прогнозирование на относительных горизонтах (1%, 5%, 10% от длины ряда)
3. Multi-horizon кривые ошибки + AUC для интегральной оценки качества
4. Нормализованные метрики (MAE/σ, MAE/range) для честного кросс-датасет сравнения
5. Детальные графики сравнения прогноза с реальностью для ЛУЧШЕЙ модели
6. Сравнение ВСЕХ моделей: Naive, RF базовая/auto FE, GB базовая/auto FE, LSTM

Оптимизировано для исследований:
- Относительные горизонты вместо абсолютных
- Нормализация метрик
- Интегральная оценка через AUC
- Детальная визуализация лучшей модели на каждом горизонте
"""

import os
import json
import warnings
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from numpy import trapezoid

from ts_feature_eng import AutoFeatureEngineer
from ts_feature_eng.utils.experiment_logger import ExperimentManager

warnings.filterwarnings('ignore', message='Provided model function fails when applied to the provided data set.')

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
    from tensorflow.keras.callbacks import EarlyStopping
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False
    warnings.warn("TensorFlow не установлен. LSTM модели будут пропущены.")

# Подавляем несущественные предупреждения
warnings.filterwarnings('ignore', message='Precision loss')
warnings.filterwarnings('ignore', message='Degrees of freedom')
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


# ============================================================================
# КОНФИГУРАЦИЯ СРАВНИТЕЛЬНОГО ЭКСПЕРИМЕНТА
# ============================================================================
COMPARISON_CONFIG = {
    # ← ОПТИМИЗАЦИЯ
    "random_state": 42,
    "train_test_split": 0.8,
    "n_calls": 2,                
    "n_initial_points": 2,
    
    # Параметры моделей 
    "n_estimators": 20,         
    "max_depth": 3,             
    "learning_rate": 0.1,
    
    # Относительные горизонты
    "relative_horizons": [0.01, 0.05, 0.10],
    
    # Нормализация метрик
    "normalize_metrics": True,
    "normalization_method": "std",
    
    # Multi-horizon AUC
    "compute_auc": True,
    "auc_max_relative_horizon": 0.10,
    "auc_n_points": 3,            
    
    # Параметры LSTM (оптимизировано)
    "lstm_timesteps": 24,
    "lstm_units": 32,
    "lstm_epochs": 5,            
    "lstm_batch_size": 32,
    "use_lstm": True,             
    
    # ← НОВОЕ: Явный список моделей для сравнения
    "models_to_compare": [
        "naive", 
        "rf_base", 
        "rf_auto_fe", 
        "gb_base", 
        "gb_auto_fe", 
        "lstm_base"  
    ],
    
    # Индикатор прогресса
    "show_progress": True,
    "progress_bar_width": 40,
    
    # Директории
    "results_base_dir": "results/comparison",
    "json_subdir": "json_reports",
    "plots_subdir": "plots",
}

# ============================================================================
# КОНФИГУРАЦИИ ДАТАСЕТОВ 
# ============================================================================
DATASET_CONFIGS = {
    # ПЕРВЫЙ ДАТАСЕТ: Энергия Марокко
    "energy": {
        "path": "data/morocco zone 1 - powerconsumption_resampled (1).csv",
        "time_col": "Datetime",
        "target_col": "consumption",
        "freq": "15min",
        "target_unit": "МВт",
        "description": "Энергопотребление Марокко, зона 1 (2017)",
    },
    # Второй: Температура
    "temperature": {
        "path": "data/daily-minimum-temperatures-in-me.csv",
        "time_col": "Date",
        "target_col": "Daily minimum temperatures",
        "freq": "D",
        "target_unit": "°C",
        "description": "Минимальные суточные температуры (1981-1990)",
    },
    # Третий: ACN Load
    "acn_load": {
        "path": "data/acn_aggregate_load_timeseries.csv",
        "time_col": "timestamp",
        "target_col": "load_smart_kw",
        "freq": "15min",
        "target_unit": "kWh",
        "description": "Нагрузка зарядки EV (ACN Dataset, 2025)",
    },
}


# ============================================================================
# УТИЛИТЫ ПРОГРЕССА
# ============================================================================

class ProgressTracker:
    """Трекер прогресса с оценкой времени выполнения."""
    
    def __init__(self, total_tasks: int, description: str = "Прогресс"):
        self.total_tasks = total_tasks
        self.completed_tasks = 0
        self.description = description
        self.start_time = time.time()
        self.task_times = []
    
    def update(self, n: int = 1):
        """Обновить прогресс на n задач."""
        self.completed_tasks += n
        current_time = time.time()
        elapsed = current_time - self.start_time
        
        if n > 0:
            self.task_times.append(elapsed / self.completed_tasks)
        
        self._display_progress(elapsed)
    
    def _display_progress(self, elapsed: float):
        """Отобразить текущий прогресс."""
        if not COMPARISON_CONFIG.get("show_progress", True):
            return
        
        percent = self.completed_tasks / self.total_tasks * 100
        bar_width = COMPARISON_CONFIG.get("progress_bar_width", 40)
        filled = int(bar_width * self.completed_tasks / self.total_tasks)
        bar = "█" * filled + "░" * (bar_width - filled)
        
        if self.completed_tasks > 0:
            avg_time_per_task = elapsed / self.completed_tasks
            remaining_tasks = self.total_tasks - self.completed_tasks
            eta_seconds = remaining_tasks * avg_time_per_task
            
            if eta_seconds < 60:
                eta_str = f"{int(eta_seconds)}с"
            elif eta_seconds < 3600:
                eta_str = f"{int(eta_seconds/60)}м {int(eta_seconds%60)}с"
            else:
                eta_str = f"{int(eta_seconds/3600)}ч {int((eta_seconds%3600)/60)}м"
        else:
            eta_str = "???"
        
        if elapsed < 60:
            elapsed_str = f"{int(elapsed)}с"
        elif elapsed < 3600:
            elapsed_str = f"{int(elapsed/60)}м {int(elapsed%60)}с"
        else:
            elapsed_str = f"{int(elapsed/3600)}ч {int((elapsed%3600)/60)}м"
        
        print(f"\r  [{bar}] {percent:5.1f}% | {self.completed_tasks}/{self.total_tasks} | "
              f"⏱ {elapsed_str} | ⏳ ETA: {eta_str}", end='', flush=True)
    
    def finish(self):
        """Завершить отслеживание прогресса."""
        self.completed_tasks = self.total_tasks
        elapsed = time.time() - self.start_time
        self._display_progress(elapsed)
        print()


def print_section_header(text: str, char: str = "=", width: int = 80):
    """Вывод заголовка раздела."""
    print(f"\n{char * width}")
    print(f" {text}")
    print(f"{char * width}")


def print_subsection(text: str, char: str = "-", width: int = 60):
    """Вывод подзаголовка."""
    print(f"\n{char * width}")
    print(f"  {text}")
    print(f"{char * width}")


# ============================================================================
# УТИЛИТЫ ДЛЯ НОРМАЛИЗАЦИИ И СРАВНЕНИЯ
# ============================================================================

def normalize_mae(mae: float, y_train: np.ndarray, method: str = "std") -> float:
    """Нормализует MAE для кросс-датасет сравнения."""
    if method == "std":
        std = np.std(y_train)
        return mae / std if std > 1e-10 else mae
    elif method == "range":
        rng = np.max(y_train) - np.min(y_train)
        return mae / rng if rng > 1e-10 else mae
    elif method == "mean":
        mean = np.mean(np.abs(y_train))
        return mae / mean if mean > 1e-10 else mae
    return mae


def compute_auc_curve(errors: List[float], horizons: List[float]) -> float:
    """Вычисляет площадь под кривой ошибки (AUC)."""
    if len(errors) < 2 or len(horizons) < 2:
        return np.nan
    
    h_norm = np.array(horizons) / np.max(horizons)
    e_norm = np.array(errors)
    auc = trapezoid(e_norm, h_norm)
    
    return auc


# ============================================================================
# ЗАГРУЗЧИКИ ДАННЫХ
# ============================================================================

def load_temperature_data(path: str) -> pd.DataFrame:
    """Загружает датасет температур."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    
    time_col = [c for c in df.columns if 'time' in c.lower() or 'date' in c.lower()][0]
    temp_col = [c for c in df.columns if 'temp' in c.lower() or 'min' in c.lower()][0]
    
    df[time_col] = pd.to_datetime(df[time_col])
    
    mask_invalid = df[temp_col].astype(str).str.contains(r'[?]', na=False)
    if mask_invalid.any():
        df = df[~mask_invalid]
    
    df[temp_col] = df[temp_col].astype(str).str.replace(',', '.', regex=True).astype(float)
    
    result = pd.DataFrame({
        "timestamp": df[time_col],
        "target": df[temp_col]
    }).set_index("timestamp").sort_index().dropna(subset=["target"])
    
    return result


def load_energy_data(path: str) -> pd.DataFrame:
    """Загружает датасет энергопотребления Марокко."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    
    time_col = [c for c in df.columns if any(kw in c.lower() for kw in ['time', 'date', 'timestamp'])][0]
    power_col = [c for c in df.columns if any(kw in c.lower() for kw in ['power', 'consum', 'load'])][0]
    
    df[time_col] = pd.to_datetime(df[time_col])
    
    result = pd.DataFrame({
        "timestamp": df[time_col],
        "target": df[power_col]
    }).set_index("timestamp").sort_index().dropna(subset=["target"])
    
    return result


def load_acn_data(path: str) -> pd.DataFrame:
    """Загружает датасет нагрузки зарядки EV."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    
    required_cols = ["timestamp", "load_smart_kw"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Отсутствуют колонки: {missing}")
    
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    result = df.set_index("timestamp").sort_index()[["load_smart_kw"]].copy()
    result.columns = ["target"]
    result = result.dropna(subset=["target"])
    
    return result


def load_dataset(name: str, config: Dict[str, str]) -> pd.DataFrame:
    """Универсальный загрузчик датасета."""
    loaders = {
        "temperature": load_temperature_data,
        "energy": load_energy_data,
        "acn_load": load_acn_data,
    }
    
    loader = loaders.get(name)
    if not loader:
        raise ValueError(f"Неизвестный датасет: {name}")
    
    print(f"  📥 Загрузка {name} из {config['path']}...")
    df = loader(config["path"])
    
    print(f"    ✓ {len(df):,} наблюдений")
    print(f"    ✓ {df.index.min()} — {df.index.max()}")
    print(f"    ✓ Частота: {config['freq']}, Единица: {config['target_unit']}")
    
    return df


# ============================================================================
# ФУНКЦИИ ПРОГНОЗИРОВАНИЯ
# ============================================================================

def prepare_lstm_data(X: pd.DataFrame, y: pd.Series, timesteps: int, 
                      train_split: float = 0.8) -> Tuple:
    """Подготовка данных для LSTM."""
    combined = pd.concat([X, y.to_frame()], axis=1).dropna()
    
    if len(combined) <= timesteps + 10:
        raise ValueError(f"Недостаточно данных для LSTM: {len(combined)} <= {timesteps + 10}")
    
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_scaled = scaler_X.fit_transform(combined[X.columns])
    y_values = combined[[y.name]].values
    y_scaled = scaler_y.fit_transform(y_values).ravel()
    
    X_lstm, y_lstm = [], []
    for i in range(timesteps, len(X_scaled)):
        X_lstm.append(X_scaled[i-timesteps:i])
        y_lstm.append(y_scaled[i])
    
    X_lstm = np.array(X_lstm)
    y_lstm = np.array(y_lstm)
    
    split_idx = int(len(X_lstm) * train_split)
    return (X_lstm[:split_idx], X_lstm[split_idx:], 
            y_lstm[:split_idx], y_lstm[split_idx:], scaler_y)


def train_lstm_model(X_train: np.ndarray, y_train: np.ndarray, 
                     X_test: np.ndarray, config: Dict, verbose: int = 0):
    """Обучение LSTM модели."""
    if not HAS_TENSORFLOW:
        return None, None
    
    try:
        timesteps = config.get("lstm_timesteps", 24)
        units = config.get("lstm_units", 32)
        epochs = config.get("lstm_epochs", 5)
        
        model = Sequential([
            Input(shape=(timesteps, X_train.shape[2])),
            LSTM(units, activation='relu'),
            Dropout(0.2),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
        
        model.fit(X_train, y_train, epochs=epochs, 
                 batch_size=config.get("lstm_batch_size", 32),
                 validation_split=0.1, callbacks=[early_stop], verbose=verbose)
        
        y_pred_scaled = model.predict(X_test, verbose=0).ravel()
        return model, y_pred_scaled
        
    except Exception as e:
        if verbose >= 1:
            print(f"    ⚠️  Ошибка LSTM: {e}")
        return None, None


def evaluate_models(df: pd.DataFrame, horizon: int, config: Dict, 
                   dataset_name: str, manager: ExperimentManager = None) -> Dict[str, Any]:
    """
    Оценка ВСЕХ моделей на заданном горизонте.
    
    Сравнивает модели из config["models_to_compare"]:
    - naive, rf_base, rf_auto_fe, gb_base, gb_auto_fe, lstm_base
    """
    target = df["target"]
    y = target.shift(-horizon)
    X = pd.DataFrame({"target": target})
    
    valid_mask = ~y.isna()
    X, y = X[valid_mask], y[valid_mask]
    
    if len(X) < 20:
        return {}
    
    split_idx = int(len(X) * config["train_test_split"])
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    results = {}
    
    # 1. Наивный прогноз
    naive_pred = y_test.shift(horizon).fillna(y_test.mean())
    results["naive"] = {
        "mae": mean_absolute_error(y_test, naive_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, naive_pred)),
        "r2": r2_score(y_test, naive_pred),
        "pred": naive_pred.values,
        "model": None
    }
    
    # 2. Auto FE + модели (RF и GB)
    try:
        engineer = AutoFeatureEngineer(
            optimize=True,
            n_calls=config["n_calls"],
            n_initial_points=config["n_initial_points"],
            apply_selection=True,
            selection_threshold=config["selection_threshold"],
            variance_threshold=config["variance_threshold"],
            shap_selection=False,
            random_state=config["random_state"],
            verbose=0
        )
        
        X_train_fe = engineer.fit_transform(X_train, y_train)
        X_test_fe = engineer.transform(X_test)
        
        # Random Forest - базовые
        rf_base = RandomForestRegressor(n_estimators=config["n_estimators"], 
                                      max_depth=config["max_depth"],
                                      random_state=config["random_state"])
        rf_base.fit(X_train.fillna(0), y_train)
        rf_base_pred = rf_base.predict(X_test.fillna(0))
        results["rf_base"] = {
            "mae": mean_absolute_error(y_test, rf_base_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, rf_base_pred)),
            "r2": r2_score(y_test, rf_base_pred),
            "pred": rf_base_pred,
            "model": rf_base
        }
        
        # Random Forest - auto FE
        rf_auto = RandomForestRegressor(n_estimators=config["n_estimators"], 
                                      max_depth=config["max_depth"],
                                      random_state=config["random_state"])
        rf_auto.fit(X_train_fe.fillna(0), y_train)
        rf_auto_pred = rf_auto.predict(X_test_fe.fillna(0))
        results["rf_auto_fe"] = {
            "mae": mean_absolute_error(y_test, rf_auto_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, rf_auto_pred)),
            "r2": r2_score(y_test, rf_auto_pred),
            "pred": rf_auto_pred,
            "model": rf_auto,
            "n_features": X_train_fe.shape[1]
        }
        
        # Gradient Boosting - базовые
        gb_base = GradientBoostingRegressor(
            n_estimators=config["n_estimators"],
            max_depth=config["max_depth"],
            learning_rate=config["learning_rate"],
            random_state=config["random_state"]
        )
        gb_base.fit(X_train.fillna(0), y_train)
        gb_base_pred = gb_base.predict(X_test.fillna(0))
        results["gb_base"] = {
            "mae": mean_absolute_error(y_test, gb_base_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, gb_base_pred)),
            "r2": r2_score(y_test, gb_base_pred),
            "pred": gb_base_pred,
            "model": gb_base
        }
        
        # Gradient Boosting - auto FE
        gb_auto = GradientBoostingRegressor(
            n_estimators=config["n_estimators"],
            max_depth=config["max_depth"],
            learning_rate=config["learning_rate"],
            random_state=config["random_state"]
        )
        gb_auto.fit(X_train_fe.fillna(0), y_train)
        gb_auto_pred = gb_auto.predict(X_test_fe.fillna(0))
        results["gb_auto_fe"] = {
            "mae": mean_absolute_error(y_test, gb_auto_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, gb_auto_pred)),
            "r2": r2_score(y_test, gb_auto_pred),
            "pred": gb_auto_pred,
            "model": gb_auto,
            "n_features": X_train_fe.shape[1]
        }
        
    except Exception as e:
        if config.get("verbose", 0) >= 1:
            print(f"    ⚠️  Auto FE ошибка: {e}")
    
    # 3. LSTM - базовый (ТОЛЬКО ЕСЛИ В СПИСКЕ МОДЕЛЕЙ)
    if "lstm_base" in config.get("models_to_compare", []) and config.get("use_lstm", False) and HAS_TENSORFLOW:
        try:
            # Используем только целевую переменную для LSTM
            X_lstm = pd.DataFrame({"target": df["target"]})
            
            X_tr, X_te, y_tr, y_te, scaler_y = prepare_lstm_data(
                X_lstm, target.shift(-horizon),
                timesteps=config.get("lstm_timesteps", 24),
                train_split=config["train_test_split"]
            )
            
            lstm_model, lstm_pred_scaled = train_lstm_model(
                X_tr, y_tr, X_te, config, verbose=0
            )
            
            if lstm_pred_scaled is not None and len(lstm_pred_scaled) > 0:
                lstm_pred = scaler_y.inverse_transform(
                    lstm_pred_scaled.reshape(-1, 1)
                ).ravel()
                
                # Проверка на валидность
                if np.any(np.isnan(lstm_pred)) or np.any(np.isinf(lstm_pred)):
                    results["lstm_base"] = {"mae": np.nan, "r2": np.nan, "pred": None}
                else:
                    # Обрезаем до длины y_test
                    min_len = min(len(lstm_pred), len(y_test))
                    lstm_pred = lstm_pred[:min_len]
                    y_test_trim = y_test.iloc[-min_len:]
                    
                    results["lstm_base"] = {
                        "mae": mean_absolute_error(y_test_trim, lstm_pred),
                        "r2": r2_score(y_test_trim, lstm_pred),
                        "pred": lstm_pred,
                        "model": lstm_model
                    }
            else:
                results["lstm_base"] = {"mae": np.nan, "r2": np.nan, "pred": None}
                
        except Exception as e:
            if config.get("verbose", 0) >= 1:
                print(f"    ⚠️  LSTM base ошибка: {e}")
            results["lstm_base"] = {"mae": np.nan, "r2": np.nan, "pred": None}
    
    elif "lstm_base" in config.get("models_to_compare", []) and not HAS_TENSORFLOW:
        results["lstm_base"] = {"mae": np.nan, "r2": np.nan, "pred": None}
    
    elif "lstm_base" in config.get("models_to_compare", []) and not config.get("use_lstm", False):
        results["lstm_base"] = {"mae": np.nan, "r2": np.nan, "pred": None}
    
    return results


def compute_multi_horizon_curve(df: pd.DataFrame, config: Dict, 
                                dataset_name: str, manager: ExperimentManager,
                                progress: ProgressTracker = None) -> Dict[str, Any]:
    """Вычисление multi-horizon кривой ошибки + AUC."""
    n_samples = len(df)
    
    # Горизонты для AUC кривой
    max_h = int(n_samples * config["auc_max_relative_horizon"])
    horizons = np.linspace(1, max_h, config["auc_n_points"], dtype=int)
    horizons = np.unique(horizons)
    
    curves = {
        "horizons": horizons.tolist(),
        "horizons_relative": [h / n_samples for h in horizons],
        "models": {},
        "best_per_horizon": {}
    }
    
    # ← ИСПОЛЬЗУЕМ СПИСОК ИЗ КОНФИГА
    models_to_evaluate = config.get("models_to_compare", [
        "naive", "rf_base", "rf_auto_fe", "gb_base", "gb_auto_fe"
    ])
    
    total_evaluations = len(horizons) * len(models_to_evaluate)
    
    if progress:
        dataset_progress = ProgressTracker(total_evaluations, f"  {dataset_name}")
    
    for model_name in models_to_evaluate:
        maes = []
        r2s = []
        
        for h_idx, h in enumerate(horizons):
            try:
                results = evaluate_models(df, h, config, dataset_name, manager)
                
                if model_name in results:
                    maes.append(results[model_name]["mae"])
                    r2s.append(results[model_name]["r2"])
                else:
                    maes.append(np.nan)
                    r2s.append(np.nan)
            except:
                maes.append(np.nan)
                r2s.append(np.nan)
            
            if progress:
                dataset_progress.update(1)
        
        if progress:
            dataset_progress.finish()
        
        # Нормализация MAE
        y_train = df["target"].iloc[:int(len(df) * config["train_test_split"])].values
        if config.get("normalize_metrics", True):
            maes_norm = [normalize_mae(m, y_train, config.get("normalization_method", "std")) 
                        for m in maes]
        else:
            maes_norm = maes
        
        auc = compute_auc_curve(maes_norm, curves["horizons_relative"])
        
        curves["models"][model_name] = {
            "mae_raw": maes,
            "mae_normalized": maes_norm,
            "r2": r2s,
            "auc": auc
        }
    
    # Определяем лучшую модель для каждого горизонта
    for h_idx, h in enumerate(horizons):
        try:
            results = evaluate_models(df, h, config, dataset_name, None)
            if results:
                # Фильтруем только модели из списка
                valid_results = {k: v for k, v in results.items() 
                               if k in models_to_evaluate and not np.isnan(v.get("mae", np.nan))}
                if valid_results:
                    best_model = min(valid_results.items(), key=lambda x: x[1]["mae"])
                    curves["best_per_horizon"][str(h)] = {
                        "model": best_model[0],
                        "mae": best_model[1]["mae"],
                        "r2": best_model[1]["r2"]
                    }
        except:
            pass
    
    return curves


# ============================================================================
# ВИЗУАЛИЗАЦИЯ
# ============================================================================

def plot_mae_curves(all_curves: Dict[str, Dict], output_path: str):
    """График: MAE vs относительный горизонт для всех датасетов."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    datasets = list(all_curves.keys())
    
    # Все модели включая базовые и LSTM
    models = ["naive", "rf_base", "rf_auto_fe", "gb_base", "gb_auto_fe"]
    if any("lstm" in str(curves["models"]) for curves in all_curves.values()):
        models.extend(["lstm_base"])
    
    colors = {
        "naive": "gray",
        "rf_base": "steelblue", "rf_auto_fe": "darkblue",
        "gb_base": "green", "gb_auto_fe": "darkgreen",
        "lstm_base": "orange"
    }
    labels = {
        "naive": "Наивный",
        "rf_base": "RF базовая", "rf_auto_fe": "RF + Auto FE",
        "gb_base": "GB базовая", "gb_auto_fe": "GB + Auto FE",
        "lstm_base": "LSTM базовая"
    }
    
    # 1. Кривые MAE по датасетам
    ax = axes[0, 0]
    for ds in datasets:
        curves = all_curves[ds]
        for model in models:
            if model in curves["models"]:
                h_rel = curves["horizons_relative"]
                mae = curves["models"][model]["mae_normalized"]
                ax.plot(h_rel, mae, label=f"{ds}: {labels[model]}", 
                       color=colors.get(model, 'black'), alpha=0.8, linewidth=2)
    
    ax.set_xlabel("Относительный горизонт")
    ax.set_ylabel("MAE (нормализованный)")
    ax.set_title("Кривые ошибки по датасетам")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    
    # 2. Кривые R²
    ax = axes[0, 1]
    for ds in datasets:
        curves = all_curves[ds]
        for model in models:
            if model in curves["models"]:
                h_rel = curves["horizons_relative"]
                r2 = curves["models"][model]["r2"]
                ax.plot(h_rel, r2, label=f"{ds}: {labels[model]}", 
                       color=colors.get(model, 'black'), alpha=0.8, linewidth=2)
    
    ax.set_xlabel("Относительный горизонт")
    ax.set_ylabel("R²")
    ax.set_title("Доля объяснённой дисперсии")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1)
    
    # 3. Улучшение относительно наивного
    ax = axes[1, 0]
    for ds in datasets:
        curves = all_curves[ds]
        if "gb_auto_fe" in curves["models"] and "naive" in curves["models"]:
            h_rel = curves["horizons_relative"]
            mae_gb = np.array(curves["models"]["gb_auto_fe"]["mae_normalized"])
            mae_naive = np.array(curves["models"]["naive"]["mae_normalized"])
            
            improvement = ((mae_naive - mae_gb) / mae_naive) * 100
            ax.plot(h_rel, improvement, label=ds, linewidth=2)
    
    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel("Относительный горизонт")
    ax.set_ylabel("Улучшение относительно наивного (%)")
    ax.set_title("Эффективность Auto FE")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. AUC сравнение
    ax = axes[1, 1]
    auc_data = []
    for ds in datasets:
        for model in models:
            if model in all_curves[ds]["models"]:
                auc = all_curves[ds]["models"][model]["auc"]
                if not np.isnan(auc):
                    auc_data.append({
                        "dataset": ds,
                        "model": labels[model],
                        "auc": auc
                    })
    
    if auc_data:
        auc_df = pd.DataFrame(auc_data)
        pivot = auc_df.pivot(index="dataset", columns="model", values="auc")
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlOrRd_r", ax=ax)
        ax.set_title("AUC ошибки (меньше — лучше)")
    
    plt.suptitle("Сравнительный анализ: Multi-horizon кривые + AUC", 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ График сохранён: {output_path}")


def plot_auc_comparison(all_curves: Dict[str, Dict], output_path: str):
    """График: Сравнение AUC по моделям и датасетам."""
    data = []
    
    for ds_name, curves in all_curves.items():
        for model_name, model_data in curves["models"].items():
            auc = model_data.get("auc")
            if auc is not None and not np.isnan(auc):
                data.append({
                    "dataset": ds_name,
                    "model": model_name.replace("_", " ").title(),
                    "auc": auc
                })
    
    if not data:
        print("  ⚠️  Нет данных для AUC графика")
        return
    
    df = pd.DataFrame(data)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    pivot = df.pivot(index="dataset", columns="model", values="auc")
    pivot.plot(kind="bar", ax=ax, figsize=(10, 6), colormap="viridis")
    
    ax.set_ylabel("AUC ошибки (меньше — лучше)")
    ax.set_xlabel("Датасет")
    ax.set_title("Интегральная оценка качества (AUC под кривой MAE)")
    ax.legend(title="Модель", bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ График сохранён: {output_path}")


# ============================================================================
# MAIN ФУНКЦИЯ
# ============================================================================

def main():
    print_section_header("🔬 СРАВНИТЕЛЬНЫЙ АНАЛИЗ: Multi-dataset, Multi-horizon, Multi-model")
    
    start_time = datetime.now()
    print(f" Время запуска: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Инициализация ExperimentManager
    manager = ExperimentManager(experiment_type="comparison")
    
    print(f"\n Результаты: {manager.full_dir}/")
    print(f" JSON-отчёты: {manager.full_dir}/{COMPARISON_CONFIG['json_subdir']}/")
    print(f" Графики: {manager.full_dir}/{COMPARISON_CONFIG['plots_subdir']}/")
    
    if not HAS_TENSORFLOW:
        print("\n    TensorFlow не установлен. LSTM модели будут пропущены.")
    
    # Вычисляем общее количество задач для прогресса
    n_datasets = len(DATASET_CONFIGS)
    n_points = COMPARISON_CONFIG["auc_n_points"]
    n_models = len(COMPARISON_CONFIG.get("models_to_compare", []))
    total_tasks = n_datasets * n_points * n_models
    
    print(f"\n План выполнения:")
    print(f"   • Датасетов: {n_datasets}")
    print(f"   • Точек на кривой: {n_points}")
    print(f"   • Моделей на точку: {n_models} ({', '.join(COMPARISON_CONFIG.get('models_to_compare', []))})")
    print(f"   • Всего оценок моделей: {total_tasks:,}")
    print(f"   • Ориентировочное время: ~{total_tasks * 2 / 60:.0f}-{total_tasks * 5 / 60:.0f} минут")
    
    all_curves = {}
    all_summaries = {}
    
    # ========================================================================
    # 1. Обработка каждого датасета
    # ========================================================================
    print_section_header(f"[1/3] Обработка датасетов", "=")
    
    # Создаём общий трекер прогресса
    progress = ProgressTracker(total_tasks, "Общий прогресс")
    
    for ds_idx, (ds_name, ds_config) in enumerate(DATASET_CONFIGS.items(), 1):
        print_subsection(f"[{ds_idx}/{n_datasets}] {ds_name}: {ds_config['description']}")
        
        try:
            # Загрузка
            df = load_dataset(ds_name, ds_config)
            
            if len(df) < 100:
                print(f"      Слишком мало данных ({len(df)}), пропускаем")
                continue
            
            # Вычисление multi-horizon кривой
            print(f"\n     Вычисление кривых ({COMPARISON_CONFIG['auc_n_points']} точек)...")
            curves = compute_multi_horizon_curve(
                df, COMPARISON_CONFIG, ds_name, manager, progress
            )
            all_curves[ds_name] = curves
            
            # Сохранение детального отчёта
            summary = {
                "dataset": ds_name,
                "n_samples": len(df),
                "freq": ds_config["freq"],
                "target_unit": ds_config["target_unit"],
                "curves": curves,
                "best_auc": min(
                    (m["auc"] for m in curves["models"].values() 
                     if not np.isnan(m.get("auc", np.nan))),
                    default=np.nan
                ),
                "best_per_horizon": curves.get("best_per_horizon", {})
            }
            all_summaries[ds_name] = summary
            
            # JSON-отчёт
            report_path = os.path.join(
                manager.full_dir, COMPARISON_CONFIG["json_subdir"],
                f"{ds_name}_curves.json"
            )
            os.makedirs(os.path.dirname(report_path), exist_ok=True)
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
            
            print(f"\n     AUC: {summary['best_auc']:.4f} (лучшая модель)")
            
            # Вывод лучшей модели для каждого горизонта
            if curves.get("best_per_horizon"):
                print(f"\n     Лучшие модели по горизонтам:")
                for h, best in curves["best_per_horizon"].items():
                    print(f"       Горизонт {h}: {best['model']} (MAE={best['mae']:.3f}, R²={best['r2']:.3f})")
            
        except Exception as e:
            print(f"     Ошибка: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Завершаем прогресс
    progress.finish()
    
    if not all_curves:
        print("\n Нет данных для построения графиков!")
        return
    
    # ========================================================================
    # 2. Генерация сравнительных графиков
    # ========================================================================
    print_section_header(f"[2/3] Генерация графиков", "=")
    
    plots_dir = os.path.join(manager.full_dir, COMPARISON_CONFIG["plots_subdir"])
    os.makedirs(plots_dir, exist_ok=True)
    
    # 1. Кривые MAE/R²
    plot_mae_curves(
        all_curves, 
        os.path.join(plots_dir, "mae_curves_comparison.png")
    )
    
    # 2. AUC сравнение
    plot_auc_comparison(
        all_curves,
        os.path.join(plots_dir, "auc_comparison.png")
    )
    
    print(f"\n  Все графики сохранены в {plots_dir}/")
    
    # ========================================================================
    # 3. Сводный отчёт
    # ========================================================================
    print_section_header(f"[3/3] Формирование сводного отчёта", "=")
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print(f"\n{'Датасет':<15} {'Частота':<10} {'Лучшая AUC':<12} {'Модель':<20}")
    print("-" * 60)
    
    for ds_name, summary in all_summaries.items():
        best_auc = summary["best_auc"]
        if not np.isnan(best_auc):
            best_model = min(
                ((name, data["auc"]) for name, data in summary["curves"]["models"].items() 
                 if not np.isnan(data.get("auc", np.nan))),
                key=lambda x: x[1],
                default=("N/A", np.nan)
            )[0]
            print(f"{ds_name:<15} {summary['freq']:<10} {best_auc:>10.4f}   {best_model:<20}")
    
    print("-" * 60)
    
    # Сохранение summary
    final_report = {
        "experiment_id": manager.experiment_id,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "duration_seconds": duration,
        "config": COMPARISON_CONFIG,
        "datasets": {k: {
            "n_samples": v["n_samples"],
            "freq": v["freq"],
            "best_auc": v["best_auc"]
        } for k, v in all_summaries.items()},
        "curves_summary": {
            ds: {
                model: {"auc": data["auc"]} 
                for model, data in curves["models"].items()
            }
            for ds, curves in all_curves.items()
        },
        "best_per_horizon": {
            ds: summary.get("best_per_horizon", {})
            for ds, summary in all_summaries.items()
        }
    }
    
    summary_path = os.path.join(manager.full_dir, "comparison_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2, ensure_ascii=False, default=str)
    
    # Форматируем длительность
    if duration < 60:
        duration_str = f"{int(duration)}с"
    elif duration < 3600:
        duration_str = f"{int(duration/60)}м {int(duration%60)}с"
    else:
        duration_str = f"{int(duration/3600)}ч {int((duration%3600)/60)}м"
    
    print(f"\n  Длительность: {duration_str} ({duration/60:.1f} минут)")
    print(f" Сводный отчёт: {summary_path}")
    
    # Ключевой инсайт
    print_section_header("🔑 КЛЮЧЕВОЙ ИНСАЙТ", "=")
    
    if all_summaries:
        best_overall = min(
            ((ds, s["best_auc"]) for ds, s in all_summaries.items() 
             if not np.isnan(s["best_auc"])),
            key=lambda x: x[1],
            default=("N/A", np.nan)
        )
        print(f"\n Наилучшая предсказуемость: {best_overall[0]} (AUC={best_overall[1]:.4f})")
        print(f"   → Меньшее AUC = более пологая кривая роста ошибки")
        print(f"   → Датасет лучше поддаётся прогнозу на длинных горизонтах")
    
    print_section_header("Сравнительный анализ завершён!", "=")


if __name__ == "__main__":
    try:
        import matplotlib
        import seaborn
    except ImportError:
        print("WARNING: matplotlib/seaborn не установлены.")
        print("Установите: pip install matplotlib seaborn")
    
    main()