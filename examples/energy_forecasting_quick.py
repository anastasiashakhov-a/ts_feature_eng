# examples/energy_forecasting_quick.py
"""
Быстрый тестовый пример прогнозирования энергопотребления.

Этот скрипт предназначен для:
- Быстрой проверки работоспособности всех компонентов
- Отладки пайплайна без долгого ожидания
- CI/CD тестов

Оптимизации для скорости:
- Байесовская оптимизация: 3 итерации вместо 20
- Уменьшенный размер данных (выборка 10%)
- Упрощённые модели (меньше деревьев, меньшая глубина)
- Отключены тяжёлые операции (SHAP, сложные визуализации)
- Минимальное количество признаков

Использует Optimizer v2.0:
- Out-of-sample penalties для честной оценки
- Semantic entropy для разнообразия признаков
- Relative score vs naive для переносимости между задачами
"""

import os
import json
import hashlib
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Неинтерактивный бэкенд для скорости
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from ts_feature_eng import AutoFeatureEngineer, OptimizerConfig  # ← НОВЫЙ ИМПОРТ
from ts_feature_eng.utils.metrics import relative_score, naive_gain  # ← НОВЫЕ МЕТРИКИ
from ts_feature_eng.transformers.lag import LagTransformer
from ts_feature_eng.transformers.window import WindowTransformer


# ============================================================================
# КОНФИГУРАЦИЯ ЭКСПЕРИМЕНТА
# ============================================================================
EXPERIMENT_CONFIG = {
    "sample_ratio": 0.1,          # Доля данных для выборки
    "n_calls": 3,                 # Итерации байесовской оптимизации
    "n_initial_points": 2,        # Начальные точки оптимизации
    "selection_threshold": 0.3,   # Порог отбора признаков
    "variance_threshold": 0.05,   # Порог дисперсии
    "shap_selection": False,      # Отключить SHAP для скорости
    "n_estimators": 50,           # Деревья в модели
    "max_depth": 3,               # Глубина деревьев
    "learning_rate": 0.1,         # Скорость обучения
    "random_state": 42,           # Сид для воспроизводимости
    "train_test_split": 0.8,      # Доля обучающей выборки
}

# ============================================================================
# КОНФИГУРАЦИЯ ОПТИМИЗАТОРА (УПРОЩЁННАЯ ДЛЯ БЫСТРЫХ ТЕСТОВ)
# ============================================================================
OPTIMIZER_CONFIG = OptimizerConfig(
    # Упрощённые штрафы для скорости
    dominance_lambda=0.3,          # Мягкий штраф за доминирование
    naive_lambda=0.2,              # Мягкий штраф за копирование lag_1
    entropy_lambda=0.05,           # Минимальный бонус за разнообразие
    collapse_lambda=0.2,           # Штраф за feature collapse
    
    # Пороги
    dominance_threshold=0.75,      # Макс. доля важности одного признака
    naive_corr_threshold=0.95,     # Макс. корреляция с lag_1
    
    # Отключаем тяжёлые операции для скорости
    scale_penalties=False,         # Не масштабируем штрафы (быстрее)
    use_oof_penalties=False,       # Не считаем OOF (быстрее)
    
    # Диагностика
    log_diagnostics=True,          # Логировать диагностические метрики
    diagnostics_file="results/optimization_diagnostics.csv",         
        
    # Отключаем horizon-aware для скорости
    use_horizon_aware=False,
    forecast_horizons=[1],
    
    # Search space
    use_conditional_space=True,
    use_progressive_modes=False,
)


def get_experiment_id(config):
    """Генерирует уникальный ID эксперимента на основе конфигурации."""
    config_str = json.dumps(config, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()[:8]


def save_results_to_csv(metrics, config, experiment_id, results_dir="results"):
    """
    Сохраняет метрики и параметры эксперимента в CSV файлы.
    
    Параметры
    ----------
    metrics : dict
        Метрики качества модели.
    config : dict
        Параметры эксперимента.
    experiment_id : str
        Уникальный ID эксперимента.
    results_dir : str
        Директория для сохранения результатов.
    """
    # Создаем директорию для результатов
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # 1. Сохраняем метрики
    metrics_file = os.path.join(results_dir, "metrics_history.csv")
    metrics_record = {
        "timestamp": timestamp,
        "experiment_id": experiment_id,
        "mae_mw": metrics.get("MAE", np.nan),
        "r2": metrics.get("R2", np.nan),
        "relative_score": metrics.get("Relative Score vs Naive", np.nan),  # ← НОВАЯ МЕТРИКА
        "naive_gain": metrics.get("Naive Gain", np.nan),                  # ← НОВАЯ МЕТРИКА
        "n_features": metrics.get("n_features", 0),
        "n_train_samples": metrics.get("n_train_samples", 0),
        "n_test_samples": metrics.get("n_test_samples", 0),
        # Диагностика оптимизатора
        "max_feature_share": metrics.get("max_feature_share", np.nan),
        "naive_corr": metrics.get("naive_corr", np.nan),
    }
    
    # Проверяем, существует ли файл
    file_exists = os.path.exists(metrics_file)
    
    with open(metrics_file, "a", encoding="utf-8") as f:
        if not file_exists:
            # Записываем заголовок
            header = ",".join(metrics_record.keys()) + "\n"
            f.write(header)
        
        # Записываем данные
        values = [str(v) for v in metrics_record.values()]
        f.write(",".join(values) + "\n")
    
    # 2. Сохраняем параметры эксперимента
    params_file = os.path.join(results_dir, "experiments_config.csv")
    params_record = {
        "timestamp": timestamp,
        "experiment_id": experiment_id,
        **config
    }
    
    file_exists = os.path.exists(params_file)
    
    with open(params_file, "a", encoding="utf-8") as f:
        if not file_exists:
            # Записываем заголовок
            header = ",".join(params_record.keys()) + "\n"
            f.write(header)
        
        # Записываем данные
        values = [str(v) for v in params_record.values()]
        f.write(",".join(values) + "\n")
    
    # 3. Сохраняем полный отчёт в JSON (для детального анализа)
    report_file = os.path.join(results_dir, f"experiment_{experiment_id}_{timestamp}.json")
    full_report = {
        "timestamp": timestamp,
        "experiment_id": experiment_id,
        "config": config,
        "optimizer_config": {
            "dominance_lambda": OPTIMIZER_CONFIG.dominance_lambda,
            "naive_lambda": OPTIMIZER_CONFIG.naive_lambda,
            "entropy_lambda": OPTIMIZER_CONFIG.entropy_lambda,
            "use_oof_penalties": OPTIMIZER_CONFIG.use_oof_penalties,
        },
        "metrics": metrics,
    }
    
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False)
    
    print(f"   Результаты сохранены в {results_dir}/")
    print(f"   ID эксперимента: {experiment_id}")
    
    return metrics_file, params_file, report_file


def load_morocco_energy_data_sample(data_path=None, sample_ratio=0.1, random_state=42):
    """
    Загружает данные с выборкой для ускорения тестов.
    
    Параметры
    ----------
    data_path : str, опционально
        Путь к CSV файлу.
    sample_ratio : float, по умолчанию 0.1
        Доля данных для загрузки (0.1 = 10%).
    random_state : int, опционально
        Сид для воспроизводимости выборки.
    
    Возвращает
    ----------
    df : pd.DataFrame
        DataFrame с выборкой данных.
    """
    if data_path is None:
        possible_paths = [
            "data/morocco zone 1 - powerconsumption_resampled (1).csv",
            "data/morocco_zone_1_powerconsumption_resampled.csv",
            "../data/morocco zone 1 - powerconsumption_resampled (1).csv",
        ]
        for path in possible_paths:
            if os.path.exists(path):
                data_path = path
                break
        if data_path is None:
            raise FileNotFoundError("Файл данных не найден")
    
    print(f"Загрузка данных из: {data_path} (выборка {sample_ratio*100:.0f}%)")
    
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.strip()
    
    # Поиск столбцов
    time_col = next((c for c in df.columns if any(k in c.lower() for k in ['time', 'date', 'timestamp'])), df.columns[0])
    power_col = next((c for c in df.columns if any(k in c.lower() for k in ['power', 'consum', 'load'])), df.columns[1])
    
    df[time_col] = pd.to_datetime(df[time_col])
    
    result = pd.DataFrame({
        "timestamp": df[time_col],
        "power_consumption": pd.to_numeric(df[power_col], errors="coerce")
    }).dropna(subset=["power_consumption"])
    
    result = result.sort_values("timestamp").set_index("timestamp")
    
    # Выборка для ускорения
    if sample_ratio < 1.0:
        np.random.seed(random_state)
        n_samples = max(1000, int(len(result) * sample_ratio))  # Минимум 1000 наблюдений
        indices = np.sort(np.random.choice(len(result), n_samples, replace=False))
        result = result.iloc[indices]
    
    print(f"  Итоговый размер: {len(result)} наблюдений")
    return result


def quick_feature_engineering(X, y, n_calls=3, random_state=42):
    """
    Быстрая автоматическая инженерия признаков для тестов.
    
    Параметры
    ----------
    X : pd.DataFrame
        Признаки.
    y : pd.Series
        Целевая переменная.
    n_calls : int, по умолчанию 3
        Количество итераций оптимизации.
    random_state : int, опционально
        Сид для воспроизводимости.
    
    Возвращает
    ----------
    X_transformed : pd.DataFrame
        Трансформированные признаки.
    engineer : AutoFeatureEngineer
        Обученный инженер признаков.
    """
    print(f"  Запуск быстрой оптимизации ({n_calls} итераций)...")
    
    engineer = AutoFeatureEngineer(
        optimize=True,
        n_calls=n_calls,
        n_initial_points=2,
        apply_selection=True,
        selection_threshold=0.3,  # Более агрессивная фильтрация
        variance_threshold=0.05,
        shap_selection=False,  # Отключаем SHAP для скорости
        random_state=random_state,
        verbose=1,
        optimizer_config=OPTIMIZER_CONFIG  # ← ПЕРЕДАЁМ КОНФИГУРАЦИЮ
    )
    
    X_transformed = engineer.fit_transform(X, y)
    
    print(f"  Сгенерировано признаков: {X_transformed.shape[1]}")
    if X_transformed.shape[1] > 0:
        print(f"  Примеры: {list(X_transformed.columns[:3])}")
    
    return X_transformed, engineer


def quick_model_training(X_train, y_train, X_test, y_test, config):
    """
    Быстрое обучение и оценка модели.
    
    Возвращает
    ----------
    metrics : dict
        Метрики качества.
    model : object
        Обученная модель.
    """
    print("  Обучение упрощённой модели...")
    
    model = GradientBoostingRegressor(
        n_estimators=config.get("n_estimators", 50),
        max_depth=config.get("max_depth", 3),
        learning_rate=config.get("learning_rate", 0.1),
        random_state=config.get("random_state", 42)
    )
    
    model.fit(X_train.fillna(0), y_train)
    
    y_pred = model.predict(X_test.fillna(0))
    
    metrics = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "R2": r2_score(y_test, y_pred),
        "n_features": X_train.shape[1],
        "n_train_samples": len(X_train),
        "n_test_samples": len(X_test),
    }
    
    return metrics, model


def main():
    print("=" * 70)
    print("БЫСТРЫЙ ТЕСТ: ПРОГНОЗИРОВАНИЕ ЭНЕРГОПОТРЕБЛЕНИЯ (OPTIMIZER v2.0)")
    print("=" * 70)
    
    # 🔑 ЛОГИРОВАНИЕ КОНФИГУРАЦИИ ОПТИМИЗАТОРА
    print(f"\nКонфигурация оптимизатора:")
    print(f"  • dominance_lambda: {OPTIMIZER_CONFIG.dominance_lambda}")
    print(f"  • naive_lambda: {OPTIMIZER_CONFIG.naive_lambda}")
    print(f"  • entropy_lambda: {OPTIMIZER_CONFIG.entropy_lambda}")
    print(f"  • use_oof_penalties: {OPTIMIZER_CONFIG.use_oof_penalties}")
    print(f"  • scale_penalties: {OPTIMIZER_CONFIG.scale_penalties}")
    
    # Генерируем ID эксперимента
    experiment_id = get_experiment_id(EXPERIMENT_CONFIG)
    print(f"ID эксперимента: {experiment_id}")
    print(f"Время запуска: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. Загрузка данных с выборкой
    print("\n1. Загрузка данных (выборка 10%)...")
    df = load_morocco_energy_data_sample(
        sample_ratio=EXPERIMENT_CONFIG["sample_ratio"],
        random_state=EXPERIMENT_CONFIG["random_state"]
    )
    
    # 2. Подготовка
    print("\n2. Подготовка данных...")
    y = df["power_consumption"].shift(-1).dropna()
    X = df.loc[y.index, ["power_consumption"]].copy()
    
    split_idx = int(len(X) * EXPERIMENT_CONFIG["train_test_split"])
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"   Train: {len(X_train)}, Test: {len(X_test)}")
    
    # 3. Инженерия признаков (быстрая)
    print("\n3. Автоматическая инженерия признаков (быстрый режим)...")
    X_train_eng, engineer = quick_feature_engineering(
        X_train, y_train,
        n_calls=EXPERIMENT_CONFIG["n_calls"],
        random_state=EXPERIMENT_CONFIG["random_state"]
    )
    X_test_eng = engineer.transform(X_test)
    
    # 4. Обучение модели
    print("\n4. Обучение модели...")
    metrics, model = quick_model_training(
        X_train_eng, y_train, X_test_eng, y_test,
        config=EXPERIMENT_CONFIG
    )
    
    # 5. Результаты
    print("\n5. Результаты:")
    print(f"   MAE: {metrics['MAE']:.2f} МВт")
    print(f"   R²:  {metrics['R2']:.4f}")
    print(f"   Признаков: {metrics['n_features']}")
    
    # 6. Проверка важности признаков
    print("\n6. Топ-5 важных признаков:")
    if hasattr(model, 'feature_importances_'):
        importance = pd.Series(model.feature_importances_, index=X_train_eng.columns)
        top = importance.nlargest(5)
        for feat, imp in top.items():
            print(f"   {feat[:40]:40s} {imp*100:5.1f}%")
    
    # 7. Проверка истории оптимизации
    print("\n7. История оптимизации:")
    history = engineer.get_optimization_history()
    if history is not None and not history.empty:
        print(f"   Итераций выполнено: {len(history)}")
        print(f"   Лучшая метрика: {history['score'].max():.4f}")
    
    # 8. Вычисление relative metrics (с обработкой NaN)
    print("\n8. Вычисление relative metrics...")
    lag1_col = [c for c in X_test_eng.columns if "lag_1" in c and "core_lags" in c]
    if lag1_col and lag1_col[0] in X_test_eng.columns:
        y_test_lag1 = X_test_eng[lag1_col[0]].values
        
        # Удаляем NaN для корректного сравнения
        valid_mask = ~np.isnan(y_test_lag1) & ~np.isnan(y_test.values)
        if valid_mask.sum() > 10:  # Минимум 10 валидных наблюдений
            y_test_clean = y_test.values[valid_mask]
            y_test_lag1_clean = y_test_lag1[valid_mask]
            y_pred_clean = model.predict(X_test_eng.fillna(0))[valid_mask]
            
            naive_mae = mean_absolute_error(y_test_clean, y_test_lag1_clean)
            metrics["Relative Score vs Naive"] = relative_score(metrics["MAE"], naive_mae)
            metrics["Naive Gain"] = naive_gain(y_test_clean, y_pred_clean, y_test_lag1_clean)
            metrics["Naive MAE"] = naive_mae
            
            rel_score = metrics["Relative Score vs Naive"]
            status = "ЛУЧШЕ NAIVE" if rel_score < 1.0 else "⚠ ХУЖЕ NAIVE"
            print(f"   Relative Score vs Naive: {rel_score:.4f} {status}")
            print(f"   Naive Gain: {metrics['Naive Gain']:.2f} МВт")
        else:
            print("   ⚠ Недостаточно валидных данных для relative metrics")
    else:
        print("   ⚠ Lag_1 признак не найден")
    
    # 9. Сохранение результатов в CSV
    print("\n9. Сохранение результатов...")
    metrics_file, params_file, report_file = save_results_to_csv(
        metrics=metrics,
        config=EXPERIMENT_CONFIG,
        experiment_id=experiment_id,
        results_dir="results"
    )
    
    # 10. Минимальная визуализация
    print("\n10. Сохранение мини-графика...")
    try:
        fig, ax = plt.subplots(figsize=(8, 4))
        viz_size = min(100, len(y_test))
        ax.plot(y_test.iloc[-viz_size:].values, label="Факт", linewidth=1)
        ax.plot(model.predict(X_test_eng.fillna(0))[-viz_size:], label="Прогноз", linestyle='--')
        ax.set_title(f"Прогноз (MAE={metrics['MAE']:.1f} МВт)", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig("quick_forecast_test.png", dpi=100, bbox_inches='tight')
        plt.close()
        print("   График: quick_forecast_test.png")
    except Exception as e:
        print(f"   ⚠ Не удалось сохранить график: {e}")
    
    print("\n" + "=" * 70)
    print("БЫСТРЫЙ ТЕСТ ЗАВЕРШЕН ✓")
    print("=" * 70)
    
    # Возвращаем статус для CI/CD
    return metrics["R2"] > -1  # Простая проверка, что модель обучилась


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)