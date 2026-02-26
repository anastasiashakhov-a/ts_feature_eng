# examples/energy_forecasting_quick.py
"""
Упрощённая версия прогнозирования энергопотребления для быстрой проверки.

Использует минимальные настройки для быстрого запуска:
- Меньше итераций оптимизации (n_calls=5)
- Упрощённая визуализация
- Отключены тяжёлые вычисления по умолчанию
- Поддержка горизонтов [1, 7, 30] дней
10. Детальная визуализация прогнозов с сравнением фактических и предсказанных значений
"""
import os
import json
import warnings
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from ts_feature_eng import AutoFeatureEngineer
from ts_feature_eng.utils.experiment_logger import ExperimentManager

# Подавляем предупреждения для чистого вывода
warnings.filterwarnings('ignore', message='Precision loss')
warnings.filterwarnings('ignore', message='Degrees of freedom')
warnings.filterwarnings('ignore', category=FutureWarning)

# Попытка импортировать TensorFlow/Keras для LSTM
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
    from tensorflow.keras.callbacks import EarlyStopping
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False
    warnings.warn("TensorFlow не установлен. LSTM модели будут пропущены. Установите: pip install tensorflow")


# ============================================================================
# КОНФИГУРАЦИЯ ДЛЯ БЫСТРОЙ ПРОВЕРКИ
# ============================================================================
EXPERIMENT_CONFIG = {
    "sample_ratio": 1.0,
    "n_calls": 5,                 # ← Меньше итераций для скорости
    "n_initial_points": 2,        # ← Меньше начальных точек
    "selection_threshold": 0.25,
    "variance_threshold": 0.01,
    "shap_selection": False,
    "n_estimators": 50,           # ← Меньше деревьев для скорости
    "max_depth": 4,
    "learning_rate": 0.1,
    "random_state": 42,
    "train_test_split": 0.8,
    "forecast_horizons": [1, 7, 30],  # ← Горизонты в ДНЯХ
    "aggregation_freq": "15min",
    # Параметры LSTM
    "lstm_timesteps": 96,         # ← 24 часа × 4 интервала = 96 шагов
    "lstm_units": 32,             # ← Меньше нейронов для скорости
    "lstm_epochs": 5,             # ← Меньше эпох
    "lstm_batch_size": 32,
    "use_lstm": False,            # ← Отключить LSTM по умолчанию (включите для тестов)
}

# Константы для конвертации дней в интервалы
INTERVALS_PER_HOUR = 4  # 15-минутные интервалы
INTERVALS_PER_DAY = 24 * INTERVALS_PER_HOUR  # 96 интервалов в дне


def load_morocco_energy_data(data_path=None):
    """Загружает данные энергопотребления Марокко."""
    if data_path is None:
        possible_paths = [
            "data/morocco zone 1 - powerconsumption_resampled (1).csv",
            "data/morocco_zone_1_powerconsumption_resampled.csv",
            "../data/morocco zone 1 - powerconsumption_resampled (1).csv",
            os.path.expanduser("~/ts_feature_eng/data/morocco zone 1 - powerconsumption_resampled (1).csv")
        ]
        for path in possible_paths:
            if os.path.exists(path):
                data_path = path
                break
        if data_path is None:
            raise FileNotFoundError(f"Файл не найден. Поиск: {possible_paths}")
    
    print(f"Загрузка данных из: {data_path}")
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.strip()
    
    time_col = [c for c in df.columns if 'time' in c.lower() or 'date' in c.lower()][0]
    power_col = [c for c in df.columns if 'power' in c.lower() or 'consum' in c.lower()][0]
    
    df[time_col] = pd.to_datetime(df[time_col])
    result = pd.DataFrame({"timestamp": df[time_col], "power_consumption": df[power_col]})
    result = result.set_index("timestamp").sort_index().dropna(subset=["power_consumption"])
    
    print(f"  Загружено {len(result)} наблюдений, диапазон: {result.index.min()} — {result.index.max()}")
    return result


def evaluate_forecast(y_true, y_pred, horizon_days=1):
    """Оценивает качество прогноза."""
    # ← ИСПРАВЛЕНИЕ: если y_pred — скаляр, создаём массив той же длины
    if np.isscalar(y_pred):
        y_pred = np.full(len(y_true), y_pred)
    
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if len(y_true) != len(y_pred):
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    mask = np.abs(y_true) > 1e-6
    if np.sum(mask) > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = np.nan
    
    peak_consumption = np.max(y_true)
    mean_abs_error_pct = (mae / peak_consumption) * 100 if peak_consumption > 0 else np.nan
    
    metrics = {
        "MAE (МВт)": mae,
        "RMSE (МВт)": rmse,
        "MAPE (%)": mape,
        "R²": r2,
        "Ошибка от пика (%)": mean_abs_error_pct,
        "Пик потребления (МВт)": peak_consumption
    }
    
    day_word = "день" if horizon_days == 1 else "дня" if horizon_days in [2,3,4] else "дней"
    print(f"\n  Оценка качества (горизонт: {horizon_days} {day_word}):")
    print(f"  MAE: {mae:.2f} МВт | RMSE: {rmse:.2f} МВт | R²: {r2:.4f} | Ошибка от пика: {mean_abs_error_pct:.2f}%")
    
    return metrics


def prepare_lstm_data(X, y, timesteps=96, train_split=0.8):
    """Подготавливает данные для LSTM: создаёт последовательности [samples, timesteps, features]."""
    combined = pd.concat([X, y.to_frame()], axis=1).dropna()
    
    if len(combined) <= timesteps:
        raise ValueError(f"Недостаточно данных для LSTM: {len(combined)} <= {timesteps}")
    
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_scaled = scaler_X.fit_transform(combined[X.columns])
    
    # ← ИСПРАВЛЕНИЕ: y_values должен быть 2D для fit_transform
    y_values = combined[[y.name]].values
    y_scaled = scaler_y.fit_transform(y_values).ravel()
    
    X_lstm, y_lstm = [], []
    for i in range(timesteps, len(X_scaled)):
        X_lstm.append(X_scaled[i-timesteps:i])
        y_lstm.append(y_scaled[i])
    
    X_lstm = np.array(X_lstm)
    y_lstm = np.array(y_lstm)
    
    split_idx = int(len(X_lstm) * train_split)
    X_train, X_test = X_lstm[:split_idx], X_lstm[split_idx:]
    y_train, y_test = y_lstm[:split_idx], y_lstm[split_idx:]
    
    return X_train, X_test, y_train, y_test, scaler_y


def train_lstm_model(X_train, y_train, X_test, config, verbose=0):
    """Обучает LSTM модель и делает прогноз."""
    if not HAS_TENSORFLOW:
        return None, None
    
    timesteps = config.get("lstm_timesteps", 96)
    units = config.get("lstm_units", 32)
    epochs = config.get("lstm_epochs", 5)
    batch_size = config.get("lstm_batch_size", 32)
    
    # ← ИСПРАВЛЕНИЕ: Используем Input() вместо input_shape в LSTM слое
    model = Sequential([
        Input(shape=(timesteps, X_train.shape[2])),
        LSTM(units, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
    
    if verbose >= 1:
        print(f"   Обучение LSTM: {epochs} эпох, batch_size={batch_size}")
    
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
              validation_split=0.1, callbacks=[early_stop], verbose=verbose)
    
    y_pred_scaled = model.predict(X_test, verbose=0).ravel()
    return model, y_pred_scaled


def forecast_on_horizon(df, horizon_days, config, experiment_id, manager):
    """Прогнозирование на заданный горизонт в ДНЯХ."""
    horizon_intervals = horizon_days * INTERVALS_PER_DAY
    
    day_word = "день" if horizon_days == 1 else "дня" if horizon_days in [2,3,4] else "дней"
    print(f"\n[QUICK] Горизонт: {horizon_days} {day_word} ({horizon_intervals} интервалов)")
    
    y = df["power_consumption"].shift(-horizon_intervals)
    X = pd.DataFrame({"power_consumption": df["power_consumption"]})
    
    valid_mask = ~y.isna() & X.notna().all(axis=1)
    X = X[valid_mask]
    y = y[valid_mask]
    
    split_idx = int(len(X) * config["train_test_split"])
    X_train_base, X_test_base = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"  Train: {len(X_train_base)} | Test: {len(X_test_base)}")
    
    # Auto FE
    print("  Генерация признаков...")
    engineer = AutoFeatureEngineer(
        optimize=True, n_calls=config["n_calls"], n_initial_points=config["n_initial_points"],
        apply_selection=True, selection_threshold=config["selection_threshold"],
        variance_threshold=config["variance_threshold"], shap_selection=config["shap_selection"],
        random_state=config["random_state"], verbose=0
    )
    X_train_auto = engineer.fit_transform(X_train_base, y_train)
    X_test_auto = engineer.transform(X_test_base)
    print(f"  Сгенерировано признаков: {X_train_auto.shape[1]}")
    
    # Сравнение моделей
    results = []
    
    # Наивный
    naive_pred = y_test.shift(horizon_intervals).fillna(y_test.mean())
    naive_mae = mean_absolute_error(y_test, naive_pred)
    results.append(("Наивный", naive_mae, None))
    
    # Сезонный наивный (24 часа = 96 интервалов назад)
    seasonal_pred = y_test.shift(INTERVALS_PER_DAY).fillna(y_test.mean())
    seasonal_mae = mean_absolute_error(y_test, seasonal_pred)
    results.append(("Сезонный наивный", seasonal_mae, None))
    
    # Random Forest (базовые)
    rf = RandomForestRegressor(n_estimators=20, max_depth=3, random_state=42)
    rf.fit(X_train_base.fillna(0), y_train)
    rf_pred = rf.predict(X_test_base.fillna(0))
    rf_mae = mean_absolute_error(y_test, rf_pred)
    results.append(("RF (базовые)", rf_mae, rf_pred))
    
    # Random Forest (auto FE)
    rf_auto = RandomForestRegressor(n_estimators=20, max_depth=3, random_state=42)
    rf_auto.fit(X_train_auto.fillna(0), y_train)
    rf_auto_pred = rf_auto.predict(X_test_auto.fillna(0))
    rf_auto_mae = mean_absolute_error(y_test, rf_auto_pred)
    results.append(("RF (auto FE)", rf_auto_mae, rf_auto_pred))
    
    # Gradient Boosting (базовые)
    gb = GradientBoostingRegressor(n_estimators=30, max_depth=3, learning_rate=0.1, random_state=42)
    gb.fit(X_train_base.fillna(0), y_train)
    gb_pred = gb.predict(X_test_base.fillna(0))
    gb_mae = mean_absolute_error(y_test, gb_pred)
    results.append(("GB (базовые)", gb_mae, gb_pred))
    
    # Gradient Boosting (auto FE)
    gb_auto = GradientBoostingRegressor(n_estimators=30, max_depth=3, learning_rate=0.1, random_state=42)
    gb_auto.fit(X_train_auto.fillna(0), y_train)
    gb_auto_pred = gb_auto.predict(X_test_auto.fillna(0))
    gb_auto_mae = mean_absolute_error(y_test, gb_auto_pred)
    results.append(("GB (auto FE)", gb_auto_mae, gb_auto_pred))
    
    # LSTM (базовые) - опционально
    if HAS_TENSORFLOW and config.get("use_lstm", False):
        print("  Обучение LSTM (базовые)...")
        try:
            X_lstm = df[["power_consumption"]].copy()
            X_tr, X_te, y_tr, y_te, scaler_y = prepare_lstm_data(
                X_lstm, df["power_consumption"].shift(-horizon_intervals),
                timesteps=config["lstm_timesteps"], train_split=config["train_test_split"]
            )
            lstm_model, lstm_pred_scaled = train_lstm_model(X_tr, y_tr, X_te, config, verbose=0)
            if lstm_pred_scaled is not None and len(lstm_pred_scaled) > 0:
                # Денормализация с правильными размерами
                lstm_pred = scaler_y.inverse_transform(lstm_pred_scaled.reshape(-1, 1)).ravel()
                
                # ← НОВАЯ ПРОВЕРКА: отбрасываем нереалистичные прогнозы
                if np.any(np.isnan(lstm_pred)) or np.any(np.isinf(lstm_pred)):
                    print(f"     LSTM прогноз содержит NaN/Inf, пропускаем")
                    results.append(("LSTM (базовые)", np.nan, None))
                else:
                    min_len = min(len(lstm_pred), len(y_test))
                    lstm_pred = lstm_pred[:min_len]
                    y_test_trimmed = y_test.iloc[-min_len:]
                    lstm_mae = mean_absolute_error(y_test_trimmed, lstm_pred)
                    results.append(("LSTM (базовые)", lstm_mae, lstm_pred))
            else:
                results.append(("LSTM (базовые)", np.nan, None))
        except Exception as e:
            print(f"      Ошибка LSTM: {e}")
            results.append(("LSTM (базовые)", np.nan, None))
    elif HAS_TENSORFLOW and not config.get("use_lstm", False):
        results.append(("LSTM (базовые)", np.nan, None))
        print("    LSTM отключён в конфиге (use_lstm=False)")
    else:
        results.append(("LSTM (базовые)", np.nan, None))
        print("    LSTM пропущен: TensorFlow не установлен")
    
    # Вывод результатов
    print(f"\n  {'='*60}")
    print(f"  Сравнение моделей (MAE, меньше — лучше):")
    print(f"  {'='*60}")
    valid_results = [(n, m, p) for n, m, p in results if not np.isnan(m)]
    if valid_results:
        best = min(valid_results, key=lambda x: x[1])
        for name, mae, _ in results:
            star = " ← BEST" if (name, mae, _) == best else ""
            if np.isnan(mae):
                print(f"  {name:25s} | N/A")
            else:
                imp = ((naive_mae - mae) / naive_mae) * 100 if naive_mae > 0 else 0
                print(f"  {name:25s} | {mae:7.2f} МВт | {imp:+6.1f}%{star}")
    print(f"  {'='*60}")
    
    # Возвращаем лучшую модель
    best_result = min((r for r in results if not np.isnan(r[1])), key=lambda x: x[1])
    best_name, best_mae, best_pred = best_result
    
    # ← ИСПРАВЛЕНИЕ: безопасный fallback для best_pred
    if best_pred is None:
        print(f"    Лучшая модель ({best_name}) не вернула прогноз, используем наивный")
        best_pred = y_test.shift(horizon_intervals).fillna(y_test.mean()).values
    
    metrics = evaluate_forecast(y_test, best_pred, horizon_days=horizon_days)
    metrics["model_type"] = best_name
    metrics["n_features"] = X_train_auto.shape[1] if "auto FE" in best_name else X_train_base.shape[1]
    metrics["n_train_samples"] = len(X_train_base)
    metrics["n_test_samples"] = len(X_test_base)
    
    # ← ИСПОЛЬЗУЕМ ExperimentManager для сохранения
    manager.save_metrics(metrics, config, experiment_id, horizon_days)
    manager.save_config(config, experiment_id, horizon_days)
    manager.save_json_report(metrics, config, experiment_id, horizon_days)
    
    # ← НОВОЕ: Создаём детальный график сравнения прогноза с реальностью
    print(f"\n  Создание графика сравнения прогноза с фактическими значениями...")
    try:
        # Получаем временные метки для тестовой выборки
        timestamps = y_test.index if hasattr(y_test, 'index') else None
        
        # Сохраняем график сравнения
        manager.save_forecast_comparison(
            y_true=y_test.values,
            y_pred=best_pred if best_pred is not None else y_test.mean(),
            timestamps=timestamps,
            horizon=horizon_days,
            model_name=best_name,
            mae=metrics.get("MAE (МВт)"),
            rmse=metrics.get("RMSE (МВт)"),
            r2=metrics.get("R²"),
            max_points=300,  # Показываем последние 300 точек для читаемости
            save_name=f"forecast_h{horizon_days}d_{best_name.replace(' ', '_').replace('(', '').replace(')', '')}"
        )
    except Exception as e:
        print(f"    ⚠️  Ошибка при создании графика сравнения: {e}")
    
    return metrics, naive_mae, best_mae, results


def main():
    start_time = datetime.now()
    
    print("=" * 70)
    print(" QUICK CHECK: Прогнозирование энергопотребления (Марокко)")
    print("=" * 70)
    
    # ← ИНИЦИАЛИЗИРУЕМ ExperimentManager
    manager = ExperimentManager(experiment_type="quick")
    experiment_id = manager.get_experiment_id(EXPERIMENT_CONFIG)
    
    print(f"ID: {experiment_id}")
    print(f"Время: {start_time.strftime('%H:%M:%S')}")
    print(f"Горизонты: {EXPERIMENT_CONFIG['forecast_horizons']} дней")
    print(f"Частота: {EXPERIMENT_CONFIG['aggregation_freq']}")
    print(f"Результаты сохраняются в: {manager.full_dir}/")
    
    if not HAS_TENSORFLOW:
        print("    TensorFlow не установлен. LSTM модели будут пропущены.")
    if not EXPERIMENT_CONFIG.get("use_lstm", False):
        print("    LSTM отключён в конфиге (use_lstm=False)")
    
    # Загрузка данных
    print("\n[1/4] Загрузка данных...")
    try:
        df = load_morocco_energy_data()
    except Exception as e:
        print(f" Ошибка загрузки: {e}")
        return False
    
    # Прогнозирование на всех горизонтах
    print("\n[2/4] Запуск прогнозирования...")
    all_results = []
    
    for horizon in EXPERIMENT_CONFIG["forecast_horizons"]:
        metrics, naive_mae, best_mae, results = forecast_on_horizon(
            df, horizon, EXPERIMENT_CONFIG, experiment_id, manager
        )
        all_results.append({
            "horizon": horizon,
            "metrics": metrics,
            "naive_mae": naive_mae,
            "best_mae": best_mae,
        })
    
    # Простая визуализация
    print("\n[3/4] Генерация графика...")
    try:
        fig, ax = plt.subplots(figsize=(10, 5))
        horizons = [r["horizon"] for r in all_results]
        naive_maes = [r["naive_mae"] for r in all_results]
        best_maes = [r["best_mae"] for r in all_results]
        
        x = np.arange(len(horizons))
        width = 0.35
        ax.bar(x - width/2, naive_maes, width, label='Наивный', color='gray', alpha=0.7)
        ax.bar(x + width/2, best_maes, width, label='Лучшая модель', color='steelblue', alpha=0.7)
        
        ax.set_xlabel('Горизонт прогноза (дни)')
        ax.set_ylabel('MAE (МВт)')
        ax.set_title('Сравнение MAE по горизонтам (quick check)')
        ax.set_xticks(x)
        ax.set_xticklabels(horizons)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # ← ИСПОЛЬЗУЕМ ExperimentManager для сохранения
        manager.save_plot(fig, "comparison.png")
        print("  ✓ График: results/quick/comparison.png")
        plt.close()
    except Exception as e:
        print(f"    Ошибка визуализации: {e}")
    
    # Итог
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print("\n" + "=" * 70)
    print(" QUICK CHECK ЗАВЕРШЕН")
    print("=" * 70)
    print(f"  Время выполнения: {duration:.1f} секунд ({duration/60:.1f} минут)")
    
    # Сводная таблица
    print(f"\n{'Горизонт':<12} {'Наивный MAE':<15} {'Лучшая MAE':<15} {'Улучшение':<12} {'R²':<10}")
    print("-" * 70)
    for r in all_results:
        imp = ((r["naive_mae"] - r["best_mae"]) / r["naive_mae"] * 100) if r["naive_mae"] > 0 else 0
        print(f"{r['horizon']:<12} {r['naive_mae']:>10.2f} МВт   {r['best_mae']:>10.2f} МВт   {imp:>10.1f}%   {r['metrics']['R²']:>8.4f}")
    
    print("-" * 70)
    print(f" Результаты: {manager.full_dir}/")
    print(f" Детальные графики: {manager.full_dir}/plots/")
    print(f" Глобальные метрики: {manager.global_metrics_file}")
    
    # ← ИСПОЛЬЗУЕМ ExperimentManager для сохранения summary
    summary_path = manager.save_summary(all_results, experiment_id, start_time, end_time)
    print(f" Отчёт: {summary_path}")
    
    print("=" * 70)
    
    # Быстрая проверка: если улучшение > 0% хотя бы на одном горизонте — всё работает
    any_improvement = any(((r["naive_mae"] - r["best_mae"]) / r["naive_mae"] > 0) for r in all_results if r["naive_mae"] > 0)
    if any_improvement:
        print(" ✓ ВСЁ РАБОТАЕТ! Улучшение достигнуто на одном или нескольких горизонтах.")
    else:
        print("  Нет улучшения, но скрипт отработал без ошибок.")
    
    return True


if __name__ == "__main__":
    main()