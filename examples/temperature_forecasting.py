# examples/temperature_forecasting.py
"""
Пример прогнозирования минимальных суточных температур.

Демонстрирует применение автоматической инженерии признаков для реальной задачи:
1. Загрузка и анализ данных минимальных суточных температур
2. Обнаружение сезонных паттернов (суточных, недельных, годовых)
3. Генерация календарных признаков с учетом особенностей временного ряда
4. Автоматический подбор оптимальных методов инженерии признаков
5. Прогнозирование на РАЗНЫЕ горизонты (1, 7, 30 дней)
6. Сравнение с базовыми моделями и интерпретация результатов
7. Гибридная инженерия признаков (обязательные лаги + адаптивные преобразования)
8. Сравнение моделей: базовые признаки vs Auto FE для всех алгоритмов + LSTM
9. Отслеживание времени выполнения эксперимента
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
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from ts_feature_eng import AutoFeatureEngineer
from ts_feature_eng.utils.experiment_logger import ExperimentManager

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
# КОНФИГУРАЦИЯ ЭКСПЕРИМЕНТА
# ============================================================================
EXPERIMENT_CONFIG = {
    "sample_ratio": 1.0,
    "n_calls": 25,
    "n_initial_points": 3,
    "selection_threshold": 0.25,
    "variance_threshold": 0.01,
    "shap_selection": False,
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
    "random_state": 42,
    "train_test_split": 0.8,
    "forecast_horizons": [1, 7, 30],  # Горизонты в днях
    # Параметры LSTM
    "lstm_timesteps": 30,  # Количество дней для LSTM (история)
    "lstm_units": 50,
    "lstm_epochs": 20,
    "lstm_batch_size": 32,
}


def load_temperature_data(data_path=None):
    """
    Загружает данные минимальных суточных температур.
    """
    if data_path is None:
        possible_paths = [
            "data/daily-minimum-temperatures-in-me.csv",
            "data/daily_minimum_temperatures_in_me.csv",
            "../data/daily-minimum-temperatures-in-me.csv",
            os.path.expanduser("~/ts_feature_eng/data/daily-minimum-temperatures-in-me.csv")
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                data_path = path
                break
        
        if data_path is None:
            raise FileNotFoundError(
                "Файл данных не найден. Укажите путь явно через параметр data_path.\n"
                f"Поиск производился в: {possible_paths}"
            )
    
    print(f"Загрузка данных из: {data_path}")
    
    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        raise RuntimeError(f"Ошибка при загрузке файла: {e}")
    
    df.columns = df.columns.str.strip()
    
    time_cols = [col for col in df.columns if any(kw in col.lower() for kw in ['time', 'date', 'timestamp', 'dt'])]
    temp_cols = [col for col in df.columns if any(kw in col.lower() for kw in ['temp', 'min'])]
    
    if not time_cols or not temp_cols:
        raise ValueError("Не найдены необходимые столбцы в данных")
    
    time_col = time_cols[0]
    temp_col = temp_cols[0]
    print(f"  Найден столбец временных меток: '{time_col}'")
    print(f"  Найден столбец температуры: '{temp_col}'")
    
    try:
        df[time_col] = pd.to_datetime(df[time_col])
    except Exception as e:
        raise ValueError(f"Ошибка преобразования временных меток: {e}")
    
    try:
        mask_invalid = df[temp_col].astype(str).str.contains(r'[?]', na=False)
        if mask_invalid.any():
            print(f"  Обнаружено {mask_invalid.sum()} некорректных значений в температуре. Удаляем...")
            df = df[~mask_invalid]
        
        df[temp_col] = df[temp_col].astype(str).str.replace(',', '.', regex=True).astype(float)
    except Exception as e:
        raise ValueError(f"Ошибка преобразования температуры в числовой формат: {e}")
    
    result = pd.DataFrame({
        "timestamp": df[time_col],
        "min_temperature": df[temp_col]
    })
    
    result = result.sort_values("timestamp").drop_duplicates(subset="timestamp").reset_index(drop=True)
    result = result.set_index("timestamp")
    
    initial_len = len(result)
    result = result.dropna(subset=["min_temperature"])
    final_len = len(result)
    
    print(f"  Удалено {initial_len - final_len} наблюдений с пропусками")
    print(f"  Итоговый размер данных: {len(result)} наблюдений")
    print(f"  Диапазон дат: {result.index.min()} — {result.index.max()}")
    
    return result


def plot_temperature_patterns(df, manager, title="Паттерны минимальных температур"):
    """
    Визуализирует ключевые паттерны минимальных температур и сохраняет через ExperimentManager.
    """
    print(f"\nВизуализация паттернов температур: {title}")
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    
    # 1. Общий тренд температуры
    ax = axes[0, 0]
    df["min_temperature"].plot(ax=ax, color='blue', linewidth=1, alpha=0.7)
    ax.set_title("Общий тренд минимальных температур", fontsize=12, fontweight='bold')
    ax.set_ylabel("Температура (°C)", fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 2. Сезонный паттерн (по месяцам)
    ax = axes[0, 1]
    monthly_avg = df.groupby(df.index.month)["min_temperature"].mean()
    month_labels = ['Янв', 'Фев', 'Мар', 'Апр', 'Май', 'Июн', 'Июл', 'Авг', 'Сен', 'Окт', 'Ноя', 'Дек']
    ax.plot(month_labels, monthly_avg.values, marker='o', color='darkred', linewidth=2, markersize=6)
    ax.set_title("Сезонный паттерн температур", fontsize=12, fontweight='bold')
    ax.set_ylabel("Средняя температура (°C)", fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 3. Распределение температур
    ax = axes[1, 0]
    sns.histplot(df["min_temperature"], bins=50, kde=True, ax=ax, color='purple')
    ax.axvline(df["min_temperature"].mean(), color='red', linestyle='--', label=f'Среднее: {df["min_temperature"].mean():.1f} °C')
    ax.axvline(df["min_temperature"].median(), color='orange', linestyle='--', label=f'Медиана: {df["min_temperature"].median():.1f} °C')
    ax.set_title("Распределение минимальных температур", fontsize=12, fontweight='bold')
    ax.set_xlabel("Температура (°C)", fontsize=10)
    ax.set_ylabel("Частота", fontsize=10)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Автокорреляция
    ax = axes[1, 1]
    from statsmodels.graphics.tsaplots import plot_acf
    plot_acf(df["min_temperature"].dropna(), lags=365, ax=ax, alpha=0.05)
    ax.set_title("Автокорреляция температур (до 365 лагов)", fontsize=12, fontweight='bold')
    ax.set_xlabel("Лаг (дни)", fontsize=10)
    ax.set_ylabel("ACF", fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 5. Недельный паттерн
    ax = axes[2, 0]
    weekday_avg = df.groupby(df.index.dayofweek)["min_temperature"].mean()
    weekday_labels = ['Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб', 'Вс']
    ax.bar(weekday_labels, weekday_avg.values, color='steelblue', alpha=0.8)
    ax.set_title("Недельный паттерн температур", fontsize=12, fontweight='bold')
    ax.set_ylabel("Средняя температура (°C)", fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 6. Годовой тренд
    ax = axes[2, 1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        yearly_trend = df.resample('YE').mean()
    ax.plot(yearly_trend.index.year, yearly_trend["min_temperature"], marker='o', color='green', linewidth=2, markersize=6)
    ax.set_title("Годовой тренд температур", fontsize=12, fontweight='bold')
    ax.set_ylabel("Средняя температура (°C)", fontsize=10)
    ax.set_xlabel("Год", fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # ← ИСПОЛЬЗУЕМ ExperimentManager для сохранения
    manager.save_plot(fig, "temperature_patterns.png")
    print("  График сохранён через ExperimentManager")
    plt.close()


def evaluate_forecast(y_true, y_pred, horizon_days=1):
    """
    Оценивает качество прогноза с интерпретацией для задачи прогнозирования температуры.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    mask = np.abs(y_true) > 1e-6
    if np.sum(mask) > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = np.nan
    
    temp_range = np.max(y_true) - np.min(y_true)
    mean_abs_error_pct = (mae / temp_range) * 100 if temp_range > 0 else np.nan
    
    metrics = {
        "MAE (°C)": mae,
        "RMSE (°C)": rmse,
        "MAPE (%)": mape,
        "R²": r2,
        "Ошибка от диапазона (%)": mean_abs_error_pct,
        "Диапазон температур (°C)": temp_range
    }
    
    print(f"\nОценка качества прогноза (горизонт: {horizon_days} день{'я' if horizon_days in [2,3,4] else 'ей' if horizon_days > 4 else 'день'}):")
    print("-" * 65)
    print(f"{'Метрика':<25} {'Значение':<15} {'Интерпретация'}")
    print("-" * 65)
    print(f"{'MAE':<25} {mae:>10.2f} °C  {'Средняя ошибка прогноза'}")
    print(f"{'RMSE':<25} {rmse:>10.2f} °C  {'Чувствительность к крупным ошибкам'}")
    print(f"{'MAPE':<25} {mape:>10.2f} %    {'Относительная ошибка'}")
    print(f"{'R²':<25} {r2:>10.4f}       {'Доля объясненной дисперсии'}")
    print(f"{'Ошибка от диапазона':<25} {mean_abs_error_pct:>10.2f} %    {'Критичность для точности'}")
    print("-" * 65)
    
    if mean_abs_error_pct < 5.0:
        reliability = "ОЧЕНЬ ВЫСОКАЯ"
        recommendation = "Подходит для точного прогнозирования"
    elif mean_abs_error_pct < 10.0:
        reliability = "ВЫСОКАЯ"
        recommendation = "Подходит для общих прогнозов"
    elif mean_abs_error_pct < 20.0:
        reliability = "СРЕДНЯЯ"
        recommendation = "Требуется уточнение прогноза"
    else:
        reliability = "НИЗКАЯ"
        recommendation = "Требуется ручная коррекция"
    
    print(f"\nНадежность прогноза: {reliability}")
    print(f"Рекомендация: {recommendation}")
    
    return metrics


def prepare_lstm_data(X, y, timesteps=30, train_split=0.8):
    """
    Подготавливает данные для LSTM: создаёт последовательности [samples, timesteps, features].
    """
    combined = pd.concat([X, y.to_frame()], axis=1).dropna()
    
    if len(combined) <= timesteps:
        raise ValueError(f"Недостаточно данных для LSTM: {len(combined)} <= {timesteps}")
    
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_scaled = scaler_X.fit_transform(combined[X.columns])
    y_values = combined[[y.name]].values.reshape(-1, 1)
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
    """
    Обучает LSTM модель и делает прогноз.
    """
    if not HAS_TENSORFLOW:
        return None, None
    
    timesteps = config.get("lstm_timesteps", 30)
    units = config.get("lstm_units", 50)
    epochs = config.get("lstm_epochs", 20)
    batch_size = config.get("lstm_batch_size", 32)
    
    model = Sequential([
        Input(shape=(timesteps, X_train.shape[2])),
        LSTM(units, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    
    if verbose >= 1:
        print(f"   Обучение LSTM: {epochs} эпох, batch_size={batch_size}")
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=verbose
    )
    
    y_pred_scaled = model.predict(X_test, verbose=0).ravel()
    
    return model, y_pred_scaled


def forecast_on_horizon(df, horizon_days, config, experiment_id, manager):
    """
    Прогнозирование на заданный горизонт.
    """
    print(f"\n{'='*80}")
    print(f"ГОРИЗОНТ ПРОГНОЗА: {horizon_days} день{'я' if horizon_days in [2,3,4] else 'ей' if horizon_days > 4 else ''}")
    print(f"{'='*80}")
    
    y = df["min_temperature"].shift(-horizon_days)
    
    valid_mask = ~y.isna()
    X = df.loc[y.index, ["min_temperature"]].copy()
    y = y[valid_mask]
    X = X[valid_mask]
    
    split_idx = int(len(X) * config["train_test_split"])
    X_train_base, X_test_base = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"\n   Обучающая выборка: {len(X_train_base)} наблюдений")
    print(f"   Тестовая выборка: {len(X_test_base)} наблюдений")
    
    # Автоматическая инженерия признаков
    print(f"\n   Автоматическая инженерия признаков...")
    
    engineer = AutoFeatureEngineer(
        optimize=True,
        n_calls=config["n_calls"],
        n_initial_points=config["n_initial_points"],
        apply_selection=True,
        selection_threshold=config["selection_threshold"],
        variance_threshold=config["variance_threshold"],
        shap_selection=config["shap_selection"],
        random_state=config["random_state"],
        verbose=0
    )
    
    X_train_auto = engineer.fit_transform(X_train_base, y_train)
    X_test_auto = engineer.transform(X_test_base)
    
    print(f"   Сгенерировано признаков: {X_train_auto.shape[1]}")
    
    print(f"\n   Сравнение моделей...")
    
    results_comparison = []
    
    # 1. Наивный прогноз
    naive_pred = y_test.shift(horizon_days).fillna(y_test.mean())
    naive_mae = mean_absolute_error(y_test, naive_pred)
    results_comparison.append(("Наивный прогноз", naive_mae, None))
    
    # 2. Сезонный наивный (365 дней назад)
    seasonal_naive_pred = y_test.shift(365).fillna(y_test.mean())
    seasonal_naive_mae = mean_absolute_error(y_test, seasonal_naive_pred)
    results_comparison.append(("Сезонный наивный (365д)", seasonal_naive_mae, None))
    
    # 3. Random Forest - базовые признаки
    rf_base = RandomForestRegressor(n_estimators=50, max_depth=3, random_state=config["random_state"])
    rf_base.fit(X_train_base.fillna(0), y_train)
    rf_base_pred = rf_base.predict(X_test_base.fillna(0))
    rf_base_mae = mean_absolute_error(y_test, rf_base_pred)
    results_comparison.append(("Random Forest (базовые)", rf_base_mae, rf_base_pred))
    
    # 4. Random Forest - auto FE
    rf_auto = RandomForestRegressor(n_estimators=50, max_depth=3, random_state=config["random_state"])
    rf_auto.fit(X_train_auto.fillna(0), y_train)
    rf_auto_pred = rf_auto.predict(X_test_auto.fillna(0))
    rf_auto_mae = mean_absolute_error(y_test, rf_auto_pred)
    results_comparison.append(("Random Forest (auto FE)", rf_auto_mae, rf_auto_pred))
    
    # 5. Gradient Boosting - базовые признаки
    gb_base = GradientBoostingRegressor(
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        learning_rate=config["learning_rate"],
        random_state=config["random_state"]
    )
    gb_base.fit(X_train_base.fillna(0), y_train)
    gb_base_pred = gb_base.predict(X_test_base.fillna(0))
    gb_base_mae = mean_absolute_error(y_test, gb_base_pred)
    results_comparison.append(("Gradient Boosting (базовые)", gb_base_mae, gb_base_pred))
    
    # 6. Gradient Boosting - auto FE
    gb_auto = GradientBoostingRegressor(
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        learning_rate=config["learning_rate"],
        random_state=config["random_state"]
    )
    gb_auto.fit(X_train_auto.fillna(0), y_train)
    gb_auto_pred = gb_auto.predict(X_test_auto.fillna(0))
    gb_auto_mae = mean_absolute_error(y_test, gb_auto_pred)
    results_comparison.append(("Gradient Boosting (auto FE)", gb_auto_mae, gb_auto_pred))
    
    # 7. LSTM - базовые признаки (только температура)
    if HAS_TENSORFLOW:
        print(f"   Обучение LSTM (базовые)...")
        try:
            X_lstm_base = df[["min_temperature"]].copy()
            X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm, scaler_y = prepare_lstm_data(
                X_lstm_base, df["min_temperature"].shift(-horizon_days),
                timesteps=config["lstm_timesteps"], train_split=config["train_test_split"]
            )
            
            lstm_model, lstm_pred_scaled = train_lstm_model(
                X_train_lstm, y_train_lstm, X_test_lstm, config, verbose=0
            )
            
            if lstm_pred_scaled is not None and len(lstm_pred_scaled) > 0:
                lstm_pred = scaler_y.inverse_transform(lstm_pred_scaled.reshape(-1, 1)).ravel()
                min_len = min(len(lstm_pred), len(y_test))
                lstm_pred = lstm_pred[:min_len]
                y_test_lstm_trimmed = y_test.iloc[-min_len:]
                
                lstm_mae = mean_absolute_error(y_test_lstm_trimmed, lstm_pred)
                results_comparison.append(("LSTM (базовые)", lstm_mae, lstm_pred))
            else:
                results_comparison.append(("LSTM (базовые)", np.nan, None))
        except Exception as e:
            print(f"    ⚠️  Ошибка LSTM (базовые): {e}")
            results_comparison.append(("LSTM (базовые)", np.nan, None))
        
        # 8. LSTM - auto FE
        print(f"   Обучение LSTM (auto FE)...")
        try:
            if X_train_auto.shape[1] > 10:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=min(10, X_train_auto.shape[1]))
                X_train_pca = pca.fit_transform(X_train_auto.fillna(0))
                X_test_pca = pca.transform(X_test_auto.fillna(0))
                X_lstm_auto = pd.DataFrame(X_train_pca, columns=[f"pc_{i}" for i in range(X_train_pca.shape[1])])
                X_test_lstm_auto = pd.DataFrame(X_test_pca, columns=[f"pc_{i}" for i in range(X_test_pca.shape[1])])
            else:
                X_lstm_auto = X_train_auto.copy()
                X_test_lstm_auto = X_test_auto.copy()
            
            X_train_lstm_auto, X_test_lstm_auto_arr, y_train_lstm_auto, y_test_lstm_auto, scaler_y_auto = prepare_lstm_data(
                X_lstm_auto, y, timesteps=config["lstm_timesteps"], train_split=config["train_test_split"]
            )
            
            lstm_auto_model, lstm_auto_pred_scaled = train_lstm_model(
                X_train_lstm_auto, y_train_lstm_auto, X_test_lstm_auto_arr, config, verbose=0
            )
            
            if lstm_auto_pred_scaled is not None and len(lstm_auto_pred_scaled) > 0:
                lstm_auto_pred = scaler_y_auto.inverse_transform(lstm_auto_pred_scaled.reshape(-1, 1)).ravel()
                min_len = min(len(lstm_auto_pred), len(y_test))
                lstm_auto_pred = lstm_auto_pred[:min_len]
                y_test_lstm_auto_trimmed = y_test.iloc[-min_len:]
                
                lstm_auto_mae = mean_absolute_error(y_test_lstm_auto_trimmed, lstm_auto_pred)
                results_comparison.append(("LSTM (auto FE)", lstm_auto_mae, lstm_auto_pred))
            else:
                results_comparison.append(("LSTM (auto FE)", np.nan, None))
        except Exception as e:
            print(f"    ⚠️  Ошибка LSTM (auto FE): {e}")
            results_comparison.append(("LSTM (auto FE)", np.nan, None))
    else:
        results_comparison.append(("LSTM (базовые)", np.nan, None))
        results_comparison.append(("LSTM (auto FE)", np.nan, None))
        print(f"    ⚠️  LSTM пропущен: TensorFlow не установлен")
    
    print(f"\n   {'='*70}")
    print(f"   Сравнение моделей по MAE (меньше — лучше):")
    print(f"   {'='*70}")
    
    best_mae = min(r[1] for r in results_comparison if not np.isnan(r[1]))
    
    for model_name, mae_value, _ in results_comparison:
        if np.isnan(mae_value):
            print(f"   {model_name:35s} | {'N/A':>6s}    | пропущен")
        else:
            star = " ← ЛУЧШАЯ" if mae_value == best_mae else ""
            improvement = ((naive_mae - mae_value) / naive_mae) * 100 if naive_mae > 0 else 0
            print(f"   {model_name:35s} | {mae_value:6.2f} °C | улучшение: {improvement:6.1f}%{star}")
    
    print(f"   {'='*70}")
    
    best_result = min((r for r in results_comparison if not np.isnan(r[1])), key=lambda x: x[1])
    best_model_name, best_mae, best_pred = best_result
    
    test_metrics = evaluate_forecast(y_test, best_pred if best_pred is not None else y_test.mean(), horizon_days=horizon_days)
    test_metrics["n_features"] = X_train_auto.shape[1] if "auto FE" in best_model_name else X_train_base.shape[1]
    test_metrics["n_train_samples"] = len(X_train_base)
    test_metrics["n_test_samples"] = len(X_test_base)
    test_metrics["model_type"] = best_model_name
    
    # ← ИСПОЛЬЗУЕМ ExperimentManager для сохранения
    manager.save_metrics(test_metrics, config, experiment_id, horizon_days)
    manager.save_config(config, experiment_id, horizon_days)
    manager.save_json_report(test_metrics, config, experiment_id, horizon_days)
    
    # ← НОВОЕ: Создаём детальный график сравнения прогноза с реальностью
    print(f"\n   Создание графика сравнения прогноза с фактическими значениями...")
    try:
        # Получаем временные метки для тестовой выборки
        timestamps = y_test.index if hasattr(y_test, 'index') else None
        
        # Сохраняем график сравнения
        manager.save_forecast_comparison(
            y_true=y_test.values,
            y_pred=best_pred if best_pred is not None else y_test.mean(),
            timestamps=timestamps,
            horizon=horizon_days,
            model_name=best_model_name,
            mae=test_metrics.get("MAE (°C)"),
            rmse=test_metrics.get("RMSE (°C)"),
            r2=test_metrics.get("R²"),
            max_points=300,  # Показываем последние 300 точек для читаемости
            save_name=f"forecast_h{horizon_days}d_{best_model_name.replace(' ', '_').replace('(', '').replace(')', '')}"
        )
    except Exception as e:
        print(f"    ⚠️  Ошибка при создании графика сравнения: {e}")
    
    if "Gradient Boosting" in best_model_name or "Random Forest" in best_model_name:
        print(f"\n   Важность по группам признаков:")
        
        model_for_importance = gb_auto if "auto FE" in best_model_name else gb_base
        X_for_importance = X_train_auto if "auto FE" in best_model_name else X_train_base
        
        feature_importance = pd.DataFrame({
            "feature": X_for_importance.columns,
            "importance": model_for_importance.feature_importances_
        }).sort_values("importance", ascending=False)
        
        groups = {
            "Core (лаги)": ["lag_"],
            "Оконные преобразования": ["window"],
            "Спектральные методы": ["stl", "dwt"],
            "Временное кодирование": ["time."],
            "Календарные признаки": ["calendar", "is_weekend", "part_of_day", "season", "month"]
        }
        
        group_importance = {}
        for group_name, keywords in groups.items():
            mask = pd.Series(False, index=feature_importance.index)
            for kw in keywords:
                mask |= feature_importance["feature"].str.contains(kw, case=False, na=False)
            group_importance[group_name] = feature_importance.loc[mask, "importance"].sum() * 100
        
        for group_name, importance in sorted(group_importance.items(), key=lambda x: x[1], reverse=True):
            bar = "█" * int(importance / 2)
            print(f"   {group_name:25s} | {importance:5.1f}% {bar}")
    
    return test_metrics, naive_mae, best_mae, results_comparison


def plot_horizon_comparison(all_results, manager):
    """
    Визуализация сравнения результатов по горизонтам и сохранение через ExperimentManager.
    """
    print(f"\n{'='*80}")
    print("СРАВНЕНИЕ РЕЗУЛЬТАТОВ ПО ГОРИЗОНТАМ")
    print(f"{'='*80}")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    horizons = [r["horizon"] for r in all_results]
    naive_maes = [r["naive_mae"] for r in all_results]
    auto_maes = [r["best_mae"] for r in all_results]
    r2_scores = [r["metrics"]["R²"] for r in all_results]
    improvements = [((r["naive_mae"] - r["best_mae"]) / r["naive_mae"]) * 100 for r in all_results]
    
    ax = axes[0, 0]
    x = np.arange(len(horizons))
    width = 0.35
    ax.bar(x - width/2, naive_maes, width, label='Наивный', color='gray', alpha=0.7)
    ax.bar(x + width/2, auto_maes, width, label='Лучшая модель', color='steelblue', alpha=0.7)
    ax.set_xlabel('Горизонт прогноза (дни)')
    ax.set_ylabel('MAE (°C)')
    ax.set_title('Сравнение MAE по горизонтам')
    ax.set_xticks(x)
    ax.set_xticklabels(horizons)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    ax = axes[0, 1]
    colors = ['green' if imp > 0 else 'red' for imp in improvements]
    ax.bar(horizons, improvements, color=colors, alpha=0.7)
    ax.axhline(0, color='black', linewidth=1)
    ax.set_xlabel('Горизонт прогноза (дни)')
    ax.set_ylabel('Улучшение относительно наивного (%)')
    ax.set_title('Улучшение лучшей модели относительно наивного прогноза')
    ax.grid(True, alpha=0.3, axis='y')
    
    ax = axes[1, 0]
    ax.plot(horizons, r2_scores, marker='o', linewidth=2, markersize=8, color='darkred')
    ax.set_xlabel('Горизонт прогноза (дни)')
    ax.set_ylabel('R²')
    ax.set_title('Доля объяснённой дисперсии (R²)')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    ax = axes[1, 1]
    n_features = [r["metrics"]["n_features"] for r in all_results]
    ax.plot(horizons, n_features, marker='s', linewidth=2, markersize=8, color='darkgreen')
    ax.set_xlabel('Горизонт прогноза (дни)')
    ax.set_ylabel('Количество признаков')
    ax.set_title('Количество сгенерированных признаков (auto FE)')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Сравнение результатов прогнозирования по горизонтам', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # ← ИСПОЛЬЗУЕМ ExperimentManager для сохранения
    manager.save_plot(fig, "horizon_comparison.png")
    print("   График сохранён через ExperimentManager")
    plt.close()
    
    print(f"\n{'='*80}")
    print("ИТОГОВАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
    print(f"{'='*80}")
    print(f"{'Горизонт':<12} {'Наивный MAE':<15} {'Лучшая MAE':<15} {'Улучшение':<12} {'R²':<10}")
    print(f"{'-'*80}")
    
    for r in all_results:
        improvement = ((r["naive_mae"] - r["best_mae"]) / r["naive_mae"]) * 100 if r["naive_mae"] > 0 else 0
        print(f"{r['horizon']:<12} {r['naive_mae']:>10.2f} °C   {r['best_mae']:>10.2f} °C   {improvement:>10.1f}%   {r['metrics']['R²']:>8.4f}")
    
    print(f"{'-'*80}")


def main():
    experiment_start_time = datetime.now()
    
    print("=" * 80)
    print("ПРОГНОЗИРОВАНИЕ МИНИМАЛЬНЫХ ТЕМПЕРАТУР (MULTI-HORIZON)")
    print("=" * 80)
    
    # ← ИНИЦИАЛИЗИРУЕМ ExperimentManager
    manager = ExperimentManager(experiment_type="temperature")
    experiment_id = manager.get_experiment_id(EXPERIMENT_CONFIG)
    
    print(f"ID эксперимента: {experiment_id}")
    print(f"Время запуска: {experiment_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Горизонты прогнозирования: {EXPERIMENT_CONFIG['forecast_horizons']} дней")
    print(f"Результаты сохраняются в: {manager.full_dir}/")
    
    if not HAS_TENSORFLOW:
        print("   ⚠️  TensorFlow не установлен. LSTM модели будут пропущены.")
    
    print("\n1. Загрузка данных минимальных суточных температур...")
    try:
        df = load_temperature_data()
    except Exception as e:
        print(f"Ошибка при загрузке данных: {e}")
        return
    
    print("\n2. Анализ структуры временного ряда...")
    print(f"   Частота дискретизации: {pd.infer_freq(df.index)}")
    print(f"   Количество пропусков: {df['min_temperature'].isna().sum()}")
    print(f"   Минимальная температура: {df['min_temperature'].min():.2f} °C")
    print(f"   Максимальная температура: {df['min_temperature'].max():.2f} °C")
    print(f"   Средняя температура: {df['min_temperature'].mean():.2f} °C")
    
    print("\n3. Визуализация ключевых паттернов температур...")
    plot_temperature_patterns(df, manager, title="Паттерны минимальных температур")
    
    print("\n4. Прогнозирование на горизонтах...")
    
    all_results = []
    
    for horizon in EXPERIMENT_CONFIG["forecast_horizons"]:
        test_metrics, naive_mae, best_mae, model_comparison = forecast_on_horizon(
            df, horizon, EXPERIMENT_CONFIG, experiment_id, manager
        )
        all_results.append({
            "horizon": horizon,
            "metrics": test_metrics,
            "naive_mae": naive_mae,
            "best_mae": best_mae,
            "model_comparison": model_comparison
        })
    
    print("\n5. Сравнение результатов по горизонтам...")
    plot_horizon_comparison(all_results, manager)
    
    experiment_end_time = datetime.now()
    experiment_duration = (experiment_end_time - experiment_start_time).total_seconds()
    
    print("\n" + "=" * 80)
    print("ЭКСПЕРИМЕНТ ЗАВЕРШЕН")
    print("=" * 80)
    
    hours = int(experiment_duration // 3600)
    minutes = int((experiment_duration % 3600) // 60)
    seconds = int(experiment_duration % 60)
    
    print(f"\n⏱️  Длительность эксперимента: {hours}ч {minutes}м {seconds}с ({experiment_duration:.1f} секунд)")
    
    print("\nСгенерированные файлы:")
    print(f"  • {manager.full_dir}/temperature_patterns.png : анализ паттернов температур")
    print(f"  • {manager.full_dir}/horizon_comparison.png : сравнение по горизонтам")
    print(f"  • {manager.full_dir}/plots/ : детальные графики прогнозов")
    print(f"  • {manager.global_metrics_file} : глобальная история метрик")
    print(f"  • {manager.full_dir}/metrics_history.csv : локальная история метрик")
    
    # ← ИСПОЛЬЗУЕМ ExperimentManager для сохранения summary
    summary_path = manager.save_summary(all_results, experiment_id, experiment_start_time, experiment_end_time)
    print(f"  • {summary_path} : итоговый отчёт")
    
    print("\n" + "=" * 80)
    print("КЛЮЧЕВОЙ ИНСАЙТ")
    print("=" * 80)
    
    short_horizon = all_results[0]
    long_horizon = all_results[-1]
    
    short_imp = ((short_horizon["naive_mae"] - short_horizon["best_mae"]) / short_horizon["naive_mae"]) * 100
    long_imp = ((long_horizon["naive_mae"] - long_horizon["best_mae"]) / long_horizon["naive_mae"]) * 100
    
    print(f"\nНа коротком горизонте ({short_horizon['horizon']} день):")
    print(f"  • Улучшение лучшей модели: {short_imp:.1f}%")
    
    print(f"\nНа длинном горизонте ({long_horizon['horizon']} дней):")
    print(f"  • Улучшение лучшей модели: {long_imp:.1f}%")
    
    print("\n" + "=" * 80)
    print("ПРИМЕР ЗАВЕРШЕН")
    print("=" * 80)


if __name__ == "__main__":
    try:
        import matplotlib
        import seaborn
    except ImportError:
        print("WARNING: matplotlib или seaborn не установлены. Визуализация будет ограничена.")
        print("Установите через: pip install matplotlib seaborn")
    
    main()