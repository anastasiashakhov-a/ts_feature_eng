# examples/energy_forecasting.py
"""
Пример прогнозирования энергопотребления на данных из Марокко (MULTI-HORIZON).
Демонстрирует применение автоматической инженерии признаков для реальной задачи:
1. Загрузка и анализ данных энергопотребления Марокко (зона 1)
2. Обнаружение сезонных паттернов (суточных, недельных, годовых)
3. Генерация календарных признаков с учетом особенностей Марокко
4. Автоматический подбор оптимальных методов инженерии признаков
5. Прогнозирование на РАЗНЫЕ горизонты (1, 7, 30 ДНЕЙ)
6. Сравнение с базовыми моделями и интерпретация результатов
7. Гибридная инженерия признаков (обязательные лаги + адаптивные преобразования)
8. Сравнение моделей: базовые признаки vs Auto FE для всех алгоритмов + LSTM
9. Отслеживание времени выполнения эксперимента
10. Детальная визуализация прогнозов с сравнением фактических и предсказанных значений

Оптимизировано для скорости:
- Уменьшено количество итераций байесовской оптимизации
- Отключена SHAP-фильтрация для скорости
"""
import os
import json
import warnings
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
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
    "n_calls": 10,
    "n_initial_points": 3,
    "selection_threshold": 0.25,
    "variance_threshold": 0.01,
    "shap_selection": False,
    "n_estimators": 100,
    "max_depth": 4,
    "learning_rate": 0.1,
    "random_state": 42,
    "train_test_split": 0.8,
    "forecast_horizons": [1, 7, 30],  # ← ГОРИЗОНТЫ В ДНЯХ
    "aggregation_freq": "15min",
    # Параметры LSTM
    "lstm_timesteps": 96,
    "lstm_units": 50,
    "lstm_epochs": 15,
    "lstm_batch_size": 32,
    "use_lstm": False,  # ← Отключено по умолчанию для скорости
}

# Константы для конвертации дней в интервалы
INTERVALS_PER_HOUR = 4
INTERVALS_PER_DAY = 24 * INTERVALS_PER_HOUR


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
    if not time_cols:
        raise ValueError("Не найден столбец с временными метками в данных")
    time_col = time_cols[0]
    print(f"  Найден столбец временных меток: '{time_col}'")
    
    power_cols = [col for col in df.columns if any(kw in col.lower() for kw in ['power', 'consum', 'load', 'energy'])]
    if not power_cols:
        raise ValueError("Не найден столбец с данными энергопотребления")
    power_col = power_cols[0]
    print(f"  Найден столбец потребления: '{power_col}'")
    
    try:
        df[time_col] = pd.to_datetime(df[time_col])
    except Exception as e:
        raise ValueError(f"Ошибка преобразования временных меток: {e}")
    
    result = pd.DataFrame({
        "timestamp": df[time_col],
        "power_consumption": df[power_col]
    })
    result = result.sort_values("timestamp").drop_duplicates(subset="timestamp").reset_index(drop=True)
    result = result.set_index("timestamp")
    
    initial_len = len(result)
    result = result.dropna(subset=["power_consumption"])
    final_len = len(result)
    print(f"  Удалено {initial_len - final_len} наблюдений с пропусками")
    print(f"  Итоговый размер данных: {len(result)} наблюдений")
    print(f"  Диапазон дат: {result.index.min()} — {result.index.max()}")
    
    return result


def detect_ramadan_periods(df, years=None):
    """Определяет периоды Рамадана для создания календарных признаков."""
    if years is None:
        years = sorted(df.index.year.unique())
    
    ramadan_approx_dates = {
        2018: ("2018-05-16", "2018-06-14"),
        2019: ("2019-05-05", "2019-06-04"),
        2020: ("2020-04-23", "2020-05-23"),
        2021: ("2021-04-12", "2021-05-12"),
        2022: ("2022-04-02", "2022-05-01"),
        2023: ("2023-03-22", "2023-04-21"),
    }
    
    ramadan_mask = pd.Series(False, index=df.index)
    for year in years:
        if year in ramadan_approx_dates:
            start, end = ramadan_approx_dates[year]
            mask = (df.index >= start) & (df.index <= end)
            ramadan_mask[mask] = True
            print(f"  Год {year}: Рамадан приблизительно {start} — {end}")
        else:
            print(f"  Год {year}: данные о Рамадане недоступны (пропущен)")
    
    return ramadan_mask


def create_ramadan_features(df):
    """Создаёт признаки, связанные с Рамаданом."""
    print("\nСоздание признаков Рамадана (демонстрационная реализация)...")
    ramadan_mask = detect_ramadan_periods(df)
    
    df_ramadan = pd.DataFrame(index=df.index)
    df_ramadan["is_ramadan"] = ramadan_mask.astype(int)
    df_ramadan["days_until_ramadan"] = np.nan
    df_ramadan["days_since_ramadan"] = np.nan
    
    ramadan_days = df_ramadan[df_ramadan["is_ramadan"] == 1].index
    if len(ramadan_days) > 0:
        for idx in df_ramadan.index:
            if idx in ramadan_days:
                df_ramadan.loc[idx, "days_until_ramadan"] = 0
                df_ramadan.loc[idx, "days_since_ramadan"] = 0
            else:
                future_ramadan = ramadan_days[ramadan_days > idx]
                if len(future_ramadan) > 0:
                    days_until = (future_ramadan[0] - idx).days
                    df_ramadan.loc[idx, "days_until_ramadan"] = min(days_until, 30)
                
                past_ramadan = ramadan_days[ramadan_days < idx]
                if len(past_ramadan) > 0:
                    days_since = (idx - past_ramadan[-1]).days
                    df_ramadan.loc[idx, "days_since_ramadan"] = min(days_since, 30)
    
    df_ramadan["days_until_ramadan"] = df_ramadan["days_until_ramadan"].fillna(30)
    df_ramadan["days_since_ramadan"] = df_ramadan["days_since_ramadan"].fillna(30)
    
    print(f"  Создано признаков Рамадана: {len(df_ramadan.columns)}")
    print(f"  Наблюдений в периоде Рамадана: {df_ramadan['is_ramadan'].sum()} ({df_ramadan['is_ramadan'].mean()*100:.1f}%)")
    
    return df_ramadan


def plot_consumption_patterns(df, manager, title="Паттерны энергопотребления"):
    """Визуализирует паттерны энергопотребления и сохраняет через ExperimentManager."""
    print(f"\nВизуализация паттернов потребления: {title}")
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    
    ax = axes[0, 0]
    df["power_consumption"].plot(ax=ax, color='blue', linewidth=1, alpha=0.7)
    ax.set_title("Общий тренд энергопотребления", fontsize=12, fontweight='bold')
    ax.set_ylabel("Потребление (МВт)", fontsize=10)
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    hourly_avg = df.groupby(df.index.hour)["power_consumption"].mean()
    hourly_std = df.groupby(df.index.hour)["power_consumption"].std()
    ax.plot(hourly_avg.index, hourly_avg.values, color='darkgreen', linewidth=2, label='Среднее')
    ax.fill_between(
        hourly_avg.index,
        hourly_avg.values - hourly_std.values,
        hourly_avg.values + hourly_std.values,
        alpha=0.3, color='green', label='±1 std'
    )
    ax.set_title("Суточной паттерн потребления", fontsize=12, fontweight='bold')
    ax.set_xlabel("Час суток", fontsize=10)
    ax.set_ylabel("Потребление (МВт)", fontsize=10)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 0]
    weekday_avg = df.groupby(df.index.dayofweek)["power_consumption"].mean()
    weekday_labels = ['Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб', 'Вс']
    ax.bar(weekday_labels, weekday_avg.values, color='steelblue', alpha=0.8)
    ax.set_title("Недельный паттерн потребления", fontsize=12, fontweight='bold')
    ax.set_ylabel("Среднее потребление (МВт)", fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    ax = axes[1, 1]
    monthly_avg = df.groupby(df.index.month)["power_consumption"].mean()
    month_labels = ['Янв', 'Фев', 'Мар', 'Апр', 'Май', 'Июн', 'Июл', 'Авг', 'Сен', 'Окт', 'Ноя', 'Дек']
    ax.plot(month_labels, monthly_avg.values, marker='o', color='darkred', linewidth=2, markersize=6)
    ax.set_title("Сезонный паттерн потребления", fontsize=12, fontweight='bold')
    ax.set_ylabel("Среднее потребление (МВт)", fontsize=10)
    ax.grid(True, alpha=0.3)
    
    ax = axes[2, 0]
    sns.histplot(df["power_consumption"], bins=50, kde=True, ax=ax, color='purple')
    ax.axvline(df["power_consumption"].mean(), color='red', linestyle='--', 
               label=f'Среднее: {df["power_consumption"].mean():.0f} МВт')
    ax.axvline(df["power_consumption"].median(), color='orange', linestyle='--', 
               label=f'Медиана: {df["power_consumption"].median():.0f} МВт')
    ax.set_title("Распределение энергопотребления", fontsize=12, fontweight='bold')
    ax.set_xlabel("Потребление (МВт)", fontsize=10)
    ax.set_ylabel("Частота", fontsize=10)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[2, 1]
    from statsmodels.graphics.tsaplots import plot_acf
    max_lag = min(168 * INTERVALS_PER_HOUR, len(df) // 4)
    plot_acf(df["power_consumption"].dropna(), lags=max_lag, ax=ax, alpha=0.05)
    ax.set_title(f"Автокорреляция (до {max_lag // INTERVALS_PER_HOUR} часов)", fontsize=12, fontweight='bold')
    ax.set_xlabel("Лаг (15-мин интервалы)", fontsize=10)
    ax.set_ylabel("ACF", fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # ← ИСПОЛЬЗУЕМ ExperimentManager для сохранения
    manager.save_plot(fig, "consumption_patterns.png")
    print("  График сохранён через ExperimentManager")
    plt.close()


def evaluate_forecast(y_true, y_pred, horizon_days=1):
    """Оценивает качество прогноза с интерпретацией для энергосистемы."""
    # Если y_pred — скаляр, создаём массив той же длины
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
    print(f"\nОценка качества прогноза (горизонт: {horizon_days} {day_word}):")
    print("-" * 65)
    print(f"{'Метрика':<25} {'Значение':<15} {'Интерпретация'}")
    print("-" * 65)
    print(f"{'MAE':<25} {mae:>10.2f} МВт  {'Средняя ошибка прогноза'}")
    print(f"{'RMSE':<25} {rmse:>10.2f} МВт  {'Чувствительность к крупным ошибкам'}")
    print(f"{'MAPE':<25} {mape:>10.2f} %    {'Относительная ошибка'}")
    print(f"{'R²':<25} {r2:>10.4f}       {'Доля объясненной дисперсии'}")
    print(f"{'Ошибка от пика':<25} {mean_abs_error_pct:>10.2f} %    {'Критичность для балансировки'}")
    print("-" * 65)
    
    if mean_abs_error_pct < 2.0:
        reliability = "ОЧЕНЬ ВЫСОКАЯ"
        recommendation = "Подходит для автоматической балансировки"
    elif mean_abs_error_pct < 5.0:
        reliability = "ВЫСОКАЯ"
        recommendation = "Подходит для планирования генерации"
    elif mean_abs_error_pct < 10.0:
        reliability = "СРЕДНЯЯ"
        recommendation = "Требуется резерв мощности 10-15%"
    else:
        reliability = "НИЗКАЯ"
        recommendation = "Требуется ручная коррекция и большой резерв"
    
    print(f"\nНадежность прогноза: {reliability}")
    print(f"Рекомендация: {recommendation}")
    
    return metrics


def prepare_lstm_data(X, y, timesteps=96, train_split=0.8):
    """Подготавливает данные для LSTM: создаёт последовательности [samples, timesteps, features]."""
    combined = pd.concat([X, y.to_frame()], axis=1).dropna()
    if len(combined) <= timesteps:
        raise ValueError(f"Недостаточно данных для LSTM: {len(combined)} <= {timesteps}")
    
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
    X_train, X_test = X_lstm[:split_idx], X_lstm[split_idx:]
    y_train, y_test = y_lstm[:split_idx], y_lstm[split_idx:]
    
    return X_train, X_test, y_train, y_test, scaler_y


def train_lstm_model(X_train, y_train, X_test, config, verbose=0):
    """Обучает LSTM модель и делает прогноз."""
    if not HAS_TENSORFLOW:
        return None, None
    
    timesteps = config.get("lstm_timesteps", 96)
    units = config.get("lstm_units", 50)
    epochs = config.get("lstm_epochs", 15)
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
    """Прогнозирование на заданный горизонт в ДНЯХ."""
    horizon_intervals = horizon_days * INTERVALS_PER_DAY
    day_word = "день" if horizon_days == 1 else "дня" if horizon_days in [2,3,4] else "дней"
    
    print(f"\n{'='*80}")
    print(f"ГОРИЗОНТ ПРОГНОЗА: {horizon_days} {day_word} ({horizon_intervals} интервалов по 15 мин)")
    print(f"{'='*80}")
    
    y = df["power_consumption"].shift(-horizon_intervals)
    ramadan_features = create_ramadan_features(df)
    X = pd.DataFrame({"power_consumption": df["power_consumption"]})
    X = pd.concat([X, ramadan_features], axis=1)
    
    valid_mask = ~y.isna()
    X = X[valid_mask]
    y = y[valid_mask]
    
    split_idx = int(len(X) * config["train_test_split"])
    X_train_base, X_test_base = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"\nОбучающая выборка: {len(X_train_base)} наблюдений")
    print(f"   Тестовая выборка: {len(X_test_base)} наблюдений")
    
    print(f"\nАвтоматическая инженерия признаков...")
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
    
    print(f"\nСравнение моделей...")
    results_comparison = []
    
    # 1. Наивный прогноз
    naive_pred = y_test.shift(horizon_intervals).fillna(y_test.mean())
    naive_mae = mean_absolute_error(y_test, naive_pred)
    results_comparison.append(("Наивный прогноз", naive_mae, None))
    
    # 2. Сезонный наивный (24 часа = 96 интервалов назад)
    seasonal_naive_pred = y_test.shift(INTERVALS_PER_DAY).fillna(y_test.mean())
    seasonal_naive_mae = mean_absolute_error(y_test, seasonal_naive_pred)
    results_comparison.append(("Сезонный наивный (24ч)", seasonal_naive_mae, None))
    
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
    
    # 7. LSTM - базовые признаки
    if HAS_TENSORFLOW and config.get("use_lstm", False):
        print(f"   Обучение LSTM (базовые)...")
        try:
            X_lstm_base = df[["power_consumption"]].copy()
            X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm, scaler_y = prepare_lstm_data(
                X_lstm_base, df["power_consumption"].shift(-horizon_intervals),
                timesteps=config["lstm_timesteps"], train_split=config["train_test_split"]
            )
            lstm_model, lstm_pred_scaled = train_lstm_model(
                X_train_lstm, y_train_lstm, X_test_lstm, config, verbose=0
            )
            if lstm_pred_scaled is not None and len(lstm_pred_scaled) > 0:
                lstm_pred = scaler_y.inverse_transform(lstm_pred_scaled.reshape(-1, 1)).ravel()
                if np.any(np.isnan(lstm_pred)) or np.any(np.isinf(lstm_pred)):
                    print(f"    ⚠️  LSTM прогноз содержит NaN/Inf, пропускаем")
                    results_comparison.append(("LSTM (базовые)", np.nan, None))
                else:
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
            if len(X_train_lstm_auto) < 10:
                raise ValueError("Слишком мало данных для обучения LSTM")
            
            lstm_auto_model, lstm_auto_pred_scaled = train_lstm_model(
                X_train_lstm_auto, y_train_lstm_auto, X_test_lstm_auto_arr, config, verbose=0
            )
            if lstm_auto_pred_scaled is not None and len(lstm_auto_pred_scaled) > 0:
                lstm_auto_pred = scaler_y_auto.inverse_transform(lstm_auto_pred_scaled.reshape(-1, 1)).ravel()
                if np.any(np.isnan(lstm_auto_pred)) or np.any(np.isinf(lstm_auto_pred)):
                    print(f"    ⚠️  LSTM (auto FE) прогноз содержит NaN/Inf, пропускаем")
                    results_comparison.append(("LSTM (auto FE)", np.nan, None))
                else:
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
        if not HAS_TENSORFLOW:
            print(f"    ⚠️  LSTM пропущен: TensorFlow не установлен")
        else:
            print(f"    ⚠️  LSTM отключён в конфиге (use_lstm=False)")
    
    print(f"\n{'='*70}")
    print(f"   Сравнение моделей по MAE (меньше — лучше):")
    print(f"   {'='*70}")
    best_mae = min(r[1] for r in results_comparison if not np.isnan(r[1]))
    for model_name, mae_value, _ in results_comparison:
        if np.isnan(mae_value):
            print(f"   {model_name:35s} | {'N/A':>6s}    | пропущен")
        else:
            star = " ← ЛУЧШАЯ" if mae_value == best_mae else ""
            improvement = ((naive_mae - mae_value) / naive_mae) * 100 if naive_mae > 0 else 0
            print(f"   {model_name:35s} | {mae_value:6.2f} МВт | улучшение: {improvement:6.1f}%{star}")
    print(f"   {'='*70}")
    
    best_result = min((r for r in results_comparison if not np.isnan(r[1])), key=lambda x: x[1])
    best_model_name, best_mae, best_pred = best_result
    
    if best_pred is None:
        print(f"   ⚠️  Лучшая модель ({best_model_name}) не вернула прогноз, используем наивный")
        best_pred = y_test.shift(horizon_intervals).fillna(y_test.mean()).values
    
    test_metrics = evaluate_forecast(y_test, best_pred, horizon_days=horizon_days)
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
            mae=test_metrics.get("MAE (МВт)"),
            rmse=test_metrics.get("RMSE (МВт)"),
            r2=test_metrics.get("R²"),
            max_points=300,  # Показываем последние 300 точек для читаемости
            save_name=f"forecast_h{horizon_days}d_{best_model_name.replace(' ', '_').replace('(', '').replace(')', '')}"
        )
    except Exception as e:
        print(f"    ⚠️  Ошибка при создании графика сравнения: {e}")
    
    if "Gradient Boosting" in best_model_name or "Random Forest" in best_model_name:
        print(f"\nВажность по группам признаков:")
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
            "Календарные признаки": ["calendar", "is_weekend", "part_of_day", "season", "month", "ramadan"]
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
    """Визуализация сравнения результатов по горизонтам."""
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
    ax.set_ylabel('MAE (МВт)')
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
        print(f"{r['horizon']:<12} {r['naive_mae']:>10.2f} МВт   {r['best_mae']:>10.2f} МВт   {improvement:>10.1f}%   {r['metrics']['R²']:>8.4f}")
    print(f"{'-'*80}")


def main():
    experiment_start_time = datetime.now()
    
    print("=" * 80)
    print("ПРОГНОЗИРОВАНИЕ ЭНЕРГОПОТРЕБЛЕНИЯ НА ДАННЫХ ИЗ МАРОККО (MULTI-HORIZON)")
    print("=" * 80)
    
    # ← ИНИЦИАЛИЗИРУЕМ ExperimentManager
    manager = ExperimentManager(experiment_type="energy")
    experiment_id = manager.get_experiment_id(EXPERIMENT_CONFIG)
    
    print(f"ID эксперимента: {experiment_id}")
    print(f"Время запуска: {experiment_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Горизонты прогнозирования: {EXPERIMENT_CONFIG['forecast_horizons']} дней")
    print(f"Частота данных: {EXPERIMENT_CONFIG['aggregation_freq']}")
    print(f"Результаты сохраняются в: {manager.full_dir}/")
    
    if not HAS_TENSORFLOW:
        print("    ⚠️  TensorFlow не установлен. LSTM модели будут пропущены.")
    if not EXPERIMENT_CONFIG.get("use_lstm", False):
        print("    ⚠️  LSTM отключён в конфиге (use_lstm=False)")
    
    print("\n1. Загрузка данных энергопотребления Марокко...")
    try:
        df = load_morocco_energy_data()
    except Exception as e:
        print(f"Ошибка при загрузке данных: {e}")
        print("\nСовет: Убедитесь, что файл находится в одной из следующих директорий:")
        print("  - data/morocco zone 1 - powerconsumption_resampled (1).csv")
        print("  - ~/ts_feature_eng/data/...")
        return
    
    print("\n2. Анализ структуры временного ряда...")
    print(f"   Частота дискретизации: {pd.infer_freq(df.index)}")
    print(f"   Количество пропусков: {df['power_consumption'].isna().sum()}")
    print(f"   Минимальное потребление: {df['power_consumption'].min():.2f} МВт")
    print(f"   Максимальное потребление: {df['power_consumption'].max():.2f} МВт")
    print(f"   Среднее потребление: {df['power_consumption'].mean():.2f} МВт")
    
    print("\n3. Визуализация ключевых паттернов потребления...")
    plot_consumption_patterns(df, manager, title="Паттерны энергопотребления в Марокко (Зона 1)")
    
    print("\n4. Прогнозирование на разных горизонтах...")
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
    print(f"\n⏱  Длительность эксперимента: {hours}ч {minutes}м {seconds}с ({experiment_duration:.1f} секунд)")
    
    print("\nСгенерированные файлы:")
    print(f"  • {manager.full_dir}/consumption_patterns.png : анализ паттернов потребления")
    print(f"  • {manager.full_dir}/horizon_comparison.png   : сравнение по горизонтам")
    print(f"  • {manager.full_dir}/plots/ : детальные графики прогнозов")
    print(f"  • {manager.global_metrics_file}     : глобальная история метрик")
    print(f"  • {manager.full_dir}/metrics_history.csv     : локальная история метрик")
    
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