# examples/temperature_forecasting.py 

"""
Пример прогнозирования минимальных суточных температур.

Демонстрирует применение автоматической инженерии признаков для реальной задачи:
1. Загрузка и анализ данных минимальных суточных температур
2. Обнаружение сезонных паттернов (суточных, недельных, годовых)
3. Генерация календарных признаков с учетом особенностей временного ряда:
   - Суточные циклы температур
   - Недельные паттерны
   - Сезонные колебания (лето vs зима)
4. Автоматический подбор оптимальных методов инженерии признаков
5. Прогнозирование на разные горизонты (1 день, 7 дней)
6. Сравнение с базовыми моделями и интерпретация результатов
7. Гибридная инженерия признаков (обязательные лаги + адаптивные преобразования)
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
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

from ts_feature_eng import AutoFeatureEngineer
from ts_feature_eng.transformers.time_encoding import CalendarFeaturesTransformer
from ts_feature_eng.transformers.lag import LagTransformer


# ============================================================================
# КОНФИГУРАЦИЯ ЭКСПЕРИМЕНТА
# ============================================================================
EXPERIMENT_CONFIG = {
    "sample_ratio": 1.0,          # Доля данных (1.0 = 100%)
    "n_calls": 15,                # Итерации байесовской оптимизации
    "n_initial_points": 3,        # Начальные точки оптимизации
    "selection_threshold": 0.25,  # Порог отбора признаков
    "variance_threshold": 0.01,   # Порог дисперсии
    "shap_selection": True,      # Отключить SHAP для скорости
    "n_estimators": 100,          # Деревья в модели
    "max_depth": 6,               # Глубина деревьев
    "learning_rate": 0.1,         # Скорость обучения
    "random_state": 42,           # Сид для воспроизводимости
    "train_test_split": 0.8,      # Доля обучающей выборки
    "forecast_horizons": [1, 7],  # Горизонты прогнозирования (дни)
}


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
        "mae_c": metrics.get("MAE (°C)", np.nan),
        "rmse_c": metrics.get("RMSE (°C)", np.nan),
        "mape_pct": metrics.get("MAPE (%)", np.nan),
        "r2": metrics.get("R²", np.nan),
        "error_from_range_pct": metrics.get("Ошибка от диапазона (%)", np.nan),
        "temp_range_c": metrics.get("Диапазон температур (°C)", np.nan),
        "n_features": metrics.get("n_features", 0),
        "n_train_samples": metrics.get("n_train_samples", 0),
        "n_test_samples": metrics.get("n_test_samples", 0),
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
        "metrics": metrics,
    }
    
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False)
    
    print(f"   Результаты сохранены в {results_dir}/")
    print(f"   ID эксперимента: {experiment_id}")
    
    return metrics_file, params_file, report_file


def load_temperature_data(data_path=None):
    """
    Загружает данные минимальных суточных температур.
    
    Параметры
    ----------
    data_path : str, опционально
        Путь к CSV файлу. Если не указан, ищет в стандартных местах.
    
    Возвращает
    ----------
    df : pd.DataFrame
        DataFrame с колонками 'timestamp' и 'min_temperature'.
    """
    # Определяем путь к данным
    if data_path is None:
        # Ищем в нескольких возможных местах
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
    
    # Загрузка данных
    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        raise RuntimeError(f"Ошибка при загрузке файла: {e}")
    
    # Очистка имен столбцов от лишних пробелов
    df.columns = df.columns.str.strip()
    
    # Определение столбцов с временными метками и температурой
    time_cols = [col for col in df.columns if any(kw in col.lower() for kw in ['time', 'date', 'timestamp', 'dt'])]
    temp_cols = [col for col in df.columns if any(kw in col.lower() for kw in ['temp', 'min'])]
    
    if not time_cols or not temp_cols:
        raise ValueError("Не найдены необходимые столбцы в данных")
    
    time_col = time_cols[0]
    temp_col = temp_cols[0]
    print(f"  Найден столбец временных меток: '{time_col}'")
    print(f"  Найден столбец температуры: '{temp_col}'")
    
    # Преобразование временных меток
    try:
        df[time_col] = pd.to_datetime(df[time_col])
    except Exception as e:
        raise ValueError(f"Ошибка преобразования временных меток: {e}")
    
    # Очистка некорректных значений в температуре
    try:
        # Удаляем строки с некорректными значениями (например, "?0.2")
        mask_invalid = df[temp_col].astype(str).str.contains(r'[?]', na=False)
        if mask_invalid.any():
            print(f"  Обнаружено {mask_invalid.sum()} некорректных значений в температуре. Удаляем...")
            df = df[~mask_invalid]
        
        # Заменяем запятые на точки и преобразуем в float
        df[temp_col] = df[temp_col].astype(str).str.replace(',', '.', regex=True).astype(float)
    except Exception as e:
        raise ValueError(f"Ошибка преобразования температуры в числовой формат: {e}")
    
    # Создание итогового DataFrame
    result = pd.DataFrame({
        "timestamp": df[time_col],
        "min_temperature": df[temp_col]
    })
    
    # Сортировка по времени и удаление дубликатов
    result = result.sort_values("timestamp").drop_duplicates(subset="timestamp").reset_index(drop=True)
    
    # Установка временного индекса
    result = result.set_index("timestamp")
    
    # Удаление пропусков в температуре
    initial_len = len(result)
    result = result.dropna(subset=["min_temperature"])
    final_len = len(result)
    
    print(f"  Удалено {initial_len - final_len} наблюдений с пропусками")
    print(f"  Итоговый размер данных: {len(result)} наблюдений")
    print(f"  Диапазон дат: {result.index.min()} — {result.index.max()}")
    
    return result


def plot_temperature_patterns(df, title="Паттерны минимальных температур"):
    """
    Визуализирует ключевые паттерны минимальных температур.
    
    Параметры
    ----------
    df : pd.DataFrame
        Данные с колонкой 'min_temperature'.
    title : str
        Заголовок графика.
    """
    print(f"\nВизуализация паттернов температур: {title}")
    
    # Создаем подграфики
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
    plot_acf(df["min_temperature"].dropna(), lags=365, ax=ax, alpha=0.05)  # 365 дней = 1 год
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
    plt.savefig("temperature_patterns.png", dpi=150, bbox_inches='tight')
    print("  График сохранен как 'temperature_patterns.png'")
    plt.close()


def evaluate_forecast(y_true, y_pred, horizon_days=1):
    """
    Оценивает качество прогноза с интерпретацией для задачи прогнозирования температуры.
    
    Параметры
    ----------
    y_true : array-like
        Фактические значения.
    y_pred : array-like
        Предсказанные значения.
    horizon_days : int
        Горизонт прогнозирования в днях.
    
    Возвращает
    ----------
    metrics : dict
        Словарь с метриками качества.
    """
    # Базовые метрики
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # MAPE с защитой
    mask = np.abs(y_true) > 1e-6
    if np.sum(mask) > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = np.nan
    
    # Средняя ошибка в процентах от диапазона температур
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
    
    # Интерпретация для задачи прогнозирования температуры
    print(f"\nОценка качества прогноза (горизонт: {horizon_days} день{'а' if horizon_days in [2,3,4] else 'ей'}):")
    print("-" * 65)
    print(f"{'Метрика':<25} {'Значение':<15} {'Интерпретация'}")
    print("-" * 65)
    print(f"{'MAE':<25} {mae:>10.2f} °C  {'Средняя ошибка прогноза'}")
    print(f"{'RMSE':<25} {rmse:>10.2f} °C  {'Чувствительность к крупным ошибкам'}")
    print(f"{'MAPE':<25} {mape:>10.2f} %    {'Относительная ошибка'}")
    print(f"{'R²':<25} {r2:>10.4f}       {'Доля объясненной дисперсии'}")
    print(f"{'Ошибка от диапазона':<25} {mean_abs_error_pct:>10.2f} %    {'Критичность для точности'}")
    print("-" * 65)
    
    # Рекомендации по надежности прогноза
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


def main():
    print("=" * 80)
    print("ПРОГНОЗИРОВАНИЕ МИНИМАЛЬНЫХ ТЕМПЕРАТУР (ОПТИМИЗИРОВАННАЯ ВЕРСИЯ)")
    print("=" * 80)
    
    # Генерируем ID эксперимента
    experiment_id = get_experiment_id(EXPERIMENT_CONFIG)
    print(f"ID эксперимента: {experiment_id}")
    print(f"Время запуска: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Шаг 1: Загрузка данных
    print("\n1. Загрузка данных минимальных суточных температур...")
    try:
        df = load_temperature_data()
    except Exception as e:
        print(f"Ошибка при загрузке данных: {e}")
        print("\nСовет: Убедитесь, что файл находится в одной из следующих директорий:")
        print("  - data/daily-minimum-temperatures-in-me.csv")
        print("  - ~/ts_feature_eng/data/...")
        return
    
    # Шаг 2: Анализ структуры данных
    print("\n2. Анализ структуры временного ряда...")
    print(f"   Частота дискретизации: {pd.infer_freq(df.index)}")
    print(f"   Количество пропусков: {df['min_temperature'].isna().sum()}")
    print(f"   Минимальная температура: {df['min_temperature'].min():.2f} °C")
    print(f"   Максимальная температура: {df['min_temperature'].max():.2f} °C")
    print(f"   Средняя температура: {df['min_temperature'].mean():.2f} °C")
    
    # Шаг 3: Визуализация паттернов температур
    print("\n3. Визуализация ключевых паттернов температур...")
    plot_temperature_patterns(df, title="Паттерны минимальных температур")
    
    # Шаг 4: Подготовка данных для прогнозирования
    print("\n4. Подготовка данных для задачи прогнозирования...")
    
    # Целевая переменная: прогноз на 1 день вперед
    y = df["min_temperature"].shift(-1)
    
    # Удаляем последние наблюдения с пропусками
    valid_mask = ~y.isna()
    X = df.loc[y.index, ["min_temperature"]].copy()
    y = y[valid_mask]
    X = X[valid_mask]
    
    print(f"   Размер признакового пространства до инженерии: {X.shape[1]} признаков")
    print(f"   Количество наблюдений для обучения: {len(X)}")
    
    # Шаг 5: Разделение на обучающую и тестовую выборки (временное разделение)
    print("\n5. Разделение данных с сохранением временного порядка...")
    split_idx = int(len(X) * EXPERIMENT_CONFIG["train_test_split"])
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"   Обучающая выборка: {len(X_train)} наблюдений ({X_train.index.min()} — {X_train.index.max()})")
    print(f"   Тестовая выборка: {len(X_test)} наблюдений ({X_test.index.min()} — {X_test.index.max()})")
    
    # Шаг 6: Автоматическая инженерия признаков с гибридным подходом
    print("\n6. Автоматическая инженерия признаков с гибридным подходом...")
    print("   Режим: core pipeline (обязательные лаги) + auto pipeline (адаптивные преобразования)")
    
    engineer = AutoFeatureEngineer(
        optimize=True,
        n_calls=EXPERIMENT_CONFIG["n_calls"],
        n_initial_points=EXPERIMENT_CONFIG["n_initial_points"],
        apply_selection=True,
        selection_threshold=EXPERIMENT_CONFIG["selection_threshold"],
        variance_threshold=EXPERIMENT_CONFIG["variance_threshold"],
        shap_selection=EXPERIMENT_CONFIG["shap_selection"],
        random_state=EXPERIMENT_CONFIG["random_state"],
        verbose=1
    )
    
    # Обучение инженера
    X_train_transformed = engineer.fit_transform(X_train, y_train)
    
    print(f"\n   Сгенерировано признаков: {X_train_transformed.shape[1]}")
    print(f"   Примеры ключевых признаков:")
    
    # Группируем признаки по типу для лучшей интерпретации
    feature_groups = {
        "Core (лаги)": [col for col in X_train_transformed.columns if "lag_" in col][:3],
        "Оконные (тренд/волатильность)": [col for col in X_train_transformed.columns if "window" in col and "lag_" not in col][:3],
        "Спектральные (сезонность)": [col for col in X_train_transformed.columns if "stl" in col or "dwt" in col][:3],
        "Временные (цикличность)": [col for col in X_train_transformed.columns if "time." in col][:3],
        "Календарные": [col for col in X_train_transformed.columns if "calendar" in col or "ramadan" in col.lower()][:3]
    }
    
    for group_name, features in feature_groups.items():
        if features:
            print(f"\n   {group_name}:")
            for feat in features[:2]:  # Показываем только первые 2 из каждой группы
                print(f"      • {feat}")
    
    # Шаг 7: Применение к тестовым данным
    print("\n7. Применение обученного инженера к тестовым данным...")
    X_test_transformed = engineer.transform(X_test)
    print(f"   Размер трансформированных тестовых данных: {X_test_transformed.shape}")
    
    # Шаг 8: Обучение модели прогнозирования
    print("\n8. Обучение модели градиентного бустинга на сгенерированных признаках...")
    model = GradientBoostingRegressor(
        n_estimators=EXPERIMENT_CONFIG["n_estimators"],
        max_depth=EXPERIMENT_CONFIG["max_depth"],
        learning_rate=EXPERIMENT_CONFIG["learning_rate"],
        random_state=EXPERIMENT_CONFIG["random_state"]
    )
    
    # Заполняем оставшиеся пропуски нулями (для начальных наблюдений оконных признаков)
    model.fit(X_train_transformed.fillna(0), y_train)
    
    # Прогнозирование
    y_pred_train = model.predict(X_train_transformed.fillna(0))
    y_pred_test = model.predict(X_test_transformed.fillna(0))
    
    # Шаг 9: Оценка качества прогноза
    print("\n9. Оценка качества прогноза на тестовой выборке...")
    test_metrics = evaluate_forecast(y_test, y_pred_test, horizon_days=1)
    
    # Добавляем дополнительную информацию в метрики
    test_metrics["n_features"] = X_train_transformed.shape[1]
    test_metrics["n_train_samples"] = len(X_train)
    test_metrics["n_test_samples"] = len(X_test)
    
    # Шаг 10: Анализ важности признаков
    print("\n10. Анализ важности сгенерированных признаков...")
    
    # Получаем важность признаков из модели
    feature_importance = pd.DataFrame({
        "feature": X_train_transformed.columns,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)
    
    print("\n   Топ-10 наиболее важных признаков:")
    print("   " + "-" * 70)
    for idx, row in feature_importance.head(10).iterrows():
        importance_pct = row["importance"] * 100
        # Упрощаем имя признака для лучшей читаемости
        feature_name = row["feature"]
        if len(feature_name) > 50:
            feature_name = feature_name[:47] + "..."
        print(f"   {importance_pct:>6.2f}% | {feature_name}")
    print("   " + "-" * 70)
    
    # Анализ по группам признаков
    print("\n   Важность по группам признаков:")
    groups = {
        "Core (лаги)": ["lag_"],
        "Оконные преобразования": ["window"],
        "Спектральные методы": ["stl", "dwt"],
        "Временное кодирование": ["time."],
        "Календарные признаки": ["calendar", "is_weekend", "part_of_day"]
    }
    
    group_importance = {}
    for group_name, keywords in groups.items():
        mask = pd.Series(False, index=feature_importance.index)
        for kw in keywords:
            mask |= feature_importance["feature"].str.contains(kw, case=False)
        group_importance[group_name] = feature_importance.loc[mask, "importance"].sum() * 100
    
    # Сортируем по важности
    for group_name, importance in sorted(group_importance.items(), key=lambda x: x[1], reverse=True):
        bar = "█" * int(importance / 2)
        print(f"   {group_name:25s} | {importance:5.1f}% {bar}")
    
    # Шаг 11: Прогнозирование на разные горизонты (упрощённое)
    print("\n11. Прогнозирование на разные горизонты (демонстрация подхода)...")
    print("   ⚠ Пропущено для ускорения (см. energy_forecasting_quick.py для полной версии)")
    
    # Шаг 12: Сравнение с базовыми моделями
    print("\n12. Сравнение с базовыми подходами...")
    
    # Базовая модель 1: Наивный прогноз (последнее наблюдение)
    naive_pred = y_test.shift(1).fillna(y_test.mean())
    naive_mae = mean_absolute_error(y_test, naive_pred)
    
    # Базовая модель 2: Сезонный наивный (значение 365 дней назад)
    seasonal_naive_pred = y_test.shift(365).fillna(y_test.mean())
    seasonal_naive_mae = mean_absolute_error(y_test, seasonal_naive_pred)
    
    # Базовая модель 3: Случайный лес на исходных признаках
    rf_base = RandomForestRegressor(n_estimators=50, max_depth=3, random_state=EXPERIMENT_CONFIG["random_state"])
    rf_base.fit(X_train.fillna(0), y_train)
    rf_base_pred = rf_base.predict(X_test.fillna(0))
    rf_base_mae = mean_absolute_error(y_test, rf_base_pred)
    
    # Наша модель
    auto_mae = test_metrics["MAE (°C)"]
    
    # Вывод сравнения
    print("\n   Сравнение моделей по MAE (меньше — лучше):")
    print("   " + "-" * 60)
    models_comparison = [
        ("Наивный прогноз", naive_mae),
        ("Сезонный наивный (365д)", seasonal_naive_mae),
        ("Случайный лес (базовые признаки)", rf_base_mae),
        ("Градиентный бустинг (авто. инж. признаков)", auto_mae)
    ]
    
    best_mae = min(m[1] for m in models_comparison)
    
    for model_name, mae_value in models_comparison:
        star = " ← ЛУЧШАЯ" if mae_value == best_mae else ""
        improvement = ((naive_mae - mae_value) / naive_mae) * 100
        print(f"   {model_name:35s} | {mae_value:6.2f} °C | улучшение: {improvement:5.1f}%{star}")
    
    print("   " + "-" * 60)
    
    # Шаг 13: Визуализация прогноза
    print("\n13. Визуализация прогноза на тестовом периоде...")
    
    # Выбираем период для визуализации (последние 200 наблюдений тестовой выборки)
    viz_end = len(y_test)
    viz_start = max(0, viz_end - 200)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
    
    # Верхний график: фактическая температура и прогноз
    ax1.plot(
        y_test.index[viz_start:viz_end],
        y_test.iloc[viz_start:viz_end],
        label="Фактическая температура",
        color='blue',
        linewidth=2,
        alpha=0.8
    )
    ax1.plot(
        y_test.index[viz_start:viz_end],
        y_pred_test[viz_start:viz_end],
        label="Прогноз (наш метод)",
        color='red',
        linewidth=2,
        alpha=0.8,
        linestyle='--'
    )
    ax1.fill_between(
        y_test.index[viz_start:viz_end],
        y_test.iloc[viz_start:viz_end],
        y_pred_test[viz_start:viz_end],
        alpha=0.3,
        color='gray',
        label=f'Ошибка (MAE: {auto_mae:.2f} °C)'
    )
    ax1.set_title("Прогноз минимальных температур (последние 200 дней тестовой выборки)", 
                  fontsize=14, fontweight='bold')
    ax1.set_ylabel("Температура (°C)", fontsize=11)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Нижний график: ошибка прогноза
    error = y_test.iloc[viz_start:viz_end] - y_pred_test[viz_start:viz_end]
    ax2.plot(
        y_test.index[viz_start:viz_end],
        error,
        color='purple',
        linewidth=1.5,
        alpha=0.7
    )
    ax2.axhline(0, color='black', linestyle='-', linewidth=0.8)
    ax2.fill_between(
        y_test.index[viz_start:viz_end],
        error,
        0,
        where=(error > 0),
        alpha=0.4,
        color='red',
        label='Переоценка'
    )
    ax2.fill_between(
        y_test.index[viz_start:viz_end],
        error,
        0,
        where=(error < 0),
        alpha=0.4,
        color='green',
        label='Недооценка'
    )
    ax2.set_title("Ошибка прогноза", fontsize=14, fontweight='bold')
    ax2.set_ylabel("Ошибка (°C)", fontsize=11)
    ax2.set_xlabel("Время", fontsize=11)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("temperature_forecast_comparison.png", dpi=150, bbox_inches='tight')
    print("   График сохранен как 'temperature_forecast_comparison.png'")
    plt.close()
    
    # Шаг 14: Сохранение результатов в CSV
    print("\n14. Сохранение результатов эксперимента...")
    metrics_file, params_file, report_file = save_results_to_csv(
        metrics=test_metrics,
        config=EXPERIMENT_CONFIG,
        experiment_id=experiment_id,
        results_dir="results"
    )

    print("\n" + "=" * 80)
    print("ПРИМЕР ЗАВЕРШЕН")
    print("=" * 80)
    print("\nСгенерированные файлы:")
    print("  • temperature_patterns.png : анализ паттернов температур")
    print("  • temperature_forecast_comparison.png : сравнение прогноза с фактом")
    print("  • results/metrics_history.csv : история метрик")
    print("  • results/experiments_config.csv : история конфигураций")
    print(f"  • Автоматическая инженерия признаков улучшила прогноз на {((naive_mae - auto_mae) / naive_mae * 100):.1f}%")



if __name__ == "__main__":
    # Проверка наличия matplotlib и seaborn для визуализации
    try:
        import matplotlib  # noqa: F401
        import seaborn  # noqa: F401
    except ImportError:
        print("WARNING: matplotlib или seaborn не установлены. Визуализация будет ограничена.")
        print("Установите через: pip install matplotlib seaborn")
    
    import warnings  # Для подавления FutureWarning
    main()