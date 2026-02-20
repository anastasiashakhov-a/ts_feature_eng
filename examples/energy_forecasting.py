# examples/energy_forecasting.py 

"""
Пример прогнозирования энергопотребления на данных из Марокко.

Демонстрирует применение автоматической инженерии признаков для реальной задачи:
1. Загрузка и анализ данных энергопотребления Марокко (зона 1)
2. Обнаружение сезонных паттернов (суточных, недельных, годовых)
3. Генерация календарных признаков с учетом особенностей Марокко:
   - Суточные циклы потребления (утро/день/вечер/ночь)
   - Недельные паттерны (будни vs выходные)
   - Сезонные колебания (лето vs зима)
   - Учет Рамадана через календарные признаки (демонстрация подхода)
4. Автоматический подбор оптимальных методов инженерии признаков
5. Прогнозирование на горизонт 1 час
6. Сравнение с базовыми моделями и интерпретация результатов
7. Гибридная инженерия признаков (обязательные лаги + адаптивные преобразования)

Оптимизировано для скорости:
- Убрано прогнозирование на 24 часа (экономия ~50 минут)
- Уменьшено количество итераций байесовской оптимизации
- Отключена SHAP-фильтрация для скорости
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
    "n_calls": 5,                 # Итерации байесовской оптимизации (было 20)
    "n_initial_points": 3,        # Начальные точки оптимизации (было 5)
    "selection_threshold": 0.25,  # Порог отбора признаков
    "variance_threshold": 0.01,   # Порог дисперсии
    "shap_selection": True,      
    "n_estimators": 100,          # Деревья в модели (было 200)
    "max_depth": 4,               # Глубина деревьев (было 6)
    "learning_rate": 0.1,         # Скорость обучения
    "random_state": 42,           # Сид для воспроизводимости
    "train_test_split": 0.8,      # Доля обучающей выборки
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
        "mae_mw": metrics.get("MAE (МВт)", np.nan),
        "rmse_mw": metrics.get("RMSE (МВт)", np.nan),
        "mape_pct": metrics.get("MAPE (%)", np.nan),
        "r2": metrics.get("R²", np.nan),
        "error_from_peak_pct": metrics.get("Ошибка от пика (%)", np.nan),
        "peak_consumption_mw": metrics.get("Пик потребления (МВт)", np.nan),
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


def load_morocco_energy_data(data_path=None):
    """
    Загружает данные энергопотребления Марокко.
    
    Параметры
    ----------
    data_path : str, опционально
        Путь к CSV файлу. Если не указан, ищет в стандартных местах.
    
    Возвращает
    ----------
    df : pd.DataFrame
        DataFrame с колонками 'timestamp' и 'power_consumption'.
    """
    # Определяем путь к данным
    if data_path is None:
        # Ищем в нескольких возможных местах
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
    
    # Загрузка данных
    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        raise RuntimeError(f"Ошибка при загрузке файла: {e}")
    
    # Очистка имен столбцов от лишних пробелов
    df.columns = df.columns.str.strip()
    
    # Определение столбцов с временными метками и потреблением
    # Ищем столбцы, содержащие 'time', 'date', 'timestamp', 'dt'
    time_cols = [col for col in df.columns if any(kw in col.lower() for kw in ['time', 'date', 'timestamp', 'dt'])]
    
    if not time_cols:
        raise ValueError("Не найден столбец с временными метками в данных")
    
    time_col = time_cols[0]
    print(f"  Найден столбец временных меток: '{time_col}'")
    
    # Ищем столбцы с потреблением (содержащие 'power', 'consum', 'load', 'energy')
    power_cols = [col for col in df.columns if any(kw in col.lower() for kw in ['power', 'consum', 'load', 'energy'])]
    
    if not power_cols:
        raise ValueError("Не найден столбец с данными энергопотребления")
    
    power_col = power_cols[0]
    print(f"  Найден столбец потребления: '{power_col}'")
    
    # Преобразование временных меток
    try:
        df[time_col] = pd.to_datetime(df[time_col])
    except Exception as e:
        raise ValueError(f"Ошибка преобразования временных меток: {e}")
    
    # Создание итогового DataFrame
    result = pd.DataFrame({
        "timestamp": df[time_col],
        "power_consumption": df[power_col]
    })
    
    # Сортировка по времени и удаление дубликатов
    result = result.sort_values("timestamp").drop_duplicates(subset="timestamp").reset_index(drop=True)
    
    # Установка временного индекса
    result = result.set_index("timestamp")
    
    # Удаление пропусков в потреблении
    initial_len = len(result)
    result = result.dropna(subset=["power_consumption"])
    final_len = len(result)
    
    print(f"  Удалено {initial_len - final_len} наблюдений с пропусками")
    print(f"  Итоговый размер данных: {len(result)} наблюдений")
    print(f"  Диапазон дат: {result.index.min()} — {result.index.max()}")
    
    return result


def detect_ramadan_periods(df, years=None):
    """
    Определяет периоды Рамадана для указанных лет.
    
    Примечание: В реальном применении следует использовать точные даты Рамадана
    из исламского календаря. Здесь приведена упрощенная демонстрация.
    
    Параметры
    ----------
    df : pd.DataFrame
        Данные с временным индексом.
    years : list, опционально
        Список лет для анализа. Если не указан — все годы в данных.
    
    Возвращает
    ----------
    ramadan_mask : pd.Series
        Булев маска, где True соответствует периодам Рамадана.
    """
    if years is None:
        years = sorted(df.index.year.unique())
    
    # Упрощенная демонстрация: Рамадан обычно приходится на март-апрель-май
    # В реальном применении замените на точные даты из исламского календаря
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
    """
    Создает признаки, связанные с Рамаданом.
    
    Примечание: Это демонстрационная реализация. Для промышленного применения
    требуется точный календарь исламских праздников.
    
    Параметры
    ----------
    df : pd.DataFrame
        Исходные данные с временным индексом.
    
    Возвращает
    ----------
    df_features : pd.DataFrame
        DataFrame с дополнительными признаками Рамадана.
    """
    print("\nСоздание признаков Рамадана (демонстрационная реализация)...")
    
    # Определение периодов Рамадана
    ramadan_mask = detect_ramadan_periods(df)
    
    # Базовый признак: находится ли наблюдение в периоде Рамадана
    df_ramadan = pd.DataFrame(index=df.index)
    df_ramadan["is_ramadan"] = ramadan_mask.astype(int)
    
    # Признаки до/после Рамадана (для учета подготовки и празднования)
    df_ramadan["days_until_ramadan"] = np.nan
    df_ramadan["days_since_ramadan"] = np.nan
    
    # Расчет дней до/после для каждого наблюдения
    ramadan_days = df_ramadan[df_ramadan["is_ramadan"] == 1].index
    if len(ramadan_days) > 0:
        for idx in df_ramadan.index:
            if idx in ramadan_days:
                df_ramadan.loc[idx, "days_until_ramadan"] = 0
                df_ramadan.loc[idx, "days_since_ramadan"] = 0
            else:
                # Дни до ближайшего Рамадана
                future_ramadan = ramadan_days[ramadan_days > idx]
                if len(future_ramadan) > 0:
                    days_until = (future_ramadan[0] - idx).days
                    df_ramadan.loc[idx, "days_until_ramadan"] = min(days_until, 30)  # Ограничиваем 30 днями
                
                # Дни после последнего Рамадана
                past_ramadan = ramadan_days[ramadan_days < idx]
                if len(past_ramadan) > 0:
                    days_since = (idx - past_ramadan[-1]).days
                    df_ramadan.loc[idx, "days_since_ramadan"] = min(days_since, 30)
    
    # Заполняем пропуски
    df_ramadan["days_until_ramadan"] = df_ramadan["days_until_ramadan"].fillna(30)
    df_ramadan["days_since_ramadan"] = df_ramadan["days_since_ramadan"].fillna(30)
    
    print(f"  Создано признаков Рамадана: {len(df_ramadan.columns)}")
    print(f"  Наблюдений в периоде Рамадана: {df_ramadan['is_ramadan'].sum()} ({df_ramadan['is_ramadan'].mean()*100:.1f}%)")
    
    return df_ramadan


def plot_consumption_patterns(df, title="Паттерны энергопотребления"):
    """
    Визуализирует ключевые паттерны энергопотребления.
    
    Параметры
    ----------
    df : pd.DataFrame
        Данные с колонкой 'power_consumption'.
    title : str
        Заголовок графика.
    """
    print(f"\nВизуализация паттернов потребления: {title}")
    
    # Создаем подграфики
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    
    # 1. Общий тренд потребления
    ax = axes[0, 0]
    df["power_consumption"].plot(ax=ax, color='blue', linewidth=1, alpha=0.7)
    ax.set_title("Общий тренд энергопотребления", fontsize=12, fontweight='bold')
    ax.set_ylabel("Потребление (МВт)", fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 2. Суточной паттерн (усредненный по всем дням)
    ax = axes[0, 1]
    hourly_avg = df.groupby(df.index.hour)["power_consumption"].mean()
    hourly_std = df.groupby(df.index.hour)["power_consumption"].std()
    ax.plot(hourly_avg.index, hourly_avg.values, color='darkgreen', linewidth=2, label='Среднее')
    ax.fill_between(
        hourly_avg.index,
        hourly_avg.values - hourly_std.values,
        hourly_avg.values + hourly_std.values,
        alpha=0.3,
        color='green',
        label='±1 std'
    )
    ax.set_title("Суточной паттерн потребления", fontsize=12, fontweight='bold')
    ax.set_xlabel("Час суток", fontsize=10)
    ax.set_ylabel("Потребление (МВт)", fontsize=10)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Недельный паттерн
    ax = axes[1, 0]
    weekday_avg = df.groupby(df.index.dayofweek)["power_consumption"].mean()
    weekday_labels = ['Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб', 'Вс']
    ax.bar(weekday_labels, weekday_avg.values, color='steelblue', alpha=0.8)
    ax.set_title("Недельный паттерн потребления", fontsize=12, fontweight='bold')
    ax.set_ylabel("Среднее потребление (МВт)", fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 4. Сезонный паттерн (по месяцам)
    ax = axes[1, 1]
    monthly_avg = df.groupby(df.index.month)["power_consumption"].mean()
    month_labels = ['Янв', 'Фев', 'Мар', 'Апр', 'Май', 'Июн', 'Июл', 'Авг', 'Сен', 'Окт', 'Ноя', 'Дек']
    ax.plot(month_labels, monthly_avg.values, marker='o', color='darkred', linewidth=2, markersize=6)
    ax.set_title("Сезонный паттерн потребления", fontsize=12, fontweight='bold')
    ax.set_ylabel("Среднее потребление (МВт)", fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 5. Распределение потребления
    ax = axes[2, 0]
    sns.histplot(df["power_consumption"], bins=50, kde=True, ax=ax, color='purple')
    ax.axvline(df["power_consumption"].mean(), color='red', linestyle='--', label=f'Среднее: {df["power_consumption"].mean():.0f} МВт')
    ax.axvline(df["power_consumption"].median(), color='orange', linestyle='--', label=f'Медиана: {df["power_consumption"].median():.0f} МВт')
    ax.set_title("Распределение энергопотребления", fontsize=12, fontweight='bold')
    ax.set_xlabel("Потребление (МВт)", fontsize=10)
    ax.set_ylabel("Частота", fontsize=10)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. Автокорреляция
    ax = axes[2, 1]
    from statsmodels.graphics.tsaplots import plot_acf
    plot_acf(df["power_consumption"].dropna(), lags=168, ax=ax, alpha=0.05)  # 168 часов = 1 неделя
    ax.set_title("Автокорреляция потребления (до 168 лагов)", fontsize=12, fontweight='bold')
    ax.set_xlabel("Лаг (часы)", fontsize=10)
    ax.set_ylabel("ACF", fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig("energy_consumption_patterns.png", dpi=150, bbox_inches='tight')
    print("  График сохранен как 'energy_consumption_patterns.png'")
    plt.close()


def evaluate_forecast(y_true, y_pred, horizon_hours=1):
    """
    Оценивает качество прогноза с интерпретацией для энергосистемы.
    
    Параметры
    ----------
    y_true : array-like
        Фактические значения.
    y_pred : array-like
        Предсказанные значения.
    horizon_hours : int
        Горизонт прогнозирования в часах.
    
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
    
    # Средняя ошибка в процентах от пикового потребления
    peak_consumption = np.max(y_true)
    mean_abs_error_pct = (mae / peak_consumption) * 100
    
    metrics = {
        "MAE (МВт)": mae,
        "RMSE (МВт)": rmse,
        "MAPE (%)": mape,
        "R²": r2,
        "Ошибка от пика (%)": mean_abs_error_pct,
        "Пик потребления (МВт)": peak_consumption
    }
    
    # Интерпретация для энергосистемы
    print(f"\nОценка качества прогноза (горизонт: {horizon_hours} час{'а' if horizon_hours in [2,3,4] else 'ов'}):")
    print("-" * 65)
    print(f"{'Метрика':<25} {'Значение':<15} {'Интерпретация'}")
    print("-" * 65)
    print(f"{'MAE':<25} {mae:>10.2f} МВт  {'Средняя ошибка прогноза'}")
    print(f"{'RMSE':<25} {rmse:>10.2f} МВт  {'Чувствительность к крупным ошибкам'}")
    print(f"{'MAPE':<25} {mape:>10.2f} %    {'Относительная ошибка'}")
    print(f"{'R²':<25} {r2:>10.4f}       {'Доля объясненной дисперсии'}")
    print(f"{'Ошибка от пика':<25} {mean_abs_error_pct:>10.2f} %    {'Критичность для балансировки'}")
    print("-" * 65)
    
    # Рекомендации по надежности прогноза
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


def main():
    print("=" * 80)
    print("ПРОГНОЗИРОВАНИЕ ЭНЕРГОПОТРЕБЛЕНИЯ НА ДАННЫХ ИЗ МАРОККО (ОПТИМИЗИРОВАННАЯ ВЕРСИЯ)")
    print("=" * 80)
    
    # Генерируем ID эксперимента
    experiment_id = get_experiment_id(EXPERIMENT_CONFIG)
    print(f"ID эксперимента: {experiment_id}")
    print(f"Время запуска: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Шаг 1: Загрузка данных
    print("\n1. Загрузка данных энергопотребления Марокко...")
    try:
        df = load_morocco_energy_data()
    except Exception as e:
        print(f"Ошибка при загрузке данных: {e}")
        print("\nСовет: Убедитесь, что файл находится в одной из следующих директорий:")
        print("  - data/morocco zone 1 - powerconsumption_resampled (1).csv")
        print("  - ~/ts_feature_eng/data/...")
        return
    
    # Шаг 2: Анализ структуры данных
    print("\n2. Анализ структуры временного ряда...")
    print(f"   Частота дискретизации: {pd.infer_freq(df.index)}")
    print(f"   Количество пропусков: {df['power_consumption'].isna().sum()}")
    print(f"   Минимальное потребление: {df['power_consumption'].min():.2f} МВт")
    print(f"   Максимальное потребление: {df['power_consumption'].max():.2f} МВт")
    print(f"   Среднее потребление: {df['power_consumption'].mean():.2f} МВт")
    
    # Шаг 3: Визуализация паттернов потребления
    print("\n3. Визуализация ключевых паттернов потребления...")
    plot_consumption_patterns(df, title="Паттерны энергопотребления в Марокко (Зона 1)")
    
    # Шаг 4: Подготовка данных для прогнозирования
    print("\n4. Подготовка данных для задачи прогнозирования...")
    
    # Создаем признаки Рамадана (демонстрационные)
    ramadan_features = create_ramadan_features(df)
    
    # Объединяем с основными данными
    X = pd.DataFrame({"power_consumption": df["power_consumption"]})
    X = pd.concat([X, ramadan_features], axis=1)
    
    # Целевая переменная: прогноз на 1 час вперед
    y = df["power_consumption"].shift(-1)
    
    # Удаляем последние наблюдения с пропусками
    valid_mask = ~y.isna()
    X = X[valid_mask]
    y = y[valid_mask]
    
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
        "Календарные (Рамадан/будни)": [col for col in X_train_transformed.columns if "calendar" in col or "ramadan" in col.lower()][:3]
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
    test_metrics = evaluate_forecast(y_test, y_pred_test, horizon_hours=1)
    
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
        "Календарные признаки": ["calendar", "ramadan", "is_weekend", "part_of_day"]
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
  
    
    # Шаг 11: Сравнение с базовыми моделями (теперь шаг 11)
    print("\n11. Сравнение с базовыми подходами...")
    
    # Базовая модель 1: Наивный прогноз (последнее наблюдение)
    naive_pred = y_test.shift(1).fillna(y_test.mean())
    naive_mae = mean_absolute_error(y_test, naive_pred)
    
    # Базовая модель 2: Сезонный наивный (значение 24 часа назад)
    seasonal_naive_pred = y_test.shift(24).fillna(y_test.mean())
    seasonal_naive_mae = mean_absolute_error(y_test, seasonal_naive_pred)
    
    # Базовая модель 3: Случайный лес на исходных признаках
    rf_base = RandomForestRegressor(n_estimators=50, max_depth=3, random_state=EXPERIMENT_CONFIG["random_state"])
    rf_base.fit(X_train.fillna(0), y_train)
    rf_base_pred = rf_base.predict(X_test.fillna(0))
    rf_base_mae = mean_absolute_error(y_test, rf_base_pred)
    
    # Наша модель
    auto_mae = test_metrics["MAE (МВт)"]
    
    # Вывод сравнения
    print("\n   Сравнение моделей по MAE (меньше — лучше):")
    print("   " + "-" * 60)
    models_comparison = [
        ("Наивный прогноз", naive_mae),
        ("Сезонный наивный (24ч)", seasonal_naive_mae),
        ("Случайный лес (базовые признаки)", rf_base_mae),
        ("Градиентный бустинг (авто. инж. признаков)", auto_mae)
    ]
    
    best_mae = min(m[1] for m in models_comparison)
    
    for model_name, mae_value in models_comparison:
        star = " ← ЛУЧШАЯ" if mae_value == best_mae else ""
        improvement = ((naive_mae - mae_value) / naive_mae) * 100
        print(f"   {model_name:35s} | {mae_value:6.2f} МВт | улучшение: {improvement:5.1f}%{star}")
    
    print("   " + "-" * 60)
    
    # Шаг 12: Визуализация прогноза (теперь шаг 12)
    print("\n12. Визуализация прогноза на тестовом периоде...")
    
    # Выбираем период для визуализации (последние 200 наблюдений тестовой выборки)
    viz_end = len(y_test)
    viz_start = max(0, viz_end - 200)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
    
    # Верхний график: фактическое потребление и прогноз
    ax1.plot(
        y_test.index[viz_start:viz_end],
        y_test.iloc[viz_start:viz_end],
        label="Фактическое потребление",
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
        label=f'Ошибка (MAE: {auto_mae:.2f} МВт)'
    )
    ax1.set_title("Прогноз энергопотребления в Марокко (последние 200 часов тестовой выборки)", 
                  fontsize=14, fontweight='bold')
    ax1.set_ylabel("Потребление (МВт)", fontsize=11)
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
    ax2.set_ylabel("Ошибка (МВт)", fontsize=11)
    ax2.set_xlabel("Время", fontsize=11)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("energy_forecast_comparison.png", dpi=150, bbox_inches='tight')
    print("   График сохранен как 'energy_forecast_comparison.png'")
    plt.close()
    
    # Шаг 13: Сохранение результатов в CSV (теперь шаг 13)
    print("\n13. Сохранение результатов эксперимента...")
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
    print("  • energy_consumption_patterns.png : анализ паттернов потребления")
    print("  • energy_forecast_comparison.png  : сравнение прогноза с фактом")
    print("  • results/metrics_history.csv     : история метрик")
    print("  • results/experiments_config.csv  : история конфигураций")
    print(f"  • Автоматическая инженерия признаков улучшила прогноз на {((naive_mae - auto_mae) / naive_mae * 100):.1f}%")



if __name__ == "__main__":
    # Проверка наличия matplotlib и seaborn для визуализации
    try:
        import matplotlib  # noqa: F401
        import seaborn  # noqa: F401
    except ImportError:
        print("WARNING: matplotlib или seaborn не установлены. Визуализация будет ограничена.")
        print("Установите через: pip install matplotlib seaborn")
    
    main()