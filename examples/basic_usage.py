# examples/basic_usage.py 

"""
Базовый пример использования модуля автоматической инженерии признаков для временных рядов.

Демонстрирует:
1. Генерацию синтетического временного ряда с сезонностью и трендом
2. Автоматическое извлечение мета-признаков
3. Байесовскую оптимизацию выбора методов инженерии признаков
4. Генерацию признаков через оконные, спектральные и временные преобразования
5. Отбор наиболее информативных признаков
6. Интеграцию с моделью прогнозирования и оценку качества
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from ts_feature_eng import AutoFeatureEngineer
from ts_feature_eng.pipeline import AutoFeatureEngineer as AutoFeatureEngineerClass


def create_synthetic_time_series(n_samples=1000, freq="H", seed=42):
    """
    Создает синтетический временной ряд с сезонностью, трендом и шумом.
    
    Параметры
    ----------
    n_samples : int
        Количество наблюдений.
    freq : str
        Частота временного индекса ('H' для часовой).
    seed : int
        Фиксация случайного состояния.
    
    Возвращает
    ----------
    df : pd.DataFrame
        DataFrame с временным индексом и столбцом 'value'.
    """
    np.random.seed(seed)
    
    # Создаем временной индекс
    dates = pd.date_range("2023-01-01", periods=n_samples, freq=freq)
    
    # Генерируем компоненты ряда
    t = np.arange(n_samples)
    
    # Линейный тренд
    trend = 0.02 * t
    
    # Суточная сезонность (24 часа)
    daily_seasonality = 10 * np.sin(2 * np.pi * t / 24)
    
    # Недельная сезонность (168 часов)
    weekly_seasonality = 5 * np.sin(2 * np.pi * t / 168)
    
    # Шум
    noise = np.random.randn(n_samples) * 2.0
    
    # Комбинируем компоненты
    values = trend + daily_seasonality + weekly_seasonality + noise
    
    # Создаем DataFrame
    df = pd.DataFrame({"value": values}, index=dates)
    
    return df


def plot_time_series_and_features(df_original, df_transformed, n_features_to_show=5):
    """
    Визуализирует исходный временной ряд и сгенерированные признаки.
    
    Параметры
    ----------
    df_original : pd.DataFrame
        Исходный временной ряд.
    df_transformed : pd.DataFrame
        Трансформированный набор признаков.
    n_features_to_show : int
        Количество признаков для отображения.
    """
    # Выбираем подмножество признаков для визуализации
    feature_cols = df_transformed.columns[:n_features_to_show]
    
    # Создаем фигуру
    fig, axes = plt.subplots(
        n_features_to_show + 1, 1, figsize=(12, 3 * (n_features_to_show + 1)), sharex=True
    )
    
    # Плотим исходный ряд
    axes[0].plot(df_original.index, df_original["value"], color="blue", linewidth=1)
    axes[0].set_ylabel("Исходный ряд", fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Плотим сгенерированные признаки
    for i, col in enumerate(feature_cols):
        axes[i + 1].plot(df_transformed.index, df_transformed[col], color="green", linewidth=1)
        axes[i + 1].set_ylabel(col, fontsize=9)
        axes[i + 1].grid(True, alpha=0.3)
    
    axes[-1].set_xlabel("Время", fontsize=11)
    plt.suptitle("Исходный временной ряд и сгенерированные признаки", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("time_series_features.png", dpi=150, bbox_inches="tight")
    print("График сохранен как 'time_series_features.png'")
    plt.close()


def evaluate_forecast(y_true, y_pred, model_name="Model"):
    """
    Оценивает качество прогноза по нескольким метрикам.
    
    Параметры
    ----------
    y_true : array-like
        Фактические значения.
    y_pred : array-like
        Предсказанные значения.
    model_name : str
        Название модели для отчета.
    
    Возвращает
    ----------
    metrics : dict
        Словарь с метриками качества.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # MAPE с защитой от деления на ноль
    mask = np.abs(y_true) > 1e-6
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE (%)": mape,
        "R²": r2
    }
    
    print(f"\nОценка качества {model_name}:")
    print("-" * 40)
    for metric, value in metrics.items():
        if metric == "R²":
            print(f"{metric:15s}: {value:.4f}")
        else:
            print(f"{metric:15s}: {value:.2f}")
    print("-" * 40)
    
    return metrics


def main():
    print("=" * 70)
    print("БАЗОВЫЙ ПРИМЕР ИСПОЛЬЗОВАНИЯ МОДУЛЯ АВТОМАТИЧЕСКОЙ ИНЖЕНЕРИИ ПРИЗНАКОВ")
    print("=" * 70)
    
    # Шаг 1: Генерация синтетического временного ряда
    print("\n1. Генерация синтетического временного ряда...")
    df = create_synthetic_time_series(n_samples=1000, freq="H", seed=42)
    print(f"   Создан ряд из {len(df)} наблюдений с часовой частотой")
    print(f"   Диапазон дат: {df.index.min()} — {df.index.max()}")
    
    # Шаг 2: Подготовка данных для прогнозирования (прогноз на 1 шаг вперед)
    print("\n2. Подготовка данных для задачи прогнозирования...")
    X = df[["value"]].copy()
    y = df["value"].shift(-1)  # Прогноз на 1 шаг вперед
    
    # Удаляем последнюю строку с пропуском
    X = X.iloc[:-1]
    y = y.iloc[:-1]
    
    print(f"   Размер признаков X: {X.shape}")
    print(f"   Размер целевой переменной y: {y.shape}")
    
    # Шаг 3: Разделение на обучающую и тестовую выборки (с сохранением временного порядка)
    print("\n3. Разделение данных на обучающую (80%) и тестовую (20%) выборки...")
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"   Обучающая выборка: {len(X_train)} наблюдений")
    print(f"   Тестовая выборка: {len(X_test)} наблюдений")
    
    # Шаг 4: Создание и обучение автоматического инженера признаков
    print("\n4. Создание и обучение AutoFeatureEngineer...")
    print("   Режим: с байесовской оптимизацией (15 итераций)")
    
    engineer = AutoFeatureEngineer(
        optimize=True,           # Включить байесовскую оптимизацию
        n_calls=15,              # Количество итераций оптимизации
        n_initial_points=5,      # Количество разведочных точек
        apply_selection=True,    # Применить постфильтрацию признаков
        selection_threshold=0.3, # Порог пропусков для фильтрации
        variance_threshold=0.01, # Порог дисперсии
        random_state=42,
        verbose=1                # Детальное логирование
    )
    
    # Обучение инженера на обучающих данных
    X_train_transformed = engineer.fit_transform(X_train, y_train)
    
    print(f"\n   Сгенерировано признаков: {X_train_transformed.shape[1]}")
    print(f"   Примеры сгенерированных признаков:")
    for i, col in enumerate(X_train_transformed.columns[:5]):
        print(f"      {i+1}. {col}")
    
    # Шаг 5: Применение обученного инженера к тестовым данным
    print("\n5. Применение обученного инженера к тестовым данным...")
    X_test_transformed = engineer.transform(X_test)
    print(f"   Размер трансформированных тестовых данных: {X_test_transformed.shape}")
    
    # Шаг 6: Визуализация сгенерированных признаков
    print("\n6. Визуализация исходного ряда и сгенерированных признаков...")
    plot_time_series_and_features(X_train, X_train_transformed, n_features_to_show=4)
    
    # Шаг 7: Обучение модели прогнозирования
    print("\n7. Обучение модели случайного леса на сгенерированных признаках...")
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train_transformed.fillna(0), y_train)  # Заполняем оставшиеся пропуски нулями
    
    # Прогнозирование
    y_pred_train = model.predict(X_train_transformed.fillna(0))
    y_pred_test = model.predict(X_test_transformed.fillna(0))
    
    # Шаг 8: Оценка качества прогноза
    print("\n8. Оценка качества прогноза...")
    train_metrics = evaluate_forecast(y_train, y_pred_train, "Обучение")
    test_metrics = evaluate_forecast(y_test, y_pred_test, "Тестирование")
    
    # Шаг 9: Анализ мета-признаков временного ряда
    print("\n9. Анализ извлеченных мета-признаков ряда...")
    meta_features = engineer.get_meta_features()
    
    if meta_features:
        # Отбираем наиболее информативные мета-признаки для отображения
        key_meta_features = {
            "length": "Длина ряда",
            "missing_ratio": "Доля пропусков",
            "stationarity_adf": "Стационарность (ADF p-value)",
            "linearity": "Линейность (R²)",
            "dominant_freq": "Доминирующая частота",
            "acf_24": "Автокорреляция (лаг 24)",
            "acf_168": "Автокорреляция (лаг 168)"
        }
        
        print("\n   Ключевые мета-признаки:")
        print("   " + "-" * 50)
        for key, description in key_meta_features.items():
            if key in meta_features:
                value = meta_features[key]
                if isinstance(value, float):
                    print(f"   {description:30s}: {value:.4f}")
                else:
                    print(f"   {description:30s}: {value}")
        print("   " + "-" * 50)
    
    # Шаг 10: Анализ истории оптимизации
    print("\n10. Анализ истории байесовской оптимизации...")
    history = engineer.get_optimization_history()
    
    if history is not None and len(history) > 0:
        best_score = history["score"].max()
        best_iter = history["score"].idxmax()
        
        print(f"    Лучшая метрика качества: {-best_score:.4f} (итерация {best_iter + 1})")
        print(f"    Количество итераций: {len(history)}")
        print(f"    Среднее количество признаков за итерации: {history['n_features'].mean():.1f}")
        
        # Вывод лучших конфигураций
        print("\n    Топ-3 конфигурации по качеству:")
        top3 = history.nlargest(3, "score")
        for idx, row in top3.iterrows():
            print(f"      Итерация {idx + 1}: качество = {-row['score']:.4f}, признаков = {row['n_features']}")
    else:
        print("    История оптимизации недоступна (оптимизация не проводилась)")
    
    # Шаг 11: Сохранение и загрузка состояния инженера
    print("\n11. Сохранение и загрузка состояния инженера...")
    engineer.save("feature_engineer.pkl")
    print("    Состояние сохранено в 'feature_engineer.pkl'")
    
    engineer_loaded = AutoFeatureEngineerClass.load("feature_engineer.pkl")
    print("    Состояние успешно загружено")
    
    # Проверка идентичности результатов
    X_test_transformed_loaded = engineer_loaded.transform(X_test)
    are_identical = np.allclose(
        X_test_transformed.fillna(0).values,
        X_test_transformed_loaded.fillna(0).values,
        rtol=1e-5,
        atol=1e-8,
        equal_nan=True
    )
    
    if are_identical:
        print("    ✓ Результаты идентичны после загрузки")
    else:
        print("    ✗ Обнаружены различия в результатах после загрузки")
    
    # Шаг 12: Сравнение с базовым подходом (без автоматической инженерии признаков)
    print("\n12. Сравнение с базовым подходом (исходный признак без трансформации)...")
    
    # Базовая модель на исходном признаке
    base_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    base_model.fit(X_train.fillna(0), y_train)
    y_pred_base = base_model.predict(X_test.fillna(0))
    base_metrics = evaluate_forecast(y_test, y_pred_base, "Базовая модель")
    
    # Сравнение метрик
    print("\nСравнение метрик качества:")
    print("-" * 60)
    print(f"{'Метрика':<15} {'Базовая модель':<20} {'С автоматической':<20}")
    print(f"{'':<15} {'':<20} {'инженерией признаков':<20}")
    print("-" * 60)
    
    for metric in ["MAE", "RMSE", "MAPE (%)", "R²"]:
        base_val = base_metrics[metric]
        auto_val = test_metrics[metric]
        
        if metric == "R²":
            improvement = auto_val - base_val
            symbol = "↑" if improvement > 0 else "↓"
        else:
            improvement = base_val - auto_val
            symbol = "↓" if improvement > 0 else "↑"
        
        print(f"{metric:<15} {base_val:<20.2f} {auto_val:<20.2f} {symbol} {abs(improvement):.2f}")
    
    print("-" * 60)
    print("\nЗаключение:")
    if test_metrics["MAE"] < base_metrics["MAE"]:
        print("✓ Автоматическая инженерия признаков улучшила качество прогноза!")
    else:
        print("⚠ Автоматическая инженерия признаков не улучшила качество прогноза")
        print("  (возможно, требуется больше итераций оптимизации или другой набор данных)")
    
    print("\n" + "=" * 70)
    print("ПРИМЕР ЗАВЕРШЕН УСПЕШНО")
    print("=" * 70)
    print("\nСгенерированные файлы:")
    print("  - time_series_features.png : график исходного ряда и признаков")
    print("  - feature_engineer.pkl     : сохраненное состояние инженера")
    print("\nРекомендации для дальнейшего использования:")
    print("  1. Для реальных данных увеличьте n_calls до 30-50 для лучшей оптимизации")
    print("  2. Используйте параметр 'shap_selection=True' для более агрессивного отбора")
    print("  3. Интегрируйте AutoFeatureEngineer в sklearn Pipeline для удобства")
    print("  4. Сохраняйте состояние инженера для применения к новым данным без повторной оптимизации")


if __name__ == "__main__":
    # Проверка наличия необходимых библиотек для визуализации
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print("WARNING: matplotlib не установлен. Визуализация будет пропущена.")
        print("Установите через: pip install matplotlib")
    
    main()