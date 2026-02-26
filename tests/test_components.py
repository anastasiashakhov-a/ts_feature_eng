"""
Быстрый тест всех компонентов ts_feature_eng.

Проверяет работоспособность всех ключевых модулей за 1-2 минуты:
- Трансформеры (Lag, Window, DWT, STL, TimeEncoding, Calendar)
- AutoFeatureEngineer
- Селекторы признаков
- Извлечение мета-признаков
- Базовый пайплайн
- Отслеживание времени экспериментов (НОВОЕ!)
- Сохранение результатов в CSV/JSON (НОВОЕ!)

Идеально для:
- Проверки после изменений в коде
- CI/CD пайплайнов
- Быстрой валидации перед долгими экспериментами
"""

import os
import sys
import time
import json
import traceback
import numpy as np
import pandas as pd
from datetime import datetime

# ============================================================================
# КОНФИГУРАЦИЯ ТЕСТА
# ============================================================================
TEST_CONFIG = {
    "n_samples": 500,           # Маленькая выборка для скорости
    "n_calls": 2,               # Минимум итераций оптимизации
    "n_initial_points": 1,      # Минимум начальных точек
    "random_state": 42,
    "verbose": 1,
    "results_dir": "test_results",  # Директория для тестовых результатов
}


class TestResult:
    """Класс для хранения результатов теста."""
    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.error = None
        self.duration = 0.0
        self.details = ""
    
    def __str__(self):
        status = "✓ PASS" if self.passed else "✗ FAIL"
        return f"{status} | {self.name} ({self.duration:.2f}s)"


def create_test_data(n_samples=500, freq="H"):
    """Создание тестовых временных рядов."""
    dates = pd.date_range("2023-01-01", periods=n_samples, freq=freq)
    
    # Синусоида + тренд + шум
    t = np.arange(n_samples)
    signal = (
        10 * np.sin(2 * np.pi * t / 24) +      # Суточная сезонность
        5 * np.sin(2 * np.pi * t / 168) +      # Недельная сезонность
        0.01 * t +                              # Тренд
        np.random.randn(n_samples) * 2          # Шум
    )
    
    X = pd.DataFrame({"value": signal}, index=dates)
    y = pd.Series(signal[1:], index=dates[1:], name="target")
    X = X.iloc[:-1]  # Выравниваем длины
    
    return X, y


# ============================================================================
# ТЕСТЫ КОМПОНЕНТОВ
# ============================================================================

def test_lag_transformer(X, y):
    """Тест LagTransformer."""
    result = TestResult("LagTransformer")
    start = time.time()
    try:
        from ts_feature_eng.transformers.lag import LagTransformer
        
        transformer = LagTransformer(lags=[1, 24, 168])
        X_transformed = transformer.fit_transform(X, y)
        
        # Проверки
        assert X_transformed.shape[0] == X.shape[0], "Неверное количество строк"
        assert X_transformed.shape[1] == 3, f"Ожидалось 3 признака, получено {X_transformed.shape[1]}"
        assert "value_lag_1" in X_transformed.columns, "Отсутствует lag_1"
        
        result.passed = True
        result.details = f"Сгенерировано {X_transformed.shape[1]} признаков"
    except Exception as e:
        result.error = str(e)
    finally:
        result.duration = time.time() - start
    
    return result


def test_window_transformer(X, y):
    """Тест WindowTransformer."""
    result = TestResult("WindowTransformer")
    start = time.time()
    try:
        from ts_feature_eng.transformers.window import WindowTransformer
        
        transformer = WindowTransformer(
            window_size=24,
            transformations=["identity", "diff"],
            statistics=["mean", "std", "min", "max"]
        )
        X_transformed = transformer.fit_transform(X, y)
        
        # Проверки
        assert X_transformed.shape[0] == X.shape[0], "Неверное количество строк"
        assert X_transformed.shape[1] > 0, "Признаки не сгенерированы"
        
        result.passed = True
        result.details = f"Сгенерировано {X_transformed.shape[1]} признаков"
    except Exception as e:
        result.error = str(e)
    finally:
        result.duration = time.time() - start
    
    return result


def test_dwt_transformer(X, y):
    """Тест DWTTransformer (быстрый)."""
    result = TestResult("DWTTransformer")
    start = time.time()
    try:
        from ts_feature_eng.transformers.spectral import DWTTransformer
        
        transformer = DWTTransformer(
            wavelet="db4",
            max_level=2,  # Минимум для скорости
            statistics=["mean", "std", "energy"]
        )
        X_transformed = transformer.fit_transform(X, y)
        
        # Проверки
        assert X_transformed.shape[0] == X.shape[0], "Неверное количество строк"
        assert X_transformed.shape[1] > 0, "Признаки не сгенерированы"
        
        result.passed = True
        result.details = f"Сгенерировано {X_transformed.shape[1]} признаков"
    except Exception as e:
        result.error = str(e)
    finally:
        result.duration = time.time() - start
    
    return result


def test_stl_transformer(X, y):
    """Тест STLTransformer (быстрый)."""
    result = TestResult("STLTransformer")
    start = time.time()
    try:
        from ts_feature_eng.transformers.spectral import STLTransformer
        
        transformer = STLTransformer(
            period=24,
            seasonal=7,
            statistics=["mean", "std"]
        )
        X_transformed = transformer.fit_transform(X, y)
        
        # Проверки
        assert X_transformed.shape[0] == X.shape[0], "Неверное количество строк"
        assert X_transformed.shape[1] > 0, "Признаки не сгенерированы"
        
        result.passed = True
        result.details = f"Сгенерировано {X_transformed.shape[1]} признаков"
    except Exception as e:
        result.error = str(e)
    finally:
        result.duration = time.time() - start
    
    return result


def test_time_encoding_transformer(X, y):
    """Тест TimeEncodingTransformer."""
    result = TestResult("TimeEncodingTransformer")
    start = time.time()
    try:
        from ts_feature_eng.transformers.time_encoding import TimeEncodingTransformer
        
        transformer = TimeEncodingTransformer(
            mode="cyclic",
            cyclic_components=["hour", "day_of_week"]
        )
        X_transformed = transformer.fit_transform(X, y)
        
        # Проверки
        assert X_transformed.shape[0] == X.shape[0], "Неверное количество строк"
        assert "time.hour_sin" in X_transformed.columns, "Отсутствует hour_sin"
        assert "time.hour_cos" in X_transformed.columns, "Отсутствует hour_cos"
        
        result.passed = True
        result.details = f"Сгенерировано {X_transformed.shape[1]} признаков"
    except Exception as e:
        result.error = str(e)
    finally:
        result.duration = time.time() - start
    
    return result


def test_calendar_transformer(X, y):
    """Тест CalendarFeaturesTransformer."""
    result = TestResult("CalendarFeaturesTransformer")
    start = time.time()
    try:
        from ts_feature_eng.transformers.time_encoding import CalendarFeaturesTransformer
        
        transformer = CalendarFeaturesTransformer(
            features=["part_of_day", "is_weekend", "hour"]
        )
        X_transformed = transformer.fit_transform(X, y)
        
        # Проверки
        assert X_transformed.shape[0] == X.shape[0], "Неверное количество строк"
        assert "time.part_of_day" in X_transformed.columns, "Отсутствует part_of_day"
        
        result.passed = True
        result.details = f"Сгенерировано {X_transformed.shape[1]} признаков"
    except Exception as e:
        result.error = str(e)
    finally:
        result.duration = time.time() - start
    
    return result


def test_meta_features(X, y):
    """Тест MetaFeatureExtractor."""
    result = TestResult("MetaFeatureExtractor")
    start = time.time()
    try:
        from ts_feature_eng.meta_features import MetaFeatureExtractor
        
        extractor = MetaFeatureExtractor(
            categories=["simple", "statistical"],
            fill_method="linear"
        )
        meta_df = extractor.fit_transform(X, y)
        
        # Проверки
        assert len(meta_df.columns) > 0, "Мета-признаки не извлечены"
        assert "length" in meta_df.columns, "Отсутствует length"
        assert "stationarity_adf" in meta_df.columns, "Отсутствует stationarity_adf"
        
        result.passed = True
        result.details = f"Извлечено {len(meta_df.columns)} мета-признаков"
    except Exception as e:
        result.error = str(e)
    finally:
        result.duration = time.time() - start
    
    return result


def test_auto_feature_engineer(X, y):
    """Тест AutoFeatureEngineer (минимальный)."""
    result = TestResult("AutoFeatureEngineer")
    start = time.time()
    try:
        from ts_feature_eng import AutoFeatureEngineer
        
        engineer = AutoFeatureEngineer(
            optimize=True,
            n_calls=TEST_CONFIG["n_calls"],
            n_initial_points=TEST_CONFIG["n_initial_points"],
            apply_selection=True,
            selection_threshold=0.3,
            variance_threshold=0.01,
            shap_selection=False,
            random_state=TEST_CONFIG["random_state"],
            verbose=0
        )
        X_transformed = engineer.fit_transform(X, y)
        
        # Проверки
        assert X_transformed.shape[0] == X.shape[0], "Неверное количество строк"
        assert X_transformed.shape[1] > 0, "Признаки не сгенерированы"
        
        result.passed = True
        result.details = f"Сгенерировано {X_transformed.shape[1]} признаков"
    except Exception as e:
        result.error = str(e)
    finally:
        result.duration = time.time() - start
    
    return result


def test_feature_selector(X, y):
    """Тест CombinedFeatureSelector."""
    result = TestResult("CombinedFeatureSelector")
    start = time.time()
    try:
        from ts_feature_eng.selection import CombinedFeatureSelector
        
        # Сначала создаём признаки
        from ts_feature_eng.transformers.window import WindowTransformer
        transformer = WindowTransformer(
            window_size=24,
            statistics=["mean", "std", "min", "max"]
        )
        X_transformed = transformer.fit_transform(X, y)
        
        # Затем отбираем
        selector = CombinedFeatureSelector(
            missing_threshold=0.2,
            variance_threshold=0.01,
            skip_selection=True  # Пропускаем SHAP для скорости
        )
        selector.fit(X_transformed, y)
        X_selected = selector.transform(X_transformed)
        
        # Проверки
        assert X_selected.shape[0] == X_transformed.shape[0], "Неверное количество строк"
        assert X_selected.shape[1] > 0, "Все признаки отфильтрованы"
        
        result.passed = True
        result.details = f"Отобрано {X_selected.shape[1]} из {X_transformed.shape[1]} признаков"
    except Exception as e:
        result.error = str(e)
    finally:
        result.duration = time.time() - start
    
    return result


def test_pipeline_integration(X, y):
    """Тест полной интеграции пайплайна."""
    result = TestResult("Pipeline Integration")
    start = time.time()
    try:
        from ts_feature_eng import AutoFeatureEngineer
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.metrics import mean_absolute_error
        
        # 1. Инженерия признаков
        engineer = AutoFeatureEngineer(
            optimize=False,  # Без оптимизации для скорости
            apply_selection=True,
            selection_threshold=0.3,
            variance_threshold=0.01,
            shap_selection=False,
            random_state=TEST_CONFIG["random_state"],
            verbose=0
        )
        X_transformed = engineer.fit_transform(X, y)
        
        # 2. Разделение
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X_transformed.iloc[:split_idx], X_transformed.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # 3. Обучение модели
        model = GradientBoostingRegressor(
            n_estimators=10,
            max_depth=2,
            random_state=TEST_CONFIG["random_state"]
        )
        model.fit(X_train.fillna(0), y_train)
        
        # 4. Прогноз
        y_pred = model.predict(X_test.fillna(0))
        mae = mean_absolute_error(y_test, y_pred)
        
        # Проверки
        assert len(y_pred) == len(y_test), "Неверная длина прогноза"
        assert mae < np.std(y_test), f"MAE слишком высокий: {mae}"
        
        result.passed = True
        result.details = f"MAE={mae:.2f}, признаков={X_transformed.shape[1]}"
    except Exception as e:
        result.error = str(e)
    finally:
        result.duration = time.time() - start
    
    return result


# ============================================================================
# НОВЫЕ ТЕСТЫ: ОТСЛЕЖИВАНИЕ ВРЕМЕНИ И СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
# ============================================================================

def test_experiment_timing():
    """Тест отслеживания времени эксперимента."""
    result = TestResult("Experiment Timing")
    start = time.time()
    try:
        from experiment_tracker import ExperimentTracker
        
        # Создаём трекер
        tracker = ExperimentTracker(experiment_id="test_exp_001")
        
        # Записываем время начала
        tracker.start()
        time.sleep(0.5)  # Имитация работы
        
        # Записываем время окончания
        tracker.stop()
        
        # Проверки
        assert tracker.start_time is not None, "Время начала не записано"
        assert tracker.end_time is not None, "Время окончания не записано"
        assert tracker.duration_seconds >= 0.5, f"Длительность слишком короткая: {tracker.duration_seconds}s"
        assert tracker.duration_formatted, "Форматированная длительность пуста"
        
        result.passed = True
        result.details = f"Длительность: {tracker.duration_formatted} ({tracker.duration_seconds:.2f}s)"
    except Exception as e:
        result.error = str(e)
        traceback.print_exc()
    finally:
        result.duration = time.time() - start
    
    return result


def test_csv_results_save():
    """Тест сохранения результатов в CSV."""
    result = TestResult("CSV Results Save")
    start = time.time()
    try:
        import os
        from results_manager import ResultsManager
        
        # Создаём директорию для тестов
        os.makedirs(TEST_CONFIG["results_dir"], exist_ok=True)
        
        # Создаём менеджер результатов
        manager = ResultsManager(results_dir=TEST_CONFIG["results_dir"])
        
        # Тестовые метрики
        metrics = {
            "MAE": 0.05,
            "RMSE": 0.07,
            "R2": 0.95,
            "n_features": 50,
        }
        
        # Сохраняем результаты
        manager.save_metrics(
            experiment_id="test_csv_001",
            horizon=1,
            metrics=metrics,
            duration_seconds=10.5
        )
        
        # Проверяем файл
        metrics_file = os.path.join(TEST_CONFIG["results_dir"], "metrics_history.csv")
        assert os.path.exists(metrics_file), "CSV файл не создан"
        
        # Читаем и проверяем содержимое
        df = pd.read_csv(metrics_file)
        assert "duration_seconds" in df.columns, "Отсутствует колонка duration_seconds"
        assert "duration_formatted" in df.columns, "Отсутствует колонка duration_formatted"
        assert len(df) > 0, "CSV файл пуст"
        
        # Проверяем значение длительности
        last_row = df.iloc[-1]
        assert last_row["duration_seconds"] == 10.5, f"Неверная длительность: {last_row['duration_seconds']}"
        assert "0ч 0м 10с" in last_row["duration_formatted"], f"Неверный формат: {last_row['duration_formatted']}"
        
        result.passed = True
        result.details = f"CSV сохранён: {len(df)} записей, duration_seconds={last_row['duration_seconds']}"
    except Exception as e:
        result.error = str(e)
        traceback.print_exc()
    finally:
        result.duration = time.time() - start
    
    return result


def test_json_summary_save():
    """Тест сохранения JSON отчёта."""
    result = TestResult("JSON Summary Save")
    start = time.time()
    try:
        import os
        from results_manager import ResultsManager
        from datetime import datetime
        
        # Создаём директорию для тестов
        os.makedirs(TEST_CONFIG["results_dir"], exist_ok=True)
        
        # Создаём менеджер результатов
        manager = ResultsManager(results_dir=TEST_CONFIG["results_dir"])
        
        # Тестовые данные
        start_time = datetime.now()
        time.sleep(0.5)
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        results_data = [
            {"horizon": 1, "mae": 0.05, "r2": 0.95},
            {"horizon": 7, "mae": 0.08, "r2": 0.90},
        ]
        
        # Сохраняем JSON отчёт
        manager.save_summary(
            experiment_id="test_json_001",
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration,
            horizons_tested=[1, 7],
            config={"n_calls": 2, "random_state": 42},
            results=results_data
        )
        
        # Проверяем файл
        summary_file = os.path.join(TEST_CONFIG["results_dir"], "experiment_test_json_001_summary.json")
        assert os.path.exists(summary_file), "JSON файл не создан"
        
        # Читаем и проверяем содержимое
        with open(summary_file, "r", encoding="utf-8") as f:
            summary = json.load(f)
        
        assert "experiment_id" in summary, "Отсутствует experiment_id"
        assert "start_time" in summary, "Отсутствует start_time"
        assert "end_time" in summary, "Отсутствует end_time"
        assert "duration_seconds" in summary, "Отсутствует duration_seconds"
        assert "duration_formatted" in summary, "Отсутствует duration_formatted"
        assert "results" in summary, "Отсутствует results"
        assert summary["experiment_id"] == "test_json_001", "Неверный experiment_id"
        assert summary["duration_seconds"] >= 0.5, f"Длительность слишком короткая: {summary['duration_seconds']}"
        
        result.passed = True
        result.details = f"JSON сохранён: duration={summary['duration_formatted']}"
    except Exception as e:
        result.error = str(e)
        traceback.print_exc()
    finally:
        result.duration = time.time() - start
    
    return result


def test_duration_formatting():
    """Тест форматирования длительности."""
    result = TestResult("Duration Formatting")
    start = time.time()
    try:
        from experiment_tracker import format_duration
        
        # Тестовые случаи
        test_cases = [
            (0.5, "0ч 0м 0с"),
            (10.5, "0ч 0м 10с"),
            (65.0, "0ч 1м 5с"),
            (3661.0, "1ч 1м 1с"),
            (7325.0, "2ч 2м 5с"),
        ]
        
        for seconds, expected in test_cases:
            formatted = format_duration(seconds)
            assert formatted == expected, f"Для {seconds}s ожидалось '{expected}', получено '{formatted}'"
        
        result.passed = True
        result.details = f"Все {len(test_cases)} тестовых случаев пройдены"
    except Exception as e:
        result.error = str(e)
        traceback.print_exc()
    finally:
        result.duration = time.time() - start
    
    return result


def test_csv_update_duration():
    """Тест обновления длительности в существующем CSV."""
    result = TestResult("CSV Update Duration")
    start = time.time()
    try:
        import os
        from results_manager import ResultsManager
        
        # Создаём директорию для тестов
        os.makedirs(TEST_CONFIG["results_dir"], exist_ok=True)
        
        # Создаём менеджер результатов
        manager = ResultsManager(results_dir=TEST_CONFIG["results_dir"])
        
        # Сначала сохраняем без длительности
        manager.save_metrics(
            experiment_id="test_update_001",
            horizon=1,
            metrics={"MAE": 0.05},
            duration_seconds=None  # Пока None
        )
        
        # Затем обновляем с длительностью
        manager.update_experiment_duration(
            experiment_id="test_update_001",
            duration_seconds=15.7
        )
        
        # Проверяем файл
        metrics_file = os.path.join(TEST_CONFIG["results_dir"], "metrics_history.csv")
        df = pd.read_csv(metrics_file)
        
        # Находим строку с нашим experiment_id
        mask = df["experiment_id"] == "test_update_001"
        assert mask.any(), "Запись не найдена в CSV"
        
        row = df[mask].iloc[-1]
        assert row["duration_seconds"] == 15.7, f"Длительность не обновилась: {row['duration_seconds']}"
        assert "0ч 0м 15с" in row["duration_formatted"], f"Формат неверный: {row['duration_formatted']}"
        
        result.passed = True
        result.details = f"Длительность обновлена: {row['duration_formatted']}"
    except Exception as e:
        result.error = str(e)
        traceback.print_exc()
    finally:
        result.duration = time.time() - start
    
    return result


# ============================================================================
# ЗАПУСК ТЕСТОВ
# ============================================================================

def run_all_tests():
    """Запуск всех тестов."""
    print("=" * 80)
    print("БЫСТРЫЙ ТЕСТ ВСЕХ КОМПОНЕНТОВ ts_feature_eng")
    print("=" * 80)
    print(f"Время запуска: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Конфигурация: {TEST_CONFIG['n_samples']} сэмплов, {TEST_CONFIG['n_calls']} итераций оптимизации")
    print("=" * 80)
    
    # Создание тестовых данных
    print("\n[0/14] Создание тестовых данных...")
    start = time.time()
    X, y = create_test_data(n_samples=TEST_CONFIG["n_samples"])
    print(f"   ✓ Данные созданы: {X.shape[0]} наблюдений, {X.shape[1]} признаков")
    print(f"   Время: {time.time() - start:.2f}s")
    
    # Список тестов
    tests = [
        # Базовые трансформеры
        ("LagTransformer", test_lag_transformer),
        ("WindowTransformer", test_window_transformer),
        ("DWTTransformer", test_dwt_transformer),
        ("STLTransformer", test_stl_transformer),
        ("TimeEncodingTransformer", test_time_encoding_transformer),
        ("CalendarFeaturesTransformer", test_calendar_transformer),
        
        # Мета-признаки и пайплайн
        ("MetaFeatureExtractor", test_meta_features),
        ("AutoFeatureEngineer", test_auto_feature_engineer),
        ("CombinedFeatureSelector", test_feature_selector),
        ("Pipeline Integration", test_pipeline_integration),
        
        # Тесты времени и сохранения
        ("Experiment Timing", test_experiment_timing),
        ("Duration Formatting", test_duration_formatting),
        ("CSV Results Save", test_csv_results_save),
        ("JSON Summary Save", test_json_summary_save),
        ("CSV Update Duration", test_csv_update_duration),
    ]
    
    # Запуск тестов
    results = []
    print("\n" + "=" * 80)
    print("ЗАПУСК ТЕСТОВ")
    print("=" * 80)
    
    for i, (name, test_func) in enumerate(tests, 1):
        print(f"\n[{i}/{len(tests)}] Тест: {name}...")
        result = test_func(X, y) if test_func.__name__.startswith("test_") and "timing" not in test_func.__name__ and "csv" not in test_func.__name__ and "json" not in test_func.__name__ and "duration" not in test_func.__name__ else test_func()
        results.append(result)
        print(f"   {result}")
        if not result.passed:
            print(f"   Ошибка: {result.error}")
    
    # Итоговый отчёт
    print("\n" + "=" * 80)
    print("ИТОГОВЫЙ ОТЧЁТ")
    print("=" * 80)
    
    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed
    total_duration = sum(r.duration for r in results)
    
    # Группировка по категориям
    core_tests = [r for r in results if r.name in ["LagTransformer", "WindowTransformer", "DWTTransformer", "STLTransformer", "TimeEncodingTransformer", "CalendarFeaturesTransformer", "MetaFeatureExtractor", "AutoFeatureEngineer", "CombinedFeatureSelector", "Pipeline Integration"]]
    timing_tests = [r for r in results if r.name in ["Experiment Timing", "Duration Formatting", "CSV Results Save", "JSON Summary Save", "CSV Update Duration"]]
    
    print(f"\n ВСЕГО:")
    print(f"   Всего тестов: {len(results)}")
    print(f"   ✓ Пройдено: {passed}")
    print(f"   ✗ Провалено: {failed}")
    print(f"   Общее время: {total_duration:.2f}s ({total_duration/60:.2f} мин)")
    
    print(f"\n БАЗОВЫЕ КОМПОНЕНТЫ ({len(core_tests)}):")
    core_passed = sum(1 for r in core_tests if r.passed)
    print(f"   ✓ Пройдено: {core_passed}/{len(core_tests)}")
    
    print(f"\n ОТСЛЕЖИВАНИЕ ВРЕМЕНИ ({len(timing_tests)}):")
    timing_passed = sum(1 for r in timing_tests if r.passed)
    print(f"   ✓ Пройдено: {timing_passed}/{len(timing_tests)}")
    
    print("\n Детали:")
    for result in results:
        status = "✓" if result.passed else "✗"
        print(f"  {status} {result.name}: {result.details if result.passed else result.error}")
    
    # Сохранение отчёта
    report_file = "test_report.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("ОТЧЁТ О ТЕСТИРОВАНИИ ts_feature_eng\n")
        f.write("=" * 80 + "\n")
        f.write(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Всего тестов: {len(results)}\n")
        f.write(f"Пройдено: {passed}\n")
        f.write(f"Провалено: {failed}\n")
        f.write(f"Общее время: {total_duration:.2f}s\n\n")
        f.write("Детали:\n")
        for result in results:
            status = "PASS" if result.passed else "FAIL"
            f.write(f"  [{status}] {result.name}: {result.details if result.passed else result.error}\n")
    
    print(f"\n Отчёт сохранён в: {report_file}")
    
    print("\n" + "=" * 80)
    if failed == 0:
        print(" ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
    else:
        print(f" {failed} ТЕСТ(ОВ) ПРОВАЛЕНО!")
        print("Проверьте ошибки выше перед запуском долгих экспериментов.")
    print("=" * 80)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)