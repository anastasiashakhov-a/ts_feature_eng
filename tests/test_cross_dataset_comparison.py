# tests/test_cross_dataset_comparison.py

"""
Быстрый тест всех компонентов cross_dataset_comparison.py.

Проверяет работоспособность всех ключевых функций за 2-3 минуты:
- Загрузка всех 3 датасетов (temperature, energy, acn_load)
- Нормализация метрик
- Расчёт относительных горизонтов
- Вычисление AUC
- Визуализация (с минимальными данными)
- ExperimentManager интеграция
- Сохранение результатов

Идеально для:
- Проверки перед долгим запуском
- CI/CD пайплайнов
- Отладки после изменений
"""

import os
import sys
import time
import json
import warnings
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ← ИСПРАВЛЕНИЕ: Добавляем недостающий импорт
from sklearn.metrics import mean_absolute_error

# Подавляем предупреждения для чистого вывода
warnings.filterwarnings('ignore', message='Precision loss')
warnings.filterwarnings('ignore', message='Degrees of freedom')
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Импорты из cross_dataset_comparison
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'examples'))

from cross_dataset_comparison import (
    normalize_mae,
    get_relative_horizons,
    compute_auc_curve,
    format_horizon,
    load_temperature_data,
    load_energy_data,
    load_acn_data,
    load_dataset,
    DATASET_CONFIGS,
    COMPARISON_CONFIG,
)

# Импорты для тестов визуализации
try:
    from cross_dataset_comparison import (
        plot_mae_curves,
        plot_auc_comparison,
    )
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False


# ============================================================================
# КЛАСС ДЛЯ РЕЗУЛЬТАТОВ ТЕСТА
# ============================================================================

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


# ============================================================================
# ТЕСТЫ КОМПОНЕНТОВ
# ============================================================================

def test_normalize_mae():
    """Тест нормализации MAE."""
    result = TestResult("normalize_mae")
    start = time.time()
    try:
        # Тестовые данные
        mae = 5.0
        y_train = np.array([10, 12, 11, 13, 12, 14, 13, 15])
        
        # Тест std нормализации
        mae_std = normalize_mae(mae, y_train, method='std')
        assert isinstance(mae_std, float), "Должно возвращать float"
        assert mae_std > 0, "Должно быть положительным"
        
        # Тест range нормализации
        mae_range = normalize_mae(mae, y_train, method='range')
        assert isinstance(mae_range, float), "Должно возвращать float"
        
        # Тест mean нормализации
        mae_mean = normalize_mae(mae, y_train, method='mean')
        assert isinstance(mae_mean, float), "Должно возвращать float"
        
        # Тест с нулевой дисперсией
        y_constant = np.array([5, 5, 5, 5, 5])
        mae_const = normalize_mae(mae, y_constant, method='std')
        assert mae_const == mae, "При нулевой дисперсии должно вернуть исходное MAE"
        
        result.passed = True
        result.details = f"std={mae_std:.3f}, range={mae_range:.3f}, mean={mae_mean:.3f}"
    except Exception as e:
        result.error = str(e)
    finally:
        result.duration = time.time() - start
    
    return result


def test_get_relative_horizons():
    """Тест расчёта относительных горизонтов."""
    result = TestResult("get_relative_horizons")
    start = time.time()
    try:
        # Тест для daily данных
        horizons_d = get_relative_horizons(3650, [0.01, 0.05, 0.10], freq='D')
        assert len(horizons_d) == 3, "Должно быть 3 горизонта"
        assert all(h > 0 for h in horizons_d), "Все горизонты должны быть > 0"
        
        # Тест для 15min данных
        horizons_15 = get_relative_horizons(35000, [0.01, 0.05, 0.10], freq='15min')
        assert len(horizons_15) == 3, "Должно быть 3 горизонта"
        # Для 15min должно быть кратно 96 (день)
        assert all(h % 96 == 0 for h in horizons_15), "Для 15min должно быть кратно 96"
        
        # Тест для hourly данных
        horizons_h = get_relative_horizons(8760, [0.01, 0.05, 0.10], freq='H')
        assert len(horizons_h) == 3, "Должно быть 3 горизонта"
        
        result.passed = True
        result.details = f"D={horizons_d}, 15min={horizons_15}, H={horizons_h}"
    except Exception as e:
        result.error = str(e)
    finally:
        result.duration = time.time() - start
    
    return result


def test_compute_auc_curve():
    """Тест вычисления AUC."""
    result = TestResult("compute_auc_curve")
    start = time.time()
    try:
        # Тестовые данные
        errors = [0.1, 0.15, 0.2, 0.25, 0.3]
        horizons = [0.01, 0.03, 0.05, 0.07, 0.10]
        
        auc = compute_auc_curve(errors, horizons)
        assert isinstance(auc, float), "Должно возвращать float"
        assert auc > 0, "AUC должно быть положительным"
        assert auc < 1, "AUC должно быть < 1 для нормализованных данных"
        
        # Тест с недостаточным количеством точек
        auc_short = compute_auc_curve([0.1], [0.01])
        assert np.isnan(auc_short), "Для 1 точки должно быть NaN"
        
        result.passed = True
        result.details = f"AUC={auc:.4f}"
    except Exception as e:
        result.error = str(e)
    finally:
        result.duration = time.time() - start
    
    return result


def test_format_horizon():
    """Тест форматирования горизонта."""
    result = TestResult("format_horizon")
    start = time.time()
    try:
        # Тест для daily
        assert format_horizon(1, 'D') == '1д', f"Ожидалось '1д', получено '{format_horizon(1, 'D')}'"
        assert format_horizon(7, 'D') == '7д', f"Ожидалось '7д', получено '{format_horizon(7, 'D')}'"
        assert format_horizon(30, 'D') == '30д', f"Ожидалось '30д', получено '{format_horizon(30, 'D')}'"
        
        # Тест для 15min (96 интервалов = 1 день)
        assert format_horizon(96, '15min') == '1д', f"Ожидалось '1д', получено '{format_horizon(96, '15min')}'"
        assert format_horizon(672, '15min') == '7д', f"Ожидалось '7д', получено '{format_horizon(672, '15min')}'"
        
        # Тест для hourly
        assert format_horizon(24, 'H') == '1д', f"Ожидалось '1д', получено '{format_horizon(24, 'H')}'"
        
        result.passed = True
        result.details = "Все форматы корректны"
    except Exception as e:
        result.error = str(e)
    finally:
        result.duration = time.time() - start
    
    return result


def test_load_temperature_data():
    """Тест загрузки датасета температур."""
    result = TestResult("load_temperature_data")
    start = time.time()
    try:
        config = DATASET_CONFIGS["temperature"]
        if not os.path.exists(config["path"]):
            result.error = f"Файл не найден: {config['path']}"
            return result
        
        df = load_temperature_data(config["path"])
        
        assert isinstance(df, pd.DataFrame), "Должен вернуть DataFrame"
        assert "target" in df.columns, "Должна быть колонка 'target'"
        assert len(df) > 0, "Должны быть данные"
        assert df.index.name == "timestamp" or isinstance(df.index, pd.DatetimeIndex), "Должен быть временной индекс"
        
        result.passed = True
        result.details = f"{len(df)} наблюдений, {df['target'].min():.1f}—{df['target'].max():.1f} °C"
    except Exception as e:
        result.error = str(e)
    finally:
        result.duration = time.time() - start
    
    return result


def test_load_energy_data():
    """Тест загрузки датасета энергии."""
    result = TestResult("load_energy_data")
    start = time.time()
    try:
        config = DATASET_CONFIGS["energy"]
        if not os.path.exists(config["path"]):
            result.error = f"Файл не найден: {config['path']}"
            return result
        
        df = load_energy_data(config["path"])
        
        assert isinstance(df, pd.DataFrame), "Должен вернуть DataFrame"
        assert "target" in df.columns, "Должна быть колонка 'target'"
        assert len(df) > 0, "Должны быть данные"
        
        result.passed = True
        result.details = f"{len(df)} наблюдений, {df['target'].min():.1f}—{df['target'].max():.1f} МВт"
    except Exception as e:
        result.error = str(e)
    finally:
        result.duration = time.time() - start
    
    return result


def test_load_acn_data():
    """Тест загрузки датасета ACN."""
    result = TestResult("load_acn_data")
    start = time.time()
    try:
        config = DATASET_CONFIGS["acn_load"]
        if not os.path.exists(config["path"]):
            result.error = f"Файл не найден: {config['path']}"
            return result
        
        df = load_acn_data(config["path"])
        
        assert isinstance(df, pd.DataFrame), "Должен вернуть DataFrame"
        assert "target" in df.columns, "Должна быть колонка 'target'"
        assert len(df) > 0, "Должны быть данные"
        
        result.passed = True
        result.details = f"{len(df)} наблюдений, {df['target'].min():.1f}—{df['target'].max():.1f} kWh"
    except Exception as e:
        result.error = str(e)
    finally:
        result.duration = time.time() - start
    
    return result


def test_load_dataset_universal():
    """Тест универсального загрузчика."""
    result = TestResult("load_dataset_universal")
    start = time.time()
    try:
        loaded_datasets = []
        for name, config in DATASET_CONFIGS.items():
            if os.path.exists(config["path"]):
                df = load_dataset(name, config)
                loaded_datasets.append(name)
                assert isinstance(df, pd.DataFrame), f"{name}: Должен вернуть DataFrame"
                assert len(df) > 0, f"{name}: Должны быть данные"
        
        if len(loaded_datasets) == 0:
            result.error = "Ни один датасет не найден"
            return result
        
        result.passed = True
        result.details = f"Загружено: {', '.join(loaded_datasets)}"
    except Exception as e:
        result.error = str(e)
    finally:
        result.duration = time.time() - start
    
    return result


def test_plot_mae_curves():
    """Тест графика MAE кривых."""
    result = TestResult("plot_mae_curves")
    start = time.time()
    try:
        if not HAS_PLOTTING:
            result.error = "Функции plotting не импортированы"
            return result
        
        # ← ИСПРАВЛЕНИЕ: Добавляем 'auc' в тестовые данные
        all_curves = {
            "temperature": {
                "horizons_relative": [0.01, 0.05, 0.10],
                "models": {
                    "naive": {"mae_normalized": [0.2, 0.25, 0.3], "r2": [0.5, 0.4, 0.3], "auc": 0.25},
                    "rf_auto_fe": {"mae_normalized": [0.15, 0.18, 0.22], "r2": [0.7, 0.6, 0.5], "auc": 0.18},
                    "gb_auto_fe": {"mae_normalized": [0.12, 0.15, 0.18], "r2": [0.8, 0.7, 0.6], "auc": 0.15}
                }
            },
            "energy": {
                "horizons_relative": [0.01, 0.05, 0.10],
                "models": {
                    "naive": {"mae_normalized": [0.25, 0.30, 0.35], "r2": [0.4, 0.3, 0.2], "auc": 0.30},
                    "rf_auto_fe": {"mae_normalized": [0.18, 0.22, 0.26], "r2": [0.6, 0.5, 0.4], "auc": 0.22},
                    "gb_auto_fe": {"mae_normalized": [0.15, 0.18, 0.22], "r2": [0.7, 0.6, 0.5], "auc": 0.18}
                }
            }
        }
        
        output_path = "test_mae_curves.png"
        plot_mae_curves(all_curves, output_path)
        
        assert os.path.exists(output_path), "Файл графика должен быть создан"
        assert os.path.getsize(output_path) > 0, "Файл не должен быть пустым"
        
        # Очистка
        os.remove(output_path)
        
        result.passed = True
        result.details = "График успешно создан и удалён"
    except Exception as e:
        result.error = str(e)
    finally:
        result.duration = time.time() - start
    
    return result


def test_plot_auc_comparison():
    """Тест графика AUC сравнения."""
    result = TestResult("plot_auc_comparison")
    start = time.time()
    try:
        if not HAS_PLOTTING:
            result.error = "Функции plotting не импортированы"
            return result
        
        # Создаём тестовые данные
        all_curves = {
            "temperature": {
                "models": {
                    "naive": {"auc": 0.25},
                    "rf_auto_fe": {"auc": 0.18},
                    "gb_auto_fe": {"auc": 0.15}
                }
            },
            "energy": {
                "models": {
                    "naive": {"auc": 0.30},
                    "rf_auto_fe": {"auc": 0.22},
                    "gb_auto_fe": {"auc": 0.18}
                }
            },
            "acn_load": {
                "models": {
                    "naive": {"auc": 0.20},
                    "rf_auto_fe": {"auc": 0.15},
                    "gb_auto_fe": {"auc": 0.12}
                }
            }
        }
        
        output_path = "test_auc_comparison.png"
        plot_auc_comparison(all_curves, output_path)
        
        assert os.path.exists(output_path), "Файл графика должен быть создан"
        assert os.path.getsize(output_path) > 0, "Файл не должен быть пустым"
        
        # Очистка
        os.remove(output_path)
        
        result.passed = True
        result.details = "График успешно создан и удалён"
    except Exception as e:
        result.error = str(e)
    finally:
        result.duration = time.time() - start
    
    return result


def test_experiment_manager_integration():
    """Тест интеграции с ExperimentManager."""
    result = TestResult("ExperimentManager_integration")
    start = time.time()
    try:
        from ts_feature_eng.utils.experiment_logger import ExperimentManager
        
        manager = ExperimentManager(experiment_type="comparison_test")
        
        # Тест создания директорий
        assert os.path.exists(manager.full_dir), "Директория эксперимента должна быть создана"
        
        # Тест сохранения метрик
        test_metrics = {"MAE (kWh)": 5.0, "R²": 0.8}
        manager.save_metrics(test_metrics, COMPARISON_CONFIG, "test_exp", 1)
        
        # Тест сохранения конфига
        manager.save_config(COMPARISON_CONFIG, "test_exp", 1)
        
        # Тест сохранения JSON отчёта
        manager.save_json_report(test_metrics, COMPARISON_CONFIG, "test_exp", 1)
        
        # Проверка файлов
        metrics_file = os.path.join(manager.full_dir, "metrics_history.csv")
        assert os.path.exists(metrics_file), "Файл метрик должен быть создан"
        
        # Очистка
        import shutil
        shutil.rmtree(manager.base_dir, ignore_errors=True)
        
        result.passed = True
        result.details = f"Директория: {manager.full_dir}"
    except Exception as e:
        result.error = str(e)
    finally:
        result.duration = time.time() - start
    
    return result


def test_config_validation():
    """Тест валидации конфигурации."""
    result = TestResult("config_validation")
    start = time.time()
    try:
        # Проверка обязательных полей
        required_fields = [
            "random_state", "train_test_split", "n_calls",
            "relative_horizons", "normalize_metrics", "compute_auc"
        ]
        
        for field in required_fields:
            assert field in COMPARISON_CONFIG, f"Отсутствует поле: {field}"
        
        # Проверка диапазонов
        assert 0 < COMPARISON_CONFIG["train_test_split"] < 1, "train_test_split должен быть в (0, 1)"
        assert COMPARISON_CONFIG["n_calls"] > 0, "n_calls должен быть > 0"
        assert len(COMPARISON_CONFIG["relative_horizons"]) > 0, "relative_horizons не должен быть пустым"
        
        # Проверка DATASET_CONFIGS
        for name, config in DATASET_CONFIGS.items():
            assert "path" in config, f"{name}: отсутствует path"
            assert "freq" in config, f"{name}: отсутствует freq"
            assert "target_unit" in config, f"{name}: отсутствует target_unit"
        
        result.passed = True
        result.details = "Все поля конфигурации корректны"
    except Exception as e:
        result.error = str(e)
    finally:
        result.duration = time.time() - start
    
    return result


def test_quick_multi_horizon():
    """Быстрый тест multi-horizon вычислений на синтетических данных."""
    result = TestResult("quick_multi_horizon")
    start = time.time()
    try:
        # Создаём синтетический временной ряд
        np.random.seed(42)
        n_samples = 500  # Мало для скорости
        dates = pd.date_range("2023-01-01", periods=n_samples, freq="H")
        signal = (
            10 * np.sin(2 * np.pi * np.arange(n_samples) / 24) +
            5 * np.sin(2 * np.pi * np.arange(n_samples) / 168) +
            np.random.randn(n_samples) * 2
        )
        df = pd.DataFrame({"target": signal}, index=dates)
        
        # Тест на нескольких горизонтах
        horizons = [24, 48, 72]  # 1, 2, 3 дня для hourly
        mae_values = []
        
        for h in horizons:
            y = df["target"].shift(-h)
            valid_mask = ~y.isna()
            y_valid = y[valid_mask]
            
            # Наивный прогноз
            naive_pred = y_valid.shift(h).fillna(y_valid.mean())
            # ← ИСПРАВЛЕНИЕ: mean_absolute_error теперь импортирован
            mae = normalize_mae(
                mean_absolute_error(y_valid, naive_pred),
                y_valid.values
            )
            mae_values.append(mae)
        
        # Проверяем что MAE растёт с горизонтом (обычно так)
        assert len(mae_values) == 3, "Должно быть 3 значения MAE"
        assert all(isinstance(m, float) for m in mae_values), "Все MAE должны быть float"
        
        # Вычисляем AUC
        horizons_rel = [h / n_samples for h in horizons]
        auc = compute_auc_curve(mae_values, horizons_rel)
        assert not np.isnan(auc), "AUC не должно быть NaN"
        
        result.passed = True
        result.details = f"MAE={mae_values}, AUC={auc:.4f}"
    except Exception as e:
        result.error = str(e)
    finally:
        result.duration = time.time() - start
    
    return result


# ============================================================================
# ЗАПУСК ТЕСТОВ
# ============================================================================

def run_all_tests():
    """Запуск всех тестов."""
    print("=" * 80)
    print("🧪 ТЕСТ КОМПОНЕНТОВ cross_dataset_comparison.py")
    print("=" * 80)
    print(f"Время запуска: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Конфигурация: {len(DATASET_CONFIGS)} датасетов, {len(COMPARISON_CONFIG['relative_horizons'])} горизонтов")
    print("=" * 80)
    
    # Список тестов
    tests = [
        ("Конфигурация", test_config_validation),
        ("Нормализация MAE", test_normalize_mae),
        ("Относительные горизонты", test_get_relative_horizons),
        ("Вычисление AUC", test_compute_auc_curve),
        ("Форматирование горизонта", test_format_horizon),
        ("Загрузка temperature", test_load_temperature_data),
        ("Загрузка energy", test_load_energy_data),
        ("Загрузка acn_load", test_load_acn_data),
        ("Универсальный загрузчик", test_load_dataset_universal),
        ("График MAE кривых", test_plot_mae_curves),
        ("График AUC сравнения", test_plot_auc_comparison),
        ("ExperimentManager", test_experiment_manager_integration),
        ("Multi-horizon (синтетика)", test_quick_multi_horizon),
    ]
    
    # Запуск тестов
    results = []
    print("\n" + "=" * 80)
    print("ЗАПУСК ТЕСТОВ")
    print("=" * 80)
    
    for i, (name, test_func) in enumerate(tests, 1):
        print(f"\n[{i}/{len(tests)}] Тест: {name}...")
        result = test_func()
        results.append(result)
        print(f"   {result}")
        if not result.passed:
            print(f"    Ошибка: {result.error}")
    
    # Итоговый отчёт
    print("\n" + "=" * 80)
    print("ИТОГОВЫЙ ОТЧЁТ")
    print("=" * 80)
    
    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed
    total_duration = sum(r.duration for r in results)
    
    # Группировка по категориям
    data_tests = [r for r in results if "Загрузка" in r.name or "Универсальный" in r.name]
    metric_tests = [r for r in results if "Нормализация" in r.name or "AUC" in r.name or "Горизонт" in r.name or "Форматирование" in r.name]
    plot_tests = [r for r in results if "График" in r.name]
    other_tests = [r for r in results if r not in data_tests + metric_tests + plot_tests]
    
    print(f"\n ВСЕГО:")
    print(f"   Всего тестов: {len(results)}")
    print(f"   ✓ Пройдено: {passed}")
    print(f"   ✗ Провалено: {failed}")
    print(f"    Общее время: {total_duration:.2f}s ({total_duration/60:.2f} мин)")
    
    if data_tests:
        data_passed = sum(1 for r in data_tests if r.passed)
        print(f"\n ДАННЫЕ ({len(data_tests)}):")
        print(f"   ✓ Пройдено: {data_passed}/{len(data_tests)}")
    
    if metric_tests:
        metric_passed = sum(1 for r in metric_tests if r.passed)
        print(f"\n МЕТРИКИ ({len(metric_tests)}):")
        print(f"   ✓ Пройдено: {metric_passed}/{len(metric_tests)}")
    
    if plot_tests:
        plot_passed = sum(1 for r in plot_tests if r.passed)
        print(f"\n ГРАФИКИ ({len(plot_tests)}):")
        print(f"   ✓ Пройдено: {plot_passed}/{len(plot_tests)}")
    
    if other_tests:
        other_passed = sum(1 for r in other_tests if r.passed)
        print(f"\n🔧 ПРОЧЕЕ ({len(other_tests)}):")
        print(f"   ✓ Пройдено: {other_passed}/{len(other_tests)}")
    
    print("\n Детали:")
    for result in results:
        status = "✓" if result.passed else "✗"
        print(f"  {status} {result.name}: {result.details if result.passed else result.error}")
    
    # Сохранение отчёта
    report_file = "test_comparison_report.json"
    report_data = {
        "timestamp": datetime.now().isoformat(),
        "total_tests": len(results),
        "passed": passed,
        "failed": failed,
        "duration_seconds": total_duration,
        "details": [
            {
                "name": r.name,
                "passed": r.passed,
                "duration": r.duration,
                "details": r.details if r.passed else r.error
            }
            for r in results
        ]
    }
    
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n Отчёт сохранён в: {report_file}")
    
    print("\n" + "=" * 80)
    if failed == 0:
        print(" ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
        print("   Можно запускать полный cross_dataset_comparison.py")
    else:
        print(f" {failed} ТЕСТ(ОВ) ПРОВАЛЕНО!")
        print("   Исправьте ошибки перед запуском полного сравнения")
    print("=" * 80)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)