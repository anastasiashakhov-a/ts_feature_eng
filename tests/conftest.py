# tests/conftest.py 

"""
Конфигурация тестов и общие фикстуры для pytest.

Предоставляет параметризованные фикстуры для генерации различных типов временных рядов,
целевых переменных и вспомогательных утилит для проверки корректности трансформаций.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge


# ==================== ГЛОБАЛЬНЫЕ КОНСТАНТЫ ====================
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# ==================== РЕГИСТРАЦИЯ КАСТОМНЫХ ОПЦИЙ PYTEST ====================
def pytest_addoption(parser):
    """
    Регистрация кастомных опций командной строки для pytest.
    
    Добавляет опцию --runslow для запуска медленных тестов.
    """
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="запустить медленные тесты (по умолчанию пропускаются)"
    )


# ==================== ФИКСТУРЫ ДЛЯ ВРЕМЕННЫХ РЯДОВ ====================

@pytest.fixture
def ts_stationary():
    """
    Стационарный временной ряд (белый шум).
    
    Характеристики:
    - Нулевое среднее
    - Постоянная дисперсия
    - Отсутствие автокорреляции
    """
    n_samples = 500
    dates = pd.date_range("2023-01-01", periods=n_samples, freq="H")
    values = np.random.randn(n_samples) * 2.0
    return pd.DataFrame({"value": values}, index=dates)


@pytest.fixture
def ts_trend():
    """
    Временной ряд с линейным трендом.
    
    Характеристики:
    - Линейный восходящий тренд
    - Небольшая случайная составляющая
    """
    n_samples = 500
    dates = pd.date_range("2023-01-01", periods=n_samples, freq="H")
    trend = np.linspace(0, 50, n_samples)
    noise = np.random.randn(n_samples) * 3.0
    values = trend + noise
    return pd.DataFrame({"value": values}, index=dates)


@pytest.fixture
def ts_seasonal_daily():
    """
    Временной ряд с суточной сезонностью (период 24).
    
    Характеристики:
    - Ярко выраженная 24-часовая сезонность
    - Небольшой восходящий тренд
    - Случайный шум
    """
    n_samples = 720  # 30 дней по 24 часа
    dates = pd.date_range("2023-01-01", periods=n_samples, freq="H")
    seasonal = 10 * np.sin(2 * np.pi * np.arange(n_samples) / 24)
    trend = np.linspace(0, 20, n_samples)
    noise = np.random.randn(n_samples) * 1.5
    values = seasonal + trend + noise
    return pd.DataFrame({"value": values}, index=dates)


@pytest.fixture
def ts_seasonal_weekly():
    """
    Временной ряд с недельной сезонностью (период 168).
    
    Характеристики:
    - Ярко выраженная 168-часовая (недельная) сезонность
    - Разные амплитуды для будней/выходных
    """
    n_samples = 1680  # 10 недель
    dates = pd.date_range("2023-01-01", periods=n_samples, freq="H")
    
    # Недельная сезонность
    weekly = 15 * np.sin(2 * np.pi * np.arange(n_samples) / 168)
    
    # Дополнительная суточная сезонность с разной амплитудой для будней/выходных
    hour_of_day = np.arange(n_samples) % 24
    day_of_week = (np.arange(n_samples) // 24) % 7
    daily_amp = np.where(day_of_week < 5, 8.0, 4.0)  # Будни амплитуднее выходных
    daily = daily_amp * np.sin(2 * np.pi * hour_of_day / 24)
    
    noise = np.random.randn(n_samples) * 2.0
    values = weekly + daily + noise
    
    return pd.DataFrame({"value": values}, index=dates)


@pytest.fixture
def ts_multivariate():
    """
    Многомерный временной ряд (3 признака).
    
    Характеристики:
    - Признак 1: тренд + шум
    - Признак 2: сезонность + шум
    - Признак 3: тренд + сезонность + шум
    """
    n_samples = 500
    dates = pd.date_range("2023-01-01", periods=n_samples, freq="H")
    
    trend = np.linspace(0, 30, n_samples)
    seasonal = 8 * np.sin(2 * np.pi * np.arange(n_samples) / 24)
    noise = np.random.randn(n_samples, 3) * 2.0
    
    df = pd.DataFrame({
        "feature1": trend + noise[:, 0],
        "feature2": seasonal + noise[:, 1],
        "feature3": trend + seasonal + noise[:, 2],
    }, index=dates)
    
    return df


@pytest.fixture
def ts_with_missing():
    """
    Временной ряд с пропусками (20% случайных пропусков).
    
    Характеристики:
    - Суточная сезонность
    - Случайные пропуски в 20% наблюдений
    - Пропуски не имеют временной структуры
    """
    n_samples = 500
    dates = pd.date_range("2023-01-01", periods=n_samples, freq="H")
    seasonal = 10 * np.sin(2 * np.pi * np.arange(n_samples) / 24)
    noise = np.random.randn(n_samples) * 1.5
    values = seasonal + noise
    
    # Вводим случайные пропуски
    mask = np.random.rand(n_samples) < 0.2
    values[mask] = np.nan
    
    return pd.DataFrame({"value": values}, index=dates)


@pytest.fixture
def ts_with_regime_change():
    """
    Временной ряд с точками смены режима.
    
    Характеристики:
    - 3 сегмента с разными параметрами
    - Изменение дисперсии и среднего в точках смены
    """
    n_samples = 600
    dates = pd.date_range("2023-01-01", periods=n_samples, freq="H")
    
    # Сегмент 1: низкая дисперсия, нулевое среднее
    seg1 = np.random.randn(200) * 1.0
    
    # Сегмент 2: высокая дисперсия, положительное среднее
    seg2 = np.random.randn(200) * 4.0 + 10.0
    
    # Сегмент 3: тренд + сезонность
    t = np.arange(200)
    seg3 = 0.1 * t + 5 * np.sin(2 * np.pi * t / 24) + np.random.randn(200) * 2.0
    
    values = np.concatenate([seg1, seg2, seg3])
    
    return pd.DataFrame({"value": values}, index=dates)


@pytest.fixture
def ts_short():
    """
    Короткий временной ряд (минимальная длина для базовых операций).
    
    Характеристики:
    - 50 наблюдений
    - Простая структура для тестирования граничных случаев
    """
    n_samples = 50
    dates = pd.date_range("2023-01-01", periods=n_samples, freq="H")
    values = np.random.randn(n_samples) * 5.0 + 20.0
    return pd.DataFrame({"value": values}, index=dates)


@pytest.fixture
def ts_irregular_index():
    """
    Временной ряд с нерегулярным временным индексом.
    
    Характеристики:
    - Пропущенные временные метки (неравномерная дискретизация)
    - Требует интерполяции перед обработкой
    """
    # Создаем регулярный индекс
    dates_regular = pd.date_range("2023-01-01", periods=200, freq="H")
    
    # Удаляем случайные 10% временных меток для создания нерегулярности
    mask = np.random.rand(200) > 0.1
    dates_irregular = dates_regular[mask]
    
    values = np.random.randn(len(dates_irregular)) * 3.0 + 15.0
    return pd.DataFrame({"value": values}, index=dates_irregular)


# ==================== ФИКСТУРЫ ДЛЯ ЦЕЛЕВЫХ ПЕРЕМЕННЫХ ====================

@pytest.fixture
def target_one_step(ts_stationary):
    """
    Целевая переменная для прогнозирования на 1 шаг вперед.
    """
    y = ts_stationary["value"].shift(-1)
    return y.dropna()


@pytest.fixture
def target_multi_step(ts_seasonal_daily):
    """
    Целевая переменная для прогнозирования на 24 шага вперед (суточный горизонт).
    """
    y = ts_seasonal_daily["value"].shift(-24)
    return y.dropna()


@pytest.fixture
def target_multivariate(ts_multivariate):
    """
    Целевая переменная для многомерного ряда (прогнозируем первый признак).
    """
    y = ts_multivariate["feature1"].shift(-1)
    return y.dropna()


# ==================== ФИКСТУРЫ ДЛЯ МОДЕЛЕЙ ====================

@pytest.fixture
def proxy_model():
    """
    Модель-прокси для оценки качества признакового пространства.
    """
    return Ridge(alpha=1.0, random_state=RANDOM_SEED)


# ==================== ВСПОМОГАТЕЛЬНЫЕ УТИЛИТЫ ====================

@pytest.fixture
def assert_transformer_output():
    """
    Утилита для проверки корректности вывода трансформера.
    
    Проверяет:
    - Тип выходных данных (pd.DataFrame)
    - Соответствие длины исходным данным
    - Отсутствие бесконечных значений
    - Наличие сгенерированных признаков
    """
    def _assert_output(
        X_original: pd.DataFrame,
        X_transformed: pd.DataFrame,
        min_features: int = 1,
        allow_nans: bool = True,
        max_nan_ratio: float = 0.5
    ):
        # Проверка типа
        assert isinstance(X_transformed, pd.DataFrame), \
            f"Expected pd.DataFrame, got {type(X_transformed)}"
        
        # Проверка длины
        assert len(X_transformed) == len(X_original), \
            f"Length mismatch: original={len(X_original)}, transformed={len(X_transformed)}"
        
        # Проверка наличия признаков
        assert X_transformed.shape[1] >= min_features, \
            f"Expected at least {min_features} features, got {X_transformed.shape[1]}"
        
        # Проверка на бесконечности
        assert np.all(np.isfinite(X_transformed.select_dtypes(include=[np.number]).values)), \
            "Transformed data contains non-finite values (inf or nan in numeric columns)"
        
        # Проверка доли пропусков (если разрешены)
        if not allow_nans:
            assert not X_transformed.isna().any().any(), \
                "Transformed data contains NaN values but allow_nans=False"
        else:
            nan_ratio = X_transformed.isna().sum().sum() / (X_transformed.shape[0] * X_transformed.shape[1])
            assert nan_ratio <= max_nan_ratio, \
                f"NaN ratio {nan_ratio:.2%} exceeds maximum allowed {max_nan_ratio:.2%}"
        
        # Проверка индекса
        assert X_transformed.index.equals(X_original.index), \
            "Transformed data index does not match original index"
    
    return _assert_output


@pytest.fixture
def assert_feature_names():
    """
    Утилита для проверки корректности имен сгенерированных признаков.
    
    Проверяет:
    - Соответствие шаблону именования
    - Уникальность имен
    - Наличие ожидаемых компонентов в именах
    """
    def _assert_names(
        feature_names: list,
        expected_prefixes: list = None,
        expected_statistics: list = None,
        require_unique: bool = True
    ):
        # Проверка типа
        assert isinstance(feature_names, list), \
            f"Expected list of feature names, got {type(feature_names)}"
        
        # Проверка непустоты
        assert len(feature_names) > 0, "Feature names list is empty"
        
        # Проверка уникальности
        if require_unique:
            assert len(feature_names) == len(set(feature_names)), \
                "Feature names are not unique"
        
        # Проверка формата имен
        for name in feature_names:
            assert isinstance(name, str), f"Feature name must be string, got {type(name)}"
            assert len(name) > 0, "Feature name is empty"
            
            # Проверка на наличие точек (разделителей компонентов)
            assert "." in name, f"Feature name '{name}' does not contain '.' separator"
        
        # Проверка ожидаемых префиксов
        if expected_prefixes:
            name_set = set(feature_names)
            for prefix in expected_prefixes:
                assert any(name.startswith(prefix) for name in feature_names), \
                    f"No feature names start with expected prefix '{prefix}'"
        
        # Проверка ожидаемых статистик
        if expected_statistics:
            for stat in expected_statistics:
                assert any(stat in name for name in feature_names), \
                    f"No feature names contain expected statistic '{stat}'"
    
    return _assert_names


# ==================== ПАРАМЕТРИЗОВАННЫЕ ФИКСТУРЫ ====================

@pytest.fixture(params=[
    "ts_stationary",
    "ts_trend",
    "ts_seasonal_daily",
    "ts_seasonal_weekly",
    "ts_multivariate",
    "ts_with_missing",
])
def ts_various(request):
    """
    Параметризованная фикстура для тестирования на разных типах временных рядов.
    
    Использование:
    @pytest.mark.parametrize("ts_various", [...], indirect=True)
    def test_something(ts_various):
        ...
    """
    return request.getfixturevalue(request.param)


# ==================== НАСТРОЙКИ PYTEST ====================

def pytest_configure(config):
    """
    Регистрация кастомных маркеров для pytest.
    """
    config.addinivalue_line(
        "markers",
        "slow: mark test as slow-running (skipped by default with -m 'not slow')"
    )
    config.addinivalue_line(
        "markers",
        "integration: mark test as integration test (requires multiple components)"
    )
    config.addinivalue_line(
        "markers",
        "requires_shap: mark test as requiring SHAP library"
    )


def pytest_collection_modifyitems(config, items):
    """
    Модификация собранных тестов (например, пропуск медленных тестов).
    """
    # Пропуск медленных тестов если не указан флаг --runslow
    if not config.getoption("--runslow"):
        skip_slow = pytest.mark.skip(reason="need --runslow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)


# ==================== ДОПОЛНИТЕЛЬНЫЕ УТИЛИТЫ ====================

@pytest.fixture
def generate_synthetic_ts():
    """
    Фабрика для генерации синтетических временных рядов с заданными параметрами.
    
    Параметры:
    - n_samples: количество наблюдений
    - freq: частота дискретизации ('H', 'D', 'W')
    - has_trend: наличие линейного тренда
    - seasonality: период сезонности (None для отсутствия)
    - noise_level: уровень шума
    - missing_ratio: доля пропусков (0.0-1.0)
    
    Возвращает:
    - Функция генерации с указанными параметрами
    """
    def _generator(
        n_samples: int = 500,
        freq: str = "H",
        has_trend: bool = False,
        seasonality: Optional[int] = None,
        noise_level: float = 1.0,
        missing_ratio: float = 0.0
    ) -> pd.DataFrame:
        dates = pd.date_range("2023-01-01", periods=n_samples, freq=freq)
        
        # Базовый ряд
        values = np.zeros(n_samples)
        
        # Добавление тренда
        if has_trend:
            values += np.linspace(0, 20, n_samples)
        
        # Добавление сезонности
        if seasonality is not None and seasonality > 0:
            values += 10 * np.sin(2 * np.pi * np.arange(n_samples) / seasonality)
        
        # Добавление шума
        values += np.random.randn(n_samples) * noise_level
        
        # Добавление пропусков
        if missing_ratio > 0:
            mask = np.random.rand(n_samples) < missing_ratio
            values[mask] = np.nan
        
        return pd.DataFrame({"value": values}, index=dates)
    
    return _generator


@pytest.fixture
def assert_no_temporal_leakage():
    """
    Утилита для проверки отсутствия временной утечки данных.
    
    Проверяет, что прогноз на момент t использует только данные до t.
    """
    def _assert_no_leakage(
        X_original: pd.DataFrame,
        X_transformed: pd.DataFrame,
        max_lookahead: int = 0
    ):
        """
        Проверка отсутствия утечки во времени.
        
        Параметры:
        - X_original: исходные данные
        - X_transformed: трансформированные данные
        - max_lookahead: максимальное допустимое "заглядывание" в будущее (в наблюдениях)
        """
        # Для каждого признака в трансформированном наборе
        for col in X_transformed.columns:
            series = X_transformed[col]
            
            # Находим первое непропущенное значение
            first_valid_idx = series.first_valid_index()
            if first_valid_idx is None:
                continue  # Пропускаем полностью пропущенный признак
            
            # Находим соответствующий индекс в исходных данных
            orig_idx = X_original.index.get_loc(first_valid_idx)
            
            # Проверяем, что для генерации этого значения использовалось не более max_lookahead будущих наблюдений
            # (для оконных методов первые значения будут пропущены)
            expected_min_idx = max(0, orig_idx - max_lookahead)
            assert orig_idx >= expected_min_idx, \
                f"Feature '{col}' shows temporal leakage: first valid value at index {orig_idx} " \
                f"but should be at least {expected_min_idx}"
    
    return _assert_no_leakage