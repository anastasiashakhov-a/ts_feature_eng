# src/ts_feature_eng/utils/time_series.py 

"""
Утилиты для обработки и анализа временных рядов.

Предоставляет векторизованные функции для оконных операций, спектрального анализа,
обработки пропусков и создания производных признаков с учетом временной структуры данных.
"""

from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import signal, stats
from scipy.fft import fft, fftfreq
from sklearn.preprocessing import StandardScaler


def create_sliding_windows(
    series: Union[pd.Series, np.ndarray],
    window_size: int,
    stride: int = 1,
    drop_incomplete: bool = True
) -> np.ndarray:
    """
    Создание скользящих окон из временного ряда.
    
    Параметры
    ----------
    series : pd.Series или np.ndarray
        Входной временной ряд.
    window_size : int
        Размер окна в наблюдениях.
    stride : int, по умолчанию 1
        Шаг сдвига окна.
    drop_incomplete : bool, по умолчанию True
        Удалять неполные окна в конце ряда.
    
    Возвращает
    ----------
    windows : np.ndarray
        Массив формы (n_windows, window_size) со скользящими окнами.
    
    Примеры
    --------
    >>> series = np.arange(10)
    >>> windows = create_sliding_windows(series, window_size=3, stride=2)
    >>> print(windows)
    [[0 1 2]
     [2 3 4]
     [4 5 6]
     [6 7 8]]
    """
    if isinstance(series, pd.Series):
        series = series.values
    
    if series.ndim != 1:
        raise ValueError(f"Input must be 1D array, got shape {series.shape}")
    
    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}")
    
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")
    
    n_samples = len(series)
    n_windows = (n_samples - window_size) // stride + 1
    
    if n_windows <= 0:
        if drop_incomplete:
            return np.empty((0, window_size))
        else:
            # Возвращаем одно неполное окно
            return series.reshape(1, -1)
    
    # Векторизованное создание окон через stride_tricks
    itemsize = series.itemsize
    shape = (n_windows, window_size)
    strides = (stride * itemsize, itemsize)
    
    windows = np.lib.stride_tricks.as_strided(
        series,
        shape=shape,
        strides=strides,
        writeable=False
    )
    
    return windows.copy()  # Копия для безопасности


def compute_rolling_statistics(
    series: Union[pd.Series, np.ndarray],
    window_size: int,
    statistics: List[str] = None,
    center: bool = False,
    min_periods: Optional[int] = None
) -> pd.DataFrame:
    """
    Вычисление множества статистик по скользящему окну за один проход.
    
    Параметры
    ----------
    series : pd.Series или np.ndarray
        Входной временной ряд.
    window_size : int
        Размер скользящего окна.
    statistics : List[str], опционально
        Список вычисляемых статистик. По умолчанию:
        ["mean", "std", "min", "max", "skew", "kurtosis", "slope", "acf1"]
    center : bool, по умолчанию False
        Центрировать окно относительно текущей точки.
    min_periods : int, опционально
        Минимальное количество наблюдений в окне.
    
    Возвращает
    ----------
    stats_df : pd.DataFrame
        DataFrame со столбцами для каждой статистики.
    """
    if statistics is None:
        statistics = ["mean", "std", "min", "max", "skew", "kurtosis", "slope", "acf1"]
    
    if isinstance(series, np.ndarray):
        if series.ndim != 1:
            raise ValueError("Input array must be 1D")
        series = pd.Series(series)
    
    if not isinstance(series, pd.Series):
        raise TypeError(f"Input must be pd.Series or 1D np.ndarray, got {type(series)}")
    
    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}")
    
    min_periods = min_periods or max(1, window_size // 4)
    
    # Создаем объект скользящего окна
    rolling = series.rolling(window=window_size, center=center, min_periods=min_periods)
    
    # Словарь для накопления результатов
    results = {}
    
    # Базовые статистики (векторизованные)
    if "mean" in statistics:
        results["mean"] = rolling.mean()
    
    if "std" in statistics:
        results["std"] = rolling.std(ddof=1)
    
    if "min" in statistics:
        results["min"] = rolling.min()
    
    if "max" in statistics:
        results["max"] = rolling.max()
    
    if "median" in statistics:
        results["median"] = rolling.median()
    
    if "sum" in statistics:
        results["sum"] = rolling.sum()
    
    # Сложные статистики (требуют применения функции)
    windows = create_sliding_windows(series.values, window_size, stride=1, drop_incomplete=False)
    
    if "skew" in statistics:
        results["skew"] = _apply_window_function(windows, stats.skew, min_periods=min_periods)
    
    if "kurtosis" in statistics:
        results["kurtosis"] = _apply_window_function(windows, stats.kurtosis, min_periods=min_periods)
    
    if "slope" in statistics:
        results["slope"] = _apply_window_function(windows, _compute_linear_slope, min_periods=min_periods)
    
    if "acf1" in statistics:
        results["acf1"] = _apply_window_function(windows, _compute_acf1, min_periods=min_periods)
    
    if "energy" in statistics:
        results["energy"] = _apply_window_function(windows, lambda x: np.sum(x**2), min_periods=min_periods)
    
    if "entropy" in statistics:
        results["entropy"] = _apply_window_function(windows, _compute_entropy, min_periods=min_periods)
    
    # Создаем итоговый DataFrame
    stats_df = pd.DataFrame(results, index=series.index)
    
    return stats_df


def _apply_window_function(
    windows: np.ndarray,
    func: Callable,
    min_periods: int = 1
) -> pd.Series:
    """
    Применение функции к каждому окну с обработкой пропусков.
    
    Параметры
    ----------
    windows : np.ndarray
        Массив окон формы (n_windows, window_size).
    func : Callable
        Функция для применения к каждому окну.
    min_periods : int
        Минимальное количество непропущенных значений.
    
    Возвращает
    ----------
    result : pd.Series
        Результат применения функции.
    """
    results = []
    
    for window in windows:
        # Подсчет непропущенных значений
        valid_mask = ~np.isnan(window)
        n_valid = np.sum(valid_mask)
        
        if n_valid < min_periods:
            results.append(np.nan)
        else:
            try:
                results.append(func(window[valid_mask]))
            except:
                results.append(np.nan)
    
    return pd.Series(results)


def _compute_linear_slope(window: np.ndarray) -> float:
    """Вычисление наклона линейной регрессии по окну."""
    if len(window) < 2:
        return np.nan
    
    x = np.arange(len(window))
    y = window
    
    # Простая линейная регрессия через МНК
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    numerator = np.sum(x_centered * y_centered)
    denominator = np.sum(x_centered ** 2)
    
    if denominator == 0:
        return np.nan
    
    return numerator / denominator


def _compute_acf1(window: np.ndarray) -> float:
    """Вычисление автокорреляции первого лага."""
    if len(window) < 3:
        return np.nan
    
    y = window - window.mean()
    autocov_1 = np.sum(y[1:] * y[:-1]) / len(window)
    variance = np.sum(y ** 2) / len(window)
    
    if variance == 0:
        return np.nan
    
    return autocov_1 / variance


def _compute_entropy(window: np.ndarray, bins: int = 10) -> float:
    """Вычисление энтропии Шеннона нормированного ряда."""
    if len(window) < 2:
        return np.nan
    
    # Нормализация в диапазон [0, 1]
    window_norm = (window - window.min()) / (window.max() - window.min() + 1e-10)
    
    # Гистограмма
    hist, _ = np.histogram(window_norm, bins=bins, range=(0, 1))
    prob = hist / hist.sum()
    prob = prob[prob > 0]
    
    if len(prob) == 0:
        return 0.0
    
    return -np.sum(prob * np.log2(prob + 1e-10))


def detect_seasonality(
    series: Union[pd.Series, np.ndarray],
    max_period: int = 100,
    method: str = "acf"
) -> Tuple[int, float]:
    """
    Автоматическое обнаружение периода сезонности временного ряда.
    
    Параметры
    ----------
    series : pd.Series или np.ndarray
        Входной временной ряд.
    max_period : int, по умолчанию 100
        Максимальный период для поиска.
    method : str, по умолчанию "acf"
        Метод обнаружения:
        - "acf": анализ автокорреляции
        - "fft": спектральный анализ через FFT
    
    Возвращает
    ----------
    period : int
        Обнаруженный период сезонности (0 если не обнаружена).
    strength : float
        Сила сезонности (0.0-1.0).
    """
    if isinstance(series, pd.Series):
        series = series.dropna().values
    else:
        series = series[~np.isnan(series)]
    
    if len(series) < 2 * max_period:
        return 0, 0.0
    
    if method == "acf":
        return _detect_seasonality_acf(series, max_period)
    elif method == "fft":
        return _detect_seasonality_fft(series, max_period)
    else:
        raise ValueError(f"Unknown method: {method}. Supported: 'acf', 'fft'")


def _detect_seasonality_acf(series: np.ndarray, max_period: int) -> Tuple[int, float]:
    """Обнаружение сезонности через анализ автокорреляции."""
    # Вычисление автокорреляции
    acf_vals = _autocorr(series, nlags=max_period * 2)
    
    # Поиск локальных максимумов после лага 1
    peaks, _ = signal.find_peaks(acf_vals[2:max_period + 2], height=0.2, distance=3)
    peaks = peaks + 2  # Коррекция индекса
    
    if len(peaks) == 0:
        return 0, 0.0
    
    # Выбор первого значимого пика
    period = int(peaks[0])
    strength = float(acf_vals[period])
    
    return period, strength


def _detect_seasonality_fft(series: np.ndarray, max_period: int) -> Tuple[int, float]:
    """Обнаружение сезонности через спектральный анализ."""
    # Детрендирование
    series_detrended = series - np.mean(series)
    
    # FFT
    n = len(series_detrended)
    fft_vals = np.abs(fft(series_detrended))
    freqs = fftfreq(n, d=1)
    
    # Рассматриваем только положительные частоты
    positive_mask = (freqs > 0) & (freqs <= 0.5)
    fft_vals = fft_vals[positive_mask]
    freqs = freqs[positive_mask]
    
    # Преобразуем частоты в периоды
    periods = 1 / freqs
    
    # Фильтруем периоды в допустимом диапазоне
    valid_mask = (periods >= 2) & (periods <= max_period)
    if not np.any(valid_mask):
        return 0, 0.0
    
    fft_vals = fft_vals[valid_mask]
    periods = periods[valid_mask]
    
    # Находим период с максимальной мощностью
    idx = np.argmax(fft_vals)
    period = int(round(periods[idx]))
    strength = float(fft_vals[idx] / fft_vals.sum())
    
    return period, strength


def _autocorr(x: np.ndarray, nlags: int = 40) -> np.ndarray:
    """Вычисление автокорреляционной функции."""
    x = np.asarray(x)
    n = len(x)
    x = x - np.mean(x)
    var = np.var(x)
    
    if var == 0:
        return np.zeros(nlags + 1)
    
    acf = np.correlate(x, x, mode="full")[-n:]
    acf = acf / (var * n)
    return acf[:nlags + 1]


def detrend_series(
    series: Union[pd.Series, np.ndarray],
    method: str = "linear",
    order: int = 1
) -> Union[pd.Series, np.ndarray]:
    """
    Удаление тренда из временного ряда.
    
    Параметры
    ----------
    series : pd.Series или np.ndarray
        Входной временной ряд.
    method : str, по умолчанию "linear"
        Метод детрендирования:
        - "linear": линейная регрессия
        - "poly": полиномиальная регрессия (требует параметр order)
        - "diff": первая разность
        - "median": вычитание скользящего медианного фильтра
    order : int, по умолчанию 1
        Порядок полинома для метода "poly".
    
    Возвращает
    ----------
    detrended : pd.Series или np.ndarray
        Ряд без тренда.
    """
    is_series = isinstance(series, pd.Series)
    index = series.index if is_series else None
    
    if is_series:
        series = series.values
    
    n = len(series)
    x = np.arange(n)
    
    if method == "linear":
        # Линейная регрессия
        slope, intercept = np.polyfit(x, series, 1)
        trend = slope * x + intercept
        detrended = series - trend
    
    elif method == "poly":
        # Полиномиальная регрессия
        coeffs = np.polyfit(x, series, order)
        trend = np.polyval(coeffs, x)
        detrended = series - trend
    
    elif method == "diff":
        # Первая разность
        detrended = np.diff(series, prepend=series[0])
    
    elif method == "median":
        # Медианный фильтр
        window_size = max(5, n // 20)
        if window_size % 2 == 0:
            window_size += 1
        trend = signal.medfilt(series, kernel_size=window_size)
        detrended = series - trend
    
    else:
        raise ValueError(
            f"Unknown detrend method: {method}. "
            f"Supported: 'linear', 'poly', 'diff', 'median'"
        )
    
    if is_series:
        return pd.Series(detrended, index=index)
    return detrended


def create_lagged_features(
    X: pd.DataFrame,
    lags: List[int],
    columns: Optional[List[str]] = None,
    drop_original: bool = False
) -> pd.DataFrame:
    """
    Создание лаговых признаков для временного ряда.
    
    Параметры
    ----------
    X : pd.DataFrame
        Исходный DataFrame с временными рядами.
    lags : List[int]
        Список лагов для создания (положительные значения = прошлое).
    columns : List[str], опционально
        Столбцы, для которых создавать лаги. Если None — все числовые столбцы.
    drop_original : bool, по умолчанию False
        Удалять исходные столбцы после создания лагов.
    
    Возвращает
    ----------
    X_lagged : pd.DataFrame
        DataFrame с добавленными лаговыми признаками.
    """
    if columns is None:
        columns = X.select_dtypes(include=[np.number]).columns.tolist()
    
    X_lagged = X.copy()
    new_columns = []
    
    for col in columns:
        for lag in lags:
            if lag <= 0:
                raise ValueError(f"Lags must be positive integers, got {lag}")
            
            lagged_col = f"{col}_lag_{lag}"
            X_lagged[lagged_col] = X[col].shift(lag)
            new_columns.append(lagged_col)
    
    if drop_original:
        X_lagged = X_lagged.drop(columns=columns)
    
    return X_lagged


def create_rolling_features(
    X: pd.DataFrame,
    windows: List[int],
    statistics: List[str] = None,
    columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Создание признаков на основе скользящих окон для нескольких столбцов.
    
    Параметры
    ----------
    X : pd.DataFrame
        Исходный DataFrame.
    windows : List[int]
        Список размеров окон.
    statistics : List[str], опционально
        Статистики для вычисления (см. compute_rolling_statistics).
    columns : List[str], опционально
        Столбцы для обработки. Если None — все числовые столбцы.
    
    Возвращает
    ----------
    X_rolling : pd.DataFrame
        DataFrame с добавленными признаками скользящих окон.
    """
    if columns is None:
        columns = X.select_dtypes(include=[np.number]).columns.tolist()
    
    if statistics is None:
        statistics = ["mean", "std"]
    
    X_rolling = X.copy()
    
    for col in columns:
        for window in windows:
            stats_df = compute_rolling_statistics(
                X[col],
                window_size=window,
                statistics=statistics,
                center=False,
                min_periods=max(1, window // 4)
            )
            
            # Переименовываем столбцы для уникальности
            stats_df = stats_df.rename(columns={
                stat: f"{col}_roll_{window}_{stat}" for stat in statistics
            })
            
            # Добавляем в результат
            X_rolling = pd.concat([X_rolling, stats_df], axis=1)
    
    return X_rolling


def fill_missing_temporal(
    series: Union[pd.Series, pd.DataFrame],
    method: str = "linear",
    max_gap: Optional[int] = None
) -> Union[pd.Series, pd.DataFrame]:
    """
    Заполнение пропусков с учетом временной структуры данных.
    
    Параметры
    ----------
    series : pd.Series или pd.DataFrame
        Временной ряд с пропусками.
    method : str, по умолчанию "linear"
        Метод заполнения:
        - "linear": линейная интерполяция
        - "time": интерполяция с учетом временного индекса
        - "ffill": заполнение предыдущим значением
        - "bfill": заполнение следующим значением
        - "seasonal": сезонное заполнение (требует обнаружения периода)
    max_gap : int, опционально
        Максимальный размер пропуска для заполнения. Пропуски больше будут сохранены как NaN.
    
    Возвращает
    ----------
    filled : pd.Series или pd.DataFrame
        Ряд с заполненными пропусками.
    """
    if method == "seasonal":
        return _fill_missing_seasonal(series, max_gap)
    
    # Используем встроенные методы pandas с ограничением по длине пропуска
    if max_gap is not None:
        # Создаем маску для пропусков, которые нужно заполнить
        is_na = series.isna()
        na_groups = (is_na != is_na.shift()).cumsum()
        na_counts = is_na.groupby(na_groups).transform('sum')
        
        # Заполняем только пропуски меньше max_gap
        to_fill = is_na & (na_counts <= max_gap)
        series_filled = series.copy()
        series_filled[to_fill] = series_filled[to_fill].interpolate(
            method=method,
            limit_direction="both"
        )
        return series_filled
    
    # Без ограничения по длине пропуска
    return series.interpolate(method=method, limit_direction="both")


def _fill_missing_seasonal(series: Union[pd.Series, pd.DataFrame], max_gap: Optional[int]) -> Union[pd.Series, pd.DataFrame]:
    """Заполнение пропусков с использованием сезонной структуры."""
    if isinstance(series, pd.DataFrame):
        return pd.DataFrame({
            col: _fill_missing_seasonal(series[col], max_gap)
            for col in series.columns
        })
    
    # Обнаружение периода сезонности
    period, _ = detect_seasonality(series, max_period=168)  # До недельного периода
    
    if period < 2:
        # Нет сезонности — используем линейную интерполяцию
        return fill_missing_temporal(series, method="linear", max_gap=max_gap)
    
    # Заполнение с использованием сезонных аналогов
    values = series.values.copy()
    n = len(values)
    
    for i in range(n):
        if np.isnan(values[i]):
            # Поиск значений на расстоянии ± периода
            candidates = []
            
            for offset in range(1, 3):  # Ближайшие 2 сезона в прошлом и будущем
                past_idx = i - offset * period
                future_idx = i + offset * period
                
                if past_idx >= 0 and not np.isnan(values[past_idx]):
                    candidates.append(values[past_idx])
                
                if future_idx < n and not np.isnan(values[future_idx]):
                    candidates.append(values[future_idx])
            
            if candidates:
                values[i] = np.mean(candidates)
    
    return pd.Series(values, index=series.index)


def normalize_series(
    series: Union[pd.Series, np.ndarray],
    method: str = "zscore",
    window_size: Optional[int] = None
) -> Union[pd.Series, np.ndarray]:
    """
    Нормализация временного ряда.
    
    Параметры
    ----------
    series : pd.Series или np.ndarray
        Входной временной ряд.
    method : str, по умолчанию "zscore"
        Метод нормализации:
        - "zscore": (x - mean) / std
        - "minmax": (x - min) / (max - min)
        - "robust": (x - median) / IQR
        - "log": log(1 + x) для положительных рядов
    window_size : int, опционально
        Размер окна для адаптивной нормализации (скользящее окно).
    
    Возвращает
    ----------
    normalized : pd.Series или np.ndarray
        Нормализованный ряд.
    """
    is_series = isinstance(series, pd.Series)
    index = series.index if is_series else None
    
    if is_series:
        series = series.values
    
    if window_size is None:
        # Глобальная нормализация
        if method == "zscore":
            mean = np.mean(series)
            std = np.std(series, ddof=1) + 1e-10
            normalized = (series - mean) / std
        
        elif method == "minmax":
            min_val = np.min(series)
            max_val = np.max(series)
            normalized = (series - min_val) / (max_val - min_val + 1e-10)
        
        elif method == "robust":
            median = np.median(series)
            q1 = np.percentile(series, 25)
            q3 = np.percentile(series, 75)
            iqr = q3 - q1 + 1e-10
            normalized = (series - median) / iqr
        
        elif method == "log":
            if np.any(series < 0):
                raise ValueError("Log normalization requires non-negative values")
            normalized = np.log1p(series)
        
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    else:
        # Адаптивная нормализация через скользящее окно
        normalized = np.empty_like(series)
        normalized[:] = np.nan
        
        for i in range(window_size, len(series) + 1):
            window = series[i - window_size:i]
            
            if method == "zscore":
                mean = np.mean(window)
                std = np.std(window, ddof=1) + 1e-10
                normalized[i - 1] = (series[i - 1] - mean) / std
            
            elif method == "minmax":
                min_val = np.min(window)
                max_val = np.max(window)
                normalized[i - 1] = (series[i - 1] - min_val) / (max_val - min_val + 1e-10)
        
        # Заполнение начальных значений глобальной нормализацией
        if np.isnan(normalized[0]):
            fallback = normalize_series(series[:window_size], method=method)
            normalized[:window_size] = fallback
    
    if is_series:
        return pd.Series(normalized, index=index)
    return normalized


def detect_change_points(
    series: Union[pd.Series, np.ndarray],
    method: str = "pelt",
    min_size: int = 10,
    penalty: float = 10.0
) -> List[int]:
    """
    Обнаружение точек смены режима (change points) во временном ряде.
    
    Параметры
    ----------
    series : pd.Series или np.ndarray
        Входной временной ряд.
    method : str, по умолчанию "pelt"
        Метод обнаружения:
        - "pelt": оптимальный алгоритм с линейной сложностью
        - "binary": бинарная сегментация
        - "window": скользящее окно
    min_size : int, по умолчанию 10
        Минимальный размер сегмента между точками смены.
    penalty : float, по умолчанию 10.0
        Штраф за добавление новой точки смены (регуляризация).
    
    Возвращает
    ----------
    change_points : List[int]
        Список индексов точек смены режима (исключая конец ряда).
    """
    try:
        from ruptures import Pelt, BinSeg, Window
        from ruptures.costs import CostNormal
    except ImportError:
        raise ImportError(
            "ruptures library required for change point detection. "
            "Install with: pip install ruptures"
        )
    
    if isinstance(series, pd.Series):
        series = series.dropna().values
    else:
        series = series[~np.isnan(series)]
    
    if len(series) < 2 * min_size:
        return []
    
    # Преобразуем в формат для ruptures (2D массив)
    signal = series.reshape(-1, 1)
    
    # Выбор алгоритма
    if method == "pelt":
        algo = Pelt(model="l2", min_size=min_size, jump=1).fit(signal)
    elif method == "binary":
        algo = BinSeg(model="l2", min_size=min_size, jump=1).fit(signal)
    elif method == "window":
        algo = Window(width=min_size, model="l2", jump=1).fit(signal)
    else:
        raise ValueError(f"Unknown method: {method}. Supported: 'pelt', 'binary', 'window'")
    
    # Обнаружение точек смены
    change_points = algo.predict(pen=penalty)
    
    # Удаляем последнюю точку (конец ряда) и преобразуем в 0-based индексы
    change_points = [cp - 1 for cp in change_points[:-1]]
    
    return change_points


def resample_time_series(
    series: Union[pd.Series, pd.DataFrame],
    freq: str,
    method: str = "mean",
    fill_method: Optional[str] = "ffill"
) -> Union[pd.Series, pd.DataFrame]:
    """
    Ресемплирование временного ряда на новую частоту.
    
    Параметры
    ----------
    series : pd.Series или pd.DataFrame
        Временной ряд с временным индексом.
    freq : str
        Целевая частота (например, "1H", "1D", "1W").
    method : str, по умолчанию "mean"
        Метод агрегации для укрупнения:
        - "mean": среднее значение
        - "sum": сумма
        - "median": медиана
        - "last": последнее значение
        - "first": первое значение
    fill_method : str, опционально
        Метод заполнения пропусков после ресемплирования:
        - "ffill": заполнение предыдущим значением
        - "bfill": заполнение следующим значением
        - "interpolate": линейная интерполяция
        - None: без заполнения
    
    Возвращает
    ----------
    resampled : pd.Series или pd.DataFrame
        Ресемплированный временной ряд.
    """
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("Index must be DatetimeIndex for resampling")
    
    # Ресемплирование
    resampled = series.resample(freq)
    
    if method == "mean":
        resampled = resampled.mean()
    elif method == "sum":
        resampled = resampled.sum()
    elif method == "median":
        resampled = resampled.median()
    elif method == "last":
        resampled = resampled.last()
    elif method == "first":
        resampled = resampled.first()
    else:
        raise ValueError(
            f"Unknown resampling method: {method}. "
            f"Supported: 'mean', 'sum', 'median', 'last', 'first'"
        )
    
    # Заполнение пропусков
    if fill_method == "ffill":
        resampled = resampled.ffill()
    elif fill_method == "bfill":
        resampled = resampled.bfill()
    elif fill_method == "interpolate":
        resampled = resampled.interpolate(method="time")
    elif fill_method is not None:
        raise ValueError(
            f"Unknown fill method: {fill_method}. "
            f"Supported: 'ffill', 'bfill', 'interpolate', None"
        )
    
    return resampled