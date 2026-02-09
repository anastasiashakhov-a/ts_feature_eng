# src/ts_feature_eng/utils/validation.py

"""
Утилиты для валидации временных рядов и параметров трансформеров.

Обеспечивают строгую проверку входных данных перед применением методов
инженерии признаков, предотвращая распространение ошибок на последующие этапы.
"""

from typing import Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ..base import TimeSeriesError


def validate_time_series(
    X: Union[pd.DataFrame, np.ndarray],
    require_datetime_index: bool = False,
    allow_missing: bool = True,
    max_missing_ratio: float = 0.5,
    require_monotonic: bool = True,
    require_unique_index: bool = True,
) -> pd.DataFrame:
    """
    Комплексная валидация временного ряда.
    
    Параметры
    ----------
    X : pd.DataFrame или np.ndarray
        Входные данные временного ряда.
    require_datetime_index : bool, по умолчанию False
        Требовать временной индекс (DatetimeIndex).
    allow_missing : bool, по умолчанию True
        Разрешить пропуски в данных.
    max_missing_ratio : float, по умолчанию 0.5
        Максимальная допустимая доля пропусков (от 0.0 до 1.0).
    require_monotonic : bool, по умолчанию True
        Требовать монотонно возрастающий индекс.
    require_unique_index : bool, по умолчанию True
        Требовать уникальные временные метки.
    
    Возвращает
    ----------
    X_validated : pd.DataFrame
        Валидированный DataFrame.
    
    Выбрасывает
    ----------
    TimeSeriesError
        При нарушении любого из условий валидации.
    """
    # Конвертация в DataFrame
    X = _convert_to_dataframe(X)
    
    # Базовая валидация
    _validate_non_empty(X)
    _validate_no_infs(X)
    _validate_numeric(X)
    
    # Валидация индекса
    if require_datetime_index:
        _validate_datetime_index(X)
    
    if require_unique_index:
        _validate_unique_index(X)
    
    if require_monotonic:
        _validate_monotonic_index(X)
    
    # Валидация пропусков
    if not allow_missing:
        _validate_no_missing(X)
    else:
        _validate_missing_ratio(X, max_missing_ratio)
    
    # Валидация константных столбцов
    _validate_no_constant_columns(X)
    
    return X


def validate_target_series(
    y: Union[pd.Series, np.ndarray, pd.DataFrame],
    X: Optional[pd.DataFrame] = None
) -> pd.Series:
    """
    Валидация целевой переменной для задачи прогнозирования.
    
    Параметры
    ----------
    y : pd.Series, np.ndarray или pd.DataFrame
        Целевая переменная.
    X : pd.DataFrame, опционально
        Соответствующие признаки для проверки согласованности длин.
    
    Возвращает
    ----------
    y_validated : pd.Series
        Валидированная целевая переменная.
    
    Выбрасывает
    ----------
    TimeSeriesError
        При нарушении условий валидации.
    """
    # Конвертация в Series
    if isinstance(y, pd.DataFrame):
        if y.shape[1] != 1:
            raise TimeSeriesError(
                f"Target DataFrame must have exactly 1 column, got {y.shape[1]}"
            )
        y = y.iloc[:, 0]
    elif isinstance(y, np.ndarray):
        if y.ndim > 1:
            if y.shape[1] != 1:
                raise TimeSeriesError(
                    f"Target array must be 1D or have 1 column, got shape {y.shape}"
                )
            y = y.ravel()
        y = pd.Series(y)
    elif not isinstance(y, pd.Series):
        raise TimeSeriesError(
            f"Target must be pd.Series, np.ndarray or pd.DataFrame, got {type(y)}"
        )
    
    # Проверка на пустоту
    if y.empty:
        raise TimeSeriesError("Target series is empty")
    
    # Проверка на бесконечности и нечисловые значения
    if not np.all(np.isfinite(y.dropna().values)):
        raise TimeSeriesError("Target contains non-finite values (inf, nan, or non-numeric)")
    
    # Проверка согласованности с X
    if X is not None:
        if len(y) != len(X):
            raise TimeSeriesError(
                f"Target length ({len(y)}) does not match features length ({len(X)})"
            )
        # Выравнивание индексов при необходимости
        if not y.index.equals(X.index):
            y = y.reindex(X.index)
    
    return y


def validate_transformer_params(
    params: dict,
    required_keys: List[str],
    value_ranges: Optional[dict] = None,
    allowed_values: Optional[dict] = None,
) -> None:
    """
    Валидация гиперпараметров трансформера.
    
    Параметры
    ----------
    params : dict
        Словарь параметров для валидации.
    required_keys : List[str]
        Обязательные ключи, которые должны присутствовать в params.
    value_ranges : dict, опционально
        Словарь диапазонов допустимых значений в формате {ключ: (мин, макс)}.
    allowed_values : dict, опционально
        Словарь допустимых значений в формате {ключ: [значение1, значение2, ...]}.
    
    Выбрасывает
    ----------
    TimeSeriesError
        При нарушении условий валидации.
    """
    # Проверка обязательных ключей
    missing_keys = set(required_keys) - set(params.keys())
    if missing_keys:
        raise TimeSeriesError(f"Missing required parameters: {missing_keys}")
    
    # Проверка диапазонов значений
    if value_ranges:
        for key, (min_val, max_val) in value_ranges.items():
            if key in params:
                value = params[key]
                if not (min_val <= value <= max_val):
                    raise TimeSeriesError(
                        f"Parameter '{key}'={value} is outside valid range [{min_val}, {max_val}]"
                    )
    
    # Проверка допустимых значений
    if allowed_values:
        for key, allowed in allowed_values.items():
            if key in params:
                value = params[key]
                if value not in allowed:
                    raise TimeSeriesError(
                        f"Parameter '{key}'={value} is not in allowed values {allowed}"
                    )


def validate_frequency(
    X: pd.DataFrame,
    min_freq: Optional[str] = None,
    max_freq: Optional[str] = None,
    require_regular: bool = True,
) -> str:
    """
    Валидация и определение частоты временного ряда.
    
    Параметры
    ----------
    X : pd.DataFrame
        Временной ряд с временным индексом.
    min_freq : str, опционально
        Минимально допустимая частота (например, "1H" для часовой).
    max_freq : str, опционально
        Максимально допустимая частота (например, "1D" для дневной).
    require_regular : bool, по умолчанию True
        Требовать регулярную частоту дискретизации.
    
    Возвращает
    ----------
    inferred_freq : str
        Определенная частота ряда (например, "H", "D", "W").
    
    Выбрасывает
    ----------
    TimeSeriesError
        При нарушении условий валидации частоты.
    """
    if not isinstance(X.index, pd.DatetimeIndex):
        raise TimeSeriesError("Index must be DatetimeIndex for frequency validation")
    
    # Определение частоты
    inferred_freq = pd.infer_freq(X.index)
    
    if require_regular and inferred_freq is None:
        # Попытка ручного определения частоты через медианный интервал
        if len(X) < 2:
            raise TimeSeriesError("Insufficient data points to determine frequency")
        
        diffs = np.diff(X.index.astype(np.int64))
        median_diff = np.median(diffs)
        std_diff = np.std(diffs)
        
        # Проверка регулярности (стандартное отклонение < 10% от медианы)
        if std_diff > 0.1 * median_diff:
            raise TimeSeriesError(
                "Irregular time series: timestamp intervals are not consistent. "
                "Consider resampling or interpolating the series."
            )
        
        # Определение приблизительной частоты
        hours = median_diff / (3600 * 1e9)
        if hours < 1.5:
            inferred_freq = "H"
        elif hours < 25:
            inferred_freq = "D"
        elif hours < 169:
            inferred_freq = "W"
        else:
            inferred_freq = "M"
    
    # Валидация против минимальной и максимальной частот
    if min_freq is not None and inferred_freq is not None:
        min_td = pd.Timedelta(min_freq)
        inferred_td = pd.Timedelta(1, inferred_freq)
        if inferred_td < min_td:
            raise TimeSeriesError(
                f"Frequency {inferred_freq} is higher than minimum allowed {min_freq}"
            )
    
    if max_freq is not None and inferred_freq is not None:
        max_td = pd.Timedelta(max_freq)
        inferred_td = pd.Timedelta(1, inferred_freq)
        if inferred_td > max_td:
            raise TimeSeriesError(
                f"Frequency {inferred_freq} is lower than maximum allowed {max_freq}"
            )
    
    return inferred_freq or "irregular"


def validate_window_size(
    window_size: int,
    n_samples: int,
    min_ratio: float = 0.01,
    max_ratio: float = 0.5,
) -> int:
    """
    Валидация размера скользящего окна относительно длины ряда.
    
    Параметры
    ----------
    window_size : int
        Запрашиваемый размер окна.
    n_samples : int
        Длина временного ряда.
    min_ratio : float, по умолчанию 0.01
        Минимальное отношение размера окна к длине ряда.
    max_ratio : float, по умолчанию 0.5
        Максимальное отношение размера окна к длине ряда.
    
    Возвращает
    ----------
    validated_size : int
        Валидированный размер окна.
    
    Выбрасывает
    ----------
    TimeSeriesError
        При нарушении условий валидации.
    """
    if not isinstance(window_size, int) or window_size <= 0:
        raise TimeSeriesError(f"Window size must be positive integer, got {window_size}")
    
    if n_samples < 10:
        raise TimeSeriesError(f"Insufficient data length ({n_samples}) for window operations")
    
    min_size = max(2, int(n_samples * min_ratio))
    max_size = min(n_samples - 1, int(n_samples * max_ratio))
    
    if window_size < min_size:
        raise TimeSeriesError(
            f"Window size {window_size} is too small for series length {n_samples} "
            f"(minimum: {min_size})"
        )
    
    if window_size > max_size:
        raise TimeSeriesError(
            f"Window size {window_size} is too large for series length {n_samples} "
            f"(maximum: {max_size})"
        )
    
    return window_size


def detect_time_column(
    X: pd.DataFrame,
    time_col: Optional[str] = None
) -> Tuple[str, pd.Series]:
    """
    Автоматическое обнаружение столбца или индекса с временными метками.
    
    Параметры
    ----------
    X : pd.DataFrame
        Входные данные.
    time_col : str, опционально
        Явно указанное имя столбца с временными метками.
    
    Возвращает
    ----------
    source : str
        Источник временных меток: "index" или имя столбца.
    timestamps : pd.Series
        Серия временных меток в формате datetime64.
    
    Выбрасывает
    ----------
    TimeSeriesError
        Если временные метки не могут быть обнаружены.
    """
    # Проверка временного индекса
    if isinstance(X.index, pd.DatetimeIndex):
        return "index", X.index.to_series()
    
    # Явно указанный столбец
    if time_col is not None:
        if time_col not in X.columns:
            raise TimeSeriesError(f"Specified time column '{time_col}' not found in DataFrame")
        
        if pd.api.types.is_datetime64_any_dtype(X[time_col]):
            return time_col, pd.to_datetime(X[time_col])
        
        try:
            return time_col, pd.to_datetime(X[time_col])
        except:
            raise TimeSeriesError(
                f"Column '{time_col}' cannot be converted to datetime format"
            )
    
    # Автоматический поиск
    # 1. Поиск столбцов с типом datetime
    datetime_cols = [
        col for col in X.columns
        if pd.api.types.is_datetime64_any_dtype(X[col])
    ]
    
    if datetime_cols:
        return datetime_cols[0], pd.to_datetime(X[datetime_cols[0]])
    
    # 2. Поиск по именам столбцов
    time_hints = ["timestamp", "date", "datetime", "time", "ds", "dt"]
    hinted_cols = [
        col for col in X.columns
        if any(hint in col.lower() for hint in time_hints)
    ]
    
    for col in hinted_cols:
        try:
            return col, pd.to_datetime(X[col])
        except:
            continue
    
    # 3. Попытка конвертации всех столбцов
    for col in X.columns:
        try:
            timestamps = pd.to_datetime(X[col])
            # Проверка, что конвертация успешна для большинства значений
            if timestamps.notna().mean() > 0.9:
                return col, timestamps
        except:
            continue
    
    raise TimeSeriesError(
        "No time column detected. DataFrame has no DatetimeIndex and no columns "
        "that can be converted to datetime format. Specify time_col parameter explicitly."
    )


def validate_no_leakage(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    time_col: Optional[str] = None
) -> bool:
    """
    Проверка отсутствия утечки данных во времени (temporal leakage).
    
    Убеждается, что все временные метки в тестовом наборе строго следуют
    после меток в обучающем наборе.
    
    Параметры
    ----------
    X_train : pd.DataFrame
        Обучающий набор данных.
    X_test : pd.DataFrame
        Тестовый набор данных.
    time_col : str, опционально
        Имя столбца с временными метками (если не индекс).
    
    Возвращает
    ----------
    is_valid : bool
        True если утечки нет, иначе выбрасывает исключение.
    
    Выбрасывает
    ----------
    TimeSeriesError
        При обнаружении временной утечки данных.
    """
    # Получение временных меток
    _, train_times = detect_time_column(X_train, time_col)
    _, test_times = detect_time_column(X_test, time_col)
    
    # Проверка пересечения временных интервалов
    train_min = train_times.min()
    train_max = train_times.max()
    test_min = test_times.min()
    test_max = test_times.max()
    
    if test_min <= train_max:
        raise TimeSeriesError(
            f"Temporal leakage detected: test set starts at {test_min} "
            f"which is before or at training set end {train_max}. "
            "Ensure test set contains only future observations relative to training set."
        )
    
    # Проверка нахлестывающихся временных меток
    train_set = set(train_times)
    test_set = set(test_times)
    
    overlap = train_set & test_set
    if overlap:
        raise TimeSeriesError(
            f"Temporal leakage detected: {len(overlap)} overlapping timestamps found "
            f"between train and test sets. Overlap example: {list(overlap)[:3]}"
        )
    
    return True


# Вспомогательные внутренние функции
def _convert_to_dataframe(X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
    """Конвертация входных данных в pd.DataFrame."""
    if isinstance(X, pd.DataFrame):
        return X.copy()
    
    if isinstance(X, np.ndarray):
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    
    raise TimeSeriesError(f"Unsupported input type: {type(X)}")


def _validate_non_empty(X: pd.DataFrame) -> None:
    """Проверка на пустоту данных."""
    if X.empty:
        raise TimeSeriesError("Input DataFrame is empty")
    if len(X) < 2:
        raise TimeSeriesError(f"Insufficient data points: {len(X)} (minimum 2 required)")


def _validate_no_infs(X: pd.DataFrame) -> None:
    """Проверка на бесконечные значения."""
    if not np.all(np.isfinite(X.select_dtypes(include=[np.number]).values)):
        raise TimeSeriesError("Input contains non-finite values (inf or -inf)")


def _validate_numeric(X: pd.DataFrame) -> None:
    """Проверка на наличие хотя бы одного числового столбца."""
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        raise TimeSeriesError("Input contains no numeric columns")


def _validate_datetime_index(X: pd.DataFrame) -> None:
    """Проверка наличия временного индекса."""
    if not isinstance(X.index, pd.DatetimeIndex):
        raise TimeSeriesError(
            "DatetimeIndex required but not found. "
            "Consider setting index to datetime format or using time_col parameter."
        )


def _validate_unique_index(X: pd.DataFrame) -> None:
    """Проверка уникальности временного индекса."""
    if X.index.has_duplicates:
        duplicates = X.index[X.index.duplicated()].unique()
        raise TimeSeriesError(
            f"DatetimeIndex contains duplicate timestamps: {duplicates[:5]} "
            f"({len(duplicates)} duplicates total)"
        )


def _validate_monotonic_index(X: pd.DataFrame) -> None:
    """Проверка монотонности временного индекса."""
    if not X.index.is_monotonic_increasing:
        # Найти первое нарушение монотонности
        diffs = np.diff(X.index.astype(np.int64))
        violation_idx = np.where(diffs <= 0)[0]
        if len(violation_idx) > 0:
            idx = violation_idx[0] + 1
            raise TimeSeriesError(
                f"DatetimeIndex is not monotonically increasing. "
                f"Violation at position {idx}: {X.index[idx-1]} -> {X.index[idx]}"
            )


def _validate_no_missing(X: pd.DataFrame) -> None:
    """Проверка отсутствия пропусков."""
    if X.isna().any().any():
        missing_counts = X.isna().sum()
        missing_cols = missing_counts[missing_counts > 0]
        raise TimeSeriesError(
            f"Input contains missing values in columns: {missing_cols.to_dict()}"
        )


def _validate_missing_ratio(X: pd.DataFrame, max_ratio: float) -> None:
    """Проверка доли пропусков."""
    missing_ratio = X.isna().sum().sum() / (X.shape[0] * X.shape[1])
    if missing_ratio > max_ratio:
        raise TimeSeriesError(
            f"Missing value ratio {missing_ratio:.2%} exceeds maximum allowed {max_ratio:.2%}"
        )


def _validate_no_constant_columns(X: pd.DataFrame) -> None:
    """Проверка отсутствия константных столбцов."""
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    constant_cols = [col for col in numeric_cols if X[col].nunique() <= 1]
    
    if constant_cols:
        raise TimeSeriesError(
            f"Input contains constant columns: {constant_cols}. "
            "Constant features provide no information for modeling."
        )

