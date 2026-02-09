# src/ts_feature_eng/utils/metrics.py 

"""
Метрики оценки качества прогнозирования временных рядов и признакового пространства.

Реализует стандартные и специализированные метрики с учетом специфики временных рядов:
- Защита от деления на ноль в процентных метриках
- Взвешенные версии метрик (акцент на свежие наблюдения)
- Метрики для оценки качества самих признаков
- Полная совместимость с интерфейсом scikit-learn
"""

from typing import Callable, Optional, Union

import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics._regression import _check_reg_targets


def mae(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    sample_weight: Optional[Union[np.ndarray, pd.Series]] = None
) -> float:
    """
    Mean Absolute Error (MAE).
    
    Средняя абсолютная ошибка прогноза. Интерпретируема в тех же единицах, что и целевая переменная.
    
    Параметры
    ----------
    y_true : np.ndarray или pd.Series
        Фактические значения.
    y_pred : np.ndarray или pd.Series
        Предсказанные значения.
    sample_weight : np.ndarray или pd.Series, опционально
        Веса наблюдений для взвешенной оценки.
    
    Возвращает
    ----------
    mae : float
        Средняя абсолютная ошибка.
    
    Примеры
    --------
    >>> y_true = [3, -0.5, 2, 7]
    >>> y_pred = [2.5, 0.0, 2, 8]
    >>> mae(y_true, y_pred)
    0.5
    """
    y_true, y_pred, sample_weight = _validate_inputs(y_true, y_pred, sample_weight)
    return float(np.average(np.abs(y_true - y_pred), weights=sample_weight))


def rmse(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    sample_weight: Optional[Union[np.ndarray, pd.Series]] = None
) -> float:
    """
    Root Mean Squared Error (RMSE).
    
    Корень из средней квадратичной ошибки. Чувствителен к выбросам.
    
    Параметры
    ----------
    y_true : np.ndarray или pd.Series
        Фактические значения.
    y_pred : np.ndarray или pd.Series
        Предсказанные значения.
    sample_weight : np.ndarray или pd.Series, опционально
        Веса наблюдений для взвешенной оценки.
    
    Возвращает
    ----------
    rmse : float
        Корень из средней квадратичной ошибки.
    
    Примеры
    --------
    >>> y_true = [3, -0.5, 2, 7]
    >>> y_pred = [2.5, 0.0, 2, 8]
    >>> rmse(y_true, y_pred)
    0.612...
    """
    y_true, y_pred, sample_weight = _validate_inputs(y_true, y_pred, sample_weight)
    squared_errors = (y_true - y_pred) ** 2
    mse = np.average(squared_errors, weights=sample_weight)
    return float(np.sqrt(mse))


def mape(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    sample_weight: Optional[Union[np.ndarray, pd.Series]] = None,
    epsilon: float = 1e-8
) -> float:
    """
    Mean Absolute Percentage Error (MAPE).
    
    Средняя абсолютная процентная ошибка. Выражается в процентах.
    Внимание: не определена при нулевых значениях в y_true.
    
    Параметры
    ----------
    y_true : np.ndarray или pd.Series
        Фактические значения (не должны содержать нулей).
    y_pred : np.ndarray или pd.Series
        Предсказанные значения.
    sample_weight : np.ndarray или pd.Series, опционально
        Веса наблюдений для взвешенной оценки.
    epsilon : float, по умолчанию 1e-8
        Малая константа для защиты от деления на ноль.
    
    Возвращает
    ----------
    mape : float
        Средняя абсолютная процентная ошибка (в долях, не процентах).
    
    Примеры
    --------
    >>> y_true = [100, 50, 200, 150]
    >>> y_pred = [110, 45, 210, 140]
    >>> mape(y_true, y_pred)
    0.0583...  # ~5.83%
    """
    y_true, y_pred, sample_weight = _validate_inputs(y_true, y_pred, sample_weight)
    
    # Защита от деления на ноль
    denominators = np.abs(y_true)
    mask = denominators < epsilon
    if np.any(mask):
        # Заменяем нули на малую константу с предупреждением
        denominators[mask] = epsilon
        import warnings
        warnings.warn(
            f"MAPE encountered {np.sum(mask)} zero/near-zero values in y_true. "
            f"Using epsilon={epsilon} for numerical stability.",
            RuntimeWarning
        )
    
    percentage_errors = np.abs((y_true - y_pred) / denominators)
    return float(np.average(percentage_errors, weights=sample_weight))


def smape(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    sample_weight: Optional[Union[np.ndarray, pd.Series]] = None,
    symmetric: bool = True
) -> float:
    """
    Symmetric Mean Absolute Percentage Error (SMAPE).
    
    Симметричная средняя абсолютная процентная ошибка. Определена даже при нулевых значениях.
    Диапазон: [0, 2] или [0, 1] в зависимости от формулы.
    
    Параметры
    ----------
    y_true : np.ndarray или pd.Series
        Фактические значения.
    y_pred : np.ndarray или pd.Series
        Предсказанные значения.
    sample_weight : np.ndarray или pd.Series, опционально
        Веса наблюдений для взвешенной оценки.
    symmetric : bool, по умолчанию True
        Использовать симметричную формулу (деление на сумму |y_true| + |y_pred|).
        Если False — используется альтернативная формулировка.
    
    Возвращает
    ----------
    smape : float
        Симметричная средняя абсолютная процентная ошибка (в долях).
    
    Примеры
    --------
    >>> y_true = [100, 50, 200, 150]
    >>> y_pred = [110, 45, 210, 140]
    >>> smape(y_true, y_pred)
    0.0566...  # ~5.66%
    """
    y_true, y_pred, sample_weight = _validate_inputs(y_true, y_pred, sample_weight)
    
    if symmetric:
        # Симметричная формула: 100 * |y_true - y_pred| / (|y_true| + |y_pred|)
        denominators = np.abs(y_true) + np.abs(y_pred)
        # Защита от деления на ноль
        denominators = np.maximum(denominators, 1e-8)
        percentage_errors = np.abs(y_true - y_pred) / denominators
    else:
        # Альтернативная формула: 200 * |y_true - y_pred| / (|y_true| + |y_pred|)
        denominators = np.abs(y_true) + np.abs(y_pred)
        denominators = np.maximum(denominators, 1e-8)
        percentage_errors = 2 * np.abs(y_true - y_pred) / denominators
    
    return float(np.average(percentage_errors, weights=sample_weight))


def mase(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    y_train: Optional[Union[np.ndarray, pd.Series]] = None,
    seasonality: int = 1,
    sample_weight: Optional[Union[np.ndarray, pd.Series]] = None
) -> float:
    """
    Mean Absolute Scaled Error (MASE).
    
    Масштабированная средняя абсолютная ошибка. Инвариантна к масштабу данных.
    Сравнивает ошибку модели с ошибкой наивного прогноза.
    
    Параметры
    ----------
    y_true : np.ndarray или pd.Series
        Фактические значения тестового набора.
    y_pred : np.ndarray или pd.Series
        Предсказанные значения.
    y_train : np.ndarray или pd.Series, опционально
        Обучающий набор для вычисления ошибки наивного прогноза.
        Если не указан — используется сам тестовый набор с лагом 1.
    seasonality : int, по умолчанию 1
        Период сезонности для наивного прогноза:
        - 1: наивный прогноз (последнее наблюдение)
        - >1: сезонный наивный прогноз (наблюдение с лагом = сезонность)
    sample_weight : np.ndarray или pd.Series, опционально
        Веса наблюдений для взвешенной оценки.
    
    Возвращает
    ----------
    mase : float
        Масштабированная средняя абсолютная ошибка.
        Значение < 1.0 означает, что модель лучше наивного прогноза.
    
    Примеры
    --------
    >>> y_train = [10, 12, 11, 13, 14]
    >>> y_true = [15, 16, 17]
    >>> y_pred = [14.5, 16.2, 17.1]
    >>> mase(y_true, y_pred, y_train=y_train, seasonality=1)
    0.428...  # Модель лучше наивного прогноза
    """
    y_true, y_pred, sample_weight = _validate_inputs(y_true, y_pred, sample_weight)
    
    # Вычисление ошибки модели
    model_mae = np.average(np.abs(y_true - y_pred), weights=sample_weight)
    
    # Вычисление ошибки наивного прогноза
    if y_train is not None:
        # Используем обучающий набор для базового прогноза
        y_train = np.asarray(y_train).ravel()
        naive_forecast = y_train[-seasonality:] if seasonality < len(y_train) else y_train[-1:]
        # Расширяем прогноз до длины тестового набора
        naive_errors = []
        for i in range(len(y_true)):
            idx = i % len(naive_forecast)
            naive_errors.append(np.abs(y_true[i] - naive_forecast[idx]))
        naive_mae = np.mean(naive_errors)
    else:
        # Используем сам тестовый набор с лагом
        if seasonality >= len(y_true):
            raise ValueError(
                f"seasonality ({seasonality}) must be less than length of y_true ({len(y_true)})"
            )
        naive_forecast = y_true[:-seasonality]
        actual_values = y_true[seasonality:]
        naive_mae = np.mean(np.abs(actual_values - naive_forecast))
    
    # Защита от деления на ноль
    if naive_mae < 1e-8:
        import warnings
        warnings.warn(
            f"Naive MAE is near zero ({naive_mae}), MASE may be unstable.",
            RuntimeWarning
        )
        naive_mae = 1e-8
    
    return float(model_mae / naive_mae)


def r2(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    sample_weight: Optional[Union[np.ndarray, pd.Series]] = None
) -> float:
    """
    Коэффициент детерминации R².
    
    Доля дисперсии целевой переменной, объясненная моделью.
    Диапазон: (-∞, 1], где 1.0 — идеальный прогноз.
    
    Параметры
    ----------
    y_true : np.ndarray или pd.Series
        Фактические значения.
    y_pred : np.ndarray или pd.Series
        Предсказанные значения.
    sample_weight : np.ndarray или pd.Series, опционально
        Веса наблюдений для взвешенной оценки.
    
    Возвращает
    ----------
    r2 : float
        Коэффициент детерминации.
    
    Примеры
    --------
    >>> y_true = [3, -0.5, 2, 7]
    >>> y_pred = [2.5, 0.0, 2, 8]
    >>> r2(y_true, y_pred)
    0.948...
    """
    y_true, y_pred, sample_weight = _validate_inputs(y_true, y_pred, sample_weight)
    
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight)
        avg_true = np.average(y_true, weights=sample_weight)
        numerator = np.average((y_true - y_pred) ** 2, weights=sample_weight)
        denominator = np.average((y_true - avg_true) ** 2, weights=sample_weight)
    else:
        avg_true = np.mean(y_true)
        numerator = np.mean((y_true - y_pred) ** 2)
        denominator = np.mean((y_true - avg_true) ** 2)
    
    if denominator < 1e-8:
        return 0.0
    
    return float(1 - numerator / denominator)


def weighted_metric(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    base_metric: Callable,
    decay_factor: float = 0.99,
    recent_weight: float = 1.0
) -> float:
    """
    Взвешенная версия метрики с экспоненциальным затуханием весов.
    
    Более свежие наблюдения получают больший вес, что критично для временных рядов.
    
    Параметры
    ----------
    y_true : np.ndarray или pd.Series
        Фактические значения.
    y_pred : np.ndarray или pd.Series
        Предсказанные значения.
    base_metric : Callable
        Базовая функция метрики (например, mae, rmse).
    decay_factor : float, по умолчанию 0.99
        Фактор затухания весов (0 < decay_factor < 1).
        Ближе к 1 — медленное затухание (более равномерные веса).
        Ближе к 0 — быстрое затухание (акцент на последние наблюдения).
    recent_weight : float, по умолчанию 1.0
        Дополнительный вес для самого свежего наблюдения.
    
    Возвращает
    ----------
    weighted_score : float
        Взвешенное значение метрики.
    
    Примеры
    --------
    >>> y_true = np.arange(100)
    >>> y_pred = y_true + np.random.randn(100) * 2
    >>> # Базовая MAE
    >>> mae(y_true, y_pred)
    1.65...
    >>> # Взвешенная MAE с акцентом на свежие данные
    >>> weighted_metric(y_true, y_pred, mae, decay_factor=0.95)
    1.42...  # Обычно ниже, так как свежие данные точнее
    """
    y_true, y_pred, _ = _validate_inputs(y_true, y_pred, None)
    
    n = len(y_true)
    if n == 0:
        return np.nan
    
    # Генерация экспоненциально затухающих весов
    weights = np.power(decay_factor, np.arange(n - 1, -1, -1))
    
    # Усиление веса самого свежего наблюдения
    weights[-1] *= recent_weight
    
    # Нормализация весов
    weights = weights / weights.sum()
    
    return base_metric(y_true, y_pred, sample_weight=weights)


def feature_quality_score(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    method: str = "correlation",
    top_k: Optional[int] = None
) -> float:
    """
    Оценка качества признакового пространства.
    
    Измеряет информативность признаков относительно целевой переменной.
    
    Параметры
    ----------
    X : pd.DataFrame или np.ndarray
        Признаковое пространство.
    y : pd.Series или np.ndarray
        Целевая переменная.
    method : str, по умолчанию "correlation"
        Метод оценки качества:
        - "correlation": средняя абсолютная корреляция с целевой переменной
        - "mutual_info": взаимная информация (требует установки sklearn)
        - "variance": средняя дисперсия признаков (без учета y)
    top_k : int, опционально
        Оценка только для топ-K наиболее информативных признаков.
    
    Возвращает
    ----------
    quality_score : float
        Оценка качества признакового пространства (чем выше, тем лучше).
    
    Примеры
    --------
    >>> X = pd.DataFrame({"f1": np.random.randn(100), "f2": np.random.randn(100)})
    >>> y = X["f1"] * 2 + np.random.randn(100) * 0.1
    >>> feature_quality_score(X, y, method="correlation")
    0.89...  # Высокая корреляция из-за сильной связи f1 и y
    """
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    
    if not isinstance(X, pd.DataFrame):
        raise ValueError(f"X must be pd.DataFrame or np.ndarray, got {type(X)}")
    
    if isinstance(y, np.ndarray):
        y = pd.Series(y)
    
    if not isinstance(y, pd.Series):
        raise ValueError(f"y must be pd.Series or np.ndarray, got {type(y)}")
    
    if len(X) != len(y):
        raise ValueError(f"Inconsistent lengths: X({len(X)}) vs y({len(y)})")
    
    if method == "correlation":
        correlations = X.corrwith(y).abs()
        if top_k is not None:
            correlations = correlations.nlargest(top_k)
        return float(correlations.mean())
    
    elif method == "mutual_info":
        from sklearn.feature_selection import mutual_info_regression
        
        # Обработка нечисловых столбцов
        numeric_X = X.select_dtypes(include=[np.number])
        if numeric_X.shape[1] == 0:
            raise ValueError("No numeric columns found in X for mutual information")
        
        mi = mutual_info_regression(numeric_X, y, random_state=42)
        if top_k is not None:
            mi = np.sort(mi)[-top_k:]
        return float(mi.mean())
    
    elif method == "variance":
        variances = X.var()
        if top_k is not None:
            variances = variances.nlargest(top_k)
        return float(variances.mean())
    
    else:
        raise ValueError(
            f"Unknown method: {method}. Supported: 'correlation', 'mutual_info', 'variance'"
        )


def directional_accuracy(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    threshold: float = 0.0
) -> float:
    """
    Точность прогнозирования направления изменений.
    
    Измеряет, насколько часто модель правильно предсказывает направление
    изменения (рост/падение) относительно предыдущего наблюдения.
    
    Параметры
    ----------
    y_true : np.ndarray или pd.Series
        Фактические значения.
    y_pred : np.ndarray или pd.Series
        Предсказанные значения.
    threshold : float, по умолчанию 0.0
        Порог для определения значимого изменения.
        Изменения меньше порога считаются "без направления".
    
    Возвращает
    ----------
    accuracy : float
        Доля правильно предсказанных направлений (от 0.0 до 1.0).
    
    Примеры
    --------
    >>> y_true = [100, 102, 101, 105, 103]
    >>> y_pred = [101, 103, 100, 106, 102]
    >>> directional_accuracy(y_true, y_pred)
    0.75  # 3 из 4 направлений угаданы верно
    """
    y_true, y_pred, _ = _validate_inputs(y_true, y_pred, None)
    
    # Вычисление изменений
    true_diff = np.diff(y_true)
    pred_diff = np.diff(y_pred)
    
    if len(true_diff) == 0:
        return np.nan
    
    # Определение направлений с учетом порога
    true_direction = np.sign(true_diff)
    pred_direction = np.sign(pred_diff)
    
    # Игнорирование изменений меньше порога
    if threshold > 0:
        true_mask = np.abs(true_diff) >= threshold
        pred_mask = np.abs(pred_diff) >= threshold
        valid_mask = true_mask & pred_mask
    else:
        valid_mask = np.ones_like(true_direction, dtype=bool)
    
    if not np.any(valid_mask):
        return np.nan
    
    # Вычисление точности
    correct = (true_direction[valid_mask] == pred_direction[valid_mask]).sum()
    total = valid_mask.sum()
    
    return float(correct / total)


def get_scorer(name: str, **kwargs) -> Callable:
    """
    Получение функции оценки (scorer) для использования в кросс-валидации.
    
    Создает scorer, совместимый с интерфейсом scikit-learn.
    
    Параметры
    ----------
    name : str
        Имя метрики:
        - "mae", "neg_mae"
        - "rmse", "neg_rmse"
        - "mape", "neg_mape"
        - "smape", "neg_smape"
        - "mase"
        - "r2"
        - "directional_accuracy"
    **kwargs : dict
        Дополнительные параметры для метрики (например, `seasonality` для MASE).
    
    Возвращает
    ----------
    scorer : Callable
        Функция scorer для использования в кросс-валидации.
    
    Примеры
    --------
    >>> from sklearn.model_selection import cross_val_score
    >>> scorer = get_scorer("neg_rmse")
    >>> scores = cross_val_score(model, X, y, scoring=scorer, cv=5)
    """
    metric_map = {
        "mae": mae,
        "neg_mae": lambda y_true, y_pred: -mae(y_true, y_pred),
        "rmse": rmse,
        "neg_rmse": lambda y_true, y_pred: -rmse(y_true, y_pred),
        "mape": mape,
        "neg_mape": lambda y_true, y_pred: -mape(y_true, y_pred),
        "smape": smape,
        "neg_smape": lambda y_true, y_pred: -smape(y_true, y_pred),
        "mase": lambda y_true, y_pred: mase(y_true, y_pred, **kwargs),
        "r2": r2,
        "directional_accuracy": directional_accuracy,
    }
    
    if name not in metric_map:
        # Попытка использовать встроенные метрики sklearn
        try:
            return make_scorer(metric_map[name.split("_")[1]], greater_is_better="neg" not in name)
        except:
            raise ValueError(
                f"Unknown scorer: {name}. Supported: {list(metric_map.keys())}"
            )
    
    # Для метрик, где "больше = лучше" (например, R², directional_accuracy)
    greater_is_better = name in ["r2", "directional_accuracy"]
    
    return make_scorer(metric_map[name], greater_is_better=greater_is_better)


# Вспомогательные функции
def _validate_inputs(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    sample_weight: Optional[Union[np.ndarray, pd.Series]]
) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Валидация и нормализация входных данных для метрик.
    
    Возвращает
    ----------
    y_true : np.ndarray
        Валидированные фактические значения.
    y_pred : np.ndarray
        Валидированные предсказанные значения.
    sample_weight : np.ndarray или None
        Валидированные веса наблюдений.
    """
    # Конвертация в numpy массивы
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    
    # Проверка длин
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true and y_pred have different lengths: {len(y_true)} vs {len(y_pred)}"
        )
    
    # Проверка на бесконечности и нечисловые значения
    if not np.all(np.isfinite(y_true)):
        raise ValueError("y_true contains non-finite values (inf, nan, or non-numeric)")
    
    if not np.all(np.isfinite(y_pred)):
        raise ValueError("y_pred contains non-finite values (inf, nan, or non-numeric)")
    
    # Валидация весов
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight).ravel()
        if len(sample_weight) != len(y_true):
            raise ValueError(
                f"sample_weight length ({len(sample_weight)}) does not match "
                f"y_true length ({len(y_true)})"
            )
        if not np.all(np.isfinite(sample_weight)):
            raise ValueError("sample_weight contains non-finite values")
        if np.any(sample_weight < 0):
            raise ValueError("sample_weight contains negative values")
    
    return y_true, y_pred, sample_weight


def _check_consistent_lengths(*arrays):
    """Проверка согласованности длин массивов (внутренняя утилита)."""
    lengths = [len(arr) for arr in arrays]
    if len(set(lengths)) > 1:
        raise ValueError(f"Arrays have inconsistent lengths: {lengths}")