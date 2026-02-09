# src/ts_feature_eng/transformers/window.py
"""
Трансформеры на основе скользящего окна для временных рядов.

Реализует преобразования на основе статистик, вычисленных в скользящих окнах.
Поддерживает различные типы преобразований (identity, diff, sma) и статистик
(среднее, стандартное отвление, минимум, максимум, асимметрия, эксцесс, наклон, автокорреляция).
"""

from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression

from ..base import TimeSeriesError, TimeSeriesTransformer


class WindowTransformer(TimeSeriesTransformer):
    """
    Трансформер временных рядов на основе скользящего окна.
    
    Генерирует признаки путем применения статистик к скользящим окнам
    различных преобразований исходного ряда.
    
    Параметры
    ----------
    window_size : int, по умолчанию 24
        Размер скользящего окна в наблюдениях.
    transformations : List[str], по умолчанию ["identity", "diff"]
        Список преобразований для применения:
        - "identity": исходный ряд
        - "diff": первая разность (Δxₜ = xₜ - xₜ₋₁)
        - "sma": простая скользящая средняя (сглаживание)
    statistics : List[str], по умолчанию все доступные
        Список статистик для вычисления над окном:
        - "mean": среднее значение
        - "std": стандартное отклонение
        - "min": минимум
        - "max": максимум
        - "skewness": коэффициент асимметрии
        - "kurtosis": коэффициент эксцесса
        - "slope": наклон линейной регрессии по окну
        - "acf1": автокорреляция первого лага
        - "last": последнее значение в окне
    center : bool, по умолчанию False
        Если True, окно центрировано относительно текущей точки (симметричное окно).
        Если False, окно включает текущую точку и `window_size-1` предыдущих точек.
    min_periods : int, по умолчанию 1
        Минимальное количество наблюдений в окне для вычисления статистики.
        Если наблюдений меньше — результат будет NaN.
    
    Атрибуты
    ----------
    feature_names_ : List[str]
        Имена сгенерированных признаков в формате:
        "{исходный_столбец}.{преобразование}.{статистика}"
    is_fitted_ : bool
        Флаг, указывающий, был ли вызван метод fit().
    
    Примеры
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from ts_feature_eng.transformers.window import WindowTransformer
    >>> 
    >>> # Создаем тестовый временной ряд
    >>> dates = pd.date_range("2023-01-01", periods=100, freq="H")
    >>> df = pd.DataFrame({"value": np.random.randn(100)}, index=dates)
    >>> 
    >>> # Создаем и применяем трансформер
    >>> transformer = WindowTransformer(
    ...     window_size=24,
    ...     transformations=["identity", "diff"],
    ...     statistics=["mean", "std", "slope"]
    ... )
    >>> X_transformed = transformer.fit_transform(df)
    >>> 
    >>> print(X_transformed.columns.tolist())
    ['value.identity.mean', 'value.identity.std', 'value.identity.slope',
     'value.diff.mean', 'value.diff.std', 'value.diff.slope']
    """
    
    _valid_transformations = ["identity", "diff", "sma"]
    _valid_statistics = ["mean", "std", "min", "max", "skewness", "kurtosis", "slope", "acf1", "last"]
    
    def __init__(
        self,
        window_size: int = 24,
        transformations: Optional[List[str]] = None,
        statistics: Optional[List[str]] = None,
        center: bool = False,
        min_periods: int = 1,
    ):
        super().__init__()
        
        # ИСПРАВЛЕНИЕ: Конвертируем numpy.int64 и другие числовые типы в int
        if isinstance(window_size, (int, np.integer)):
            window_size = int(window_size)
        elif isinstance(window_size, float):
            if window_size.is_integer():
                window_size = int(window_size)
            else:
                raise ValueError(f"window_size must represent an integer, got {window_size}")
        else:
            raise ValueError(f"window_size must be an integer, got {type(window_size)}")
        
        if isinstance(min_periods, (int, np.integer)):
            min_periods = int(min_periods)
        elif isinstance(min_periods, float):
            if min_periods.is_integer():
                min_periods = int(min_periods)
            else:
                raise ValueError(f"min_periods must represent an integer, got {min_periods}")
        else:
            raise ValueError(f"min_periods must be an integer, got {type(min_periods)}")
        
        self.window_size = window_size
        self.transformations = transformations or ["identity", "diff"]
        self.statistics = statistics or self._valid_statistics.copy()
        self.center = center
        self.min_periods = min_periods
        
        # Валидация параметров при инициализации
        self._validate_params()

    def _validate_params(self) -> None:
        """Валидация гиперпараметров трансформера."""
        # Проверка window_size
        if not isinstance(self.window_size, int) or self.window_size <= 0:
            raise ValueError(f"window_size must be positive integer, got {self.window_size}")
        
        # Проверка min_periods
        if not isinstance(self.min_periods, int) or self.min_periods <= 0:
            raise ValueError(f"min_periods must be positive integer, got {self.min_periods}")
        
        if self.min_periods > self.window_size:
            raise ValueError(
                f"min_periods ({self.min_periods}) cannot be greater than window_size ({self.window_size})"
            )
        
        invalid_transforms = set(self.transformations) - set(self._valid_transformations)
        if invalid_transforms:
            raise ValueError(
                f"Invalid transformations: {invalid_transforms}. "
                f"Valid options: {self._valid_transformations}"
            )
        
        invalid_stats = set(self.statistics) - set(self._valid_statistics)
        if invalid_stats:
            raise ValueError(
                f"Invalid statistics: {invalid_stats}. "
                f"Valid options: {self._valid_statistics}"
            )

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y=None) -> "WindowTransformer":
        """
        Обучение трансформера (в данном случае — только валидация данных).
        
        Параметры
        ----------
        X : pd.DataFrame или np.ndarray
            Входные данные временного ряда.
        y : любое, опционально
            Целевая переменная (игнорируется, так как трансформер не использует целевую переменную).
        
        Возвращает
        ----------
        self : WindowTransformer
            Обученный трансформер.
        """
        X = self._validate_input(X)
        
        # Проверка на константные столбцы (могут вызвать проблемы со статистиками)
        constant_cols = X.columns[X.nunique() <= 1]
        if len(constant_cols) > 0:
            raise TimeSeriesError(
                f"Input contains constant columns: {list(constant_cols)}. "
                "Window statistics cannot be computed for constant series."
            )
        
        self.is_fitted_ = True
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """
        Применение трансформации к данным и генерация признаков.
        
        Параметры
        ----------
        X : pd.DataFrame или np.ndarray
            Входные данные временного ряда.
        
        Возвращает
        ----------
        X_transformed : pd.DataFrame
            DataFrame с сгенерированными признаками. Индекс сохраняется от исходных данных.
        """
        if not self.is_fitted_:
            raise TimeSeriesError("Transformer is not fitted. Call fit() first.")
        
        X = self._validate_input(X)
        
        # Словарь для накопления сгенерированных признаков
        features = {}
        feature_names = []
        
        # Обработка каждого столбца исходного DataFrame
        for col in X.columns:
            series = X[col]
            
            # Применение каждого преобразования
            for trans in self.transformations:
                transformed = self._apply_transformation(series, trans)
                
                # Вычисление каждой статистики над преобразованным рядом
                for stat in self.statistics:
                    feature_name = f"{col}.{trans}.{stat}"
                    feature_values = self._compute_statistic(transformed, stat)
                    features[feature_name] = feature_values
                    feature_names.append(feature_name)
        
        # Формирование результата
        X_transformed = pd.DataFrame(features, index=X.index)
        self.feature_names_ = feature_names
        
        return X_transformed

    def _apply_transformation(self, series: pd.Series, transformation: str) -> pd.Series:
        """
        Применение преобразования к временному ряду.
        
        Параметры
        ----------
        series : pd.Series
            Исходный временной ряд.
        transformation : str
            Тип преобразования ("identity", "diff", "sma").
        
        Возвращает
        ----------
        transformed : pd.Series
            Преобразованный временной ряд.
        """
        if transformation == "identity":
            return series
        
        elif transformation == "diff":
            return series.diff()
        
        elif transformation == "sma":
            # Простая скользящая средняя с тем же размером окна
            return series.rolling(window=self.window_size, center=self.center, min_periods=1).mean()
        
        else:
            raise ValueError(f"Unknown transformation: {transformation}")

    def _compute_statistic(self, series: pd.Series, statistic: str) -> pd.Series:
        """
        Вычисление статистики над скользящем окне.
        
        Параметры
        ----------
        series : pd.Series
            Временной ряд (возможно, уже преобразованный).
        statistic : str
            Тип статистики для вычисления.
        
        Возвращает
        ----------
        result : pd.Series
            Результат вычисления статистики для каждого окна.
        """
        # Создаем объект скользящего окна
        rolling = series.rolling(
            window=self.window_size,
            center=self.center,
            min_periods=self.min_periods
        )
        
        if statistic == "mean":
            return rolling.mean()
        
        elif statistic == "std":
            return rolling.std(ddof=1)  # ddof=1 для несмещенной оценки
        
        elif statistic == "min":
            return rolling.min()
        
        elif statistic == "max":
            return rolling.max()
        
        elif statistic == "skewness":
            # Используем метод скользящего окна для асимметрии
            return rolling.apply(lambda x: stats.skew(x, nan_policy="omit"), raw=True)
        
        elif statistic == "kurtosis":
            # Используем метод скользящего окна для эксцесса
            return rolling.apply(lambda x: stats.kurtosis(x, nan_policy="omit"), raw=True)
        
        elif statistic == "slope":
            # Наклон линейной регрессии по окну
            return rolling.apply(self._compute_slope, raw=True)
        
        elif statistic == "acf1":
            # Автокорреляция первого лага
            return rolling.apply(self._compute_acf1, raw=True)
        
        elif statistic == "last":
            # Последнее значение в окне (эквивалентно сдвигу)
            return series.shift(1) if not self.center else series
        
        else:
            raise ValueError(f"Unknown statistic: {statistic}")

    @staticmethod
    def _compute_slope(window: np.ndarray) -> float:
        """
        Вычисление наклона линейной регрессии по окну.
        
        Параметры
        ----------
        window : np.ndarray
            Массив значений в окне.
        
        Возвращает
        ----------
        slope : float
            Наклон линии регрессии (коэффициент при x).
        """
        # Удаляем NaN значения
        valid = ~np.isnan(window)
        if valid.sum() < 2:  # Нужно минимум 2 точки для регрессии
            return np.nan
        
        x = np.arange(len(window))[valid]
        y = window[valid]
        
        # Простая линейная регрессия через МНК
        if len(x) < 2:
            return np.nan
        
        # Вычисление наклона: cov(x,y)/var(x)
        x_centered = x - x.mean()
        y_centered = y - y.mean()
        numerator = np.sum(x_centered * y_centered)
        denominator = np.sum(x_centered ** 2)
        
        if denominator == 0:
            return np.nan
        
        return numerator / denominator

    @staticmethod
    def _compute_acf1(window: np.ndarray) -> float:
        """
        Вычисление автокорреляции первого лага по окну.
        
        Параметры
        ----------
        window : np.ndarray
            Массив значений в окне.
        
        Возвращает
        ----------
        acf1 : float
            Автокорреляция первого лага.
        """
        # Удаляем NaN значения
        valid = ~np.isnan(window)
        if valid.sum() < 3:  # Нужно минимум 3 точки для автокорреляции
            return np.nan
        
        series = window[valid]
        n = len(series)
        
        if n < 2:
            return np.nan
        
        # Центрируем ряд
        series_centered = series - series.mean()
        
        # Автоковариация лага 1
        autocov_1 = np.sum(series_centered[1:] * series_centered[:-1]) / n
        
        # Дисперсия
        variance = np.sum(series_centered ** 2) / n
        
        if variance == 0:
            return np.nan
        
        return autocov_1 / variance

    def get_feature_names(self) -> List[str]:
        """
        Получение имен сгенерированных признаков.
        
        Возвращает
        ----------
        feature_names : List[str]
            Список имен признаков в формате "{столбец}.{преобразование}.{статистика}".
        """
        if not self.is_fitted_:
            raise TimeSeriesError("Transformer is not fitted. Call fit() first.")
        
        return self.feature_names_
    
    def get_params(self, deep: bool = True) -> Dict[str, Union[int, List[str], bool]]:
        """
        Получение параметров трансформера (совместимость с sklearn).
        
        Параметры
        ----------
        deep : bool, по умолчанию True
            Игнорируется (требуется для совместимости с интерфейсом sklearn).
        
        Возвращает
        ----------
        params : Dict[str, Any]
            Словарь параметров трансформера.
        """
        return {
            "window_size": self.window_size,
            "transformations": self.transformations,
            "statistics": self.statistics,
            "center": self.center,
            "min_periods": self.min_periods,
        }

    def set_params(self, **params) -> "WindowTransformer":
        """
        Установка параметров трансформера (совместимость с sklearn).
        
        Параметры
        ----------
        **params : Dict[str, Any]
            Параметры для установки.
        
        Возвращает
        ----------
        self : WindowTransformer
            Трансформер с обновленными параметрами.
        """
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter {key} for WindowTransformer")
        
        # Повторная валидация после изменения параметров
        self._validate_params()
        return self