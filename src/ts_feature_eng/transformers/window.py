# src/ts_feature_eng/transformers/window.py

"""
Модуль для оконных преобразований временных рядов.

Предоставляет WindowTransformer — универсальный трансформер для генерации
оконных статистик, разностных преобразований и лагов на основе скользящих окон.
"""

from typing import List, Optional, Union, Dict, Any

import numpy as np
import pandas as pd
from scipy import stats

from ..base import TimeSeriesTransformer


class WindowTransformer(TimeSeriesTransformer):
    """
    Трансформер для генерации оконных признаков временного ряда.
    
    Поддерживает различные типы преобразований (identity, diff, pct_change)
    и статистик (mean, std, min, max, slope, acf1, lag_N).
    
    Параметры
    ----------
    window_size : int
        Размер скользящего окна в наблюдениях.
    transformations : List[str], по умолчанию ["identity"]
        Список преобразований для применения к исходному ряду:
        - "identity": исходный ряд
        - "diff": первая разность
        - "pct_change": процентное изменение
        - "sma": простое скользящее среднее
    statistics : List[str], по умолчанию ["mean", "std"]
        Список статистик для вычисления над окном:
        - "mean": среднее значение
        - "std": стандартное отклонение
        - "min": минимальное значение
        - "max": максимальное значение
        - "slope": наклон линейной регрессии
        - "acf1": автокорреляция первого порядка
        - "skewness": асимметрия
        - "kurtosis": эксцесс
        - "lag_N": лаг на N шагов (например, "lag_1", "lag_24")
    min_periods : int, по умолчанию 1
        Минимальное количество наблюдений в окне для вычисления статистики.
    
    Атрибуты
    ----------
    window_size_ : int
        Валидированный размер окна после обучения.
    transformations_ : List[str]
        Валидированные преобразования после обучения.
    statistics_ : List[str]
        Валидированные статистики после обучения.
    feature_names_ : List[str]
        Имена сгенерированных признаков.
    
    Примеры
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from ts_feature_eng.transformers.window import WindowTransformer
    >>> 
    >>> # Создаем тестовый временной ряд
    >>> dates = pd.date_range("2023-01-01", periods=100, freq="H")
    >>> df = pd.DataFrame({"value": np.sin(np.arange(100) / 10) + np.random.randn(100) * 0.1}, index=dates)
    >>> 
    >>> # Создаем и применяем трансформер
    >>> window_transformer = WindowTransformer(
    ...     window_size=24,
    ...     transformations=["identity", "diff"],
    ...     statistics=["mean", "std", "slope", "lag_1", "lag_24"]
    ... )
    >>> df_windowed = window_transformer.fit_transform(df)
    >>> 
    >>> print(f"Сгенерировано признаков: {df_windowed.shape[1]}")
    >>> print(f"Примеры признаков: {list(df_windowed.columns[:5])}")
    """
    
    def __init__(
        self,
        window_size: int,
        transformations: List[str] = None,
        statistics: List[str] = None,
        min_periods: int = 1
    ):
        if window_size <= 0:
            raise ValueError("window_size must be a positive integer")
        
        self.window_size = window_size
        self.transformations = transformations or ["identity"]
        self.statistics = statistics or ["mean", "std"]
        self.min_periods = min_periods
        
        # Валидация параметров
        valid_transformations = {"identity", "diff", "pct_change", "sma"}
        invalid_transforms = set(self.transformations) - valid_transformations
        if invalid_transforms:
            raise ValueError(f"Invalid transformations: {invalid_transforms}. Valid options: {valid_transformations}")
        
        # Извлекаем лаги из статистик
        self.lag_statistics = []
        self.base_statistics = []
        
        for stat in self.statistics:
            if stat.startswith("lag_"):
                try:
                    lag_value = int(stat.split("_")[1])
                    if lag_value <= 0:
                        raise ValueError(f"lag value must be positive: {stat}")
                    self.lag_statistics.append(lag_value)
                except (ValueError, IndexError):
                    raise ValueError(f"Invalid lag statistic format: {stat}. Use 'lag_N' where N is positive integer.")
            else:
                self.base_statistics.append(stat)
        
        valid_statistics = {
            "mean", "std", "min", "max", "slope", "acf1", 
            "skewness", "kurtosis"
        }
        invalid_stats = set(self.base_statistics) - valid_statistics
        if invalid_stats:
            raise ValueError(f"Invalid statistics: {invalid_stats}. Valid options: {valid_statistics}")
    
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> "WindowTransformer":
        """
        Обучение трансформера оконных признаков.
        
        Параметры
        ----------
        X : pd.DataFrame или np.ndarray
            Входные данные временного ряда.
        y : pd.Series, np.ndarray или None, опционально
            Целевая переменная (игнорируется).
        
        Возвращает
        ----------
        self : WindowTransformer
            Обученный трансформер.
        """
        # Валидация входных данных
        X_validated = self._validate_data(X)
        
        # Проверка, что размер окна меньше длины ряда
        if self.window_size >= len(X_validated):
            raise ValueError(
                f"window_size ({self.window_size}) must be less than the number of observations ({len(X_validated)})"
            )
        
        # Сохраняем параметры
        self.window_size_ = self.window_size
        self.transformations_ = self.transformations.copy()
        self.statistics_ = self.statistics.copy()
        self.n_features_in_ = X_validated.shape[1]
        self.feature_names_in_ = (
            X_validated.columns.tolist() 
            if isinstance(X_validated, pd.DataFrame) 
            else [f"feature_{i}" for i in range(X_validated.shape[1])]
        )
        
        # Генерируем имена выходных признаков
        self.feature_names_ = []
        for feature_name in self.feature_names_in_:
            for transform in self.transformations_:
                for stat in self.base_statistics:
                    self.feature_names_.append(f"{feature_name}.{transform}.{stat}")
            
            # Добавляем лаги (они не зависят от преобразования)
            for lag in self.lag_statistics:
                self.feature_names_.append(f"{feature_name}.lag_{lag}")
        
        return self
    
    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        """
        Получение имен выходных признаков.
        
        Параметры
        ----------
        input_features : List[str], опционально
            Имена входных признаков (игнорируются, используются сохраненные имена).
        
        Возвращает
        ----------
        feature_names : List[str]
            Список имен сгенерированных признаков.
        """
        if not hasattr(self, 'feature_names_'):
            raise ValueError("WindowTransformer has not been fitted. Call fit() first.")
        
        return self.feature_names_.copy()

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """
        Применение трансформера оконных признаков к данным.
        
        Параметры
        ----------
        X : pd.DataFrame или np.ndarray
            Входные данные временного ряда.
        
        Возвращает
        ----------
        X_transformed : pd.DataFrame
            DataFrame с сгенерированными оконными признаками.
        
        Выбрасывает
        ----------
        ValueError
            Если метод вызван до обучения (fit).
        """
        if not hasattr(self, 'window_size_'):
            raise ValueError("WindowTransformer has not been fitted. Call fit() before transform().")
        
        # Валидация входных данных
        X_validated = self._validate_data(X)
        
        # Проверка соответствия количества признаков
        if X_validated.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, but got {X_validated.shape[1]}"
            )
        
        # Создаем словарь для результатов
        transformed_features = {}
        
        # Обрабатываем каждый признак
        for i, feature_name in enumerate(self.feature_names_in_):
            feature_series = (
                X_validated.iloc[:, i] 
                if isinstance(X_validated, pd.DataFrame) 
                else X_validated[:, i]
            )
            
            # Применяем преобразования
            transformed_series = {}
            for transform in self.transformations_:
                if transform == "identity":
                    transformed_series[transform] = feature_series
                elif transform == "diff":
                    transformed_series[transform] = feature_series.diff()
                elif transform == "pct_change":
                    transformed_series[transform] = feature_series.pct_change()
                elif transform == "sma":
                    transformed_series[transform] = feature_series.rolling(
                        window=self.window_size_, 
                        min_periods=self.min_periods
                    ).mean()
            
            # Вычисляем статистики для каждого преобразования
            for transform, series in transformed_series.items():
                windowed = series.rolling(window=self.window_size_, min_periods=self.min_periods)
                
                for stat in self.base_statistics:
                    feature_key = f"{feature_name}.{transform}.{stat}"
                    
                    if stat == "mean":
                        transformed_features[feature_key] = windowed.mean()
                    elif stat == "std":
                        transformed_features[feature_key] = windowed.std()
                    elif stat == "min":
                        transformed_features[feature_key] = windowed.min()
                    elif stat == "max":
                        transformed_features[feature_key] = windowed.max()
                    elif stat == "slope":
                        transformed_features[feature_key] = self._compute_slope(series, self.window_size_)
                    elif stat == "acf1":
                        transformed_features[feature_key] = self._compute_acf1(series, self.window_size_)
                    elif stat == "skewness":
                        transformed_features[feature_key] = windowed.apply(
                            lambda x: stats.skew(x, nan_policy='omit'), raw=False
                        )
                    elif stat == "kurtosis":
                        transformed_features[feature_key] = windowed.apply(
                            lambda x: stats.kurtosis(x, nan_policy='omit'), raw=False
                        )
            
            # Добавляем лаги
            for lag in self.lag_statistics:
                lag_key = f"{feature_name}.lag_{lag}"
                transformed_features[lag_key] = feature_series.shift(lag)
        
        # Создаем итоговый DataFrame
        X_transformed = pd.DataFrame(transformed_features, index=X_validated.index)
        
        return X_transformed
    
    def fit_transform(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> pd.DataFrame:
        """
        Обучение и применение трансформера за один шаг.
        
        Параметры
        ----------
        X : pd.DataFrame или np.ndarray
            Входные данные временного ряда.
        y : pd.Series, np.ndarray или None, опционально
            Целевая переменная (игнорируется).
        
        Возвращает
        ----------
        X_transformed : pd.DataFrame
            DataFrame с сгенерированными оконными признаками.
        """
        return self.fit(X, y).transform(X)
    
    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        """
        Получение имен выходных признаков.
        
        Параметры
        ----------
        input_features : List[str], опционально
            Имена входных признаков (игнорируются, используются сохраненные имена).
        
        Возвращает
        ----------
        feature_names : List[str]
            Список имен сгенерированных признаков.
        """
        if not hasattr(self, 'feature_names_'):
            raise ValueError("WindowTransformer has not been fitted. Call fit() first.")
        
        return self.feature_names_.copy()
    
    def _compute_slope(self, series: pd.Series, window_size: int) -> pd.Series:
        """Вычисление наклона линейной регрессии в скользящем окне."""
        def slope_func(window):
            if len(window.dropna()) < 2:
                return np.nan
            x = np.arange(len(window))
            y = window.values
            mask = ~np.isnan(y)
            if sum(mask) < 2:
                return np.nan
            slope, _ = np.polyfit(x[mask], y[mask], 1)
            return slope
        
        return series.rolling(window=window_size, min_periods=self.min_periods).apply(
            slope_func, raw=False
        )
    
    def _compute_acf1(self, series: pd.Series, window_size: int) -> pd.Series:
        """Вычисление автокорреляции первого порядка в скользящем окне."""
        def acf1_func(window):
            if len(window.dropna()) < 2:
                return np.nan
            return np.corrcoef(window[:-1], window[1:])[0, 1] if len(window) > 1 else np.nan
        
        return series.rolling(window=window_size, min_periods=self.min_periods).apply(
            acf1_func, raw=False
        )
    
    def _validate_data(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """
        Валидация и нормализация входных данных.
        
        Параметры
        ----------
        X : pd.DataFrame или np.ndarray
            Входные данные.
        
        Возвращает
        ----------
        X_validated : pd.DataFrame
            Валидированный DataFrame.
        """
        if isinstance(X, np.ndarray):
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        
        if not isinstance(X, pd.DataFrame):
            raise ValueError(f"X must be pd.DataFrame or np.ndarray, got {type(X)}")
        
        if X.empty:
            raise ValueError("Input DataFrame is empty")
        
        # Проверка на наличие временного индекса или RangeIndex
        valid_index_types = (pd.DatetimeIndex, pd.RangeIndex, pd.Index)
        if not isinstance(X.index, valid_index_types):
            # Попытка конвертации в RangeIndex
            X = X.reset_index(drop=True)
        
        return X