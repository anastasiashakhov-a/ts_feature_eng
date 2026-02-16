# src/ts_feature_eng/transformers/lag.py

"""
Модуль для генерации обязательных лагов временного ряда.

Предоставляет LagTransformer — специализированный трансформер для создания
фиксированных лагов, которые не участвуют в процессе отбора признаков и всегда
включаются в финальное признаковое пространство.
"""

from typing import List, Optional, Union

import numpy as np
import pandas as pd

from ..base import TimeSeriesTransformer


class LagTransformer(TimeSeriesTransformer):
    """
    Трансформер для генерации фиксированных лагов временного ряда.
    
    Создает заданные лаги для каждой колонки входного DataFrame. Лаги
    являются обязательными признаками и не подлежат удалению в процессе
    постфильтрации признаков.
    
    Параметры
    ----------
    lags : List[int]
        Список лагов для генерации. Каждый лаг должен быть положительным
        целым числом, представляющим количество шагов назад.
        Пример: [1, 4, 24, 168] для 15 мин, 1 час, 24 часа, 7 дней (при 15-минутной частоте).
    
    Атрибуты
    ----------
    lags_ : List[int]
        Валидированный список лагов после обучения.
    feature_names_ : List[str]
        Имена сгенерированных признаков.
    
    Примеры
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from ts_feature_eng.transformers.lag import LagTransformer
    >>> 
    >>> # Создаем тестовый временной ряд
    >>> dates = pd.date_range("2023-01-01", periods=100, freq="H")
    >>> df = pd.DataFrame({"value": np.arange(100)}, index=dates)
    >>> 
    >>> # Создаем и применяем трансформер лагов
    >>> lag_transformer = LagTransformer(lags=[1, 24])
    >>> df_lagged = lag_transformer.fit_transform(df)
    >>> 
    >>> print(f"Оригинальные колонки: {df.columns.tolist()}")
    >>> print(f"Лаг-колонки: {[col for col in df_lagged.columns if 'lag_' in col]}")
    >>> print(f"Значение lag_1 для последнего наблюдения: {df_lagged['value_lag_1'].iloc[-1]}")
    """
    
    def __init__(self, lags: List[int]):
        if not isinstance(lags, list) or not all(isinstance(lag, int) and lag > 0 for lag in lags):
            raise ValueError("lags must be a list of positive integers")
        
        if len(lags) == 0:
            raise ValueError("lags list cannot be empty")
        
        self.lags = sorted(list(set(lags)))  # Удаляем дубликаты и сортируем
    
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> "LagTransformer":
        """
        Обучение трансформера лагов.
        
        Поскольку лаги являются детерминированными преобразованиями,
        метод fit только валидирует входные данные и сохраняет параметры.
        
        Параметры
        ----------
        X : pd.DataFrame или np.ndarray
            Входные данные временного ряда.
        y : pd.Series, np.ndarray или None, опционально
            Целевая переменная (игнорируется).
        
        Возвращает
        ----------
        self : LagTransformer
            Обученный трансформер.
        """
        # Валидация входных данных
        X_validated = self._validate_data(X)
        
        # Проверка, что максимальный лаг меньше длины ряда
        max_lag = max(self.lags)
        if max_lag >= len(X_validated):
            raise ValueError(
                f"Maximum lag ({max_lag}) must be less than the number of observations ({len(X_validated)})"
            )
        
        self.lags_ = self.lags.copy()
        self.n_features_in_ = X_validated.shape[1]
        self.feature_names_in_ = X_validated.columns.tolist() if isinstance(X_validated, pd.DataFrame) else [f"feature_{i}" for i in range(X_validated.shape[1])]
        
        # Генерируем имена выходных признаков
        self.feature_names_ = []
        for feature_name in self.feature_names_in_:
            for lag in self.lags_:
                self.feature_names_.append(f"{feature_name}_lag_{lag}")
        
        self.is_fitted_ = True
        return self
    
    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """
        Применение трансформера лагов к данным.
        
        Генерирует указанные лаги для каждой колонки входного DataFrame.
        
        Параметры
        ----------
        X : pd.DataFrame или np.ndarray
            Входные данные временного ряда.
        
        Возвращает
        ----------
        X_lagged : pd.DataFrame
            DataFrame с добавленными колонками лагов.
        
        Выбрасывает
        ----------
        ValueError
            Если метод вызван до обучения (fit).
        """
        if not self.is_fitted_:
            raise ValueError("LagTransformer has not been fitted. Call fit() before transform().")
        
        # Валидация входных данных
        X_validated = self._validate_data(X)
        
        # Проверка соответствия количества признаков
        if X_validated.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, but got {X_validated.shape[1]}"
            )
        
        # Создаем DataFrame для результатов
        lagged_features = {}
        
        # Генерируем лаги для каждой колонки
        for i, feature_name in enumerate(self.feature_names_in_):
            feature_series = X_validated.iloc[:, i] if isinstance(X_validated, pd.DataFrame) else X_validated[:, i]
            
            for lag in self.lags_:
                lagged_col_name = f"{feature_name}_lag_{lag}"
                # Создаем лаг с заполнением NaN в начале
                lagged_series = feature_series.shift(lag)
                lagged_features[lagged_col_name] = lagged_series
        
        # Создаем итоговый DataFrame
        X_lagged = pd.DataFrame(lagged_features, index=X_validated.index)
        
        return X_lagged
    
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
        X_lagged : pd.DataFrame
            DataFrame с добавленными колонками лагов.
        """
        return self.fit(X, y).transform(X)
    
    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """
        Получение имен выходных признаков.
        
        Параметры
        ----------
        input_features : array-like, опционально
            Имена входных признаков (игнорируются, возвращаются сохраненные имена).
        
        Возвращает
        ----------
        feature_names : np.ndarray
            Массив имен сгенерированных признаков.
        """
        if not self.is_fitted_:
            raise ValueError("LagTransformer has not been fitted. Call fit() first.")
        
        return np.array(self.feature_names_)
    
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