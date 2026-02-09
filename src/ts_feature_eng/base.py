# src/ts_feature_eng/base.py
"""
Базовые абстрактные классы для трансформеров и селекторов временных рядов.

Совместимы с интерфейсом scikit-learn для легкой интеграции в существующие пайплайны.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class TimeSeriesError(ValueError):
    """Исключение, специфичное для ошибок обработки временных рядов."""
    pass


class TimeSeriesTransformer(BaseEstimator, TransformerMixin, ABC):
    """
    Абстрактный базовый класс для трансформеров временных рядов.
    
    Все трансформеры должны реализовывать методы:
    - fit: обучение на данных (может быть пустым для stateless-трансформеров)
    - transform: применение преобразования к данным
    - get_feature_names: получение имен сгенерированных признаков
    
    Поддерживает интерфейс scikit-learn (BaseEstimator + TransformerMixin),
    что позволяет использовать трансформеры в пайплайнах sklearn.
    """
    
    def __init__(self):
        self.feature_names_: List[str] = []
        self.is_fitted_: bool = False
    
    @abstractmethod
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> "TimeSeriesTransformer":
        """
        Обучение трансформера на данных.
        
        Параметры
        ----------
        X : pd.DataFrame или np.ndarray
            Входные данные временного ряда. Для многомерных рядов ожидается DataFrame.
        y : pd.Series или np.ndarray, опционально
            Целевая переменная (требуется только для supervised-трансформеров)
        
        Возвращает
        ----------
        self : TimeSeriesTransformer
            Обученный трансформер
        """
        pass
    
    @abstractmethod
    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """
        Применение трансформации к данным.
        
        Параметры
        ----------
        X : pd.DataFrame или np.ndarray
            Входные данные временного ряда
        
        Возвращает
        ----------
        X_transformed : pd.DataFrame
            DataFrame с сгенерированными признаками. Индекс сохраняется от исходных данных.
        """
        pass
    
    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """
        Получение имен сгенерированных признаков.
        
        Возвращает
        ----------
        feature_names : List[str]
            Список имен признаков в порядке их генерации
        """
        pass
    
    def fit_transform(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> pd.DataFrame:
        """
        Обучение трансформера и применение преобразования за один шаг.
        
        Параметры
        ----------
        X : pd.DataFrame или np.ndarray
            Входные данные временного ряда
        y : pd.Series или np.ndarray, опционально
            Целевая переменная
        
        Возвращает
        ----------
        X_transformed : pd.DataFrame
            DataFrame с сгенерированными признаками
        """
        return self.fit(X, y).transform(X)
    
    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """
        Совместимость с интерфейсом scikit-learn 1.0+.
        
        Возвращает имена признаков в формате numpy array.
        
        Параметры
        ----------
        input_features : array-like, опционально
            Игнорируется (требуется для совместимости с интерфейсом sklearn)
        
        Возвращает
        ----------
        feature_names : np.ndarray
            Массив имен признаков
        """
        if not self.is_fitted_:
            raise TimeSeriesError("Transformer is not fitted. Call fit() first.")
        return np.array(self.get_feature_names())
    
    def _validate_input(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """
        Валидация и нормализация входных данных.
        
        Параметры
        ----------
        X : pd.DataFrame или np.ndarray
            Входные данные
        
        Возвращает
        ----------
        X_validated : pd.DataFrame
            Валидированный DataFrame с корректным индексом
        
        Выбрасывает
        ----------
        TimeSeriesError
            Если данные не прошли валидацию
        """
        if X is None:
            raise TimeSeriesError("Input data cannot be None")
        
        # Конвертация в DataFrame при необходимости
        if isinstance(X, np.ndarray):
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        
        if not isinstance(X, pd.DataFrame):
            raise TimeSeriesError(f"Expected pd.DataFrame or np.ndarray, got {type(X)}")
        
        if X.empty:
            raise TimeSeriesError("Input DataFrame is empty")
        
        # Проверка на пропуски в индексе (для временных рядов)
        if isinstance(X.index, pd.DatetimeIndex):
            if X.index.has_duplicates:
                raise TimeSeriesError("DatetimeIndex contains duplicate timestamps")
            if not X.index.is_monotonic_increasing:
                raise TimeSeriesError("DatetimeIndex is not monotonically increasing")
        
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            # ИСПРАВЛЕНО: разрешаем NaN (пропуски), запрещаем ТОЛЬКО бесконечности
            # Пропуски будут обработаны позже в _handle_missing_values
            if np.any(np.isinf(X[numeric_cols].values)):
                raise TimeSeriesError("Input contains infinite values (inf or -inf)")
        
        return X
    
    def _set_feature_names(self, base_name: str, statistics: List[str]) -> None:
        """
        Установка имен признаков по шаблону {базовое_имя}.{статистика}.
        
        Параметры
        ----------
        base_name : str
            Базовое имя признака (например, "value.diff")
        statistics : List[str]
            Список статистик (например, ["mean", "std", "slope"])
        """
        self.feature_names_ = [f"{base_name}.{stat}" for stat in statistics]
        self.is_fitted_ = True


class FeatureSelector(BaseEstimator, TransformerMixin, ABC):
    """
    Абстрактный базовый класс для селекторов признаков.
    
    Все селекторы должны реализовывать методы:
    - fit: обучение на данных с использованием целевой переменной
    - transform: применение отбора признаков
    - get_selected_features: получение списка отобранных признаков
    
    Поддерживает интерфейс scikit-learn для интеграции в пайплайны.
    """
    
    def __init__(self):
        self.selected_features_: List[str] = []
        self.feature_importances_: Optional[np.ndarray] = None
        self.is_fitted_: bool = False
    
    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray]
    ) -> "FeatureSelector":
        """
        Обучение селектора на данных.
        
        Параметры
        ----------
        X : pd.DataFrame
            DataFrame с признаками
        y : pd.Series или np.ndarray
            Целевая переменная (обязательна для обучения)
        
        Возвращает
        ----------
        self : FeatureSelector
            Обученный селектор
        """
        pass
    
    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Применение отбора признаков к данным.
        
        Параметры
        ----------
        X : pd.DataFrame
            DataFrame с признаками
        
        Возвращает
        ----------
        X_transformed : pd.DataFrame
            DataFrame только с отобранными признаками
        """
        pass
    
    @abstractmethod
    def get_selected_features(self) -> List[str]:
        """
        Получение списка отобранных признаков.
        
        Возвращает
        ----------
        selected_features : List[str]
            Список имен отобранных признаков
        """
        pass
    
    def fit_transform(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray]
    ) -> pd.DataFrame:
        """
        Обучение селектора и применение отбора за один шаг.
        
        Параметры
        ----------
        X : pd.DataFrame
            DataFrame с признаками
        y : pd.Series или np.ndarray
            Целевая переменная
        
        Возвращает
        ----------
        X_transformed : pd.DataFrame
            DataFrame только с отобранными признаками
        """
        return self.fit(X, y).transform(X)
    
    def get_support(self, indices: bool = False) -> Union[List[bool], np.ndarray]:
        """
        Получение маски отобранных признаков (совместимость с sklearn).
        
        Параметры
        ----------
        indices : bool, по умолчанию False
            Если True, возвращает индексы отобранных признаков,
            иначе — булеву маску
        
        Возвращает
        ----------
        mask : List[bool] или np.ndarray
            Булева маска или массив индексов отобранных признаков
        """
        if not self.is_fitted_:
            raise TimeSeriesError("Selector is not fitted. Call fit() first.")
        
        mask = [col in self.selected_features_ for col in self.feature_names_in_]
        
        if indices:
            return np.where(mask)[0]
        return mask
    
    def _validate_input(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray]
    ) -> tuple[pd.DataFrame, np.ndarray]:
        """
        Валидация входных данных для селектора.
        
        Параметры
        ----------
        X : pd.DataFrame
            DataFrame с признаками
        y : pd.Series или np.ndarray
            Целевая переменная
        
        Возвращает
        ----------
        X_validated : pd.DataFrame
            Валидированный DataFrame
        y_validated : np.ndarray
            Валидированный массив целевой переменной
        
        Выбрасывает
        ----------
        TimeSeriesError
            Если данные не прошли валидацию
        """
        if X is None or y is None:
            raise TimeSeriesError("X and y cannot be None")
        
        if not isinstance(X, pd.DataFrame):
            raise TimeSeriesError(f"X must be a pandas DataFrame, got {type(X)}")
        
        if X.empty:
            raise TimeSeriesError("Input DataFrame is empty")
        
        # Конвертация y в numpy array
        if isinstance(y, pd.Series):
            y = y.values
        elif not isinstance(y, np.ndarray):
            raise TimeSeriesError(f"y must be pd.Series or np.ndarray, got {type(y)}")
        
        if len(y) != len(X):
            raise TimeSeriesError(f"X and y have inconsistent lengths: {len(X)} vs {len(y)}")
        
        # Сохранение имен входных признаков для последующего использования
        self.feature_names_in_ = X.columns.tolist()

        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            # ИСПРАВЛЕНО: разрешаем NaN (пропуски), запрещаем ТОЛЬКО бесконечности
            if np.any(np.isinf(X[numeric_cols].values)):
                raise TimeSeriesError("Input contains infinite values (inf or -inf)")
        
        return X, y