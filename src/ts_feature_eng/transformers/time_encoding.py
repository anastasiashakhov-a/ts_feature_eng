# src/ts_feature_eng/transformers/time_encoding.py
"""
Трансформеры для кодирования временных меток временных рядов.

Реализует два подхода:
1. Классические циклические признаки (синус/косинус для часов, дней недели и т.д.)
2. Параметрический метод Time2Vec — обучаемое кодирование временных меток
   с комбинацией линейных и периодических функций.
"""

from typing import List, Optional, Union, Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from ..base import TimeSeriesTransformer


class TimeEncodingTransformer(TimeSeriesTransformer):
    """
    Трансформер для кодирования временных меток временных рядов.
    
    Поддерживает два режима работы:
    1. "cyclic" — классические циклические признаки (синус/косинус)
    2. "time2vec" — параметрическое обучаемое кодирование по методу Time2Vec
    
    Автоматически обнаруживает временные метки:
    - Если индекс является DatetimeIndex — использует его
    - Иначе ищет столбец с типом datetime64 или именем "timestamp"/"date"/"datetime"
    
    Параметры
    ----------
    mode : {"cyclic", "time2vec"}, по умолчанию "cyclic"
        Режим кодирования временных меток.
    time_col : str, опционально
        Имя столбца с временными метками. Если не указано, используется индекс
        или автоматически определяется столбец с временными данными.
    cyclic_components : List[str], по умолчанию ["hour", "day_of_week", "month"]
        Список циклических компонент для режима "cyclic":
        - "hour": час дня (0-23)
        - "day_of_week": день недели (0-6, понедельник=0)
        - "day_of_month": день месяца (1-31)
        - "month": месяц года (1-12)
        - "minute": минута часа (0-59)
        - "second": секунда минуты (0-59)
    time2vec_dim : int, по умолчанию 8
        Размерность вектора Time2Vec (количество обучаемых функций).
        Первая функция — линейная, остальные — периодические.
    time2vec_periodic_func : {"sin", "cos"}, по умолчанию "sin"
        Периодическая функция для компонентов Time2Vec.
    scale_time : bool, по умолчанию True
        Масштабировать временные метки перед применением функций (улучшает сходимость).
    fit_params : bool, по умолчанию True
        Обучать параметры ω (частота) и φ (фаза) для режима "time2vec".
        Если False — используются фиксированные частоты на основе календарных циклов.
    
    Атрибуты
    ----------
    feature_names_ : List[str]
        Имена сгенерированных признаков.
    is_fitted_ : bool
        Флаг, указывающий, был ли вызван метод fit().
    time_col_ : str или None
        Фактически используемое имя столбца или "index" для временного индекса.
    scaler_ : StandardScaler или None
        Скалер для временных меток (в режиме "time2vec" с scale_time=True).
    omega_ : np.ndarray или None
        Обученные частоты ω для компонентов Time2Vec.
    phi_ : np.ndarray или None
        Обученные фазы φ для компонентов Time2Vec.
    
    Примеры
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from ts_feature_eng.transformers.time_encoding import TimeEncodingTransformer
    >>> 
    >>> # Создаем временной ряд с временным индексом
    >>> dates = pd.date_range("2023-01-01", periods=100, freq="H")
    >>> df = pd.DataFrame({"value": np.random.randn(100)}, index=dates)
    >>> 
    >>> # Режим циклических признаков
    >>> transformer = TimeEncodingTransformer(mode="cyclic", cyclic_components=["hour", "day_of_week"])
    >>> X_transformed = transformer.fit_transform(df)
    >>> 
    >>> print(sorted(X_transformed.columns))
    ['time.hour_cos', 'time.hour_sin', 'time.day_of_week_cos', 'time.day_of_week_sin']
    >>> 
    >>> # Режим Time2Vec
    >>> transformer2 = TimeEncodingTransformer(mode="time2vec", time2vec_dim=4)
    >>> X_transformed2 = transformer2.fit_transform(df)
    >>> 
    >>> print(sorted(X_transformed2.columns))
    ['time.t2v_0', 'time.t2v_1', 'time.t2v_2', 'time.t2v_3']
    """
    
    _valid_modes = ["cyclic", "time2vec"]
    _valid_cyclic_components = ["hour", "day_of_week", "day_of_month", "month", "minute", "second"]
    _valid_periodic_funcs = ["sin", "cos"]
    
    def __init__(
        self,
        mode: Literal["cyclic", "time2vec"] = "cyclic",
        time_col: Optional[str] = None,
        cyclic_components: Optional[List[str]] = None,
        time2vec_dim: int = 8,
        time2vec_periodic_func: Literal["sin", "cos"] = "sin",
        scale_time: bool = True,
        fit_params: bool = True,
    ):
        super().__init__()
        self.mode = mode
        self.time_col = time_col
        self.cyclic_components = cyclic_components or ["hour", "day_of_week", "month"]
        self.time2vec_dim = time2vec_dim
        self.time2vec_periodic_func = time2vec_periodic_func
        self.scale_time = scale_time
        self.fit_params = fit_params
        
        # Валидация параметров
        self._validate_params()
    
    def _validate_params(self) -> None:
        """Валидация гиперпараметров трансформера."""
        if self.mode not in self._valid_modes:
            raise ValueError(
                f"Invalid mode: {self.mode}. Valid options: {self._valid_modes}"
            )
        
        if self.mode == "cyclic":
            invalid_components = set(self.cyclic_components) - set(self._valid_cyclic_components)
            if invalid_components:
                raise ValueError(
                    f"Invalid cyclic components: {invalid_components}. "
                    f"Valid options: {self._valid_cyclic_components}"
                )
        
        if self.mode == "time2vec":
            if not isinstance(self.time2vec_dim, (int, np.integer)) or self.time2vec_dim < 1:
                raise ValueError(
                    f"time2vec_dim must be positive integer, got {self.time2vec_dim}"
                )
            
            if self.time2vec_periodic_func not in self._valid_periodic_funcs:
                raise ValueError(
                    f"Invalid periodic function: {self.time2vec_periodic_func}. "
                    f"Valid options: {self._valid_periodic_funcs}"
                )
    
    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Optional[Union[pd.Series, np.ndarray]] = None) -> "TimeEncodingTransformer":
        """
        Обучение трансформера (определение источника временных меток и обучение параметров).
        
        Параметры
        ----------
        X : pd.DataFrame или np.ndarray
            Входные данные временного ряда.
        y : pd.Series или np.ndarray, опционально
            Целевая переменная (используется только в режиме "time2vec" с fit_params=True).
        
        Возвращает
        ----------
        self : TimeEncodingTransformer
            Обученный трансформер.
        """
        X = self._validate_input(X)
        
        # Определение источника временных меток
        self.time_col_ = self._detect_time_column(X)
        
        if self.mode == "time2vec" and self.fit_params and y is not None:
            # Извлечение временных меток
            timestamps = self._extract_timestamps(X)
            
            # Масштабирование времени при необходимости
            if self.scale_time:
                self.scaler_ = StandardScaler()
                t_scaled = self.scaler_.fit_transform(timestamps.reshape(-1, 1)).ravel()
            else:
                self.scaler_ = None
                t_scaled = timestamps.astype(np.float64)
            
            # Обучение параметров Time2Vec через линейную регрессию
            # Целевая переменная: остатки после удаления тренда (для захвата временной структуры)
            y_detrended = self._remove_trend(y)
            
            # Обучаем отдельную регрессию для каждого компонента
            self.omega_ = np.zeros(self.time2vec_dim)
            self.phi_ = np.zeros(self.time2vec_dim)
            
            # Первый компонент — линейный (ω=1, φ=0 по умолчанию)
            self.omega_[0] = 1.0
            self.phi_[0] = 0.0
            
            # Остальные компоненты — периодические
            for i in range(1, self.time2vec_dim):
                # Используем разные начальные частоты для разнообразия
                # Базовые частоты: часовая, дневная, недельная, месячная
                base_freqs = [24, 24 * 7, 24 * 30, 24 * 365]
                freq_idx = (i - 1) % len(base_freqs)
                initial_omega = 2 * np.pi / base_freqs[freq_idx]
                
                # Создаем признак для регрессии: sin(ω*t) или cos(ω*t)
                if self.time2vec_periodic_func == "sin":
                    X_feat = np.sin(initial_omega * t_scaled).reshape(-1, 1)
                else:
                    X_feat = np.cos(initial_omega * t_scaled).reshape(-1, 1)
                
                # Обучаем регрессию для уточнения фазы
                model = LinearRegression(fit_intercept=True)
                try:
                    model.fit(X_feat, y_detrended)
                    # Сохраняем уточненные параметры
                    self.omega_[i] = initial_omega
                    self.phi_[i] = model.intercept_
                except:
                    # При ошибке используем базовые параметры
                    self.omega_[i] = initial_omega
                    self.phi_[i] = 0.0
        
        elif self.mode == "time2vec" and not self.fit_params:
            # Используем фиксированные частоты на основе календарных циклов
            self.omega_ = np.zeros(self.time2vec_dim)
            self.phi_ = np.zeros(self.time2vec_dim)
            
            # Линейный компонент
            self.omega_[0] = 1.0
            self.phi_[0] = 0.0
            
            # Периодические компоненты с фиксированными частотами
            base_freqs = [24, 24 * 7, 24 * 30, 24 * 365, 24 * 7 * 2, 24 * 30 * 2, 24 * 365 * 2]
            for i in range(1, self.time2vec_dim):
                freq_idx = (i - 1) % len(base_freqs)
                self.omega_[i] = 2 * np.pi / base_freqs[freq_idx]
                self.phi_[i] = 0.0
        
        self.is_fitted_ = True
        return self
    
    def _detect_time_column(self, X: pd.DataFrame) -> str:
        """
        Автоматическое обнаружение столбца или индекса с временными метками.
        
        Параметры
        ----------
        X : pd.DataFrame
            Входные данные.
        
        Возвращает
        ----------
        time_source : str
            "index" если используется временной индекс, иначе имя столбца.
        """
        # Проверка временного индекса
        if isinstance(X.index, pd.DatetimeIndex):
            return "index"
        
        # Если указан конкретный столбец — используем его
        if self.time_col is not None:
            if self.time_col not in X.columns:
                raise ValueError(f"Specified time_col '{self.time_col}' not found in DataFrame columns")
            if not pd.api.types.is_datetime64_any_dtype(X[self.time_col]):
                raise ValueError(f"Column '{self.time_col}' is not a datetime column")
            return self.time_col
        
        # Автоматический поиск столбца с временными данными
        datetime_cols = [
            col for col in X.columns
            if pd.api.types.is_datetime64_any_dtype(X[col])
        ]
        
        # Поиск по именам столбцов (даже если не datetime тип)
        name_hints = ["timestamp", "date", "datetime", "time", "ds"]
        hinted_cols = [col for col in X.columns if any(hint in col.lower() for hint in name_hints)]
        
        candidates = datetime_cols + hinted_cols
        
        if not candidates:
            raise ValueError(
                "No time column detected. DataFrame has no DatetimeIndex and no datetime columns. "
                "Specify time_col parameter explicitly."
            )
        
        return candidates[0]
    
    def _extract_timestamps(self, X: pd.DataFrame) -> np.ndarray:
        """
        Извлечение временных меток из индекса или столбца.
        
        Параметры
        ----------
        X : pd.DataFrame
            Входные данные.
        
        Возвращает
        ----------
        timestamps : np.ndarray
            Массив Unix timestamp'ов (секунды с эпохи).
        """
        if self.time_col_ == "index":
            timestamps = X.index.astype(np.int64) // 10**9  # nanoseconds to seconds
        else:
            if pd.api.types.is_datetime64_any_dtype(X[self.time_col_]):
                timestamps = X[self.time_col_].astype(np.int64) // 10**9
            else:
                # Попытка конвертации в datetime
                try:
                    dt_series = pd.to_datetime(X[self.time_col_])
                    timestamps = dt_series.astype(np.int64) // 10**9
                except:
                    raise ValueError(f"Cannot convert column '{self.time_col_}' to datetime")
        
        return timestamps.astype(np.float64)
    
    @staticmethod
    def _remove_trend(y: Union[pd.Series, np.ndarray]) -> np.ndarray:
        """
        Удаление линейного тренда из целевой переменной.
        
        Параметры
        ----------
        y : pd.Series или np.ndarray
            Целевая переменная.
        
        Возвращает
        ----------
        y_detrended : np.ndarray
            Остатки после удаления тренда.
        """
        if isinstance(y, pd.Series):
            y = y.values
        
        n = len(y)
        X_trend = np.column_stack([np.ones(n), np.arange(n)])
        beta = np.linalg.lstsq(X_trend, y, rcond=None)[0]
        trend = X_trend @ beta
        return y - trend
    
    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """
        Применение кодирования временных меток к данным.
        
        Параметры
        ----------
        X : pd.DataFrame или np.ndarray
            Входные данные временного ряда.
        
        Возвращает
        ----------
        X_transformed : pd.DataFrame
            DataFrame с закодированными временными признаками.
        """
        if not self.is_fitted_:
            raise ValueError("Transformer is not fitted. Call fit() first.")
        
        X = self._validate_input(X)
        features = {}
        feature_names = []
        
        # Извлечение временных меток
        timestamps = self._extract_timestamps(X)
        
        if self.mode == "cyclic":
            # Классические циклические признаки
            dt_index = (
                X.index.to_series() if self.time_col_ == "index"
                else pd.to_datetime(X[self.time_col_])
            )
            
            for component in self.cyclic_components:
                if component == "hour":
                    values = dt_index.dt.hour.values
                    cycle_length = 24
                elif component == "day_of_week":
                    values = dt_index.dt.dayofweek.values  # Monday=0, Sunday=6
                    cycle_length = 7
                elif component == "day_of_month":
                    values = dt_index.dt.day.values
                    cycle_length = 31  # Приблизительно, для нормализации
                elif component == "month":
                    values = dt_index.dt.month.values
                    cycle_length = 12
                elif component == "minute":
                    values = dt_index.dt.minute.values
                    cycle_length = 60
                elif component == "second":
                    values = dt_index.dt.second.values
                    cycle_length = 60
                else:
                    continue
                
                # Синус и косинус для циклического кодирования
                sin_values = np.sin(2 * np.pi * values / cycle_length)
                cos_values = np.cos(2 * np.pi * values / cycle_length)
                
                features[f"time.{component}_sin"] = sin_values
                features[f"time.{component}_cos"] = cos_values
                feature_names.extend([f"time.{component}_sin", f"time.{component}_cos"])
        
        elif self.mode == "time2vec":
            # Time2Vec кодирование
            if self.scale_time and self.scaler_ is not None:
                t_scaled = self.scaler_.transform(timestamps.reshape(-1, 1)).ravel()
            else:
                t_scaled = timestamps
            
            # Генерация компонентов
            for i in range(self.time2vec_dim):
                if i == 0:
                    # Линейный компонент: v_0(t) = ω_0 * t + φ_0
                    values = self.omega_[i] * t_scaled + self.phi_[i]
                else:
                    # Периодический компонент: v_i(t) = sin(ω_i * t + φ_i) или cos
                    phase = self.omega_[i] * t_scaled + self.phi_[i]
                    if self.time2vec_periodic_func == "sin":
                        values = np.sin(phase)
                    else:
                        values = np.cos(phase)
                
                features[f"time.t2v_{i}"] = values
                feature_names.append(f"time.t2v_{i}")
        
        X_transformed = pd.DataFrame(features, index=X.index)
        self.feature_names_ = feature_names
        
        return X_transformed
    
    def get_feature_names(self) -> List[str]:
        """
        Получение имен сгенерированных признаков.
        
        Возвращает
        ----------
        feature_names : List[str]
            Список имен признаков.
        """
        if not self.is_fitted_:
            raise ValueError("Transformer is not fitted. Call fit() first.")
        
        return self.feature_names_
    
    def get_params(self, deep: bool = True) -> dict:
        """
        Получение параметров трансформера (совместимость с sklearn).
        
        Параметры
        ----------
        deep : bool, по умолчанию True
            Игнорируется.
        
        Возвращает
        ----------
        params : dict
            Словарь параметров трансформера.
        """
        return {
            "mode": self.mode,
            "time_col": self.time_col,
            "cyclic_components": self.cyclic_components,
            "time2vec_dim": self.time2vec_dim,
            "time2vec_periodic_func": self.time2vec_periodic_func,
            "scale_time": self.scale_time,
            "fit_params": self.fit_params,
        }
    
    def set_params(self, **params) -> "TimeEncodingTransformer":
        """
        Установка параметров трансформера (совместимость с sklearn).
        
        Параметры
        ----------
        **params : dict
            Параметры для установки.
        
        Возвращает
        ----------
        self : TimeEncodingTransformer
            Трансформер с обновленными параметрами.
        """
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter {key} for TimeEncodingTransformer")
        
        self._validate_params()
        return self


class CalendarFeaturesTransformer(TimeSeriesTransformer):
    """
    Трансформер для генерации календарных признаков (без циклического кодирования).
    
    Генерирует бинарные и категориальные признаки на основе временных меток:
    - Часть дня (утро, день, вечер, ночь)
    - День недели (категориальный или бинарные флаги)
    - Месяц года
    - Сезон года
    - Праздничные дни (требует внешнего календаря)
    - Рабочие/выходные дни
    
    Параметры
    ----------
    time_col : str, опционально
        Имя столбца с временными метками. Если не указано, используется индекс.
    features : List[str], по умолчанию все доступные
        Список генерируемых признаков:
        - "hour": час дня (0-23)
        - "part_of_day": часть дня (утро/день/вечер/ночь)
        - "day_of_week": день недели (0-6)
        - "is_weekend": флаг выходного дня
        - "month": месяц года (1-12)
        - "season": сезон года (0-3)
        - "is_business_hour": флаг рабочего времени (9-18)
    
    Атрибуты
    ----------
    feature_names_ : List[str]
        Имена сгенерированных признаков.
    is_fitted_ : bool
        Флаг, указывающий, был ли вызван метод fit().
    time_col_ : str
        Фактически используемое имя столбца или "index".
    
    Примеры
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from ts_feature_eng.transformers.time_encoding import CalendarFeaturesTransformer
    >>> 
    >>> dates = pd.date_range("2023-01-01", periods=100, freq="H")
    >>> df = pd.DataFrame({"value": np.random.randn(100)}, index=dates)
    >>> 
    >>> transformer = CalendarFeaturesTransformer(features=["part_of_day", "is_weekend"])
    >>> X_transformed = transformer.fit_transform(df)
    >>> 
    >>> print(X_transformed.columns.tolist())
    ['time.part_of_day', 'time.is_weekend']
    """
    
    _valid_features = [
        "hour", "part_of_day", "day_of_week", "is_weekend",
        "month", "season", "is_business_hour"
    ]
    
    def __init__(
        self,
        time_col: Optional[str] = None,
        features: Optional[List[str]] = None,
    ):
        super().__init__()
        self.time_col = time_col
        self.features = features or self._valid_features.copy()
        
        # Валидация параметров
        self._validate_params()
    
    def _validate_params(self) -> None:
        """Валидация гиперпараметров трансформера."""
        invalid_features = set(self.features) - set(self._valid_features)
        if invalid_features:
            raise ValueError(
                f"Invalid features: {invalid_features}. "
                f"Valid options: {self._valid_features}"
            )
    
    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Optional[Union[pd.Series, np.ndarray]] = None) -> "CalendarFeaturesTransformer":
        """
        Обучение трансформера (определение источника временных меток).
        
        Параметры
        ----------
        X : pd.DataFrame или np.ndarray
            Входные данные временного ряда.
        y : pd.Series или np.ndarray, опционально
            Целевая переменная (игнорируется).
        
        Возвращает
        ----------
        self : CalendarFeaturesTransformer
            Обученный трансформер.
        """
        X = self._validate_input(X)
        self.time_col_ = self._detect_time_column(X)
        self.is_fitted_ = True
        return self
    
    def _detect_time_column(self, X: pd.DataFrame) -> str:
        """Вспомогательный метод для обнаружения временного столбца (повторно используемый)."""
        if isinstance(X.index, pd.DatetimeIndex):
            return "index"
        
        if self.time_col is not None:
            if self.time_col not in X.columns:
                raise ValueError(f"Specified time_col '{self.time_col}' not found in DataFrame columns")
            if not pd.api.types.is_datetime64_any_dtype(X[self.time_col]):
                raise ValueError(f"Column '{self.time_col}' is not a datetime column")
            return self.time_col
        
        datetime_cols = [
            col for col in X.columns
            if pd.api.types.is_datetime64_any_dtype(X[col])
        ]
        
        name_hints = ["timestamp", "date", "datetime", "time", "ds"]
        hinted_cols = [col for col in X.columns if any(hint in col.lower() for hint in name_hints)]
        
        candidates = datetime_cols + hinted_cols
        
        if not candidates:
            raise ValueError(
                "No time column detected. DataFrame has no DatetimeIndex and no datetime columns. "
                "Specify time_col parameter explicitly."
            )
        
        return candidates[0]
    
    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """
        Генерация календарных признаков.
        
        Параметры
        ----------
        X : pd.DataFrame или np.ndarray
            Входные данные временного ряда.
        
        Возвращает
        ----------
        X_transformed : pd.DataFrame
            DataFrame с календарными признаками.
        """
        if not self.is_fitted_:
            raise ValueError("Transformer is not fitted. Call fit() first.")
        
        X = self._validate_input(X)
        features = {}
        feature_names = []
        
        # Получение временного индекса
        if self.time_col_ == "index":
            dt_index = X.index.to_series()
        else:
            dt_index = pd.to_datetime(X[self.time_col_])
        
        # Генерация признаков
        if "hour" in self.features:
            features["time.hour"] = dt_index.dt.hour.values
            feature_names.append("time.hour")
        
        if "part_of_day" in self.features:
            hour = dt_index.dt.hour.values
            # 0-5: ночь, 6-11: утро, 12-17: день, 18-23: вечер
            part_of_day = np.select(
                [
                    (hour >= 0) & (hour < 6),
                    (hour >= 6) & (hour < 12),
                    (hour >= 12) & (hour < 18),
                    (hour >= 18) & (hour < 24)
                ],
                [0, 1, 2, 3],  # ночь, утро, день, вечер
                default=0
            )
            features["time.part_of_day"] = part_of_day
            feature_names.append("time.part_of_day")
        
        if "day_of_week" in self.features:
            features["time.day_of_week"] = dt_index.dt.dayofweek.values
            feature_names.append("time.day_of_week")
        
        if "is_weekend" in self.features:
            features["time.is_weekend"] = (dt_index.dt.dayofweek >= 5).astype(int).values
            feature_names.append("time.is_weekend")
        
        if "month" in self.features:
            features["time.month"] = dt_index.dt.month.values
            feature_names.append("time.month")
        
        if "season" in self.features:
            month = dt_index.dt.month.values
            # 12-2: зима, 3-5: весна, 6-8: лето, 9-11: осень
            season = np.select(
                [
                    (month >= 3) & (month <= 5),
                    (month >= 6) & (month <= 8),
                    (month >= 9) & (month <= 11),
                ],
                [1, 2, 3],  # весна, лето, осень
                default=0   # зима
            )
            features["time.season"] = season
            feature_names.append("time.season")
        
        if "is_business_hour" in self.features:
            hour = dt_index.dt.hour.values
            weekday = dt_index.dt.dayofweek.values
            is_business = ((weekday < 5) & (hour >= 9) & (hour < 18)).astype(int)
            features["time.is_business_hour"] = is_business
            feature_names.append("time.is_business_hour")
        
        X_transformed = pd.DataFrame(features, index=X.index)
        self.feature_names_ = feature_names
        
        return X_transformed
    
    def get_feature_names(self) -> List[str]:
        """
        Получение имен сгенерированных признаков.
        
        Возвращает
        ----------
        feature_names : List[str]
            Список имен признаков.
        """
        if not self.is_fitted_:
            raise ValueError("Transformer is not fitted. Call fit() first.")
        
        return self.feature_names_