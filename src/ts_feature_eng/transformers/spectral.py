# src/ts_feature_eng/transformers/spectral.py

"""
Спектральные трансформеры для анализа временных рядов.

Реализует два метода декомпозиции:
1. DWT (Discrete Wavelet Transform) — вейвлет-анализ для выделения
   многочастотных компонент ряда.
2. STL (Seasonal-Trend decomposition using Loess) — классическая
   декомпозиция на тренд, сезонность и остаток.
"""

from typing import List, Optional, Union

import numpy as np
import pandas as pd
import pywt
from scipy import signal, stats
from statsmodels.tsa.seasonal import STL as StatsmodelsSTL

from ..base import TimeSeriesTransformer


class DWTTransformer(TimeSeriesTransformer):
    """
    Трансформер на основе дискретного вейвлет-преобразования (DWT).
    
    Разлагает временной ряд на коэффициенты аппроксимации (низкочастотные)
    и детализации (высокочастотные) на нескольких уровнях декомпозиции.
    Из коэффициентов извлекаются статистики для генерации признаков.
    
    Параметры
    ----------
    wavelet : str, по умолчанию "db4"
        Тип вейвлета. Поддерживаемые вейвлеты:
        - Daubechies: "db1"-"db20"
        - Symlets: "sym2"-"sym20"
        - Coiflets: "coif1"-"coif5"
        Полный список: https://pywavelets.readthedocs.io/en/latest/ref/wavelets.html  
    max_level : int, опционально
        Максимальный уровень декомпозиции. Если не указан, вычисляется автоматически
        как floor(log2(n_samples)).
    statistics : List[str], по умолчанию ["mean", "std", "energy", "entropy"]
        Список статистик для извлечения из коэффициентов:
        - "mean": среднее значение коэффициентов
        - "std": стандартное отклонение коэффициентов
        - "energy": энергия компоненты (сумма квадратов коэффициентов)
        - "entropy": энтропия Шеннона коэффициентов
        - "max": максимальное абсолютное значение
        - "zero_crossings": количество пересечений нуля
    
    Атрибуты
    ----------
    feature_names_ : List[str]
        Имена сгенерированных признаков в формате:
        "{столбец}.dwt.level_{n}.{тип}.{статистика}"
        где тип ∈ {approx, detail}
    is_fitted_ : bool
        Флаг, указывающий, был ли вызван метод fit().
    n_levels_ : int
        Фактическое количество уровней декомпозиции (может быть меньше max_level).
    
    Примеры
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from ts_feature_eng.transformers.spectral import DWTTransformer
    >>> 
    >>> # Создаем тестовый временной ряд с сезонностью
    >>> dates = pd.date_range("2023-01-01", periods=200, freq="H")
    >>> signal = np.sin(2 * np.pi * np.arange(200) / 24)  # 24-часовая сезонность
    >>> noise = np.random.normal(0, 0.1, 200)
    >>> df = pd.DataFrame({"value": signal + noise}, index=dates)
    >>> 
    >>> # Применяем DWT-трансформер
    >>> transformer = DWTTransformer(wavelet="db4", max_level=3)
    >>> X_transformed = transformer.fit_transform(df)
    >>> 
    >>> print(sorted(X_transformed.columns)[:6])
    ['value.dwt.level_1.detail.energy', 'value.dwt.level_1.detail.entropy',
     'value.dwt.level_1.detail.mean', 'value.dwt.level_1.detail.std',
     'value.dwt.level_1.approx.energy', 'value.dwt.level_1.approx.entropy']
    """
    
    _valid_statistics = ["mean", "std", "energy", "entropy", "max", "zero_crossings"]
    
    def __init__(
        self,
        wavelet: str = "db4",
        max_level: Optional[int] = None,
        statistics: Optional[List[str]] = None,
    ):
        super().__init__()
        self.wavelet = wavelet
        self.max_level = max_level
        self.statistics = statistics or ["mean", "std", "energy", "entropy"]
        
        # Валидация параметров
        self._validate_params()
    
    def _validate_params(self) -> None:
        """Валидация гиперпараметров трансформера."""
        # Проверка существования вейвлета
        if self.wavelet not in pywt.wavelist(kind="discrete"):
            raise ValueError(
                f"Wavelet '{self.wavelet}' is not supported. "
                f"Available wavelets: {pywt.wavelist(kind='discrete')[:10]}..."
            )
        
        # Проверка статистик
        invalid_stats = set(self.statistics) - set(self._valid_statistics)
        if invalid_stats:
            raise ValueError(
                f"Invalid statistics: {invalid_stats}. "
                f"Valid options: {self._valid_statistics}"
            )
    
    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Optional[Union[pd.Series, np.ndarray]] = None) -> "DWTTransformer":
        """
        Обучение трансформера (определение количества уровней декомпозиции).
        
        Параметры
        ----------
        X : pd.DataFrame или np.ndarray
            Входные данные временного ряда.
        y : pd.Series или np.ndarray, опционально
            Целевая переменная (игнорируется).
        
        Возвращает
        ----------
        self : DWTTransformer
            Обученный трансформер.
        """
        X = self._validate_input(X)
        
        # Определение максимального уровня декомпозиции
        n_samples = len(X)
        if self.max_level is None:
            self.n_levels_ = pywt.dwt_max_level(n_samples, self.wavelet)
        else:
            max_possible = pywt.dwt_max_level(n_samples, self.wavelet)
            self.n_levels_ = min(self.max_level, max_possible)
        
        if self.n_levels_ < 1:
            raise ValueError(
                f"Insufficient data length ({n_samples}) for DWT decomposition "
                f"with wavelet '{self.wavelet}'. Minimum required: 2 samples."
            )
        
        # Создание имен признаков
        self.feature_names_ = []
        for col in X.columns:
            for level in range(1, self.n_levels_ + 1):
                for stat in self.statistics:
                    # Признаки для детализации (high-frequency)
                    self.feature_names_.append(f"{col}.dwt.level_{level}.detail.{stat}")
                    # Признаки для аппроксимации (low-frequency) - только на последнем уровне
                    if level == self.n_levels_:
                        self.feature_names_.append(f"{col}.dwt.level_{level}.approx.{stat}")
        
        self.is_fitted_ = True
        return self
    
    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """
        Применение вейвлет-преобразования и генерация признаков.
        
        Параметры
        ----------
        X : pd.DataFrame или np.ndarray
            Входные данные временного ряда.
        
        Возвращает
        ----------
        X_transformed : pd.DataFrame
            DataFrame с сгенерированными признаками.
        """
        if not self.is_fitted_:
            raise ValueError("Transformer is not fitted. Call fit() first.")
        
        X = self._validate_input(X)
        features = {}
        
        # Обработка каждого столбца
        for col in X.columns:
            series = X[col].values.astype(np.float64)
            
            # Применение многомерной вейвлет-декомпозиции
            coeffs = pywt.wavedec(series, self.wavelet, level=self.n_levels_)
            
            # coeffs[0] — коэффициенты аппроксимации самого высокого уровня
            # coeffs[1:] — коэффициенты детализации от высокого к низкому уровню
            for level in range(1, self.n_levels_ + 1):
                # Коэффициенты детализации для текущего уровня
                detail_coeffs = coeffs[-level]
                
                # Статистики для детализации
                for stat in self.statistics:
                    feat_name = f"{col}.dwt.level_{level}.detail.{stat}"
                    features[feat_name] = self._compute_wavelet_statistic(detail_coeffs, stat)
                
                # Статистики для аппроксимации (только на самом высоком уровне)
                if level == self.n_levels_:
                    approx_coeffs = coeffs[0]
                    for stat in self.statistics:
                        feat_name = f"{col}.dwt.level_{level}.approx.{stat}"
                        features[feat_name] = self._compute_wavelet_statistic(approx_coeffs, stat)
        
        X_transformed = pd.DataFrame(features, index=X.index)
        return X_transformed
    
    def _compute_wavelet_statistic(self, coeffs: np.ndarray, statistic: str) -> float:
        """
        Вычисление статистики из вейвлет-коэффициентов.
        
        Параметры
        ----------
        coeffs : np.ndarray
            Массив вейвлет-коэффициентов.
        statistic : str
            Тип статистики для вычисления.
        
        Возвращает
        ----------
        value : float
            Результат вычисления статистики.
        """
        if len(coeffs) == 0:
            return np.nan
        
        if statistic == "mean":
            return np.mean(coeffs)
        
        elif statistic == "std":
            return np.std(coeffs, ddof=1) if len(coeffs) > 1 else 0.0
        
        elif statistic == "energy":
            # Энергия = сумма квадратов коэффициентов
            return np.sum(coeffs ** 2)
        
        elif statistic == "entropy":
            # Энтропия Шеннона нормированных коэффициентов
            energy = np.sum(coeffs ** 2)
            if energy == 0:
                return 0.0
            
            normalized = (coeffs ** 2) / energy
            # Игнорируем нулевые значения для избежания log(0)
            positive = normalized[normalized > 0]
            if len(positive) == 0:
                return 0.0
            
            return -np.sum(positive * np.log2(positive + 1e-10))
        
        elif statistic == "max":
            return np.max(np.abs(coeffs))
        
        elif statistic == "zero_crossings":
            # Количество пересечений нуля (изменений знака)
            signs = np.sign(coeffs)
            # Удаляем нули для корректного подсчета
            signs = signs[signs != 0]
            if len(signs) < 2:
                return 0
            
            crossings = np.sum(signs[1:] != signs[:-1])
            return crossings
        
        else:
            raise ValueError(f"Unknown statistic: {statistic}")
    
    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """
        Получение имен сгенерированных признаков.
        
        Параметры
        ----------
        input_features : array-like, опционально
            Игнорируется (требуется для совместимости с интерфейсом sklearn).
        
        Возвращает
        ----------
        feature_names : np.ndarray
            Массив имен сгенерированных признаков.
        """
        if not self.is_fitted_:
            raise ValueError("DWTTransformer is not fitted. Call fit() first.")
        
        return np.array(self.feature_names_)


class STLTransformer(TimeSeriesTransformer):
    """
    Трансформер на основе сезонно-трендовой декомпозиции (STL).
    
    Разлагает временной ряд на три компоненты:
    - Тренд (долгосрочная динамика)
    - Сезонность (периодические колебания)
    - Остаток (шум и аномалии)
    
    Из каждой компоненты извлекаются статистики для генерации признаков.
    
    Параметры
    ----------
    period : int, опционально
        Период сезонности в наблюдениях. Если не указан, определяется автоматически
        через анализ автокорреляции.
    seasonal : int, по умолчанию 7
        Длина окна для сглаживания сезонной компоненты (должна быть нечетной).
    trend : int, опционально
        Длина окна для сглаживания трендовой компоненты (должна быть нечетной).
        Если не указано, вычисляется как 1.5 * период + 1 (округленное до нечетного).
    low_pass : int, опционально
        Длина окна для фильтра низких частот.
    statistics : List[str], по умолчанию ["mean", "std", "min", "max", "skewness", "kurtosis"]
        Список статистик для извлечения из каждой компоненты:
        - "mean": среднее значение
        - "std": стандартное отклонение
        - "min": минимум
        - "max": максимум
        - "skewness": коэффициент асимметрии
        - "kurtosis": коэффициент эксцесса
        - "range": размах (max - min)
        - "energy": энергия компоненты
    
    Атрибуты
    ----------
    feature_names_ : List[str]
        Имена сгенерированных признаков в формате:
        "{столбец}.stl.{компонента}.{статистика}"
        где компонента ∈ {trend, seasonal, resid}
    is_fitted_ : bool
        Флаг, указывающий, был ли вызван метод fit().
    period_ : int
        Фактически используемый период сезонности.
    
    Примеры
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from ts_feature_eng.transformers.spectral import STLTransformer
    >>> 
    >>> # Создаем тестовый временной ряд с 24-часовой сезонностью
    >>> dates = pd.date_range("2023-01-01", periods=200, freq="H")
    >>> trend = np.linspace(0, 10, 200)
    >>> seasonal = 5 * np.sin(2 * np.pi * np.arange(200) / 24)
    >>> noise = np.random.normal(0, 0.5, 200)
    >>> df = pd.DataFrame({"value": trend + seasonal + noise}, index=dates)
    >>> 
    >>> # Применяем STL-трансформер с автоматическим определением периода
    >>> transformer = STLTransformer(period=24)
    >>> X_transformed = transformer.fit_transform(df)
    >>> 
    >>> print(sorted(X_transformed.columns)[:6])
    ['value.stl.resid.energy', 'value.stl.resid.kurtosis', 'value.stl.resid.max',
     'value.stl.resid.mean', 'value.stl.resid.min', 'value.stl.resid.range']
    """
    
    _valid_statistics = ["mean", "std", "min", "max", "skewness", "kurtosis", "range", "energy"]
    _valid_components = ["trend", "seasonal", "resid"]
    
    def __init__(
        self,
        period: Optional[int] = None,
        seasonal: int = 7,
        trend: Optional[int] = None,
        low_pass: Optional[int] = None,
        statistics: Optional[List[str]] = None,
    ):
        super().__init__()
        self.period = period
        self.seasonal = seasonal
        self.trend = trend
        self.low_pass = low_pass
        self.statistics = statistics or ["mean", "std", "min", "max", "skewness", "kurtosis"]
        
        # Валидация параметров
        self._validate_params()
    
    def _validate_params(self) -> None:
        """Валидация гиперпараметров трансформера."""
        if self.seasonal % 2 == 0:
            raise ValueError(f"seasonal parameter must be odd, got {self.seasonal}")
        
        if self.trend is not None and self.trend % 2 == 0:
            raise ValueError(f"trend parameter must be odd, got {self.trend}")
        
        if self.low_pass is not None and self.low_pass % 2 == 0:
            raise ValueError(f"low_pass parameter must be odd, got {self.low_pass}")
        
        invalid_stats = set(self.statistics) - set(self._valid_statistics)
        if invalid_stats:
            raise ValueError(
                f"Invalid statistics: {invalid_stats}. "
                f"Valid options: {self._valid_statistics}"
            )
    
    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Optional[Union[pd.Series, np.ndarray]] = None) -> "STLTransformer":
        """
        Обучение трансформера (определение периода сезонности при необходимости).
        
        Параметры
        ----------
        X : pd.DataFrame или np.ndarray
            Входные данные временного ряда.
        y : pd.Series или np.ndarray, опционально
            Целевая переменная (игнорируется).
        
        Возвращает
        ----------
        self : STLTransformer
            Обученный трансформер.
        """
        X = self._validate_input(X)
        
        # Автоматическое определение периода через автокорреляцию
        if self.period is None:
            self.period_ = self._estimate_period(X)
        else:
            self.period_ = self.period
        
        # Автоматическое определение параметра тренда, если не задан
        if self.trend is None:
            self.trend_ = int(1.5 * self.period_ + 1)
            if self.trend_ % 2 == 0:
                self.trend_ += 1  # Делаем нечетным
        else:
            self.trend_ = self.trend
        
        # Создание имен признаков
        self.feature_names_ = []
        for col in X.columns:
            for component in ["trend", "seasonal", "resid"]:
                for stat in self.statistics:
                    self.feature_names_.append(f"{col}.stl.{component}.{stat}")
        
        self.is_fitted_ = True
        return self
    
    def _estimate_period(self, X: pd.DataFrame) -> int:
        """
        Автоматическая оценка периода сезонности через автокорреляцию.
        
        Параметры
        ----------
        X : pd.DataFrame
            Входные данные.
        
        Возвращает
        ----------
        period : int
            Оцененный период сезонности.
        """
        # Используем первый столбец для оценки периода
        series = X.iloc[:, 0].dropna().values
        
        if len(series) < 24:  # Минимальная длина для оценки
            return 24  # Значение по умолчанию
        
        # Вычисляем автокорреляцию до лага 100 или половины длины ряда
        max_lag = min(100, len(series) // 2)
        acf = np.correlate(series - np.mean(series), series - np.mean(series), mode="full")
        acf = acf[acf.size // 2:][:max_lag + 1]
        acf = acf / acf[0]  # Нормализация
        
        # Находим локальные максимумы автокорреляции
        peaks, _ = signal.find_peaks(acf[1:], height=0.3, distance=5)
        peaks = peaks + 1  # Коррекция индекса
        
        if len(peaks) == 0:
            return 24  # Значение по умолчанию
        
        # Выбираем первый значимый пик как период
        period = int(peaks[0])
        return max(2, period)  # Минимальный период = 2
    
    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """
        Применение STL-декомпозиции и генерация признаков.
        
        Параметры
        ----------
        X : pd.DataFrame или np.ndarray
            Входные данные временного ряда.
        
        Возвращает
        ----------
        X_transformed : pd.DataFrame
            DataFrame с сгенерированными признаками.
        """
        if not self.is_fitted_:
            raise ValueError("Transformer is not fitted. Call fit() first.")
        
        X = self._validate_input(X)
        features = {}
        
        # Обработка каждого столбца
        for col in X.columns:
            series = X[col]
            
            # Пропуски могут нарушить работу STL — заполняем линейной интерполяцией
            if series.isna().any():
                series = series.interpolate(method="linear", limit_direction="both")
            
            # Применение декомпозиции STL
            try:
                stl = StatsmodelsSTL(
                    series,
                    period=self.period_,
                    seasonal=self.seasonal,
                    trend=self.trend_,
                    low_pass=self.low_pass,
                    seasonal_deg=1,
                    trend_deg=1,
                    low_pass_deg=1,
                    robust=True,
                )
                result = stl.fit()
            except Exception as e:
                raise ValueError(
                    f"STL decomposition failed for column '{col}' with period={self.period_}. "
                    f"Error: {str(e)}"
                )
            
            # Извлечение компонент
            components = {
                "trend": result.trend,
                "seasonal": result.seasonal,
                "resid": result.resid,
            }
            
            # Вычисление статистик для каждой компоненты
            for component_name, component in components.items():
                for stat in self.statistics:
                    feat_name = f"{col}.stl.{component_name}.{stat}"
                    features[feat_name] = self._compute_component_statistic(component, stat)
        
        X_transformed = pd.DataFrame(features, index=X.index)
        return X_transformed
    
    def _compute_component_statistic(self, component: pd.Series, statistic: str) -> float:
        """
        Вычисление статистики из компоненты декомпозиции.
        
        Параметры
        ----------
        component : pd.Series
            Временной ряд компоненты (тренд, сезонность или остаток).
        statistic : str
            Тип статистики для вычисления.
        
        Возвращает
        ----------
        value : float
            Результат вычисления статистики.
        """
        # Удаляем пропуски перед вычислением
        values = component.dropna().values
        
        if len(values) == 0:
            return np.nan
        
        if statistic == "mean":
            return np.mean(values)
        
        elif statistic == "std":
            return np.std(values, ddof=1) if len(values) > 1 else 0.0
        
        elif statistic == "min":
            return np.min(values)
        
        elif statistic == "max":
            return np.max(values)
        
        elif statistic == "skewness":
            return stats.skew(values, nan_policy="omit")
        
        elif statistic == "kurtosis":
            return stats.kurtosis(values, nan_policy="omit")
        
        elif statistic == "range":
            return np.max(values) - np.min(values)
        
        elif statistic == "energy":
            return np.sum(values ** 2)
        
        else:
            raise ValueError(f"Unknown statistic: {statistic}")
    
    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """
        Получение имен сгенерированных признаков.
        
        Параметры
        ----------
        input_features : array-like, опционально
            Игнорируется (требуется для совместимости с интерфейсом sklearn).
        
        Возвращает
        ----------
        feature_names : np.ndarray
            Массив имен сгенерированных признаков.
        """
        if not self.is_fitted_:
            raise ValueError("STLTransformer is not fitted. Call fit() first.")
        
        return np.array(self.feature_names_)
    
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
            "period": self.period,
            "seasonal": self.seasonal,
            "trend": self.trend,
            "low_pass": self.low_pass,
            "statistics": self.statistics,
        }
    
    def set_params(self, **params) -> "STLTransformer":
        """
        Установка параметров трансформера (совместимость с sklearn).
        
        Параметры
        ----------
        **params : dict
            Параметры для установки.
        
        Возвращает
        ----------
        self : STLTransformer
            Трансформер с обновленными параметрами.
        """
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter {key} for STLTransformer")
        
        self._validate_params()
        return self