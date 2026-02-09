# src/ts_feature_eng/meta_features.py
"""
Извлечение мета-признаков временных рядов для адаптивного выбора методов инженерии признаков.
Мета-признаки описывают глобальные свойства временного ряда и используются для:
1. Инициализации байесовской оптимизации (приорные знания о структуре ряда)
2. Адаптивного выбора методов преобразования (оконные, спектральные, временные кодирования)
3. Оценки "прогнозируемости" ряда перед обучением модели
Категории мета-признаков [12][22]:
- Простые: длина, частота, пропуски
- Статистические: нестационарность, линейность, дисперсия, асимметрия
- Информационно-теоретические: энтропия, сложность, нелинейность
- Спектральные: доминирующие частоты, автокорреляция, результаты FFT
- Ландмарковые: ошибки простых моделей (отражающие прогнозируемость)
"""
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
from scipy import stats
from scipy.fft import fft
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import adfuller, kpss
from .base import TimeSeriesTransformer


class MetaFeatureExtractor(TimeSeriesTransformer):
    """
    Извлекатель мета-признаков временного ряда.
    Генерирует 30+ мета-признаков из 5 категорий для характеристики структуры ряда.
    Мета-признаки используются для адаптивного выбора методов инженерии признаков
    через байесовскую оптимизацию.
    
    Параметры
    ----------
    categories : List[str], по умолчанию все категории
        Список категорий мета-признаков для извлечения:
        - "simple": простые характеристики (длина, пропуски, частота)
        - "statistical": статистические свойства (нестационарность, линейность)
        - "information_theoretic": информационные метрики (энтропия, сложность)
        - "spectral": спектральные характеристики (частоты, автокорреляция)
        - "landmarking": ошибки простых моделей (прогнозируемость)
        - "all": все категории (специальное значение)
    
    fill_method : str, по умолчанию "linear"
        Метод заполнения пропусков перед вычислением мета-признаков:
        - "linear": линейная интерполяция
        - "ffill": заполнение предыдущим значением
        - "none": без заполнения (требует отсутствия пропусков)
    
    Атрибуты
    ----------
    meta_features_ : Dict[str, float]
        Словарь извлеченных мета-признаков в формате {имя: значение}.
    feature_names_ : List[str]
        Имена всех извлеченных мета-признаков.
    is_fitted_ : bool
        Флаг, указывающий, был ли вызван метод fit().
    
    Примеры
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from ts_feature_eng.meta_features import MetaFeatureExtractor
    >>>
    >>> # Создаем тестовый временной ряд
    >>> dates = pd.date_range("2023-01-01", periods=1000, freq="h")
    >>> trend = np.linspace(0, 10, 1000)
    >>> seasonal = 5 * np.sin(2 * np.pi * np.arange(1000) / 24)
    >>> noise = np.random.normal(0, 1, 1000)
    >>> df = pd.DataFrame({"value": trend + seasonal + noise}, index=dates)
    >>>
    >>> # Извлекаем мета-признаки
    >>> extractor = MetaFeatureExtractor(categories=["simple", "statistical", "spectral"])
    >>> meta_features = extractor.fit_transform(df)
    >>>
    >>> print(meta_features.shape)
    (1, 18)  # 18 мета-признаков из 3 категорий
    >>> print(sorted(meta_features.columns)[:5])
    ['acf_1', 'acf_24', 'length', 'linearity', 'missing_ratio']
    """
    _valid_categories = [
        "simple",
        "statistical",
        "information_theoretic",
        "spectral",
        "landmarking"
    ]
    _valid_fill_methods = ["linear", "ffill", "none"]

    def __init__(
        self,
        categories: Optional[List[str]] = None,
        fill_method: str = "linear",
    ):
        super().__init__()
        # Обработка специального значения "all"
        if categories is None or (isinstance(categories, list) and "all" in categories):
            self.categories = self._valid_categories.copy()
        else:
            self.categories = categories if isinstance(categories, list) else [categories]
        self.fill_method = fill_method
        # Валидация параметров
        self._validate_params()

    def _validate_params(self) -> None:
        """Валидация гиперпараметров извлекателя."""
        # Фильтруем специальное значение "all" перед проверкой
        categories_to_check = [c for c in self.categories if c != "all"]
        invalid_categories = set(categories_to_check) - set(self._valid_categories)
        if invalid_categories:
            raise ValueError(
                f"Invalid categories: {invalid_categories}. "
                f"Valid options: {self._valid_categories + ['all']}"
            )
        if self.fill_method not in self._valid_fill_methods:
            raise ValueError(
                f"Invalid fill_method: {self.fill_method}. "
                f"Valid options: {self._valid_fill_methods}"
            )

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> "MetaFeatureExtractor":
        """
        Извлечение мета-признаков из временного ряда.
        
        Параметры
        ----------
        X : pd.DataFrame или np.ndarray
            Входные данные временного ряда (одномерный или многомерный).
        y : pd.Series или np.ndarray, опционально
            Целевая переменная (используется только для ландмарковых мета-признаков).
        
        Возвращает
        ----------
        self : MetaFeatureExtractor
            Обученный извлекатель с сохраненными мета-признаками.
        """
        # Сохраняем исходные данные ДО валидации для вычисления missing_ratio
        if isinstance(X, pd.DataFrame):
            X_original = X.copy()
        else:
            X_original = pd.DataFrame(X) if isinstance(X, np.ndarray) else pd.DataFrame(X.reshape(-1, 1))
        
        X_validated = self._validate_input(X)
        
        # Обработка пропусков (после сохранения исходных данных для missing_ratio)
        X_clean = self._handle_missing_values(X_validated, X_original)
        
        # Извлечение мета-признаков по категориям
        self.meta_features_ = {}
        if "simple" in self.categories:
            self.meta_features_.update(self._extract_simple_features(X_clean, X_original))
        if "statistical" in self.categories:
            self.meta_features_.update(self._extract_statistical_features(X_clean))
        if "information_theoretic" in self.categories:
            self.meta_features_.update(self._extract_information_theoretic_features(X_clean))
        if "spectral" in self.categories:
            self.meta_features_.update(self._extract_spectral_features(X_clean))
        if "landmarking" in self.categories and y is not None:
            # Для ландмарковых мета-признаков требуется целевая переменная
            if isinstance(y, pd.Series):
                y_clean = y.reindex(X_clean.index).dropna()
                X_clean = X_clean.loc[y_clean.index]
            else:
                y_clean = y
            self.meta_features_.update(
                self._extract_landmarking_features(X_clean, y_clean)
            )
        elif "landmarking" in self.categories and y is None:
            # Предупреждение: ландмарковые мета-признаки требуют целевую переменную
            import warnings
            warnings.warn(
                "Landmarking meta-features require target variable 'y'. "
                "Skipping landmarking features.",
                UserWarning
            )
        
        # Сохранение имен признаков
        self.feature_names_ = sorted(self.meta_features_.keys())
        self.is_fitted_ = True
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """
        Возвращает извлеченные мета-признаки как однорядный DataFrame.
        
        Примечание: мета-признаки вычисляются один раз при вызове fit().
        Transform всегда возвращает один набор мета-признаков (1 строка),
        независимо от количества строк во входных данных.
        Это соответствует семантике мета-признаков как глобальных характеристик ряда.
        
        Параметры
        ----------
        X : pd.DataFrame или np.ndarray
            Входные данные (используются только для проверки типа).
        
        Возвращает
        ----------
        meta_df : pd.DataFrame
            DataFrame с одной строкой и столбцами-мета-признаками.
        """
        if not self.is_fitted_:
            raise ValueError("MetaFeatureExtractor is not fitted. Call fit() first.")
        
        # Всегда возвращаем 1 строку - глобальные мета-признаки всего ряда
        return pd.DataFrame([self.meta_features_], columns=self.feature_names_, index=[0])

    def get_feature_names(self) -> List[str]:
        """
        Получение имен извлеченных мета-признаков.
        
        Возвращает
        ----------
        feature_names : List[str]
            Список имен мета-признаков.
        """
        if not self.is_fitted_:
            raise ValueError("MetaFeatureExtractor is not fitted. Call fit() first.")
        return self.feature_names_

    def _handle_missing_values(self, X: pd.DataFrame, X_original: pd.DataFrame) -> pd.DataFrame:
        """
        Обработка пропусков в данных перед извлечением мета-признаков.
        
        Параметры
        ----------
        X : pd.DataFrame
            Валидированные данные.
        X_original : pd.DataFrame
            Исходные данные (для вычисления missing_ratio до заполнения).
        
        Возвращает
        ----------
        X_clean : pd.DataFrame
            Данные с обработанными пропусками.
        """
        if self.fill_method == "none":
            if X.isna().any().any():
                raise ValueError(
                    "Input contains missing values but fill_method='none'. "
                    "Either fill missing values manually or use fill_method='linear' or 'ffill'."
                )
            return X
        elif self.fill_method == "linear":
            # Линейная интерполяция с заполнением крайних значений
            return X.interpolate(method="linear", limit_direction="both").bfill().ffill()
        elif self.fill_method == "ffill":
            return X.fillna(method="ffill").fillna(method="bfill")
        else:
            raise ValueError(f"Unknown fill_method: {self.fill_method}")

    def _extract_simple_features(self, X: pd.DataFrame, X_original: pd.DataFrame) -> Dict[str, float]:
        """
        Извлечение простых мета-признаков.
        
        Параметры
        ----------
        X : pd.DataFrame
            Данные после обработки пропусков.
        X_original : pd.DataFrame
            Исходные данные (для вычисления доли пропусков ДО заполнения).
        
        Возвращает
        ----------
        features : Dict[str, float]
            Словарь мета-признаков:
            - length: длина ряда
            - missing_ratio: доля пропусков в исходных данных (до интерполяции)
            - num_features: количество признаков (для многомерных рядов)
            - freq_hourly: частота дискретизации в часах (если временной индекс)
        """
        features = {}
        # Длина ряда
        features["length"] = float(len(X))
        # Количество признаков (столбцов)
        features["num_features"] = float(X.shape[1])
        # Доля пропусков в ИСХОДНЫХ данных (до интерполяции) - критически важно!
        missing_ratio = X_original.isna().sum().sum() / (X_original.shape[0] * X_original.shape[1])
        features["missing_ratio"] = float(missing_ratio)
        # Частота дискретизации (если временной индекс)
        if isinstance(X.index, pd.DatetimeIndex):
            # Вычисляем медианный интервал между наблюдениями в часах
            if len(X) > 1:
                freq_ns = np.median(np.diff(X.index.astype(np.int64)))
                freq_hours = freq_ns / (3600 * 1e9)
                features["freq_hourly"] = float(freq_hours)
            else:
                features["freq_hourly"] = np.nan
        return features

    def _extract_statistical_features(self, X: pd.DataFrame) -> Dict[str, float]:
        """
        Извлечение статистических мета-признаков.
        
        Возвращает
        ----------
        features : Dict[str, float]
            Словарь мета-признаков:
            - stationarity_adf: p-value теста Дики-Фуллера (чем меньше, тем стационарнее)
            - stationarity_kpss: p-value теста KPSS (чем больше, тем стационарнее)
            - linearity: R² линейной регрессии по времени
            - variance: дисперсия ряда
            - skewness: асимметрия распределения
            - kurtosis: эксцесс распределения
            - cv: коэффициент вариации (std / mean)
        """
        features = {}
        # Работаем с первым столбцом (предполагаем одномерный ряд для мета-признаков)
        series = X.iloc[:, 0].dropna().values
        if len(series) < 10:
            # Недостаточно данных для надежных статистик
            return {
                "stationarity_adf": np.nan,
                "stationarity_kpss": np.nan,
                "linearity": np.nan,
                "variance": np.nan,
                "skewness": np.nan,
                "kurtosis": np.nan,
                "cv": np.nan,
            }
        # Тест стационарности Дики-Фуллера
        try:
            adf_result = adfuller(series, maxlag=min(10, len(series) // 10))
            features["stationarity_adf"] = float(adf_result[1])  # p-value
        except Exception:
            features["stationarity_adf"] = np.nan
        # Тест KPSS на стационарность
        try:
            kpss_result = kpss(series, nlags="auto")
            features["stationarity_kpss"] = float(kpss_result[1])  # p-value
        except Exception:
            features["stationarity_kpss"] = np.nan
        # Линейность: R² линейной регрессии по времени
        X_time = np.arange(len(series)).reshape(-1, 1)
        model = LinearRegression()
        model.fit(X_time, series)
        features["linearity"] = float(model.score(X_time, series))
        # Базовые статистики
        features["variance"] = float(np.var(series, ddof=1))
        features["skewness"] = float(stats.skew(series))
        features["kurtosis"] = float(stats.kurtosis(series))
        features["cv"] = float(np.std(series, ddof=1) / (np.mean(series) + 1e-10))  # Коэффициент вариации
        return features

    def _extract_information_theoretic_features(self, X: pd.DataFrame) -> Dict[str, float]:
        """
        Извлечение информационно-теоретических мета-признаков.
        
        Возвращает
        ----------
        features : Dict[str, float]
            Словарь мета-признаков:
            - entropy: энтропия Шеннона нормированного ряда
            - permutation_entropy: перестановочная энтропия (сложность)
            - hurst_exponent: показатель Херста (долгосрочная зависимость)
            - nonlinearity: мера нелинейности через остатки линейной модели
        """
        features = {}
        series = X.iloc[:, 0].dropna().values
        if len(series) < 20:
            return {
                "entropy": np.nan,
                "permutation_entropy": np.nan,
                "hurst_exponent": np.nan,
                "nonlinearity": np.nan,
            }
        # Энтропия Шеннона
        # Нормализуем ряд и дискретизируем в 10 бинов
        if series.max() - series.min() > 1e-10:
            normalized = (series - series.min()) / (series.max() - series.min())
        else:
            normalized = np.zeros_like(series)
        hist, _ = np.histogram(normalized, bins=10, range=(0, 1))
        prob = hist / (hist.sum() + 1e-10)
        prob = prob[prob > 0]
        features["entropy"] = float(-np.sum(prob * np.log2(prob + 1e-10))) if len(prob) > 0 else np.nan
        # Перестановочная энтропия (мера сложности/хаотичности)
        try:
            pe = self._permutation_entropy(series, order=3, delay=1)
            features["permutation_entropy"] = float(pe) if not np.isnan(pe) else np.nan
        except Exception:
            features["permutation_entropy"] = np.nan
        # Показатель Херста (долгосрочная зависимость)
        try:
            hurst = self._hurst_exponent(series)
            features["hurst_exponent"] = float(hurst) if not np.isnan(hurst) else np.nan
        except Exception:
            features["hurst_exponent"] = np.nan
        # Нелинейность: доля дисперсии, не объясненная линейной моделью
        X_time = np.arange(len(series)).reshape(-1, 1)
        model = LinearRegression()
        model.fit(X_time, series)
        residuals = series - model.predict(X_time)
        features["nonlinearity"] = float(np.var(residuals) / (np.var(series) + 1e-10))
        return features

    def _permutation_entropy(self, x: np.ndarray, order: int = 3, delay: int = 1) -> float:
        """
        Вычисление перестановочной энтропии временного ряда.
        
        Параметры
        ----------
        x : np.ndarray
            Временной ряд.
        order : int
            Порядок перестановки (количество точек в паттерне).
        delay : int
            Задержка между точками в паттерне.
        
        Возвращает
        ----------
        pe : float
            Значение перестановочной энтропии (нормированное от 0 до 1).
        """
        x = np.array(x)
        # Минимальная длина для вычисления
        min_length = (order - 1) * delay + 1
        if len(x) < min_length:
            return np.nan
        
        # Формируем паттерны с обработкой дубликатов (стабильная сортировка)
        patterns = []
        for i in range(len(x) - min_length + 1):
            segment = x[i:i + min_length:delay]
            # Используем стабильную сортировку для обработки дубликатов
            sorted_indices = np.argsort(segment, kind='mergesort')
            pattern = tuple(sorted_indices)
            patterns.append(pattern)
        
        if not patterns:
            return np.nan
        
        # Подсчитываем частоты паттернов
        from collections import Counter
        pattern_counts = Counter(patterns)
        n_patterns = len(patterns)
        
        # Вычисляем вероятности
        probs = np.array(list(pattern_counts.values())) / n_patterns
        probs = probs[probs > 0]  # Удаляем нулевые вероятности
        
        if len(probs) == 0:
            return np.nan
        
        # Вычисляем энтропию
        pe = -np.sum(probs * np.log2(probs + 1e-10))
        max_pe = np.log2(np.math.factorial(order))
        
        return pe / max_pe if max_pe > 0 else np.nan

    def _hurst_exponent(self, ts: np.ndarray, max_lag: int = 20) -> float:
        """
        Оценка показателя Херста методом анализа изменчивости (R/S анализ).
        
        Параметры
        ----------
        ts : np.ndarray
            Временной ряд.
        max_lag : int
            Максимальный лаг для анализа.
        
        Возвращает
        ----------
        hurst : float
            Оценка показателя Херста:
            - H < 0.5: антиперсистентность (среднее-реверсионный процесс)
            - H ≈ 0.5: случайное блуждание
            - H > 0.5: персистентность (трендовый процесс)
        """
        # Убираем тренд
        ts = ts - np.mean(ts)
        # Вычисляем кумулятивную сумму (процесс случайного блуждания)
        walk = np.cumsum(ts)
        # Вычисляем размах (R) и стандартное отклонение (S) для разных лагов
        lags = range(2, min(max_lag, len(ts) // 2) + 1)
        rs_values = []
        for lag in lags:
            # Разбиваем ряд на сегменты длиной lag
            segments = len(walk) // lag
            if segments < 2:
                continue
            rs_lag = []
            for i in range(segments):
                segment = walk[i * lag:(i + 1) * lag]
                r = np.max(segment) - np.min(segment)
                s = np.std(ts[i * lag:(i + 1) * lag]) + 1e-10
                if s > 0:
                    rs_lag.append(r / s)
            if rs_lag:
                rs_values.append(np.mean(rs_lag))
        
        if len(rs_values) < 2:
            return 0.5
        
        # Логарифмическая регрессия для оценки показателя Херста
        log_lags = np.log(list(lags)[:len(rs_values)])
        log_rs = np.log(rs_values)
        # Удаляем бесконечности и NaN
        valid = np.isfinite(log_lags) & np.isfinite(log_rs)
        if np.sum(valid) < 2:
            return 0.5
        try:
            slope, _ = np.polyfit(log_lags[valid], log_rs[valid], 1)
            return float(slope) if np.isfinite(slope) else 0.5
        except Exception:
            return 0.5

    def _extract_spectral_features(self, X: pd.DataFrame) -> Dict[str, float]:
        """
        Извлечение спектральных мета-признаков.
        
        Возвращает
        ----------
        features : Dict[str, float]
            Словарь мета-признаков:
            - dominant_freq: доминирующая частота из спектра мощности
            - spectral_entropy: энтропия спектра мощности
            - acf_1: автокорреляция лага 1
            - acf_24: автокорреляция лага 24 (суточная сезонность)
            - pacf_1: частичная автокорреляция лага 1
            - peak_frequency_ratio: отношение мощности пика к общей мощности
        """
        features = {}
        series = X.iloc[:, 0].dropna().values
        if len(series) < 50:
            return {
                "dominant_freq": np.nan,
                "spectral_entropy": np.nan,
                "acf_1": np.nan,
                "acf_24": np.nan,
                "pacf_1": np.nan,
                "peak_frequency_ratio": np.nan,
            }
        # Спектр мощности через FFT
        n = len(series)
        fft_vals = np.abs(fft(series - np.mean(series)))[:n // 2]
        freqs = np.fft.fftfreq(n, d=1)[:n // 2]
        # Доминирующая частота (исключая нулевую частоту)
        if len(fft_vals) > 1 and np.sum(fft_vals[1:]) > 0:
            dominant_idx = np.argmax(fft_vals[1:]) + 1
            features["dominant_freq"] = float(freqs[dominant_idx])
            # Энтропия спектра мощности
            power = fft_vals ** 2
            power_norm = power / (power.sum() + 1e-10)
            power_norm = power_norm[power_norm > 1e-10]
            if len(power_norm) > 0:
                features["spectral_entropy"] = float(-np.sum(power_norm * np.log2(power_norm + 1e-10)))
            else:
                features["spectral_entropy"] = np.nan
            # Отношение мощности пика к общей мощности
            features["peak_frequency_ratio"] = float(power[dominant_idx] / (power.sum() + 1e-10))
        else:
            features["dominant_freq"] = np.nan
            features["spectral_entropy"] = np.nan
            features["peak_frequency_ratio"] = np.nan
        # Автокорреляция
        acf_vals = self._autocorr(series, nlags=48)
        features["acf_1"] = float(acf_vals[1]) if len(acf_vals) > 1 else np.nan
        features["acf_24"] = float(acf_vals[24]) if len(acf_vals) > 24 else np.nan
        # Частичная автокорреляция (упрощенная оценка)
        try:
            # PACF(1) ≈ ACF(1)
            pacf_1 = acf_vals[1] if len(acf_vals) > 1 else 0
            features["pacf_1"] = float(pacf_1)
        except Exception:
            features["pacf_1"] = np.nan
        return features

    def _autocorr(self, x: np.ndarray, nlags: int = 40) -> np.ndarray:
        """
        Вычисление автокорреляционной функции.
        
        Параметры
        ----------
        x : np.ndarray
            Временной ряд.
        nlags : int
            Количество лагов для вычисления.
        
        Возвращает
        ----------
        acf : np.ndarray
            Массив значений автокорреляции от лага 0 до nlags.
        """
        x = np.asarray(x)
        n = len(x)
        if n == 0:
            return np.zeros(nlags + 1)
        x = x - np.mean(x)
        var = np.var(x)
        if var == 0:
            return np.zeros(nlags + 1)
        acf = np.correlate(x, x, mode="full")[-n:]
        acf = acf / (var * n)
        return acf[:nlags + 1]

    def _extract_landmarking_features(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray]
    ) -> Dict[str, float]:
        """
        Извлечение ландмарковых мета-признаков (ошибки простых моделей).
        
        Возвращает
        ----------
        features : Dict[str, float]
            Словарь мета-признаков:
            - ar_error: MAE модели авторегрессии (AR)
            - ridge_error: MAE модели гребневой регрессии
            - naive_error: MAE наивного прогноза (последнее значение)
            - seasonal_naive_error: MAE сезонного наивного прогноза (лаг 24)
            - error_ratio: отношение ошибки сложной модели к наивной
        """
        features = {}
        # Преобразуем в массивы numpy
        if isinstance(X, pd.DataFrame):
            X_vals = X.values
        else:
            X_vals = X
        if isinstance(y, pd.Series):
            y_vals = y.values
        else:
            y_vals = y
        # Требуем минимум 50 наблюдений для надежной оценки
        if len(y_vals) < 50:
            return {
                "ar_error": np.nan,
                "ridge_error": np.nan,
                "naive_error": np.nan,
                "seasonal_naive_error": np.nan,
                "error_ratio": np.nan,
            }
        # Наивный прогноз: последнее наблюдение
        naive_pred = np.roll(y_vals, 1)[1:]
        naive_error = mean_absolute_error(y_vals[1:], naive_pred)
        features["naive_error"] = float(naive_error)
        # Сезонный наивный прогноз (лаг 24 для часовых данных)
        seasonal_lag = min(24, len(y_vals) // 2)
        if seasonal_lag > 0:
            seasonal_naive_pred = np.roll(y_vals, seasonal_lag)[seasonal_lag:]
            seasonal_naive_error = mean_absolute_error(
                y_vals[seasonal_lag:],
                seasonal_naive_pred
            )
            features["seasonal_naive_error"] = float(seasonal_naive_error)
        else:
            features["seasonal_naive_error"] = np.nan
        # Модель авторегрессии (AR)
        try:
            # Используем последние 50% данных для обучения
            train_size = max(10, int(len(y_vals) * 0.5))
            if train_size < len(y_vals):
                ar_model = AutoReg(y_vals[:train_size], lags=min(24, train_size - 1), old_names=False)
                ar_result = ar_model.fit()
                start_idx = train_size
                end_idx = len(y_vals) - 1
                if start_idx <= end_idx:
                    ar_pred = ar_result.predict(start=start_idx, end=end_idx)
                    ar_error = mean_absolute_error(y_vals[start_idx:end_idx + 1], ar_pred)
                    features["ar_error"] = float(ar_error)
                else:
                    features["ar_error"] = np.nan
            else:
                features["ar_error"] = np.nan
        except Exception:
            features["ar_error"] = np.nan
        # Гребневая регрессия с лагами
        try:
            # Создаем признаки лагов
            lags = [1, 2, 3, 24]
            valid_lags = [lag for lag in lags if lag < len(y_vals)]
            if valid_lags:
                X_lagged = np.column_stack([np.roll(y_vals, lag) for lag in valid_lags])
                # Обрезаем начальные значения с пропусками из-за лагов
                valid_idx = np.all(~np.isnan(X_lagged), axis=1)
                X_train = X_lagged[valid_idx]
                y_train = y_vals[valid_idx]
                if len(X_train) > 10:
                    ridge = Ridge(alpha=1.0)
                    ridge.fit(X_train, y_train)
                    ridge_pred = ridge.predict(X_train)
                    ridge_error = mean_absolute_error(y_train, ridge_pred)
                    features["ridge_error"] = float(ridge_error)
                else:
                    features["ridge_error"] = np.nan
            else:
                features["ridge_error"] = np.nan
        except Exception:
            features["ridge_error"] = np.nan
        # Отношение ошибок (мера прогнозируемости)
        # Значение < 1 означает, что модель лучше наивного прогноза
        if "ridge_error" in features and not np.isnan(features["ridge_error"]) and naive_error > 1e-10:
            features["error_ratio"] = float(features["ridge_error"] / naive_error)
        else:
            features["error_ratio"] = np.nan
        return features