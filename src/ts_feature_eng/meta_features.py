# src/ts_feature_eng/meta_features.py

"""
Модуль для извлечения мета-признаков временных рядов.

Предоставляет инструменты для автоматического анализа структуры временного ряда
и извлечения мета-признаков из 5 категорий:
1. Простые характеристики (длина, пропуски, частота)
2. Статистические свойства (нестационарность, линейность, нормальность)
3. Информационно-теоретические метрики (энтропия, сложность)
4. Спектральные характеристики (частоты, автокорреляция)
5. Ландмарковые метрики (ошибки простых моделей)

Используется для адаптивной инициализации поиска оптимальных методов
инженерии признаков в FeatureEngineeringOptimizer.
"""

import warnings
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, kpss

from .base import TimeSeriesError


class MetaFeatureExtractor:
    """
    Извлекатель мета-признаков временного ряда.
    
    Автоматически анализирует структуру временного ряда и извлекает
    мета-признаки из заданных категорий. Поддерживает обработку
    многомерных рядов (анализ первого столбца).
    
    Параметры
    ----------
    categories : List[str] или "all", по умолчанию "all"
        Категории мета-признаков для извлечения:
        - "simple": базовые характеристики
        - "statistical": статистические свойства  
        - "information_theoretic": информационные метрики
        - "spectral": спектральные характеристики
        - "landmarking": ландмарковые метрики
    fill_method : str, по умолчанию "linear"
        Метод заполнения пропусков перед анализом:
        - "linear": линейная интерполяция
        - "forward": forward fill
        - "backward": backward fill  
        - "none": не заполнять (требует отсутствия пропусков)
    n_jobs : int, по умолчанию None
        Количество потоков для параллельных вычислений (не реализовано).
    
    Атрибуты
    ----------
    meta_features_ : Dict[str, float]
        Извлеченные мета-признаки после вызова fit_transform.
    
    Примеры
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from ts_feature_eng.meta_features import MetaFeatureExtractor
    >>> 
    >>> # Создаем тестовый временной ряд
    >>> dates = pd.date_range("2023-01-01", periods=1000, freq="H")
    >>> df = pd.DataFrame({
    ...     "value": np.sin(2 * np.pi * np.arange(1000) / 24) + np.random.randn(1000) * 0.1
    ... }, index=dates)
    >>> y = df["value"].shift(-1).dropna()  # Прогноз на 1 шаг вперед
    >>> X = df.iloc[:-1]
    >>> 
    >>> # Создаем и применяем извлекатель мета-признаков
    >>> extractor = MetaFeatureExtractor(
    ...     categories=["simple", "statistical", "spectral"],
    ...     fill_method="linear"
    ... )
    >>> meta_df = extractor.fit_transform(X, y)
    >>> 
    >>> print(f"Извлечено {len(meta_df.columns)} мета-признаков")
    >>> print(f"Ключевые мета-признаки: {list(meta_df.columns[:5])}")
    """
    
    _valid_categories = [
        "simple",
        "statistical", 
        "information_theoretic",
        "spectral",
        "landmarking"
    ]
    
    def __init__(
        self,
        categories: Union[List[str], str] = "all",
        fill_method: str = "linear",
        n_jobs: Optional[int] = None
    ):
        if categories == "all":
            self.categories = self._valid_categories.copy()
        elif isinstance(categories, list):
            invalid_cats = set(categories) - set(self._valid_categories)
            if invalid_cats:
                raise ValueError(
                    f"Invalid categories: {invalid_cats}. "
                    f"Valid options: {self._valid_categories}"
                )
            self.categories = categories
        else:
            raise ValueError("categories must be 'all' or list of valid categories")
        
        if fill_method not in ["linear", "forward", "backward", "none"]:
            raise ValueError(
                f"Invalid fill_method: {fill_method}. "
                f"Valid options: ['linear', 'forward', 'backward', 'none']"
            )
        
        self.fill_method = fill_method
        self.n_jobs = n_jobs
        self.meta_features_: Optional[Dict[str, float]] = None
    
    def fit_transform(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> pd.DataFrame:
        """
        Извлечение мета-признаков из временного ряда.
        
        Параметры
        ----------
        X : pd.DataFrame или np.ndarray
            Входные данные временного ряда.
        y : pd.Series или np.ndarray, опционально
            Целевая переменная (требуется только для landmarking мета-признаков).
        
        Возвращает
        ----------
        meta_df : pd.DataFrame
            DataFrame с одной строкой и колонками-мета-признаками.
        """
        # Валидация входных данных
        if X is None:
            raise ValueError("X cannot be None")
        
        if isinstance(X, np.ndarray):
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        
        if not isinstance(X, pd.DataFrame):
            raise ValueError(f"X must be pd.DataFrame or np.ndarray, got {type(X)}")
        
        if X.empty:
            raise ValueError("Input DataFrame is empty")
        
        # Обработка пропусков
        X_filled = self._handle_missing_values(X)
        
        # Извлечение мета-признаков по категориям
        meta_features = {}
        
        if "simple" in self.categories:
            meta_features.update(self._extract_simple_features(X_filled))
        
        if "statistical" in self.categories:
            meta_features.update(self._extract_statistical_features(X_filled))
        
        if "information_theoretic" in self.categories:
            meta_features.update(self._extract_information_theoretic_features(X_filled))
        
        if "spectral" in self.categories:
            meta_features.update(self._extract_spectral_features(X_filled))
        
        if "landmarking" in self.categories:
            if y is None:
                warnings.warn(
                    "Landmarking meta-features require target variable y. "
                    "Skipping landmarking features."
                )
            else:
                meta_features.update(self._extract_landmarking_features(X_filled, y))
        
        self.meta_features_ = meta_features
        
        return pd.DataFrame([meta_features])
    
    def _handle_missing_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """Обработка пропусков в данных."""
        if self.fill_method == "none":
            if X.isna().any().any():
                raise ValueError("Input contains missing values but fill_method='none'")
            return X
        
        X_filled = X.copy()
        
        if self.fill_method == "linear":
            X_filled = X_filled.interpolate(method="linear", limit_direction="both")
        elif self.fill_method == "forward":
            X_filled = X_filled.ffill().bfill()
        elif self.fill_method == "backward":
            X_filled = X_filled.bfill().ffill()
        
        # Заполняем оставшиеся пропуски средним значением
        for col in X_filled.columns:
            if X_filled[col].isna().any():
                X_filled[col] = X_filled[col].fillna(X_filled[col].mean())
        
        return X_filled
    
    def _extract_simple_features(self, X: pd.DataFrame) -> Dict[str, float]:
        """Извлечение простых характеристик ряда."""
        features = {}
        
        # Базовые характеристики
        features["length"] = float(len(X))
        features["missing_ratio"] = float(X.isna().sum().sum() / (X.shape[0] * X.shape[1]))
        features["num_features"] = float(X.shape[1])
        
        # Частота дискретизации (если есть временной индекс)
        if isinstance(X.index, pd.DatetimeIndex):
            try:
                freq = pd.infer_freq(X.index)
                if freq:
                    # Преобразуем частоту в часы
                    if freq.endswith('T') or freq.endswith('min'):
                        hours = float(freq[:-1]) / 60
                    elif freq.endswith('H'):
                        hours = float(freq[:-1])
                    elif freq.endswith('D'):
                        hours = float(freq[:-1]) * 24
                    elif freq.endswith('W'):
                        hours = float(freq[:-1]) * 24 * 7
                    else:
                        hours = 1.0
                    features["freq_hourly"] = hours
                else:
                    features["freq_hourly"] = 1.0
            except:
                features["freq_hourly"] = 1.0
        else:
            features["freq_hourly"] = np.nan
        
        return features
    
    def _extract_statistical_features(self, X: pd.DataFrame) -> Dict[str, float]:
        """Извлечение статистических свойств ряда."""
        # Работаем с первым столбцом для многомерных рядов
        series = X.iloc[:, 0].dropna().values
        features = {}
        
        if len(series) < 10:
            # Возвращаем NaN значения для коротких рядов
            return {
                "stationarity_adf": np.nan,
                "stationarity_kpss": np.nan,
                "linearity": np.nan,
                "variance": np.nan,
                "skewness": np.nan,
                "kurtosis": np.nan,
                "normality_shapiro": np.nan,
                "normality_jarque": np.nan,
                "autocorrelation_ljungbox": np.nan,
                "homoskedasticity_bp": np.nan,
                "acf_1": np.nan,
                "acf_24": np.nan,
                "acf_168": np.nan,
                "acf_30": np.nan,
                "acf_365": np.nan
            }
        
        # Тест на стационарность (ADF)
        try:
            adf_result = adfuller(series, autolag="AIC")
            features["stationarity_adf"] = float(adf_result[1])  # p-value
        except:
            features["stationarity_adf"] = np.nan
        
        # Тест на стационарность (KPSS)
        try:
            kpss_result = kpss(series, nlags="auto")
            features["stationarity_kpss"] = float(kpss_result[1])  # p-value
        except:
            features["stationarity_kpss"] = np.nan
        
        # Линейность (R² линейной регрессии)
        try:
            X_reg = np.arange(len(series)).reshape(-1, 1)
            model = LinearRegression().fit(X_reg, series)
            features["linearity"] = float(model.score(X_reg, series))
        except:
            features["linearity"] = np.nan
        
        # Дисперсия
        features["variance"] = float(np.var(series, ddof=1))
        
        # Асимметрия
        try:
            features["skewness"] = float(stats.skew(series, nan_policy="omit"))
        except:
            features["skewness"] = np.nan
        
        # Эксцесс
        try:
            features["kurtosis"] = float(stats.kurtosis(series, nan_policy="omit"))
        except:
            features["kurtosis"] = np.nan
        
        # Тест на нормальность (Shapiro-Wilk)
        try:
            # Shapiro-Wilk имеет ограничение на размер выборки (<= 5000)
            sample_size = min(len(series), 5000)
            _, p_shapiro = stats.shapiro(series[:sample_size])
            features["normality_shapiro"] = float(p_shapiro)
        except:
            features["normality_shapiro"] = np.nan
        
        # Тест на нормальность (Jarque-Bera)
        try:
            _, p_jarque = stats.jarque_bera(series)
            features["normality_jarque"] = float(p_jarque)
        except:
            features["normality_jarque"] = np.nan
        
        # Тест на автокорреляцию (Ljung-Box)
        try:
            # Ограничиваем количество лагов разумным значением
            max_lags = min(10, len(series) // 10)
            if max_lags > 0:
                lb_pvals = []
                for lag in range(1, max_lags + 1):
                    try:
                        _, pval = stats.box_ljung(series, lags=lag, return_df=False)
                        lb_pvals.append(pval[-1] if isinstance(pval, np.ndarray) else pval)
                    except:
                        continue
                
                if lb_pvals:
                    features["autocorrelation_ljungbox"] = float(min(lb_pvals))
                else:
                    features["autocorrelation_ljungbox"] = np.nan
            else:
                features["autocorrelation_ljungbox"] = np.nan
        except:
            features["autocorrelation_ljungbox"] = np.nan
        
        # Тест на гомоскедастичность (Breusch-Pagan)
        try:
            # Для Breusch-Pagan нужна регрессия, используем линейный тренд
            X_reg = np.arange(len(series)).reshape(-1, 1)
            model = LinearRegression().fit(X_reg, series)
            residuals = series - model.predict(X_reg)
            
            # Выполняем тест Breusch-Pagan
            from statsmodels.stats.diagnostic import het_breuschpagan
            _, p_bp, _, _ = het_breuschpagan(residuals, X_reg)
            features["homoskedasticity_bp"] = float(p_bp)
        except:
            features["homoskedasticity_bp"] = np.nan
        
        # Автокорреляции для разных горизонтов
        try:
            from statsmodels.tsa.stattools import acf
            max_lag = min(len(series) // 2, 400)
            acf_vals = acf(series, nlags=max_lag, fft=True)
            
            features["acf_1"] = float(acf_vals[1]) if len(acf_vals) > 1 else np.nan
            features["acf_24"] = float(acf_vals[24]) if len(acf_vals) > 24 else np.nan
            features["acf_168"] = float(acf_vals[168]) if len(acf_vals) > 168 else np.nan
            features["acf_30"] = float(acf_vals[30]) if len(acf_vals) > 30 else np.nan
            features["acf_365"] = float(acf_vals[365]) if len(acf_vals) > 365 else np.nan
        except:
            features["acf_1"] = np.nan
            features["acf_24"] = np.nan
            features["acf_168"] = np.nan
            features["acf_30"] = np.nan
            features["acf_365"] = np.nan
        
        return features
    
    def _extract_information_theoretic_features(self, X: pd.DataFrame) -> Dict[str, float]:
        """Извлечение информационно-теоретических метрик."""
        series = X.iloc[:, 0].dropna().values
        features = {}
        
        if len(series) < 10:
            return {
                "entropy": np.nan,
                "permutation_entropy": np.nan,
                "hurst_exponent": np.nan,
                "nonlinearity": np.nan
            }
        
        # Энтропия Шеннона
        try:
            # Нормализуем данные для вычисления энтропии
            normalized = (series - series.mean()) / series.std()
            # Дискретизируем в 10 бинов
            hist, _ = np.histogram(normalized, bins=10, density=True)
            hist = hist[hist > 0]  # Удаляем нулевые вероятности
            if len(hist) > 0:
                features["entropy"] = float(-np.sum(hist * np.log(hist)))
            else:
                features["entropy"] = np.nan
        except:
            features["entropy"] = np.nan
        
        # Перестановочная энтропия
        try:
            features["permutation_entropy"] = float(self._permutation_entropy(series, order=3, delay=1))
        except:
            features["permutation_entropy"] = np.nan
        
        # Показатель Херста
        try:
            features["hurst_exponent"] = float(self._compute_hurst_exponent(series))
        except:
            features["hurst_exponent"] = np.nan
        
        # Мера нелинейности (сравнение линейной и нелинейной модели)
        try:
            # Линейная модель
            X_reg = np.arange(len(series)).reshape(-1, 1)
            linear_model = LinearRegression().fit(X_reg, series)
            linear_pred = linear_model.predict(X_reg)
            linear_mae = mean_absolute_error(series, linear_pred)
            
            # Простая нелинейная модель (скользящее среднее)
            window_size = min(10, len(series) // 10)
            if window_size > 1:
                nonlinear_pred = pd.Series(series).rolling(window=window_size, center=True).mean().ffill().bfill().values
                nonlinear_mae = mean_absolute_error(series, nonlinear_pred)
                # Меньшая ошибка нелинейной модели указывает на нелинейность
                features["nonlinearity"] = float((linear_mae - nonlinear_mae) / linear_mae)
            else:
                features["nonlinearity"] = np.nan
        except:
            features["nonlinearity"] = np.nan
        
        return features
    
    def _extract_spectral_features(self, X: pd.DataFrame) -> Dict[str, float]:
        """Извлечение спектральных характеристик ряда."""
        series = X.iloc[:, 0].dropna().values
        features = {}
        
        if len(series) < 20:
            return {
                "dominant_freq": np.nan,
                "acf_1": np.nan,
                "acf_24": np.nan,
                "acf_168": np.nan,
                "spectral_entropy": np.nan
            }
        
        # Доминирующая частота через FFT
        try:
            n = len(series)
            fft_vals = np.fft.fft(series - np.mean(series))
            fft_freqs = np.fft.fftfreq(n)
            # Находим частоту с максимальной амплитудой (игнорируем DC компоненту)
            power_spectrum = np.abs(fft_vals[1:n//2])**2
            dominant_idx = np.argmax(power_spectrum) + 1
            dominant_freq = abs(fft_freqs[dominant_idx])
            features["dominant_freq"] = float(dominant_freq)
        except:
            features["dominant_freq"] = np.nan
        
        # Спектральная энтропия
        try:
            if 'power_spectrum' in locals():
                ps_norm = power_spectrum / np.sum(power_spectrum)
                ps_norm = ps_norm[ps_norm > 0]
                if len(ps_norm) > 0:
                    features["spectral_entropy"] = float(-np.sum(ps_norm * np.log(ps_norm)))
                else:
                    features["spectral_entropy"] = np.nan
            else:
                features["spectral_entropy"] = np.nan
        except:
            features["spectral_entropy"] = np.nan
        
        return features
    
    def _extract_landmarking_features(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]) -> Dict[str, float]:
        """Извлечение ландмарковых мета-признаков (ошибки простых моделей)."""
        # Конвертируем y в numpy array если нужно
        if isinstance(y, pd.Series):
            y = y.values
        
        # Убеждаемся, что длины совпадают
        min_len = min(len(X), len(y))
        X_subset = X.iloc[:min_len]
        y_subset = y[:min_len]
        
        features = {}
        
        if len(y_subset) < 10:
            return {
                "naive_error": np.nan,
                "seasonal_naive_error": np.nan,
                "ar_error": np.nan,
                "error_ratio": np.nan
            }
        
        # Наивный прогноз (последнее наблюдение)
        try:
            naive_pred = np.roll(y_subset, 1)
            naive_pred[0] = y_subset[0]  # Первое значение берем как есть
            naive_mae = mean_absolute_error(y_subset, naive_pred)
            features["naive_error"] = float(naive_mae)
        except:
            features["naive_error"] = np.nan
        
        # Сезонный наивный прогноз (значение 24 периода назад)
        try:
            seasonal_lag = 24
            if len(y_subset) > seasonal_lag:
                seasonal_naive_pred = np.roll(y_subset, seasonal_lag)
                seasonal_naive_pred[:seasonal_lag] = y_subset[:seasonal_lag]  # Первые значения берем как есть
                seasonal_naive_mae = mean_absolute_error(y_subset, seasonal_naive_pred)
                features["seasonal_naive_error"] = float(seasonal_naive_mae)
            else:
                features["seasonal_naive_error"] = np.nan
        except:
            features["seasonal_naive_error"] = np.nan
        
        # Модель авторегрессии (AR(1))
        try:
            ar_pred = np.roll(y_subset, 1)
            ar_pred[0] = y_subset[0]
            # Простая линейная регрессия AR(1)
            X_ar = ar_pred[:-1].reshape(-1, 1)
            y_ar = y_subset[1:]
            if len(X_ar) > 1:
                ar_model = LinearRegression().fit(X_ar, y_ar)
                ar_full_pred = np.zeros_like(y_subset)
                ar_full_pred[0] = y_subset[0]
                for i in range(1, len(y_subset)):
                    ar_full_pred[i] = ar_model.predict(ar_full_pred[i-1:i].reshape(-1, 1))[0]
                ar_mae = mean_absolute_error(y_subset, ar_full_pred)
                features["ar_error"] = float(ar_mae)
            else:
                features["ar_error"] = np.nan
        except:
            features["ar_error"] = np.nan
        
        # Отношение ошибок (лучшая модель / наивный прогноз)
        try:
            errors = [features.get("naive_error", np.inf), 
                     features.get("seasonal_naive_error", np.inf), 
                     features.get("ar_error", np.inf)]
            best_error = min(err for err in errors if not np.isnan(err) and err != np.inf)
            if not np.isnan(features.get("naive_error", np.nan)) and features["naive_error"] > 0:
                features["error_ratio"] = float(best_error / features["naive_error"])
            else:
                features["error_ratio"] = np.nan
        except:
            features["error_ratio"] = np.nan
        
        return features
    
    def _permutation_entropy(self, x: np.ndarray, order: int = 3, delay: int = 1) -> float:
        """Вычисление перестановочной энтропии."""
        if len(x) < order + delay:
            return np.nan
        
        # Генерация паттернов
        patterns = []
        for i in range(len(x) - (order - 1) * delay):
            pattern = x[i:i + order * delay:delay]
            patterns.append(tuple(np.argsort(pattern)))
        
        # Подсчет частот паттернов
        pattern_counts = {}
        for pattern in patterns:
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        
        # Вычисление энтропии
        total_patterns = len(patterns)
        entropy = 0.0
        for count in pattern_counts.values():
            p = count / total_patterns
            entropy -= p * np.log(p)
        
        # Нормализация
        max_entropy = np.log(np.math.factorial(order))
        return entropy / max_entropy if max_entropy > 0 else 0.0
    
    def _compute_hurst_exponent(self, series: np.ndarray) -> float:
        """Вычисление показателя Херста."""
        lags = range(2, min(100, len(series) // 2))
        if len(lags) < 2:
            return np.nan
        
        tau = [np.sqrt(np.std(np.subtract(series[lag:], series[:-lag]))) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0] * 2.0
    
    def get_feature_names(self) -> List[str]:
        """Получение имен извлеченных мета-признаков."""
        if self.meta_features_ is None:
            raise ValueError("MetaFeatureExtractor has not been fitted yet.")
        return list(self.meta_features_.keys())
    
    def recommend_core_features(self, meta_features: Dict[str, float]) -> Dict[str, bool]:
        """
        Рекомендация обязательных признаков на основе мета-признаков.
        
        Параметры
        ----------
        meta_features : Dict[str, float]
            Словарь мета-признаков временного ряда.
        
        Возвращает
        ----------
        recommendations : Dict[str, bool]
            Рекомендации по включению обязательных признаков:
            - "include_lags": Включать ли лаги
            - "include_rolling": Включать ли rolling-статистики
            - "min_lag": Минимальный лаг для включения
            - "max_lag": Максимальный лаг для включения
        """
        recommendations = {
            "include_lags": False,
            "include_rolling": False,
            "min_lag": 1,
            "max_lag": 1
        }
        
        # Анализ автокорреляции для рекомендации лагов
        acf_1 = meta_features.get("acf_1", 0)
        acf_24 = meta_features.get("acf_24", 0)
        acf_168 = meta_features.get("acf_168", 0)
        acf_365 = meta_features.get("acf_365", 0)
        
        # Если есть значимая автокорреляция на лаге 1, включаем лаги
        if abs(acf_1) > 0.1:
            recommendations["include_lags"] = True
            recommendations["min_lag"] = 1
            
            # Определяем максимальный лаг на основе значимых ACF
            if abs(acf_365) > 0.1:
                recommendations["max_lag"] = 365
            elif abs(acf_168) > 0.1:
                recommendations["max_lag"] = 168
            elif abs(acf_24) > 0.1:
                recommendations["max_lag"] = 24
            else:
                recommendations["max_lag"] = 1
        
        # Всегда включаем rolling-статистики для временных рядов
        recommendations["include_rolling"] = True
        
        return recommendations