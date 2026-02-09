# src/ts_feature_eng/__init__.py 
"""
Модуль автоматической инженерии признаков для временных рядов.

Предоставляет единый интерфейс AutoFeatureEngineer для адаптивного подбора
оптимальных методов преобразования временных рядов с минимальным участием пользователя.
"""

# Версия пакета
__version__ = "0.1.0"

# Основной интерфейс
from .pipeline import AutoFeatureEngineer

# Базовые классы и исключения
from .base import TimeSeriesTransformer, FeatureSelector, TimeSeriesError

# Трансформеры
from .transformers.window import WindowTransformer
from .transformers.spectral import DWTTransformer, STLTransformer
from .transformers.time_encoding import TimeEncodingTransformer, CalendarFeaturesTransformer

# Утилиты
from .meta_features import MetaFeatureExtractor
from .selection import (
    VarianceThresholdSelector,
    MissingValueSelector,
    SHAPFeatureSelector,
    CombinedFeatureSelector,
)
from .optimization import FeatureEngineeringOptimizer, FeatureEngineeringPipeline

# Утилиты для работы с временными рядами
from .utils.validation import validate_time_series
from .utils.time_series import (
    create_sliding_windows,
    compute_rolling_statistics,
    detect_seasonality,
    detrend_series,
)
from .utils.metrics import mae, rmse, mape, smape, mase, r2

# Экспорт основных компонентов для удобства использования
__all__ = [
    # Основной интерфейс
    "AutoFeatureEngineer",
    
    # Базовые классы
    "TimeSeriesTransformer",
    "FeatureSelector",
    "TimeSeriesError",
    
    # Трансформеры
    "WindowTransformer",
    "DWTTransformer", 
    "STLTransformer",
    "TimeEncodingTransformer",
    "CalendarFeaturesTransformer",
    
    # Селекторы признаков
    "VarianceThresholdSelector",
    "MissingValueSelector", 
    "SHAPFeatureSelector",
    "CombinedFeatureSelector",
    
    # Мета-признаки и оптимизация
    "MetaFeatureExtractor",
    "FeatureEngineeringOptimizer",
    "FeatureEngineeringPipeline",
    
    # Утилиты
    "validate_time_series",
    "create_sliding_windows",
    "compute_rolling_statistics", 
    "detect_seasonality",
    "detrend_series",
    "mae",
    "rmse", 
    "mape",
    "smape",
    "mase",
    "r2",
]