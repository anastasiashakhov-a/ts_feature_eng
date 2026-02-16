# src/ts_feature_eng/pipeline.py

"""
Основной пайплайн автоматической инженерии признаков для временных рядов.

Предоставляет единый интерфейс AutoFeatureEngineer для адаптивного подбора
оптимальных методов преобразования временных рядов с минимальным участием пользователя.
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .base import TimeSeriesError
from .meta_features import MetaFeatureExtractor
from .optimization import FeatureEngineeringOptimizer, FeatureEngineeringPipeline
from .selection import CombinedFeatureSelector
from .transformers.window import WindowTransformer
from .transformers.lag import LagTransformer
from .transformers.spectral import DWTTransformer, STLTransformer
from .transformers.time_encoding import TimeEncodingTransformer, CalendarFeaturesTransformer

class AutoFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Автоматический инженер признаков для временных рядов.
    
    Адаптивно подбирает оптимальную комбинацию методов инженерии признаков
    на основе структуры конкретного временного ряда. Использует мета-признаки
    для инициализации поиска и байесовскую оптимизацию для нахождения
    наилучшего пайплайна преобразований.
    
    Поддерживает интерфейс scikit-learn (fit/transform) для интеграции
    в существующие пайплайны машинного обучения.
    
    Параметры
    ----------
    optimize : bool, по умолчанию True
        Включить байесовскую оптимизацию выбора методов. Если False,
        используется фиксированный пайплайн с разумными настройками по умолчанию.
    use_meta_features : bool, по умолчанию True
        Использовать мета-признаки для адаптивной инициализации поиска.
    n_calls : int, по умолчанию 30
        Количество итераций байесовской оптимизации (актуально при optimize=True).
    n_initial_points : int, по умолчанию 10
        Количество начальных точек для разведочного поиска.
    model : object, опционально
        Модель-прокси для оценки качества признакового пространства.
        По умолчанию используется Ridge регрессия.
    metric : str или Callable, по умолчанию "neg_mean_absolute_error"
        Метрика для оптимизации. Поддерживаемые строки:
        - "neg_mean_absolute_error"
        - "neg_mean_squared_error"
        - "neg_root_mean_squared_error"
        - "r2"
    apply_selection : bool, по умолчанию True
        Применять постфильтрацию признаков (дисперсия, пропуски, SHAP).
    selection_threshold : float, по умолчанию 0.2
        Порог для фильтрации признаков с высокой долей пропусков.
    variance_threshold : float, по умолчанию 0.01
        Порог дисперсии для удаления неинформативных признаков.
    shap_selection : bool, по умолчанию False
        Применять SHAP-отбор признаков (требует установки shap).
    shap_n_features : int или float, опционально
        Количество или доля признаков для сохранения после SHAP-отбора.
    random_state : int, опционально
        Фиксация случайного состояния для воспроизводимости.
    verbose : int, по умолчанию 0
        Уровень детализации логирования (0=минимум, 1=итерации оптимизации).
    
    Атрибуты
    ----------
    best_pipeline_ : FeatureEngineeringPipeline или None
        Оптимальный пайплайн преобразований, найденный в процессе оптимизации.
    meta_features_ : Dict[str, float] или None
        Извлеченные мета-признаки временного ряда.
    selected_features_ : List[str] или None
        Список отобранных признаков после постфильтрации.
    feature_names_ : List[str]
        Имена всех сгенерированных признаков.
    optimization_history_ : pd.DataFrame или None
        История байесовской оптимизации (доступна при optimize=True).
    is_fitted_ : bool
        Флаг, указывающий, был ли вызван метод fit().
    
    Примеры
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from ts_feature_eng import AutoFeatureEngineer
    >>> 
    >>> # Создаем тестовый временной ряд
    >>> dates = pd.date_range("2023-01-01", periods=1000, freq="H")
    >>> df = pd.DataFrame({
    ...     "value": np.sin(2 * np.pi * np.arange(1000) / 24) + np.random.randn(1000) * 0.1
    ... }, index=dates)
    >>> y = df["value"].shift(-1).dropna()  # Прогноз на 1 шаг вперед
    >>> X = df.iloc[:-1]
    >>> 
    >>> # Создаем и обучаем автоматический инженер признаков
    >>> engineer = AutoFeatureEngineer(optimize=True, n_calls=20, verbose=1)
    >>> X_transformed = engineer.fit_transform(X, y)
    >>> 
    >>> print(f"Сгенерировано признаков: {X_transformed.shape[1]}")
    >>> print(f"Примеры признаков: {list(X_transformed.columns[:5])}")
    """
    
    def __init__(
        self,
        optimize: bool = True,
        use_meta_features: bool = True,
        n_calls: int = 30,
        n_initial_points: int = 10,
        model: Optional[Any] = None,
        metric: Union[str, callable] = "neg_mean_absolute_error",
        apply_selection: bool = True,
        selection_threshold: float = 0.2,
        variance_threshold: float = 0.01,
        shap_selection: bool = False,
        shap_n_features: Optional[Union[int, float]] = None,
        random_state: Optional[int] = 42,
        verbose: int = 0,
    ):
        self.optimize = optimize
        self.use_meta_features = use_meta_features
        self.n_calls = n_calls
        self.n_initial_points = n_initial_points
        self.model = model
        self.metric = metric
        self.apply_selection = apply_selection
        self.selection_threshold = selection_threshold
        self.variance_threshold = variance_threshold
        self.shap_selection = shap_selection
        self.shap_n_features = shap_n_features
        self.random_state = random_state
        self.verbose = verbose
        
        # Внутренние атрибуты
        self.best_pipeline_ = None
        self.meta_features_ = None
        self.selector_ = None
        self.selected_features_ = None
        self.feature_names_ = []
        self.optimization_history_ = None
        self.is_fitted_ = False
    
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> "AutoFeatureEngineer":
        """
        Обучение автоматического инженера признаков.
        
        Выполняет следующие этапы:
        1. Валидация и нормализация входных данных
        2. Извлечение мета-признаков временного ряда
        3. Поиск оптимального пайплайна преобразований (если optimize=True)
        4. Применение постфильтрации признаков (если apply_selection=True)
        
        Параметры
        ----------
        X : pd.DataFrame или np.ndarray
            Входные данные временного ряда.
        y : pd.Series или np.ndarray, опционально
            Целевая переменная для задачи прогнозирования.
        
        Возвращает
        ----------
        self : AutoFeatureEngineer
            Обученный инженер признаков.
        
        Выбрасывает
        ----------
        TimeSeriesError
            При ошибках валидации данных или процесса обучения.
        """
        # Валидация входных данных
        X, y = self._validate_input(X, y)
        
        # Шаг 1: Извлечение мета-признаков
        if self.verbose >= 1:
            print("Извлечение мета-признаков временного ряда...")
        
        meta_extractor = MetaFeatureExtractor(
            categories=["simple", "statistical", "spectral"],
            fill_method="linear"
        )
        meta_df = meta_extractor.fit_transform(X, y)
        self.meta_features_ = meta_df.iloc[0].to_dict()
        
        if self.verbose >= 2:
            print(f"Извлечено {len(self.meta_features_)} мета-признаков")
            print(f"Ключевые мета-признаки: {list(self.meta_features_.keys())[:5]}")
        
        # Шаг 2: Создание core pipeline (обязательные признаки)
        core_pipeline = self._create_core_pipeline(X, y)
        
        # Шаг 3: Поиск оптимального auto pipeline (если требуется)
        if self.optimize:
            if self.verbose >= 1:
                print(f"Запуск байесовской оптимизации ({self.n_calls} итераций)...")
            
            # Создаем оптимизатор с настройками пользователя
            optimizer = FeatureEngineeringOptimizer(
                model=self.model,
                metric=self.metric,
                n_calls=self.n_calls,
                n_initial_points=self.n_initial_points,
                random_state=self.random_state,
                use_meta_features=self.use_meta_features,
                verbose=self.verbose >= 2
            )
            
            # Запускаем оптимизацию только для auto-компонентов
            best_auto_pipeline, best_params, best_score = optimizer.optimize(
                X, y, meta_features=self.meta_features_
            )
            
            # Комбинируем core и auto пайплайны
            combined_transformers = core_pipeline.transformers + best_auto_pipeline.transformers
            self.best_pipeline_ = FeatureEngineeringPipeline(combined_transformers)
            
            self.optimization_history_ = optimizer.get_search_history()
            
            if self.verbose >= 1:
                print(f"Оптимизация завершена. Лучшая метрика: {best_score:.4f}")
                active_transformers = [name for name, _, active in combined_transformers if active]
                print(f"Активные трансформеры: {active_transformers}")
        
        else:
            # Используем фиксированный пайплайн по умолчанию
            if self.verbose >= 1:
                print("Использование фиксированного пайплайна по умолчанию (без оптимизации)...")
            
            # Создаем разумный пайплайн по умолчанию на основе мета-признаков
            default_pipeline = self._create_default_pipeline(X)
            combined_transformers = core_pipeline.transformers + default_pipeline.transformers
            self.best_pipeline_ = FeatureEngineeringPipeline(combined_transformers)
            
            if self.verbose >= 1:
                print("Фиксированный пайплайн создан")
        
        # Шаг 4: Применение постфильтрации признаков
        if self.apply_selection:
            if self.verbose >= 1:
                print("Применение постфильтрации признаков...")
            
            selector = CombinedFeatureSelector(
                missing_threshold=self.selection_threshold,
                variance_threshold=self.variance_threshold,
                shap_n_features=self.shap_n_features if self.shap_selection else None,
                shap_model=self.model,
                skip_shap=not self.shap_selection
            )
            
            # Генерируем признаки для обучения селектора
            X_temp = self.best_pipeline_.transform(X)
            
            # Обучаем селектор
            selector.fit(X_temp, y)
            self.selector_ = selector
            self.selected_features_ = selector.get_selected_features()
            
            if self.verbose >= 1:
                print(f"Отобрано {len(self.selected_features_)} признаков из {X_temp.shape[1]}")
        
        # Сохраняем имена признаков
        if self.best_pipeline_ is not None:
            all_features = self.best_pipeline_.get_feature_names()
            if self.apply_selection and self.selected_features_ is not None:
                # Фильтруем только отобранные признаки
                self.feature_names_ = [f for f in all_features if f in self.selected_features_]
            else:
                self.feature_names_ = all_features
        
        self.is_fitted_ = True
        
        if self.verbose >= 1:
            print(f"Готово! Сгенерировано {len(self.feature_names_)} признаков")
        
        return self
    
    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """
        Применение обученного пайплайна к новым данным.
        
        Параметры
        ----------
        X : pd.DataFrame или np.ndarray
            Входные данные временного ряда.
        
        Возвращает
        ----------
        X_transformed : pd.DataFrame
            DataFrame с сгенерированными и отфильтрованными признаками.
        
        Выбрасывает
        ----------
        TimeSeriesError
            Если метод вызван до обучения (fit).
        """
        if not self.is_fitted_:
            raise TimeSeriesError(
                "AutoFeatureEngineer is not fitted. Call fit() before transform()."
            )
        
        # Валидация входных данных
        X, _ = self._validate_input(X, None)
        
        # Применяем оптимальный пайплайн
        X_transformed = self.best_pipeline_.transform(X)
        
        # Применяем постфильтрацию, если она была обучена
        if self.apply_selection and self.selector_ is not None:
            # Проверяем наличие всех необходимых признаков
            missing_features = set(self.selected_features_) - set(X_transformed.columns)
            if missing_features:
                # Добавляем отсутствующие признаки со значениями NaN
                for feat in missing_features:
                    X_transformed[feat] = np.nan
            
            # Применяем селектор
            X_transformed = self.selector_.transform(X_transformed)
        
        # Упорядочиваем столбцы в соответствии с обученным порядком
        if self.feature_names_:
            X_transformed = X_transformed[self.feature_names_]
        
        return X_transformed
    
    def fit_transform(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> pd.DataFrame:
        """
        Обучение и применение пайплайна за один шаг.
        
        Параметры
        ----------
        X : pd.DataFrame или np.ndarray
            Входные данные временного ряда.
        y : pd.Series или np.ndarray, опционально
            Целевая переменная.
        
        Возвращает
        ----------
        X_transformed : pd.DataFrame
            DataFrame с сгенерированными и отфильтрованными признаками.
        """
        return self.fit(X, y).transform(X)
    
    def get_feature_names(self) -> List[str]:
        """
        Получение имен всех сгенерированных признаков.
        
        Возвращает
        ----------
        feature_names : List[str]
            Список имен признаков в порядке их генерации.
        """
        if not self.is_fitted_:
            raise TimeSeriesError(
                "AutoFeatureEngineer is not fitted. Call fit() first."
            )
        
        return self.feature_names_
    
    def get_optimization_history(self) -> Optional[pd.DataFrame]:
        """
        Получение истории байесовской оптимизации.
        
        Возвращает
        ----------
        history : pd.DataFrame или None
            DataFrame с историей оценок конфигураций во время оптимизации.
            Возвращает None, если оптимизация не проводилась (optimize=False).
        """
        return self.optimization_history_
    
    def get_meta_features(self) -> Optional[Dict[str, float]]:
        """
        Получение извлеченных мета-признаков временного ряда.
        
        Возвращает
        ----------
        meta_features : Dict[str, float] или None
            Словарь мета-признаков. Возвращает None, если метод не был обучен.
        """
        return self.meta_features_ if self.is_fitted_ else None
    
    def _validate_input(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]]
    ) -> tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Валидация и нормализация входных данных.
        
        Параметры
        ----------
        X : pd.DataFrame или np.ndarray
            Входные данные.
        y : pd.Series, np.ndarray или None
            Целевая переменная.
        
        Возвращает
        ----------
        X_validated : pd.DataFrame
            Валидированный DataFrame.
        y_validated : pd.Series или None
            Валидированная целевая переменная.
        """
        # Конвертация в DataFrame при необходимости
        if isinstance(X, np.ndarray):
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        
        if not isinstance(X, pd.DataFrame):
            raise TimeSeriesError(f"X must be pd.DataFrame or np.ndarray, got {type(X)}")
        
        if X.empty:
            raise TimeSeriesError("Input DataFrame is empty")
        
        # Проверка временного индекса
        valid_index_types = (
            pd.DatetimeIndex, 
            pd.RangeIndex,
            pd.Index  # Общий тип для целочисленных индексов
        )
        is_valid_index = (
            isinstance(X.index, valid_index_types) or 
            pd.api.types.is_integer_dtype(X.index)
        )
        if not is_valid_index:
            # Попытка конвертации в временной индекс
            try:
                X.index = pd.to_datetime(X.index)
            except:
                # Если не удается конвертировать — используем RangeIndex
                X = X.reset_index(drop=True)
        
        # Валидация целевой переменной
        if y is not None:
            if isinstance(y, np.ndarray):
                if y.ndim > 1:
                    y = y.ravel()
                y = pd.Series(y, index=X.index[:len(y)])
            
            if not isinstance(y, pd.Series):
                raise TimeSeriesError(f"y must be pd.Series or np.ndarray, got {type(y)}")
            
            # Проверка соответствия длин
            if len(y) != len(X):
                raise TimeSeriesError(
                    f"X and y have inconsistent lengths: {len(X)} vs {len(y)}"
                )
        
        return X, y
    
    def _create_core_pipeline(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> FeatureEngineeringPipeline:
        """Создание core пайплайна с обязательными признаками (лаги + rolling)."""
        n_samples = len(X)
        has_time_index = isinstance(X.index, pd.DatetimeIndex)
        
        # Определяем лаги на основе частоты данных
        lags = []
        if has_time_index:
            freq = pd.infer_freq(X.index)
            if freq and 'T' in freq:  # минутная частота
                lags = [1, 4, 96, 672]  # 15 мин, 1 час, 24 часа, 7 дней
            elif freq and 'H' in freq:  # часовая частота
                lags = [1, 24, 168]      # 1 час, 24 часа, 7 дней
            elif freq and 'D' in freq:  # дневная частота
                lags = [1, 7, 30, 365]   # 1 день, неделя, месяц, год
            else:
                lags = [1, min(24, n_samples//10), min(168, n_samples//2)]
        else:
            # Для не временных индексов используем относительные лаги
            lags = [1, min(10, n_samples//20), min(50, n_samples//5)]
        
        # Фильтруем лаги, которые возможны для данного размера данных
        lags = [lag for lag in lags if lag < n_samples]
        
        transformers = []
        
        # Добавляем обязательные лаги
        if lags:
            transformers.append(
                (
                    "core_lags",
                    LagTransformer(lags=lags),
                    True
                )
            )
        
        # Добавляем rolling-статистики
        if lags:
            window_size = max(lags) + 1
            transformers.append(
                (
                    "core_rolling",
                    WindowTransformer(
                        window_size=window_size,
                        transformations=["identity"],
                        statistics=["mean", "std", "min", "max"]
                    ),
                    True
                )
            )
        
        return FeatureEngineeringPipeline(transformers)
    
    def _create_default_pipeline(self, X: pd.DataFrame) -> FeatureEngineeringPipeline:
        """
        Создание фиксированного пайплайна по умолчанию без оптимизации.
        
        Пайплайн адаптируется на основе базовых мета-признаков ряда.
        
        Параметры
        ----------
        X : pd.DataFrame
            Входные данные для анализа структуры.
        
        Возвращает
        ----------
        pipeline : FeatureEngineeringPipeline
            Настроенный пайплайн преобразований.
        """
        # Анализируем базовые характеристики ряда
        has_time_index = isinstance(X.index, pd.DatetimeIndex)
        n_samples = len(X)
        
        # Определяем разумные параметры по умолчанию
        window_size = 24 if n_samples >= 100 else max(6, n_samples // 10)
        use_stl = has_time_index and n_samples >= 100
        
        # Создаем трансформеры
        transformers = [
            (
                "window",
                WindowTransformer(
                    window_size=window_size,
                    transformations=["identity", "diff"],
                    statistics=["mean", "std", "min", "max", "slope", "acf1"]
                ),
                True
            ),
            (
                "dwt",
                DWTTransformer(
                    wavelet="db4",
                    max_level=min(3, pywt.dwt_max_level(n_samples, "db4"))
                ),
                n_samples >= 50  # DWT требует минимум 50 наблюдений
            ),
        ]
        
        # Добавляем STL при наличии временного индекса и достаточной длины
        if use_stl:
            transformers.append(
                (
                    "stl",
                    STLTransformer(period=24),
                    True
                )
            )
        
        # Добавляем временные кодировки при наличии временного индекса
        if has_time_index:
            transformers.append(
                (
                    "time_encoding",
                    TimeEncodingTransformer(
                        mode="cyclic",
                        cyclic_components=["hour", "day_of_week"]
                    ),
                    True
                )
            )
            transformers.append(
                (
                    "calendar",
                    CalendarFeaturesTransformer(
                        features=["part_of_day", "is_weekend"]
                    ),
                    True
                )
            )
        
        return FeatureEngineeringPipeline(transformers)
    
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """
        Получение параметров для совместимости с интерфейсом sklearn.
        
        Параметры
        ----------
        deep : bool, по умолчанию True
            Игнорируется (требуется для совместимости).
        
        Возвращает
        ----------
        params : Dict[str, Any]
            Словарь параметров.
        """
        return {
            "optimize": self.optimize,
            "use_meta_features": self.use_meta_features,
            "n_calls": self.n_calls,
            "n_initial_points": self.n_initial_points,
            "model": self.model,
            "metric": self.metric,
            "apply_selection": self.apply_selection,
            "selection_threshold": self.selection_threshold,
            "variance_threshold": self.variance_threshold,
            "shap_selection": self.shap_selection,
            "shap_n_features": self.shap_n_features,
            "random_state": self.random_state,
            "verbose": self.verbose,
        }
    
    def set_params(self, **params) -> "AutoFeatureEngineer":
        """
        Установка параметров для совместимости с интерфейсом sklearn.
        
        Параметры
        ----------
        **params : Dict[str, Any]
            Параметры для установки.
        
        Возвращает
        ----------
        self : AutoFeatureEngineer
            Обновленный инженер признаков.
        """
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter {key} for AutoFeatureEngineer")
        return self
    
    def save(self, path: str) -> None:
        """
        Сохранение состояния обученного инженера признаков.
        
        Параметры
        ----------
        path : str
            Путь для сохранения состояния (в формате pickle).
        
        Примечание
        ----------
        Для полного восстановления требуется сохранить также:
        - Оптимальный пайплайн преобразований
        - Селектор признаков (если использовался)
        - Мета-признаки
        
        Рекомендуется использовать вместе с `load()` для восстановления.
        """
        try:
            import pickle
            
            state = {
                "best_pipeline": self.best_pipeline_,
                "selector": self.selector_,
                "meta_features": self.meta_features_,
                "selected_features": self.selected_features_,
                "feature_names": self.feature_names_,
                "is_fitted": self.is_fitted_,
                "params": self.get_params()
            }
            
            with open(path, "wb") as f:
                pickle.dump(state, f)
            
            if self.verbose >= 1:
                print(f"Состояние сохранено в {path}")
        
        except Exception as e:
            raise TimeSeriesError(f"Ошибка при сохранении состояния: {e}")
    
    @classmethod
    def load(cls, path: str) -> "AutoFeatureEngineer":
        """
        Загрузка состояния обученного инженера признаков.
        
        Параметры
        ----------
        path : str
            Путь к сохраненному состоянию.
        
        Возвращает
        ----------
        engineer : AutoFeatureEngineer
            Восстановленный инженер признаков.
        
        Примечание
        ----------
        Требует установки всех зависимостей, использованных при обучении.
        """
        try:
            import pickle
            
            with open(path, "rb") as f:
                state = pickle.load(f)
            
            # Создаем новый инстанс с сохраненными параметрами
            engineer = cls(**state["params"])
            
            # Восстанавливаем внутреннее состояние
            engineer.best_pipeline_ = state["best_pipeline"]
            engineer.selector_ = state["selector"]
            engineer.meta_features_ = state["meta_features"]
            engineer.selected_features_ = state["selected_features"]
            engineer.feature_names_ = state["feature_names"]
            engineer.is_fitted_ = state["is_fitted"]
            
            return engineer
        
        except Exception as e:
            raise TimeSeriesError(f"Ошибка при загрузке состояния: {e}")


    def analyze_feature_selection(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        comparison_methods: List[str] = ["shap", "pearson", "f_regression", "mutual_info"],
        top_k: int = 20,
        plot_correlation_matrix: bool = True,
        save_plots: bool = True
    ) -> Dict[str, Any]:
        """
        Анализ эффективности различных стратегий отбора признаков.
        
        Параметры
        ----------
        X : pd.DataFrame или np.ndarray
            Входные данные временного ряда.
        y : pd.Series или np.ndarray, опционально
            Целевая переменная (требуется для корреляции/модельных методов).
        comparison_methods : List[str], по умолчанию ["shap", "pearson", ...]
            Список методов для сравнения:
            - "shap": SHAP важность (требует обученной модели)
            - "pearson": Корреляция Пирсона с целевой переменной
            - "f_regression": F-статистика ANOVA
            - "mutual_info": Взаимная информация
        top_k : int, по умолчанию 20
            Количество признаков для отбора по каждой стратегии.
        plot_correlation_matrix : bool, по умолчанию True
            Строить ли матрицу корреляций между отобранными признаками.
        save_plots : bool, по умолчанию True
            Сохранять ли графики на диск.
        
        Возвращает
        ----------
        analysis_results : Dict[str, Any]
            Словарь с результатами анализа:
            - 'selected_features_by_method': Dict[str, List[str]]
            - 'correlation_matrices': Dict[str, pd.DataFrame]
            - 'intersection_analysis': pd.DataFrame
            - 'performance_comparison': pd.DataFrame (если y предоставлена)
        """
        from sklearn.feature_selection import f_regression, mutual_info_regression, SelectKBest
        from sklearn.linear_model import Ridge
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        if not self.is_fitted_:
            raise TimeSeriesError("AutoFeatureEngineer must be fitted before calling analyze_feature_selection.")
        
        if y is None and any(m in comparison_methods for m in ["pearson", "f_regression", "mutual_info"]):
            raise ValueError("Target variable 'y' is required for correlation/model-based selection methods.")
        
        # Шаг 1: Генерация признаков через обученный пайплайн
        X_transformed = self.best_pipeline_.transform(X)
        
        if self.apply_selection and self.selector_ is not None:
            # Если пост-селектор уже обучен, применяем его для согласованности
            X_transformed = self.selector_.transform(X_transformed)
        
        print(f"Анализ отбора признаков на {X_transformed.shape[1]} сгенерированных признаках...")
        
        results = {}
        
        # Метод 1: SHAP (если модель обучена)
        if "shap" in comparison_methods and self.model_ is not None:
            try:
                import shap
                explainer = shap.LinearExplainer(self.model_, X_transformed.fillna(0).iloc[:100])
                shap_values = explainer.shap_values(X_transformed.fillna(0).iloc[:100])
                shap_importance = np.abs(shap_values).mean(0)
                shap_feature_names = X_transformed.columns
                top_shap_idx = np.argsort(shap_importance)[-top_k:][::-1]
                results["shap"] = [shap_feature_names[i] for i in top_shap_idx]
            except ImportError:
                print("SHAP не установлен. Пропускаем SHAP-анализ.")
            except Exception as e:
                print(f"Ошибка при SHAP-анализе: {e}")
        
        # Метод 2: Корреляция Пирсона
        if "pearson" in comparison_methods and y is not None:
            correlations = X_transformed.corrwith(y).abs().sort_values(ascending=False)
            results["pearson"] = correlations.head(top_k).index.tolist()
        
        # Метод 3: F-регрессия
        if "f_regression" in comparison_methods and y is not None:
            selector_f = SelectKBest(score_func=f_regression, k=top_k)
            X_f_selected = selector_f.fit_transform(X_transformed.fillna(0), y)
            results["f_regression"] = X_transformed.columns[selector_f.get_support()].tolist()
        
        # Метод 4: Взаимная информация
        if "mutual_info" in comparison_methods and y is not None:
            mi_scores = mutual_info_regression(X_transformed.fillna(0), y, random_state=42)
            mi_feature_scores = pd.Series(mi_scores, index=X_transformed.columns).sort_values(ascending=False)
            results["mutual_info"] = mi_feature_scores.head(top_k).index.tolist()
        
        # Проверяем, что хотя бы один метод сработал
        if not results:
            raise ValueError("Ни один из запрошенных методов отбора не может быть применен.")
        
        print(f"Отбор выполнено по {len(results)} методам: {list(results.keys())}")
        
        # Шаг 2: Анализ пересечений
        from itertools import combinations
        intersection_analysis = []
        for method1, method2 in combinations(results.keys(), 2):
            set1 = set(results[method1])
            set2 = set(results[method2])
            intersection = len(set1 & set2)
            union = len(set1 | set2)
            jaccard = intersection / union if union > 0 else 0
            intersection_analysis.append({
                "Method_1": method1,
                "Method_2": method2,
                "Intersection_Count": intersection,
                "Jaccard_Similarity": jaccard
            })
        
        intersection_df = pd.DataFrame(intersection_analysis)
        
        # Шаг 3: Матрицы корреляций
        correlation_matrices = {}
        if plot_correlation_matrix:
            for method_name, selected_features in results.items():
                if len(selected_features) == 0:
                    continue
                X_subset = X_transformed[selected_features].fillna(0)
                corr_matrix = X_subset.corr()
                correlation_matrices[method_name] = corr_matrix
                
                if save_plots:
                    plt.figure(figsize=(10, 8))
                    sns.heatmap(
                        corr_matrix,
                        annot=True,
                        fmt=".2f",
                        cmap="coolwarm",
                        center=0,
                        square=True,
                        cbar_kws={"shrink": 0.8}
                    )
                    plt.title(f"Матрица корреляций: {method_name} (top-{top_k} features)")
                    plt.tight_layout()
                    plt.savefig(f"correlation_matrix_{method_name}.png", dpi=150, bbox_inches='tight')
                    plt.close()
                    print(f"  Матрица корреляций для {method_name} сохранена как 'correlation_matrix_{method_name}.png'")
        
        # Шаг 4: Сравнение производительности (если предоставлена целевая переменная)
        performance_comparison = None
        if y is not None:
            model_for_comparison = Ridge(alpha=1.0, random_state=42)
            perf_data = []
            
            for method_name, selected_features in results.items():
                if len(selected_features) == 0:
                    continue
                X_subset = X_transformed[selected_features].fillna(0)
                
                # Кросс-валидация
                from sklearn.model_selection import cross_val_score
                scores = cross_val_score(model_for_comparison, X_subset, y, cv=3, scoring='neg_mean_absolute_error')
                mean_mae = -scores.mean()
                std_mae = scores.std()
                
                perf_data.append({
                    "Method": method_name,
                    "Mean_MAE": mean_mae,
                    "Std_MAE": std_mae,
                    "Num_Features_Selected": len(selected_features)
                })
            
            performance_comparison = pd.DataFrame(perf_data)
            performance_comparison = performance_comparison.sort_values("Mean_MAE")
        
        analysis_results = {
            "selected_features_by_method": results,
            "intersection_analysis": intersection_df,
            "correlation_matrices": correlation_matrices,
            "performance_comparison": performance_comparison
        }
        
        return analysis_results