# src/ts_feature_eng/optimization.py
"""
Байесовская оптимизация выбора методов инженерии признаков для временных рядов.
Реализует адаптивный поиск оптимальной комбинации трансформеров и их гиперпараметров
на основе мета-признаков временного ряда. Использует вероятностную модель для
эффективного исследования пространства решений с минимальным количеством оценок.
"""
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from skopt import gp_minimize
from skopt.space import Categorical, Integer, Real
from skopt.utils import use_named_args
from .base import TimeSeriesTransformer
from .meta_features import MetaFeatureExtractor
from .transformers.window import WindowTransformer
from .transformers.spectral import DWTTransformer, STLTransformer
from .transformers.time_encoding import TimeEncodingTransformer, CalendarFeaturesTransformer


class FeatureEngineeringPipeline:
    """
    Пайплайн последовательного применения трансформеров инженерии признаков.
    Объединяет несколько трансформеров в единый конвейер с возможностью
    условного включения/исключения компонентов на основе конфигурации.
    """

    def __init__(self, transformers: List[Tuple[str, TimeSeriesTransformer, bool]]):
        self.transformers = transformers
        self.feature_names_ = []
        self.is_fitted_ = False

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Optional[Union[pd.Series, np.ndarray]] = None) -> "FeatureEngineeringPipeline":
        """
        Последовательное обучение всех активных трансформеров в пайплайне.
        """
        X_current = X
        all_feature_names = []
        for name, transformer, active in self.transformers:
            if not active:
                continue
            # Для трансформеров, требующих целевую переменную (например, для обучения параметров)
            if isinstance(transformer, TimeEncodingTransformer) and transformer.fit_params:
                transformer.fit(X_current, y)
            else:
                transformer.fit(X_current)
            # Сохраняем имена признаков для каждого трансформера
            if hasattr(transformer, "get_feature_names"):
                all_feature_names.extend(transformer.get_feature_names())
        self.feature_names_ = all_feature_names
        self.is_fitted_ = True
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """
        Применение всех активных трансформеров к данным.
        """
        if not self.is_fitted_:
            raise ValueError("Pipeline is not fitted. Call fit() first.")
        features_dict = {}
        for name, transformer, active in self.transformers:
            if not active:
                continue
            # Применяем трансформер
            X_transformed = transformer.transform(X)
            # Добавляем признаки в общий словарь с префиксом имени трансформера
            for col in X_transformed.columns:
                features_dict[f"{name}.{col}"] = X_transformed[col]
        # Если нет активных трансформеров — возвращаем пустой DataFrame с тем же индексом
        if not features_dict:
            return pd.DataFrame(index=X.index)
        return pd.DataFrame(features_dict, index=X.index)

    def fit_transform(self, X: Union[pd.DataFrame, np.ndarray], y: Optional[Union[pd.Series, np.ndarray]] = None) -> pd.DataFrame:
        """
        Обучение и применение пайплайна за один шаг.
        """
        return self.fit(X, y).transform(X)

    def get_feature_names(self) -> List[str]:
        """
        Получение имен всех сгенерированных признаков.
        """
        if not self.is_fitted_:
            raise ValueError("Pipeline is not fitted. Call fit() first.")
        return self.feature_names_


class FeatureEngineeringOptimizer:
    """
    Оптимизатор выбора методов инженерии признаков через байесовскую оптимизацию.
    Адаптивно подбирает оптимальную комбинацию трансформеров и их гиперпараметров
    на основе структуры конкретного временного ряда. Использует мета-признаки
    для инициализации поиска и ускорения сходимости.
    """

    def __init__(
        self,
        model: Optional[BaseEstimator] = None,
        metric: Union[str, Callable] = "neg_mean_absolute_error",
        n_calls: int = 50,
        n_initial_points: int = 10,
        cv: Union[int, Any] = None,
        random_state: Optional[int] = 42,
        use_meta_features: bool = True,
        verbose: int = 0,
    ):
        from sklearn.linear_model import Ridge
        self.model = model or Ridge(alpha=1.0, random_state=random_state)
        self.metric = metric
        self.n_calls = n_calls
        self.n_initial_points = n_initial_points
        self.cv = cv or TimeSeriesSplit(n_splits=5)
        self.random_state = random_state
        self.use_meta_features = use_meta_features
        self.verbose = verbose
        self.best_params_ = None
        self.best_score_ = -np.inf
        self.search_space_ = None
        self.history_ = []

    def _define_search_space(self, meta_features: Optional[Dict[str, float]] = None) -> List:
        """
        Определение пространства поиска с корректными ограничениями.
        """
        space = []
        # 1. Категориальные параметры: какие трансформеры включать
        space.append(Categorical([True, False], name="use_window"))
        space.append(Categorical([True, False], name="use_dwt"))
        space.append(Categorical([True, False], name="use_stl"))
        space.append(Categorical([True, False], name="use_time_encoding"))
        space.append(Categorical([True, False], name="use_calendar_features"))

        # 2. Гиперпараметры оконного трансформера — НЕЗАВИСИМЫЕ Integer
        space.append(Integer(6, 168, name="window_size"))
        space.append(Categorical(["identity", "diff", "sma", "identity,diff", "identity,sma", "diff,sma", "identity,diff,sma"], name="window_transformations"))
        space.append(Integer(1, 24, name="window_min_periods"))  # Отдельно!

        # 3. Гиперпараметры DWT
        space.append(Categorical(["db4", "db8", "sym4", "coif1"], name="dwt_wavelet"))
        space.append(Integer(1, 5, name="dwt_max_level"))

        # 4. Гиперпараметры STL — seasonal ДОЛЖЕН быть нечётным
        space.append(Integer(12, 168, name="stl_period"))
        space.append(Categorical([7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31], name="stl_seasonal"))

        # 5. Гиперпараметры временного кодирования
        space.append(Categorical(["cyclic", "time2vec"], name="time_encoding_mode"))
        space.append(Categorical(["hour", "day_of_week", "month", "hour,day_of_week", "hour,month", "day_of_week,month", "hour,day_of_week,month"], name="cyclic_components"))
        space.append(Integer(4, 16, name="time2vec_dim"))

        # 6. Бинарные флаги постобработки
        space.append(Categorical([True, False], name="apply_shap_filter"))
        space.append(Real(0.0, 0.5, name="missing_threshold"))
        space.append(Real(0.0, 0.1, name="variance_threshold"))

        # Адаптация на основе мета-признаков
        if meta_features is not None and self.use_meta_features:
            self._adapt_search_space(space, meta_features)
        return space

    def _adapt_search_space(self, space: List, meta_features: Dict[str, float]) -> None:
        """Адаптация пространства поиска (оставлено без изменений)."""
        if "dominant_freq" in meta_features:
            dominant_freq = meta_features["dominant_freq"]
            if dominant_freq > 0.04:
                for dim in space:
                    if dim.name == "stl_period":
                        dim.low = 20
                        dim.high = 30
                        break

    def _build_pipeline(self, params: Dict[str, Any]) -> FeatureEngineeringPipeline:
        """
        Построение пайплайна с корректной защитой min_periods.
        """
        transformers = []
        # WindowTransformer
        if params["use_window"]:
            window_size = params["window_size"]
            # 🔑 КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ: min_periods не может превышать window_size
            min_periods = min(params["window_min_periods"], window_size)
            transformations = params["window_transformations"].split(",")
            transformers.append((
                "window",
                WindowTransformer(
                    window_size=window_size,
                    transformations=transformations,
                    statistics=["mean", "std", "min", "max", "skewness", "kurtosis", "slope", "acf1"],
                    min_periods=min_periods
                ),
                True
            ))
        # DWTTransformer
        if params["use_dwt"]:
            transformers.append((
                "dwt",
                DWTTransformer(
                    wavelet=params["dwt_wavelet"],
                    max_level=params["dwt_max_level"],
                    statistics=["mean", "std", "energy", "entropy"]
                ),
                True
            ))
        # STLTransformer
        if params["use_stl"]:
            transformers.append((
                "stl",
                STLTransformer(
                    period=params["stl_period"],
                    seasonal=params["stl_seasonal"],
                    statistics=["mean", "std", "min", "max", "skewness", "kurtosis"]
                ),
                True
            ))
        # TimeEncodingTransformer
        if params["use_time_encoding"]:
            if params["time_encoding_mode"] == "cyclic":
                components = params["cyclic_components"].split(",")
                transformers.append((
                    "time_encoding",
                    TimeEncodingTransformer(
                        mode="cyclic",
                        cyclic_components=components
                    ),
                    True
                ))
            else:  # time2vec
                transformers.append((
                    "time_encoding",
                    TimeEncodingTransformer(
                        mode="time2vec",
                        time2vec_dim=params["time2vec_dim"],
                        fit_params=False  # Без обучения параметров для ускорения
                    ),
                    True
                ))
        # CalendarFeaturesTransformer
        if params["use_calendar_features"]:
            transformers.append((
                "calendar",
                CalendarFeaturesTransformer(
                    features=["part_of_day", "is_weekend", "month", "season", "is_business_hour"]
                ),
                True
            ))
        return FeatureEngineeringPipeline(transformers)

    def _objective(self, X: pd.DataFrame, y: pd.Series, **params) -> float:
        """
        Целевая функция с защитой от ошибок и штрафами.
        """
        pipeline = self._build_pipeline(params)
        try:
            X_transformed = pipeline.fit_transform(X, y)
            # Проверка на вырожденность признакового пространства
            if X_transformed.shape[1] == 0:
                return 1e6  # Штраф за пустое пространство признаков
            if X_transformed.isna().all().all():
                return 1e6  # Штраф за полностью пропущенные признаки

            # Применяем фильтрацию признаков при необходимости
            if params["apply_shap_filter"]:
                X_transformed = self._apply_shap_filter(X_transformed, y)

            # Удаляем признаки с высоким % пропусков
            missing_ratio = X_transformed.isna().mean()
            X_transformed = X_transformed.loc[:, missing_ratio < params["missing_threshold"]]

            # Удаляем признаки с низкой дисперсией
            if X_transformed.shape[1] > 0:
                variance = X_transformed.var()
                X_transformed = X_transformed.loc[:, variance > params["variance_threshold"]]

            # Если после фильтрации не осталось признаков — штраф
            if X_transformed.shape[1] == 0:
                return 1e6

            # Заполняем оставшиеся пропуски средним значением
            X_transformed = X_transformed.fillna(X_transformed.mean())

            # Оцениваем качество через кросс-валидацию
            if isinstance(self.metric, str):
                scores = cross_val_score(
                    self.model,
                    X_transformed,
                    y,
                    cv=self.cv,
                    scoring=self.metric,
                    n_jobs=-1
                )
                mean_score = np.mean(scores)
            else:
                # Кастомная метрика — требуется ручная кросс-валидация
                scores = []
                for train_idx, test_idx in self.cv.split(X_transformed):
                    X_train, X_test = X_transformed.iloc[train_idx], X_transformed.iloc[test_idx]
                    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                    self.model.fit(X_train, y_train)
                    y_pred = self.model.predict(X_test)
                    score = self.metric(y_test, y_pred)
                    scores.append(score)
                mean_score = np.mean(scores)

            # Сохраняем историю
            self.history_.append({
                "params": params.copy(),
                "score": mean_score,
                "n_features": X_transformed.shape[1]
            })

            # Для минимизации возвращаем отрицательное значение (для метрик, где больше = лучше)
            # или положительное (для метрик ошибки, где меньше = лучше)
            if self.metric in ["r2", "neg_mean_absolute_error", "neg_mean_squared_error", "neg_root_mean_squared_error"]:
                return -mean_score
            else:
                return mean_score

        except Exception as e:
            if self.verbose >= 2:
                print(f"Ошибка при оценке конфигурации: {e}")
            return 1e6  # Большой штраф за ошибку

    def _apply_shap_filter(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Применение SHAP-фильтрации для отбора наиболее важных признаков."""
        try:
            import shap
            # Обучаем модель для вычисления SHAP-значений
            model = self.model.__class__(**self.model.get_params())
            model.fit(X.fillna(0), y)
            # Вычисляем SHAP-значения
            explainer = shap.Explainer(model, X.fillna(0).iloc[:100])  # Ограничиваем для скорости
            shap_values = explainer(X.fillna(0).iloc[:100])
            # Вычисляем среднюю абсолютную важность каждого признака
            shap_abs = np.abs(shap_values.values).mean(axis=0)
            # Отбираем признаки выше медианы по важности
            threshold = np.median(shap_abs)
            selected_mask = shap_abs >= threshold
            return X.loc[:, selected_mask]
        except ImportError:
            if self.verbose >= 1:
                print("SHAP не установлен. Пропускаем SHAP-фильтрацию.")
            return X
        except Exception as e:
            if self.verbose >= 2:
                print(f"Ошибка при SHAP-фильтрации: {e}")
            return X

    def optimize(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        meta_features: Optional[Union[Dict[str, float], pd.DataFrame]] = None
    ) -> Tuple[FeatureEngineeringPipeline, Dict[str, Any], float]:
        """
        Запуск байесовской оптимизации для поиска оптимального пайплайна.
        """
        # Валидация входных данных
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        if isinstance(y, np.ndarray):
            y = pd.Series(y, index=X.index)

        # Извлечение мета-признаков при необходимости
        if meta_features is None and self.use_meta_features:
            extractor = MetaFeatureExtractor(
                categories=["simple", "statistical", "spectral"],
                fill_method="linear"
            )
            meta_df = extractor.fit_transform(X, y)
            meta_features = meta_df.iloc[0].to_dict()
        elif isinstance(meta_features, pd.DataFrame):
            meta_features = meta_features.iloc[0].to_dict()

        # Определение пространства поиска
        self.search_space_ = self._define_search_space(meta_features)

        # Создание целевой функции с фиксированными X и y
        @use_named_args(self.search_space_)
        def objective_function(**kwargs):
            return self._objective(X, y, **kwargs)

        # Запуск байесовской оптимизации
        if self.verbose >= 1:
            print(f"Запуск байесовской оптимизации ({self.n_calls} итераций)...")

        result = gp_minimize(
            objective_function,
            self.search_space_,
            n_calls=self.n_calls,
            n_initial_points=self.n_initial_points,
            random_state=self.random_state,
            verbose=self.verbose >= 2
        )

        # Сохранение результатов
        self.best_params_ = {dim.name: result.x[i] for i, dim in enumerate(self.search_space_)}
        self.best_score_ = -result.fun if self.metric in [
            "neg_mean_absolute_error", "neg_mean_squared_error", "neg_root_mean_squared_error", "r2"
        ] else result.fun

        # Построение оптимального пайплайна
        best_pipeline = self._build_pipeline(self.best_params_)
        best_pipeline.fit(X, y)  # Обучаем на всех данных

        if self.verbose >= 1:
            active = [name for name, _, active in best_pipeline.transformers if active]
            print(f"\nОптимизация завершена!")
            print(f"Лучшая метрика: {self.best_score_:.4f}")
            print(f"Количество признаков: {len(best_pipeline.get_feature_names())}")
            print(f"Активные трансформеры: {active}")

        return best_pipeline, self.best_params_, self.best_score_

    def get_search_history(self) -> pd.DataFrame:
        """Получение истории поиска в виде DataFrame."""
        if not self.history_:
            return pd.DataFrame()
        records = []
        for entry in self.history_:
            record = entry["params"].copy()
            record["score"] = entry["score"]
            record["n_features"] = entry["n_features"]
            records.append(record)
        return pd.DataFrame(records)

    def suggest_initial_points(self, meta_features: Dict[str, float], n_points: int = 5) -> List[Dict[str, Any]]:
        """Генерация разумных начальных точек на основе мета-признаков."""
        points = []
        for i in range(n_points):
            point = {}
            point["use_window"] = True
            point["window_size"] = self._suggest_window_size(meta_features)
            point["window_transformations"] = self._suggest_transformations(meta_features)
            point["window_min_periods"] = max(1, point["window_size"] // 4)
            has_seasonality = meta_features.get("acf_24", 0) > 0.3 or meta_features.get("dominant_freq", 0) > 0.04
            point["use_dwt"] = has_seasonality or i % 2 == 0
            point["use_stl"] = has_seasonality and meta_features.get("acf_24", 0) > 0.5
            if point["use_stl"]:
                point["stl_period"] = self._suggest_stl_period(meta_features)
                point["stl_seasonal"] = 7 if point["stl_period"] < 48 else 15
            has_time_index = isinstance(meta_features.get("freq_hourly", None), (int, float))
            point["use_time_encoding"] = has_time_index
            point["use_calendar_features"] = has_time_index and meta_features.get("acf_168", 0) > 0.2
            if point["use_time_encoding"]:
                point["time_encoding_mode"] = "cyclic" if i % 2 == 0 else "time2vec"
                point["cyclic_components"] = "hour,day_of_week" if meta_features.get("acf_24", 0) > 0.3 else "hour"
                point["time2vec_dim"] = 8
            point["apply_shap_filter"] = i % 3 == 0
            point["missing_threshold"] = 0.2
            point["variance_threshold"] = 0.01
            point["window_size"] = max(6, min(168, point["window_size"] + np.random.randint(-6, 7)))
            points.append(point)
        return points

    def _suggest_window_size(self, meta_features: Dict[str, float]) -> int:
        """Рекомендация размера окна на основе мета-признаков."""
        base_size = 24
        dominant_freq = meta_features.get("dominant_freq", 0)
        if dominant_freq > 0.1:
            base_size = 6
        elif dominant_freq > 0.04:
            base_size = 24
        elif dominant_freq > 0.006:
            base_size = 168
        else:
            base_size = 720
        return min(168, max(6, base_size))

    def _suggest_transformations(self, meta_features: Dict[str, float]) -> str:
        """Рекомендация преобразований на основе стационарности."""
        adf_pvalue = meta_features.get("stationarity_adf", 1.0)
        if adf_pvalue > 0.1:
            return "diff"
        elif adf_pvalue > 0.05:
            return "identity,diff"
        else:
            return "identity,sma"

    def _suggest_stl_period(self, meta_features: Dict[str, float]) -> int:
        """Рекомендация периода сезонности для STL."""
        acf_24 = meta_features.get("acf_24", 0)
        acf_168 = meta_features.get("acf_168", 0)
        if acf_168 > acf_24 and acf_168 > 0.3:
            return 168
        elif acf_24 > 0.3:
            return 24
        else:
            return 12