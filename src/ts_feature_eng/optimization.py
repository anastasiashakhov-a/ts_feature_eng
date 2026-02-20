# src/ts_feature_eng/optimization.py
# src/ts_feature_eng/optimization.py
"""
Байесовская оптимизация выбора методов инженерии признаков для временных рядов.

Реализует адаптивный поиск оптимальной комбинации трансформеров и их гиперпараметров
на основе мета-признаков временного ряда. Использует вероятностную модель для
эффективного исследования пространства решений с минимальным количеством оценок.

Новые возможности (v2.0):
- Multi-objective оптимизация с индуктивными смещениями
- Anti-trivial penalty против доминирования одного признака
- Naive similarity penalty против копирования y(t-1)
- Entropy-based diversity bonus за разнообразие признаков
- Иерархический search space с авто-режимами пайплайна
- Early abort для ускорения поиска
"""
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from collections import Counter
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from skopt import gp_minimize
from skopt.space import Categorical, Integer, Real
from skopt.utils import use_named_args

from .base import TimeSeriesTransformer
from .meta_features import MetaFeatureExtractor
from .transformers.window import WindowTransformer
from .transformers.spectral import DWTTransformer, STLTransformer
from .transformers.time_encoding import TimeEncodingTransformer, CalendarFeaturesTransformer


# ============================================================================
# КОНФИГУРАЦИЯ ОПТИМИЗАТОРА (INDUCTIVE BIASES)
# ============================================================================
class OptimizerConfig:
    """
    Конфигурация индуктивных смещений для оптимизации.
    
    Позволяет управлять философией поиска без изменения кода трансформеров.
    """
    # Штрафы (чем больше — тем строже)
    dominance_lambda: float = 0.5      # Штраф за доминирование одного признака
    naive_lambda: float = 0.3          # Штраф за копирование lag_1
    entropy_lambda: float = 0.1        # Бонус за разнообразие признаков
    
    # Пороги
    dominance_threshold: float = 0.7   # Макс. доля важности одного признака
    naive_corr_threshold: float = 0.95 # Макс. корреляция с lag_1
    early_abort_penalty: float = 10.0  # Порог для ранней остановки
    
    # Режимы пайплайна
    pipeline_modes: Dict[str, List[str]] = {
        "baseline": ["core_lags"],
        "structure": ["core_lags", "window", "calendar"],
        "full": ["core_lags", "window", "calendar", "spectral", "dwt", "stl"]
    }
    
    # Ранняя остановка
    enable_early_abort: bool = True
    min_features_for_evaluation: int = 2


# ============================================================================
# ПЛАЙПЛАЙН ИНЖЕНЕРИИ ПРИЗНАКОВ
# ============================================================================
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
            # Для трансформеров, требующих целевую переменную
            if isinstance(transformer, TimeEncodingTransformer) and transformer.fit_params:
                transformer.fit(X_current, y)
            else:
                transformer.fit(X_current)
            # Сохраняем имена признаков
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
            X_transformed = transformer.transform(X)
            for col in X_transformed.columns:
                features_dict[f"{name}.{col}"] = X_transformed[col]
        
        if not features_dict:
            return pd.DataFrame(index=X.index)
        return pd.DataFrame(features_dict, index=X.index)

    def fit_transform(self, X: Union[pd.DataFrame, np.ndarray], y: Optional[Union[pd.Series, np.ndarray]] = None) -> pd.DataFrame:
        """Обучение и применение пайплайна за один шаг."""
        return self.fit(X, y).transform(X)

    def get_feature_names(self) -> List[str]:
        """Получение имен всех сгенерированных признаков."""
        if not self.is_fitted_:
            raise ValueError("Pipeline is not fitted. Call fit() first.")
        return self.feature_names_


# ============================================================================
# ОПТИМИЗАТОР С MULTI-OBJECTIVE ПОДДЕРЖКОЙ
# ============================================================================
class FeatureEngineeringOptimizer:
    """
    Оптимизатор выбора методов инженерии признаков через байесовскую оптимизацию.
    
    Адаптивно подбирает оптимальную комбинацию трансформеров и их гиперпараметров
    на основе структуры конкретного временного ряда. Использует мета-признаки
    для инициализации поиска и ускорения сходимости.
    
    Новые возможности:
    - Multi-objective оптимизация: MAE + diversity - triviality
    - Индуктивные смещения через штрафы, а не hard rules
    - Иерархический search space с авто-режимами
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
        config: Optional[OptimizerConfig] = None,
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
        self.config = config or OptimizerConfig()
        
        self.best_params_ = None
        self.best_score_ = -np.inf
        self.search_space_ = None
        self.history_ = []

    # =========================================================================
    # 1. ANTI-TRIVIAL PENALTY (DOMINANCE)
    # =========================================================================
    def _dominance_penalty(self, model: BaseEstimator, feature_names: List[str]) -> float:
        """
        Штраф за доминирование одного признака.
        
        Если один признак имеет >70% важности — пайплайн слишком "тривиален".
        """
        if not hasattr(model, "feature_importances_"):
            return 0.0
        
        importances = np.abs(model.feature_importances_)
        if len(importances) == 0 or importances.sum() == 0:
            return 0.0
        
        # Нормализуем важности
        importances = importances / importances.sum()
        max_share = importances.max()
        
        # Штраф если один признак доминирует
        if max_share > self.config.dominance_threshold:
            penalty = (max_share - self.config.dominance_threshold) * self.config.dominance_lambda
            return penalty
        return 0.0

    # =========================================================================
    # 2. NAIVE SIMILARITY PENALTY (LAG-COPY DETECTION)
    # =========================================================================
    def _naive_similarity_penalty(self, y_pred: np.ndarray, y_lag1: np.ndarray) -> float:
        """
        Штраф за слишком похожие на lag_1 прогнозы.
        
        Если прогноз ≈ y(t-1), значит модель просто копирует прошлое.
        """
        # Удаляем NaN для корреляции
        mask = ~np.isnan(y_pred) & ~np.isnan(y_lag1)
        if mask.sum() < 10:
            return 0.0
        
        y_pred_clean = y_pred[mask]
        y_lag1_clean = y_lag1[mask]
        
        # Корреляция Пирсона
        if np.std(y_pred_clean) < 1e-6 or np.std(y_lag1_clean) < 1e-6:
            return 0.0
        
        corr = np.corrcoef(y_pred_clean, y_lag1_clean)[0, 1]
        if np.isnan(corr):
            return 0.0
        
        # Штраф если слишком похожи
        if corr > self.config.naive_corr_threshold:
            penalty = (corr - self.config.naive_corr_threshold) * self.config.naive_lambda
            return penalty
        return 0.0

    # =========================================================================
    # 3. ENTROPY-BASED DIVERSITY BONUS
    # =========================================================================
    def _feature_group_entropy(self, feature_names: List[str]) -> float:
        """
        Вычисление энтропии распределения признаков по группам.
        
        Высокая энтропия = разнообразие = хорошо.
        """
        if not feature_names:
            return 0.0
        
        # Группируем по первому уровню имени (transformer name)
        groups = [name.split(".")[0] for name in feature_names]
        counts = Counter(groups)
        
        if len(counts) <= 1:
            return 0.0
        
        # Вычисляем энтропию Шеннона
        total = sum(counts.values())
        probs = np.array([count / total for count in counts.values()])
        entropy = -np.sum(probs * np.log(probs + 1e-9))
        
        # Нормализуем к [0, 1]
        max_entropy = np.log(len(counts))
        return entropy / max_entropy if max_entropy > 0 else 0.0

    # =========================================================================
    # 4. HIERARCHICAL SEARCH SPACE (PIPELINE MODES)
    # =========================================================================
    def _define_search_space(self, meta_features: Optional[Dict[str, float]] = None) -> List:
        """
        Определение пространства поиска с поддержкой pipeline modes.
        """
        space = []
        
        # режим пайплайна
        space.append(Categorical(
            list(self.config.pipeline_modes.keys()),
            name="pipeline_mode"
        ))
        
        # 1. Категориальные параметры (всегда доступны)
        space.append(Categorical([True, False], name="use_window"))
        space.append(Categorical([True, False], name="use_dwt"))
        space.append(Categorical([True, False], name="use_stl"))
        space.append(Categorical([True, False], name="use_time_encoding"))
        space.append(Categorical([True, False], name="use_calendar_features"))

        # 2. Гиперпараметры оконного трансформера
        space.append(Integer(6, 168, name="window_size"))
        space.append(Categorical(
            ["identity", "diff", "sma", "identity,diff", "identity,sma", "diff,sma", "identity,diff,sma"],
            name="window_transformations"
        ))
        space.append(Integer(1, 24, name="window_min_periods"))

        # 3. Гиперпараметры DWT
        space.append(Categorical(["db4", "db8", "sym4", "coif1"], name="dwt_wavelet"))
        space.append(Integer(1, 5, name="dwt_max_level"))

        # 4. Гиперпараметры STL
        space.append(Integer(12, 168, name="stl_period"))
        space.append(Categorical(
            [7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31],
            name="stl_seasonal"
        ))

        # 5. Гиперпараметры временного кодирования
        space.append(Categorical(["cyclic", "time2vec"], name="time_encoding_mode"))
        space.append(Categorical(
            ["hour", "day_of_week", "month", "hour,day_of_week", "hour,month", "day_of_week,month", "hour,day_of_week,month"],
            name="cyclic_components"
        ))
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
        """Адаптация пространства поиска на основе мета-признаков."""
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
        Построение пайплайна с поддержкой pipeline modes.
        """
        transformers = []
        
        # 🔑 ОПРЕДЕЛЯЕМ АКТИВНЫЕ ГРУППЫ ПО РЕЖИМУ
        mode = params.get("pipeline_mode", "full")
        active_groups = self.config.pipeline_modes.get(mode, self.config.pipeline_modes["full"])
        
        # WindowTransformer
        if params.get("use_window") and "window" in active_groups:
            window_size = params["window_size"]
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
        if params.get("use_dwt") and "dwt" in active_groups:
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
        if params.get("use_stl") and "stl" in active_groups:
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
        if params.get("use_time_encoding") and "time_encoding" in active_groups:
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
                        fit_params=False
                    ),
                    True
                ))
        
        # CalendarFeaturesTransformer
        if params.get("use_calendar_features") and "calendar" in active_groups:
            transformers.append((
                "calendar",
                CalendarFeaturesTransformer(
                    features=["part_of_day", "is_weekend", "month", "season", "is_business_hour"]
                ),
                True
            ))

        if "core_lags" in active_groups or mode == "baseline":
            from .transformers.lag import LagTransformer
            # Лаги определяются автоматически в pipeline.py
            # Здесь просто резервируем место
            pass
        
        return FeatureEngineeringPipeline(transformers)

    # =========================================================================
    # 5. EARLY ABORT FOR BAD PIPELINES
    # =========================================================================
    def _early_abort_check(self, X_transformed: pd.DataFrame, dominance_penalty: float) -> Optional[float]:
        """
        Ранняя остановка для заведомо плохих пайплайнов.
        
        Возвращает штраф если нужно остановиться, иначе None.
        """
        if not self.config.enable_early_abort:
            return None
        
        # Слишком мало признаков
        if X_transformed.shape[1] < self.config.min_features_for_evaluation:
            return 1e5
        
        # Слишком высокий штраф доминирования
        if dominance_penalty > self.config.early_abort_penalty:
            return 1e5
        
        return None

    # =========================================================================
    # MAIN OBJECTIVE FUNCTION (MULTI-OBJECTIVE)
    # =========================================================================
    def _objective(self, X: pd.DataFrame, y: pd.Series, **params) -> float:
        """
        Целевая функция с multi-objective поддержкой.
        
        Формула:
            final_score = MAE + dominance_penalty + naive_penalty - entropy_bonus
        
        Возвращает значение для минимизации.
        """
        pipeline = self._build_pipeline(params)
        
        try:
            # Генерация признаков
            X_transformed = pipeline.fit_transform(X, y)
            
            # Базовые проверки
            if X_transformed.shape[1] == 0:
                return 1e6
            if X_transformed.isna().all().all():
                return 1e6

            # Пост-фильтрация
            if params.get("apply_shap_filter"):
                X_transformed = self._apply_shap_filter(X_transformed, y)
            
            missing_ratio = X_transformed.isna().mean()
            X_transformed = X_transformed.loc[:, missing_ratio < params["missing_threshold"]]
            
            if X_transformed.shape[1] > 0:
                variance = X_transformed.var()
                X_transformed = X_transformed.loc[:, variance > params["variance_threshold"]]
            
            if X_transformed.shape[1] == 0:
                return 1e6

            X_transformed = X_transformed.fillna(X_transformed.mean())
            
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
                scores = []
                for train_idx, test_idx in self.cv.split(X_transformed):
                    X_train, X_test = X_transformed.iloc[train_idx], X_transformed.iloc[test_idx]
                    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                    self.model.fit(X_train, y_train)
                    y_pred = self.model.predict(X_test)
                    score = self.metric(y_test, y_pred)
                    scores.append(score)
                mean_score = np.mean(scores)
            
            
            # 1. Dominance penalty
            dominance_penalty = self._dominance_penalty(self.model, list(X_transformed.columns))
            
            # 2. Naive similarity penalty
            naive_penalty = 0.0
            lag1_col = [c for c in X_transformed.columns if "lag_1" in c and "core_lags" in c]
            if lag1_col and hasattr(self.model, "predict"):
                # Обучаем на всех данных для получения предсказаний
                self.model.fit(X_transformed.fillna(0), y)
                y_pred = self.model.predict(X_transformed.fillna(0))
                y_lag1 = X_transformed[lag1_col[0]].values
                naive_penalty = self._naive_similarity_penalty(y_pred, y_lag1)
            
            # 3. Entropy bonus (разнообразие)
            entropy = self._feature_group_entropy(list(X_transformed.columns))
            entropy_bonus = self.config.entropy_lambda * entropy
            
            # EARLY ABORT CHECK
            early_penalty = self._early_abort_check(X_transformed, dominance_penalty)
            if early_penalty is not None:
                return early_penalty
            
            # ФИНАЛЬНАЯ ФОРМУЛА
            # Для метрик где больше = лучше (R², neg_MAE):
            if self.metric in ["r2", "neg_mean_absolute_error", "neg_mean_squared_error", "neg_root_mean_squared_error"]:
                # Инвертируем score для минимизации
                base_score = -mean_score
            else:
                base_score = mean_score
            
            final_score = (
                base_score
                + dominance_penalty
                + naive_penalty
                - entropy_bonus  # Вычитаем бонус = уменьшаем score = лучше
            )
            
            # Сохраняем историю с расширенными метриками
            self.history_.append({
                "params": params.copy(),
                "base_score": mean_score,
                "dominance_penalty": dominance_penalty,
                "naive_penalty": naive_penalty,
                "entropy_bonus": entropy_bonus,
                "final_score": final_score,
                "n_features": X_transformed.shape[1]
            })
            
            return final_score

        except Exception as e:
            if self.verbose >= 2:
                print(f"Ошибка при оценке конфигурации: {e}")
            return 1e6

    def _apply_shap_filter(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Применение SHAP-фильтрации для отбора наиболее важных признаков."""
        try:
            import shap
            model = self.model.__class__(**self.model.get_params())
            model.fit(X.fillna(0), y)
            explainer = shap.Explainer(model, X.fillna(0).iloc[:100])
            shap_values = explainer(X.fillna(0).iloc[:100])
            shap_abs = np.abs(shap_values.values).mean(axis=0)
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

        # Извлечение мета-признаков
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

        # Создание целевой функции
        @use_named_args(self.search_space_)
        def objective_function(**kwargs):
            return self._objective(X, y, **kwargs)

        # Запуск оптимизации
        if self.verbose >= 1:
            print(f"Запуск байесовской оптимизации ({self.n_calls} итераций)...")
            print(f"  Режимы: {list(self.config.pipeline_modes.keys())}")
            print(f"  Штрафы: dominance={self.config.dominance_lambda}, naive={self.config.naive_lambda}")

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
        
        # Инвертируем score обратно для отчёта
        if self.metric in ["neg_mean_absolute_error", "neg_mean_squared_error", "neg_root_mean_squared_error", "r2"]:
            self.best_score_ = -result.fun
        else:
            self.best_score_ = result.fun

        # Построение оптимального пайплайна
        best_pipeline = self._build_pipeline(self.best_params_)
        best_pipeline.fit(X, y)

        if self.verbose >= 1:
            active = [name for name, _, active in best_pipeline.transformers if active]
            print(f"\nОптимизация завершена!")
            print(f"Лучшая метрика: {self.best_score_:.4f}")
            print(f"Режим пайплайна: {self.best_params_.get('pipeline_mode', 'N/A')}")
            print(f"Количество признаков: {len(best_pipeline.get_feature_names())}")
            print(f"Активные трансформеры: {active}")

        return best_pipeline, self.best_params_, self.best_score_

    def get_search_history(self) -> pd.DataFrame:
        """Получение истории поиска в виде DataFrame."""
        if not self.history_:
            return pd.DataFrame()
        return pd.DataFrame(self.history_)

    def suggest_initial_points(self, meta_features: Dict[str, float], n_points: int = 5) -> List[Dict[str, Any]]:
        """Генерация разумных начальных точек на основе мета-признаков."""
        points = []
        for i in range(n_points):
            point = {}
            
            # 🔑 ВЫБИРАЕМ РЕЖИМ ПЛАЙПЛАЙНА
            if i == 0:
                point["pipeline_mode"] = "baseline"
            elif i == 1:
                point["pipeline_mode"] = "structure"
            else:
                point["pipeline_mode"] = "full"
            
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
            
            # Небольшая вариация для exploration
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