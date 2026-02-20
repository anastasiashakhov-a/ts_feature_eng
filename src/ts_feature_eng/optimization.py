# src/ts_feature_eng/optimization.py
# src/ts_feature_eng/optimization.py
"""
Байесовская оптимизация выбора методов инженерии признаков для временных рядов.

Реализует адаптивный поиск оптимальной комбинации трансформеров и их гиперпараметров
на основе мета-признаков временного ряда. Использует вероятностную модель для
эффективного исследования пространства решений с минимальным количеством оценок.

Версия 2.0 — Улучшения:
- Out-of-sample penalties (OOF predictions)
- Unified penalty scaling (relative to MAE)
- Explicit collapse diagnostics
- Semantic entropy (by information type)
- Horizon-aware scoring
- Conditional search space
- Progressive complexity modes
- Relative score vs naive baseline
"""
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_predict
from skopt import gp_minimize
from skopt.space import Categorical, Integer, Real, Space
from skopt.utils import use_named_args
import warnings

from .base import TimeSeriesTransformer
from .meta_features import MetaFeatureExtractor
from .transformers.window import WindowTransformer
from .transformers.spectral import DWTTransformer, STLTransformer
from .transformers.time_encoding import TimeEncodingTransformer, CalendarFeaturesTransformer
from .transformers.lag import LagTransformer


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
    collapse_lambda: float = 0.4       # Штраф за feature collapse
    
    # Пороги
    dominance_threshold: float = 0.7   # Макс. доля важности одного признака
    naive_corr_threshold: float = 0.95 # Макс. корреляция с lag_1
    early_abort_penalty: float = 10.0  # Порог для ранней остановки
    
    # Масштабирование штрафов
    scale_penalties: bool = True       # Приводить штрафы к масштабу MAE
    use_oof_penalties: bool = True     # Считать штрафы на OOF predictions
    
    # Режимы пайплайна
    pipeline_modes: Dict[str, List[str]] = None
    use_progressive_modes: bool = False  # Запускать BO в фазах
    
    # Горизонты
    forecast_horizons: List[int] = None  # [1, 7, 24] для horizon-aware scoring
    use_horizon_aware: bool = False      # Использовать gain(h) метрику
    
    # Диагностика
    log_diagnostics: bool = True       # Логировать diagnostic metrics
    diagnostics_file: str = None       # Путь к файлу логов
    
    # Search space
    use_conditional_space: bool = True # Активные параметры только если группа включена
    
    def __init__(
        self,
        dominance_lambda: float = 0.5,
        naive_lambda: float = 0.3,
        entropy_lambda: float = 0.1,
        collapse_lambda: float = 0.4,
        dominance_threshold: float = 0.7,
        naive_corr_threshold: float = 0.95,
        scale_penalties: bool = True,
        use_oof_penalties: bool = True,
        use_progressive_modes: bool = False,
        use_horizon_aware: bool = False,
        forecast_horizons: Optional[List[int]] = None,
        log_diagnostics: bool = True,
        diagnostics_file: Optional[str] = None,  # ДОБАВЬТЕ ЭТУ СТРОКУ
        use_conditional_space: bool = True,
    ):
        self.dominance_lambda = dominance_lambda
        self.naive_lambda = naive_lambda
        self.entropy_lambda = entropy_lambda
        self.collapse_lambda = collapse_lambda
        self.dominance_threshold = dominance_threshold
        self.naive_corr_threshold = naive_corr_threshold
        self.scale_penalties = scale_penalties
        self.use_oof_penalties = use_oof_penalties
        self.use_progressive_modes = use_progressive_modes
        self.use_horizon_aware = use_horizon_aware
        self.forecast_horizons = forecast_horizons or [1, 7, 24]
        self.log_diagnostics = log_diagnostics
        self.diagnostics_file = diagnostics_file  
        self.use_conditional_space = use_conditional_space
        
        # Режимы пайплайна по умолчанию
        self.pipeline_modes = {
            "baseline": ["core_lags"],
            "structure": ["core_lags", "window", "calendar"],
            "full": ["core_lags", "window", "calendar", "spectral", "dwt", "stl"]
        }
        
        # Диагностика
        self.diagnostics_history = []


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
        """Последовательное обучение всех активных трансформеров в пайплайне."""
        X_current = X
        all_feature_names = []
        for name, transformer, active in self.transformers:
            if not active:
                continue
            if isinstance(transformer, TimeEncodingTransformer) and transformer.fit_params:
                transformer.fit(X_current, y)
            else:
                transformer.fit(X_current)
            if hasattr(transformer, "get_feature_names"):
                all_feature_names.extend(transformer.get_feature_names())
        self.feature_names_ = all_feature_names
        self.is_fitted_ = True
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """Применение всех активных трансформеров к данным."""
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
# ДИАГНОСТИКА И ЛОГИРОВАНИЕ
# ============================================================================
class DiagnosticsLogger:
    """
    Логгер для диагностики процесса оптимизации.
    """
    def __init__(self, filepath: Optional[str] = None):
        self.filepath = filepath
        self.history = []
    
    def log(self, iteration: int, diagnostics: Dict[str, Any]):
        """Запись диагностических данных."""
        record = {
            "iteration": iteration,
            "timestamp": pd.Timestamp.now(),
            **diagnostics
        }
        self.history.append(record)
        
        if self.filepath:
            import os
            os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
            
            df = pd.DataFrame(self.history)
            df.to_csv(self.filepath, index=False)
    
    def get_summary(self) -> pd.DataFrame:
        """Получение сводки по диагностике."""
        if not self.history:
            return pd.DataFrame()
        return pd.DataFrame(self.history)


# ============================================================================
# ОПТИМИЗАТОР С MULTI-OBJECTIVE ПОДДЕРЖКОЙ
# ============================================================================
class FeatureEngineeringOptimizer:
    """
    Оптимизатор выбора методов инженерии признаков через байесовскую оптимизацию.
    
    Адаптивно подбирает оптимальную комбинацию трансформеров и их гиперпараметров
    на основе структуры конкретного временного ряда. Использует мета-признаки
    для инициализации поиска и ускорения сходимости.
    
    Новые возможности (v2.0):
    - Multi-objective оптимизация: MAE + diversity - triviality
    - Out-of-sample penalties (OOF predictions)
    - Unified penalty scaling (relative to MAE)
    - Semantic entropy (by information type)
    - Horizon-aware scoring
    - Conditional search space
    - Progressive complexity modes
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
        self.config = config or Optimizer()
        
        self.best_params_ = None
        self.best_score_ = -np.inf
        self.search_space_ = None
        self.history_ = []
        self.diagnostics_logger = None
        
        if self.config.log_diagnostics:
            self.diagnostics_logger = DiagnosticsLogger(self.config.diagnostics_file)
        
        # Для horizon-aware scoring
        self.naive_baseline_mae = {}
        
        # Для progressive modes
        self.current_mode_idx = 0
        self.mode_iterations = {
            "baseline": max(3, n_calls // 5),
            "structure": max(5, n_calls // 3),
            "full": n_calls
        }

    # =========================================================================
    # 1. OUT-OF-SAMPLE PENALTIES (OOF PREDICTIONS)
    # =========================================================================
    def _compute_oof_predictions(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        model: BaseEstimator
    ) -> np.ndarray:
        """
        Вычисление OOF предсказаний для честной оценки штрафов.
        """
        try:
            oof_pred = cross_val_predict(
                model, X.fillna(0), y, cv=self.cv, n_jobs=-1
            )
            return oof_pred
        except Exception as e:
            if self.verbose >= 2:
                print(f"OOF prediction failed: {e}. Using in-sample predictions.")
            model.fit(X.fillna(0), y)
            return model.predict(X.fillna(0))

    def _dominance_penalty(
        self, 
        model: BaseEstimator, 
        feature_names: List[str],
        X: pd.DataFrame = None,
        y: pd.Series = None
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Штраф за доминирование одного признака.
        
        Вычисляется на OOF предсказаниях если use_oof_penalties=True.
        Возвращает штраф и диагностические данные.
        """
        diagnostics = {
            "max_feature_share": 0.0,
            "top_feature": None,
            "top_3_share": 0.0
        }
        
        if not hasattr(model, "feature_importances_"):
            return 0.0, diagnostics
        
        importances = np.abs(model.feature_importances_)
        if len(importances) == 0 or importances.sum() == 0:
            return 0.0, diagnostics
        
        # Нормализуем важности
        importances = importances / importances.sum()
        max_share = importances.max()
        top_idx = np.argmax(importances)
        
        diagnostics["max_feature_share"] = float(max_share)
        diagnostics["top_feature"] = feature_names[top_idx] if top_idx < len(feature_names) else "unknown"
        diagnostics["top_3_share"] = float(np.sum(np.sort(importances)[-3:]))
        
        # Штраф если один признак доминирует
        if max_share > self.config.dominance_threshold:
            penalty = (max_share - self.config.dominance_threshold) * self.config.dominance_lambda
            return penalty, diagnostics
        return 0.0, diagnostics

    def _naive_similarity_penalty(
        self, 
        y_pred: np.ndarray, 
        y_lag1: np.ndarray,
        y_true: np.ndarray = None
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Штраф за слишком похожие на lag_1 прогнозы.
        
        Вычисляется на OOF предсказаниях если use_oof_penalties=True.
        """
        diagnostics = {
            "naive_corr": 0.0,
            "naive_mae_ratio": 1.0
        }
        
        # Удаляем NaN для корреляции
        mask = ~np.isnan(y_pred) & ~np.isnan(y_lag1)
        if mask.sum() < 10:
            return 0.0, diagnostics
        
        y_pred_clean = y_pred[mask]
        y_lag1_clean = y_lag1[mask]
        
        # Корреляция Пирсона
        if np.std(y_pred_clean) < 1e-6 or np.std(y_lag1_clean) < 1e-6:
            return 0.0, diagnostics
        
        corr = np.corrcoef(y_pred_clean, y_lag1_clean)[0, 1]
        if np.isnan(corr):
            return 0.0, diagnostics
        
        diagnostics["naive_corr"] = float(corr)
        
        # Дополнительно: отношение MAE модели к MAE naive
        if y_true is not None:
            y_true_clean = y_true[mask]
            model_mae = mean_absolute_error(y_true_clean, y_pred_clean)
            naive_mae = mean_absolute_error(y_true_clean, y_lag1_clean)
            if naive_mae > 0:
                diagnostics["naive_mae_ratio"] = float(model_mae / naive_mae)
        
        # Штраф если слишком похожи
        if corr > self.config.naive_corr_threshold:
            penalty = (corr - self.config.naive_corr_threshold) * self.config.naive_lambda
            return penalty, diagnostics
        return 0.0, diagnostics

    def _collapse_penalty(
        self, 
        X_transformed: pd.DataFrame,
        feature_names: List[str]
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Штраф за feature collapse (слишком мало уникальных признаков).
        """
        diagnostics = {
            "n_features": X_transformed.shape[1],
            "n_unique_patterns": 0,
            "effective_rank": 0
        }
        
        n_features = X_transformed.shape[1]
        diagnostics["n_features"] = n_features
        
        if n_features < 2:
            diagnostics["collapse_detected"] = True
            return self.config.collapse_lambda * 2.0, diagnostics
        
        # Вычисляем эффективный ранг через корреляционную матрицу
        try:
            corr_matrix = X_transformed.fillna(0).corr().values
            eigenvalues = np.linalg.eigvalsh(corr_matrix)
            effective_rank = np.sum(eigenvalues > 1e-6)
            diagnostics["effective_rank"] = int(effective_rank)
            diagnostics["n_unique_patterns"] = int(effective_rank)
            
            # Штраф если эффективный ранг слишком мал
            if effective_rank < n_features * 0.3:
                penalty = self.config.collapse_lambda * (1 - effective_rank / n_features)
                diagnostics["collapse_detected"] = True
                return penalty, diagnostics
        except:
            pass
        
        diagnostics["collapse_detected"] = False
        return 0.0, diagnostics

    # =========================================================================
    # 2. UNIFIED PENALTY SCALING
    # =========================================================================
    def _scale_penalty(self, penalty: float, mae: float) -> float:
        """
        Приведение штрафа к масштабу MAE.
        
        penalty_scaled = penalty * mae
        
        Это делает штрафы сопоставимыми на разных датасетах.
        """
        if not self.config.scale_penalties or mae == 0:
            return penalty
        return penalty * mae

    # =========================================================================
    # 3. SEMANTIC ENTROPY (BY INFORMATION TYPE)
    # =========================================================================
    def _semantic_feature_grouping(self, feature_names: List[str]) -> Dict[str, List[str]]:
        """
        Группировка признаков по семантическому типу, а не по имени трансформера.
        
        Возвращает словарь: {semantic_type: [feature_names]}
        """
        groups = defaultdict(list)
        
        for feat_name in feature_names:
            # Определяем семантический тип по паттернам в имени
            if "lag_" in feat_name:
                groups["lag"].append(feat_name)
            elif "window" in feat_name:
                if "mean" in feat_name or "std" in feat_name:
                    groups["rolling_stats"].append(feat_name)
                elif "slope" in feat_name or "trend" in feat_name:
                    groups["rolling_trend"].append(feat_name)
                elif "diff" in feat_name:
                    groups["rolling_diff"].append(feat_name)
                else:
                    groups["rolling_other"].append(feat_name)
            elif "stl" in feat_name:
                if "trend" in feat_name:
                    groups["stl_trend"].append(feat_name)
                elif "seasonal" in feat_name:
                    groups["stl_seasonal"].append(feat_name)
                else:
                    groups["stl_resid"].append(feat_name)
            elif "dwt" in feat_name:
                groups["wavelet"].append(feat_name)
            elif "time." in feat_name or "calendar" in feat_name:
                if "hour" in feat_name or "minute" in feat_name:
                    groups["time_intraday"].append(feat_name)
                elif "day_of_week" in feat_name or "weekend" in feat_name:
                    groups["time_weekly"].append(feat_name)
                elif "month" in feat_name or "season" in feat_name:
                    groups["time_monthly"].append(feat_name)
                else:
                    groups["time_other"].append(feat_name)
            else:
                groups["other"].append(feat_name)
        
        return dict(groups)

    def _feature_group_entropy(self, feature_names: List[str]) -> Tuple[float, Dict[str, Any]]:
        """
        Вычисление энтропии распределения признаков по семантическим группам.
        
        Возвращает энтропию и диагностические данные.
        """
        diagnostics = {
            "n_groups": 0,
            "group_distribution": {},
            "max_group_share": 0.0
        }
        
        if not feature_names:
            return 0.0, diagnostics
        
        # Группируем по семантическому типу
        groups = self._semantic_feature_grouping(feature_names)
        diagnostics["n_groups"] = len(groups)
        
        if len(groups) <= 1:
            diagnostics["group_distribution"] = {k: len(v) for k, v in groups.items()}
            return 0.0, diagnostics
        
        # Вычисляем энтропию Шеннона
        counts = np.array([len(v) for v in groups.values()])
        total = counts.sum()
        probs = counts / total
        
        diagnostics["group_distribution"] = {k: len(v) for k, v in groups.items()}
        diagnostics["max_group_share"] = float(np.max(probs))
        
        entropy = -np.sum(probs * np.log(probs + 1e-9))
        
        # Нормализуем к [0, 1]
        max_entropy = np.log(len(groups))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        
        return normalized_entropy, diagnostics

    # =========================================================================
    # 4. HORIZON-AWARE SCORING
    # =========================================================================
    def _compute_horizon_aware_gain(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model: BaseEstimator,
        horizon: int = 1
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Вычисление gain(h) = MAE_naive(h) - MAE_model(h).
        
        Положительный gain означает, что модель лучше naive на горизонте h.
        """
        diagnostics = {
            f"mae_naive_h{horizon}": 0.0,
            f"mae_model_h{horizon}": 0.0,
            f"gain_h{horizon}": 0.0
        }
        
        # Создаём target для горизонта h
        y_h = y.shift(-horizon).dropna()
        X_h = X.loc[y_h.index]
        y_h = y_h.loc[X_h.index]
        
        if len(y_h) < 20:
            diagnostics[f"gain_h{horizon}_insufficient_data"] = True
            return 0.0, diagnostics
        
        # Naive прогноз на горизонте h
        naive_pred = y_h.shift(horizon).fillna(y_h.mean())
        naive_mae = mean_absolute_error(y_h, naive_pred)
        diagnostics[f"mae_naive_h{horizon}"] = float(naive_mae)
        
        # Модель
        try:
            scores = cross_val_score(
                model, X_h.fillna(0), y_h, cv=self.cv, 
                scoring="neg_mean_absolute_error", n_jobs=-1
            )
            model_mae = -np.mean(scores)
        except:
            diagnostics[f"gain_h{horizon}_model_failed"] = True
            return 0.0, diagnostics
        
        diagnostics[f"mae_model_h{horizon}"] = float(model_mae)
        
        gain = naive_mae - model_mae
        diagnostics[f"gain_h{horizon}"] = float(gain)
        diagnostics[f"relative_gain_h{horizon}"] = float(gain / naive_mae) if naive_mae > 0 else 0.0
        
        return gain, diagnostics

    # =========================================================================
    # 5. RELATIVE SCORE VS NAIVE
    # =========================================================================
    def _compute_relative_score(
        self,
        mae_model: float,
        mae_naive: float
    ) -> float:
        """
        Вычисление относительного score: MAE_model / MAE_naive.
        
        Значения < 1.0 означают улучшение относительно naive.
        """
        if mae_naive == 0:
            return mae_model
        return mae_model / mae_naive

    # =========================================================================
    # 6. CONDITIONAL SEARCH SPACE
    # =========================================================================
    def _define_search_space(
        self, 
        meta_features: Optional[Dict[str, float]] = None,
        mode: str = "full"
    ) -> Space:
        """
        Определение пространства поиска с поддержкой conditional parameters.
        """
        from skopt.space import Space
        
        space = []
        
        # НОВЫЙ ПАРАМЕТР: режим пайплайна (для progressive modes)
        if self.config.use_progressive_modes:
            space.append(Categorical(
                list(self.config.pipeline_modes.keys()),
                name="pipeline_mode"
            ))
        
        # Определяем активные группы по режиму
        if mode in self.config.pipeline_modes:
            active_groups = self.config.pipeline_modes[mode]
        else:
            active_groups = self.config.pipeline_modes["full"]
        
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

    def _adapt_search_space(self, space: Space, meta_features: Dict[str, float]) -> None:
        """Адаптация пространства поиска на основе мета-признаков."""
        if "dominant_freq" in meta_features:
            dominant_freq = meta_features["dominant_freq"]
            if dominant_freq > 0.04:
                for dim in space:
                    if dim.name == "stl_period":
                        dim.low = 20
                        dim.high = 30
                        break

    def _build_pipeline(self, params: Dict[str, Any], mode: str = None) -> FeatureEngineeringPipeline:
        """
        Построение пайплайна с поддержкой pipeline modes.
        """
        transformers = []
        
        # Определяем активные группы по режиму
        if mode is None:
            mode = params.get("pipeline_mode", "full")
        
        if mode in self.config.pipeline_modes:
            active_groups = self.config.pipeline_modes[mode]
        else:
            active_groups = self.config.pipeline_modes["full"]
        
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
            else:
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
        
        return FeatureEngineeringPipeline(transformers)

    # =========================================================================
    # 7. PROGRESSIVE COMPLEXITY MODES
    # =========================================================================
    def _get_current_mode(self, iteration: int) -> str:
        """
        Определение текущего режима для progressive complexity.
        """
        if not self.config.use_progressive_modes:
            return "full"
        
        cumulative = 0
        for mode, max_iter in self.mode_iterations.items():
            cumulative += max_iter
            if iteration < cumulative:
                return mode
        return "full"

    # =========================================================================
    # MAIN OBJECTIVE FUNCTION (MULTI-OBJECTIVE)
    # =========================================================================
    def _objective(self, X: pd.DataFrame, y: pd.Series, **params) -> float:
        """
        Целевая функция с multi-objective поддержкой.
        
        Формула:
            final_score = (MAE + penalties) / MAE_naive - entropy_bonus
        
        Все штрафы считаются на OOF predictions если use_oof_penalties=True.
        """
        # Определяем текущий режим для progressive modes
        current_iteration = len(self.history_)
        mode = self._get_current_mode(current_iteration)
        params["pipeline_mode"] = mode
        
        pipeline = self._build_pipeline(params, mode)
        
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
            feature_names = list(X_transformed.columns)
            
            # ОЦЕНКА КАЧЕСТВА (базовый MAE)
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
            
            # Конвертируем к MAE для штрафов
            if self.metric in ["neg_mean_absolute_error"]:
                mae = -mean_score
            elif self.metric in ["neg_mean_squared_error", "neg_root_mean_squared_error"]:
                mae = np.sqrt(-mean_score) if mean_score < 0 else np.abs(mean_score)
            else:
                mae = np.abs(mean_score)
            
            # ВЫЧИСЛЕНИЕ НАИВНОГО БАЗЛАЙНА
            lag1_col = [c for c in feature_names if "lag_1" in c and "core_lags" in c]
            if lag1_col and lag1_col[0] in X_transformed.columns:
                y_lag1 = X_transformed[lag1_col[0]].values
            else:
                y_lag1 = y.shift(1).fillna(y.mean()).values
            
            naive_mae = mean_absolute_error(y, y_lag1)
            if naive_mae > 0:
                self.naive_baseline_mae[1] = naive_mae
            
            # ДОПОЛНИТЕЛЬНЫЕ МЕТРИКИ КАЧЕСТВА FE
            
            # 1. Dominance penalty (OOF)
            dominance_penalty = 0.0
            dominance_diagnostics = {}
            if self.config.use_oof_penalties:
                oof_pred = self._compute_oof_predictions(X_transformed, y, self.model)
                dominance_penalty, dominance_diagnostics = self._dominance_penalty(
                    self.model, feature_names, X_transformed, y
                )
            else:
                self.model.fit(X_transformed.fillna(0), y)
                dominance_penalty, dominance_diagnostics = self._dominance_penalty(
                    self.model, feature_names
                )
            
            # 2. Naive similarity penalty (OOF)
            naive_penalty = 0.0
            naive_diagnostics = {}
            if self.config.use_oof_penalties:
                oof_pred = self._compute_oof_predictions(X_transformed, y, self.model)
                naive_penalty, naive_diagnostics = self._naive_similarity_penalty(
                    oof_pred, y_lag1, y.values
                )
            else:
                naive_penalty, naive_diagnostics = self._naive_similarity_penalty(
                    self.model.predict(X_transformed.fillna(0)), y_lag1, y.values
                )
            
            # 3. Collapse penalty
            collapse_penalty, collapse_diagnostics = self._collapse_penalty(
                X_transformed, feature_names
            )
            
            # 4. Entropy bonus (semantic)
            entropy, entropy_diagnostics = self._feature_group_entropy(feature_names)
            entropy_bonus = self.config.entropy_lambda * entropy
            
            # 5. Horizon-aware gain (опционально)
            horizon_gain = 0.0
            horizon_diagnostics = {}
            if self.config.use_horizon_aware:
                for h in self.config.forecast_horizons[:2]:  # Ограничиваем для скорости
                    gain, h_diag = self._compute_horizon_aware_gain(
                        X_transformed, y, self.model, horizon=h
                    )
                    horizon_gain += gain
                    horizon_diagnostics.update(h_diag)
            
            # МАСШТАБИРОВАНИЕ ШТРАФОВ
            if self.config.scale_penalties and mae > 0:
                dominance_penalty = self._scale_penalty(dominance_penalty, mae)
                naive_penalty = self._scale_penalty(naive_penalty, mae)
                collapse_penalty = self._scale_penalty(collapse_penalty, mae)
            
            # EARLY ABORT CHECK
            total_penalty = dominance_penalty + naive_penalty + collapse_penalty
            if total_penalty > self.config.early_abort_penalty:
                if self.config.log_diagnostics and self.diagnostics_logger:
                    self.diagnostics_logger.log(current_iteration, {
                        "mae": float(mae),
                        "dominance_penalty": float(dominance_penalty),
                        "naive_penalty": float(naive_penalty),
                        "collapse_penalty": float(collapse_penalty),
                        "entropy_bonus": float(entropy_bonus),
                        "early_abort": True,
                        **dominance_diagnostics,
                        **naive_diagnostics,
                        **collapse_diagnostics,
                        **entropy_diagnostics
                    })
                return 1e5
            
            # ФИНАЛЬНАЯ ФОРМУЛА
            # Для метрик где больше = лучше (R², neg_MAE):
            if self.metric in ["r2", "neg_mean_absolute_error", "neg_mean_squared_error", "neg_root_mean_squared_error"]:
                base_score = -mean_score
            else:
                base_score = mean_score
            
            # Relative score vs naive
            if naive_mae > 0:
                relative_score = base_score / naive_mae
            else:
                relative_score = base_score
            
            final_score = (
                relative_score
                + dominance_penalty
                + naive_penalty
                + collapse_penalty
                - entropy_bonus
                - (horizon_gain / 100.0 if horizon_gain > 0 else 0)  # Нормализуем gain
            )
            
            # Сохраняем историю с расширенными метриками
            self.history_.append({
                "params": params.copy(),
                "base_score": float(mean_score),
                "mae": float(mae),
                "naive_mae": float(naive_mae),
                "relative_score": float(relative_score),
                "dominance_penalty": float(dominance_penalty),
                "naive_penalty": float(naive_penalty),
                "collapse_penalty": float(collapse_penalty),
                "entropy_bonus": float(entropy_bonus),
                "horizon_gain": float(horizon_gain),
                "final_score": float(final_score),
                "n_features": X_transformed.shape[1],
                "mode": mode,
                **dominance_diagnostics,
                **naive_diagnostics,
                **collapse_diagnostics,
                **entropy_diagnostics,
                **horizon_diagnostics
            })
            
            # Логирование диагностики
            if self.config.log_diagnostics and self.diagnostics_logger:
                self.diagnostics_logger.log(current_iteration, {
                    "mae": float(mae),
                    "naive_mae": float(naive_mae),
                    "relative_score": float(relative_score),
                    "dominance_penalty": float(dominance_penalty),
                    "naive_penalty": float(naive_penalty),
                    "collapse_penalty": float(collapse_penalty),
                    "entropy_bonus": float(entropy_bonus),
                    "horizon_gain": float(horizon_gain),
                    "final_score": float(final_score),
                    "n_features": X_transformed.shape[1],
                    "mode": mode,
                    **dominance_diagnostics,
                    **naive_diagnostics,
                    **collapse_diagnostics,
                    **entropy_diagnostics
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
            if self.config.use_progressive_modes:
                print(f"  Режимы: {list(self.config.pipeline_modes.keys())}")
            print(f"  Штрафы: dominance={self.config.dominance_lambda}, naive={self.config.naive_lambda}")
            if self.config.scale_penalties:
                print("  Масштабирование штрафов: включено")
            if self.config.use_oof_penalties:
                print("  OOF penalties: включено")

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
            
            # Вывод диагностики
            if self.history_:
                last_record = self.history_[-1]
                print(f"\nДиагностика последнего trial:")
                print(f"  Dominance penalty: {last_record.get('dominance_penalty', 0):.4f}")
                print(f"  Naive penalty: {last_record.get('naive_penalty', 0):.4f}")
                print(f"  Entropy bonus: {last_record.get('entropy_bonus', 0):.4f}")
                print(f"  Max feature share: {last_record.get('max_feature_share', 0):.2%}")
                print(f"  Naive correlation: {last_record.get('naive_corr', 0):.3f}")

        return best_pipeline, self.best_params_, self.best_score_

    def get_search_history(self) -> pd.DataFrame:
        """Получение истории поиска в виде DataFrame."""
        if not self.history_:
            return pd.DataFrame()
        return pd.DataFrame(self.history_)

    def get_diagnostics_summary(self) -> pd.DataFrame:
        """Получение сводки по диагностике."""
        if self.diagnostics_logger:
            return self.diagnostics_logger.get_summary()
        return pd.DataFrame()

    def suggest_initial_points(self, meta_features: Dict[str, float], n_points: int = 5) -> List[Dict[str, Any]]:
        """Генерация разумных начальных точек на основе мета-признаков."""
        points = []
        for i in range(n_points):
            point = {}
            
            # ВЫБИРАЕМ РЕЖИМ ПЛАЙПЛАЙНА
            if self.config.use_progressive_modes:
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