# src/ts_feature_eng/selection.py

"""
Модули отбора признаков для временных рядов.

Реализует многоступенчатый подход к отбору признаков:
1. Фильтрация по дисперсии (удаление константных/почти константных признаков)
2. Фильтрация по пропускам (удаление признаков с высоким % пропусков)
3. Отбор на основе важности признаков (SHAP-значения)

Все селекторы совместимы с интерфейсом scikit-learn и могут использоваться
в пайплайнах как самостоятельные компоненты или в комбинации.
"""

from typing import List, Optional, Union, Dict, Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.feature_selection import VarianceThreshold as SklearnVarianceThreshold, SelectKBest, f_regression, mutual_info_regression

from .base import FeatureSelector, TimeSeriesError


class VarianceThresholdSelector(FeatureSelector):
    """
    Селектор признаков на основе порога дисперсии.
    
    Удаляет признаки с дисперсией ниже заданного порога. Полезен для
    удаления константных или почти константных признаков, которые не
    несут информативности для модели.
    
    Параметры
    ----------
    threshold : float, по умолчанию 0.0
        Минимальная дисперсия для сохранения признака. Признаки с дисперсией
        ниже порога будут удалены.
    skipna : bool, по умолчанию True
        Игнорировать пропуски при вычислении дисперсии. Если False, признаки
        с пропусками будут иметь дисперсию = NaN и удалены.
    
    Атрибуты
    ----------
    selected_features_ : List[str]
        Список отобранных признаков.
    feature_importances_ : np.ndarray или None
        Важность признаков (не используется в этом селекторе, всегда None).
    variance_ : pd.Series
        Дисперсия каждого признака до фильтрации.
    is_fitted_ : bool
        Флаг, указывающий, был ли вызван метод fit().
    
    Примеры
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from ts_feature_eng.selection import VarianceThresholdSelector
    >>> 
    >>> # Создаем данные с константным признаком
    >>> X = pd.DataFrame({
    ...     "feature1": np.random.randn(100),
    ...     "feature2": np.random.randn(100),
    ...     "constant": np.ones(100)  # Константный признак
    ... })
    >>> y = pd.Series(np.random.randn(100))
    >>> 
    >>> selector = VarianceThresholdSelector(threshold=0.01)
    >>> X_selected = selector.fit_transform(X, y)
    >>> 
    >>> print(X_selected.columns.tolist())
    ['feature1', 'feature2']  # 'constant' удален
    """
    
    def __init__(self, threshold: float = 0.0, skipna: bool = True):
        super().__init__()
        self.threshold = threshold
        self.skipna = skipna
    
    def fit(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]) -> "VarianceThresholdSelector":
        """
        Вычисление дисперсии признаков и определение отобранных признаков.
        
        Параметры
        ----------
        X : pd.DataFrame
            DataFrame с признаками.
        y : pd.Series или np.ndarray
            Целевая переменная (игнорируется, требуется для совместимости).
        
        Возвращает
        ----------
        self : VarianceThresholdSelector
            Обученный селектор.
        """
        X, y = self._validate_input(X, y)
        
        # Вычисление дисперсии с обработкой пропусков
        if self.skipna:
            self.variance_ = X.var(skipna=True)
        else:
            # Признаки с пропусками получат дисперсию = NaN
            self.variance_ = X.var(skipna=False)
        
        # Отбор признаков с дисперсией выше порога
        mask = self.variance_ > self.threshold
        self.selected_features_ = X.columns[mask].tolist()
        
        # Защита от полного удаления всех признаков
        if len(self.selected_features_) == 0:
            # Сохраняем признак с максимальной дисперсией как минимум
            max_var_feature = self.variance_.idxmax()
            self.selected_features_ = [max_var_feature]
            if hasattr(self, "logger"):
                self.logger.warning(
                    f"All features filtered by variance threshold {self.threshold}. "
                    f"Keeping feature with max variance: {max_var_feature}"
                )
        
        self.feature_importances_ = None
        self.is_fitted_ = True
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Применение отбора признаков к данным.
        
        Параметры
        ----------
        X : pd.DataFrame
            DataFrame с признаками.
        
        Возвращает
        ----------
        X_transformed : pd.DataFrame
            DataFrame только с отобранными признаками.
        """
        if not self.is_fitted_:
            raise TimeSeriesError("Selector is not fitted. Call fit() first.")
        
        # Проверка наличия всех отобранных признаков в X
        missing_features = set(self.selected_features_) - set(X.columns)
        if missing_features:
            raise TimeSeriesError(
                f"Input DataFrame is missing features selected during fit: {missing_features}"
            )
        
        return X[self.selected_features_]
    
    def get_selected_features(self) -> List[str]:
        """
        Получение списка отобранных признаков.
        
        Возвращает
        ----------
        selected_features : List[str]
            Список имен отобранных признаков.
        """
        if not self.is_fitted_:
            raise TimeSeriesError("Selector is not fitted. Call fit() first.")
        
        return self.selected_features_


class MissingValueSelector(FeatureSelector):
    """
    Селектор признаков на основе допустимого уровня пропусков.
    
    Удаляет признаки, в которых доля пропусков превышает заданный порог.
    Полезен для очистки признакового пространства от ненадежных признаков.
    
    Параметры
    ----------
    threshold : float, по умолчанию 0.2
        Максимально допустимая доля пропусков (от 0.0 до 1.0). Признаки с долей
        пропусков выше порога будут удалены.
    
    Атрибуты
    ----------
    selected_features_ : List[str]
        Список отобранных признаков.
    feature_importances_ : np.ndarray или None
        Важность признаков (не используется в этом селекторе, всегда None).
    missing_ratio_ : pd.Series
        Доля пропусков для каждого признака.
    is_fitted_ : bool
        Флаг, указывающий, был ли вызван метод fit().
    
    Примеры
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from ts_feature_eng.selection import MissingValueSelector
    >>> 
    >>> # Создаем данные с признаком, содержащим много пропусков
    >>> X = pd.DataFrame({
    ...     "feature1": np.random.randn(100),
    ...     "feature2": np.random.randn(100),
    ...     "noisy": np.random.randn(100)
    ... })
    >>> X.loc[:70, "noisy"] = np.nan  # 70% пропусков
    >>> y = pd.Series(np.random.randn(100))
    >>> 
    >>> selector = MissingValueSelector(threshold=0.5)
    >>> X_selected = selector.fit_transform(X, y)
    >>> 
    >>> print(X_selected.columns.tolist())
    ['feature1', 'feature2']  # 'noisy' удален (70% > 50%)
    """
    
    def __init__(self, threshold: float = 0.2):
        super().__init__()
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be between 0.0 and 1.0, got {threshold}")
        self.threshold = threshold
    
    def fit(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]) -> "MissingValueSelector":
        """
        Вычисление доли пропусков и определение отобранных признаков.
        
        Параметры
        ----------
        X : pd.DataFrame
            DataFrame с признаками.
        y : pd.Series или np.ndarray
            Целевая переменная (игнорируется).
        
        Возвращает
        ----------
        self : MissingValueSelector
            Обученный селектор.
        """
        X, y = self._validate_input(X, y)
        
        # Вычисление доли пропусков для каждого признака
        self.missing_ratio_ = X.isna().mean()
        
        # Отбор признаков с долей пропусков ниже порога
        mask = self.missing_ratio_ <= self.threshold
        self.selected_features_ = X.columns[mask].tolist()
        
        # Защита от полного удаления всех признаков
        if len(self.selected_features_) == 0:
            # Сохраняем признак с минимальной долей пропусков
            min_missing_feature = self.missing_ratio_.idxmin()
            self.selected_features_ = [min_missing_feature]
            if hasattr(self, "logger"):
                self.logger.warning(
                    f"All features filtered by missing value threshold {self.threshold}. "
                    f"Keeping feature with min missing ratio: {min_missing_feature}"
                )
        
        self.feature_importances_ = None
        self.is_fitted_ = True
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Применение отбора признаков к данным.
        
        Параметры
        ----------
        X : pd.DataFrame
            DataFrame с признаками.
        
        Возвращает
        ----------
        X_transformed : pd.DataFrame
            DataFrame только с отобранными признаками.
        """
        if not self.is_fitted_:
            raise TimeSeriesError("Selector is not fitted. Call fit() first.")
        
        missing_features = set(self.selected_features_) - set(X.columns)
        if missing_features:
            raise TimeSeriesError(
                f"Input DataFrame is missing features selected during fit: {missing_features}"
            )
        
        return X[self.selected_features_]
    
    def get_selected_features(self) -> List[str]:
        """
        Получение списка отобранных признаков.
        
        Возвращает
        ----------
        selected_features : List[str]
            Список имен отобранных признаков.
        """
        if not self.is_fitted_:
            raise TimeSeriesError("Selector is not fitted. Call fit() first.")
        
        return self.selected_features_


class SHAPFeatureSelector(FeatureSelector):
    """
    Селектор признаков на основе SHAP-значений (SHapley Additive exPlanations).
    
    Оценивает важность признаков через их вклад в предсказания модели и отбирает
    наиболее важные признаки. Поддерживает различные типы моделей:
    - Линейные модели (через LinearExplainer)
    - Деревья и ансамбли (через TreeExplainer)
    - Произвольные модели (через KernelExplainer или PermutationExplainer)
    
    Параметры
    ----------
    model : object, опционально
        Модель для оценки важности признаков. Если не указана, используется
        Ridge регрессия по умолчанию.
    n_features : int или float, опционально
        Количество или доля признаков для отбора:
        - Если int: абсолютное количество признаков
        - Если float: доля от общего числа признаков (от 0.0 до 1.0)
        - Если None: используется порог по важности (threshold)
    threshold : float, по умолчанию 'median'
        Порог важности для отбора признаков:
        - 'median': отбор признаков с важностью выше медианы
        - 'mean': отбор признаков с важностью выше среднего
        - Число: абсолютный порог важности
    explainer_type : str, по умолчанию 'auto'
        Тип SHAP-эксплейнера:
        - 'auto': автоматический выбор на основе типа модели
        - 'tree': TreeExplainer для моделей на основе деревьев
        - 'linear': LinearExplainer для линейных моделей
        - 'kernel': KernelExplainer для произвольных моделей (медленнее)
        - 'permutation': PermutationExplainer (требует shap >= 0.40)
    max_samples : int, по умолчанию 1000
        Максимальное количество наблюдений для вычисления SHAP-значений
        (для ускорения на больших данных).
    random_state : int, опционально
        Фиксация случайного состояния для воспроизводимости.
    
    Атрибуты
    ----------
    selected_features_ : List[str]
        Список отобранных признаков.
    feature_importances_ : np.ndarray
        SHAP-значения важности признаков (средняя абсолютная важность).
    shap_values_ : np.ndarray или объект SHAP
        Полные SHAP-значения для всех наблюдений и признаков.
    is_fitted_ : bool
        Флаг, указывающий, был ли вызван метод fit().
    
    Примеры
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> from ts_feature_eng.selection import SHAPFeatureSelector
    >>> 
    >>> # Создаем синтетические данные
    >>> X = pd.DataFrame(np.random.randn(1000, 20), columns=[f"f{i}" for i in range(20)])
    >>> y = X["f0"] * 2 + X["f1"] * 1.5 + np.random.randn(1000) * 0.1  # f0 и f1 наиболее важны
    >>> 
    >>> # Отбор 5 наиболее важных признаков
    >>> selector = SHAPFeatureSelector(
    ...     model=RandomForestRegressor(n_estimators=50, random_state=42),
    ...     n_features=5,
    ...     explainer_type="tree"
    ... )
    >>> X_selected = selector.fit_transform(X, y)
    >>> 
    >>> print(X_selected.columns.tolist())
    ['f0', 'f1', 'f2', 'f3', 'f4']  # f0 и f1 должны быть в топе
    """
    
    def __init__(
        self,
        model: Optional[object] = None,
        n_features: Optional[Union[int, float]] = None,
        threshold: Union[str, float] = "median",
        explainer_type: str = "auto",
        max_samples: int = 1000,
        random_state: Optional[int] = 42,
    ):
        super().__init__()
        self.model = model
        self.n_features = n_features
        self.threshold = threshold
        self.explainer_type = explainer_type
        self.max_samples = max_samples
        self.random_state = random_state
    
    def fit(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]) -> "SHAPFeatureSelector":
        """
        Обучение модели и вычисление SHAP-значений для отбора признаков.
        
        Параметры
        ----------
        X : pd.DataFrame
            DataFrame с признаками.
        y : pd.Series или np.ndarray
            Целевая переменная.
        
        Возвращает
        ----------
        self : SHAPFeatureSelector
            Обученный селектор.
        """
        X, y = self._validate_input(X, y)
        
        # Импорт SHAP внутри метода для отложенной загрузки
        try:
            import shap
        except ImportError:
            raise ImportError(
                "SHAP library is required for SHAPFeatureSelector. "
                "Install it with: pip install shap>=0.44.0"
            )
        
        # Создание модели по умолчанию при необходимости
        if self.model is None:
            from sklearn.linear_model import Ridge
            self.model_ = Ridge(alpha=1.0, random_state=self.random_state)
        else:
            # Клонирование модели для избежания изменения оригинала
            self.model_ = clone(self.model)
        
        # Ограничение размера выборки для ускорения
        if len(X) > self.max_samples:
            np.random.seed(self.random_state)
            sample_idx = np.random.choice(len(X), self.max_samples, replace=False)
            X_sample = X.iloc[sample_idx]
            y_sample = y[sample_idx] if isinstance(y, np.ndarray) else y.iloc[sample_idx]
        else:
            X_sample = X
            y_sample = y
        
        # Обучение модели
        self.model_.fit(X_sample, y_sample)
        
        # Выбор эксплейнера на основе типа модели
        explainer = self._get_explainer(self.model_, X_sample)
        
        # Вычисление SHAP-значений
        try:
            self.shap_values_ = explainer(X_sample)
            
            # Для мультиклассовой классификации shap_values может быть списком
            if isinstance(self.shap_values_, list):
                # Берем SHAP-значения для первого класса
                shap_values_abs = np.abs(self.shap_values_[0].values).mean(axis=0)
            else:
                shap_values_abs = np.abs(self.shap_values_.values).mean(axis=0)
            
            # Сохранение важности признаков
            self.feature_importances_ = shap_values_abs
            
            # Отбор признаков
            self.selected_features_ = self._select_features(
                X_sample.columns.tolist(),
                shap_values_abs
            )
        
        except Exception as e:
            # Резервный метод: пермутационная важность при ошибке SHAP
            if hasattr(self, "logger"):
                self.logger.warning(
                    f"SHAP computation failed: {str(e)}. Falling back to permutation importance."
                )
            self.selected_features_ = self._fallback_selection(X_sample, y_sample)
            self.feature_importances_ = None
            self.shap_values_ = None
        
        # Защита от полного удаления всех признаков
        if len(self.selected_features_) == 0:
            # Сохраняем топ-5 признаков по умолчанию
            if self.feature_importances_ is not None:
                top_indices = np.argsort(self.feature_importances_)[-5:][::-1]
                self.selected_features_ = [X.columns[i] for i in top_indices]
            else:
                self.selected_features_ = X.columns[:5].tolist()
            
            if hasattr(self, "logger"):
                self.logger.warning(
                    "All features filtered by SHAP selector. Keeping top 5 features by default."
                )
        
        self.is_fitted_ = True
        return self
    
    def _get_explainer(self, model: object, X_sample: pd.DataFrame):
        """Выбор подходящего SHAP-эксплейнера на основе типа модели."""
        import shap
        
        explainer_type = self.explainer_type
        
        # Автоматическое определение типа эксплейнера
        if explainer_type == "auto":
            model_type = type(model).__module__ + "." + type(model).__name__
            
            # Деревья и ансамбли на основе деревьев
            tree_models = [
                "sklearn.ensemble.RandomForestRegressor",
                "sklearn.ensemble.RandomForestClassifier",
                "sklearn.ensemble.GradientBoostingRegressor",
                "sklearn.ensemble.GradientBoostingClassifier",
                "sklearn.ensemble.ExtraTreesRegressor",
                "sklearn.ensemble.ExtraTreesClassifier",
                "sklearn.tree.DecisionTreeRegressor",
                "sklearn.tree.DecisionTreeClassifier",
                "xgboost.XGBRegressor",
                "xgboost.XGBClassifier",
                "lightgbm.LGBMRegressor",
                "lightgbm.LGBMClassifier",
                "catboost.CatBoostRegressor",
                "catboost.CatBoostClassifier",
            ]
            
            if any(m in model_type for m in tree_models):
                explainer_type = "tree"
            # Линейные модели
            elif "sklearn.linear_model" in model_type:
                explainer_type = "linear"
            else:
                explainer_type = "kernel"
        
        # Создание эксплейнера
        if explainer_type == "tree":
            return shap.TreeExplainer(model)
        elif explainer_type == "linear":
            return shap.LinearExplainer(model, X_sample)
        elif explainer_type == "kernel":
            return shap.KernelExplainer(model.predict, shap.sample(X_sample, 100, random_state=self.random_state))
        elif explainer_type == "permutation":
            if hasattr(shap, "PermutationExplainer"):
                return shap.PermutationExplainer(model.predict, X_sample)
            else:
                raise ValueError("PermutationExplainer requires shap >= 0.40.0")
        else:
            raise ValueError(f"Unknown explainer_type: {explainer_type}")
    
    def _select_features(self, feature_names: List[str], importances: np.ndarray) -> List[str]:
        """
        Отбор признаков на основе их важности.
        
        Параметры
        ----------
        feature_names : List[str]
            Имена признаков.
        importances : np.ndarray
            Важность каждого признака.
        
        Возвращает
        ----------
        selected : List[str]
            Список отобранных признаков.
        """
        if self.n_features is not None:
            # Отбор по количеству признаков
            if isinstance(self.n_features, float):
                if not 0.0 < self.n_features <= 1.0:
                    raise ValueError("n_features as float must be in (0, 1]")
                n_select = int(len(feature_names) * self.n_features)
                n_select = max(1, min(n_select, len(feature_names)))
            else:  # int
                n_select = min(self.n_features, len(feature_names))
                n_select = max(1, n_select)
            
            # Сортировка по важности и выбор топ-n
            indices = np.argsort(importances)[-n_select:][::-1]
            return [feature_names[i] for i in indices]
        
        else:
            # Отбор по порогу важности
            if self.threshold == "median":
                threshold_value = np.median(importances)
            elif self.threshold == "mean":
                threshold_value = np.mean(importances)
            elif isinstance(self.threshold, (int, float)):
                threshold_value = self.threshold
            else:
                raise ValueError(
                    f"threshold must be 'median', 'mean' or a number, got {self.threshold}"
                )
            
            # Отбор признаков выше порога
            mask = importances >= threshold_value
            selected = [feature_names[i] for i in range(len(feature_names)) if mask[i]]
            
            # Гарантируем минимум 1 признак
            if len(selected) == 0:
                max_idx = np.argmax(importances)
                selected = [feature_names[max_idx]]
            
            return selected
    
    def _fallback_selection(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]) -> List[str]:
        """
        Резервный метод отбора признаков через пермутационную важность.
        
        Используется при ошибке вычисления SHAP-значений.
        
        Параметры
        ----------
        X : pd.DataFrame
            Признаки.
        y : pd.Series или np.ndarray
            Целевая переменная.
        
        Возвращает
        ----------
        selected_features : List[str]
            Список отобранных признаков.
        """
        from sklearn.inspection import permutation_importance
        
        # Вычисление пермутационной важности
        result = permutation_importance(
            self.model_,
            X,
            y,
            n_repeats=5,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        # Отбор топ-50% признаков по важности
        n_select = max(1, len(X.columns) // 2)
        indices = np.argsort(result.importances_mean)[-n_select:][::-1]
        return X.columns[indices].tolist()
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Применение отбора признаков к данным.
        
        Параметры
        ----------
        X : pd.DataFrame
            DataFrame с признаками.
        
        Возвращает
        ----------
        X_transformed : pd.DataFrame
            DataFrame только с отобранными признаками.
        """
        if not self.is_fitted_:
            raise TimeSeriesError("Selector is not fitted. Call fit() first.")
        
        missing_features = set(self.selected_features_) - set(X.columns)
        if missing_features:
            raise TimeSeriesError(
                f"Input DataFrame is missing features selected during fit: {missing_features}"
            )
        
        return X[self.selected_features_]
    
    def get_selected_features(self) -> List[str]:
        """
        Получение списка отобранных признаков.
        
        Возвращает
        ----------
        selected_features : List[str]
            Список имен отобранных признаков.
        """
        if not self.is_fitted_:
            raise TimeSeriesError("Selector is not fitted. Call fit() first.")
        
        return self.selected_features_
    
    def get_feature_importance_df(self) -> pd.DataFrame:
        """
        Получение DataFrame с важностью признаков.
        
        Возвращает
        ----------
        importance_df : pd.DataFrame
            DataFrame с колонками ['feature', 'importance', 'rank'].
        """
        if not self.is_fitted_ or self.feature_importances_ is None:
            raise TimeSeriesError(
                "Feature importances not available. Call fit() first or SHAP computation failed."
            )
        
        importance_df = pd.DataFrame({
            "feature": self.feature_names_in_,
            "importance": self.feature_importances_
        })
        
        importance_df = importance_df.sort_values("importance", ascending=False)
        importance_df["rank"] = range(1, len(importance_df) + 1)
        
        return importance_df.reset_index(drop=True)


class HybridFeatureSelector(FeatureSelector):
    """
    Гибридный селектор признаков, комбинирующий несколько стратегий отбора.
    
    Поддерживает следующие стратегии:
    - "union": объединение признаков из всех методов
    - "intersection": пересечение признаков из всех методов
    - "staged": последовательное применение методов (фильтрация → модель → SHAP)
    
    Параметры
    ----------
    methods : List[str], по умолчанию ["pearson", "distance_corr"]
        Список методов для гибридизации:
        - "pearson": корреляция Пирсона
        - "f_regression": F-статистика ANOVA  
        - "mutual_info": взаимная информация
        - "distance_corr": корреляция расстояний
    strategy : str, по умолчанию "union"
        Стратегия комбинирования:
        - "union": объединение всех признаков
        - "intersection": пересечение признаков
        - "staged": многоступенчатый отбор
    top_k : int, по умолчанию 20
        Количество признаков для каждого метода.
    model : object, опционально
        Модель для staged-стратегии (по умолчанию Ridge).
    random_state : int, опционально
        Фиксация случайного состояния.
    
    Атрибуты
    ----------
    selected_features_ : List[str]
        Список отобранных признаков.
    feature_importances_ : Dict[str, List[float]]
        Важность признаков по каждому методу.
    is_fitted_ : bool
        Флаг, указывающий, был ли вызван метод fit().
    
    Примеры
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from ts_feature_eng.selection import HybridFeatureSelector
    >>> 
    >>> X = pd.DataFrame(np.random.randn(1000, 20), columns=[f"f{i}" for i in range(20)])
    >>> y = X["f0"] * 2 + X["f1"] * 1.5 + np.random.randn(1000) * 0.1
    >>> 
    >>> selector = HybridFeatureSelector(
    ...     methods=["pearson", "distance_corr"],
    ...     strategy="union",
    ...     top_k=10
    ... )
    >>> X_selected = selector.fit_transform(X, y)
    >>> 
    >>> print(f"Отобрано признаков: {len(X_selected.columns)}")
    """
    
    def __init__(
        self,
        methods: List[str] = None,
        strategy: str = "union",
        top_k: int = 20,
        model: Optional[object] = None,
        random_state: Optional[int] = 42,
    ):
        super().__init__()
        self.methods = methods or ["pearson", "distance_corr"]
        self.strategy = strategy
        self.top_k = top_k
        self.model = model
        self.random_state = random_state
        
        valid_strategies = {"union", "intersection", "staged"}
        if strategy not in valid_strategies:
            raise ValueError(f"strategy must be one of {valid_strategies}, got {strategy}")
        
        valid_methods = {"pearson", "f_regression", "mutual_info", "distance_corr"}
        invalid_methods = set(self.methods) - valid_methods
        if invalid_methods:
            raise ValueError(f"Invalid methods: {invalid_methods}. Valid options: {valid_methods}")
    
    def fit(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]) -> "HybridFeatureSelector":
        """
        Применение гибридной стратегии отбора признаков.
        
        Параметры
        ----------
        X : pd.DataFrame
            DataFrame с признаками.
        y : pd.Series или np.ndarray
            Целевая переменная.
        
        Возвращает
        ----------
        self : HybridFeatureSelector
            Обученный селектор.
        """
        X, y = self._validate_input(X, y)
        
        # Вычисление признаков по каждому методу
        self.feature_sets_ = {}
        self.feature_importances_ = {}
        
        for method in self.methods:
            if method == "pearson":
                correlations = X.corrwith(y).abs()
                top_features = correlations.nlargest(self.top_k).index.tolist()
                self.feature_sets_[method] = top_features
                self.feature_importances_[method] = correlations[top_features].tolist()
                
            elif method == "f_regression":
                selector = SelectKBest(score_func=f_regression, k=self.top_k)
                selector.fit(X.fillna(0), y)
                selected_mask = selector.get_support()
                self.feature_sets_[method] = X.columns[selected_mask].tolist()
                self.feature_importances_[method] = selector.scores_[selected_mask].tolist()
                
            elif method == "mutual_info":
                mi_scores = mutual_info_regression(X.fillna(0), y, random_state=self.random_state)
                top_indices = np.argsort(mi_scores)[-self.top_k:][::-1]
                self.feature_sets_[method] = X.columns[top_indices].tolist()
                self.feature_importances_[method] = mi_scores[top_indices].tolist()
                
            elif method == "distance_corr":
                try:
                    from dcor import distance_correlation
                    correlations = {}
                    for col in X.columns:
                        corr = distance_correlation(X[col].fillna(0).values, y.values)
                        correlations[col] = corr
                    sorted_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
                    self.feature_sets_[method] = [feat for feat, _ in sorted_features[:self.top_k]]
                    self.feature_importances_[method] = [corr for _, corr in sorted_features[:self.top_k]]
                except ImportError:
                    # Резервный метод: используем Pearson вместо distance correlation
                    correlations = X.corrwith(y).abs()
                    top_features = correlations.nlargest(self.top_k).index.tolist()
                    self.feature_sets_[method] = top_features
                    self.feature_importances_[method] = correlations[top_features].tolist()
                    if hasattr(self, "logger"):
                        self.logger.warning("dcor not installed. Using Pearson correlation as fallback for distance_corr.")
        
        # Применение стратегии комбинирования
        if self.strategy == "union":
            all_features = set()
            for features in self.feature_sets_.values():
                all_features.update(features)
            self.selected_features_ = list(all_features)
            
        elif self.strategy == "intersection":
            if len(self.feature_sets_) == 1:
                self.selected_features_ = self.feature_sets_[list(self.feature_sets_.keys())[0]]
            else:
                common_features = set(self.feature_sets_[self.methods[0]])
                for method in self.methods[1:]:
                    common_features = common_features.intersection(set(self.feature_sets_[method]))
                self.selected_features_ = list(common_features)
                
        elif self.strategy == "staged":
            # Этап 1: фильтрация (Pearson или Distance Corr)
            filter_method = "pearson" if "pearson" in self.methods else self.methods[0]
            filtered_features = self.feature_sets_[filter_method]
            
            # Этап 2: модельный отбор (F-regression или Mutual Info)
            model_methods = [m for m in self.methods if m in ["f_regression", "mutual_info"]]
            if model_methods:
                model_method = model_methods[0]
                model_features = self.feature_sets_[model_method]
                # Пересечение фильтрации и модельного отбора
                staged_features = list(set(filtered_features) & set(model_features))
            else:
                staged_features = filtered_features
            
            # Этап 3: SHAP-отбор (если указано)
            if self.model is not None and len(staged_features) > 1:
                shap_selector = SHAPFeatureSelector(
                    model=self.model,
                    n_features=min(self.top_k, len(staged_features)),
                    random_state=self.random_state
                )
                try:
                    X_staged = X[staged_features]
                    shap_selector.fit(X_staged, y)
                    self.selected_features_ = shap_selector.selected_features_
                    self.feature_importances_["shap"] = shap_selector.feature_importances_.tolist()
                except Exception as e:
                    if hasattr(self, "logger"):
                        self.logger.warning(f"SHAP stage failed: {e}. Using staged features without SHAP.")
                    self.selected_features_ = staged_features
            else:
                self.selected_features_ = staged_features
        
        # Защита от пустого результата
        if len(self.selected_features_) == 0:
            # Берем топ-признаки из первого метода
            self.selected_features_ = self.feature_sets_[self.methods[0]][:min(5, len(self.feature_sets_[self.methods[0]]))]
        
        self.is_fitted_ = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Применение отбора признаков к данным.
        
        Параметры
        ----------
        X : pd.DataFrame
            DataFrame с признаками.
        
        Возвращает
        ----------
        X_transformed : pd.DataFrame
            DataFrame только с отобранными признаками.
        """
        if not self.is_fitted_:
            raise TimeSeriesError("Selector is not fitted. Call fit() first.")
        
        missing_features = set(self.selected_features_) - set(X.columns)
        if missing_features:
            raise TimeSeriesError(
                f"Input DataFrame is missing features selected during fit: {missing_features}"
            )
        
        return X[self.selected_features_]
    
    def get_selected_features(self) -> List[str]:
        """
        Получение списка отобранных признаков.
        
        Возвращает
        ----------
        selected_features : List[str]
            Список имен отобранных признаков.
        """
        if not self.is_fitted_:
            raise TimeSeriesError("Selector is not fitted. Call fit() first.")
        
        return self.selected_features_
    
    def get_selection_report(self) -> Dict[str, Any]:
        """
        Получение отчета о результатах гибридного отбора.
        
        Возвращает
        ----------
        report : Dict[str, Any]
            Словарь с результатами по каждому методу и стратегии.
        """
        if not self.is_fitted_:
            raise TimeSeriesError("Selector is not fitted. Call fit() first.")
        
        return {
            "strategy": self.strategy,
            "methods": self.methods,
            "feature_sets": self.feature_sets_,
            "selected_features": self.selected_features_,
            "n_selected": len(self.selected_features_)
        }


class CombinedFeatureSelector(FeatureSelector):
    """
    Комбинированный селектор признаков, применяющий несколько методов последовательно.
    
    Реализует многоступенчатый подход:
    1. Фильтрация по пропускам (MissingValueSelector)
    2. Фильтрация по дисперсии (VarianceThresholdSelector)
    3. Отбор по важности (SHAPFeatureSelector или HybridFeatureSelector)
    
    Позволяет гибко настраивать последовательность и параметры каждого этапа.
    
    Параметры
    ----------
    missing_threshold : float, по умолчанию 0.2
        Порог доли пропусков для первого этапа фильтрации.
    variance_threshold : float, по умолчанию 0.0
        Порог дисперсии для второго этапа фильтрации.
    selection_method : str, по умолчанию "shap"
        Метод отбора на третьем этапе:
        - "shap": SHAP-отбор
        - "hybrid": гибридный отбор
    shap_n_features : int или float, опционально
        Количество или доля признаков для SHAP-отбора.
    hybrid_methods : List[str], опционально
        Методы для гибридного отбора.
    hybrid_strategy : str, по умолчанию "union"
        Стратегия гибридного отбора.
    model : object, опционально
        Модель для SHAP или гибридного отбора.
    skip_selection : bool, по умолчанию False
        Пропустить этап отбора по важности.
    
    Атрибуты
    ----------
    selected_features_ : List[str]
        Список окончательно отобранных признаков.
    feature_importances_ : np.ndarray или None
        Важность признаков после отбора.
    selectors_ : List[FeatureSelector]
        Список примененных селекторов в порядке выполнения.
    is_fitted_ : bool
        Флаг, указывающий, был ли вызван метод fit().
    
    Примеры
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from ts_feature_eng.selection import CombinedFeatureSelector
    >>> 
    >>> # Создаем данные с разными типами признаков
    >>> X = pd.DataFrame({
    ...     "good1": np.random.randn(1000),
    ...     "good2": np.random.randn(1000),
    ...     "noisy": np.random.randn(1000),
    ...     "constant": np.ones(1000),
    ... })
    >>> X.loc[:199, "noisy"] = np.nan  # 20% пропусков
    >>> y = X["good1"] * 2 + X["good2"] * 1.5 + np.random.randn(1000) * 0.1
    >>> 
    >>> selector = CombinedFeatureSelector(
    ...     missing_threshold=0.15,   # Удаляем 'noisy' (20% > 15%)
    ...     variance_threshold=0.01,  # Удаляем 'constant'
    ...     selection_method="hybrid",
    ...     hybrid_methods=["pearson", "distance_corr"],
    ...     hybrid_strategy="union",
    ...     hybrid_top_k=2
    ... )
    >>> X_selected = selector.fit_transform(X, y)
    >>> 
    >>> print(X_selected.columns.tolist())
    ['good1', 'good2']  # Только информативные признаки
    """
    
    def __init__(
        self,
        missing_threshold: float = 0.2,
        variance_threshold: float = 0.0,
        selection_method: str = "shap",
        shap_n_features: Optional[Union[int, float]] = None,
        hybrid_methods: Optional[List[str]] = None,
        hybrid_strategy: str = "union",
        hybrid_top_k: int = 20,
        model: Optional[object] = None,
        skip_selection: bool = False,
    ):
        super().__init__()
        self.missing_threshold = missing_threshold
        self.variance_threshold = variance_threshold
        self.selection_method = selection_method
        self.shap_n_features = shap_n_features
        self.hybrid_methods = hybrid_methods or ["pearson", "distance_corr"]
        self.hybrid_strategy = hybrid_strategy
        self.hybrid_top_k = hybrid_top_k
        self.model = model
        self.skip_selection = skip_selection
    
    def fit(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]) -> "CombinedFeatureSelector":
        """
        Последовательное применение всех этапов отбора признаков.
        
        Параметры
        ----------
        X : pd.DataFrame
            DataFrame с признаками.
        y : pd.Series или np.ndarray
            Целевая переменная.
        
        Возвращает
        ----------
        self : CombinedFeatureSelector
            Обученный селектор.
        """
        X, y = self._validate_input(X, y)
        
        # Этап 1: Фильтрация по пропускам
        missing_selector = MissingValueSelector(threshold=self.missing_threshold)
        X_stage1 = missing_selector.fit_transform(X, y)
        
        # Этап 2: Фильтрация по дисперсии
        variance_selector = VarianceThresholdSelector(
            threshold=self.variance_threshold,
            skipna=True
        )
        X_stage2 = variance_selector.fit_transform(X_stage1, y)
        
        # Этап 3: Отбор по важности (SHAP или Hybrid)
        if not self.skip_selection and X_stage2.shape[1] > 1:
            if self.selection_method == "shap":
                final_selector = SHAPFeatureSelector(
                    model=self.model,
                    n_features=self.shap_n_features,
                    max_samples=min(1000, len(X_stage2)),
                    random_state=42
                )
            elif self.selection_method == "hybrid":
                final_selector = HybridFeatureSelector(
                    methods=self.hybrid_methods,
                    strategy=self.hybrid_strategy,
                    top_k=self.hybrid_top_k,
                    model=self.model,
                    random_state=42
                )
            else:
                raise ValueError(f"Unknown selection_method: {self.selection_method}")
            
            try:
                X_stage3 = final_selector.fit_transform(X_stage2, y)
                self.feature_importances_ = final_selector.feature_importances_
            except Exception as e:
                if hasattr(self, "logger"):
                    self.logger.warning(f"Final selection failed: {e}. Skipping final stage.")
                X_stage3 = X_stage2
                self.feature_importances_ = None
        else:
            X_stage3 = X_stage2
            self.feature_importances_ = None
        
        # Сохранение результатов
        self.selected_features_ = X_stage3.columns.tolist()
        self.selectors_ = [missing_selector, variance_selector]
        if not self.skip_selection and X_stage2.shape[1] > 1:
            self.selectors_.append(final_selector)
        
        # Защита от полного удаления признаков
        if len(self.selected_features_) == 0:
            # Возвращаемся к результату после фильтрации по дисперсии
            self.selected_features_ = X_stage2.columns.tolist()
            if len(self.selected_features_) == 0:
                # Крайний случай — возвращаем исходные признаки
                self.selected_features_ = X.columns.tolist()[:5]  # Топ-5
        
        self.is_fitted_ = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Применение отбора признаков к данным.
        
        Параметры
        ----------
        X : pd.DataFrame
            DataFrame с признаками.
        
        Возвращает
        ----------
        X_transformed : pd.DataFrame
            DataFrame только с отобранными признаками.
        """
        if not self.is_fitted_:
            raise TimeSeriesError("Selector is not fitted. Call fit() first.")
        
        missing_features = set(self.selected_features_) - set(X.columns)
        if missing_features:
            raise TimeSeriesError(
                f"Input DataFrame is missing features selected during fit: {missing_features}"
            )
        
        return X[self.selected_features_]
    
    def get_selected_features(self) -> List[str]:
        """
        Получение списка отобранных признаков.
        
        Возвращает
        ----------
        selected_features : List[str]
            Список имен отобранных признаков.
        """
        if not self.is_fitted_:
            raise TimeSeriesError("Selector is not fitted. Call fit() first.")
        
        return self.selected_features_
    
    def get_selection_report(self) -> pd.DataFrame:
        """
        Получение отчета о результатах каждого этапа отбора.
        
        Возвращает
        ----------
        report : pd.DataFrame
            DataFrame с колонками ['stage', 'n_features_before', 'n_features_after', 'removed_features'].
        """
        if not self.is_fitted_:
            raise TimeSeriesError("Selector is not fitted. Call fit() first.")
        
        report_data = []
        
        for i, selector in enumerate(self.selectors_):
            stage_name = type(selector).__name__
            n_before = len(selector.feature_names_in_) if hasattr(selector, "feature_names_in_") else "N/A"
            n_after = len(selector.selected_features_)
            removed = set(selector.feature_names_in_) - set(selector.selected_features_) if hasattr(selector, "feature_names_in_") else []
            
            report_data.append({
                "stage": f"{i+1}. {stage_name}",
                "n_features_before": n_before,
                "n_features_after": n_after,
                "removed_features": list(removed)[:5] + (["..."] if len(removed) > 5 else []) if removed else []
            })
        
        return pd.DataFrame(report_data)