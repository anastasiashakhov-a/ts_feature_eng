# src/ts_feature_eng/analysis.py

"""
Модуль для анализа эффективности инженерии признаков.

Содержит утилиты для:
- Сравнения стратегий отбора признаков (включая корреляцию расстояний)
- Построения матриц корреляций
- Анализа пересечений наборов признаков
- Оценки влияния отбора на качество модели по блокам
"""

from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.feature_selection import f_regression, mutual_info_regression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns


def distance_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """
    Вычисление корреляции расстояний между двумя массивами.
    
    Корреляция расстояний измеряет как линейные, так и нелинейные зависимости.
    
    Параметры
    ----------
    x : np.ndarray
        Первый массив данных.
    y : np.ndarray  
        Второй массив данных.
    
    Возвращает
    ----------
    dcor : float
        Значение корреляции расстояний в диапазоне [0, 1].
        0 - независимость, 1 - полная зависимость.
    """
    def _center_distance_matrix(D):
        """Центрирование матрицы расстояний."""
        n = D.shape[0]
        D_centered = D.copy()
        D_centered -= D.mean(axis=0)
        D_centered -= D_centered.mean(axis=1)[:, np.newaxis]
        D_centered += D.mean()
        return D_centered
    
    def _distance_covariance(X, Y):
        """Вычисление ковариации расстояний."""
        A = _center_distance_matrix(np.sqrt(np.square(X[:, np.newaxis] - X[np.newaxis, :])).sum(axis=2))
        B = _center_distance_matrix(np.sqrt(np.square(Y[:, np.newaxis] - Y[np.newaxis, :])).sum(axis=2))
        return np.sqrt((A * B).sum() / (X.shape[0] ** 2))
    
    # Обработка одномерных массивов
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    
    # Вычисление корреляции расстояний
    dcov_xy = _distance_covariance(x.flatten(), y.flatten())
    dcov_xx = _distance_covariance(x.flatten(), x.flatten())
    dcov_yy = _distance_covariance(y.flatten(), y.flatten())
    
    if dcov_xx * dcov_yy == 0:
        return 0.0
    
    return dcov_xy / np.sqrt(dcov_xx * dcov_yy)


def compare_feature_selection_methods(
    X: pd.DataFrame,
    y: Union[pd.Series, np.ndarray],
    methods: List[str] = ["pearson", "f_regression", "mutual_info", "distance_corr"],
    top_k: int = 20,
    model_for_shap: Optional[object] = None
) -> Dict[str, List[str]]:
    """
    Сравнение различных методов отбора признаков.
    
    Поддерживаемые методы:
    - "pearson": Корреляция Пирсона (линейная зависимость)
    - "f_regression": F-статистика ANOVA
    - "mutual_info": Взаимная информация (нелинейная зависимость)
    - "distance_corr": Корреляция расстояний (линейная + нелинейная зависимость)
    - "shap": SHAP важность (требует обученной модели)
    
    Параметры
    ----------
    X : pd.DataFrame
        Признаковое пространство.
    y : pd.Series или np.ndarray
        Целевая переменная.
    methods : List[str]
        Список методов для сравнения.
    top_k : int
        Количество признаков для отбора по каждой стратегии.
    model_for_shap : object, опционально
        Обученная модель для SHAP анализа.
    
    Возвращает
    ----------
    selected_features : Dict[str, List[str]]
        Словарь: {метод: [список отобранных признаков]}
    """
    results = {}
    y_array = np.asarray(y)
    
    # Pearson correlation
    if "pearson" in methods:
        correlations = X.corrwith(pd.Series(y_array), method="pearson").abs().sort_values(ascending=False)
        results["pearson"] = correlations.head(top_k).index.tolist()
    
    # F-regression
    if "f_regression" in methods:
        from sklearn.feature_selection import SelectKBest
        selector = SelectKBest(score_func=f_regression, k=top_k)
        X_selected = selector.fit_transform(X.fillna(0), y_array)
        selected_mask = selector.get_support()
        results["f_regression"] = X.columns[selected_mask].tolist()
    
    # Mutual Information
    if "mutual_info" in methods:
        mi_scores = mutual_info_regression(X.fillna(0), y_array, random_state=42)
        mi_series = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
        results["mutual_info"] = mi_series.head(top_k).index.tolist()
    
    # Distance Correlation
    if "distance_corr" in methods:
        dist_corr_scores = []
        for col in X.columns:
            try:
                corr = distance_correlation(X[col].fillna(0).values, y_array)
                dist_corr_scores.append((col, corr))
            except:
                dist_corr_scores.append((col, 0.0))
        
        dist_corr_scores.sort(key=lambda x: x[1], reverse=True)
        results["distance_corr"] = [col for col, _ in dist_corr_scores[:top_k]]
    
    # SHAP (требует модель)
    if "shap" in methods and model_for_shap is not None:
        try:
            import shap
            explainer = shap.LinearExplainer(model_for_shap, X.fillna(0).iloc[:100])
            shap_values = explainer.shap_values(X.fillna(0).iloc[:100])
            shap_importance = np.abs(shap_values).mean(0)
            shap_feature_names = X.columns
            top_shap_idx = np.argsort(shap_importance)[-top_k:][::-1]
            results["shap"] = [shap_feature_names[i] for i in top_shap_idx]
        except ImportError:
            print("SHAP не установлен. Пропускаем SHAP-анализ.")
        except Exception as e:
            print(f"Ошибка при SHAP-анализе: {e}")
    
    return results


def analyze_feature_intersection(
    selected_features_dict: Dict[str, List[str]]
) -> pd.DataFrame:
    """
    Анализ пересечений наборов отобранных признаков.
    
    Параметры
    ----------
    selected_features_dict : Dict[str, List[str]]
        Результат compare_feature_selection_methods.
    
    Возвращает
    ----------
    intersection_df : pd.DataFrame
        DataFrame с метриками пересечения между методами.
    """
    from itertools import combinations
    
    intersection_data = []
    for method1, method2 in combinations(selected_features_dict.keys(), 2):
        set1 = set(selected_features_dict[method1])
        set2 = set(selected_features_dict[method2])
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        jaccard = intersection / union if union > 0 else 0
        
        intersection_data.append({
            "Method_1": method1,
            "Method_2": method2,
            "Intersection_Count": intersection,
            "Union_Count": union,
            "Jaccard_Similarity": jaccard
        })
    
    return pd.DataFrame(intersection_data)


def plot_correlation_matrix(
    X: pd.DataFrame,
    features: List[str],
    title: str = "Матрица корреляций",
    save_path: Optional[str] = None
):
    """
    Построение матрицы корреляций для подмножества признаков.
    
    Параметры
    ----------
    X : pd.DataFrame
        Признаковое пространство.
    features : List[str]
        Список признаков для анализа.
    title : str
        Заголовок графика.
    save_path : str, опционально
        Путь для сохранения графика.
    """
    X_subset = X[features].fillna(0)
    corr_matrix = X_subset.corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        cbar_kws={"shrink": 0.8}
    )
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  График сохранен как '{save_path}'")
    
    plt.show()


def evaluate_block_performance(
    X: pd.DataFrame,
    y: Union[pd.Series, np.ndarray],
    feature_blocks: Dict[str, List[str]],
    model=None,
    cv: int = 3
) -> pd.DataFrame:
    """
    Оценка качества модели по блокам признаков.
    
    Вместо оценки каждого метода отбора отдельно, оцениваются
    предопределенные блоки признаков (например, оконные, спектральные).
    
    Параметры
    ----------
    X : pd.DataFrame
        Исходное признаковое пространство.
    y : pd.Series или np.ndarray
        Целевая переменная.
    feature_blocks : Dict[str, List[str]]
        Словарь: {название_блока: [список_признаков]}.
    model : object
        Модель для оценки (по умолчанию Ridge).
    cv : int
        Количество фолдов кросс-валидации.
    
    Возвращает
    ----------
    performance_df : pd.DataFrame
        DataFrame с метриками качества для каждого блока.
    """
    if model is None:
        model = Ridge(alpha=1.0, random_state=42)
    
    perf_data = []
    y_array = np.asarray(y)
    
    for block_name, features in feature_blocks.items():
        if len(features) == 0:
            continue
        
        # Проверяем, что все признаки существуют в X
        valid_features = [f for f in features if f in X.columns]
        if len(valid_features) == 0:
            continue
        
        X_subset = X[valid_features].fillna(0)
        scores = cross_val_score(model, X_subset, y_array, cv=cv, scoring='neg_mean_absolute_error')
        mean_mae = -scores.mean()
        std_mae = scores.std()
        
        perf_data.append({
            "Block": block_name,
            "Mean_MAE": mean_mae,
            "Std_MAE": std_mae,
            "Num_Features": len(valid_features)
        })
    
    if not perf_data:
        return pd.DataFrame(columns=["Block", "Mean_MAE", "Std_MAE", "Num_Features"])
    
    return pd.DataFrame(perf_data).sort_values("Mean_MAE")


def evaluate_selection_impact(
    X: pd.DataFrame,
    y: Union[pd.Series, np.ndarray],
    selected_features_dict: Dict[str, List[str]],
    model=None,
    cv: int = 3
) -> pd.DataFrame:
    """
    Оценка влияния отбора признаков на качество модели.
    
    Параметры
    ----------
    X : pd.DataFrame
        Исходное признаковое пространство.
    y : pd.Series или np.ndarray
        Целевая переменная.
    selected_features_dict : Dict[str, List[str]]
        Результат compare_feature_selection_methods.
    model : object
        Модель для оценки (по умолчанию Ridge).
    cv : int
        Количество фолдов кросс-валидации.
    
    Возвращает
    ----------
    performance_df : pd.DataFrame
        DataFrame с метриками качества для каждого метода отбора.
    """
    if model is None:
        model = Ridge(alpha=1.0, random_state=42)
    
    perf_data = []
    y_array = np.asarray(y)
    
    for method, features in selected_features_dict.items():
        if len(features) == 0:
            continue
        
        X_subset = X[features].fillna(0)
        scores = cross_val_score(model, X_subset, y_array, cv=cv, scoring='neg_mean_absolute_error')
        mean_mae = -scores.mean()
        std_mae = scores.std()
        
        perf_data.append({
            "Method": method,
            "Mean_MAE": mean_mae,
            "Std_MAE": std_mae,
            "Num_Features": len(features)
        })
    
    if not perf_data:
        return pd.DataFrame(columns=["Method", "Mean_MAE", "Std_MAE", "Num_Features"])
    
    return pd.DataFrame(perf_data).sort_values("Mean_MAE")


def create_feature_blocks(X: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Создание блоков признаков на основе их префиксов.
    
    Автоматически группирует признаки по типу трансформера:
    - window.* → Window Features
    - dwt.*, stl.* → Spectral Features  
    - time.*, calendar.* → Time Features
    - остальные → Other Features
    
    Параметры
    ----------
    X : pd.DataFrame
        Признаковое пространство с сгенерированными признаками.
    
    Возвращает
    ----------
    feature_blocks : Dict[str, List[str]]
        Словарь блоков признаков.
    """
    feature_blocks = {
        "Window Features": [],
        "Spectral Features": [],
        "Time Features": [],
        "Other Features": []
    }
    
    for col in X.columns:
        if col.startswith("window."):
            feature_blocks["Window Features"].append(col)
        elif col.startswith("dwt.") or col.startswith("stl."):
            feature_blocks["Spectral Features"].append(col)
        elif col.startswith("time.") or col.startswith("calendar."):
            feature_blocks["Time Features"].append(col)
        else:
            feature_blocks["Other Features"].append(col)
    
    # Удаляем пустые блоки
    feature_blocks = {k: v for k, v in feature_blocks.items() if v}
    
    return feature_blocks