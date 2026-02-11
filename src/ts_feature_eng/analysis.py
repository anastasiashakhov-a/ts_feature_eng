# src/ts_feature_eng/analysis.py

"""
Модуль для анализа эффективности инженерии признаков.

Содержит утилиты для:
- Сравнения стратегий отбора признаков
- Построения матриц корреляций
- Анализа пересечений наборов признаков
- Оценки влияния отбора на качество модели
"""

# === ДОБАВЬТЕ ЭТИ ИМПОРТЫ В НАЧАЛО ФАЙЛА ===
from typing import Dict, List, Optional, Union  # <-- ЭТИ СТРОКИ НЕ ХВАТАЛО
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression  # <-- Эти тоже нужны
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer, mean_absolute_error
# ============================================

def compare_feature_selection_methods(
    X: pd.DataFrame,
    y: Union[pd.Series, np.ndarray],
    methods: List[str] = ["pearson", "f_regression", "mutual_info"],
    top_k: int = 20,
    model_for_shap: Optional[object] = None
) -> Dict[str, List[str]]:
    """
    Сравнение различных методов отбора признаков.
    
    Параметры
    ----------
    X : pd.DataFrame
        Признаковое пространство.
    y : pd.Series или np.ndarray
        Целевая переменная.
    methods : List[str]
        Список методов: "pearson", "f_regression", "mutual_info", "shap".
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
    
    # Pearson correlation
    if "pearson" in methods:
        correlations = X.corrwith(pd.Series(y), method="pearson").abs().sort_values(ascending=False)
        results["pearson"] = correlations.head(top_k).index.tolist()
    
    # F-regression
    if "f_regression" in methods:
        selector = SelectKBest(score_func=f_regression, k=top_k)
        X_selected = selector.fit_transform(X.fillna(0), y)
        selected_mask = selector.get_support()
        results["f_regression"] = X.columns[selected_mask].tolist()
    
    # Mutual Information
    if "mutual_info" in methods:
        mi_scores = mutual_info_regression(X.fillna(0), y, random_state=42)
        mi_series = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
        results["mutual_info"] = mi_series.head(top_k).index.tolist()
    
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
    from sklearn.model_selection import cross_val_score
    
    if model is None:
        model = Ridge(alpha=1.0, random_state=42)
    
    perf_data = []
    for method, features in selected_features_dict.items():
        if len(features) == 0:
            continue
        
        X_subset = X[features].fillna(0)
        scores = cross_val_score(model, X_subset, y, cv=cv, scoring='neg_mean_absolute_error')
        mean_mae = -scores.mean()
        std_mae = scores.std()
        
        perf_data.append({
            "Method": method,
            "Mean_MAE": mean_mae,
            "Std_MAE": std_mae,
            "Num_Features": len(features)
        })
    
    return pd.DataFrame(perf_data).sort_values("Mean_MAE")
