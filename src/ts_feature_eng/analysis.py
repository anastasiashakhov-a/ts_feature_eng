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
    y: pd.Series,
    methods: List[str],
    top_k: int,
    sample_size: int = 1000  # Новый параметр для подвыборки
) -> Dict[str, List[str]]:
    """Сравнивает методы отбора признаков."""
    # Подвыборка данных
    if len(X) > sample_size:
        X_sample = X.sample(n=sample_size, random_state=42)
        y_sample = y.loc[X_sample.index]
    else:
        X_sample = X
        y_sample = y

    selected_features = {}

    for method in methods:
        if method == "pearson":
            correlations = X_sample.corrwith(y_sample, method="pearson")
            top_features = correlations.abs().nlargest(top_k).index.tolist()
        elif method == "distance_corr":
            try:
                from dcor import distance_correlation
                correlations = {
                    col: distance_correlation(X_sample[col].values, y_sample.values)
                    for col in X_sample.columns
                }
                top_features = sorted(correlations, key=correlations.get, reverse=True)[:top_k]
            except ImportError:
                print("Модуль dcor не установлен.")
                top_features = []
        else:
            raise ValueError(f"Неизвестный метод: {method}")

        selected_features[method] = top_features

    return selected_features


def hybrid_feature_selection(
    X: pd.DataFrame,
    y: Union[pd.Series, np.ndarray],
    methods: List[str] = ["pearson", "distance_corr"],
    top_k: int = 20,
    hybrid_strategy: str = "union"
) -> List[str]:
    """
    Гибридизация методов отбора признаков.
    
    Параметры
    ----------
    X : pd.DataFrame
        Признаковое пространство.
    y : pd.Series или np.ndarray
        Целевая переменная.
    methods : List[str]
        Методы для гибридизации (например, "pearson", "distance_corr").
    top_k : int
        Количество признаков для каждого метода.
    hybrid_strategy : str
        Стратегия комбинирования:
        - "union": объединение всех признаков.
        - "intersection": пересечение признаков.
    
    Возвращает
    ----------
    selected_features : List[str]
        Список отобранных признаков после гибридизации.
    """
    # Сравнение методов
    selected_sets = compare_feature_selection_methods(
        X=X,
        y=y,
        methods=methods,
        top_k=top_k
    )
    
    # Гибридизация
    if hybrid_strategy == "union":
        selected_features = list(set.union(*[set(features) for features in selected_sets.values()]))
    elif hybrid_strategy == "intersection":
        selected_features = list(set.intersection(*[set(features) for features in selected_sets.values()]))
    else:
        raise ValueError(f"Unknown hybrid strategy: {hybrid_strategy}")
    
    return selected_features

def plot_correlation_matrix(
    X: pd.DataFrame,
    title: str = "Матрица корреляций",
    figsize: tuple = (10, 8),
    save_path: Optional[str] = None
):
    """
    Визуализация матрицы корреляций между признаками.
    
    Параметры
    ----------
    X : pd.DataFrame
        DataFrame с признаками.
    title : str, по умолчанию "Матрица корреляций"
        Заголовок графика.
    figsize : tuple, по умолчанию (10, 8)
        Размер графика.
    save_path : str, опционально
        Путь для сохранения графика.
    """
    corr_matrix = X.corr()
    
    plt.figure(figsize=figsize)
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
        print(f"График сохранен как '{save_path}'")
    
    plt.show()


def evaluate_block_performance(
    X: pd.DataFrame,
    y: Union[pd.Series, np.ndarray],
    feature_blocks: Dict[str, List[str]],
    model: object = None,
    cv: int = 3
) -> pd.DataFrame:
    """
    Оценка качества модели по блокам признаков.
    
    Параметры
    ----------
    X : pd.DataFrame
        Признаковое пространство.
    y : pd.Series или np.ndarray
        Целевая переменная.
    feature_blocks : Dict[str, List[str]]
        Словарь блоков признаков.
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
        
        X_subset = X[features].fillna(0)
        scores = cross_val_score(model, X_subset, y_array, cv=cv, scoring='neg_mean_absolute_error')
        mean_mae = -scores.mean()
        std_mae = scores.std()
        
        perf_data.append({
            "Block": block_name,
            "Mean_MAE": mean_mae,
            "Std_MAE": std_mae,
            "Num_Features": len(features)
        })
    
    if not perf_data:
        return pd.DataFrame(columns=["Block", "Mean_MAE", "Std_MAE", "Num_Features"])
    
    return pd.DataFrame(perf_data).sort_values("Mean_MAE")

def analyze_feature_intersection(
    selected_sets: Dict[str, List[str]]
) -> Dict[str, List[str]]:
    """
    Анализ пересечений наборов признаков, отобранных разными методами.
    
    Параметры
    ----------
    selected_sets : Dict[str, List[str]]
        Словарь: {метод: [список отобранных признаков]}.
    
    Возвращает
    ----------
    intersection_results : Dict[str, List[str]]
        Словарь с результатами анализа пересечений.
    """
    methods = list(selected_sets.keys())
    intersection_results = {}
    
    # Полное пересечение всех методов
    full_intersection = set.intersection(*[set(features) for features in selected_sets.values()])
    intersection_results["full_intersection"] = list(full_intersection)
    
    # Попарные пересечения
    for i in range(len(methods)):
        for j in range(i + 1, len(methods)):
            method1 = methods[i]
            method2 = methods[j]
            intersection = set(selected_sets[method1]).intersection(set(selected_sets[method2]))
            intersection_results[f"{method1}_and_{method2}"] = list(intersection)
    
    return intersection_results

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