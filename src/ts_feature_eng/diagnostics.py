# src/ts_feature_eng/diagnostics.py
"""
Диагностика качества feature engineering.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score

def feature_survival_rate(X, y, model, cv=5, threshold=0.5):
    """
    Вычисление доли признаков, стабильно важных через CV.
    """
    from sklearn.feature_selection import SelectKBest, f_regression
    
    n_features = X.shape[1]
    survival_counts = np.zeros(n_features)
    
    for train_idx, test_idx in cv.split(X):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        
        # Отбор признаков на fold
        selector = SelectKBest(score_func=f_regression, k='all')
        selector.fit(X_train.fillna(0), y_train)
        
        # Топ-50% признаков
        scores = selector.scores_
        top_threshold = np.percentile(scores, 50)
        selected = scores >= top_threshold
        survival_counts += selected
    
    # Доля выживших признаков
    survival_rate = survival_counts / cv.n_splits
    return survival_rate

def collapse_diagnostics(X_transformed):
    """
    Диагностика feature collapse.
    """
    n_features = X_transformed.shape[1]
    
    if n_features < 2:
        return {'collapse_detected': True, 'effective_rank': 0}
    
    # Эффективный ранг через корреляционную матрицу
    corr_matrix = X_transformed.fillna(0).corr().values
    eigenvalues = np.linalg.eigvalsh(corr_matrix)
    effective_rank = np.sum(eigenvalues > 1e-6)
    
    return {
        'collapse_detected': effective_rank < n_features * 0.3,
        'effective_rank': int(effective_rank),
        'n_features': n_features,
        'rank_ratio': effective_rank / n_features
    }