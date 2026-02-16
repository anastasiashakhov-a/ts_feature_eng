# exapmples/comparison_temperature_analysis.py

"""
Сравнение стратегий отбора признаков на данных минимальных суточных температур.

Анализирует:
- Структуру временного ряда через мета-признаки (включая статистические тесты)
- Выбор оптимальных методов инженерии признаков
- Влияние отбора на качество прогноза
- Интерпретацию важности признаков через SHAP
- Сравнение линейных и нелинейных методов отбора (Pearson vs Distance Correlation)
- Блоковую оценку эффективности групп признаков
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, TimeSeriesSplit

from ts_feature_eng import AutoFeatureEngineer
from ts_feature_eng.meta_features import MetaFeatureExtractor
from ts_feature_eng.selection import CombinedFeatureSelector
from ts_feature_eng.analysis import (
    compare_feature_selection_methods,
    analyze_feature_intersection,
    plot_correlation_matrix,
    evaluate_block_performance,
    create_feature_blocks,
    hybrid_feature_selection  # Новая функция для гибридизации
)


def load_temperature_data(data_path=None):
    """Загрузка данных минимальных суточных температур."""
    if data_path is None:
        possible_paths = [
            "data/daily-minimum-temperatures-in-me.csv",
            "../data/daily-minimum-temperatures-in-me.csv",
            os.path.expanduser("~/ts_feature_eng/data/daily-minimum-temperatures-in-me.csv"),
            os.path.join(os.path.dirname(__file__), "..", "data", "daily-minimum-temperatures-in-me.csv"),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                data_path = path
                break
        
        if data_path is None:
            raise FileNotFoundError(
                "Файл данных не найден. Проверьте наличие одного из следующих файлов:\n"
                f"{possible_paths}"
            )
    
    print(f"Загрузка данных из: {data_path}")
    
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.strip()
    
    # Определение столбцов
    time_col = None
    temp_col = None
    
    for col in df.columns:
        if any(keyword in col.lower() for keyword in ["datetime", "date", "time", "timestamp"]):
            time_col = col
        if any(keyword in col.lower() for keyword in ["temp", "min", "temperature"]):
            temp_col = col
    
    if time_col is None:
        time_col = df.columns[0]
    
    if temp_col is None:
        temp_col = df.columns[1]
    
    result = pd.DataFrame({
        "timestamp": pd.to_datetime(df[time_col]),
        "min_temperature": pd.to_numeric(df[temp_col], errors="coerce")
    })
    
    # Очистка некорректных значений
    mask_invalid = result["min_temperature"].astype(str).str.contains(r'[?]', na=False)
    if mask_invalid.any():
        print(f"  Обнаружено {mask_invalid.sum()} некорректных значений в температуре. Удаляем...")
        result = result[~mask_invalid]
    
    result = result.set_index("timestamp")
    initial_len = len(result)
    result = result.dropna(subset=["min_temperature"])
    final_len = len(result)
    
    print(f"Удалено {initial_len - final_len} наблюдений с пропусками")
    print(f"Итоговый размер данных: {len(result)} наблюдений")
    print(f"Диапазон дат: {result.index.min()} — {result.index.max()}")
    
    return result


def analyze_meta_features_structure(df):
    """Анализ структуры ряда через мета-признаки."""
    print("\nИзвлечение мета-признаков ряда...")
    
    extractor = MetaFeatureExtractor(
        categories=["simple", "statistical", "spectral", "information_theoretic"],
        fill_method="linear"
    )
    
    y = df["min_temperature"].shift(-1).dropna()
    X = df.iloc[:-1][["min_temperature"]]
    
    meta_df = extractor.fit_transform(X, y)
    meta_features = meta_df.iloc[0].to_dict()
    
    print(f"Извлечено {len(meta_features)} мета-признаков")
    
    # Ключевые мета-признаки
    print("\nКлючевые мета-признаки:")
    print(f"  Длина ряда: {meta_features.get('length', 'N/A'):,.0f}")
    print(f"  Пропуски: {meta_features.get('missing_ratio', 0)*100:.2f}%")
    print(f"  Стационарность (ADF p-value): {meta_features.get('stationarity_adf', 1.0):.3f}")
    print(f"  Линейность (R²): {meta_features.get('linearity', 0):.3f}")
    print(f"  Годовая ACF: {meta_features.get('acf_365', 0):.3f}")
    print(f"  Месячная ACF: {meta_features.get('acf_30', 0):.3f}")
    print(f"  Доминирующая частота: {meta_features.get('dominant_freq', 0):.4f}")
    
    # Статистические тесты
    print(f"\nСтатистические тесты:")
    print(f"  Нормальность (Shapiro-Wilk p-value): {meta_features.get('normality_shapiro', 'N/A')}")
    print(f"  Нормальность (Jarque-Bera p-value): {meta_features.get('normality_jarque', 'N/A')}")
    print(f"  Автокорреляция (Ljung-Box p-value): {meta_features.get('autocorrelation_ljungbox', 'N/A')}")
    print(f"  Гомоскедастичность (Breusch-Pagan p-value): {meta_features.get('homoskedasticity_bp', 'N/A')}")
    
    return meta_features


def visualize_temperature_patterns(df):
    """Визуализация ключевых паттернов температур."""
    print("\nВизуализация паттернов температур...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Общий тренд
    axes[0, 0].plot(df.index, df["min_temperature"], color='blue', linewidth=0.8, alpha=0.7)
    axes[0, 0].set_title("Общий тренд минимальных температур")
    axes[0, 0].set_ylabel("Температура (°C)")
    axes[0, 0].grid(True, alpha=0.3)
    
    # Сезонный паттерн (по месяцам)
    monthly_avg = df.groupby(df.index.month)["min_temperature"].mean()
    month_labels = ['Янв', 'Фев', 'Мар', 'Апр', 'Май', 'Июн', 'Июл', 'Авг', 'Сен', 'Окт', 'Ноя', 'Дек']
    axes[0, 1].plot(month_labels, monthly_avg.values, marker='o', color='darkred', linewidth=2, markersize=6)
    axes[0, 1].set_title("Сезонный паттерн температур")
    axes[0, 1].set_ylabel("Средняя температура (°C)")
    axes[0, 1].grid(True, alpha=0.3)
    
    # Распределение
    axes[1, 0].hist(df["min_temperature"], bins=50, density=True, alpha=0.7, color='purple', edgecolor='black')
    axes[1, 0].axvline(df["min_temperature"].mean(), color='red', linestyle='--', label=f'Среднее: {df["min_temperature"].mean():.1f} °C')
    axes[1, 0].axvline(df["min_temperature"].median(), color='orange', linestyle='--', label=f'Медиана: {df["min_temperature"].median():.1f} °C')
    axes[1, 0].set_title("Распределение минимальных температур")
    axes[1, 0].set_xlabel("Температура (°C)")
    axes[1, 0].set_ylabel("Плотность")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Автокорреляция
    from statsmodels.graphics.tsaplots import plot_acf
    plot_acf(df["min_temperature"].dropna(), lags=365, ax=axes[1, 1], alpha=0.05)
    axes[1, 1].set_title("Автокорреляция температур (до 365 лагов)")
    axes[1, 1].set_xlabel("Лаг (дни)")
    axes[1, 1].set_ylabel("ACF")
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle("Паттерны минимальных температур")
    plt.tight_layout()
    plt.savefig("temperature_patterns.png", dpi=150, bbox_inches='tight')
    print("  График сохранен как 'temperature_patterns.png'")
    plt.close(fig)  # Закрываем график, чтобы избежать предупреждений


def analyze_feature_selection_process(df, meta_features):
    """Анализ выбора трансформеров на основе мета-признаков."""
    print("\nАнализ выбора трансформеров...")
    
    insights = {}
    
    # Сезонность
    acf_30 = meta_features.get('acf_30', 0)
    acf_365 = meta_features.get('acf_365', 0)
    dom_freq = meta_features.get('dominant_freq', 0)
    
    print(f"\nСезонность:")
    if acf_30 > 0.3:
        print(f"  Месячная сезонность: ACF(30) = {acf_30:.3f} (сильная)")
        insights["monthly_seasonality"] = True
    else:
        print(f"  Месячная сезонность: ACF(30) = {acf_30:.3f} (слабая)")
        insights["monthly_seasonality"] = False
    
    if acf_365 > 0.3:
        print(f"  Годовая сезонность: ACF(365) = {acf_365:.3f} (сильная)")
        insights["yearly_seasonality"] = True
    else:
        print(f"  Годовая сезонность: ACF(365) = {acf_365:.3f} (слабая)")
        insights["yearly_seasonality"] = False
    
    if dom_freq > 0.01:
        period = 1 / dom_freq if dom_freq > 0 else float('inf')
        print(f"  Доминирующая частота: {dom_freq:.4f} (период ≈ {period:.1f})")
        insights["dominant_period"] = period
    else:
        print(f"  Нет выраженной доминирующей частоты")
        insights["dominant_period"] = None
    
    # Стационарность
    adf_p = meta_features.get('stationarity_adf', 1.0)
    print(f"\nСтационарность:")
    if adf_p > 0.05:
        print(f"  Ряд нестационарен (ADF p-value = {adf_p:.3f})")
        insights["stationarity"] = "nonstationary"
    else:
        print(f"  Ряд стационарен (ADF p-value = {adf_p:.3f})")
        insights["stationarity"] = "stationary"
    
    # Линейность
    linearity = meta_features.get('linearity', 0)
    print(f"\nЛинейность:")
    if linearity > 0.5:
        print(f"  Высокая линейность: R² = {linearity:.3f}")
        insights["linearity"] = "high"
    else:
        print(f"  Низкая линейность: R² = {linearity:.3f}")
        insights["linearity"] = "low"
    
    # Сложность
    perm_entropy = meta_features.get('permutation_entropy', 0)
    print(f"\nСложность:")
    print(f"  Перестановочная энтропия: {perm_entropy:.3f}")
    
    if perm_entropy > 0.8:
        insights["complexity"] = "high"
        print("  → Высокая сложность: нелинейные методы, спектральный анализ")
    else:
        insights["complexity"] = "low"
        print("  → Низкая сложность: линейные методы, оконные статистики")
    
    # Статистические тесты
    shapiro_p = meta_features.get('normality_shapiro', 1.0)
    ljung_p = meta_features.get('autocorrelation_ljungbox', 1.0)
    bp_p = meta_features.get('homoskedasticity_bp', 1.0)
    
    print(f"\nСтатистические свойства:")
    if shapiro_p < 0.05:
        print(f"  Распределение не нормальное (Shapiro-Wilk p-value = {shapiro_p:.3f})")
        insights["normality"] = "non_normal"
    else:
        print(f"  Распределение нормальное (Shapiro-Wilk p-value = {shapiro_p:.3f})")
        insights["normality"] = "normal"
    
    if ljung_p < 0.05:
        print(f"  Значимая автокорреляция (Ljung-Box p-value = {ljung_p:.3f})")
        insights["autocorrelation"] = "significant"
    else:
        print(f"  Нет значимой автокорреляции (Ljung-Box p-value = {ljung_p:.3f})")
        insights["autocorrelation"] = "none"
    
    if bp_p < 0.05:
        print(f"  Гетероскедастичность (Breusch-Pagan p-value = {bp_p:.3f})")
        insights["homoskedasticity"] = "heteroskedastic"
    else:
        print(f"  Гомоскедастичность (Breusch-Pagan p-value = {bp_p:.3f})")
        insights["homoskedasticity"] = "homoskedastic"
    
    return insights


def compare_selection_strategies_with_insights(df, insights):
    """Сравнение стратегий отбора признаков с учетом инсайтов."""
    print("\nСравнение стратегий отбора признаков...")
    
    y = df["min_temperature"].shift(-1).dropna()
    X = df.iloc[:-1][["min_temperature"]]
    
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # 1. Автоматическая инженерия признаков
    print("  Автоматическая инженерия признаков...")
    engineer = AutoFeatureEngineer(
        optimize=True,
        n_calls=20,
        n_initial_points=5,
        apply_selection=True,
        selection_threshold=0.2,
        variance_threshold=0.01,
        shap_selection=True,
        shap_n_features=0.3,
        random_state=42,
        verbose=0
    )
    
    X_train_engineered = engineer.fit_transform(X_train, y_train)
    X_test_engineered = engineer.transform(X_test)
    
    print(f"    Сгенерировано признаков: {X_train_engineered.shape[1]}")
    print(f"    Отобрано признаков: {X_test_engineered.shape[1]}")
    
    # 2. Обучение модели на отобранных признаках
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train_engineered.fillna(0), y_train)
    
    y_pred_train = model.predict(X_train_engineered.fillna(0))
    y_pred_test = model.predict(X_test_engineered.fillna(0))
    
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_r2 = r2_score(y_test, y_pred_test)
    
    print(f"    MAE (обучение): {train_mae:.2f}")
    print(f"    MAE (тест): {test_mae:.2f}")
    print(f"    R² (тест): {test_r2:.4f}")
    
    # 3. SHAP-анализ важности признаков
    print("\n  SHAP-анализ важности признаков...")
    try:
        import shap
        
        sample_size = min(1000, len(X_train_engineered))
        X_sample = X_train_engineered.fillna(0).iloc[:sample_size]
        y_sample = y_train.iloc[:sample_size]
        
        model_shap = RandomForestRegressor(n_estimators=50, max_depth=6, random_state=42)
        model_shap.fit(X_sample, y_sample)
        
        explainer = shap.TreeExplainer(model_shap)
        shap_values = explainer.shap_values(X_sample)
        
        shap.summary_plot(
            shap_values,
            X_sample,
            max_display=20,
            show=False
        )
        plt.title("SHAP значения важности признаков (топ-20)")
        plt.tight_layout()
        plt.savefig("shap_feature_importance.png", dpi=150, bbox_inches='tight')
        print("    График SHAP важности сохранен как 'shap_feature_importance.png'")
        plt.close()  # Закрываем график, чтобы избежать предупреждений
        
        shap_importance = np.abs(shap_values).mean(0)
        feature_names = X_sample.columns
        top_indices = np.argsort(shap_importance)[-10:][::-1]
        
        print(f"\n    Топ-10 важных признаков (по SHAP):")
        for i, idx in enumerate(top_indices):
            print(f"      {i+1:2d}. {feature_names[idx]:30s} | {shap_importance[idx]:.4f}")
    
    except ImportError:
        print("    SHAP не установлен. Установите через: pip install shap")
    
    # 4. Анализ корреляций между отобранными признаками
    print("\n  Анализ корреляций между отобранными признаками...")
    
    n_features_to_analyze = min(20, X_train_engineered.shape[1])
    feature_subset = X_train_engineered.columns[:n_features_to_analyze]
    X_corr_subset = X_train_engineered[feature_subset].fillna(0)
    
    corr_matrix = X_corr_subset.corr()
    
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
    plt.title(f"Матрица корреляций между отобранными признаками (топ-{n_features_to_analyze})")
    plt.tight_layout()
    plt.savefig("feature_correlation_matrix.png", dpi=150, bbox_inches='tight')
    print(f"    Матрица корреляций сохранена как 'feature_correlation_matrix.png'")
    plt.close()  # Закрываем график, чтобы избежать предупреждений
    
    # 5. Сравнение линейных и нелинейных методов отбора
    print("\n  Сравнение линейных и нелинейных методов отбора...")
    
    # Сравниваем Pearson (линейный) и Distance Correlation (нелинейный)
    comparison_methods = ["pearson", "distance_corr"]
    selected_sets = compare_feature_selection_methods(
        X=X_train_engineered,
        y=y_train,
        methods=comparison_methods,
        top_k=15
    )
    
    print(f"    Результаты сравнения:")
    for method, features in selected_sets.items():
        print(f"      {method:15s}: {len(features)} признаков")
        if features:
            print(f"                     Примеры: {features[:3]}{'...' if len(features) > 3 else ''}")
    
    # Оценка качества для каждого метода
    model_for_comparison = RandomForestRegressor(n_estimators=50, random_state=42)
    perf_data = []
    for method, features in selected_sets.items():
        if len(features) == 0:
            continue
        
        X_subset = X_train_engineered[features].fillna(0)
        scores = cross_val_score(
            model_for_comparison, X_subset, y_train, 
            cv=3, scoring='neg_mean_absolute_error'
        )
        mean_mae = -scores.mean()
        perf_data.append({
            "Method": method,
            "Mean_MAE": mean_mae,
            "Num_Features": len(features)
        })
    
    perf_df = pd.DataFrame(perf_data).sort_values("Mean_MAE")
    print(f"\n    Качество моделей по методам отбора:")
    for _, row in perf_df.iterrows():
        print(f"      {row['Method']:15s}: MAE = {row['Mean_MAE']:.2f}")
    
    # 6. Блоковая оценка эффективности групп признаков
    print("\n  Блоковая оценка эффективности групп признаков...")
    
    # Создаем блоки признаков
    feature_blocks = create_feature_blocks(X_train_engineered)
    
    print(f"    Блоки признаков:")
    for block_name, features in feature_blocks.items():
        print(f"      {block_name:20s}: {len(features):3d} признаков")
    
    # Оцениваем каждый блок
    block_performance = evaluate_block_performance(
        X=X_train_engineered,
        y=y_train,
        feature_blocks=feature_blocks,
        model=RandomForestRegressor(n_estimators=50, random_state=42),
        cv=3
    )
    
    print(f"\n    Качество моделей по блокам признаков:")
    for _, row in block_performance.iterrows():
        print(f"      {row['Block']:20s}: MAE = {row['Mean_MAE']:.2f} ({row['Num_Features']} признаков)")
    
    # 7. Гибридизация методов отбора
    print("\n  Гибридизация методов отбора...")
    hybrid_strategy = "union"  # Можно адаптировать на основе мета-признаков
    hybrid_features = hybrid_feature_selection(
        X=X_train_engineered,
        y=y_train,
        methods=["pearson", "distance_corr"],
        top_k=15,
        hybrid_strategy=hybrid_strategy
    )
    print(f"    Гибридизация ({hybrid_strategy}): {len(hybrid_features)} признаков")
    print(f"                     Примеры: {hybrid_features[:3]}{'...' if len(hybrid_features) > 3 else ''}")
    
    # Оценка качества гибридного набора
    if hybrid_features:
        X_hybrid = X_train_engineered[hybrid_features].fillna(0)
        hybrid_scores = cross_val_score(
            model_for_comparison, X_hybrid, y_train,
            cv=3, scoring='neg_mean_absolute_error'
        )
        hybrid_mae = -hybrid_scores.mean()
        print(f"    MAE (гибридный набор): {hybrid_mae:.2f}")
    else:
        hybrid_mae = np.nan
        print("    Гибридный набор пуст.")
    
    # 8. Интерпретация отобранных признаков
    print("\n  Интерпретация отобранных признаков...")
    
    # Группируем признаки по типу трансформера
    feature_groups = {}
    for col in X_train_engineered.columns:
        if "window." in col:
            group = "window"
        elif "dwt." in col:
            group = "dwt"
        elif "stl." in col:
            group = "stl"
        elif "time." in col:
            group = "time"
        elif "calendar." in col:
            group = "calendar"
        else:
            group = "other"
        
        if group not in feature_groups:
            feature_groups[group] = []
        feature_groups[group].append(col)
    
    print(f"    Признаки по типу трансформера:")
    for group_name, features in feature_groups.items():
        print(f"      {group_name:10s}: {len(features):3d} признаков")
    
    return {
        "engineered_features_count": X_train_engineered.shape[1],
        "selected_features_count": X_test_engineered.shape[1],
        "test_mae": test_mae,
        "test_r2": test_r2,
        "comparison_methods_performance": perf_df,
        "block_performance": block_performance,
        "hybrid_performance": {
            "Method": f"hybrid_{hybrid_strategy}",
            "Mean_MAE": hybrid_mae,
            "Num_Features": len(hybrid_features)
        },
        "feature_groups": feature_groups,
    }


def main():
    print("=" * 80)
    print("СРАВНЕНИЕ СТРАТЕГИЙ ОТБОРА ПРИЗНАКОВ НА ДАННЫХ ТЕМПЕРАТУР")
    print("=" * 80)
    
    # Загрузка данных
    print("\n1. Загрузка данных минимальных суточных температур...")
    try:
        df = load_temperature_data()
    except FileNotFoundError as e:
        print(f"Ошибка: {e}")
        print("\nСовет: Убедитесь, что файл данных находится в директории data/")
        return
    
    # Анализ структуры ряда
    print("\n2. Анализ структуры временного ряда...")
    meta_features = analyze_meta_features_structure(df)
    
    # Визуализация паттернов
    print("\n3. Визуализация ключевых паттернов температур...")
    visualize_temperature_patterns(df)
    
    # Анализ выбора трансформеров
    insights = analyze_feature_selection_process(df, meta_features)
    
    # Сравнение стратегий отбора
    comparison_results = compare_selection_strategies_with_insights(df, insights)
    
    # Финальный отчет
    print("\n" + "=" * 80)
    print("ФИНАЛЬНЫЙ ОТЧЕТ")
    print("=" * 80)
    print(f"Сгенерировано признаков: {comparison_results['engineered_features_count']}")
    print(f"Отобрано признаков: {comparison_results['selected_features_count']}")
    print(f"MAE прогноза: {comparison_results['test_mae']:.2f}")
    print(f"R² прогноза: {comparison_results['test_r2']:.4f}")
    
    print(f"\nСравнение методов отбора:")
    perf_df = comparison_results['comparison_methods_performance']
    if not perf_df.empty:
        best_method = perf_df.iloc[0]['Method']
        best_mae = perf_df.iloc[0]['Mean_MAE']
        print(f"  Лучший метод отбора: {best_method} (MAE = {best_mae:.2f})")
    
    print(f"\nЭффективность блоков признаков:")
    block_perf = comparison_results['block_performance']
    if not block_perf.empty:
        best_block = block_perf.iloc[0]['Block']
        best_block_mae = block_perf.iloc[0]['Mean_MAE']
        print(f"  Лучший блок признаков: {best_block} (MAE = {best_block_mae:.2f})")
    
    print(f"\nГибридизация методов:")
    hybrid_perf = comparison_results['hybrid_performance']
    print(f"  Гибридный набор ({hybrid_perf['Method']}): MAE = {hybrid_perf['Mean_MAE']:.2f} ({hybrid_perf['Num_Features']} признаков)")
    
    print(f"\nИнсайты о структуре ряда:")
    print(f"  Месячная сезонность: {'ДА' if insights.get('monthly_seasonality', False) else 'НЕТ'}")
    print(f"  Годовая сезонность: {'ДА' if insights.get('yearly_seasonality', False) else 'НЕТ'}")
    print(f"  Стационарность: {insights.get('stationarity', 'UNKNOWN')}")
    print(f"  Сложность: {insights.get('complexity', 'UNKNOWN')}")
    print(f"  Нормальность: {insights.get('normality', 'UNKNOWN')}")
    print(f"  Автокорреляция: {insights.get('autocorrelation', 'UNKNOWN')}")
    print(f"  Гомоскедастичность: {insights.get('homoskedasticity', 'UNKNOWN')}")
    
    print(f"\nКоличество признаков по типу:")
    for group, count in comparison_results['feature_groups'].items():
        print(f"  {group:10s}: {len(count):3d}")
    
    print("\nСгенерированные файлы:")
    print("  • temperature_patterns.png")
    print("  • shap_feature_importance.png")
    print("  • feature_correlation_matrix.png")
    print("=" * 80)


if __name__ == "__main__":
    main()