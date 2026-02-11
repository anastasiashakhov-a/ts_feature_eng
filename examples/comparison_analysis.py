# examples/comparison_analysis.py
"""
Сравнение стратегий отбора признаков на реальных данных из Марокко.

Анализирует:
- Структуру временного ряда через мета-признаки
- Выбор оптимальных методов инженерии признаков
- Влияние отбора на качество прогноза
- Интерпретацию важности признаков через SHAP
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
from sklearn.model_selection import TimeSeriesSplit

from ts_feature_eng import AutoFeatureEngineer
from ts_feature_eng.meta_features import MetaFeatureExtractor
from ts_feature_eng.selection import CombinedFeatureSelector


def load_morocco_data(data_path=None):
    """Загрузка данных энергопотребления Марокко."""
    if data_path is None:
        possible_paths = [
            "data/morocco zone 1 - powerconsumption_resampled (1).csv",
            "data/morocco_zone_1_powerconsumption_resampled.csv",
            "../data/morocco zone 1 - powerconsumption_resampled (1).csv",
            os.path.expanduser("~/ts_feature_eng/data/morocco zone 1 - powerconsumption_resampled (1).csv"),
            os.path.join(os.path.dirname(__file__), "..", "data", "morocco zone 1 - powerconsumption_resampled (1).csv"),
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
    consumption_col = None
    
    for col in df.columns:
        if any(keyword in col.lower() for keyword in ["datetime", "date", "time", "timestamp"]):
            time_col = col
        if any(keyword in col.lower() for keyword in ["consumption", "power", "load", "energy"]):
            consumption_col = col
    
    if time_col is None:
        time_col = df.columns[0]
    
    if consumption_col is None:
        consumption_col = df.columns[1]
    
    result = pd.DataFrame({
        "timestamp": pd.to_datetime(df[time_col]),
        "consumption": pd.to_numeric(df[consumption_col], errors="coerce")
    })
    
    result = result.set_index("timestamp")
    initial_len = len(result)
    result = result.dropna(subset=["consumption"])
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
    
    y = df["consumption"].shift(-1).dropna()
    X = df.iloc[:-1][["consumption"]]
    
    meta_df = extractor.fit_transform(X, y)
    meta_features = meta_df.iloc[0].to_dict()
    
    print(f"Извлечено {len(meta_features)} мета-признаков")
    
    # Ключевые мета-признаки
    print("\nКлючевые мета-признаки:")
    print(f"  Длина ряда: {meta_features.get('length', 'N/A'):,.0f}")
    print(f"  Пропуски: {meta_features.get('missing_ratio', 0)*100:.2f}%")
    print(f"  Стационарность (ADF p-value): {meta_features.get('stationarity_adf', 1.0):.3f}")
    print(f"  Линейность (R²): {meta_features.get('linearity', 0):.3f}")
    print(f"  Суточная ACF: {meta_features.get('acf_24', 0):.3f}")
    print(f"  Недельная ACF: {meta_features.get('acf_168', 0):.3f}")
    print(f"  Доминирующая частота: {meta_features.get('dominant_freq', 0):.4f}")
    
    return meta_features


def visualize_consumption_patterns(df):
    """Визуализация ключевых паттернов потребления."""
    print("\nВизуализация паттернов потребления...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Общий тренд
    axes[0, 0].plot(df.index, df["consumption"], color='blue', linewidth=0.8, alpha=0.7)
    axes[0, 0].set_title("Общий тренд энергопотребления")
    axes[0, 0].set_ylabel("Потребление (МВт)")
    axes[0, 0].grid(True, alpha=0.3)
    
    # Суточной паттерн
    hourly_avg = df.groupby(df.index.hour)["consumption"].mean()
    hourly_std = df.groupby(df.index.hour)["consumption"].std()
    axes[0, 1].plot(hourly_avg.index, hourly_avg.values, color='darkgreen', linewidth=2, label='Среднее')
    axes[0, 1].fill_between(
        hourly_avg.index,
        hourly_avg.values - hourly_std.values,
        hourly_avg.values + hourly_std.values,
        alpha=0.3,
        color='green',
        label='±1 std'
    )
    axes[0, 1].set_title("Суточной паттерн потребления")
    axes[0, 1].set_xlabel("Час суток")
    axes[0, 1].set_ylabel("Потребление (МВт)")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Недельный паттерн
    weekday_avg = df.groupby(df.index.dayofweek)["consumption"].mean()
    weekday_labels = ['Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб', 'Вс']
    axes[1, 0].bar(weekday_labels, weekday_avg.values, color='steelblue', alpha=0.8)
    axes[1, 0].set_title("Недельный паттерн потребления")
    axes[1, 0].set_ylabel("Потребление (МВт)")
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Распределение
    axes[1, 1].hist(df["consumption"], bins=50, density=True, alpha=0.7, color='purple', edgecolor='black')
    axes[1, 1].axvline(df["consumption"].mean(), color='red', linestyle='--', label=f'Среднее: {df["consumption"].mean():.0f}')
    axes[1, 1].axvline(df["consumption"].median(), color='orange', linestyle='--', label=f'Медиана: {df["consumption"].median():.0f}')
    axes[1, 1].set_title("Распределение энергопотребления")
    axes[1, 1].set_xlabel("Потребление (МВт)")
    axes[1, 1].set_ylabel("Плотность")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle("Паттерны энергопотребления в Марокко (Зона 1)")
    plt.tight_layout()
    plt.savefig("morocco_energy_patterns.png", dpi=150, bbox_inches='tight')
    print("  График сохранен как 'morocco_energy_patterns.png'")
    plt.show()


def analyze_feature_selection_process(df, meta_features):
    """Анализ выбора трансформеров на основе мета-признаков."""
    print("\nАнализ выбора трансформеров...")
    
    insights = {}
    
    # Сезонность
    acf_24 = meta_features.get('acf_24', 0)
    acf_168 = meta_features.get('acf_168', 0)
    dom_freq = meta_features.get('dominant_freq', 0)
    
    print(f"\nСезонность:")
    if acf_24 > 0.3:
        print(f"  Суточная сезонность: ACF(24) = {acf_24:.3f} (сильная)")
        insights["daily_seasonality"] = True
    else:
        print(f"  Суточная сезонность: ACF(24) = {acf_24:.3f} (слабая)")
        insights["daily_seasonality"] = False
    
    if acf_168 > 0.3:
        print(f"  Недельная сезонность: ACF(168) = {acf_168:.3f} (сильная)")
        insights["weekly_seasonality"] = True
    else:
        print(f"  Недельная сезонность: ACF(168) = {acf_168:.3f} (слабая)")
        insights["weekly_seasonality"] = False
    
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
    
    return insights


def compare_selection_strategies(df, insights):
    """Сравнение стратегий отбора признаков."""
    print("\nСравнение стратегий отбора признаков...")
    
    y = df["consumption"].shift(-1).dropna()
    X = df.iloc[:-1][["consumption"]]
    
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Автоматическая инженерия признаков
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
    
    # Обучение модели
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
    
    # SHAP-анализ
    print("\n  SHAP-анализ важности признаков...")
    try:
        import shap
        
        sample_size = min(1000, len(X_train_engineered))
        X_sample = X_train_engineered.fillna(0).iloc[:sample_size]
        y_sample = y_train.iloc[:sample_size]
        
        model_shap = RandomForestRegressor(n_estimators=50, max_depth=6, random_state=42)
        model_shap.fit(X_sample, y_sample)
        
        explainer = shap.TreeExplainer(model_shap)
        shap_values = explainer(X_sample)
        
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
        plt.show()
        
        shap_importance = np.abs(shap_values.values).mean(0)
        feature_names = X_sample.columns
        top_indices = np.argsort(shap_importance)[-10:][::-1]
        
        print(f"\n    Топ-10 важных признаков (по SHAP):")
        for i, idx in enumerate(top_indices):
            print(f"      {i+1:2d}. {feature_names[idx]:30s} | {shap_importance[idx]:.4f}")
    
    except ImportError:
        print("    SHAP не установлен. Установите через: pip install shap")
    
    # Анализ корреляций
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
    plt.show()
    
    # Группировка признаков
    print("\n  Группировка признаков по типу трансформера:")
    
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
    
    for group_name, features in feature_groups.items():
        print(f"    {group_name:10s}: {len(features):3d} признаков")
    
    # Сравнение с базовой стратегией
    print("\n  Сравнение с базовой стратегией отбора...")
    
    correlations = X_train_engineered.corrwith(y_train).abs().sort_values(ascending=False)
    top_corr_features = correlations.head(20).index.tolist()
    
    model_corr = RandomForestRegressor(n_estimators=50, random_state=42)
    model_corr.fit(X_train_engineered[top_corr_features].fillna(0), y_train)
    y_pred_corr = model_corr.predict(X_test_engineered[top_corr_features].fillna(0))
    corr_mae = mean_absolute_error(y_test, y_pred_corr)
    
    print(f"    MAE (корреляция): {corr_mae:.2f}")
    print(f"    MAE (авто-инж. + SHAP): {test_mae:.2f}")
    print(f"    Улучшение: {((corr_mae - test_mae) / corr_mae * 100):+.2f}%")
    
    return {
        "engineered_features_count": X_train_engineered.shape[1],
        "selected_features_count": X_test_engineered.shape[1],
        "test_mae": test_mae,
        "test_r2": test_r2,
        "correlation_baseline_mae": corr_mae,
        "improvement_over_baseline": ((corr_mae - test_mae) / corr_mae * 100),
        "feature_groups": feature_groups,
    }


def main():
    print("=" * 80)
    print("СРАВНЕНИЕ СТРАТЕГИЙ ОТБОРА ПРИЗНАКОВ НА ДАННЫХ ИЗ МАРОККО")
    print("=" * 80)
    
    # Загрузка данных
    print("\n1. Загрузка данных энергопотребления Марокко...")
    try:
        df = load_morocco_data()
    except FileNotFoundError as e:
        print(f"Ошибка: {e}")
        print("\nСовет: Убедитесь, что файл данных находится в директории data/")
        return
    
    # Анализ структуры ряда
    print("\n2. Анализ структуры временного ряда...")
    meta_features = analyze_meta_features_structure(df)
    
    # Визуализация паттернов
    print("\n3. Визуализация ключевых паттернов потребления...")
    visualize_consumption_patterns(df)
    
    # Анализ выбора трансформеров
    insights = analyze_feature_selection_process(df, meta_features)
    
    # Сравнение стратегий отбора
    comparison_results = compare_selection_strategies(df, insights)
    
    # Финальный отчет
    print("\n" + "=" * 80)
    print("ФИНАЛЬНЫЙ ОТЧЕТ")
    print("=" * 80)
    print(f"Сгенерировано признаков: {comparison_results['engineered_features_count']}")
    print(f"Отобрано признаков: {comparison_results['selected_features_count']}")
    print(f"MAE прогноза: {comparison_results['test_mae']:.2f}")
    print(f"R² прогноза: {comparison_results['test_r2']:.4f}")
    print(f"Улучшение от автоматической инженерии: {comparison_results['improvement_over_baseline']:+.2f}%")
    
    print(f"\nИнсайты о структуре ряда:")
    print(f"  Суточная сезонность: {'ДА' if insights.get('daily_seasonality', False) else 'НЕТ'}")
    print(f"  Недельная сезонность: {'ДА' if insights.get('weekly_seasonality', False) else 'НЕТ'}")
    print(f"  Стационарность: {insights.get('stationarity', 'UNKNOWN')}")
    print(f"  Сложность: {insights.get('complexity', 'UNKNOWN')}")
    
    print(f"\nКоличество признаков по типу:")
    for group, count in comparison_results['feature_groups'].items():
        print(f"  {group:10s}: {len(count):3d}")
    
    print("\nСгенерированные файлы:")
    print("  • morocco_energy_patterns.png")
    print("  • shap_feature_importance.png")
    print("  • feature_correlation_matrix.png")
    print("=" * 80)


if __name__ == "__main__":
    main()