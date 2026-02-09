# tests/test_base.py 
"""
Тесты базовых абстрактных классов TimeSeriesTransformer и FeatureSelector.

Проверяют:
- Корректность интерфейса и наследования
- Валидацию входных данных
- Совместимость с экосистемой scikit-learn
- Обработку ошибок и крайних случаев
- Работу вспомогательных методов
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from ts_feature_eng.base import (
    FeatureSelector,
    TimeSeriesError,
    TimeSeriesTransformer,
)
from ts_feature_eng.transformers.window import WindowTransformer


# Фикстуры для тестовых данных
@pytest.fixture
def sample_time_series():
    """Создает тестовый временной ряд с временным индексом."""
    dates = pd.date_range("2023-01-01", periods=100, freq="h")  # Используем 'h' вместо 'H'
    return pd.DataFrame(
        {
            "value": np.sin(2 * np.pi * np.arange(100) / 24) + np.random.randn(100) * 0.1,
            "temperature": np.random.randn(100) * 10 + 20,
        },
        index=dates,
    )


@pytest.fixture
def sample_target():
    """Создает тестовую целевую переменную."""
    return pd.Series(np.random.randn(100), name="target")


@pytest.fixture
def invalid_inputs():
    """Создает различные невалидные входные данные для тестирования."""
    return [
        None,  # None вместо данных
        [],  # Пустой список
        pd.DataFrame(),  # Пустой DataFrame
        pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}).iloc[:0],  # DataFrame без строк
        np.array([]),  # Пустой массив
        np.array([[1, 2], [3, 4]])[:, :0],  # Массив без столбцов
    ]


# Тесты для исключения TimeSeriesError
def test_timeseries_error_inheritance():
    """Проверяет, что TimeSeriesError наследуется от ValueError."""
    with pytest.raises(ValueError):
        raise TimeSeriesError("Test error")


# Тесты для абстрактного класса TimeSeriesTransformer
class DummyTransformer(TimeSeriesTransformer):
    def __init__(self, param=1.0):  # Добавляем параметр
        super().__init__()
        self.param = param
        self.feature_names_ = []
    
    def fit(self, X, y=None):
        X = self._validate_input(X)
        # Установка имен признаков
        self._set_feature_names("dummy", ["mean", "std"])
        return self
    
    def transform(self, X):
        if not self.is_fitted_:
            raise TimeSeriesError("Transformer is not fitted. Call fit() first.")
        X = self._validate_input(X)
        # Генерация признаков
        features = {}
        for col in X.columns:
            features[f"{col}.mean"] = X[col].rolling(window=10, min_periods=1).mean()
            features[f"{col}.std"] = X[col].rolling(window=10, min_periods=1).std()
        return pd.DataFrame(features, index=X.index)
    
    def get_feature_names(self):
        if not self.is_fitted_:
            raise TimeSeriesError("Transformer is not fitted. Call fit() first.")
        return self.feature_names_


class TestTimeSeriesTransformer:
    """Тесты базового класса трансформеров временных рядов."""
    
    def test_inheritance(self):
        """Проверяет наследование от sklearn BaseEstimator и TransformerMixin."""
        transformer = DummyTransformer()
        assert isinstance(transformer, BaseEstimator)
        assert isinstance(transformer, TransformerMixin)
        assert isinstance(transformer, TimeSeriesTransformer)
    
    def test_initialization(self):
        """Проверяет корректную инициализацию атрибутов."""
        transformer = DummyTransformer()
        assert transformer.feature_names_ == []
        assert transformer.is_fitted_ is False
    
    def test_fit_transform_integration(self, sample_time_series):
        """Проверяет работу метода fit_transform."""
        transformer = DummyTransformer()
        result = transformer.fit_transform(sample_time_series)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_time_series)
        assert transformer.is_fitted_ is True
        assert len(transformer.get_feature_names()) > 0
    
    def test_get_feature_names_out(self, sample_time_series):
        """Проверяет совместимость с интерфейсом sklearn 1.0+."""
        transformer = DummyTransformer()
        transformer.fit(sample_time_series)
        
        feature_names = transformer.get_feature_names_out()
        assert isinstance(feature_names, np.ndarray)
        assert len(feature_names) == len(transformer.feature_names_)
        assert list(feature_names) == transformer.feature_names_
    
    def test_get_feature_names_before_fit(self):
        """Проверяет выброс исключения при вызове get_feature_names до fit."""
        transformer = DummyTransformer()
        with pytest.raises(TimeSeriesError, match="not fitted"):
            transformer.get_feature_names()
    
    def test_get_feature_names_out_before_fit(self):
        """Проверяет выброс исключения при вызове get_feature_names_out до fit."""
        transformer = DummyTransformer()
        with pytest.raises(TimeSeriesError, match="not fitted"):
            transformer.get_feature_names_out()
    
    @pytest.mark.parametrize("invalid_input", [
        None,
        [],
        pd.DataFrame(),
        np.array([]),
    ])
    def test_validate_input_invalid(self, invalid_input):
        """Проверяет валидацию некорректных входных данных."""
        transformer = DummyTransformer()
        with pytest.raises(TimeSeriesError):
            transformer._validate_input(invalid_input)
    
    def test_validate_input_valid_dataframe(self, sample_time_series):
        """Проверяет валидацию корректного DataFrame."""
        transformer = DummyTransformer()
        validated = transformer._validate_input(sample_time_series)
        
        assert isinstance(validated, pd.DataFrame)
        assert validated.shape == sample_time_series.shape
        assert validated.index.equals(sample_time_series.index)
    
    def test_validate_input_valid_array(self):
        """Проверяет валидацию корректного numpy массива."""
        transformer = DummyTransformer()
        array = np.random.randn(100, 3)
        validated = transformer._validate_input(array)
        
        assert isinstance(validated, pd.DataFrame)
        assert validated.shape == (100, 3)
        assert list(validated.columns) == ["feature_0", "feature_1", "feature_2"]
    
    def test_validate_input_1d_array(self):
        """Проверяет валидацию одномерного массива (автоматическое преобразование в 2D)."""
        transformer = DummyTransformer()
        array = np.random.randn(100)
        validated = transformer._validate_input(array)
        
        assert isinstance(validated, pd.DataFrame)
        assert validated.shape == (100, 1)
        assert validated.columns.tolist() == ["feature_0"]
    
    def test_validate_input_datetime_index_validation(self):
        """Проверяет валидацию временного индекса (дубликаты и не монотонность)."""
        transformer = DummyTransformer()
        
        # Тест с дубликатами в индексе
        dates = pd.date_range("2023-01-01", periods=50, freq="h")
        dates = dates.append(dates[:10])  # Добавляем дубликаты
        df_with_duplicates = pd.DataFrame({"value": np.random.randn(60)}, index=dates)
        
        with pytest.raises(TimeSeriesError, match="duplicate"):
            transformer._validate_input(df_with_duplicates)
        
        # Тест с не монотонным индексом
        dates = pd.date_range("2023-01-01", periods=50, freq="h")
        dates = dates[::-1]  # Разворачиваем для нарушения монотонности
        df_non_monotonic = pd.DataFrame({"value": np.random.randn(50)}, index=dates)
        
        with pytest.raises(TimeSeriesError, match="monotonically"):
            transformer._validate_input(df_non_monotonic)
    
    def test_set_feature_names(self):
        """Проверяет корректную установку имен признаков."""
        transformer = DummyTransformer()
        base_name = "test"
        statistics = ["mean", "std", "min", "max"]
        
        transformer._set_feature_names(base_name, statistics)
        
        assert transformer.feature_names_ == [
            "test.mean",
            "test.std",
            "test.min",
            "test.max",
        ]
        assert transformer.is_fitted_ is True
    
    def test_get_params_set_params_integration(self, sample_time_series):
        """Проверяет совместимость с интерфейсом sklearn (get_params/set_params)."""
        transformer = DummyTransformer(param=2.5)
        
        # Получение параметров
        params = transformer.get_params()
        assert "param" in params
        assert params["param"] == 2.5
        
        # Установка новых параметров
        transformer.set_params(param=3.7)
        assert transformer.param == 3.7
        
        # Проверка, что трансформер все еще работает после изменения параметров
        transformer.fit(sample_time_series)
        assert transformer.is_fitted_ is True
    
    def test_transform_without_fit(self, sample_time_series):
        """Проверяет выброс исключения при вызове transform до fit."""
        transformer = DummyTransformer()
        with pytest.raises(TimeSeriesError, match="not fitted"):
            transformer.transform(sample_time_series)


# Тесты для абстрактного класса FeatureSelector
class DummySelector(FeatureSelector):
    """Простая реализация абстрактного класса селектора для тестирования."""
    
    def __init__(self, n_features_to_select=2):
        super().__init__()
        self.n_features_to_select = n_features_to_select
    
    def fit(self, X, y):
        X, y = self._validate_input(X, y)
        
        # Простая эвристика: выбираем признаки с наибольшей дисперсией
        variances = X.var()
        top_features = variances.nlargest(self.n_features_to_select).index.tolist()
        
        self.selected_features_ = top_features
        self.feature_importances_ = np.array([
            1.0 if col in top_features else 0.0 for col in X.columns
        ])
        self.is_fitted_ = True
        
        return self
    
    def transform(self, X):
        if not self.is_fitted_:
            raise TimeSeriesError("Selector is not fitted. Call fit() first.")
        
        missing_features = set(self.selected_features_) - set(X.columns)
        if missing_features:
            raise TimeSeriesError(f"Missing features: {missing_features}")
        
        return X[self.selected_features_]
    
    def get_selected_features(self):
        if not self.is_fitted_:
            raise TimeSeriesError("Selector is not fitted. Call fit() first.")
        return self.selected_features_


class TestFeatureSelector:
    """Тесты базового класса селекторов признаков."""
    
    def test_inheritance(self):
        """Проверяет наследование от sklearn BaseEstimator и TransformerMixin."""
        selector = DummySelector()
        assert isinstance(selector, BaseEstimator)
        assert isinstance(selector, TransformerMixin)
        assert isinstance(selector, FeatureSelector)
    
    def test_initialization(self):
        """Проверяет корректную инициализацию атрибутов."""
        selector = DummySelector()
        assert selector.selected_features_ == []
        assert selector.feature_importances_ is None
        assert selector.is_fitted_ is False
    
    def test_fit_transform_integration(self, sample_time_series, sample_target):
        """Проверяет работу метода fit_transform."""
        selector = DummySelector(n_features_to_select=2)
        result = selector.fit_transform(sample_time_series, sample_target)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result.columns) == 2
        assert set(result.columns).issubset(set(sample_time_series.columns))
        assert selector.is_fitted_ is True
    
    def test_get_selected_features(self, sample_time_series, sample_target):
        """Проверяет получение списка отобранных признаков."""
        selector = DummySelector(n_features_to_select=2)
        selector.fit(sample_time_series, sample_target)
        
        selected = selector.get_selected_features()
        assert isinstance(selected, list)
        assert len(selected) == 2
        assert all(isinstance(feat, str) for feat in selected)
    
    def test_get_selected_features_before_fit(self):
        """Проверяет выброс исключения при вызове до fit."""
        selector = DummySelector()
        with pytest.raises(TimeSeriesError, match="not fitted"):
            selector.get_selected_features()
    
    def test_get_support(self, sample_time_series, sample_target):
        """Проверяет работу метода get_support (совместимость с sklearn)."""
        selector = DummySelector(n_features_to_select=2)
        selector.fit(sample_time_series, sample_target)
        
        # Булева маска
        mask = selector.get_support()
        assert isinstance(mask, list)
        assert len(mask) == len(sample_time_series.columns)
        assert sum(mask) == 2  # Ровно 2 признака отобрано
        
        # Индексы
        indices = selector.get_support(indices=True)
        assert isinstance(indices, np.ndarray)
        assert len(indices) == 2
        assert all(0 <= idx < len(sample_time_series.columns) for idx in indices)
    
    def test_get_support_before_fit(self):
        """Проверяет выброс исключения при вызове get_support до fit."""
        selector = DummySelector()
        with pytest.raises(TimeSeriesError, match="not fitted"):
            selector.get_support()
    
    @pytest.mark.parametrize("invalid_input", [
        (None, pd.Series(np.random.randn(100))),
        (pd.DataFrame(np.random.randn(100, 3)), None),
        (pd.DataFrame(np.random.randn(50, 3)), pd.Series(np.random.randn(100))),  # Несовпадение длин
    ])
    def test_validate_input_invalid(self, invalid_input):
        """Проверяет валидацию некорректных входных данных."""
        selector = DummySelector()
        X, y = invalid_input
        with pytest.raises(TimeSeriesError):
            selector._validate_input(X, y)
    
    def test_validate_input_valid(self, sample_time_series, sample_target):
        """Проверяет валидацию корректных входных данных."""
        selector = DummySelector()
        X_validated, y_validated = selector._validate_input(
            sample_time_series, sample_target
        )
        
        assert isinstance(X_validated, pd.DataFrame)
        assert isinstance(y_validated, np.ndarray)
        assert len(X_validated) == len(y_validated)
        assert hasattr(selector, "feature_names_in_")
        assert selector.feature_names_in_ == list(sample_time_series.columns)
    
    def test_transform_without_fit(self, sample_time_series):
        """Проверяет выброс исключения при вызове transform до fit."""
        selector = DummySelector()
        with pytest.raises(TimeSeriesError, match="not fitted"):
            selector.transform(sample_time_series)
    
    def test_transform_missing_features(self, sample_time_series, sample_target):
        """Проверяет обработку отсутствующих признаков при трансформации."""
        selector = DummySelector(n_features_to_select=2)
        selector.fit(sample_time_series, sample_target)
        
        # Создаем новый DataFrame без одного из отобранных признаков
        missing_feature = selector.selected_features_[0]
        X_test = sample_time_series.drop(columns=[missing_feature])
        
        with pytest.raises(TimeSeriesError, match="Missing features"):  
            selector.transform(X_test)
    
    def test_get_params_set_params_integration(self, sample_time_series, sample_target):
        """Проверяет совместимость с интерфейсом sklearn (get_params/set_params)."""
        selector = DummySelector(n_features_to_select=3)
        
        # Получение параметров
        params = selector.get_params()
        assert "n_features_to_select" in params
        assert params["n_features_to_select"] == 3
        
        # Установка новых параметров
        selector.set_params(n_features_to_select=5)
        assert selector.n_features_to_select == 5
        
        # Проверка, что селектор все еще работает после изменения параметров
        selector.fit(sample_time_series, sample_target)
        expected_n = min(5, len(sample_time_series.columns))
        assert len(selector.get_selected_features()) == expected_n


# Интеграционные тесты
class TestIntegration:
    """Интеграционные тесты базовых классов."""
    
    def test_window_transformer_inheritance(self):
        """Проверяет, что конкретные трансформеры наследуются от базового класса."""
        transformer = WindowTransformer(window_size=24)
        assert isinstance(transformer, TimeSeriesTransformer)
        assert isinstance(transformer, BaseEstimator)
        assert isinstance(transformer, TransformerMixin)
    
    def test_error_propagation(self, invalid_inputs):
        """Проверяет корректную обработку ошибок во всех методах."""
        transformer = DummyTransformer()
        
        for invalid_input in invalid_inputs:
            with pytest.raises(TimeSeriesError):
                transformer.fit(invalid_input)
            
            # После неудачного fit трансформер остается необученным
            assert transformer.is_fitted_ is False
    
    def test_non_numeric_columns_handling(self):
        """Проверяет обработку нечисловых столбцов."""
        transformer = DummyTransformer()
        
        # DataFrame с нечисловыми столбцами
        df_mixed = pd.DataFrame({
            "numeric": np.random.randn(10),
            "categorical": ["A", "B", "C", "D", "E"] * 2,
        })
        
        # Валидация должна пройти (фильтрация нечисловых столбцов происходит в конкретных трансформерах)
        validated = transformer._validate_input(df_mixed)
        assert isinstance(validated, pd.DataFrame)
        assert "numeric" in validated.columns
        # Столбец "categorical" сохраняется, но конкретные трансформеры могут его игнорировать


# Тесты крайних случаев
class TestEdgeCases:
    """Тесты для крайних случаев и граничных условий."""
    
    def test_empty_dataframe_after_validation(self):
        """Проверяет обработку пустого DataFrame после валидации."""
        transformer = DummyTransformer()
        
        # Пустой DataFrame
        empty_df = pd.DataFrame()
        
        with pytest.raises(TimeSeriesError, match="empty"):
            transformer._validate_input(empty_df)
    
    def test_single_row_dataframe(self):
        """Проверяет обработку DataFrame с одной строкой."""
        transformer = DummyTransformer()
        
        df_single = pd.DataFrame({"value": [1.0]})
        
        # Валидация должна пройти
        validated = transformer._validate_input(df_single)
        assert isinstance(validated, pd.DataFrame)
        assert len(validated) == 1
        
        # Но трансформация может вернуть пустой результат из-за оконных операций
        transformer.fit(df_single)
        result = transformer.transform(df_single)
        # Ожидаем, что результат будет иметь ту же длину, но возможно с пропусками
        assert len(result) == 1
    
    def test_inf_values_handling(self):
        """Проверяет обработку бесконечных значений."""
        transformer = DummyTransformer()
        
        df_with_inf = pd.DataFrame({
            "a": [1.0, 2.0, np.inf, 4.0],
            "b": [5.0, -np.inf, 7.0, 8.0],
        })
        
        with pytest.raises(TimeSeriesError, match="infinite values"):
            transformer._validate_input(df_with_inf)
    
    def test_complex_index_types(self):
        """Проверяет обработку разных типов индексов."""
        transformer = DummyTransformer()
        
        # RangeIndex
        df_range = pd.DataFrame(np.random.randn(10, 2))
        validated = transformer._validate_input(df_range)
        assert isinstance(validated.index, pd.RangeIndex)
        
        # Для Int64Index используйте проверку типа
        df_int = pd.DataFrame(np.random.randn(10, 2), index=np.arange(10, 20))
        validated = transformer._validate_input(df_int)
        # Проверка на целочисленный тип индекса
        assert hasattr(validated.index, 'dtype') and np.issubdtype(validated.index.dtype, np.integer)
        
        # DatetimeIndex
        dates = pd.date_range("2023-01-01", periods=10, freq="h") 
        df_dt = pd.DataFrame(np.random.randn(10, 2), index=dates)
        validated = transformer._validate_input(df_dt)
        assert isinstance(validated.index, pd.DatetimeIndex)