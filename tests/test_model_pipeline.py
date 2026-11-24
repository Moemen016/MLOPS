"""Unit tests for each function of model_pipeline."""

import numpy as np
import pandas as pd
import pytest

import model_pipeline as mp


@pytest.fixture
def paths(tmp_path):
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame(
        {
            "RowNumber": range(1, n + 1),
            "CustomerId": range(1000, 1000 + n),
            "Surname": ["A"] * n,
            "CreditScore": rng.integers(350, 850, n),
            "Geography": rng.choice(["France", "Spain"], n),
            "Gender": rng.choice(["Male", "Female"], n),
            "Age": rng.integers(18, 80, n),
            "Tenure": rng.integers(0, 11, n),
            "Balance": rng.uniform(0, 250000, n),
            "NumOfProducts": rng.integers(1, 4, n),
            "HasCrCard": rng.integers(0, 2, n),
            "IsActiveMember": rng.integers(0, 2, n),
            "EstimatedSalary": rng.uniform(10000, 200000, n),
            "Exited": rng.integers(0, 2, n),
        }
    )
    csv = tmp_path / "churn.csv"
    df.to_csv(csv, index=False)
    return str(csv), str(tmp_path / "scaler.joblib"), str(tmp_path / "model.joblib")


def test_prepare_data(paths):
    csv, scaler, _ = paths
    x_train, x_test, y_train, y_test = mp.prepare_data(csv, scaler)
    assert len(x_train) == 160 and len(x_test) == 40
    assert len(y_train) == 160 and len(y_test) == 40


def test_train_model(paths):
    csv, scaler, _ = paths
    x_train, _, y_train, _ = mp.prepare_data(csv, scaler)
    model = mp.train_model(x_train, y_train)
    assert hasattr(model, "predict")


def test_evaluate_model(paths):
    csv, scaler, _ = paths
    x_train, x_test, y_train, y_test = mp.prepare_data(csv, scaler)
    model = mp.train_model(x_train, y_train)
    accuracy = mp.evaluate_model(model, x_test, y_test)
    assert 0.0 <= accuracy <= 1.0


def test_save_and_load_model(paths):
    csv, scaler, model_path = paths
    x_train, x_test, y_train, _ = mp.prepare_data(csv, scaler)
    model = mp.train_model(x_train, y_train)
    mp.save_model(model, model_path)
    loaded = mp.load_model(model_path)
    assert (model.predict(x_test) == loaded.predict(x_test)).all()
