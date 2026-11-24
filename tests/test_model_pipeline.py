"""Example unit tests - extend them with real checks on your data/model."""

import model_pipeline


def test_expected_functions_exist():
    for name in ("prepare_data", "train_model", "save_model", "load_model", "evaluate_model"):
        assert callable(getattr(model_pipeline, name))
