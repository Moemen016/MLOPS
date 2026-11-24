"""Atelier 3 - Pipeline MLOps avec Prefect."""
# pylint: disable=missing-function-docstring

import argparse
import subprocess  # nosec B404
import sys

from prefect import flow, task, get_run_logger

from model_pipeline import (
    prepare_data,
    train_model,
    save_model,
    load_model,
    evaluate_model,
    predict,
)

# Only scan our own files (otherwise flake8/pylint/bandit scan the whole venv)
PROJECT_FILES = ["model_pipeline.py", "main.py", "pipeline_prefect.py"]


def run_cmd(cmd, check=True):
    """Run a shell command, log its output, optionally raise on failure."""
    logger = get_run_logger()
    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(  # nosec B603
        cmd, capture_output=True, text=True, check=False
    )
    if result.stdout:
        logger.info(result.stdout)
    if result.stderr:
        logger.warning(result.stderr)
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(cmd)}")
    return result.returncode


# ------------------------------------------------------------- TASKS (env)
@task(name="Fetch project (git pull)")
def fetch_project():
    run_cmd(["git", "pull", "--ff-only"], check=False)


@task(name="Install dependencies")
def install_dependencies():
    run_cmd([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])


# ------------------------------------------------------------- TASKS (code)
@task(name="Format code (black)")
def format_code():
    run_cmd([sys.executable, "-m", "black", *PROJECT_FILES], check=False)


@task(name="Code quality (flake8 + pylint)")
def code_quality():
    run_cmd(
        [sys.executable, "-m", "flake8", "--max-line-length=100", *PROJECT_FILES],
        check=False,
    )
    run_cmd([sys.executable, "-m", "pylint", *PROJECT_FILES], check=False)


@task(name="Code security (bandit)")
def code_security():
    run_cmd([sys.executable, "-m", "bandit", "-q", *PROJECT_FILES], check=False)


@task(name="Unit tests (pytest)")
def unit_tests():
    run_cmd([sys.executable, "-m", "pytest", "tests/", "-v"])


# ------------------------------------------------------------- TASKS (model)
@task(name="Prepare data")
def prepare_data_task():
    return prepare_data()


@task(name="Train model")
def train_model_task(data):
    x_train, _x_test, y_train, _y_test = data
    return train_model(x_train, y_train)


@task(name="Save model")
def save_model_task(model, path="model.joblib"):
    save_model(model, path)
    return path


@task(name="Load model")
def load_model_task(path="model.joblib"):
    return load_model(path)


@task(name="Evaluate model")
def evaluate_model_task(model, data):
    logger = get_run_logger()
    _x_train, x_test, _y_train, y_test = data
    accuracy = evaluate_model(model, x_test, y_test)
    logger.info("Accuracy: %s", accuracy)
    return accuracy


@task(name="Predict")
def predict_task(data):
    _x_train, x_test, _y_train, _y_test = data
    predictions = predict(x_test)
    get_run_logger().info("First 10 predictions: %s", predictions[:10])
    return predictions


# ----------------------------------------------------------------------- FLOWS
@flow(name="install")
def install_flow():
    install_dependencies()


@flow(name="code")
def code_flow():
    format_code()
    code_quality()
    code_security()
    unit_tests()


@flow(name="train")
def train_flow():
    data = prepare_data_task()
    model = train_model_task(data)
    save_model_task(model)


@flow(name="evaluate")
def evaluate_flow():
    data = prepare_data_task()
    model = load_model_task()
    evaluate_model_task(model, data)


@flow(name="predict")
def predict_flow():
    data = prepare_data_task()
    predict_task(data)


@flow(name="all")
def all_flow():
    fetch_project()
    install_dependencies()
    format_code()
    code_quality()
    code_security()
    unit_tests()
    data = prepare_data_task()
    model = train_model_task(data)
    save_model_task(model)
    evaluate_model_task(model, data)


FLOWS = {
    "install": install_flow,
    "code": code_flow,
    "train": train_flow,
    "entrainement": train_flow,  # alias used in the handout
    "evaluate": evaluate_flow,
    "predict": predict_flow,
    "all": all_flow,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a Prefect flow")
    parser.add_argument("--flow", choices=FLOWS.keys(), default="all")
    args = parser.parse_args()
    FLOWS[args.flow]()
