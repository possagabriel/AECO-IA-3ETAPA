"""Testes de persistência de modelos."""

from pathlib import Path

import numpy as np
import pytest

from iris_nn.data import IrisDataset
from iris_nn.experiment import ExperimentResult
from iris_nn.persistence import load_model, save_model


def test_save_and_load_model_round_trip(
    tmp_path: Path,
    experiment_result: ExperimentResult,
    dataset: IrisDataset,
) -> None:
    saved_path = save_model(tmp_path / "model", experiment_result.model)

    assert saved_path == tmp_path / "model.npz"
    loaded = load_model(saved_path)

    assert loaded.n_inputs == experiment_result.model.n_inputs
    assert loaded.n_hidden == experiment_result.model.n_hidden
    assert loaded.n_outputs == experiment_result.model.n_outputs
    assert loaded.learning_rate == experiment_result.model.learning_rate
    np.testing.assert_allclose(
        loaded.predict_proba(dataset.X_test),
        experiment_result.model.predict_proba(dataset.X_test),
    )


def test_load_model_accepts_path_without_suffix(
    tmp_path: Path,
    experiment_result: ExperimentResult,
) -> None:
    save_model(tmp_path / "model", experiment_result.model)

    loaded = load_model(tmp_path / "model")

    assert loaded.n_hidden == experiment_result.model.n_hidden


def test_load_missing_model_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="não encontrado"):
        load_model(tmp_path / "missing")


def test_load_incompatible_format_raises(tmp_path: Path) -> None:
    path = tmp_path / "invalid.npz"
    np.savez(path, format_version=np.array(99, dtype=np.int64))

    with pytest.raises(ValueError, match="formato"):
        load_model(path)
