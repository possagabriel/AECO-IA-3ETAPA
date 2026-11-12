"""Testes da orquestração de experimentos."""

import numpy as np
import pytest

from iris_nn.experiment import ExperimentConfig, ExperimentResult, run_experiment


def test_experiment_result_is_consistent(experiment_result: ExperimentResult) -> None:
    result = experiment_result

    assert len(result.history) == result.config.epochs
    assert result.initial_loss == result.history[0]
    assert result.final_loss == result.history[-1]
    assert result.final_loss < result.initial_loss
    assert result.n_parameters == 67
    assert result.elapsed_seconds >= 0.0
    assert result.dataset.n_train == 120
    assert result.dataset.n_test == 30
    assert result.train_metrics.n_samples == 120
    assert result.test_metrics.n_samples == 30
    assert 0.0 <= result.test_metrics.accuracy <= 1.0


def test_experiment_is_reproducible() -> None:
    config = ExperimentConfig(epochs=25, log_every=25, verbose=False)
    first = run_experiment(config)
    second = run_experiment(config)

    np.testing.assert_allclose(first.history, second.history)
    np.testing.assert_array_equal(
        first.model.predict(first.dataset.X_test),
        second.model.predict(second.dataset.X_test),
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"epochs": 0},
        {"learning_rate": 0.0},
        {"learning_rate": float("inf")},
        {"hidden_size": 0},
        {"test_size": 0.0},
        {"test_size": 1.0},
        {"log_every": 0},
    ],
)
def test_experiment_config_rejects_invalid_values(
    overrides: dict[str, float],
) -> None:
    with pytest.raises(ValueError):
        ExperimentConfig(**overrides)  # type: ignore[arg-type]
