"""Fixtures compartilhadas entre os testes."""

import pytest

from iris_nn.data import IrisDataset, load_prepared_iris
from iris_nn.experiment import ExperimentConfig, ExperimentResult, run_experiment


@pytest.fixture(scope="session")
def dataset() -> IrisDataset:
    """Dataset Iris preparado com a configuração padrão."""
    return load_prepared_iris(test_size=0.2, random_state=42)


@pytest.fixture(scope="session")
def experiment_result() -> ExperimentResult:
    """Experimento curto reutilizado por testes de integração."""
    return run_experiment(ExperimentConfig(epochs=100, verbose=False))
