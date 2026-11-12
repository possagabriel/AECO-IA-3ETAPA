"""Fixtures compartilhadas entre os testes."""

import pytest

from iris_nn.data import IrisDataset, load_prepared_iris


@pytest.fixture(scope="session")
def dataset() -> IrisDataset:
    """Dataset Iris preparado com a configuração padrão."""
    return load_prepared_iris(test_size=0.2, random_state=42)
