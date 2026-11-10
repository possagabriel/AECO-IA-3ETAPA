"""Testes do carregamento e pré-processamento do Iris."""

import numpy as np
import pytest

from iris_nn.data import load_prepared_iris, to_one_hot


def test_to_one_hot_creates_expected_matrix() -> None:
    encoded = to_one_hot(np.array([0, 2, 1]), n_classes=3)

    expected = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ]
    )
    np.testing.assert_array_equal(encoded, expected)


@pytest.mark.parametrize("labels", [np.array([0, 3]), np.array([-1, 1])])
def test_to_one_hot_rejects_out_of_range_labels(labels: np.ndarray) -> None:
    with pytest.raises(ValueError):
        to_one_hot(labels, n_classes=3)


def test_to_one_hot_rejects_non_vector_input() -> None:
    with pytest.raises(ValueError):
        to_one_hot(np.array([[0, 1], [1, 0]]), n_classes=2)


def test_dataset_shapes_and_class_balance() -> None:
    dataset = load_prepared_iris(test_size=0.2, random_state=42)

    assert dataset.X_train.shape == (120, 4)
    assert dataset.X_test.shape == (30, 4)
    assert dataset.y_train.shape == (120, 3)
    assert dataset.y_test.shape == (30, 3)
    np.testing.assert_array_equal(dataset.y_train.sum(axis=1), np.ones(120))
    np.testing.assert_array_equal(dataset.y_test.sum(axis=1), np.ones(30))
    np.testing.assert_array_equal(
        np.bincount(np.argmax(dataset.y_test, axis=1)),
        np.array([10, 10, 10]),
    )


def test_training_features_are_standardized() -> None:
    dataset = load_prepared_iris(test_size=0.2, random_state=42)

    np.testing.assert_allclose(dataset.X_train.mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(dataset.X_train.std(axis=0), 1.0, atol=1e-12)


def test_dataset_loading_is_reproducible() -> None:
    first = load_prepared_iris(test_size=0.2, random_state=7)
    second = load_prepared_iris(test_size=0.2, random_state=7)

    np.testing.assert_array_equal(first.X_train, second.X_train)
    np.testing.assert_array_equal(first.X_test, second.X_test)
    np.testing.assert_array_equal(first.y_train, second.y_train)
    np.testing.assert_array_equal(first.y_test, second.y_test)


@pytest.mark.parametrize("test_size", [0.0, 1.0, -0.1, 1.5])
def test_invalid_test_size_raises(test_size: float) -> None:
    with pytest.raises(ValueError):
        load_prepared_iris(test_size=test_size)
