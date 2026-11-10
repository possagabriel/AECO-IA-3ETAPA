"""Testes da rede neural, incluindo verificação numérica dos gradientes."""

import numpy as np
import pytest

from iris_nn.data import load_prepared_iris
from iris_nn.network import NeuralNetwork


def test_relu_and_derivative() -> None:
    values = np.array([[-2.0, 0.0, 3.5]])

    np.testing.assert_array_equal(
        NeuralNetwork.relu(values),
        np.array([[0.0, 0.0, 3.5]]),
    )
    np.testing.assert_array_equal(
        NeuralNetwork.relu_derivative(values),
        np.array([[0.0, 0.0, 1.0]]),
    )


def test_softmax_is_stable_and_normalized() -> None:
    logits = np.array([[1000.0, 1001.0, 1002.0], [-1000.0, -1001.0, -1002.0]])

    probabilities = NeuralNetwork.softmax(logits)

    assert np.all(np.isfinite(probabilities))
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)


def test_forward_returns_valid_probabilities() -> None:
    rng = np.random.default_rng(0)
    features = rng.normal(size=(12, 4))
    model = NeuralNetwork(n_inputs=4, n_hidden=8, n_outputs=3, seed=0)

    cache = model.forward(features)

    assert cache.a1.shape == (12, 8)
    assert cache.probs.shape == (12, 3)
    np.testing.assert_allclose(cache.probs.sum(axis=1), 1.0)
    assert np.all(cache.probs >= 0.0)


def test_cross_entropy_is_zero_for_perfect_predictions() -> None:
    targets = np.eye(3)
    model = NeuralNetwork()

    assert model.compute_loss(targets, targets) == pytest.approx(0.0, abs=1e-12)


def test_gradients_match_finite_differences() -> None:
    rng = np.random.default_rng(7)
    features = rng.normal(size=(6, 3))
    labels = rng.integers(0, 2, size=6)
    targets = np.eye(2)[labels]

    model = NeuralNetwork(n_inputs=3, n_hidden=4, n_outputs=2, seed=11)
    gradients = model.backward(model.forward(features), targets)
    epsilon = 1e-6

    for name, analytic_gradient in gradients.items():
        parameter = getattr(model, name)
        numeric_gradient = np.zeros_like(parameter)

        for index in np.ndindex(parameter.shape):
            original = parameter[index]

            parameter[index] = original + epsilon
            loss_plus = model.compute_loss(model.forward(features).probs, targets)

            parameter[index] = original - epsilon
            loss_minus = model.compute_loss(model.forward(features).probs, targets)

            parameter[index] = original
            numeric_gradient[index] = (loss_plus - loss_minus) / (2.0 * epsilon)

        np.testing.assert_allclose(
            analytic_gradient,
            numeric_gradient,
            rtol=1e-5,
            atol=1e-7,
            err_msg=f"Gradiente incorreto para {name}",
        )


def test_training_reduces_loss_and_learns_iris() -> None:
    dataset = load_prepared_iris(test_size=0.2, random_state=42)
    model = NeuralNetwork(
        n_inputs=dataset.X_train.shape[1],
        n_hidden=8,
        n_outputs=dataset.y_train.shape[1],
        learning_rate=0.1,
        seed=42,
    )

    history = model.fit(
        dataset.X_train,
        dataset.y_train,
        epochs=1000,
        verbose=False,
    )

    assert history[-1] < history[0]
    assert model.accuracy(dataset.X_test, dataset.y_test) >= 0.9


def test_same_seed_produces_same_training_history() -> None:
    dataset = load_prepared_iris(test_size=0.2, random_state=42)
    first = NeuralNetwork(seed=123)
    second = NeuralNetwork(seed=123)

    first_history = first.fit(
        dataset.X_train, dataset.y_train, epochs=20, verbose=False
    )
    second_history = second.fit(
        dataset.X_train, dataset.y_train, epochs=20, verbose=False
    )

    np.testing.assert_allclose(first_history, second_history)


def test_forward_rejects_wrong_number_of_features() -> None:
    model = NeuralNetwork(n_inputs=4)

    with pytest.raises(ValueError):
        model.forward(np.zeros((5, 3)))


def test_backward_rejects_wrong_target_shape() -> None:
    model = NeuralNetwork(n_inputs=4, n_outputs=3)
    cache = model.forward(np.zeros((5, 4)))

    with pytest.raises(ValueError):
        model.backward(cache, np.zeros((5, 2)))
