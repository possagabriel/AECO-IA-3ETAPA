"""Testes da rede neural, incluindo verificação numérica dos gradientes."""

import logging

import numpy as np
import pytest

from iris_nn.data import IrisDataset
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


@pytest.mark.parametrize(
    "logits",
    [
        np.array([1.0, 2.0]),
        np.empty((0, 3)),
        np.array([[1.0, np.inf, 2.0]]),
    ],
)
def test_softmax_rejects_invalid_logits(logits: np.ndarray) -> None:
    with pytest.raises(ValueError):
        NeuralNetwork.softmax(logits)


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


@pytest.mark.parametrize(
    "overrides",
    [
        {"n_inputs": 0},
        {"n_hidden": 0},
        {"n_outputs": 0},
        {"learning_rate": 0.0},
        {"learning_rate": float("inf")},
    ],
)
def test_init_rejects_invalid_hyperparameters(
    overrides: dict[str, float],
) -> None:
    with pytest.raises(ValueError):
        NeuralNetwork(**overrides)  # type: ignore[arg-type]


def test_n_parameters_and_repr() -> None:
    model = NeuralNetwork(n_inputs=4, n_hidden=8, n_outputs=3, seed=1)

    assert model.n_parameters == 67
    assert "n_hidden=8" in repr(model)
    assert "seed=1" in repr(model)


def test_state_dict_round_trip(dataset: IrisDataset) -> None:
    model = NeuralNetwork(seed=1)
    expected_probabilities = model.predict_proba(dataset.X_test)

    restored = NeuralNetwork(seed=999)
    restored.load_state_dict(model.state_dict())

    np.testing.assert_allclose(
        restored.predict_proba(dataset.X_test),
        expected_probabilities,
    )


def test_state_dict_rejects_missing_parameter() -> None:
    model = NeuralNetwork()
    state = model.state_dict()
    state.pop("W2")

    with pytest.raises(KeyError):
        model.load_state_dict(state)


def test_state_dict_rejects_wrong_shape() -> None:
    model = NeuralNetwork()
    state = model.state_dict()
    state["W1"] = np.zeros((2, 2))

    with pytest.raises(ValueError, match="formato"):
        model.load_state_dict(state)


def test_state_dict_rejects_non_finite_parameter() -> None:
    model = NeuralNetwork()
    state = model.state_dict()
    state["b1"][0] = np.nan

    with pytest.raises(ValueError, match="finitos"):
        model.load_state_dict(state)


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


def test_training_reduces_loss_and_learns_iris(dataset: IrisDataset) -> None:
    model = NeuralNetwork(
        n_inputs=dataset.n_features,
        n_hidden=8,
        n_outputs=dataset.n_classes,
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


def test_accuracy_accepts_integer_labels(dataset: IrisDataset) -> None:
    model = NeuralNetwork(seed=0)
    labels = np.argmax(dataset.y_test, axis=1)

    assert model.accuracy(dataset.X_test, labels) == model.accuracy(
        dataset.X_test,
        dataset.y_test,
    )


def test_same_seed_produces_same_training_history(dataset: IrisDataset) -> None:
    first = NeuralNetwork(seed=123)
    second = NeuralNetwork(seed=123)

    first_history = first.fit(
        dataset.X_train,
        dataset.y_train,
        epochs=20,
        verbose=False,
    )
    second_history = second.fit(
        dataset.X_train,
        dataset.y_train,
        epochs=20,
        verbose=False,
    )

    np.testing.assert_allclose(first_history, second_history)


def test_forward_rejects_wrong_number_of_features() -> None:
    model = NeuralNetwork(n_inputs=4)

    with pytest.raises(ValueError, match="formato"):
        model.forward(np.zeros((5, 3)))


def test_forward_rejects_non_finite_features() -> None:
    model = NeuralNetwork(n_inputs=4)
    features = np.zeros((5, 4))
    features[0, 0] = np.nan

    with pytest.raises(ValueError, match="finitos"):
        model.forward(features)


def test_forward_rejects_empty_batch() -> None:
    model = NeuralNetwork(n_inputs=4)

    with pytest.raises(ValueError, match="ao menos"):
        model.forward(np.empty((0, 4)))


def test_compute_loss_rejects_non_finite_targets() -> None:
    model = NeuralNetwork()
    targets = np.eye(3)
    targets[0, 0] = np.nan

    with pytest.raises(ValueError, match="finitos"):
        model.compute_loss(np.eye(3), targets)


def test_compute_loss_rejects_empty_probabilities() -> None:
    model = NeuralNetwork()

    with pytest.raises(ValueError, match="não vazia"):
        model.compute_loss(np.empty((0, 3)), np.empty((0, 3)))


def test_compute_loss_rejects_non_finite_probabilities() -> None:
    model = NeuralNetwork()
    probabilities = np.eye(3)
    probabilities[0, 0] = np.inf

    with pytest.raises(ValueError, match="finitos"):
        model.compute_loss(probabilities, np.eye(3))


def test_backward_rejects_wrong_target_shape() -> None:
    model = NeuralNetwork(n_inputs=4, n_outputs=3)
    cache = model.forward(np.zeros((5, 4)))

    with pytest.raises(ValueError, match="formato"):
        model.backward(cache, np.zeros((5, 2)))


@pytest.mark.parametrize("epochs", [0, -1])
def test_fit_rejects_invalid_epochs(epochs: int) -> None:
    model = NeuralNetwork()

    with pytest.raises(ValueError, match="epochs"):
        model.fit(np.zeros((2, 4)), np.eye(3)[[0, 1]], epochs=epochs)


def test_fit_rejects_invalid_log_interval() -> None:
    model = NeuralNetwork()

    with pytest.raises(ValueError, match="log_every"):
        model.fit(np.zeros((2, 4)), np.eye(3)[[0, 1]], epochs=1, log_every=0)


def test_fit_logs_training_progress(caplog: pytest.LogCaptureFixture) -> None:
    model = NeuralNetwork()
    features = np.zeros((2, 4))
    targets = np.eye(3)[[0, 1]]

    with caplog.at_level(logging.INFO, logger="iris_nn.network"):
        model.fit(features, targets, epochs=2, log_every=1, verbose=True)

    assert "Época    1/2" in caplog.text
    assert "Época    2/2" in caplog.text


def test_accuracy_rejects_length_mismatch(dataset: IrisDataset) -> None:
    model = NeuralNetwork()

    with pytest.raises(ValueError, match="mesma quantidade"):
        model.accuracy(dataset.X_test, dataset.y_test[:-1])
