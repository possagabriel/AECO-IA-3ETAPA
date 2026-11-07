"""Rede neural 4-8-3 implementada do zero com NumPy."""

from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


class ForwardCache(NamedTuple):
    """Valores intermediários reutilizados no backward pass."""

    X: FloatArray
    z1: FloatArray
    a1: FloatArray
    probs: FloatArray


class NeuralNetwork:
    """Perceptron multicamadas com uma camada escondida ReLU e saída Softmax.

    A rede é treinada por gradiente descendente em lote completo, utilizando a
    função de custo de entropia cruzada.
    """

    def __init__(
        self,
        n_inputs: int = 4,
        n_hidden: int = 8,
        n_outputs: int = 3,
        learning_rate: float = 0.1,
        seed: int = 42,
    ) -> None:
        if min(n_inputs, n_hidden, n_outputs) < 1:
            raise ValueError("As dimensões da rede devem ser positivas.")
        if learning_rate <= 0.0:
            raise ValueError("A taxa de aprendizado deve ser positiva.")

        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs
        self.learning_rate = learning_rate

        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0.0, np.sqrt(2.0 / n_inputs), (n_inputs, n_hidden))
        self.b1 = rng.normal(0.0, 0.01, n_hidden)
        self.W2 = rng.normal(0.0, np.sqrt(1.0 / n_hidden), (n_hidden, n_outputs))
        self.b2 = rng.normal(0.0, 0.01, n_outputs)

    @staticmethod
    def relu(z: FloatArray) -> FloatArray:
        """Aplica a ReLU elemento a elemento."""
        return np.maximum(0.0, z)

    @staticmethod
    def relu_derivative(z: FloatArray) -> FloatArray:
        """Calcula a derivada da ReLU (0 para z <= 0 e 1 para z > 0)."""
        return (z > 0.0).astype(np.float64)

    @staticmethod
    def softmax(z: FloatArray) -> FloatArray:
        """Converte logits em probabilidades, com estabilidade numérica."""
        shifted = z - np.max(z, axis=1, keepdims=True)
        exponentials = np.exp(shifted)
        return exponentials / np.sum(exponentials, axis=1, keepdims=True)

    def _validate_features(self, X: np.ndarray) -> FloatArray:
        features = np.asarray(X, dtype=np.float64)
        if features.ndim != 2 or features.shape[1] != self.n_inputs:
            raise ValueError(f"X deve ter formato (n_amostras, {self.n_inputs}).")
        return features

    def _validate_targets(self, y_true: np.ndarray, n_samples: int) -> FloatArray:
        targets = np.asarray(y_true, dtype=np.float64)
        if targets.shape != (n_samples, self.n_outputs):
            raise ValueError(f"y_true deve ter formato (n_amostras, {self.n_outputs}).")
        return targets

    def forward(self, X: np.ndarray) -> ForwardCache:
        """Executa o forward pass e guarda as ativações para o backward."""
        features = self._validate_features(X)

        z1 = features @ self.W1 + self.b1
        a1 = self.relu(z1)
        z2 = a1 @ self.W2 + self.b2
        probs = self.softmax(z2)

        return ForwardCache(X=features, z1=z1, a1=a1, probs=probs)

    def compute_loss(self, probs: np.ndarray, y_true: np.ndarray) -> float:
        """Calcula a entropia cruzada média entre probabilidades e alvos."""
        probabilities = np.asarray(probs, dtype=np.float64)
        if probabilities.ndim != 2:
            raise ValueError("probs deve ser uma matriz bidimensional.")

        targets = self._validate_targets(y_true, probabilities.shape[0])
        probabilities = np.clip(probabilities, 1e-12, 1.0)

        return float(-np.mean(np.sum(targets * np.log(probabilities), axis=1)))

    def backward(
        self, cache: ForwardCache, y_true: np.ndarray
    ) -> dict[str, FloatArray]:
        """Calcula os gradientes da perda em relação a pesos e vieses."""
        targets = self._validate_targets(y_true, cache.X.shape[0])
        n_samples = cache.X.shape[0]

        dz2 = (cache.probs - targets) / n_samples
        dW2 = cache.a1.T @ dz2
        db2 = np.sum(dz2, axis=0)

        da1 = dz2 @ self.W2.T
        dz1 = da1 * self.relu_derivative(cache.z1)
        dW1 = cache.X.T @ dz1
        db1 = np.sum(dz1, axis=0)

        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

    def _apply_gradients(self, gradients: dict[str, FloatArray]) -> None:
        self.W1 -= self.learning_rate * gradients["W1"]
        self.b1 -= self.learning_rate * gradients["b1"]
        self.W2 -= self.learning_rate * gradients["W2"]
        self.b2 -= self.learning_rate * gradients["b2"]

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 1000,
        log_every: int = 100,
        verbose: bool = True,
    ) -> list[float]:
        """Treina a rede por gradiente descendente e devolve o histórico da perda."""
        if epochs < 1:
            raise ValueError("epochs deve ser maior que zero.")
        if log_every < 1:
            raise ValueError("log_every deve ser maior que zero.")

        features = self._validate_features(X)
        targets = self._validate_targets(y, features.shape[0])
        history: list[float] = []

        for epoch in range(epochs):
            cache = self.forward(features)
            loss = self.compute_loss(cache.probs, targets)
            gradients = self.backward(cache, targets)
            self._apply_gradients(gradients)
            history.append(loss)

            if verbose and (
                epoch == 0 or (epoch + 1) % log_every == 0 or epoch == epochs - 1
            ):
                print(f"Época {epoch + 1:4d}/{epochs} | loss: {loss:.6f}")

        return history

    def predict_proba(self, X: np.ndarray) -> FloatArray:
        """Retorna as probabilidades previstas para cada classe."""
        return self.forward(X).probs

    def predict(self, X: np.ndarray) -> NDArray[np.int64]:
        """Retorna a classe de maior probabilidade para cada amostra."""
        return np.argmax(self.predict_proba(X), axis=1)

    def accuracy(self, X: np.ndarray, y_true: np.ndarray) -> float:
        """Calcula a acurácia aceitando alvos one-hot ou rótulos inteiros."""
        targets = np.asarray(y_true)
        if targets.ndim == 2:
            labels = np.argmax(targets, axis=1)
        elif targets.ndim == 1:
            labels = targets.astype(np.int64)
        else:
            raise ValueError("y_true deve ser unidimensional ou one-hot.")

        predictions = self.predict(X)
        if labels.shape[0] != predictions.shape[0]:
            raise ValueError("X e y_true devem ter a mesma quantidade de amostras.")

        return float(np.mean(predictions == labels))
