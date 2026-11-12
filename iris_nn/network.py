"""Rede neural 4-8-3 implementada do zero com NumPy."""

import logging
from collections.abc import Mapping
from typing import NamedTuple

import numpy as np

from iris_nn._typing import AnyArray, FloatArray, IntArray
from iris_nn.metrics import labels_from_targets

logger = logging.getLogger(__name__)


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
        if not np.isfinite(learning_rate) or learning_rate <= 0.0:
            raise ValueError("A taxa de aprendizado deve ser positiva e finita.")

        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs
        self.learning_rate = learning_rate
        self.seed = seed

        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0.0, np.sqrt(2.0 / n_inputs), (n_inputs, n_hidden))
        self.b1 = rng.normal(0.0, 0.01, n_hidden)
        self.W2 = rng.normal(0.0, np.sqrt(1.0 / n_hidden), (n_hidden, n_outputs))
        self.b2 = rng.normal(0.0, 0.01, n_outputs)

    def __repr__(self) -> str:
        return (
            "NeuralNetwork("
            f"n_inputs={self.n_inputs}, "
            f"n_hidden={self.n_hidden}, "
            f"n_outputs={self.n_outputs}, "
            f"learning_rate={self.learning_rate}, "
            f"seed={self.seed}"
            ")"
        )

    @property
    def n_parameters(self) -> int:
        """Quantidade total de parâmetros treináveis."""
        return sum(parameter.size for parameter in (self.W1, self.b1, self.W2, self.b2))

    @staticmethod
    def relu(z: AnyArray) -> FloatArray:
        """Aplica a ReLU elemento a elemento."""
        return np.maximum(0.0, np.asarray(z, dtype=np.float64))

    @staticmethod
    def relu_derivative(z: AnyArray) -> FloatArray:
        """Calcula a derivada da ReLU (0 para z <= 0 e 1 para z > 0)."""
        return (np.asarray(z) > 0.0).astype(np.float64)

    @staticmethod
    def softmax(z: AnyArray) -> FloatArray:
        """Converte logits em probabilidades, com estabilidade numérica."""
        logits = np.asarray(z, dtype=np.float64)
        if logits.ndim != 2 or logits.shape[0] == 0:
            raise ValueError("Os logits devem ser uma matriz não vazia.")
        if not np.all(np.isfinite(logits)):
            raise ValueError("Os logits devem conter apenas valores finitos.")

        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exponentials = np.exp(shifted)
        return exponentials / np.sum(exponentials, axis=1, keepdims=True)

    def _validate_features(self, X: AnyArray) -> FloatArray:
        features = np.asarray(X, dtype=np.float64)
        if features.ndim != 2 or features.shape[1] != self.n_inputs:
            raise ValueError(f"X deve ter formato (n_amostras, {self.n_inputs}).")
        if features.shape[0] == 0:
            raise ValueError("X deve conter ao menos uma amostra.")
        if not np.all(np.isfinite(features)):
            raise ValueError("X deve conter apenas valores finitos.")
        return features

    def _validate_targets(self, y_true: AnyArray, n_samples: int) -> FloatArray:
        targets = np.asarray(y_true, dtype=np.float64)
        if targets.shape != (n_samples, self.n_outputs):
            raise ValueError(f"y_true deve ter formato (n_amostras, {self.n_outputs}).")
        if not np.all(np.isfinite(targets)):
            raise ValueError("y_true deve conter apenas valores finitos.")
        return targets

    def forward(self, X: AnyArray) -> ForwardCache:
        """Executa o forward pass e guarda as ativações para o backward."""
        features = self._validate_features(X)

        z1 = features @ self.W1 + self.b1
        a1 = self.relu(z1)
        z2 = a1 @ self.W2 + self.b2
        probs = self.softmax(z2)

        return ForwardCache(X=features, z1=z1, a1=a1, probs=probs)

    def compute_loss(self, probs: AnyArray, y_true: AnyArray) -> float:
        """Calcula a entropia cruzada média entre probabilidades e alvos."""
        probabilities = np.asarray(probs, dtype=np.float64)
        if probabilities.ndim != 2 or probabilities.shape[0] == 0:
            raise ValueError("probs deve ser uma matriz não vazia.")

        targets = self._validate_targets(y_true, probabilities.shape[0])
        if not np.all(np.isfinite(probabilities)):
            raise ValueError("probs deve conter apenas valores finitos.")
        probabilities = np.clip(probabilities, 1e-12, 1.0)

        return float(-np.mean(np.sum(targets * np.log(probabilities), axis=1)))

    def backward(self, cache: ForwardCache, y_true: AnyArray) -> dict[str, FloatArray]:
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

    def _apply_gradients(self, gradients: Mapping[str, FloatArray]) -> None:
        self.W1 -= self.learning_rate * gradients["W1"]
        self.b1 -= self.learning_rate * gradients["b1"]
        self.W2 -= self.learning_rate * gradients["W2"]
        self.b2 -= self.learning_rate * gradients["b2"]

    def fit(
        self,
        X: AnyArray,
        y: AnyArray,
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
                logger.info(
                    "Época %4d/%d | loss: %.6f",
                    epoch + 1,
                    epochs,
                    loss,
                )

        return history

    def state_dict(self) -> dict[str, FloatArray]:
        """Retorna uma cópia dos parâmetros da rede."""
        return {
            "W1": self.W1.copy(),
            "b1": self.b1.copy(),
            "W2": self.W2.copy(),
            "b2": self.b2.copy(),
        }

    def load_state_dict(self, state: Mapping[str, AnyArray]) -> None:
        """Substitui os parâmetros da rede após validar nomes e formatos."""
        expected_shapes = {
            "W1": (self.n_inputs, self.n_hidden),
            "b1": (self.n_hidden,),
            "W2": (self.n_hidden, self.n_outputs),
            "b2": (self.n_outputs,),
        }
        validated: dict[str, FloatArray] = {}

        for name, shape in expected_shapes.items():
            if name not in state:
                raise KeyError(f"Parâmetro ausente: {name}")
            parameter = np.asarray(state[name], dtype=np.float64)
            if parameter.shape != shape:
                raise ValueError(
                    f"{name} deve ter formato {shape}, mas possui {parameter.shape}."
                )
            if not np.all(np.isfinite(parameter)):
                raise ValueError(f"{name} deve conter apenas valores finitos.")
            validated[name] = parameter.copy()

        self.W1 = validated["W1"]
        self.b1 = validated["b1"]
        self.W2 = validated["W2"]
        self.b2 = validated["b2"]

    def predict_proba(self, X: AnyArray) -> FloatArray:
        """Retorna as probabilidades previstas para cada classe."""
        return self.forward(X).probs

    def predict(self, X: AnyArray) -> IntArray:
        """Retorna a classe de maior probabilidade para cada amostra."""
        return np.argmax(self.predict_proba(X), axis=1).astype(np.int64)

    def accuracy(self, X: AnyArray, y_true: AnyArray) -> float:
        """Calcula a acurácia aceitando alvos one-hot ou rótulos inteiros."""
        labels = labels_from_targets(y_true, n_classes=self.n_outputs)
        predictions = self.predict(X)
        if labels.shape[0] != predictions.shape[0]:
            raise ValueError("X e y_true devem ter a mesma quantidade de amostras.")

        return float(np.mean(predictions == labels))
