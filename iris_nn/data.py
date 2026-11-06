"""Carregamento e pré-processamento do dataset Iris."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class IrisDataset:
    """Conjuntos de treino e teste já normalizados e em one-hot encoding."""

    X_train: FloatArray
    X_test: FloatArray
    y_train: FloatArray
    y_test: FloatArray
    feature_names: tuple[str, ...]
    class_names: tuple[str, ...]


def to_one_hot(labels: np.ndarray, n_classes: int) -> FloatArray:
    """Converte rótulos inteiros em uma matriz binária one-hot.

    Args:
        labels: Vetor unidimensional com rótulos entre ``0`` e ``n_classes - 1``.
        n_classes: Quantidade de classes distintas.

    Returns:
        Matriz de formato ``(n_amostras, n_classes)``.

    Raises:
        ValueError: Se os rótulos ou a quantidade de classes forem inválidos.
    """
    labels = np.asarray(labels)

    if labels.ndim != 1:
        raise ValueError("Os rótulos devem ser um vetor unidimensional.")
    if n_classes < 1:
        raise ValueError("A quantidade de classes deve ser positiva.")
    if labels.size > 0 and (labels.min() < 0 or labels.max() >= n_classes):
        raise ValueError("Os rótulos devem estar no intervalo [0, n_classes).")

    return np.eye(n_classes, dtype=np.float64)[labels.astype(np.int64)]


def load_prepared_iris(
    test_size: float = 0.2,
    random_state: int = 42,
) -> IrisDataset:
    """Carrega o Iris e devolve os dados prontos para treinamento.

    A padronização é ajustada apenas no conjunto de treino para evitar
    vazamento de informação do conjunto de teste.

    Args:
        test_size: Fração dos dados reservada para teste.
        random_state: Semente que garante a reprodutibilidade da divisão.

    Returns:
        Um :class:`IrisDataset` com atributos normalizados e alvos em one-hot.

    Raises:
        ValueError: Se ``test_size`` estiver fora do intervalo ``(0, 1)``.
    """
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size deve estar no intervalo (0, 1).")

    iris = load_iris()
    features = np.asarray(iris.data, dtype=np.float64)
    labels = np.asarray(iris.target)

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    n_classes = len(iris.target_names)

    return IrisDataset(
        X_train=X_train,
        X_test=X_test,
        y_train=to_one_hot(y_train, n_classes),
        y_test=to_one_hot(y_test, n_classes),
        feature_names=tuple(iris.feature_names),
        class_names=tuple(iris.target_names),
    )
