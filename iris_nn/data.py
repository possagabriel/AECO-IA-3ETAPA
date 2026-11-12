"""Carregamento e pré-processamento do dataset Iris."""

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from iris_nn._typing import AnyArray, FloatArray


@dataclass(frozen=True, eq=False)
class IrisDataset:
    """Conjuntos de treino e teste normalizados e com alvos em one-hot."""

    X_train: FloatArray
    X_test: FloatArray
    y_train: FloatArray
    y_test: FloatArray
    feature_names: tuple[str, ...]
    class_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.X_train.ndim != 2 or self.X_test.ndim != 2:
            raise ValueError("X_train e X_test devem ser matrizes bidimensionais.")
        if self.X_train.shape[1] != self.X_test.shape[1]:
            raise ValueError("Treino e teste devem ter a mesma quantidade de features.")

        n_classes = len(self.class_names)
        if self.y_train.shape != (self.X_train.shape[0], n_classes):
            raise ValueError("y_train não corresponde às amostras ou classes.")
        if self.y_test.shape != (self.X_test.shape[0], n_classes):
            raise ValueError("y_test não corresponde às amostras ou classes.")
        if len(self.feature_names) != self.X_train.shape[1]:
            raise ValueError("feature_names não corresponde às colunas de entrada.")

    @property
    def n_features(self) -> int:
        """Quantidade de características de entrada."""
        return int(self.X_train.shape[1])

    @property
    def n_classes(self) -> int:
        """Quantidade de classes de saída."""
        return len(self.class_names)

    @property
    def n_train(self) -> int:
        """Quantidade de amostras de treino."""
        return int(self.X_train.shape[0])

    @property
    def n_test(self) -> int:
        """Quantidade de amostras de teste."""
        return int(self.X_test.shape[0])


def to_one_hot(labels: AnyArray, n_classes: int) -> FloatArray:
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
    if labels.dtype.kind == "f":
        if not np.all(np.isfinite(labels)) or not np.all(labels == np.floor(labels)):
            raise ValueError("Os rótulos devem ser inteiros.")
    elif labels.dtype.kind not in "iub":
        raise ValueError("Os rótulos devem ser inteiros.")
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
    standardized_train = scaler.fit_transform(X_train)
    standardized_test = scaler.transform(X_test)

    n_classes = len(iris.target_names)

    return IrisDataset(
        X_train=np.asarray(standardized_train, dtype=np.float64),
        X_test=np.asarray(standardized_test, dtype=np.float64),
        y_train=to_one_hot(y_train, n_classes),
        y_test=to_one_hot(y_test, n_classes),
        feature_names=tuple(iris.feature_names),
        class_names=tuple(iris.target_names),
    )
