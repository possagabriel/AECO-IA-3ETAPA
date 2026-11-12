"""Orquestração de experimentos de treinamento e avaliação."""

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from iris_nn.data import IrisDataset, load_prepared_iris
from iris_nn.metrics import ClassificationMetrics, classification_metrics
from iris_nn.network import NeuralNetwork


@dataclass(frozen=True)
class ExperimentConfig:
    """Hiperparâmetros e opções de um experimento."""

    epochs: int = 1000
    learning_rate: float = 0.1
    hidden_size: int = 8
    test_size: float = 0.2
    seed: int = 42
    log_every: int = 100
    verbose: bool = True

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError("O número de épocas deve ser positivo.")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("A taxa de aprendizado deve ser positiva e finita.")
        if self.hidden_size < 1:
            raise ValueError("O tamanho da camada escondida deve ser positivo.")
        if not 0.0 < self.test_size < 1.0:
            raise ValueError("A fração de teste deve estar no intervalo (0, 1).")
        if self.log_every < 1:
            raise ValueError("O intervalo de log deve ser positivo.")


@dataclass(frozen=True, eq=False)
class ExperimentResult:
    """Resultado completo de um experimento, incluindo métricas e artefatos."""

    config: ExperimentConfig
    model: NeuralNetwork
    dataset: IrisDataset
    history: tuple[float, ...]
    train_metrics: ClassificationMetrics
    test_metrics: ClassificationMetrics
    elapsed_seconds: float

    @property
    def initial_loss(self) -> float:
        """Perda registrada na primeira época."""
        return self.history[0]

    @property
    def final_loss(self) -> float:
        """Perda registrada na última época."""
        return self.history[-1]

    @property
    def n_parameters(self) -> int:
        """Quantidade de parâmetros treináveis do modelo."""
        return self.model.n_parameters


def run_experiment(config: ExperimentConfig | None = None) -> ExperimentResult:
    """Executa o pipeline completo de preparação, treino e avaliação."""
    resolved_config = config or ExperimentConfig()
    dataset = load_prepared_iris(
        test_size=resolved_config.test_size,
        random_state=resolved_config.seed,
    )
    model = NeuralNetwork(
        n_inputs=dataset.n_features,
        n_hidden=resolved_config.hidden_size,
        n_outputs=dataset.n_classes,
        learning_rate=resolved_config.learning_rate,
        seed=resolved_config.seed,
    )

    started_at = perf_counter()
    history = model.fit(
        dataset.X_train,
        dataset.y_train,
        epochs=resolved_config.epochs,
        log_every=resolved_config.log_every,
        verbose=resolved_config.verbose,
    )
    elapsed_seconds = perf_counter() - started_at

    train_metrics = classification_metrics(
        dataset.y_train,
        model.predict(dataset.X_train),
        dataset.n_classes,
    )
    test_metrics = classification_metrics(
        dataset.y_test,
        model.predict(dataset.X_test),
        dataset.n_classes,
    )

    return ExperimentResult(
        config=resolved_config,
        model=model,
        dataset=dataset,
        history=tuple(history),
        train_metrics=train_metrics,
        test_metrics=test_metrics,
        elapsed_seconds=elapsed_seconds,
    )
