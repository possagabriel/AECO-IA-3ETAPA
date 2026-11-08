"""Treina e avalia a rede neural no dataset Iris."""

import argparse

from iris_nn.data import load_prepared_iris
from iris_nn.network import NeuralNetwork


def build_parser() -> argparse.ArgumentParser:
    """Cria o parser de argumentos da linha de comando."""
    parser = argparse.ArgumentParser(
        description="Rede neural 4-8-3 em NumPy para classificação do Iris.",
    )
    parser.add_argument("--epochs", type=int, default=1000, help="número de épocas")
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.1,
        help="taxa de aprendizado do gradiente descendente",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=8,
        help="quantidade de neurônios da camada escondida",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="fração dos dados reservada para teste",
    )
    parser.add_argument("--seed", type=int, default=42, help="semente aleatória")
    parser.add_argument(
        "--log-every",
        type=int,
        default=100,
        help="intervalo de épocas entre os logs de perda",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Executa o treinamento e imprime as métricas finais."""
    args = build_parser().parse_args(argv)

    dataset = load_prepared_iris(
        test_size=args.test_size,
        random_state=args.seed,
    )
    model = NeuralNetwork(
        n_inputs=dataset.X_train.shape[1],
        n_hidden=args.hidden_size,
        n_outputs=dataset.y_train.shape[1],
        learning_rate=args.learning_rate,
        seed=args.seed,
    )

    print("=" * 60)
    print("Classificação de flores Iris com rede neural em NumPy")
    print("=" * 60)
    print(f"Treino: {dataset.X_train.shape[0]} amostras")
    print(f"Teste:  {dataset.X_test.shape[0]} amostras")
    print(
        "Arquitetura: "
        f"{model.n_inputs} -> {model.n_hidden} (ReLU) -> "
        f"{model.n_outputs} (Softmax)"
    )
    print(f"Épocas: {args.epochs} | Taxa de aprendizado: {args.learning_rate}")
    print(f"Seed: {args.seed}")
    print("-" * 60)

    history = model.fit(
        dataset.X_train,
        dataset.y_train,
        epochs=args.epochs,
        log_every=args.log_every,
    )

    train_accuracy = model.accuracy(dataset.X_train, dataset.y_train)
    test_accuracy = model.accuracy(dataset.X_test, dataset.y_test)

    print("-" * 60)
    print(f"Perda final (treino): {history[-1]:.6f}")
    print(f"Acurácia de treino:  {train_accuracy:.2%}")
    print(f"Acurácia de teste:   {test_accuracy:.2%}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
