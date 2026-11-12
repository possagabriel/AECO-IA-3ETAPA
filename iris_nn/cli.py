"""Interface de linha de comando do projeto."""

import argparse
import logging
import math
import sys
from collections.abc import Sequence
from pathlib import Path

from iris_nn._version import __version__
from iris_nn.experiment import ExperimentConfig, run_experiment
from iris_nn.persistence import save_model
from iris_nn.reporting import (
    format_console_report,
    write_history_csv,
    write_json_report,
)

logger = logging.getLogger(__name__)

_LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("deve ser um número inteiro.") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("deve ser maior que zero.")
    return parsed


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("deve ser um número real.") from exc
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("deve ser positivo e finito.")
    return parsed


def _fraction(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("deve ser um número real.") from exc
    if not math.isfinite(parsed) or not 0.0 < parsed < 1.0:
        raise argparse.ArgumentTypeError("deve estar no intervalo (0, 1).")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    """Cria o parser de argumentos da aplicação."""
    parser = argparse.ArgumentParser(
        prog="iris-nn",
        description=(
            "Treina e avalia uma rede neural 4-8-3 implementada do zero com "
            "NumPy no dataset Iris."
        ),
        epilog=(
            "Exemplos: iris-nn | "
            "iris-nn --epochs 2000 --learning-rate 0.05 --seed 7 | "
            "iris-nn --report reports/run.json --history reports/loss.csv"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument(
        "--epochs",
        type=_positive_int,
        default=1000,
        help="número de épocas de treinamento",
    )
    parser.add_argument(
        "--learning-rate",
        type=_positive_float,
        default=0.1,
        help="taxa de aprendizado do gradiente descendente",
    )
    parser.add_argument(
        "--hidden-size",
        type=_positive_int,
        default=8,
        help="quantidade de neurônios da camada escondida",
    )
    parser.add_argument(
        "--test-size",
        type=_fraction,
        default=0.2,
        help="fração dos dados reservada para teste",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="semente aleatória para reprodutibilidade",
    )
    parser.add_argument(
        "--log-every",
        type=_positive_int,
        default=100,
        help="intervalo de épocas entre os logs de perda",
    )
    parser.add_argument(
        "--log-level",
        choices=tuple(_LOG_LEVELS),
        default="INFO",
        help="nível de detalhamento dos logs",
    )
    parser.add_argument(
        "--report",
        type=Path,
        metavar="ARQUIVO",
        help="grava um relatório JSON no caminho informado",
    )
    parser.add_argument(
        "--history",
        type=Path,
        metavar="ARQUIVO",
        help="grava o histórico de perda por época em CSV",
    )
    parser.add_argument(
        "--save-model",
        type=Path,
        metavar="ARQUIVO",
        help="salva o modelo treinado em um arquivo .npz",
    )
    return parser


def _configure_logging(level_name: str) -> None:
    logging.basicConfig(
        level=_LOG_LEVELS[level_name],
        format="%(message)s",
        stream=sys.stdout,
        force=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Executa a CLI e retorna o código de saída do processo."""
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.log_level)

    try:
        config = ExperimentConfig(
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            hidden_size=args.hidden_size,
            test_size=args.test_size,
            seed=args.seed,
            log_every=args.log_every,
        )
    except ValueError as exc:
        logger.error("Configuração inválida: %s", exc)
        return 2

    result = run_experiment(config)
    print(format_console_report(result))

    try:
        if args.report is not None:
            report_path = write_json_report(result, args.report)
            print(f"Relatório JSON: {report_path}")
        if args.history is not None:
            history_path = write_history_csv(result, args.history)
            print(f"Histórico CSV:  {history_path}")
        if args.save_model is not None:
            model_path = save_model(args.save_model, result.model)
            print(f"Modelo salvo:   {model_path}")
    except OSError as exc:
        logger.error("Falha ao gravar artefatos: %s", exc)
        return 1

    return 0
