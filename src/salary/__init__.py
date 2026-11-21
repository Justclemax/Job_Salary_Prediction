"""Job salary prediction: data, statistics, causal inference, ML models."""

__version__ = "0.1.0"


def main() -> None:
    from salary.train import main as train_main

    train_main()
