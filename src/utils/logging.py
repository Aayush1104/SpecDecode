"""Structured logging setup with optional WandB integration."""

import logging
import sys
from typing import Optional


_logger: Optional[logging.Logger] = None
_wandb_run = None


def setup_logging(
    level: str = "INFO",
    use_wandb: bool = False,
    wandb_project: str = "specdecode",
    wandb_run_name: Optional[str] = None,
) -> logging.Logger:
    """Initialize structured logging and optionally WandB."""
    global _logger, _wandb_run

    _logger = logging.getLogger("specdecode")
    _logger.setLevel(getattr(logging, level.upper()))
    _logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, level.upper()))
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    _logger.addHandler(handler)

    if use_wandb:
        try:
            import wandb

            _wandb_run = wandb.init(
                project=wandb_project,
                name=wandb_run_name,
                reinit=True,
            )
            _logger.info("WandB initialized: %s", wandb_run_name or wandb_project)
        except ImportError:
            _logger.warning("wandb not installed, skipping WandB integration")
        except Exception as e:
            _logger.warning("Failed to initialize WandB: %s", e)

    return _logger


def get_logger() -> logging.Logger:
    """Get the specdecode logger, initializing with defaults if needed."""
    global _logger
    if _logger is None:
        _logger = setup_logging()
    return _logger


def log_metrics(metrics: dict, step: Optional[int] = None):
    """Log metrics to console and optionally WandB."""
    logger = get_logger()
    logger.info("Metrics: %s", metrics)
    if _wandb_run is not None:
        import wandb

        wandb.log(metrics, step=step)
