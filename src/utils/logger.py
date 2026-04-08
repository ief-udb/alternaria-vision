"""
logger.py
---------
Configuración centralizada de logging con Rich para salida formateada.
"""

from __future__ import annotations

import logging
from pathlib import Path

from rich.logging import RichHandler


def get_logger(name: str, log_file: Path | None = None) -> logging.Logger:
    """
    Retorna un logger con handler Rich (consola) y opcionalmente a archivo.

    Parameters
    ----------
    name : str
        Nombre del logger (generalmente __name__).
    log_file : Path | None
        Si se indica, guarda los logs en este archivo.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        console_handler = RichHandler(
            rich_tracebacks=True,
            markup=True,
            show_path=False,
        )
        console_handler.setLevel(logging.INFO)
        logger.addHandler(console_handler)

        if log_file is not None:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(
                logging.Formatter(
                    "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )
            logger.addHandler(fh)

    return logger
