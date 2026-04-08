"""
device.py
---------
Detección automática del dispositivo de cómputo disponible.
Soporta: CUDA (GTX 1650 / Quadro T1000 / Colab T4/A100),
MPS (MacBook Air M4) y CPU.
"""

from __future__ import annotations

import torch
from rich.console import Console

console = Console()


def get_device(verbose: bool = True) -> torch.device:
    """
    Retorna el mejor dispositivo disponible.
    Orden de prioridad: CUDA > MPS > CPU.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        if verbose:
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            console.print(
                f"[bold green]✓ CUDA disponible:[/bold green] {name} ({vram:.1f} GB VRAM)"
            )
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        if verbose:
            console.print("[bold green]✓ Apple MPS disponible:[/bold green] MacBook M-series")
    else:
        device = torch.device("cpu")
        if verbose:
            console.print(
                "[bold yellow]⚠ Sin acelerador GPU — usando CPU.[/bold yellow] "
                "Se recomienda Google Colab T4 para entrenamiento completo."
            )
    return device


def get_batch_size(model_name: str, device: torch.device, override: int | None = None) -> int:
    """
    Batch size recomendado según modelo y VRAM disponible.

    Parameters
    ----------
    model_name : str
        Nombre del modelo timm (ej. 'efficientnet_b2').
    device : torch.device
    override : int | None
        Si se especifica, anula la detección automática.
    """
    if override is not None:
        return override

    table: dict[str, dict[str, int]] = {
        "efficientnet_b2": {"low": 16, "mid": 32, "high": 128},
        "convnext_tiny": {"low": 8, "mid": 16, "high": 64},
        "efficientnet_b4": {"low": 8, "mid": 16, "high": 64},
    }
    sizes = table.get(model_name, {"low": 8, "mid": 16, "high": 32})

    if device.type == "mps":
        return sizes["mid"]

    if device.type == "cuda":
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        if vram >= 12:
            return sizes["high"]
        elif vram >= 5:
            return sizes["mid"]
        else:
            return sizes["low"]

    return 4
