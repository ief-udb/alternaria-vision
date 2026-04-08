"""
evaluate.py
-----------
CLI para evaluar un checkpoint entrenado sobre el test set.
Genera el conjunto completo de métricas, visualizaciones y
un CSV de registro para comparación entre experimentos.

Uso:
    uv run evaluate \\
        --checkpoint checkpoints/classification/best_model.pt \\
        --data-dir data/processed/classification/ \\
        --output-dir outputs/evaluation/

    # Con Test-Time Augmentation (TTA):
    uv run evaluate \\
        --checkpoint checkpoints/classification/best_model.pt \\
        --data-dir data/processed/classification/ \\
        --tta
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import typer
from torch.utils.data import DataLoader

from src.data.augmentations import get_tta_transforms, get_val_transforms
from src.data.dataset import MicroscopyDataset
from src.evaluation.metrics import (
    ClassificationMetrics,
    compute_metrics,
    plot_confusion_matrix,
    plot_precision_recall_curve,
    plot_roc_curve,
    save_metrics_csv,
)
from src.models.classifier import AlternariaCLF
from src.utils.device import get_device
from src.utils.logger import get_logger

app = typer.Typer(help="Evalúa un checkpoint sobre el test set.")
logger = get_logger(__name__, log_file=Path("logs/evaluate.log"))


def _run_inference(
    model: AlternariaCLF,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Ejecuta inferencia sobre un DataLoader completo.

    Returns
    -------
    y_true : np.ndarray  — Etiquetas reales.
    y_pred : np.ndarray  — Predicciones (umbral 0.5).
    y_prob : np.ndarray  — Probabilidad de clase positiva.
    """
    model.eval()
    all_labels, all_probs = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)[:, 1]
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    y_true = np.array(all_labels)
    y_prob = np.array(all_probs)
    y_pred = (y_prob >= 0.5).astype(int)

    return y_true, y_pred, y_prob


def _run_tta_inference(
    model: AlternariaCLF,
    dataset: MicroscopyDataset,
    device: torch.device,
    image_size: int,
    batch_size: int,
    num_workers: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Inferencia con Test-Time Augmentation (TTA, 8 transformaciones).
    Promedia las probabilidades de todas las variantes aumentadas.
    """
    transforms_list = get_tta_transforms(image_size)
    all_probs_runs: list[np.ndarray] = []
    y_true_arr: np.ndarray | None = None

    logger.info(f"TTA activado: {len(transforms_list)} transformaciones.")

    for i, transform in enumerate(transforms_list, 1):
        dataset.transform = transform
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        y_true, _, y_prob = _run_inference(model, loader, device)
        all_probs_runs.append(y_prob)
        if y_true_arr is None:
            y_true_arr = y_true
        logger.info(f"  TTA {i}/{len(transforms_list)} completado.")

    # Promedio de las 8 probabilidades
    y_prob_mean = np.mean(np.stack(all_probs_runs, axis=0), axis=0)
    y_pred_mean = (y_prob_mean >= 0.5).astype(int)

    return y_true_arr, y_pred_mean, y_prob_mean


@app.command()
def main(
    checkpoint: Path = typer.Option(
        ...,
        "--checkpoint",
        "-c",
        help="Ruta al .pt del mejor modelo (best_model.pt).",
    ),
    data_dir: Path = typer.Option(
        ...,
        "--data-dir",
        "-d",
        help="Directorio raíz con subdirectorios por clase.",
    ),
    split_file: Path = typer.Option(
        Path("data/splits/test.txt"),
        "--split-file",
        help="Archivo test.txt con rutas relativas al test set.",
    ),
    use_tta: bool = typer.Option(
        False,
        "--tta/--no-tta",
        help="Aplicar Test-Time Augmentation (8 variantes).",
    ),
    output_dir: Path = typer.Option(
        Path("outputs/evaluation"),
        "--output-dir",
        "-o",
        help="Directorio de salida para métricas y visualizaciones.",
    ),
    experiment_name: str = typer.Option(
        "run_001",
        "--name",
        "-n",
        help="Nombre del experimento para el CSV de registro.",
    ),
    image_size: int = typer.Option(
        288,
        help="Tamaño de imagen (debe coincidir con el entrenamiento).",
    ),
    batch_size: int = typer.Option(
        32,
        help="Batch size para inferencia.",
    ),
    num_workers: int = typer.Option(
        2,
        help="Workers para el DataLoader.",
    ),
) -> None:
    """
    Evalúa el modelo de clasificación en el test set.

    Genera:
      - Matriz de confusión (absoluta y normalizada)
      - Curva ROC con umbral óptimo de Youden
      - Curva Precision-Recall
      - Historial de métricas CSV
      - Log completo en logs/evaluate.log
    """
    device = get_device()

    logger.info(f"{'=' * 58}")
    logger.info(f"  Evaluación: {experiment_name}")
    logger.info(f"  Checkpoint: {checkpoint}")
    logger.info(f"  Split     : {split_file}")
    logger.info(f"  TTA       : {use_tta}")
    logger.info(f"{'=' * 58}")

    # Cargar modelo
    model = AlternariaCLF.load(checkpoint, device)
    model.eval()

    # Dataset
    dataset = MicroscopyDataset(
        root=data_dir,
        transform=get_val_transforms(image_size),
        split_file=split_file if split_file.exists() else None,
    )

    if use_tta:
        y_true, y_pred, y_prob = _run_tta_inference(
            model, dataset, device, image_size, batch_size, num_workers
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        y_true, y_pred, y_prob = _run_inference(model, loader, device)

    # Calcular y registrar métricas
    metrics: ClassificationMetrics = compute_metrics(y_true, y_pred, y_prob, dataset.classes)

    # Guardar visualizaciones
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_confusion_matrix(
        metrics.cm,
        dataset.classes,
        save_path=output_dir / "confusion_matrix.png",
        title=f"Matriz de Confusión — {experiment_name}",
    )
    plot_roc_curve(
        y_true,
        y_prob,
        metrics.auc_roc,
        save_path=output_dir / "roc_curve.png",
    )
    plot_precision_recall_curve(
        y_true,
        y_prob,
        metrics.auc_pr,
        save_path=output_dir / "pr_curve.png",
    )
    save_metrics_csv(
        metrics,
        save_path=output_dir / "metrics_log.csv",
        experiment_name=experiment_name,
    )

    # Resumen final
    logger.info(f"\n{'=' * 58}")
    logger.info("  Evaluación completada.")
    logger.info(f"  Resultados guardados en: {output_dir}")
    logger.info(f"  Objetivos cumplidos: {'✓ SÍ' if metrics.meets_clinical_targets() else '✗ NO'}")
    logger.info(f"{'=' * 58}")


if __name__ == "__main__":
    app()
