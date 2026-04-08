"""
metrics.py
----------
Métricas de evaluación para clasificación binaria en micología médica.

Justificación clínica de las métricas seleccionadas:
  - Sensitivity (Recall): Prioridad máxima en contexto diagnóstico.
    Un falso negativo (no detectar Alternaria cuando está presente) tiene
    mayor costo clínico que un falso positivo. Objetivo mínimo: ≥ 0.85.
  - Specificity: Evitar sobrediagnóstico frente a otros dematiáceos
    morfológicamente similares (Cladosporium, Ulocladium). Objetivo: ≥ 0.80.
  - AUC-ROC: Evaluación global del discriminador a todos los umbrales.
    Independiente del umbral de decisión. Objetivo: ≥ 0.90.
  - Umbral óptimo (Youden J): Maximiza Sensitivity + Specificity - 1.
    Permite ajustar el umbral de decisión más allá del default 0.5.
  - AUC-PR: Más informativa que AUC-ROC en datasets desbalanceados.
    Relevante cuando hay desproporción entre alternaria y otros hongos.

Referencias:
  Youden, W.J. (1950). Index for rating diagnostic tests.
  Cancer, 3(1), 32–35. https://doi.org/10.1002/1097-0142(1950)3:1<32::AID-CNCR2820030106>3.0.CO;2-3
  Fawcett, T. (2006). An introduction to ROC analysis.
  Pattern Recognition Letters, 27(8), 861–874.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Paleta de colores consistente con el proyecto
_COLOR_PRIMARY = "#01696f"
_COLOR_SECONDARY = "#a12c7b"
_COLOR_SUCCESS = "#437a22"
_COLOR_SURFACE = "#f7f6f2"


@dataclass
class ClassificationMetrics:
    """Contenedor de todas las métricas de una evaluación completa."""

    accuracy: float = 0.0
    sensitivity: float = 0.0  # Recall clase positiva (Alternaria)
    specificity: float = 0.0
    precision: float = 0.0
    f1: float = 0.0
    auc_roc: float = 0.0
    auc_pr: float = 0.0
    optimal_threshold: float = 0.5
    youden_j: float = 0.0
    cm: np.ndarray = field(default_factory=lambda: np.zeros((2, 2), dtype=int))
    report_str: str = ""

    def to_dict(self) -> dict[str, float]:
        """Retorna las métricas como diccionario (para CSV / MLflow)."""
        return {
            "accuracy": self.accuracy,
            "sensitivity": self.sensitivity,
            "specificity": self.specificity,
            "precision": self.precision,
            "f1_score": self.f1,
            "auc_roc": self.auc_roc,
            "auc_pr": self.auc_pr,
            "optimal_threshold": self.optimal_threshold,
            "youden_j": self.youden_j,
        }

    def meets_clinical_targets(self) -> bool:
        """
        Verifica si el modelo cumple los objetivos mínimos definidos
        para uso pedagógico en la asignatura de Micología.
        """
        return self.sensitivity >= 0.85 and self.specificity >= 0.80 and self.auc_roc >= 0.90


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str],
) -> ClassificationMetrics:
    """
    Calcula el conjunto completo de métricas de clasificación binaria.

    Parameters
    ----------
    y_true : np.ndarray
        Etiquetas reales (0 = alternaria, 1 = otros_hongos o viceversa
        según el orden alfabético detectado por MicroscopyDataset).
    y_pred : np.ndarray
        Predicciones con umbral default 0.5.
    y_prob : np.ndarray
        Probabilidad de la clase positiva (índice 1).
    class_names : list[str]
        Nombres de clases en el orden del dataset.

    Returns
    -------
    ClassificationMetrics
    """
    cm = confusion_matrix(y_true, y_pred)

    # Sensitivity y Specificity (caso binario)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        sensitivity = float(tp / (tp + fn + 1e-8))
        specificity = float(tn / (tn + fp + 1e-8))
    else:
        sensitivity = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
        specificity = 0.0

    # Umbral óptimo por índice de Youden J = Sensitivity + Specificity - 1
    fpr_arr, tpr_arr, thresholds = roc_curve(y_true, y_prob)
    youden = tpr_arr - fpr_arr
    opt_idx = int(np.argmax(youden))
    opt_thresh = float(thresholds[opt_idx])
    youden_j = float(youden[opt_idx])

    # AUC-PR (Precision-Recall)
    prec_arr, rec_arr, _ = precision_recall_curve(y_true, y_prob)
    auc_pr = float(auc(rec_arr, prec_arr))

    m = ClassificationMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        sensitivity=sensitivity,
        specificity=specificity,
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        auc_roc=float(roc_auc_score(y_true, y_prob)),
        auc_pr=auc_pr,
        optimal_threshold=opt_thresh,
        youden_j=youden_j,
        cm=cm,
        report_str=classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            zero_division=0,
        ),
    )

    _log_metrics(m)

    if m.meets_clinical_targets():
        logger.info("[bold green]✓ El modelo cumple los objetivos clínicos mínimos.[/bold green]")
    else:
        logger.warning(
            "[yellow]⚠ El modelo NO cumple todos los objetivos clínicos.\n"
            "  Considera: más datos, ajuste de umbral o data augmentation adicional.[/yellow]"
        )

    return m


def _log_metrics(m: ClassificationMetrics) -> None:
    logger.info("=" * 58)
    logger.info("  REPORTE DE EVALUACIÓN — Clasificación Binaria")
    logger.info("=" * 58)
    logger.info(f"  Accuracy           : {m.accuracy:.4f}")
    logger.info(f"  Sensitivity        : {m.sensitivity:.4f}  ← Recall clase positiva")
    logger.info(f"  Specificity        : {m.specificity:.4f}")
    logger.info(f"  Precision          : {m.precision:.4f}")
    logger.info(f"  F1-Score           : {m.f1:.4f}")
    logger.info(f"  AUC-ROC            : {m.auc_roc:.4f}")
    logger.info(f"  AUC-PR             : {m.auc_pr:.4f}")
    logger.info(f"  Umbral óptimo      : {m.optimal_threshold:.4f}  (Youden J={m.youden_j:.4f})")
    logger.info("-" * 58)
    logger.info("  Reporte por clase:")
    logger.info(m.report_str)
    logger.info("  Objetivos clínicos: Sensitivity≥0.85 | Specificity≥0.80 | AUC-ROC≥0.90")
    logger.info("=" * 58)


# ── Funciones de visualización ───────────────────────────────────────────────


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    save_path: Path | None = None,
    title: str = "Matriz de Confusión",
) -> None:
    """
    Guarda la matriz de confusión con normalización porcentual.

    Muestra tanto el conteo absoluto como el porcentaje en cada celda
    para facilitar la interpretación pedagógica.
    """
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)

    # Conteos absolutos
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[0],
        linewidths=0.5,
    )
    axes[0].set_xlabel("Predicción", fontsize=11)
    axes[0].set_ylabel("Etiqueta Real", fontsize=11)
    axes[0].set_title("Conteos absolutos", fontsize=11)

    # Porcentajes normalizados
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".1f",
        cmap="Greens",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[1],
        linewidths=0.5,
    )
    axes[1].set_xlabel("Predicción", fontsize=11)
    axes[1].set_ylabel("Etiqueta Real", fontsize=11)
    axes[1].set_title("Porcentaje por fila (%)", fontsize=11)

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Matriz de confusión guardada → {save_path}")
    plt.close(fig)


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    auc_roc: float,
    save_path: Path | None = None,
) -> None:
    """
    Guarda la curva ROC con el punto óptimo de Youden y línea de referencia.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    opt_idx = int(np.argmax(tpr - fpr))

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color=_COLOR_PRIMARY, lw=2, label=f"Modelo (AUC-ROC = {auc_roc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Clasificador aleatorio (AUC = 0.50)")
    ax.scatter(
        fpr[opt_idx],
        tpr[opt_idx],
        color=_COLOR_SECONDARY,
        s=90,
        zorder=5,
        label=f"Umbral óptimo = {thresholds[opt_idx]:.3f}\n"
        f"(Sens={tpr[opt_idx]:.3f}, 1-Spec={fpr[opt_idx]:.3f})",
    )
    ax.set_xlabel("1 − Especificidad (Tasa de Falsos Positivos)", fontsize=11)
    ax.set_ylabel("Sensibilidad (Tasa de Verdaderos Positivos)", fontsize=11)
    ax.set_title(
        "Curva ROC — Alternaria alternata vs Otros Hongos",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.grid(alpha=0.3)
    ax.set_facecolor(_COLOR_SURFACE)
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Curva ROC guardada → {save_path}")
    plt.close(fig)


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    auc_pr: float,
    save_path: Path | None = None,
) -> None:
    """Guarda la curva Precision-Recall."""
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    baseline = float(y_true.sum()) / len(y_true)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rec, prec, color=_COLOR_SUCCESS, lw=2, label=f"Modelo (AUC-PR = {auc_pr:.4f})")
    ax.axhline(
        baseline,
        color="gray",
        lw=1,
        linestyle="--",
        label=f"Baseline (prevalencia = {baseline:.2f})",
    )
    ax.set_xlabel("Recall (Sensibilidad)", fontsize=11)
    ax.set_ylabel("Precision (Valor Predictivo Positivo)", fontsize=11)
    ax.set_title("Curva Precision-Recall", fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.grid(alpha=0.3)
    ax.set_facecolor(_COLOR_SURFACE)
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Curva PR guardada → {save_path}")
    plt.close(fig)


def plot_training_history(
    train_losses: list[float],
    val_losses: list[float],
    val_f1s: list[float],
    save_path: Path | None = None,
) -> None:
    """
    Guarda la curva de pérdida y F1 a lo largo del entrenamiento.
    Incluye anotación del epoch con mejor F1.
    """
    epochs = list(range(1, len(train_losses) + 1))
    best_epoch = int(np.argmax(val_f1s)) + 1
    best_f1 = max(val_f1s)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    fig.suptitle("Historial de Entrenamiento", fontsize=14, fontweight="bold")

    # Pérdida
    ax1.plot(epochs, train_losses, label="Train loss", color=_COLOR_PRIMARY, lw=2)
    ax1.plot(epochs, val_losses, label="Val loss", color=_COLOR_SECONDARY, lw=2, linestyle="--")
    ax1.axvline(best_epoch, color="gray", lw=1, linestyle=":", alpha=0.7)
    ax1.set_xlabel("Época")
    ax1.set_ylabel("Pérdida (CrossEntropy)")
    ax1.set_title("Curva de Pérdida")
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.set_facecolor(_COLOR_SURFACE)

    # F1-Score
    ax2.plot(epochs, val_f1s, color=_COLOR_SUCCESS, lw=2)
    ax2.axvline(
        best_epoch, color="gray", lw=1, linestyle=":", alpha=0.7, label=f"Mejor época: {best_epoch}"
    )
    ax2.scatter(
        [best_epoch],
        [best_f1],
        color=_COLOR_SECONDARY,
        s=80,
        zorder=5,
        label=f"Mejor F1 = {best_f1:.4f}",
    )
    ax2.set_xlabel("Época")
    ax2.set_ylabel("F1-Score (Validación)")
    ax2.set_title("F1-Score en Validación")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)
    ax2.set_facecolor(_COLOR_SURFACE)

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Historial de entrenamiento guardado → {save_path}")
    plt.close(fig)


def save_metrics_csv(
    metrics: ClassificationMetrics,
    save_path: Path,
    experiment_name: str = "run",
) -> None:
    """
    Guarda las métricas como CSV para registro científico / comparación
    entre experimentos (modelos, hiperparámetros, aumentaciones).
    """
    import pandas as pd

    row = {"experiment": experiment_name, **metrics.to_dict()}
    df = pd.DataFrame([row])
    if save_path.exists():
        df_prev = pd.read_csv(save_path)
        df = pd.concat([df_prev, df], ignore_index=True)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False)
    logger.info(f"Métricas guardadas en CSV → {save_path}")
