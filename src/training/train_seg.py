"""
train_seg.py
------------
Script de entrenamiento — Fase 2: Segmentación de estructuras fúngicas.

Entrena YOLOv11n-seg para detectar y segmentar:
  0 → conidias
  1 → conidias_multiseptadas
  2 → hifas

Prerequisitos:
  1. Anotaciones convertidas con:
     uv run convert-ann data/raw/annotations/ \
         data/processed/segmentation/labels/train/ \
         --images-dir data/raw/images/ \
         --images-out data/processed/segmentation/images/train/
  2. data.yaml generado por converter.py en:
     data/processed/segmentation/data.yaml

Uso:
    # Entrenamiento completo
    uv run train-seg --config configs/train_seg.yaml

    # Reanudar entrenamiento interrumpido
    uv run train-seg --config configs/train_seg.yaml --resume

    # Validar modelo existente sin reentrenar
    uv run train-seg --config configs/train_seg.yaml --eval-only \
        --checkpoint outputs/segmentation/alternaria_seg/weights/best.pt
"""

from __future__ import annotations

import shutil
from pathlib import Path

import typer
import yaml

from src.models.segmenter import AlternariaSEG
from src.utils.logger import get_logger

app = typer.Typer(help="Entrena YOLOv11-seg para segmentación de estructuras fúngicas.")
logger = get_logger(__name__, log_file=Path("logs/train_seg.log"))


def _validate_data_yaml(data_yaml: Path, expected_classes: list[str]) -> None:
    """
    Verifica que el data.yaml exista y contenga las clases esperadas.

    Parameters
    ----------
    data_yaml : Path
        Ruta al archivo data.yaml generado por converter.py.
    expected_classes : list[str]
        Clases esperadas en el orden correcto.
    """
    if not data_yaml.exists():
        raise FileNotFoundError(
            f"data.yaml no encontrado: {data_yaml}\nEjecuta primero: make convert-ann"
        )

    with open(data_yaml, encoding="utf-8") as f:
        content = yaml.safe_load(f)

    names = content.get("names", [])
    if names != expected_classes:
        logger.warning(
            f"Las clases en data.yaml ({names}) no coinciden con las esperadas "
            f"({expected_classes}). Verifica el archivo."
        )
    else:
        logger.info(f"data.yaml validado: {len(names)} clases → {names}")

    nc = content.get("nc", 0)
    logger.info(f"  nc={nc} | path={content.get('path', '?')}")


def _log_dataset_stats(data_yaml: Path) -> None:
    """
    Reporta estadísticas del dataset de segmentación:
    número de imágenes y anotaciones por split.
    """
    with open(data_yaml, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    base = Path(cfg.get("path", data_yaml.parent))
    logger.info("Estadísticas del dataset de segmentación:")

    for split in ["train", "val", "test"]:
        img_dir = base / cfg.get(split, f"images/{split}")
        lbl_dir = str(img_dir).replace("images", "labels")
        lbl_path = Path(lbl_dir)

        n_imgs = len(list(img_dir.glob("*.*"))) if img_dir.exists() else 0
        n_lbls = len(list(lbl_path.glob("*.txt"))) if lbl_path.exists() else 0

        logger.info(f"  {split:<6}: {n_imgs} imágenes | {n_lbls} etiquetas")


def _copy_best_checkpoint(
    training_output: Path,
    dest: Path,
) -> Path | None:
    """
    Copia best.pt desde el directorio de salida de Ultralytics
    al directorio de checkpoints del proyecto.
    """
    best_src = training_output / "weights" / "best.pt"
    if best_src.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_src, dest)
        logger.info(f"best.pt copiado a: {dest}")
        return dest
    else:
        logger.warning(f"best.pt no encontrado en: {best_src}")
        return None


@app.command()
def main(
    config: Path = typer.Option(
        Path("configs/train_seg.yaml"),
        "--config",
        "-c",
        help="Ruta al archivo de configuración YAML.",
    ),
    resume: bool = typer.Option(
        False,
        "--resume/--no-resume",
        help="Reanudar entrenamiento desde el último checkpoint.",
    ),
    eval_only: bool = typer.Option(
        False,
        "--eval-only",
        help="Solo evaluar, sin reentrenar.",
    ),
    checkpoint: Path = typer.Option(
        None,
        "--checkpoint",
        help="Checkpoint para --eval-only. Default: best_model.pt.",
    ),
) -> None:
    """
    Entrena YOLOv11n-seg para segmentación de estructuras fúngicas.

    En modo --eval-only genera el reporte de métricas sobre el test set
    sin reentrenar el modelo.
    """
    # ── Configuración ────────────────────────────────────────────────
    with open(config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    project_name = cfg["project"]["name"]
    data_yaml = Path(cfg["data"]["root"]) / "data.yaml"
    classes = cfg["data"]["classes"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    ckpt_dir = Path(cfg["checkpoints"]["save_dir"])

    logger.info(f"{'=' * 60}")
    logger.info(f"  Proyecto  : {project_name} v{cfg['project']['version']}")
    logger.info(f"  Modelo    : {model_cfg['architecture']}")
    logger.info(f"  data.yaml : {data_yaml}")
    logger.info(f"{'=' * 60}")

    # ── Validaciones previas ─────────────────────────────────────────
    _validate_data_yaml(data_yaml, classes)
    _log_dataset_stats(data_yaml)

    # ── Inicializar modelo ───────────────────────────────────────────
    weights = (
        str(checkpoint)
        if checkpoint
        else str(ckpt_dir / "best_model.pt") if eval_only else model_cfg["pretrained_weights"]
    )

    seg_model = AlternariaSEG(weights=weights)

    # ═════════════════════════════════════════════════════════════════
    # MODO EVALUACIÓN — solo métricas, sin entrenamiento
    # ═════════════════════════════════════════════════════════════════
    if eval_only:
        logger.info("\n  Modo: EVALUACIÓN ÚNICAMENTE")
        for split in ["val", "test"]:
            logger.info(f"\n  Evaluando split='{split}'...")
            seg_model.validate(
                data_yaml=data_yaml,
                imgsz=train_cfg["imgsz"],
                split=split,
                save_json=True,
            )
        logger.info("\n  Evaluación completada.")
        return

    # ═════════════════════════════════════════════════════════════════
    # ENTRENAMIENTO
    # ═════════════════════════════════════════════════════════════════
    logger.info(f"\n{'=' * 60}")
    logger.info(
        f"  Iniciando entrenamiento YOLOv11-seg\n"
        f"  epochs={train_cfg['epochs']} | "
        f"imgsz={train_cfg['imgsz']} | "
        f"batch={train_cfg['batch_size']} | "
        f"resume={resume}"
    )
    logger.info(f"{'=' * 60}")

    results = seg_model.train(
        data_yaml=data_yaml,
        epochs=train_cfg["epochs"],
        imgsz=train_cfg["imgsz"],
        batch=train_cfg["batch_size"],
        lr0=train_cfg["lr0"],
        lrf=train_cfg["lrf"],
        patience=train_cfg["patience"],
        degrees=train_cfg["degrees"],
        mosaic=train_cfg["mosaic"],
        copy_paste=train_cfg["copy_paste"],
        project="outputs/segmentation",
        name="alternaria_seg",
        resume=resume,
        # Nuevos parámetros (con defaults para compatibilidad)
        optimizer=train_cfg.get("optimizer", "auto"),
        cos_lr=train_cfg.get("cos_lr", False),
        close_mosaic=train_cfg.get("close_mosaic", 10),
        warmup_epochs=train_cfg.get("warmup_epochs", 3.0),
        mixup=train_cfg.get("mixup", 0.0),
        scale=train_cfg.get("scale", 0.0),
        translate=train_cfg.get("translate", 0.1),
    )

    # ── Copiar best.pt al directorio de checkpoints del proyecto ─────
    try:
        training_output = Path(results.save_dir)
    except AttributeError:
        training_output = Path("outputs/segmentation/alternaria_seg")

    best_dest = ckpt_dir / "best_model.pt"
    _copy_best_checkpoint(training_output, best_dest)

    # ═════════════════════════════════════════════════════════════════
    # EVALUACIÓN FINAL — TEST SET
    # ═════════════════════════════════════════════════════════════════
    logger.info(f"\n{'=' * 60}")
    logger.info("  EVALUACIÓN FINAL — TEST SET")
    logger.info(f"{'=' * 60}")

    # Recargar el mejor modelo para evaluación
    final_model = AlternariaSEG.from_checkpoint(best_dest)
    final_model.validate(
        data_yaml=data_yaml,
        imgsz=train_cfg["imgsz"],
        split="test",
        save_json=True,
    )

    logger.info("\n  Entrenamiento y evaluación completados.")
    logger.info(f"  Checkpoints en   : {ckpt_dir}")
    logger.info("  Resultados en    : outputs/segmentation/alternaria_seg/")
    logger.info("  Próximo paso     : make app  →  lanzar interfaz Streamlit")


if __name__ == "__main__":
    app()
