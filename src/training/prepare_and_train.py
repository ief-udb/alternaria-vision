"""
prepare_and_train.py
--------------------
Pipeline completo para preparar datos y entrenar el modelo de segmentación.

Pasos automatizados:
  1. Convertir anotaciones JSON (X-AnyLabeling) → YOLO Segmentation (.txt)
  2. Dividir en train/val/test (70/15/15) con estratificación
  3. Generar data.yaml para Ultralytics
  4. Ejecutar aumentación avanzada (Tiling + Oversampling → segmentation_v2)
  5. Lanzar entrenamiento del experimento 7 (data-centric con nuevos datos)

Uso:
    uv run python src/training/prepare_and_train.py
    uv run python src/training/prepare_and_train.py --skip-augment
    uv run python src/training/prepare_and_train.py --skip-train
    uv run python src/training/prepare_and_train.py --experiment 7
"""

from __future__ import annotations

import json
import random
import shutil
import time
from collections import Counter
from pathlib import Path

import typer
import yaml

from src.utils.logger import get_logger

app = typer.Typer(
    help="Pipeline completo: preparación de datos + aumentación + entrenamiento de segmentación."
)
logger = get_logger(__name__, log_file=Path("logs/prepare_and_train.log"))

# ── Directorios ──────────────────────────────────────────────────────────
RAW_IMAGES = Path("data/raw/images")
RAW_ANNOTATIONS = Path("data/raw/annotations")
PROCESSED_SEG = Path("data/processed/segmentation")
PROCESSED_SEG_V2 = Path("data/processed/segmentation_v2")

# ── Mapa de clases ───────────────────────────────────────────────────────
CLASS_MAP: dict[str, int] = {
    "conidia": 0,
    "conidia-multiseptada": 1,
    "hifa": 2,
}

# Aliases para labels con typos comunes en anotaciones
LABEL_ALIASES: dict[str, str] = {
    "conidia multiseptada": "conidia-multiseptada",
    "conidia_multiseptada": "conidia-multiseptada",
    "conidias": "conidia",
    "hifas": "hifa",
}

SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
RANDOM_SEED = 42


# ═════════════════════════════════════════════════════════════════════════
# PASO 1: Convertir anotaciones JSON → YOLO
# ═════════════════════════════════════════════════════════════════════════


def _normalize_points(points: list, width: int, height: int) -> list[str]:
    """Normaliza coordenadas de píxeles al rango [0, 1]."""
    normalized = []
    for x, y in points:
        xn = max(0.0, min(1.0, float(x) / width))
        yn = max(0.0, min(1.0, float(y) / height))
        normalized.extend([f"{xn:.6f}", f"{yn:.6f}"])
    return normalized


def convert_annotations() -> dict[str, list[str]]:
    """
    Convierte todos los JSON de anotaciones a formato YOLO .txt.
    Retorna un diccionario con las clases encontradas por imagen.

    Returns
    -------
    dict[str, list[str]]
        {image_stem: [class_labels_found]}
    """
    logger.info("=" * 60)
    logger.info("  PASO 1: Convertir anotaciones JSON → YOLO")
    logger.info("=" * 60)

    json_files = sorted(RAW_ANNOTATIONS.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(
            f"No se encontraron archivos .json en: {RAW_ANNOTATIONS}"
        )

    # Directorio temporal para todas las labels (luego se distribuyen por split)
    all_labels_dir = PROCESSED_SEG / "labels" / "_all"
    all_labels_dir.mkdir(parents=True, exist_ok=True)

    image_classes: dict[str, list[str]] = {}
    total_objects = 0
    total_ignored = 0
    empty_annotations = 0

    for jf in json_files:
        with open(jf, encoding="utf-8") as f:
            data = json.load(f)

        w = data.get("imageWidth", 1)
        h = data.get("imageHeight", 1)
        img_stem = Path(data.get("imagePath", jf.stem)).stem

        shapes = data.get("shapes", [])
        if not shapes:
            empty_annotations += 1
            logger.debug(f"Anotación vacía: {jf.name}")
            continue

        txt_path = all_labels_dir / f"{img_stem}.txt"
        found_classes = []
        converted_this = 0

        with open(txt_path, "w", encoding="utf-8") as f_out:
            for shape in shapes:
                label = shape.get("label", "").strip().lower()
                # Normalizar aliases comunes
                label = LABEL_ALIASES.get(label, label)
                stype = shape.get("shape_type", "")
                pts = shape.get("points", [])

                if stype not in {"polygon", "rotation"}:
                    continue
                if len(pts) < 3:
                    logger.warning(
                        f"Poligono con <3 puntos en {jf.name}. Ignorado."
                    )
                    continue
                if label not in CLASS_MAP:
                    logger.warning(
                        f"Clase desconocida '{label}' en {jf.name}. "
                        f"Validas: {list(CLASS_MAP.keys())}"
                    )
                    total_ignored += 1
                    continue

                coords = _normalize_points(pts, w, h)
                f_out.write(f"{CLASS_MAP[label]} " + " ".join(coords) + "\n")
                found_classes.append(label)
                converted_this += 1

        if converted_this > 0:
            image_classes[img_stem] = found_classes
            total_objects += converted_this
        else:
            # Si no se convirtió nada, eliminar el .txt vacío
            txt_path.unlink(missing_ok=True)

    logger.info(f"  Archivos JSON procesados : {len(json_files)}")
    logger.info(f"  Imágenes con anotaciones : {len(image_classes)}")
    logger.info(f"  Anotaciones vacías       : {empty_annotations}")
    logger.info(f"  Objetos convertidos      : {total_objects}")
    logger.info(f"  Clases ignoradas         : {total_ignored}")

    # Resumen por clase
    all_classes = [c for classes in image_classes.values() for c in classes]
    class_counts = Counter(all_classes)
    logger.info("  Distribución de instancias:")
    for cls_name, count in sorted(class_counts.items()):
        pct = count / len(all_classes) * 100 if all_classes else 0
        logger.info(f"    {cls_name:<25}: {count:>5} ({pct:.1f}%)")

    return image_classes


# ═════════════════════════════════════════════════════════════════════════
# PASO 2: Dividir en train/val/test
# ═════════════════════════════════════════════════════════════════════════


def split_dataset(image_classes: dict[str, list[str]]) -> dict[str, list[str]]:
    """
    Divide las imágenes anotadas en train/val/test con estratificación
    básica (asegura que cada split tenga muestras de clases minoritarias).

    Returns
    -------
    dict[str, list[str]]
        {"train": [stems...], "val": [stems...], "test": [stems...]}
    """
    logger.info("")
    logger.info("=" * 60)
    logger.info("  PASO 2: Dividir en train/val/test")
    logger.info("=" * 60)

    random.seed(RANDOM_SEED)
    all_stems = list(image_classes.keys())

    # Estratificar: priorizar imágenes con clases minoritarias
    has_hifa = [s for s in all_stems if "hifa" in image_classes[s]]
    has_conidia_only = [
        s for s in all_stems
        if "conidia" in image_classes[s] and s not in has_hifa
    ]
    rest = [s for s in all_stems if s not in has_hifa and s not in has_conidia_only]

    random.shuffle(has_hifa)
    random.shuffle(has_conidia_only)
    random.shuffle(rest)

    splits: dict[str, list[str]] = {"train": [], "val": [], "test": []}

    def _distribute(items: list[str]) -> None:
        n = len(items)
        n_val = max(1, round(n * SPLIT_RATIOS["val"]))
        n_test = max(1, round(n * SPLIT_RATIOS["test"]))
        n_train = n - n_val - n_test
        if n_train < 0:
            # Si hay muy pocas, poner todo en train
            splits["train"].extend(items)
            return
        splits["val"].extend(items[:n_val])
        splits["test"].extend(items[n_val : n_val + n_test])
        splits["train"].extend(items[n_val + n_test :])

    _distribute(has_hifa)
    _distribute(has_conidia_only)
    _distribute(rest)

    for split_name, stems in splits.items():
        logger.info(f"  {split_name:<6}: {len(stems)} imágenes")

    return splits


def distribute_files(splits: dict[str, list[str]]) -> None:
    """
    Copia imágenes y labels al directorio correcto por split.
    """
    logger.info("")
    logger.info("=" * 60)
    logger.info("  PASO 2b: Distribuir archivos por split")
    logger.info("=" * 60)

    all_labels_dir = PROCESSED_SEG / "labels" / "_all"

    for split_name, stems in splits.items():
        img_out = PROCESSED_SEG / "images" / split_name
        lbl_out = PROCESSED_SEG / "labels" / split_name
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        copied_imgs = 0
        copied_lbls = 0

        for stem in stems:
            # Buscar la imagen original (puede ser .jpg o .JPG)
            img_src = None
            for ext in [".jpg", ".JPG", ".png", ".PNG", ".jpeg", ".JPEG"]:
                candidate = RAW_IMAGES / f"{stem}{ext}"
                if candidate.exists():
                    img_src = candidate
                    break

            if img_src is None:
                logger.warning(f"Imagen no encontrada para: {stem}")
                continue

            # Copiar imagen
            dst_img = img_out / img_src.name
            if not dst_img.exists():
                shutil.copy2(img_src, dst_img)
            copied_imgs += 1

            # Copiar label
            lbl_src = all_labels_dir / f"{stem}.txt"
            if lbl_src.exists():
                dst_lbl = lbl_out / f"{stem}.txt"
                if not dst_lbl.exists():
                    shutil.copy2(lbl_src, dst_lbl)
                copied_lbls += 1

        logger.info(
            f"  {split_name:<6}: {copied_imgs} imágenes, {copied_lbls} labels copiados"
        )

    # Limpiar directorio temporal _all
    if all_labels_dir.exists():
        shutil.rmtree(all_labels_dir)
        logger.info("  Directorio temporal _all eliminado.")


# ═════════════════════════════════════════════════════════════════════════
# PASO 3: Generar data.yaml
# ═════════════════════════════════════════════════════════════════════════


def generate_data_yaml() -> Path:
    """Genera el data.yaml requerido por Ultralytics."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("  PASO 3: Generar data.yaml")
    logger.info("=" * 60)

    names = [k for k, _ in sorted(CLASS_MAP.items(), key=lambda x: x[1])]
    abs_path = str(PROCESSED_SEG.resolve()).replace("\\", "/")
    content = (
        "# data.yaml — Dataset de segmentación para YOLOv11-seg\n"
        "# Generado por prepare_and_train.py\n\n"
        f"path: {abs_path}\n"
        "train: images/train\n"
        "val:   images/val\n"
        "test:  images/test\n\n"
        f"nc: {len(CLASS_MAP)}\n"
        f"names: {names}\n"
    )
    yaml_path = PROCESSED_SEG / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(content)

    logger.info(f"  Generado: {yaml_path}")
    return yaml_path


# ═════════════════════════════════════════════════════════════════════════
# PASO 4: Aumentación avanzada (Tiling + Oversampling)
# ═════════════════════════════════════════════════════════════════════════


def run_advanced_augmentation() -> None:
    """Ejecuta el script de aumentación avanzada."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("  PASO 4: Aumentación Avanzada (Tiling + Oversampling)")
    logger.info("=" * 60)

    from src.data.advanced_augment import (
        process_dataset,
        create_data_yaml,
        DEST_DIR,
    )

    if DEST_DIR.exists():
        logger.info(f"  Limpiando directorio existente: {DEST_DIR}")
        shutil.rmtree(DEST_DIR)

    process_dataset("train")
    process_dataset("val")
    process_dataset("test")
    create_data_yaml()

    logger.info("  ¡Dataset V2 generado exitosamente!")

    # Log estadísticas del V2
    for split in ["train", "val", "test"]:
        img_dir = DEST_DIR / "images" / split
        lbl_dir = DEST_DIR / "labels" / split
        n_imgs = len(list(img_dir.glob("*.*"))) if img_dir.exists() else 0
        n_lbls = len(list(lbl_dir.glob("*.txt"))) if lbl_dir.exists() else 0
        logger.info(f"  V2 {split:<6}: {n_imgs} imágenes | {n_lbls} etiquetas")


# ═════════════════════════════════════════════════════════════════════════
# PASO 5: Entrenamiento
# ═════════════════════════════════════════════════════════════════════════


def run_training(experiment_num: int = 7) -> None:
    """Lanza el entrenamiento usando el runner de experimentos."""
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"  PASO 5: Entrenamiento — Experimento {experiment_num}")
    logger.info("=" * 60)

    from src.training.run_experiments import _run_single_experiment, _save_results
    from src.training.run_experiments import _generate_summary, _load_existing_results
    from src.training.run_experiments import RESULTS_CSV

    config_path = Path(f"configs/experiments/experiment_{experiment_num}_data_centric_v3.yaml")
    if not config_path.exists():
        logger.error(f"Configuración no encontrada: {config_path}")
        raise FileNotFoundError(f"Archivo de configuración no encontrado: {config_path}")

    exp_name = config_path.stem
    logger.info(f"  Config: {config_path}")
    logger.info(f"  Nombre: {exp_name}")

    start_time = time.time()

    try:
        metrics = _run_single_experiment(config_path, exp_name)
        elapsed_total = time.time() - start_time

        all_results = _load_existing_results()
        all_results.append(metrics)
        _save_results(all_results)
        _generate_summary(all_results)

        logger.info(f"  ✅ Entrenamiento completado en {elapsed_total / 60:.1f} min")
        logger.info(f"  Resultados guardados en: {RESULTS_CSV}")
    except Exception as e:
        logger.error(f"  ❌ Entrenamiento falló: {e}")
        raise


# ═════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════


@app.command()
def main(
    skip_convert: bool = typer.Option(
        False, "--skip-convert", help="Saltar conversión de anotaciones."
    ),
    skip_augment: bool = typer.Option(
        False, "--skip-augment", help="Saltar aumentación avanzada."
    ),
    skip_train: bool = typer.Option(
        False, "--skip-train", help="Saltar entrenamiento (solo preparar datos)."
    ),
    experiment: int = typer.Option(
        7, "--experiment", "-e", help="Número de experimento a ejecutar."
    ),
    clean: bool = typer.Option(
        False, "--clean", help="Limpiar datos procesados antes de empezar."
    ),
) -> None:
    """
    Pipeline completo de preparación y entrenamiento de segmentación.

    Ejecuta secuencialmente:
      1. Conversión de anotaciones JSON → YOLO
      2. División en train/val/test
      3. Generación de data.yaml
      4. Aumentación avanzada (Tiling + Oversampling)
      5. Entrenamiento del modelo
    """
    logger.info("=" * 70)
    logger.info("  PIPELINE COMPLETO — SEGMENTACIÓN ALTERNARIA V3")
    logger.info(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 70)

    pipeline_start = time.time()

    # Limpiar si se solicita
    if clean:
        logger.info("Limpiando datos procesados anteriores...")
        for d in [PROCESSED_SEG / "images", PROCESSED_SEG / "labels"]:
            for split in ["train", "val", "test"]:
                split_dir = d / split
                if split_dir.exists():
                    for f in split_dir.iterdir():
                        if f.name != ".gitkeep":
                            f.unlink()
        if PROCESSED_SEG_V2.exists():
            shutil.rmtree(PROCESSED_SEG_V2)
        logger.info("  Limpieza completada.")

    # PASO 1-2: Convertir y dividir
    if not skip_convert:
        image_classes = convert_annotations()
        splits = split_dataset(image_classes)
        distribute_files(splits)
        generate_data_yaml()
    else:
        logger.info("  Saltando conversión de anotaciones (--skip-convert)")

    # PASO 4: Aumentación
    if not skip_augment:
        run_advanced_augmentation()
    else:
        logger.info("  Saltando aumentación avanzada (--skip-augment)")

    # PASO 5: Entrenamiento
    if not skip_train:
        run_training(experiment_num=experiment)
    else:
        logger.info("  Saltando entrenamiento (--skip-train)")

    # Resumen final
    total_time = time.time() - pipeline_start
    logger.info("")
    logger.info("=" * 70)
    logger.info(f"  PIPELINE COMPLETADO — {total_time / 60:.1f} minutos")
    logger.info("=" * 70)


if __name__ == "__main__":
    app()
