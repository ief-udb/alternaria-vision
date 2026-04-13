"""
run_experiments.py
------------------
Script para ejecutar secuencialmente los experimentos de segmentación
y comparar resultados entre configuraciones.

Lee todos los YAML de configs/experiments/, ejecuta cada uno,
extrae métricas por clase y genera un CSV de comparación.

Uso:
    # Ejecutar todos los experimentos
    uv run python src/training/run_experiments.py

    # Ejecutar un solo experimento
    uv run python src/training/run_experiments.py --only 1

    # Ejecutar un rango de experimentos
    uv run python src/training/run_experiments.py --only 2 --only 3

    # Solo generar reporte con resultados existentes (sin reentrenar)
    uv run python src/training/run_experiments.py --report-only
"""

from __future__ import annotations

import csv
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import typer
import yaml

from src.models.segmenter import AlternariaSEG, CLASS_NAMES
from src.utils.logger import get_logger

app = typer.Typer(
    help="Ejecuta experimentos de segmentación secuencialmente y compara resultados."
)
logger = get_logger(__name__, log_file=Path("logs/run_experiments.log"))

# ── Directorio y archivos de salida ──────────────────────────────────────

EXPERIMENTS_DIR = Path("configs/experiments")
OUTPUT_BASE = Path("outputs/segmentation")
RESULTS_CSV = OUTPUT_BASE / "experiment_results.csv"
SUMMARY_FILE = OUTPUT_BASE / "experiment_summary.txt"


def _discover_experiments(only: list[int] | None = None) -> list[Path]:
    """
    Descubre y ordena los YAML de experimentos.

    Parameters
    ----------
    only : list[int] | None
        Si se especifica, solo incluye los experimentos con estos números.

    Returns
    -------
    list[Path]
        Lista ordenada de archivos YAML de configuración.
    """
    yamls = sorted(EXPERIMENTS_DIR.glob("experiment_*.yaml"))
    if not yamls:
        raise FileNotFoundError(
            f"No se encontraron archivos de experimento en {EXPERIMENTS_DIR}.\n"
            "Verifica que existan archivos experiment_*.yaml."
        )

    if only:
        filtered = []
        for y in yamls:
            # Extraer número del nombre: experiment_1_xxx.yaml → 1
            try:
                num = int(y.stem.split("_")[1])
                if num in only:
                    filtered.append(y)
            except (IndexError, ValueError):
                continue
        yamls = filtered

    logger.info(f"Descubiertos {len(yamls)} experimentos:")
    for y in yamls:
        logger.info(f"  > {y.name}")

    return yamls


def _run_single_experiment(
    config_path: Path,
    experiment_name: str,
) -> dict:
    """
    Ejecuta un solo experimento de entrenamiento y retorna métricas.

    Parameters
    ----------
    config_path : Path
        Ruta al YAML del experimento.
    experiment_name : str
        Nombre descriptivo para el directorio de salida.

    Returns
    -------
    dict
        Diccionario con métricas del experimento.
    """
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_yaml_name = cfg["data"].get("yaml_file", "data.yaml")
    data_yaml = Path(cfg["data"]["root"]) / data_yaml_name
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]

    # Determinar pesos de inicio
    weights = model_cfg["pretrained_weights"]

    logger.info(f"Inicializando modelo: {model_cfg['architecture']}")
    seg_model = AlternariaSEG(weights=weights)

    # Ejecutar entrenamiento
    start_time = time.time()
    results = seg_model.train(
        data_yaml=data_yaml,
        epochs=train_cfg["epochs"],
        imgsz=train_cfg["imgsz"],
        batch=train_cfg["batch_size"],
        lr0=train_cfg["lr0"],
        lrf=train_cfg["lrf"],
        patience=train_cfg["patience"],
        degrees=train_cfg.get("degrees", 180.0),
        mosaic=train_cfg.get("mosaic", 0.5),
        copy_paste=train_cfg.get("copy_paste", 0.3),
        project=str(OUTPUT_BASE),
        name=experiment_name,
        resume=False,
        # Parámetros de experimentos
        optimizer=train_cfg.get("optimizer", "auto"),
        cos_lr=train_cfg.get("cos_lr", False),
        close_mosaic=train_cfg.get("close_mosaic", 10),
        warmup_epochs=train_cfg.get("warmup_epochs", 3.0),
        mixup=train_cfg.get("mixup", 0.0),
        scale=train_cfg.get("scale", 0.0),
        translate=train_cfg.get("translate", 0.1),
    )
    elapsed = time.time() - start_time

    # Extraer métricas
    metrics = _extract_metrics(results, experiment_name, elapsed, config_path)

    # Evaluar con best.pt en test set
    try:
        training_output = Path(results.save_dir)
        best_pt = training_output / "weights" / "best.pt"
        if best_pt.exists():
            logger.info(f"Evaluando best.pt en test set: {best_pt}")
            eval_model = AlternariaSEG(weights=best_pt)
            test_metrics = eval_model.validate(
                data_yaml=data_yaml,
                imgsz=train_cfg["imgsz"],
                split="test",
                save_json=False,
            )
            test_data = _extract_test_metrics(test_metrics)
            metrics.update(test_data)
    except Exception as e:
        logger.warning(f"No se pudo evaluar en test set: {e}")

    return metrics


def _extract_metrics(
    results,
    experiment_name: str,
    elapsed: float,
    config_path: Path,
) -> dict:
    """Extrae métricas del objeto results de Ultralytics."""
    metrics = {
        "experiment": experiment_name,
        "config_file": config_path.name,
        "training_time_min": round(elapsed / 60, 1),
    }

    try:
        seg = results.results_dict
        # Métricas globales de segmentación (keys de Ultralytics)
        metrics["seg_mAP50"] = round(seg.get("metrics/mAP50(M)", 0.0), 4)
        metrics["seg_mAP50_95"] = round(seg.get("metrics/mAP50-95(M)", 0.0), 4)
        metrics["seg_precision"] = round(seg.get("metrics/precision(M)", 0.0), 4)
        metrics["seg_recall"] = round(seg.get("metrics/recall(M)", 0.0), 4)
        # Métricas de box
        metrics["box_mAP50"] = round(seg.get("metrics/mAP50(B)", 0.0), 4)
        metrics["box_mAP50_95"] = round(seg.get("metrics/mAP50-95(B)", 0.0), 4)
    except (AttributeError, TypeError):
        logger.warning("No se pudieron extraer métricas del entrenamiento.")

    return metrics


def _extract_test_metrics(test_metrics) -> dict:
    """Extrae métricas por clase del test set."""
    data = {}
    try:
        seg = test_metrics.seg
        data["test_seg_mAP50"] = round(seg.map50, 4)
        data["test_seg_mAP50_95"] = round(seg.map, 4)
        data["test_seg_precision"] = round(seg.mp, 4)
        data["test_seg_recall"] = round(seg.mr, 4)

        # mAP50 por clase
        for i, name in CLASS_NAMES.items():
            try:
                data[f"test_mAP50_{name}"] = round(float(seg.maps[i]), 4)
            except (IndexError, AttributeError):
                data[f"test_mAP50_{name}"] = None
    except AttributeError:
        logger.warning("No se pudieron extraer métricas de test.")

    return data


def _load_existing_results() -> list[dict]:
    """Carga resultados existentes del CSV."""
    if not RESULTS_CSV.exists():
        return []
    with open(RESULTS_CSV, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _save_results(all_results: list[dict]) -> None:
    """Guarda todos los resultados en CSV."""
    if not all_results:
        logger.warning("No hay resultados para guardar.")
        return

    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    # Unificar campos de todos los resultados
    fieldnames: list[str] = []
    for r in all_results:
        for k in r:
            if k not in fieldnames:
                fieldnames.append(k)

    with open(RESULTS_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    logger.info(f"Resultados guardados en: {RESULTS_CSV}")


def _generate_summary(all_results: list[dict]) -> None:
    """Genera un resumen legible de los experimentos."""
    if not all_results:
        return

    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    lines = [
        "=" * 70,
        "  RESUMEN DE EXPERIMENTOS — SEGMENTACIÓN ALTERNARIA",
        f"  Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 70,
        "",
    ]

    # Encontrar mejor experimento por mAP50 de segmentación en test
    best_exp = None
    best_map50 = -1.0

    for r in all_results:
        exp_name = r.get("experiment", "?")
        lines.append(f"  ── {exp_name} ──")
        lines.append(f"    Config     : {r.get('config_file', '?')}")
        lines.append(f"    Tiempo     : {r.get('training_time_min', '?')} min")
        lines.append(f"    Seg mAP50  : {r.get('seg_mAP50', '?')}")
        lines.append(f"    Seg mAP50-95: {r.get('seg_mAP50_95', '?')}")

        # Métricas de test
        test_map50 = r.get("test_seg_mAP50")
        if test_map50 is not None and test_map50 != "":
            lines.append(f"    [TEST] Seg mAP50  : {test_map50}")
            lines.append(f"    [TEST] Precision  : {r.get('test_seg_precision', '?')}")
            lines.append(f"    [TEST] Recall     : {r.get('test_seg_recall', '?')}")

            # mAP50 por clase
            for name in CLASS_NAMES.values():
                class_map = r.get(f"test_mAP50_{name}", "?")
                lines.append(f"      {name:<28}: {class_map}")

            try:
                val = float(test_map50)
                if val > best_map50:
                    best_map50 = val
                    best_exp = exp_name
            except (ValueError, TypeError):
                pass

        lines.append("")

    if best_exp:
        lines.append("=" * 70)
        lines.append(f"  🏆 MEJOR EXPERIMENTO: {best_exp}")
        lines.append(f"     Test Seg mAP50: {best_map50:.4f}")
        lines.append("=" * 70)

    summary_text = "\n".join(lines)
    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        f.write(summary_text)

    # También imprimir al logger
    for line in lines:
        logger.info(line)

    logger.info(f"Resumen guardado en: {SUMMARY_FILE}")


@app.command()
def main(
    only: list[int] = typer.Option(
        None,
        "--only",
        "-o",
        help="Ejecutar solo experimento(s) específico(s) por número. Ej: --only 1 --only 3",
    ),
    report_only: bool = typer.Option(
        False,
        "--report-only",
        help="Solo generar reporte con resultados existentes (no reentrenar).",
    ),
) -> None:
    """
    Ejecuta experimentos de segmentación secuencialmente.

    Cada experimento usa un YAML independiente de configs/experiments/.
    Los resultados se comparan en un CSV con métricas globales y por clase.
    """
    logger.info("=" * 70)
    logger.info("  PIPELINE DE EXPERIMENTOS — SEGMENTACIÓN ALTERNARIA")
    logger.info("=" * 70)

    if report_only:
        logger.info("Modo: solo reporte (sin entrenamiento)")
        existing = _load_existing_results()
        if not existing:
            logger.error("No hay resultados previos. Ejecuta experimentos primero.")
            raise typer.Exit(code=1)
        _generate_summary(existing)
        return

    # Descubrir experimentos
    experiment_files = _discover_experiments(only)

    all_results: list[dict] = _load_existing_results()
    completed_names = {r.get("experiment") for r in all_results}

    n_total = len(experiment_files)
    n_new = 0
    n_failed = 0

    for idx, config_path in enumerate(experiment_files, 1):
        # Generar nombre del experimento a partir del archivo
        exp_name = config_path.stem  # e.g. "experiment_1_baseline_adamw"

        logger.info(f"\n{'━' * 70}")
        logger.info(f"  [{idx}/{n_total}] {exp_name}")
        logger.info(f"  Config: {config_path}")
        logger.info(f"{'━' * 70}")

        if exp_name in completed_names:
            logger.info(f"  ⏭ Ya completado previamente. Saltando.")
            logger.info(f"  (Elimina la entrada de {RESULTS_CSV} para re-ejecutar)")
            continue

        try:
            metrics = _run_single_experiment(config_path, exp_name)
            all_results.append(metrics)
            n_new += 1

            # Guardar después de cada experimento (por si se interrumpe)
            _save_results(all_results)
            logger.info(f"  ✅ {exp_name} completado.")

        except Exception as e:
            n_failed += 1
            logger.error(f"  ❌ {exp_name} falló: {e}")
            traceback.print_exc()
            # Registrar el fallo parcialmente
            all_results.append({
                "experiment": exp_name,
                "config_file": config_path.name,
                "status": f"FAILED: {e}",
            })
            _save_results(all_results)
            continue

    # Generar resumen final
    _generate_summary(all_results)

    logger.info(f"\n{'=' * 70}")
    logger.info(f"  PIPELINE COMPLETADO")
    logger.info(f"  Nuevos: {n_new} | Fallidos: {n_failed} | Total: {n_total}")
    logger.info(f"  Resultados: {RESULTS_CSV}")
    logger.info(f"  Resumen:    {SUMMARY_FILE}")
    logger.info(f"{'=' * 70}")


if __name__ == "__main__":
    app()
