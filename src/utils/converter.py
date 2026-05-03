"""
converter.py
------------
Convierte anotaciones de X-AnyLabeling (polígonos SAM, formato JSON)
al formato YOLO Segmentation (.txt) requerido por YOLOv11-seg.

Formato de entrada (X-AnyLabeling JSON):
    {
      "imagePath": "img001.jpg",
      "imageWidth": 1280,
      "imageHeight": 960,
      "shapes": [
        {"label": "conidias", "shape_type": "polygon",
         "points": [[x1,y1], [x2,y2], ...]}
      ]
    }

Formato de salida (YOLO Segmentation .txt):
    <class_id> <x1_norm> <y1_norm> <x2_norm> <y2_norm> ...
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import track

from src.utils.logger import get_logger

app = typer.Typer(help="Convierte anotaciones X-AnyLabeling JSON → YOLO Segmentation.")
logger = get_logger(__name__)
console = Console()

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


def _normalize_points(points: list, width: int, height: int) -> list[str]:
    """Normaliza coordenadas de píxeles al rango [0, 1]."""
    normalized = []
    for x, y in points:
        xn = max(0.0, min(1.0, float(x) / width))
        yn = max(0.0, min(1.0, float(y) / height))
        normalized.extend([f"{xn:.6f}", f"{yn:.6f}"])
    return normalized


def convert_single(
    json_path: Path,
    labels_out: Path,
    images_out: Path | None = None,
    images_src: Path | None = None,
    class_map: dict[str, int] = CLASS_MAP,
) -> tuple[int, int]:
    """
    Convierte un archivo JSON de anotación a formato YOLO.

    Returns
    -------
    tuple[int, int]
        (n_objetos_convertidos, n_clases_ignoradas)
    """
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    w = data.get("imageWidth", 1)
    h = data.get("imageHeight", 1)
    img_stem = Path(data.get("imagePath", json_path.stem)).stem

    labels_out.mkdir(parents=True, exist_ok=True)
    txt_path = labels_out / f"{img_stem}.txt"

    converted = 0
    ignored = 0

    with open(txt_path, "w") as f_out:
        for shape in data.get("shapes", []):
            label = shape.get("label", "").strip().lower()
            label = LABEL_ALIASES.get(label, label)
            stype = shape.get("shape_type", "")
            pts = shape.get("points", [])

            if stype not in {"polygon", "rotation"}:
                continue
            if len(pts) < 3:
                logger.warning(f"Polígono con <3 puntos en {json_path.name}. Ignorado.")
                continue
            if label not in class_map:
                logger.warning(
                    f"Clase desconocida '{label}' en {json_path.name}. "
                    f"Válidas: {list(class_map.keys())}"
                )
                ignored += 1
                continue

            coords = _normalize_points(pts, w, h)
            f_out.write(f"{class_map[label]} " + " ".join(coords) + "\n")
            converted += 1

    if images_out is not None and images_src is not None:
        img_filename = data.get("imagePath", f"{img_stem}.jpg")
        src_img = images_src / img_filename
        if src_img.exists():
            images_out.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_img, images_out / src_img.name)
        else:
            logger.debug(f"Imagen no encontrada: {src_img}")

    return converted, ignored


def _write_data_yaml(base_dir: Path, class_map: dict[str, int]) -> None:
    """Genera data.yaml requerido por Ultralytics YOLOv11."""
    names = [k for k, _ in sorted(class_map.items(), key=lambda x: x[1])]
    content = (
        "# data.yaml — Dataset de segmentación para YOLOv11-seg\n"
        "# Generado por converter.py\n\n"
        f"path: {base_dir.resolve()}\n"
        "train: images/train\n"
        "val:   images/val\n"
        "test:  images/test\n\n"
        f"nc: {len(class_map)}\n"
        f"names: {names}\n"
    )
    yaml_path = base_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(content)
    logger.info(f"data.yaml generado: {yaml_path}")


@app.command()
def batch_convert(
    input_dir: Path = typer.Argument(..., help="Directorio con archivos .json de X-AnyLabeling."),
    labels_out: Path = typer.Argument(..., help="Directorio de salida para etiquetas YOLO .txt."),
    images_dir: Path = typer.Option(None, "--images-dir", help="Directorio de imágenes fuente."),
    images_out: Path = typer.Option(
        None, "--images-out", help="Directorio de salida para imágenes."
    ),
    copy_images: bool = typer.Option(True, "--copy-images/--no-copy-images"),
) -> None:
    """
    Convierte en lote todos los JSON del directorio de entrada.
    """
    json_files = sorted(input_dir.glob("*.json"))
    if not json_files:
        logger.error(f"No se encontraron archivos .json en: {input_dir}")
        raise typer.Exit(1)

    logger.info(f"Convirtiendo {len(json_files)} archivos JSON...")
    total_objs, total_ignored = 0, 0

    for jf in track(json_files, description="Convirtiendo..."):
        n_ok, n_ign = convert_single(
            json_path=jf,
            labels_out=labels_out,
            images_out=images_out if copy_images else None,
            images_src=images_dir,
        )
        total_objs += n_ok
        total_ignored += n_ign

    console.print(
        f"\n[bold green]✓ Conversión completada[/bold green]\n"
        f"  Archivos procesados : {len(json_files)}\n"
        f"  Objetos convertidos : {total_objs}\n"
        f"  Clases ignoradas    : {total_ignored}\n"
        f"  Etiquetas en        : {labels_out}\n"
    )

    _write_data_yaml(labels_out.parent.parent, CLASS_MAP)


if __name__ == "__main__":
    app()
