"""
prepare_data.py
---------------
Organiza las imágenes crudas en la estructura de directorios
esperada por MicroscopyDataset para la clasificación binaria.

Estructura de entrada (flexible):
    data/raw/
        alternaria/
        aspergillus_niger/
        fusarium_solani/
        ...

Estructura de salida:
    data/processed/classification/
        alternaria/
        otros_hongos/

También genera un reporte CSV con el inventario del dataset.

Uso:
    uv run prepare-data \
        --src-dir data/raw/ \
        --dest-dir data/processed/classification/ \
        --alternaria-folder alternaria
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

try:
    import pillow_heif
    from PIL import Image
    pillow_heif.register_heif_opener()
except ImportError:
    pass

from src.utils.logger import get_logger

app = typer.Typer(help="Prepara la estructura de directorios para clasificación binaria.")
logger = get_logger(__name__)
console = Console()

VALID_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".heic"}

OTROS_HONGOS = {
    "aspergillus_brasiliensis",
    "aspergillus_niger",
    "cladosporium_cladosporioides",
    "fusarium_solani",
    "mucor_racemosus",
    "penicillium_chrysogenum",
    "rhizopus_stolonifer",
    "trichoderma_virens",
}


def _copy_images(src_dir: Path, dest_dir: Path, label: str) -> list[dict]:
    """
    Copia imágenes de src_dir a dest_dir.
    Retorna lista de registros para el inventario CSV.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    records = []

    for img in src_dir.iterdir():
        if img.suffix.lower() not in VALID_EXT:
            continue

        dest = dest_dir / img.name
        if dest.exists():
            dest = dest_dir / f"{src_dir.name}_{img.name}"

        if img.suffix.lower() == ".heic":
            # Convert HEIC to JPG to avoid compatibility issues with PyTorch DataLoader
            dest = dest.with_suffix(".jpg")
            if not dest.exists():
                try:
                    img_heic = Image.open(img)
                    img_heic.convert("RGB").save(dest, "JPEG")
                except NameError:
                    logger.error(f"Falta pillow-heif para abrir {img.name}. Instálalo con: uv add pillow-heif")
                    continue
        else:
            shutil.copy2(img, dest)
        records.append(
            {
                "filename": dest.name,
                "label": label,
                "class_binary": label,
                "original_folder": src_dir.name,
                "path": str(dest),
            }
        )

    return records


@app.command()
def main(
    src_dir: Path = typer.Option(
        Path("data/raw"),
        "--src-dir",
        "-s",
        help="Directorio raíz con subcarpetas por especie.",
    ),
    dest_dir: Path = typer.Option(
        Path("data/processed/classification"),
        "--dest-dir",
        "-d",
        help="Directorio de destino para la estructura binaria.",
    ),
    alternaria_folder: str = typer.Option(
        "alternaria",
        help="Nombre de la carpeta que contiene Alternaria alternata.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Muestra qué haría sin copiar archivos.",
    ),
) -> None:
    """
    Organiza imágenes de data/raw/ en estructura binaria para clasificación.
    """
    if not src_dir.exists():
        logger.error(f"Directorio fuente no encontrado: {src_dir}")
        raise typer.Exit(1)

    subfolders = [d for d in src_dir.iterdir() if d.is_dir()]
    if not subfolders:
        logger.error(f"No se encontraron subcarpetas en: {src_dir}")
        raise typer.Exit(1)

    table = Table(title="Inventario de imágenes encontradas", show_header=True)
    table.add_column("Carpeta", style="cyan")
    table.add_column("Clase binaria", style="green")
    table.add_column("Imágenes", justify="right")

    all_records = []

    for folder in sorted(subfolders):
        imgs = [f for f in folder.iterdir() if f.suffix.lower() in VALID_EXT]
        if not imgs:
            continue

        folder_lower = folder.name.lower().replace(" ", "_").replace("-", "_")

        if folder_lower in [alternaria_folder.lower(), "images"]:
            binary_class = "alternaria"
        elif folder_lower == "otros_hongos" or folder_lower in OTROS_HONGOS or folder_lower.replace("_", "") in {
            k.replace("_", "") for k in OTROS_HONGOS
        }:
            binary_class = "otros_hongos"
        else:
            logger.warning(
                f"Carpeta '{folder.name}' no reconocida. "
                f"Se asignará a 'otros_hongos'. Verifica OTROS_HONGOS en prepare_data.py."
            )
            binary_class = "otros_hongos"

        table.add_row(folder.name, binary_class, str(len(imgs)))

        if not dry_run:
            dest_class_dir = dest_dir / binary_class
            records = _copy_images(folder, dest_class_dir, binary_class)
            all_records.extend(records)

    console.print(table)

    if dry_run:
        console.print("[yellow]Modo dry-run: no se copiaron archivos.[/yellow]")
        return

    if all_records:
        df = pd.DataFrame(all_records)
        report_path = dest_dir / "inventario_dataset.csv"
        df.to_csv(report_path, index=False)
        logger.info(f"Inventario guardado en: {report_path}")

        summary = df.groupby("class_binary").size()
        console.print("\n[bold]Resumen del dataset preparado:[/bold]")
        for cls, count in summary.items():
            console.print(f"  {cls}: {count} imágenes")
        console.print(f"  Total: {len(df)} imágenes")
        logger.info(f"Preparación completada. Destino: {dest_dir}")


if __name__ == "__main__":
    app()
