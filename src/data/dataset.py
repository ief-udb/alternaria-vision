"""
dataset.py
----------
Clases Dataset de PyTorch para la carga de imágenes de microscopía fúngica.

Estructura de directorios esperada:
    root/
        alternaria/
            img001.jpg
            img002.png
            ...
        otros_hongos/
            img001.jpg
            ...

Características:
  - Carga completa del directorio o desde splits pre-generados (.txt).
  - Cálculo automático de class_weights para CrossEntropyLoss.
  - Reporte de distribución de clases en la inicialización.
  - Soporte para formatos: .jpg, .jpeg, .png, .tif, .tiff, .bmp.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.utils.logger import get_logger

logger = get_logger(__name__)

VALID_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


class MicroscopyDataset(Dataset):
    """
    Dataset para imágenes de microscopía fúngica (clasificación binaria).

    Parameters
    ----------
    root : Path | str
        Directorio raíz con un subdirectorio por clase.
    transform : callable | None
        Pipeline Albumentations (Compose). Si None, retorna array numpy.
    split_file : Path | str | None
        Archivo .txt con rutas relativas para usar un subconjunto.
        Si None, carga todas las imágenes encontradas.
    """

    def __init__(
        self,
        root: Path | str,
        transform=None,
        split_file: Path | str | None = None,
    ) -> None:
        self.root = Path(root)
        self.transform = transform
        self.classes, self.class_to_idx = self._find_classes()
        self.samples = self._load_samples(split_file)

        dist = self._class_distribution()
        logger.info(
            f"Dataset '{self.root.name}' | "
            f"{len(self.samples)} imágenes | "
            f"Clases: {self.classes} | "
            f"Distribución: {dist}"
        )

    # ── Métodos internos ────────────────────────────────────────────────────

    def _find_classes(self) -> tuple[list[str], dict[str, int]]:
        classes = sorted(d.name for d in self.root.iterdir() if d.is_dir())
        if not classes:
            raise ValueError(
                f"Sin subdirectorios de clase en: {self.root}\n"
                "Verifica la estructura: root/clase_a/ root/clase_b/"
            )
        return classes, {c: i for i, c in enumerate(classes)}

    def _load_samples(self, split_file: Path | str | None) -> list[tuple[Path, int]]:
        if split_file is not None:
            return self._from_split_file(Path(split_file))
        return self._from_directory()

    def _from_directory(self) -> list[tuple[Path, int]]:
        samples = []
        for cls, idx in self.class_to_idx.items():
            cls_dir = self.root / cls
            for f in cls_dir.iterdir():
                if f.suffix.lower() in VALID_EXT:
                    samples.append((f, idx))
        return samples

    def _from_split_file(self, split_file: Path) -> list[tuple[Path, int]]:
        samples = []
        with open(split_file, encoding="utf-8") as f:
            for line in f:
                rel = line.strip()
                if not rel:
                    continue
                img_path = self.root / rel
                cls_name = img_path.parent.name
                if cls_name not in self.class_to_idx:
                    logger.warning(f"Clase desconocida: '{cls_name}' (línea: {rel})")
                    continue
                samples.append((img_path, self.class_to_idx[cls_name]))
        return samples

    def _class_distribution(self) -> dict[str, int]:
        dist: dict[str, int] = {c: 0 for c in self.classes}
        for _, lbl in self.samples:
            dist[self.classes[lbl]] += 1
        return dist

    # ── API pública ──────────────────────────────────────────────────────────

    def get_class_weights(self) -> torch.Tensor:
        """
        Pesos inversamente proporcionales a la frecuencia de cada clase.

        Úsalos en nn.CrossEntropyLoss(weight=...) para manejar desbalance.

        Ejemplo:
            alternaria: 200 imágenes, otros_hongos: 80 imágenes
            → weights ≈ [1.0, 2.5]

        Returns
        -------
        torch.Tensor  shape=(n_classes,)
        """
        dist = self._class_distribution()
        counts = np.array([dist[c] for c in self.classes], dtype=np.float32)
        weights = counts.max() / counts
        return torch.tensor(weights, dtype=torch.float32)

    def get_labels(self) -> list[int]:
        """
        Retorna la lista completa de etiquetas.
        Requerido por train_test_split con stratify.
        """
        return [lbl for _, lbl in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        try:
            image = np.array(Image.open(img_path).convert("RGB"))
        except Exception as exc:
            logger.error(f"Error leyendo {img_path}: {exc}. Retornando imagen negra.")
            image = np.zeros((288, 288, 3), dtype=np.uint8)

        if self.transform is not None:
            image = self.transform(image=image)["image"]

        return image, label
