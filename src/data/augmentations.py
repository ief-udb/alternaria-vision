"""
augmentations.py
----------------
Pipeline de aumentación de datos con Albumentations para microscopía fúngica.

Consideraciones de diseño:
  1. INVARIANZA AL COLOR: El medio de contraste varía entre preparaciones
     (azul de lactofenol, verde brillante, KOH). hue_shift_limit=180 asegura
     que el modelo aprenda morfología (texturas, formas) y NO el tono del
     colorante.
  2. INVARIANZA A ORIENTACIÓN: Conidias e hifas no tienen dirección
     preferencial en preparaciones directas. Rotaciones y volteos libres
     son obligatorios.
  3. RUIDO Y DESENFOQUE: Simulan variaciones reales del sensor electrónico
     del microscopio y desenfoque por ajuste de plano focal.
  4. CLAHE: Mejora contraste local — útil en imágenes con iluminación
     desigual, característica de microscopía óptica sin fluorescencia.
  5. CoarseDropout: Simula oclusiones parciales, fuerza al modelo a
     usar features distribuidas y no depender de una región concreta.
  6. ElasticTransform / GridDistortion: Deformaciones suaves del preparado
     y variaciones de lente del microscopio.
  7. Sharpen: Variabilidad en el enfoque del plano focal.

Referencias:
  Buslaev et al. (2020). Albumentations: Fast and Flexible Image Augmentations.
  Information, 11(2), 125. https://doi.org/10.3390/info11020125
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Estadísticas de normalización ImageNet (default para fine-tuning)
# Se conservan para el fine-tuning desde pesos preentrenados en ImageNet.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

# Alias de compatibilidad con código existente que importa _MEAN / _STD
_MEAN = _IMAGENET_MEAN
_STD = _IMAGENET_STD

# Extensiones de imagen soportadas (mismas que dataset.py)
_VALID_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


# ── Utilidad: estadísticas reales del dataset ─────────────────────────────────


def compute_dataset_stats(
    root: Path | str,
    image_size: int = 288,
    max_samples: int = 500,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """
    Calcula mean y std RGB reales del dataset de clasificación.

    Útil para sustituir las estadísticas de ImageNet cuando las imágenes
    tienen una distribución de color muy diferente (microscopía con tinciones
    específicas: azul de lactofenol, verde brillante, KOH).

    Estrategia: acumula media y varianza pixel-a-pixel con el método de
    Welford para eficiencia numérica, usando un muestreo aleatorio cuando
    el dataset es grande.

    Parameters
    ----------
    root : Path | str
        Directorio raíz con subdirectorios por clase.
    image_size : int
        Tamaño al que se redimensionan las imágenes para el cálculo.
    max_samples : int
        Máximo de imágenes a procesar (para eficiencia con datasets grandes).

    Returns
    -------
    tuple[tuple[float,float,float], tuple[float,float,float]]
        (mean_rgb, std_rgb) — valores normalizados en [0, 1].
    """
    from PIL import Image

    root = Path(root)

    # Recolectar todas las rutas válidas
    paths: list[Path] = [
        p for p in root.rglob("*") if p.suffix.lower() in _VALID_EXT
    ]

    if not paths:
        return _IMAGENET_MEAN, _IMAGENET_STD

    # Muestreo aleatorio si hay demasiadas imágenes
    if len(paths) > max_samples:
        paths = random.sample(paths, max_samples)

    mean_acc = np.zeros(3, dtype=np.float64)
    sq_acc = np.zeros(3, dtype=np.float64)
    n = 0

    for p in paths:
        try:
            img = np.array(
                Image.open(p).convert("RGB").resize((image_size, image_size)),
                dtype=np.float32,
            ) / 255.0  # HxWx3 en [0,1]
            pixels = img.reshape(-1, 3)  # (H*W, 3)
            mean_acc += pixels.mean(axis=0)
            sq_acc += (pixels ** 2).mean(axis=0)
            n += 1
        except Exception:
            continue

    if n == 0:
        return _IMAGENET_MEAN, _IMAGENET_STD

    mean = mean_acc / n
    std = np.sqrt(np.maximum(sq_acc / n - mean ** 2, 0.0))
    std = np.clip(std, 1e-6, None)  # evitar división por cero en Normalize

    mean_t = tuple(float(v) for v in mean)
    std_t = tuple(float(v) for v in std)
    return mean_t, std_t  # type: ignore[return-value]


# ── Pipelines de transformación ───────────────────────────────────────────────


def get_train_transforms(
    image_size: int = 288,
    mean: tuple[float, float, float] = _IMAGENET_MEAN,
    std: tuple[float, float, float] = _IMAGENET_STD,
) -> A.Compose:
    """
    Aumentaciones para el conjunto de entrenamiento.

    Parameters
    ----------
    image_size : int
        Tamaño del crop cuadrado final. Default=288 (EfficientNet-B2).
    mean : tuple[float, float, float]
        Mean RGB para normalización. Default: stats ImageNet.
    std : tuple[float, float, float]
        Std RGB para normalización. Default: stats ImageNet.
    """
    return A.Compose(
        [
            # ── Geometría: invarianza a orientación ───────────────────────
            A.RandomRotate90(p=0.75),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Affine(
                translate_percent=(-0.05, 0.05),
                scale=(0.9, 1.1),
                rotate=(-45, 45),
                border_mode=0,
                p=0.5,
            ),
            # ── Deformaciones elásticas (simulación de preparado) ─────────
            # ElasticTransform: deformaciones suaves del tejido o esporas.
            # Útil porque el montaje del preparado introduce distorsiones
            # que cambian entre láminas.
            A.ElasticTransform(alpha=1.0, sigma=50, p=0.2),
            # GridDistortion: simula variaciones de lente del microscopio
            # (aberraciones de campo, curvatura de la imagen).
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.2),
            # ── Invarianza al color del medio de contraste ─────────────────
            # Aplica uno de tres métodos aleatoriamente para máxima cobertura
            # del espacio de color presente en distintos laboratorios.
            A.OneOf(
                [
                    # Rotación completa del espectro H en HSV
                    A.HueSaturationValue(
                        hue_shift_limit=180,  # ← cubre azul→verde→amarillo→rojo
                        sat_shift_limit=50,
                        val_shift_limit=30,
                        p=1.0,
                    ),
                    # Permutación de canales R/G/B (agnóstico al color)
                    A.ChannelShuffle(p=1.0),
                    # Escala de grises: fuerza al modelo a aprender solo morfología
                    A.ToGray(p=1.0),
                ],
                p=0.75,
            ),
            # ── Brillo y contraste ────────────────────────────────────────
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.3,
                p=0.7,
            ),
            A.RandomGamma(gamma_limit=(70, 130), p=0.4),
            # CLAHE: realce adaptativo del histograma por regiones (8×8 tiles)
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),
            # ── Nitidez variable (variabilidad de enfoque del microscopio) ─
            # Simula que el plano focal no siempre es óptimo.
            A.Sharpen(alpha=(0.1, 0.4), lightness=(0.8, 1.2), p=0.3),
            # ── Ruido y desenfoque (simulación de microscopio óptico) ──────
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            A.GaussNoise(std_range=(0.04, 0.2), p=0.4),
            # ── Oclusiones parciales (CoarseDropout) ─────────────────────
            # Elimina parches rectangulares aleatorios de la imagen. Fuerza
            # al modelo a distribuir la atención y no depender de una sola
            # región del campo visual. Simula suciedad en el preparado.
            A.CoarseDropout(
                num_holes_range=(1, 4),
                hole_height_range=(0.05, 0.15),
                hole_width_range=(0.05, 0.15),
                fill=0,
                p=0.3,
            ),
            # ── Crop y resize ─────────────────────────────────────────────
            A.RandomResizedCrop(
                size=(image_size, image_size),
                scale=(0.7, 1.0),
                ratio=(0.9, 1.1),
                p=0.5,
            ),
            A.Resize(image_size, image_size),
            # ── Normalización y conversión a tensor ───────────────────────
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )


def get_val_transforms(
    image_size: int = 288,
    mean: tuple[float, float, float] = _IMAGENET_MEAN,
    std: tuple[float, float, float] = _IMAGENET_STD,
) -> A.Compose:
    """
    Transformaciones deterministas para validación y test.

    No aplica aumentaciones: solo resize + normalización.

    Parameters
    ----------
    image_size : int
        Tamaño final de la imagen.
    mean : tuple[float, float, float]
        Mean RGB para normalización.
    std : tuple[float, float, float]
        Std RGB para normalización.
    """
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )


def get_tta_transforms(
    image_size: int = 288,
    mean: tuple[float, float, float] = _IMAGENET_MEAN,
    std: tuple[float, float, float] = _IMAGENET_STD,
) -> list[A.Compose]:
    """
    Test-Time Augmentation (TTA): 8 transformaciones deterministas.

    Promedia las predicciones sobre 8 versiones aumentadas de cada imagen
    para mejorar la robustez en el test set final.

    Parameters
    ----------
    image_size : int
        Tamaño final de la imagen.
    mean : tuple[float, float, float]
        Mean RGB para normalización.
    std : tuple[float, float, float]
        Std RGB para normalización.

    Returns
    -------
    list[A.Compose]
        Lista de 8 pipelines de transformación.
    """
    base = [
        A.Resize(image_size, image_size),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ]
    augmentations = [
        [],  # 1. Original
        [A.HorizontalFlip(p=1.0)],  # 2. H-flip
        [A.VerticalFlip(p=1.0)],  # 3. V-flip
        [A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0)],  # 4. H+V flip
        [A.RandomRotate90(p=1.0)],  # 5. 90°
        [A.RandomRotate90(p=1.0), A.HorizontalFlip(p=1.0)],  # 6. 90° + H
        [A.RandomRotate90(p=1.0), A.VerticalFlip(p=1.0)],  # 7. 90° + V
        [A.Transpose(p=1.0)],  # 8. Transposición
    ]
    return [A.Compose(aug + base) for aug in augmentations]
