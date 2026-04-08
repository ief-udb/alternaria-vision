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

Referencias:
  Buslaev et al. (2020). Albumentations: Fast and Flexible Image Augmentations.
  Information, 11(2), 125. https://doi.org/10.3390/info11020125
"""

from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2

# Estadísticas de normalización ImageNet
# Se conservan para el fine-tuning desde pesos preentrenados en ImageNet.
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


def get_train_transforms(image_size: int = 288) -> A.Compose:
    """
    Aumentaciones para el conjunto de entrenamiento.

    Parameters
    ----------
    image_size : int
        Tamaño del crop cuadrado final. Default=288 (EfficientNet-B2).
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
            # ── Invarianza al color del medio de contraste ─────────────────
            # Aplica uno de tres métodos aleatoriamente para máxima cobertura
            # del espacio de color presente en distintos laboratorios.
            A.OneOf(
                [
                    # Rotación completa del espectro H en HSV
                    A.HueSaturationValue(
                        hue_shift_limit=180,  # ← Clave: cubre azul→verde→amarillo→rojo
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
            # ── Ruido y desenfoque (simulación de microscopio óptico) ──────
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            A.GaussNoise(std_range=(0.04, 0.2), p=0.4),
            # ── Crop y resize ─────────────────────────────────────────────
            A.RandomResizedCrop(
                size=(image_size, image_size),
                scale=(0.7, 1.0),
                ratio=(0.9, 1.1),
                p=0.5,
            ),
            A.Resize(image_size, image_size),
            # ── Normalización y conversión a tensor ───────────────────────
            A.Normalize(mean=_MEAN, std=_STD),
            ToTensorV2(),
        ]
    )


def get_val_transforms(image_size: int = 288) -> A.Compose:
    """
    Transformaciones deterministas para validación y test.
    No aplica aumentaciones: solo resize + normalización.
    """
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(mean=_MEAN, std=_STD),
            ToTensorV2(),
        ]
    )


def get_tta_transforms(image_size: int = 288) -> list[A.Compose]:
    """
    Test-Time Augmentation (TTA): 8 transformaciones deterministas.

    Promedia las predicciones sobre 8 versiones aumentadas de cada imagen
    para mejorar la robustez en el test set final.

    Returns
    -------
    list[A.Compose]
        Lista de 8 pipelines de transformación.
    """
    base = [
        A.Resize(image_size, image_size),
        A.Normalize(mean=_MEAN, std=_STD),
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
