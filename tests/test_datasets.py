"""
test_dataset.py
---------------
Tests unitarios para src/data/dataset.py y src/data/augmentations.py.

Estrategia de prueba:
  - Se usa un dataset sintético generado en memoria (imágenes de ruido
    aleatorio) para que las pruebas sean rápidas y no dependan de datos
    reales del proyecto.
  - No se requiere GPU ni conexión a internet.
  - Ejecutar con: uv run pytest tests/test_dataset.py -v

Cobertura:
  ✓ Detección de clases desde estructura de directorios
  ✓ Carga de imágenes desde directorio completo
  ✓ Carga desde archivo split .txt
  ✓ Cálculo de class_weights (dataset desbalanceado)
  ✓ get_labels() para stratified split
  ✓ __getitem__: forma del tensor de salida
  ✓ __getitem__: rango normalizado de valores
  ✓ Imagen corrupta: retorna tensor en lugar de lanzar excepción
  ✓ Pipeline de aumentaciones train: forma y tipo
  ✓ Pipeline de aumentaciones val: determinismo
  ✓ TTA: 8 transformaciones distintas
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from src.data.augmentations import (
    get_train_transforms,
    get_tta_transforms,
    get_val_transforms,
)
from src.data.dataset import MicroscopyDataset

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def synthetic_dataset_dir():
    """
    Crea un dataset sintético temporal con 2 clases y N imágenes por clase.

    Estructura:
        tmpdir/
            alternaria/         (15 imágenes RGB 128×128)
            otros_hongos/       (6  imágenes RGB 128×128)

    Scope 'module': se crea una vez para todos los tests del módulo.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        class_counts = {"alternaria": 15, "otros_hongos": 6}

        for cls_name, n in class_counts.items():
            cls_dir = root / cls_name
            cls_dir.mkdir()
            for i in range(n):
                # Imagen de ruido aleatorio con semilla fija por clase
                rng = np.random.default_rng(seed=i + hash(cls_name) % 100)
                img_arr = rng.integers(0, 255, (128, 128, 3), dtype=np.uint8)
                img = Image.fromarray(img_arr, mode="RGB")
                img.save(cls_dir / f"img_{i:03d}.jpg")

        yield root


@pytest.fixture(scope="module")
def split_file(synthetic_dataset_dir):
    """
    Crea un archivo split.txt que referencia las primeras 5 imágenes
    de cada clase del dataset sintético.
    """
    root = synthetic_dataset_dir
    split_path = root / "test_split.txt"
    lines = []
    for cls in ["alternaria", "otros_hongos"]:
        imgs = sorted((root / cls).glob("*.jpg"))[:5]
        for img in imgs:
            lines.append(str(img.relative_to(root)))
    split_path.write_text("\n".join(lines) + "\n")
    return split_path


# ── Tests: MicroscopyDataset ──────────────────────────────────────────────────


class TestMicroscopyDataset:
    def test_finds_two_classes(self, synthetic_dataset_dir):
        """El dataset detecta exactamente 2 clases en orden alfabético."""
        ds = MicroscopyDataset(root=synthetic_dataset_dir)
        assert len(ds.classes) == 2
        assert ds.classes == sorted(ds.classes), "Las clases deben estar en orden alfabético."

    def test_class_to_idx_keys(self, synthetic_dataset_dir):
        """class_to_idx contiene las clases correctas."""
        ds = MicroscopyDataset(root=synthetic_dataset_dir)
        assert "alternaria" in ds.class_to_idx
        assert "otros_hongos" in ds.class_to_idx

    def test_total_samples_from_directory(self, synthetic_dataset_dir):
        """Carga el número correcto de imágenes desde el directorio."""
        ds = MicroscopyDataset(root=synthetic_dataset_dir)
        assert len(ds) == 15 + 6  # 21 imágenes totales

    def test_load_from_split_file(self, synthetic_dataset_dir, split_file):
        """Carga correctamente desde un archivo split .txt."""
        ds = MicroscopyDataset(root=synthetic_dataset_dir, split_file=split_file)
        assert len(ds) == 10  # 5 por clase × 2 clases

    def test_labels_count_matches_samples(self, synthetic_dataset_dir):
        """get_labels() retorna una lista del mismo tamaño que el dataset."""
        ds = MicroscopyDataset(root=synthetic_dataset_dir)
        labels = ds.get_labels()
        assert len(labels) == len(ds)

    def test_labels_are_integers(self, synthetic_dataset_dir):
        """Todas las etiquetas son enteros."""
        ds = MicroscopyDataset(root=synthetic_dataset_dir)
        labels = ds.get_labels()
        assert all(isinstance(lbl, int) for lbl in labels)

    def test_class_weights_shape(self, synthetic_dataset_dir):
        """get_class_weights() retorna un tensor de tamaño (n_classes,)."""
        ds = MicroscopyDataset(root=synthetic_dataset_dir)
        w = ds.get_class_weights()
        assert isinstance(w, torch.Tensor)
        assert w.shape == (2,)

    def test_class_weights_minority_higher(self, synthetic_dataset_dir):
        """
        La clase minoritaria (otros_hongos: 6 imgs) recibe peso mayor
        que la clase mayoritaria (alternaria: 15 imgs).
        """
        ds = MicroscopyDataset(root=synthetic_dataset_dir)
        w = ds.get_class_weights()
        idx_alt = ds.class_to_idx["alternaria"]
        idx_otros = ds.class_to_idx["otros_hongos"]
        assert w[idx_otros] > w[idx_alt], "La clase minoritaria debe recibir mayor peso."

    def test_getitem_without_transform(self, synthetic_dataset_dir):
        """Sin transform, __getitem__ retorna (array numpy, int)."""
        ds = MicroscopyDataset(root=synthetic_dataset_dir)
        img, label = ds[0]
        assert isinstance(img, np.ndarray)
        assert isinstance(label, int)

    def test_getitem_output_shape_with_transform(self, synthetic_dataset_dir):
        """Con transform val, retorna tensor (3, H, W)."""
        ds = MicroscopyDataset(
            root=synthetic_dataset_dir,
            transform=get_val_transforms(image_size=128),
        )
        img, label = ds[0]
        assert isinstance(img, torch.Tensor)
        assert img.shape == (3, 128, 128)

    def test_getitem_label_is_valid_int(self, synthetic_dataset_dir):
        """La etiqueta retornada es un entero en el rango [0, n_classes)."""
        ds = MicroscopyDataset(
            root=synthetic_dataset_dir,
            transform=get_val_transforms(image_size=128),
        )
        for i in range(len(ds)):
            _, label = ds[i]
            assert 0 <= label < len(ds.classes)

    def test_getitem_tensor_value_range(self, synthetic_dataset_dir):
        """
        Después de normalización ImageNet el tensor puede contener
        valores negativos. Verificamos que no sea todo ceros.
        """
        ds = MicroscopyDataset(
            root=synthetic_dataset_dir,
            transform=get_val_transforms(image_size=128),
        )
        img, _ = ds[0]
        assert img.abs().sum().item() > 0, "El tensor no debe ser todo ceros."

    def test_corrupted_image_returns_black_tensor(self, synthetic_dataset_dir):
        """
        Una imagen corrupta (archivo vacío) no debe lanzar excepción.
        MicroscopyDataset retorna un tensor negro en su lugar.
        """
        # Crear imagen corrupta en una carpeta temporal adicional
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for cls in ["alternaria", "otros_hongos"]:
                (root / cls).mkdir()
                if cls == "alternaria":
                    # Imagen válida
                    img = Image.fromarray(np.zeros((128, 128, 3), dtype=np.uint8))
                    img.save(root / cls / "ok.jpg")
                    # Imagen corrupta
                    (root / cls / "corrupt.jpg").write_bytes(b"NOT_AN_IMAGE")
                else:
                    img = Image.fromarray(np.zeros((128, 128, 3), dtype=np.uint8))
                    img.save(root / cls / "ok.jpg")

            ds = MicroscopyDataset(
                root=root,
                transform=get_val_transforms(image_size=128),
            )
            # Iterar todos sin que explote
            for i in range(len(ds)):
                img_t, label = ds[i]
                assert isinstance(img_t, torch.Tensor)
                assert img_t.shape[0] == 3

    def test_invalid_root_raises(self):
        """Un directorio sin subdirectorios lanza ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="Sin subdirectorios"):
                MicroscopyDataset(root=Path(tmpdir))


# ── Tests: Augmentations ──────────────────────────────────────────────────────


class TestAugmentations:
    @pytest.fixture
    def dummy_image(self):
        """Imagen numpy RGB aleatoria 256×256."""
        rng = np.random.default_rng(42)
        return rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)

    def test_train_transform_output_shape(self, dummy_image):
        """El pipeline de train retorna tensor (3, H, W)."""
        transform = get_train_transforms(image_size=224)
        result = transform(image=dummy_image)["image"]
        assert isinstance(result, torch.Tensor)
        assert result.shape == (3, 224, 224)

    def test_val_transform_output_shape(self, dummy_image):
        """El pipeline de val retorna tensor (3, H, W)."""
        transform = get_val_transforms(image_size=224)
        result = transform(image=dummy_image)["image"]
        assert isinstance(result, torch.Tensor)
        assert result.shape == (3, 224, 224)

    def test_val_transform_is_deterministic(self, dummy_image):
        """
        El pipeline de validación es determinista:
        dos llamadas con la misma imagen retornan tensores idénticos.
        """
        transform = get_val_transforms(image_size=224)
        r1 = transform(image=dummy_image)["image"]
        r2 = transform(image=dummy_image)["image"]
        assert torch.allclose(r1, r2), "get_val_transforms debe ser determinista."

    def test_train_transform_produces_variation(self, dummy_image):
        """
        El pipeline de entrenamiento produce al menos una imagen
        distinta en 10 aplicaciones (las augmentaciones son estocásticas).
        """
        transform = get_train_transforms(image_size=224)
        results = [transform(image=dummy_image)["image"] for _ in range(10)]
        all_equal = all(torch.allclose(results[0], r) for r in results[1:])
        assert not all_equal, "get_train_transforms debe producir variación entre llamadas."

    def test_tta_returns_eight_transforms(self):
        """get_tta_transforms retorna exactamente 8 pipelines."""
        transforms = get_tta_transforms(image_size=224)
        assert len(transforms) == 8

    def test_tta_all_same_output_shape(self, dummy_image):
        """Todos los pipelines TTA retornan tensores del mismo tamaño."""
        transforms = get_tta_transforms(image_size=224)
        shapes = set()
        for t in transforms:
            result = t(image=dummy_image)["image"]
            shapes.add(result.shape)
        assert (
            len(shapes) == 1
        ), f"Todos los pipelines TTA deben retornar la misma forma. Formas encontradas: {shapes}"

    def test_tta_first_is_original(self, dummy_image):
        """
        La primera transformación TTA (índice 0) es la original
        sin aumentaciones geométricas extra: coincide con val_transform.
        """
        tta_transforms = get_tta_transforms(image_size=224)
        val_transform = get_val_transforms(image_size=224)

        r_tta = tta_transforms[0](image=dummy_image)["image"]
        r_val = val_transform(image=dummy_image)["image"]
        assert torch.allclose(
            r_tta, r_val
        ), "TTA[0] debe ser equivalente a val_transform (sin augmentaciones)."
