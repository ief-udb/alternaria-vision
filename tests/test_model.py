"""
test_model.py
-------------
Tests unitarios para src/models/classifier.py y src/models/segmenter.py.

Estrategia de prueba:
  - Las pruebas de AlternariaCLF se ejecutan en CPU (sin GPU requerida).
  - Los tests de AlternariaSEG son de integración ligera: verifican
    la interfaz pública sin ejecutar inferencia completa (que requiere
    el modelo descargado). Se usan mocks para aislar dependencias.
  - Ejecutar con: uv run pytest tests/test_model.py -v

Cobertura:
  ✓ Inicialización con arquitecturas soportadas
  ✓ forward pass: forma del tensor de salida
  ✓ freeze_backbone: parámetros activos reducidos
  ✓ unfreeze_last_n_blocks: parámetros activos incrementados
  ✓ unfreeze_all: todos los parámetros activos
  ✓ get_gradcam_target_layer: retorna nn.Module no nulo
  ✓ save / load checkpoint: consistencia de predicciones
  ✓ export_onnx: archivo .onnx generado correctamente
  ✓ AlternariaSEG: predict_and_count estructura del resultado
  ✓ AlternariaSEG.from_checkpoint: FileNotFoundError si no existe
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(
    scope="module",
    params=["efficientnet_b2", "convnext_tiny"],
)
def clf_model(request):
    """
    Instancia AlternariaCLF para cada arquitectura soportada.
    pretrained=False para evitar descarga de pesos en CI.
    """
    from src.models.classifier import AlternariaCLF

    return AlternariaCLF(
        model_name=request.param,
        num_classes=2,
        pretrained=False,
        dropout_rate=0.3,
    )


@pytest.fixture
def dummy_batch():
    """Batch de 4 imágenes RGB 288×288 (tamaño EfficientNet-B2)."""
    torch.manual_seed(42)
    return torch.randn(4, 3, 288, 288)


# ── Tests: AlternariaCLF — Forward ────────────────────────────────────────────


class TestAlternariaCLFForward:
    def test_output_shape(self, clf_model, dummy_batch):
        """forward() retorna tensor (batch_size, num_classes)."""
        clf_model.eval()
        with torch.no_grad():
            out = clf_model(dummy_batch)
        assert out.shape == (4, 2), f"Esperado (4, 2), obtenido {out.shape}"

    def test_output_is_logits_not_probs(self, clf_model, dummy_batch):
        """
        La salida es logits crudos, no probabilidades.
        Los logits no están acotados en [0, 1].
        """
        clf_model.eval()
        with torch.no_grad():
            out = clf_model(dummy_batch)
        # Verificar que no todas las filas suman a 1.0 (que serían softmax)
        row_sums = out.sum(dim=1)
        assert not torch.allclose(
            row_sums, torch.ones(4), atol=0.01
        ), "La salida debe ser logits, no probabilidades softmax."

    def test_softmax_sums_to_one(self, clf_model, dummy_batch):
        """Aplicando softmax a los logits, cada fila suma a 1."""
        clf_model.eval()
        with torch.no_grad():
            out = clf_model(dummy_batch)
            probs = torch.softmax(out, dim=1)
        assert torch.allclose(probs.sum(dim=1), torch.ones(4), atol=1e-5)

    def test_train_mode_forward(self, clf_model, dummy_batch):
        """El forward en modo train (con dropout) retorna la forma correcta."""
        clf_model.train()
        out = clf_model(dummy_batch)
        assert out.shape == (4, 2)
        clf_model.eval()  # Restaurar para otros tests


# ── Tests: AlternariaCLF — Fine-tuning API ────────────────────────────────────


class TestAlternariaCLFFineTuning:
    def test_freeze_backbone_reduces_trainable_params(self, clf_model):
        """
        Después de freeze_backbone(), los parámetros entrenables
        deben ser solo los de la cabeza (< total de parámetros).
        """
        clf_model.unfreeze_all()
        total_before = sum(p.numel() for p in clf_model.parameters() if p.requires_grad)

        clf_model.freeze_backbone()
        total_after = sum(p.numel() for p in clf_model.parameters() if p.requires_grad)

        assert (
            total_after < total_before
        ), "freeze_backbone() debe reducir el número de parámetros entrenables."

    def test_freeze_backbone_head_still_trainable(self, clf_model):
        """La cabeza clasificadora permanece entrenable después de freeze."""
        clf_model.freeze_backbone()
        head_params = sum(p.numel() for p in clf_model.head.parameters() if p.requires_grad)
        assert head_params > 0, "La cabeza debe permanecer entrenable."

    def test_freeze_backbone_backbone_not_trainable(self, clf_model):
        """El backbone queda congelado (ningún parámetro entrenable)."""
        clf_model.freeze_backbone()
        backbone_trainable = sum(
            p.numel() for p in clf_model.backbone.parameters() if p.requires_grad
        )
        assert backbone_trainable == 0, "El backbone debe estar completamente congelado."

    def test_unfreeze_last_blocks_increases_params(self, clf_model):
        """
        unfreeze_last_n_blocks() aumenta los parámetros entrenables
        respecto al estado congelado.
        """
        clf_model.freeze_backbone()
        params_frozen = sum(p.numel() for p in clf_model.parameters() if p.requires_grad)

        clf_model.unfreeze_last_n_blocks(n_blocks=2)
        params_partial = sum(p.numel() for p in clf_model.parameters() if p.requires_grad)

        assert (
            params_partial > params_frozen
        ), "unfreeze_last_n_blocks() debe aumentar los parámetros entrenables."

    def test_unfreeze_all_equals_total_params(self, clf_model):
        """unfreeze_all() hace que todos los parámetros sean entrenables."""
        clf_model.unfreeze_all()
        trainable = sum(p.numel() for p in clf_model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in clf_model.parameters())
        assert trainable == total

    def test_gradcam_target_layer_is_module(self, clf_model):
        """get_gradcam_target_layer() retorna un nn.Module válido."""
        layer = clf_model.get_gradcam_target_layer()
        assert isinstance(layer, nn.Module), "La capa objetivo de Grad-CAM debe ser un nn.Module."


# ── Tests: AlternariaCLF — Serialización ──────────────────────────────────────


class TestAlternariaCLFSerialization:
    def test_save_creates_file(self, clf_model):
        """save() crea el archivo .pt en el disco."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_ckpt.pt"
            clf_model.save(path, epoch=5, val_f1=0.91, val_acc=0.93)
            assert path.exists(), "El checkpoint debe existir tras save()."
            assert path.stat().st_size > 0, "El archivo no debe estar vacío."

    def test_load_restores_predictions(self, clf_model, dummy_batch):
        """
        Un modelo cargado desde checkpoint produce las mismas
        predicciones que el modelo original.
        """
        from src.models.classifier import AlternariaCLF

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_ckpt.pt"
            clf_model.eval()
            clf_model.save(path, epoch=5, val_f1=0.91, val_acc=0.93)

            loaded = AlternariaCLF.load(path, device=torch.device("cpu"))
            loaded.eval()

            with torch.no_grad():
                out_orig = clf_model(dummy_batch)
                out_loaded = loaded(dummy_batch)

            assert torch.allclose(
                out_orig, out_loaded, atol=1e-5
            ), "El modelo cargado debe producir predicciones idénticas."

    def test_save_checkpoint_contains_metadata(self, clf_model):
        """El checkpoint guardado contiene los campos de metadatos esperados."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "meta_test.pt"
            clf_model.save(path, epoch=10, val_f1=0.88, val_acc=0.90)

            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            for key in [
                "model_name",
                "num_classes",
                "dropout_rate",
                "model_state_dict",
                "epoch",
                "val_f1",
                "val_acc",
            ]:
                assert key in ckpt, f"Campo faltante en checkpoint: '{key}'"

    def test_export_onnx(self, clf_model):
        """export_onnx() genera un archivo .onnx válido."""
        pytest.importorskip("onnx", reason="onnx no instalado — saltando test ONNX.")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.onnx"
            clf_model.eval()
            clf_model.export_onnx(path, image_size=288)
            assert path.exists(), "El archivo .onnx debe existir."
            assert path.stat().st_size > 0


# ── Tests: AlternariaSEG ──────────────────────────────────────────────────────


class TestAlternariaSEG:
    """
    Tests de interfaz para AlternariaSEG.
    Se usa mock de Ultralytics para aislar la dependencia del modelo YOLO.
    """

    def test_from_checkpoint_raises_if_not_found(self):
        """from_checkpoint() lanza FileNotFoundError si el .pt no existe."""
        from src.models.segmenter import AlternariaSEG

        with pytest.raises(FileNotFoundError, match="Checkpoint no encontrado"):
            AlternariaSEG.from_checkpoint(Path("nonexistent/best.pt"))

    def test_predict_and_count_returns_all_classes(self):
        """
        predict_and_count() retorna un dict con las tres clases,
        incluso si su conteo es 0.
        """
        from src.models.segmenter import CLASS_NAMES, AlternariaSEG

        # Mock de Ultralytics YOLO
        with patch("src.models.segmenter.AlternariaSEG.__init__", return_value=None):
            seg = AlternariaSEG.__new__(AlternariaSEG)
            seg.model = MagicMock()
            seg.device = "cpu"

            # Simular resultado sin detecciones
            mock_result = MagicMock()
            mock_result.masks = None
            mock_result.orig_img = np.zeros((640, 640, 3), dtype=np.uint8)
            seg.model.predict = MagicMock(return_value=[mock_result])

            dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
            counts = seg.predict_and_count(dummy_img)

        assert isinstance(counts, dict)
        for cls_name in CLASS_NAMES.values():
            assert cls_name in counts, f"La clave '{cls_name}' debe estar en el resultado."

    def test_predict_and_count_zero_without_detections(self):
        """Sin detecciones, todos los conteos son 0."""
        from src.models.segmenter import AlternariaSEG

        with patch("src.models.segmenter.AlternariaSEG.__init__", return_value=None):
            seg = AlternariaSEG.__new__(AlternariaSEG)
            seg.model = MagicMock()
            seg.device = "cpu"

            mock_result = MagicMock()
            mock_result.masks = None
            mock_result.orig_img = np.zeros((640, 640, 3), dtype=np.uint8)
            seg.model.predict = MagicMock(return_value=[mock_result])

            counts = seg.predict_and_count(np.zeros((640, 640, 3), dtype=np.uint8))

        assert all(
            v == 0 for v in counts.values()
        ), "Sin detecciones, todos los conteos deben ser 0."

    def test_class_colors_cover_all_classes(self):
        """CLASS_COLORS contiene una entrada para cada clase de CLASS_NAMES."""
        from src.models.segmenter import CLASS_COLORS, CLASS_NAMES

        for class_id in CLASS_NAMES:
            assert (
                class_id in CLASS_COLORS
            ), f"CLASS_COLORS no tiene color para class_id={class_id}."

    def test_class_colors_are_valid_bgr(self):
        """Cada color BGR está en el rango [0, 255]³."""
        from src.models.segmenter import CLASS_COLORS

        for class_id, color in CLASS_COLORS.items():
            assert len(color) == 3, f"Color de clase {class_id} debe tener 3 canales."
            for channel_value in color:
                assert (
                    0 <= channel_value <= 255
                ), f"Valor de canal fuera de rango en clase {class_id}: {channel_value}"
