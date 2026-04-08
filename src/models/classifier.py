"""
classifier.py
-------------
Modelo de clasificación binaria para microscopía fúngica.

Arquitecturas soportadas (vía timm):
  - efficientnet_b2  →  9.1 M params  | imagen 288×288 | VRAM mín. ~2 GB
  - efficientnet_b4  → 19.3 M params  | imagen 380×380 | VRAM mín. ~3 GB
  - convnext_tiny    → 28.6 M params  | imagen 288×288 | VRAM mín. ~3.5 GB

Estrategia de fine-tuning en 2 fases:
  Fase A: Backbone congelado → solo la cabeza clasificadora se entrena.
          Permite que los pesos de ImageNet se adapten gradualmente.
  Fase B: Últimos N bloques del backbone descongelados → ajuste fino.
          Preserva las capas stem (bordes, texturas de bajo nivel).

Referencias:
  Tan & Le (2019). EfficientNet: Rethinking Model Scaling for CNNs. ICML.
  Liu et al. (2022). A ConvNet for the 2020s. CVPR.
  Wightman (2019). PyTorch Image Models (timm). GitHub.
"""

from __future__ import annotations

from pathlib import Path

import timm
import torch
import torch.nn as nn

from src.utils.logger import get_logger

logger = get_logger(__name__)


class AlternariaCLF(nn.Module):
    """
    Clasificador binario Alternaria alternata / Otros hongos.

    Arquitectura:
        Backbone preentrenado (ImageNet-1k)
        → Global Average Pooling (incluido en timm)
        → Dropout(p)
        → Linear(n_features → 256)
        → BatchNorm1d(256)
        → ReLU
        → Dropout(p/2)
        → Linear(256 → num_classes)

    Parameters
    ----------
    model_name : str
        Identificador del modelo en timm.
        Default: 'efficientnet_b2'.
    num_classes : int
        Número de clases de salida. Default: 2.
    pretrained : bool
        Cargar pesos ImageNet. Default: True.
    dropout_rate : float
        Tasa de dropout en la cabeza clasificadora. Default: 0.3.
    """

    def __init__(
        self,
        model_name: str = "efficientnet_b2",
        num_classes: int = 2,
        pretrained: bool = True,
        dropout_rate: float = 0.3,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        # Backbone sin capa FC original (num_classes=0 → extractor de features)
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            drop_rate=dropout_rate,
        )
        n_feat = self.backbone.num_features

        # Cabeza clasificadora personalizada
        self.head = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(n_feat, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate * 0.5),
            nn.Linear(256, num_classes),
        )
        self._init_head()

        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            f"AlternariaCLF | arch={model_name} | "
            f"params_total={total:,} | params_trainable={trainable:,} | "
            f"n_features={n_feat} | num_classes={num_classes}"
        )

    def _init_head(self) -> None:
        """Inicializa los pesos de la cabeza con Kaiming Normal."""
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))

    # ── API de fine-tuning ───────────────────────────────────────────────────

    def freeze_backbone(self) -> None:
        """
        Fase A: Congela el backbone completo.
        Solo la cabeza clasificadora es entrenable.
        """
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.head.parameters():
            p.requires_grad = True

        n = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"[Fase A] Backbone congelado | parámetros activos: {n:,}")

    def unfreeze_last_n_blocks(self, n_blocks: int = 3) -> None:
        """
        Fase B: Descongela los últimos n_blocks del backbone.

        Mantiene congeladas las capas iniciales (stem, primeros bloques)
        que capturan características de bajo nivel transferibles
        (bordes, texturas genéricas).

        Parameters
        ----------
        n_blocks : int
            Número de bloques a descongelar desde el final del backbone.
            Recomendado: 3 para EfficientNet-B2/B4, 2 para ConvNeXt-Tiny.
        """
        # Congelar todo primero
        for p in self.backbone.parameters():
            p.requires_grad = False

        # EfficientNet usa .blocks (Sequential de bloques MBConv)
        if hasattr(self.backbone, "blocks"):
            blocks = list(self.backbone.blocks.children())
            for blk in blocks[-n_blocks:]:
                for p in blk.parameters():
                    p.requires_grad = True

        # ConvNeXt usa .stages
        elif hasattr(self.backbone, "stages"):
            stages = list(self.backbone.stages.children())
            for st in stages[-n_blocks:]:
                for p in st.parameters():
                    p.requires_grad = True

        # Capas de normalización final siempre activas
        for attr in ("bn2", "norm_head", "conv_head", "head_norm", "final_conv"):
            if hasattr(self.backbone, attr):
                for p in getattr(self.backbone, attr).parameters():
                    p.requires_grad = True

        # Cabeza clasificadora siempre activa
        for p in self.head.parameters():
            p.requires_grad = True

        n = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            f"[Fase B] Últimos {n_blocks} bloques descongelados | parámetros activos: {n:,}"
        )

    def unfreeze_all(self) -> None:
        """Fine-tuning completo: todos los parámetros son entrenables."""
        for p in self.parameters():
            p.requires_grad = True
        n = sum(p.numel() for p in self.parameters())
        logger.info(f"[Full FT] Todos los parámetros activos: {n:,}")

    # ── API de Grad-CAM ──────────────────────────────────────────────────────

    def get_gradcam_target_layer(self) -> nn.Module:
        """
        Retorna la capa objetivo para Grad-CAM / Grad-CAM++.

        Debe ser el último bloque convolucional antes del GlobalAvgPool,
        pues produce los mapas de activación espacialmente más ricos.

        Compatible con pytorch-grad-cam:
            from pytorch_grad_cam import GradCAMPlusPlus
            cam = GradCAMPlusPlus(model, target_layers=[model.get_gradcam_target_layer()])
        """
        if hasattr(self.backbone, "blocks"):
            return self.backbone.blocks[-1]
        elif hasattr(self.backbone, "stages"):
            return self.backbone.stages[-1]
        return self.backbone

    # ── Exportación ──────────────────────────────────────────────────────────

    def export_onnx(self, path: Path, image_size: int = 288) -> None:
        """
        Exporta el modelo a formato ONNX para inferencia en producción.

        Parameters
        ----------
        path : Path
            Ruta de salida del archivo .onnx.
        image_size : int
            Resolución de entrada (cuadrada).
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        dummy = torch.randn(1, 3, image_size, image_size)
        torch.onnx.export(
            self.cpu().eval(),
            dummy,
            path,
            opset_version=17,
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
        )
        logger.info(f"Modelo exportado a ONNX: {path}")

    # ── Serialización ────────────────────────────────────────────────────────

    def save(
        self,
        path: Path,
        epoch: int,
        val_f1: float,
        val_acc: float,
    ) -> None:
        """Guarda un checkpoint completo con metadatos del entrenamiento."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_name": self.model_name,
                "num_classes": self.num_classes,
                "dropout_rate": self.dropout_rate,
                "model_state_dict": self.state_dict(),
                "epoch": epoch,
                "val_f1": val_f1,
                "val_acc": val_acc,
            },
            path,
        )
        logger.info(
            f"Checkpoint guardado: {path.name} | epoch={epoch} | "
            f"val_f1={val_f1:.4f} | val_acc={val_acc:.4f}"
        )

    @classmethod
    def load(cls, path: Path, device: torch.device) -> AlternariaCLF:
        """
        Carga un modelo desde un checkpoint guardado con .save().

        Parameters
        ----------
        path : Path
            Ruta al archivo .pt generado por .save().
        device : torch.device
            Dispositivo destino.
        """
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model = cls(
            model_name=ckpt["model_name"],
            num_classes=ckpt["num_classes"],
            pretrained=False,
            dropout_rate=ckpt.get("dropout_rate", 0.3),
        )
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info(
            f"Modelo cargado: {path.name} | "
            f"epoch={ckpt.get('epoch', '?')} | "
            f"val_f1={ckpt.get('val_f1', 0.0):.4f}"
        )
        return model.to(device)
