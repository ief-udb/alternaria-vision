"""
segmenter.py
------------
Wrapper de entrenamiento e inferencia para YOLOv11-seg (Fase 2).

Arquitectura seleccionada: YOLOv11n-seg
--------------------------------------
Justificación frente a alternativas:

  | Arquitectura     | mAP  | Vel. inf. | VRAM  | Anotaciones req. | Veredicto        |
  |------------------|------|-----------|-------|-------------------|------------------|
  | YOLOv11n-seg     | Alta | ~8 ms     | ~2 GB | ~50-100           | ✅ Recomendado   |
  | YOLOv8s-seg      | Alta | ~10 ms    | ~3 GB | ~50-100           | ✅ Alternativa   |
  | Mask R-CNN R50   | Muy alta | ~80 ms | ~6 GB | ~200+            | ⚠️ Lento, pesado |
  | U-Net            | Alta | ~15 ms    | ~4 GB | Requiere masks px | ⚠️ Sin detección |
  | SAM 2            | SOTA | ~200 ms   | ~8 GB | Pocos (prompting) | ⚠️ Sin clases    |

  YOLOv11n-seg ofrece el mejor balance para datasets pequeños (~116
  imágenes anotadas), inferencia en tiempo real, y detección +
  segmentación simultánea de las tres clases morfológicas.

Clases de segmentación:
  0 → conidias
  1 → conidias_multiseptadas
  2 → hifas

Referencias:
  Jocher et al. (2023). Ultralytics YOLOv8/v11. GitHub.
  https://github.com/ultralytics/ultralytics
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Colores BGR para visualización de cada clase (OpenCV)
CLASS_COLORS: dict[int, tuple[int, int, int]] = {
    0: (255, 180, 50),  # conidias               → amarillo-naranja
    1: (50, 200, 255),  # conidias_multiseptadas → cian
    2: (120, 255, 120),  # hifas                  → verde claro
}

CLASS_NAMES: dict[int, str] = {
    0: "conidias",
    1: "conidias_multiseptadas",
    2: "hifas",
}


class AlternariaSEG:
    """
    Wrapper para YOLOv11-seg orientado a segmentación de estructuras
    fúngicas en imágenes de microscopía.

    Encapsula: carga del modelo, entrenamiento, predicción con
    post-procesado y exportación a ONNX.

    Parameters
    ----------
    weights : str | Path
        Ruta a pesos preentrenados (.pt) o nombre del modelo base
        de Ultralytics (ej. 'yolo11n-seg.pt' para descarga automática).
    device : str | None
        Dispositivo de inferencia: 'cuda', 'mps', 'cpu' o None
        (detección automática).
    """

    def __init__(
        self,
        weights: str | Path = "yolo11n-seg.pt",
        device: str | None = None,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "Ultralytics no está instalado.\nInstala con: uv add ultralytics"
            ) from exc

        self.weights = Path(weights)
        self.device = device or self._auto_device()
        self.model = YOLO(str(weights))
        logger.info(
            f"AlternariaSEG inicializado | weights={self.weights.name} | device={self.device}"
        )

    # ── Utilidades internas ──────────────────────────────────────────────────

    @staticmethod
    def _auto_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        # Desactivado temporalmente por bug PyTorch MPS / Ultralytics en Macs
        # if torch.backends.mps.is_available():
        #     return "mps"
        return "cpu"

    # ── Entrenamiento ────────────────────────────────────────────────────────

    def train(
        self,
        data_yaml: Path,
        epochs: int = 100,
        imgsz: int = 640,
        batch: int = 16,
        lr0: float = 1e-3,
        lrf: float = 0.01,
        patience: int = 20,
        degrees: float = 180.0,
        mosaic: float = 0.5,
        copy_paste: float = 0.3,
        project: str = "outputs/segmentation",
        name: str = "alternaria_seg",
        resume: bool = False,
    ) -> Any:
        """
        Lanza el entrenamiento de YOLOv11-seg.

        Parameters
        ----------
        data_yaml : Path
            Ruta al data.yaml generado por converter.py.
        epochs : int
            Número de épocas. Default: 100.
        imgsz : int
            Tamaño de imagen de entrenamiento. Default: 640.
        batch : int
            Tamaño de batch. Default: 16.
        lr0 : float
            Learning rate inicial. Default: 1e-3.
        lrf : float
            Factor final del lr (lr_final = lr0 * lrf). Default: 0.01.
        patience : int
            Paciencia para early stopping. Default: 20.
        degrees : float
            Rango de rotación aleatoria en grados.
            180.0 = rotación libre (crítico para microscopía). Default: 180.
        mosaic : float
            Probabilidad de augmentación mosaic. Default: 0.5.
        copy_paste : float
            Probabilidad de copy-paste (útil para clases poco frecuentes).
            Default: 0.3.
        project : str
            Directorio raíz de resultados.
        name : str
            Nombre del experimento (subdirectorio en project/).
        resume : bool
            Reanudar entrenamiento desde el último checkpoint.

        Returns
        -------
        ultralytics.engine.results.Results
            Objeto de resultados de Ultralytics con métricas finales.
        """
        logger.info(
            f"Iniciando entrenamiento YOLOv11-seg | "
            f"epochs={epochs} | imgsz={imgsz} | batch={batch} | "
            f"device={self.device}"
        )
        results = self.model.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            lr0=lr0,
            lrf=lrf,
            patience=patience,
            degrees=degrees,
            mosaic=mosaic,
            copy_paste=copy_paste,
            device=self.device,
            project=project,
            name=name,
            resume=resume,
            # Aumentaciones adicionales para microscopía
            hsv_h=0.5,  # Variación de matiz (colorante)
            hsv_s=0.5,  # Variación de saturación
            hsv_v=0.3,  # Variación de valor/brillo
            flipud=0.5,  # Volteo vertical (invarianza orientación)
            fliplr=0.5,  # Volteo horizontal
            # Desactivar augmentaciones no aplicables a microscopía
            perspective=0.0,
            shear=0.0,
        )
        logger.info("Entrenamiento completado.")
        return results

    # ── Validación ───────────────────────────────────────────────────────────

    def validate(
        self,
        data_yaml: Path,
        imgsz: int = 640,
        split: str = "test",
        save_json: bool = True,
    ) -> Any:
        """
        Evalúa el modelo en el split indicado.

        Métricas reportadas por Ultralytics:
          - mAP50, mAP50-95 (métricas de detección)
          - mAP50-seg, mAP50-95-seg (métricas de segmentación)
          - Precision, Recall por clase

        Parameters
        ----------
        data_yaml : Path
            Ruta al data.yaml del dataset.
        imgsz : int
            Tamaño de imagen de evaluación.
        split : str
            Split a evaluar: 'val' o 'test'. Default: 'test'.
        save_json : bool
            Guardar predicciones en formato COCO JSON.
        """
        logger.info(f"Evaluando en split='{split}'...")
        metrics = self.model.val(
            data=str(data_yaml),
            imgsz=imgsz,
            split=split,
            save_json=save_json,
            device=self.device,
        )
        self._log_seg_metrics(metrics)
        return metrics

    @staticmethod
    def _log_seg_metrics(metrics: Any) -> None:
        """Registra las métricas de segmentación en el logger."""
        try:
            box = metrics.box
            seg = metrics.seg
            logger.info("=" * 55)
            logger.info("  MÉTRICAS DE SEGMENTACIÓN — TEST SET")
            logger.info("=" * 55)
            logger.info(f"  [Box]  mAP50      : {box.map50:.4f}")
            logger.info(f"  [Box]  mAP50-95   : {box.map:.4f}")
            logger.info(f"  [Seg]  mAP50      : {seg.map50:.4f}")
            logger.info(f"  [Seg]  mAP50-95   : {seg.map:.4f}")
            logger.info(f"  [Seg]  Precision  : {seg.mp:.4f}")
            logger.info(f"  [Seg]  Recall     : {seg.mr:.4f}")
            logger.info("-" * 55)
            logger.info("  mAP50 por clase:")
            for i, name in CLASS_NAMES.items():
                try:
                    logger.info(f"    {name:<28}: {seg.maps[i]:.4f}")
                except (IndexError, AttributeError):
                    pass
            logger.info("=" * 55)
        except AttributeError:
            logger.warning("No se pudieron extraer métricas detalladas del objeto results.")

    # ── Inferencia ───────────────────────────────────────────────────────────

    def predict(
        self,
        source: str | Path | np.ndarray,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
        return_vis: bool = True,
    ) -> tuple[list[dict], np.ndarray | None]:
        """
        Realiza predicción sobre una imagen o directorio.

        Parameters
        ----------
        source : str | Path | np.ndarray
            Imagen individual, directorio, URL o array numpy (BGR/RGB).
        conf : float
            Umbral de confianza mínimo. Default: 0.25.
        iou : float
            Umbral IoU para NMS. Default: 0.45.
        imgsz : int
            Tamaño de inferencia.
        return_vis : bool
            Si True, retorna imagen con máscaras superpuestas.

        Returns
        -------
        detections : list[dict]
            Lista de detecciones. Cada dict contiene:
            {
                "class_id"   : int,
                "class_name" : str,
                "confidence" : float,
                "bbox"       : [x1, y1, x2, y2],   # píxeles
                "mask"       : np.ndarray,           # bool H×W
                "area_px"    : int,
            }
        vis_image : np.ndarray | None
            Imagen BGR con máscaras y etiquetas superpuestas.
            None si return_vis=False.
        """
        results = self.model.predict(
            source=source,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=self.device,
            verbose=False,
        )

        detections: list[dict] = []
        vis_image: np.ndarray | None = None

        for result in results:
            # Imagen base para visualización
            img = result.orig_img.copy()

            if result.masks is None:
                logger.warning("Sin máscaras en el resultado. Verificar umbral de confianza.")
                if return_vis:
                    vis_image = img
                continue

            masks_data = result.masks.data.cpu().numpy()  # (N, H, W)
            boxes_data = result.boxes.xyxy.cpu().numpy()  # (N, 4)
            confs_data = result.boxes.conf.cpu().numpy()  # (N,)
            cls_data = result.boxes.cls.cpu().numpy().astype(int)  # (N,)

            for i in range(len(cls_data)):
                cls_id = int(cls_data[i])
                cls_name = CLASS_NAMES.get(cls_id, f"class_{cls_id}")
                conf_val = float(confs_data[i])
                bbox = boxes_data[i].tolist()

                # Redimensionar máscara al tamaño original
                mask_resized = cv2.resize(
                    masks_data[i].astype(np.uint8),
                    (img.shape[1], img.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)

                area_px = int(mask_resized.sum())

                detections.append(
                    {
                        "class_id": cls_id,
                        "class_name": cls_name,
                        "confidence": conf_val,
                        "bbox": bbox,
                        "mask": mask_resized,
                        "area_px": area_px,
                    }
                )

                # Superponer máscara en imagen de visualización
                if return_vis:
                    color = CLASS_COLORS.get(cls_id, (200, 200, 200))
                    overlay = img.copy()
                    overlay[mask_resized] = (
                        overlay[mask_resized] * 0.45 + np.array(color, dtype=np.float32) * 0.55
                    ).astype(np.uint8)
                    img = overlay

                    # Contorno
                    contours, _ = cv2.findContours(
                        mask_resized.astype(np.uint8),
                        cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE,
                    )
                    cv2.drawContours(img, contours, -1, color, 1)

                    # Etiqueta con clase y confianza
                    x1, y1 = int(bbox[0]), int(bbox[1])
                    label_txt = f"{cls_name} {conf_val:.2f}"
                    (tw, th), _ = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                    cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
                    cv2.putText(
                        img,
                        label_txt,
                        (x1 + 2, y1 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (20, 20, 20),
                        1,
                        cv2.LINE_AA,
                    )

            if return_vis:
                vis_image = img

        return detections, vis_image

    def predict_and_count(
        self,
        source: str | Path | np.ndarray,
        conf: float = 0.25,
        iou: float = 0.45,
    ) -> dict[str, int]:
        """
        Predice y retorna el conteo de estructuras por clase.

        Útil para la interfaz educativa de Streamlit,
        donde el estudiante puede comparar sus conteos manuales
        con los del modelo.

        Returns
        -------
        dict[str, int]
            Ej: {'conidias': 12, 'conidias_multiseptadas': 5, 'hifas': 3}
        """
        detections, _ = self.predict(source, conf=conf, iou=iou, return_vis=False)
        counts: dict[str, int] = {name: 0 for name in CLASS_NAMES.values()}
        for det in detections:
            counts[det["class_name"]] += 1
        return counts

    # ── Exportación ──────────────────────────────────────────────────────────

    def export_onnx(self, imgsz: int = 640) -> Path:
        """
        Exporta el modelo a ONNX para despliegue sin dependencia de PyTorch.

        Returns
        -------
        Path
            Ruta al archivo .onnx generado.
        """
        logger.info("Exportando modelo a ONNX...")
        export_path = self.model.export(
            format="onnx",
            imgsz=imgsz,
            opset=17,
            simplify=True,
            dynamic=True,
        )
        logger.info(f"Modelo ONNX exportado: {export_path}")
        return Path(export_path)

    # ── Carga desde checkpoint ───────────────────────────────────────────────

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path,
        device: str | None = None,
    ) -> AlternariaSEG:
        """
        Carga el modelo desde un checkpoint de entrenamiento.

        Parameters
        ----------
        checkpoint_path : Path
            Ruta al best.pt generado por Ultralytics durante el entrenamiento.
            Típicamente: outputs/segmentation/alternaria_seg/weights/best.pt
        device : str | None
            Dispositivo destino. None = detección automática.
        """
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint no encontrado: {checkpoint_path}\n"
                "Verifica que el entrenamiento haya completado al menos 1 época."
            )
        logger.info(f"Cargando checkpoint: {checkpoint_path}")
        return cls(weights=checkpoint_path, device=device)
