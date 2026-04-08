"""
app.py
------
Interfaz web educativa para la identificación de Alternaria alternata
mediante visión artificial.

Desarrollado con Streamlit para la asignatura de Micología
— Programa de Bacteriología, Universidad de Boyacá.

Flujo de la aplicación:
  1. Carga de imagen de microscopía
  2. Clasificación binaria: Alternaria vs Otros hongos (Fase 1)
  3. Segmentación de estructuras: conidias, conidias_multiseptadas, hifas (Fase 2)
  4. Visualización de Grad-CAM (explicabilidad del modelo)
  5. Ficha morfológica educativa de A. alternata

Lanzar con:
    uv run streamlit run app/app.py
    make app

Requisitos previos:
    - checkpoints/classification/best_model.pt  (Fase 1 entrenada)
    - checkpoints/segmentation/best_model.pt    (Fase 2 entrenada)
"""

from __future__ import annotations

import io
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image

# ── Configuración de página (debe ser el primer comando Streamlit) ───────────
st.set_page_config(
    page_title="AlternariaVision — Universidad de Boyacá",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Rutas de checkpoints ─────────────────────────────────────────────────────
CLF_CHECKPOINT = Path("checkpoints/classification/best_model.pt")
SEG_CHECKPOINT = Path("checkpoints/segmentation/best_model.pt")

# ── Constantes morfológicas ──────────────────────────────────────────────────
CLASS_NAMES_SEG = {
    0: "conidias",
    1: "conidias_multiseptadas",
    2: "hifas",
}
CLASS_COLORS_BGR = {
    0: (255, 180, 50),
    1: (50, 200, 255),
    2: (120, 255, 120),
}
CLASS_COLORS_HEX = {
    "conidias": "#FFB432",
    "conidias_multiseptadas": "#32C8FF",
    "hifas": "#78FF78",
}

MORPHOLOGY_INFO = {
    "descripcion_general": (
        "*Alternaria alternata* (Fr.) Keissl. 1912 es un hongo dematiáceo "
        "del orden **Pleosporales**, ubicuo en el ambiente. Es agente etiológico "
        "de feohifomicosis cutánea, sinusitis fúngica alérgica y queratitis "
        "fúngica en pacientes inmunocomprometidos."
    ),
    "estructuras": {
        "Conidias": {
            "descripcion": (
                "Obclavadas a ovoides, **muriformes** (septa transversales Y "
                "longitudinales). Coloración marrón media. Extremo apical en "
                "forma de pico (*beak*). Se producen en cadenas acrópetas."
            ),
            "dimensiones": "7–10 × 23–34 µm (hasta 50 µm en hábitat natural)",
            "relevancia": "Estructura diferencial primaria de la especie.",
            "color": CLASS_COLORS_HEX["conidias"],
        },
        "Conidias multiseptadas": {
            "descripcion": (
                "Conidias con **≥3 septos transversales** claramente visibles. "
                "Confirman madurez y esporulación activa del cultivo. "
                "Presentan la pigmentación marrón característica del género."
            ),
            "dimensiones": "Similares a conidias simples, mayor longitud",
            "relevancia": "Indicador de madurez y viabilidad del cultivo.",
            "color": CLASS_COLORS_HEX["conidias_multiseptadas"],
        },
        "Hifas": {
            "descripcion": (
                "Septadas, **pigmentadas** (marrón oliváceo), ramificadas. "
                "La melanización de la pared celular es característica clave "
                "que diferencia los hongos dematiáceos de los hialinos."
            ),
            "dimensiones": "2–6 µm de diámetro",
            "relevancia": "Confirma el carácter dematiáceo del hongo.",
            "color": CLASS_COLORS_HEX["hifas"],
        },
    },
    "relevancia_clinica": [
        "**Feohifomicosis** cutánea y subcutánea en inmunocomprometidos",
        "**Sinusitis fúngica alérgica** (alérgeno Alt a 1, ISAC)",
        "**Queratitis fúngica** post-traumática",
        "Oportunista en trasplantados, pacientes con SIDA y neutropénicos",
    ],
    "diagnostico_laboratorio": [
        "Examen directo: KOH 20% — hifas dematiáceas y conidias muriformes",
        "Cultivo: SDA 25°C, 7–10 días — colonias oliváceas a negras, aterciopeladas",
        "Temperatura máxima de crecimiento: 32°C (clave diferencial)",
        "Microscopía directa de colonia: conidióforos en zigzag con cadenas de conidias",
    ],
}


# ── CSS personalizado ─────────────────────────────────────────────────────────
def inject_css() -> None:
    st.markdown(
        """
    <style>
        /* Fuente y fondo */
        html, body, [class*="css"] {
            font-family: 'Segoe UI', 'Inter', sans-serif;
        }

        /* Header superior */
        .app-header {
            background: linear-gradient(135deg, #01696f 0%, #0c4e54 100%);
            padding: 1.5rem 2rem;
            border-radius: 0.75rem;
            margin-bottom: 1.5rem;
            color: white;
        }
        .app-header h1 {
            font-size: 1.8rem;
            font-weight: 700;
            margin: 0;
            color: white;
        }
        .app-header p {
            margin: 0.3rem 0 0 0;
            opacity: 0.85;
            font-size: 0.95rem;
        }

        /* Tarjetas de métricas */
        .metric-card {
            background: #f9f8f5;
            border: 1px solid #dcd9d5;
            border-radius: 0.625rem;
            padding: 1rem 1.25rem;
            text-align: center;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        }
        .metric-card .value {
            font-size: 2rem;
            font-weight: 700;
            color: #01696f;
            line-height: 1.1;
        }
        .metric-card .label {
            font-size: 0.8rem;
            color: #7a7974;
            margin-top: 0.3rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }

        /* Etiquetas de resultado */
        .result-positive {
            background: #d4dfcc;
            color: #1e3f0a;
            padding: 0.5rem 1.2rem;
            border-radius: 2rem;
            font-weight: 700;
            font-size: 1.1rem;
            display: inline-block;
        }
        .result-negative {
            background: #e0ced7;
            color: #561740;
            padding: 0.5rem 1.2rem;
            border-radius: 2rem;
            font-weight: 700;
            font-size: 1.1rem;
            display: inline-block;
        }

        /* Barra de confianza */
        .confidence-bar {
            height: 8px;
            border-radius: 4px;
            background: #edeae5;
            margin-top: 0.5rem;
            overflow: hidden;
        }
        .confidence-fill {
            height: 100%;
            border-radius: 4px;
            transition: width 0.5s ease;
        }

        /* Leyenda de colores */
        .color-dot {
            display: inline-block;
            width: 14px;
            height: 14px;
            border-radius: 50%;
            margin-right: 6px;
            vertical-align: middle;
        }

        /* Advertencia clínica */
        .clinical-warning {
            background: #ddcfc6;
            border-left: 4px solid #964219;
            padding: 0.75rem 1rem;
            border-radius: 0 0.5rem 0.5rem 0;
            font-size: 0.88rem;
            color: #4b2614;
            margin-top: 1rem;
        }

        /* Sección de pasos */
        .step-badge {
            background: #01696f;
            color: white;
            border-radius: 50%;
            width: 28px;
            height: 28px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            font-size: 0.9rem;
            margin-right: 0.5rem;
        }

        /* Ocultar menú Streamlit por defecto */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
    """,
        unsafe_allow_html=True,
    )


# ── Carga de modelos (cacheada) ───────────────────────────────────────────────


@st.cache_resource(show_spinner=False)
def load_classifier():
    """Carga AlternariaCLF desde checkpoint. Se ejecuta una sola vez."""
    if not CLF_CHECKPOINT.exists():
        return None
    try:
        from src.models.classifier import AlternariaCLF
        from src.utils.device import get_device

        device = get_device(verbose=False)
        model = AlternariaCLF.load(CLF_CHECKPOINT, device)
        model.eval()
        return model, device
    except Exception as e:
        st.error(f"Error cargando clasificador: {e}")
        return None


@st.cache_resource(show_spinner=False)
def load_segmenter():
    """Carga AlternariaSEG desde checkpoint. Se ejecuta una sola vez."""
    if not SEG_CHECKPOINT.exists():
        return None
    try:
        from src.models.segmenter import AlternariaSEG

        return AlternariaSEG.from_checkpoint(SEG_CHECKPOINT)
    except Exception as e:
        st.error(f"Error cargando segmentador: {e}")
        return None


# ── Funciones de predicción ───────────────────────────────────────────────────


def predict_classification(
    image_np: np.ndarray,
    clf_tuple,
    image_size: int = 288,
    conf_threshold: float = 0.5,
) -> dict:
    """
    Ejecuta la clasificación binaria sobre una imagen numpy RGB.

    Returns
    -------
    dict con claves:
        label (str), confidence (float), probs (dict),
        is_alternaria (bool), optimal_threshold (float)
    """
    from src.data.augmentations import get_val_transforms

    model, device = clf_tuple
    transform = get_val_transforms(image_size)
    tensor = transform(image=image_np)["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    classes = ["alternaria", "otros_hongos"]
    pred_idx = int(np.argmax(probs))

    return {
        "label": classes[pred_idx],
        "confidence": float(probs[pred_idx]),
        "probs": {c: float(p) for c, p in zip(classes, probs)},
        "is_alternaria": pred_idx == 0,
        "prob_alternaria": float(probs[0]),
        "optimal_threshold": conf_threshold,
    }


def predict_segmentation(
    image_np: np.ndarray,
    seg_model,
    conf: float = 0.25,
    iou: float = 0.45,
) -> tuple[list[dict], np.ndarray]:
    """
    Ejecuta la segmentación de estructuras sobre una imagen numpy RGB.

    Parameters
    ----------
    image_np : np.ndarray  RGB uint8
    seg_model : AlternariaSEG

    Returns
    -------
    detections : list[dict]
    vis_image  : np.ndarray BGR con máscaras superpuestas
    """
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    detections, vis_bgr = seg_model.predict(
        source=image_bgr,
        conf=conf,
        iou=iou,
        return_vis=True,
    )
    vis_rgb = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB) if vis_bgr is not None else image_np
    return detections, vis_rgb


def generate_gradcam(
    image_np: np.ndarray,
    clf_tuple,
    image_size: int = 288,
) -> np.ndarray | None:
    """
    Genera un mapa Grad-CAM++ superpuesto sobre la imagen original.

    Returns
    -------
    np.ndarray RGB | None si pytorch-grad-cam no está instalado.
    """
    try:
        from pytorch_grad_cam import GradCAMPlusPlus
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

        from src.data.augmentations import get_val_transforms

        model, device = clf_tuple
        transform = get_val_transforms(image_size)
        tensor = transform(image=image_np)["image"].unsqueeze(0).to(device)

        target_layer = [model.get_gradcam_target_layer()]
        
        with torch.no_grad():
            logits = model(tensor)
            pred_class = int(logits.argmax(1).item())

        with GradCAMPlusPlus(model=model, target_layers=target_layer) as cam:
            grayscale_cam = cam(
                input_tensor=tensor,
                targets=[ClassifierOutputTarget(pred_class)],
            )[0]

        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_disp = transform(image=image_np)["image"].cpu().numpy().transpose(1, 2, 0)
        img_disp = np.clip(img_disp * std + mean, 0, 1).astype(np.float32)

        vis = show_cam_on_image(img_disp, grayscale_cam, use_rgb=True)
        return vis

    except ImportError as e:
        print(f"ImportError en GradCAM: {e}")
        return None
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None


# ── Componentes UI ────────────────────────────────────────────────────────────


def render_header() -> None:
    st.markdown(
        """
    <div class="app-header">
        <h1>🔬 AlternariaVision</h1>
        <p>
            Prototipo de visión artificial para identificación de
            <em>Alternaria alternata</em> en microscopía óptica ·
            Instituto de Estudios del Futuro · Universidad de Boyacá
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> tuple[float, float, float, bool]:
    """
    Renderiza el panel lateral con parámetros de inferencia.

    Returns
    -------
    conf_clf (float), conf_seg (float), iou_seg (float), show_gradcam (bool)
    """
    with st.sidebar:
        st.markdown("## ⚙️ Parámetros")

        st.markdown("### Clasificación")
        conf_clf = st.slider(
            "Umbral de confianza",
            min_value=0.3,
            max_value=0.95,
            value=0.50,
            step=0.05,
            help=(
                "Umbral mínimo de probabilidad para clasificar como "
                "Alternaria alternata. El umbral óptimo de Youden se "
                "determina durante el entrenamiento."
            ),
        )

        st.markdown("### Segmentación")
        conf_seg = st.slider(
            "Confianza mínima (detección)",
            min_value=0.10,
            max_value=0.80,
            value=0.25,
            step=0.05,
            help="Umbral de confianza para aceptar una detección de estructura.",
        )
        iou_seg = st.slider(
            "Umbral IoU (NMS)",
            min_value=0.10,
            max_value=0.80,
            value=0.45,
            step=0.05,
            help=(
                "Intersection over Union para Non-Maximum Suppression. "
                "Valores menores eliminan más detecciones solapadas."
            ),
        )

        st.markdown("### Explicabilidad")
        show_gradcam = st.toggle(
            "Mostrar Grad-CAM++",
            value=True,
            help=(
                "Visualiza las regiones de la imagen que más activaron "
                "la decisión del clasificador (requiere pytorch-grad-cam)."
            ),
        )

        st.divider()
        st.markdown("### 📋 Estado de modelos")

        clf_ok = CLF_CHECKPOINT.exists()
        seg_ok = SEG_CHECKPOINT.exists()

        st.markdown(
            f"{'✅' if clf_ok else '❌'} **Clasificador** "
            f"({'listo' if clf_ok else 'checkpoint no encontrado'})"
        )
        st.markdown(
            f"{'✅' if seg_ok else '❌'} **Segmentador** "
            f"({'listo' if seg_ok else 'checkpoint no encontrado'})"
        )

        if not clf_ok:
            st.warning(
                "Ejecuta `make train-clf` para entrenar el clasificador.",
                icon="⚠️",
            )
        if not seg_ok:
            st.info(
                "Ejecuta `make train-seg` para entrenar el segmentador.",
                icon="ℹ️",
            )

        st.divider()
        st.markdown(
            "<small style='color:#7a7974'>"
            "AlternariaVision v1.0.0 · MIT License<br>"
            "Universidad de Boyacá · 2026"
            "</small>",
            unsafe_allow_html=True,
        )

    return conf_clf, conf_seg, iou_seg, show_gradcam


def render_classification_result(result: dict, conf_threshold: float) -> None:
    """Renderiza el resultado de clasificación con métricas visuales."""
    is_alt = result["prob_alternaria"] >= conf_threshold
    label = "Alternaria alternata" if is_alt else "Otro hongo"
    conf = result["prob_alternaria"] if is_alt else (1 - result["prob_alternaria"])
    css_class = "result-positive" if is_alt else "result-negative"

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        st.markdown("**Diagnóstico del modelo:**")
        st.markdown(
            f'<span class="{css_class}">{"🔴" if is_alt else "🟢"} {label}</span>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class="confidence-bar">
                <div class="confidence-fill"
                     style="width:{result["prob_alternaria"] * 100:.1f}%;
                            background:{"#437a22" if is_alt else "#a12c7b"};">
                </div>
            </div>
            <small style="color:#7a7974">
                P(Alternaria) = {result["prob_alternaria"]:.3f}
                &nbsp;|&nbsp; Umbral = {conf_threshold:.2f}
            </small>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="value">{result["prob_alternaria"]:.1%}</div>
                <div class="label">P(Alternaria)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="value">{1 - result["prob_alternaria"]:.1%}</div>
                <div class="label">P(Otros hongos)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="clinical-warning">'
        "⚠️ <strong>Aviso clínico:</strong> Este resultado es una herramienta "
        "educativa de apoyo. El diagnóstico definitivo requiere correlación con "
        "cultivo, historia clínica y criterio del bacteriólogo."
        "</div>",
        unsafe_allow_html=True,
    )


def render_segmentation_result(
    detections: list[dict],
    vis_image: np.ndarray,
    original_image: np.ndarray,
) -> None:
    """Renderiza los resultados de segmentación con conteos y leyenda."""

    # Conteo por clase
    counts = {name: 0 for name in CLASS_NAMES_SEG.values()}
    for det in detections:
        counts[det["class_name"]] += 1

    # Métricas de conteo
    st.markdown("#### Estructuras detectadas")
    c1, c2, c3, c4 = st.columns(4)
    cols = [c1, c2, c3]
    for i, (cls_name, count) in enumerate(counts.items()):
        color = CLASS_COLORS_HEX[cls_name]
        with cols[i]:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="value" style="color:{color}">{count}</div>
                    <div class="label">{cls_name.replace("_", " ")}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    with c4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="value">{len(detections)}</div>
                <div class="label">Total estructuras</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Imágenes lado a lado
    st.markdown("#### Comparación: original vs segmentado")
    img_col1, img_col2 = st.columns(2)
    with img_col1:
        st.image(original_image, caption="Imagen original", use_container_width=True)
    with img_col2:
        st.image(vis_image, caption="Segmentación YOLOv11-seg", use_container_width=True)

    # Leyenda de colores
    st.markdown(
        "**Leyenda:** "
        + " · ".join(
            [
                f'<span class="color-dot" style="background:{CLASS_COLORS_HEX[name]};"></span>'
                f"{name.replace('_', ' ')}"
                for name in CLASS_NAMES_SEG.values()
            ]
        ),
        unsafe_allow_html=True,
    )

    # Tabla de detecciones individuales
    if detections:
        with st.expander("📊 Ver tabla de detecciones individuales"):
            import pandas as pd

            rows = []
            for i, det in enumerate(detections):
                rows.append(
                    {
                        "#": i + 1,
                        "Clase": det["class_name"].replace("_", " "),
                        "Confianza": f"{det['confidence']:.3f}",
                        "Área (px²)": det["area_px"],
                        "BBox [x1,y1,x2,y2]": (
                            f"[{det['bbox'][0]:.0f}, {det['bbox'][1]:.0f}, "
                            f"{det['bbox'][2]:.0f}, {det['bbox'][3]:.0f}]"
                        ),
                    }
                )
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)


def render_morphology_tab() -> None:
    """Renderiza la ficha morfológica educativa de A. alternata."""

    st.markdown("## 📚 Ficha Morfológica — *Alternaria alternata*")
    st.markdown(MORPHOLOGY_INFO["descripcion_general"])

    st.divider()

    # Estructuras diagnósticas
    st.markdown("### Estructuras diagnósticas")
    cols = st.columns(3)
    for i, (nombre, info) in enumerate(MORPHOLOGY_INFO["estructuras"].items()):
        with cols[i]:
            st.markdown(
                f"""
                <div class="metric-card" style="text-align:left; padding:1.2rem;">
                    <div style="display:flex;align-items:center;margin-bottom:0.6rem;">
                        <span class="color-dot"
                              style="background:{info["color"]};
                                     width:18px;height:18px;"></span>
                        <strong style="font-size:1rem">{nombre}</strong>
                    </div>
                    <p style="font-size:0.88rem;color:#28251d;margin-bottom:0.4rem">
                        {info["descripcion"]}
                    </p>
                    <p style="font-size:0.82rem;color:#7a7974;margin-bottom:0.3rem">
                        📐 {info["dimensiones"]}
                    </p>
                    <p style="font-size:0.82rem;color:#437a22">
                        ✦ {info["relevancia"]}
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.divider()

    col_clin, col_lab = st.columns(2)

    with col_clin:
        st.markdown("### 🏥 Relevancia clínica")
        for item in MORPHOLOGY_INFO["relevancia_clinica"]:
            st.markdown(f"- {item}")

    with col_lab:
        st.markdown("### 🧫 Diagnóstico de laboratorio")
        for item in MORPHOLOGY_INFO["diagnostico_laboratorio"]:
            st.markdown(f"- {item}")

    st.divider()
    st.markdown("### 📖 Referencias bibliográficas")
    st.markdown("""
    - de Hoog, G. S., et al. (2019). *Atlas of Clinical Fungi* (4th ed.).
      Westerdijk Fungal Biodiversity Institute.
    - Larone, D. H. (2011). *Medically Important Fungi: A Guide to Identification*
      (5th ed.). ASM Press.
    - Murray, P. R., et al. (2020). *Manual of Clinical Microbiology* (12th ed.).
      ASM Press.
    - Revankar, S. G., & Sutton, D. A. (2010). Melanized fungi in human disease.
      *Clinical Microbiology Reviews, 23*(4), 884–928.
    """)


def render_about_tab() -> None:
    """Renderiza la pestaña Acerca de / Metodología."""

    st.markdown("## ℹ️ Acerca del Prototipo")

    st.markdown("""
    **AlternariaVision** es un prototipo de investigación desarrollado en el
    **Instituto de Estudios del Futuro** de la **Universidad de Boyacá**
    (Tunja, Colombia) para la identificación automatizada de
    *Alternaria alternata* mediante técnicas de visión artificial y
    deep learning.
    """)

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🔬 Fase 1 — Clasificación")
        st.markdown("""
        | Componente | Detalle |
        |---|---|
        | **Arquitectura** | EfficientNet-B2 (timm) |
        | **Parámetros** | 9.1 M |
        | **Entrada** | 288 × 288 px, RGB |
        | **Fine-tuning** | 2 fases (head → últimos 3 bloques) |
        | **Aumentaciones** | Albumentations + HueSaturation 360° |
        | **Pérdida** | CrossEntropy con class weights |
        | **Optimizador** | AdamW + CosineAnnealingLR |
        | **Explicabilidad** | Grad-CAM++ |
        """)

    with col2:
        st.markdown("### 🧩 Fase 2 — Segmentación")
        st.markdown("""
        | Componente | Detalle |
        |---|---|
        | **Arquitectura** | YOLOv11n-seg (Ultralytics) |
        | **Clases** | conidias, conidias_multiseptadas, hifas |
        | **Anotaciones** | X-AnyLabeling + SAM-base |
        | **Formato** | YOLO Segmentation (.txt) |
        | **Entrada** | 640 × 640 px |
        | **Latencia** | ~8 ms / imagen (T4) |
        | **Métricas** | mAP50, mAP50-95, IoU por clase |
        """)

    st.divider()

    st.markdown("### 🎯 Objetivos de aprendizaje")
    st.markdown("""
    Esta herramienta se alinea con los siguientes resultados de aprendizaje
    de la asignatura de **Micología** (Programa de Bacteriología):

    1. **Identificar** las estructuras morfológicas diagnósticas de
       *Alternaria alternata* en preparaciones directas.
    2. **Comparar** los hallazgos del modelo con la observación manual
       al microscopio.
    3. **Interpretar** los resultados del modelo con criterio clínico,
       reconociendo sus limitaciones.
    4. **Relacionar** la morfología fúngica con la patogenia de las
       micosis feohifomicóticas.
    """)

    st.divider()

    st.markdown("### ⚖️ Limitaciones")
    st.markdown("""
    - El modelo fue entrenado con un dataset limitado de microscopía óptica.
      El rendimiento puede variar con diferentes tinciones o aumentos.
    - La Fase 2 requiere al menos ~50 imágenes anotadas por clase para
      resultados confiables.
    - No es un dispositivo médico certificado. No debe usarse para
      diagnóstico clínico sin supervisión de un bacteriólogo.
    - Las predicciones de segmentación pueden fallar en imágenes con
      alta densidad de estructuras solapadas.
    """)

    st.markdown(
        """
    <div class="clinical-warning">
    ⚠️ <strong>Aviso legal:</strong> Este prototipo es exclusivamente
    una herramienta educativa e investigativa. No reemplaza el juicio
    clínico de un profesional de la salud.
    </div>
    """,
        unsafe_allow_html=True,
    )


# ── Función principal ─────────────────────────────────────────────────────────


def main() -> None:
    inject_css()
    render_header()

    conf_clf, conf_seg, iou_seg, show_gradcam = render_sidebar()

    # Pestañas principales
    tab_analysis, tab_morphology, tab_about = st.tabs(
        [
            "🔬 Análisis de muestra",
            "📚 Morfología",
            "ℹ️ Acerca de",
        ]
    )

    # ── PESTAÑA 1: ANÁLISIS ──────────────────────────────────────────
    with tab_analysis:
        st.markdown(
            "### <span class='step-badge'>1</span> Cargar imagen de microscopía",
            unsafe_allow_html=True,
        )

        uploaded = st.file_uploader(
            "Arrastra o selecciona una imagen (JPG, PNG, TIFF)",
            type=["jpg", "jpeg", "png", "tif", "tiff"],
            help=(
                "Se recomienda microscopía óptica con tinción de KOH, "
                "lactofenol azul o verde brillante. Aumento: 400× a 1000×."
            ),
        )

        if uploaded is None:
            # Estado vacío
            st.markdown(
                """
            <div style="
                text-align:center; padding:3rem;
                background:#f9f8f5; border-radius:0.75rem;
                border:2px dashed #dcd9d5; color:#7a7974;
            ">
                <div style="font-size:3rem">🧫</div>
                <div style="font-size:1.1rem;margin-top:0.5rem">
                    Carga una imagen de microscopía para comenzar el análisis
                </div>
                <div style="font-size:0.85rem;margin-top:0.4rem">
                    Formatos aceptados: JPG · PNG · TIFF · BMP
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

            st.divider()
            st.markdown("#### 💡 Instrucciones de uso")
            st.markdown("""
            1. **Carga** una imagen de microscopía de tu preparación.
            2. El sistema **clasifica** automáticamente si la muestra
               corresponde a *Alternaria alternata*.
            3. Si el clasificador detecta *Alternaria*, puedes ejecutar
               la **segmentación** para identificar estructuras individuales.
            4. Activa **Grad-CAM++** para visualizar las regiones que
               activaron la decisión del modelo.
            5. Consulta la pestaña **Morfología** para información
               educativa detallada.
            """)
            return

        # ── Imagen cargada ───────────────────────────────────────────
        if st.session_state.get("current_image_id") != uploaded.file_id:
            st.session_state["current_image_id"] = uploaded.file_id
            st.session_state["seg_done"] = False
            st.session_state["seg_detections"] = []
            st.session_state["seg_vis"] = None

        image_pil = Image.open(uploaded).convert("RGB")
        image_np = np.array(image_pil)

        st.success(
            f"✓ Imagen cargada: {uploaded.name} · {image_np.shape[1]}×{image_np.shape[0]} px",
            icon="📷",
        )

        # Vista previa
        preview_col, info_col = st.columns([3, 1])
        with preview_col:
            st.image(image_np, caption="Imagen cargada", use_container_width=True)
        with info_col:
            st.markdown("**Detalles de la imagen:**")
            st.markdown(f"- Archivo: `{uploaded.name}`")
            st.markdown(f"- Tamaño: {uploaded.size / 1024:.1f} KB")
            st.markdown(f"- Dimensiones: {image_np.shape[1]} × {image_np.shape[0]} px")
            st.markdown(f"- Canales: {'RGB' if image_np.ndim == 3 else 'Escala de grises'}")

        st.divider()

        # ── FASE 1: CLASIFICACIÓN ────────────────────────────────────
        st.markdown(
            "### <span class='step-badge'>2</span> Clasificación binaria",
            unsafe_allow_html=True,
        )

        clf_tuple = load_classifier()

        if clf_tuple is None:
            st.warning(
                "⚠️ El clasificador no está disponible. "
                "Entrena el modelo con `make train-clf` y reinicia la aplicación.",
                icon="⚠️",
            )
        else:
            with st.spinner("Ejecutando clasificación..."):
                t0 = time.perf_counter()
                clf_result = predict_classification(
                    image_np,
                    clf_tuple,
                    image_size=288,
                    conf_threshold=conf_clf,
                )
                t_clf = (time.perf_counter() - t0) * 1000

            st.markdown(
                f"<small style='color:#7a7974'>⏱ {t_clf:.1f} ms</small>", unsafe_allow_html=True
            )
            render_classification_result(clf_result, conf_clf)

            # ── GRAD-CAM ─────────────────────────────────────────────
            if show_gradcam:
                st.divider()
                st.markdown(
                    "### <span class='step-badge'>3</span> Explicabilidad — Grad-CAM++",
                    unsafe_allow_html=True,
                )
                with st.spinner("Generando mapa de activación Grad-CAM++..."):
                    gradcam_img = generate_gradcam(image_np, clf_tuple)

                if gradcam_img is not None:
                    gc_col1, gc_col2 = st.columns(2)
                    with gc_col1:
                        st.image(image_np, caption="Original", use_container_width=True)
                    with gc_col2:
                        st.image(
                            gradcam_img,
                            caption="Grad-CAM++ (regiones activadas)",
                            use_container_width=True,
                        )
                    st.info(
                        "💡 **Interpretación:** Las zonas rojas/cálidas indican "
                        "las regiones que más influyeron en la decisión del modelo. "
                        "Verifica que coincidan con estructuras morfológicas "
                        "diagnósticas (conidias, hifas) y no con artefactos.",
                        icon="🔍",
                    )
                else:
                    st.info(
                        "Grad-CAM++ no disponible. Instala: `uv add grad-cam`",
                        icon="ℹ️",
                    )

            # ── FASE 2: SEGMENTACIÓN ──────────────────────────────────
            st.divider()
            st.markdown(
                "### <span class='step-badge'>4</span> Segmentación de estructuras",
                unsafe_allow_html=True,
            )

            seg_model = load_segmenter()

            if seg_model is None:
                st.warning(
                    "⚠️ El segmentador no está disponible. Entrena el modelo con `make train-seg`.",
                    icon="⚠️",
                )
            else:
                run_seg = st.button(
                    "🧩 Ejecutar segmentación",
                    type="primary",
                    help=(
                        "Detecta y delimita conidias, conidias multiseptadas e hifas en la imagen."
                    ),
                )

                if run_seg or st.session_state.get("seg_done", False):
                    if run_seg:
                        with st.spinner("Segmentando estructuras fúngicas..."):
                            t0 = time.perf_counter()
                            detections, vis_image = predict_segmentation(
                                image_np,
                                seg_model,
                                conf=conf_seg,
                                iou=iou_seg,
                            )
                            t_seg = (time.perf_counter() - t0) * 1000
                        st.session_state["seg_detections"] = detections
                        st.session_state["seg_vis"] = vis_image
                        st.session_state["seg_done"] = True
                        st.markdown(
                            f"<small style='color:#7a7974'>⏱ {t_seg:.1f} ms · "
                            f"{len(detections)} estructuras detectadas</small>",
                            unsafe_allow_html=True,
                        )

                    render_segmentation_result(
                        st.session_state["seg_detections"],
                        st.session_state["seg_vis"],
                        image_np,
                    )

            # ── DESCARGA DE RESULTADOS ────────────────────────────────
            if clf_tuple is not None:
                st.divider()
                st.markdown(
                    "### <span class='step-badge'>5</span> Exportar resultados",
                    unsafe_allow_html=True,
                )

                report_lines = [
                    "# Reporte AlternariaVision",
                    f"## Imagen analizada: {uploaded.name}",
                    "",
                    "## Fase 1 — Clasificación",
                    f"- P(Alternaria alternata): {clf_result['prob_alternaria']:.4f}",
                    f"- P(Otros hongos): {1 - clf_result['prob_alternaria']:.4f}",
                    f"- Umbral aplicado: {conf_clf:.2f}",
                    f"- Resultado: {'Alternaria alternata' if clf_result['prob_alternaria'] >= conf_clf else 'Otro hongo'}",  # noqa: E501
                    "",
                ]

                if st.session_state.get("seg_done"):
                    dets = st.session_state["seg_detections"]
                    counts = {n: 0 for n in CLASS_NAMES_SEG.values()}
                    for d in dets:
                        counts[d["class_name"]] += 1
                    report_lines += [
                        "## Fase 2 — Segmentación",
                        f"- Conidias: {counts['conidias']}",
                        f"- Conidias multiseptadas: {counts['conidias_multiseptadas']}",
                        f"- Hifas: {counts['hifas']}",
                        f"- Total estructuras: {len(dets)}",
                        "",
                    ]

                report_lines.append(
                    "⚠️ Aviso: Este reporte es una herramienta educativa. "
                    "No reemplaza el diagnóstico microbiológico profesional."
                )

                report_txt = "\n".join(report_lines)
                st.download_button(
                    label="📥 Descargar reporte (.txt)",
                    data=report_txt.encode("utf-8"),
                    file_name=f"reporte_{uploaded.name.rsplit('.', 1)[0]}.txt",
                    mime="text/plain",
                )

                # Descarga imagen segmentada
                if st.session_state.get("seg_done") and st.session_state.get("seg_vis") is not None:
                    vis_pil = Image.fromarray(st.session_state["seg_vis"])
                    buf = io.BytesIO()
                    vis_pil.save(buf, format="PNG")
                    st.download_button(
                        label="📥 Descargar imagen segmentada (.png)",
                        data=buf.getvalue(),
                        file_name=f"segmentacion_{uploaded.name.rsplit('.', 1)[0]}.png",
                        mime="image/png",
                    )

    # ── PESTAÑA 2: MORFOLOGÍA ────────────────────────────────────────
    with tab_morphology:
        render_morphology_tab()

    # ── PESTAÑA 3: ACERCA DE ─────────────────────────────────────────
    with tab_about:
        render_about_tab()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
