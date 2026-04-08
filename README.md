# Prototipo de Visión Artificial para Identificación de *Alternaria alternata*

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Ultralytics](https://img.shields.io/badge/YOLOv11-seg-00BFFF?logo=ultralytics&logoColor=white)](https://github.com/ultralytics/ultralytics)
[![CI](https://github.com/TU_USUARIO/alternaria-vision/actions/workflows/ci.yml/badge.svg)](https://github.com/TU_USUARIO/alternaria-vision/actions)

---

## Resumen

Prototipo de investigación desarrollado en el **Instituto de Estudios del Futuro**
de la **Universidad de Boyacá** (Tunja, Colombia) para la clasificación y segmentación
de *Alternaria alternata* en imágenes de microscopía óptica.

La herramienta integra dos componentes de deep learning:

1. **Clasificación binaria** (Fase 1): determina si una preparación microscópica
   contiene *Alternaria alternata* o corresponde a otro hongo.
2. **Segmentación de instancias** (Fase 2): detecta y delimita las estructuras
   morfológicas diagnósticas — conidias, conidias multiseptadas e hifas.

Su propósito es **pedagógico**, orientado a estudiantes de Bacteriología
en la asignatura de Micología, y **científico**, documentando cada decisión
técnica con rigor investigativo para posible publicación.

---

## Tabla de Contenidos

- [Contexto micológico](#contexto-micológico)
- [Arquitectura del sistema](#arquitectura-del-sistema)
- [Requisitos](#requisitos)
- [Instalación](#instalación)
- [Estructura del repositorio](#estructura-del-repositorio)
- [Flujo de trabajo](#flujo-de-trabajo)
  - [Preparación de datos](#1-preparación-de-datos)
  - [Fase 1 — Clasificación](#2-fase-1--clasificación-binaria)
  - [Fase 2 — Segmentación](#3-fase-2--segmentación-de-estructuras)
  - [Interfaz web](#4-interfaz-web-streamlit)
- [Métricas y objetivos clínicos](#métricas-y-objetivos-clínicos)
- [Decisiones técnicas](#decisiones-técnicas)
- [Citación](#citación)
- [Licencia](#licencia)

---

## Contexto Micológico

### *Alternaria alternata* (Fr.) Keissl. 1912

*Alternaria alternata* es un hongo dematiáceo (hifas melanizadas) ubicuo,
clasificado en el orden **Pleosporales**, familia **Pleosporaceae**.

#### Características morfológicas diagnósticas

| Estructura | Descripción microscópica | Importancia diagnóstica |
|---|---|---|
| **Conidias** | Obclavadas a ovoides, muriformes (septa transversales Y longitudinales), pigmentación marrón media, extremo apical en "pico" (*beak*) | Estructura diferencial primaria |
| **Conidias multiseptadas** | Conidias con ≥3 septos transversales visibles; cadenas acrópetas | Confirma madurez y esporulación activa |
| **Hifas** | Septadas, pigmentadas (marrón oliváceo), ramificadas, 2–6 µm diámetro | Contexto estructural del crecimiento |

#### Dimensiones de referencia (cultivo SDA, 25°C)

- Conidias: 7–10 × 23–34 µm (rango reportado hasta 50 µm en hábitat natural)
- Conidióforos: 40–70 × 3–4 µm, septados, con aspecto en zigzag
- Temperatura óptima de crecimiento: 25–28°C

#### Relevancia clínica

*A. alternata* es agente etiológico de:
- **Feohifomicosis** cutánea y subcutánea en pacientes inmunocomprometidos
- **Sinusitis fúngica alérgica** (alérgeno Alt a 1)
- **Queratitis fúngica** (infección ocular post-traumática)
- Oportunista en trasplantados, pacientes con SIDA y neutropénicos

> ⚠️ **Nota clínica**: El diagnóstico definitivo requiere correlación con
> cultivo e historia clínica. Este prototipo es una herramienta educativa
> de apoyo, no un dispositivo diagnóstico certificado.

---

## Arquitectura del Sistema

```mermaid 
┌─────────────────────────────────────────────────────────────────┐
│ INTERFAZ WEB (Streamlit) │
│ Carga imagen → Clasificar → Segmentar → Educar │
└───────────────┬─────────────────────────┬───────────────────────┘
│ │
┌───────────▼──────────┐ ┌───────────▼──────────────┐
│ FASE 1 │ │ FASE 2 │
│ Clasificación │ │ Segmentación │
│ EfficientNet-B2 │ │ YOLOv11n-seg │
│ (binaria) │ │ (3 clases) │
│ │ │ │
│ Alternaria vs │ │ conidias │
│ Otros hongos │ │ conidias_multiseptadas │
│ │ │ hifas │
└───────────┬──────────┘ └───────────┬───────────────┘
│ │
┌───────────▼─────────────────────────▼───────────────┐
│ DATASET DE MICROSCOPÍA │
│ data/processed/classification/ (Fase 1) │
│ data/processed/segmentation/ (Fase 2) │
│ Anotaciones: X-AnyLabeling JSON (SAM-base) │
└─────────────────────────────────────────────────────┘
```
---

## Requisitos

### Hardware mínimo

| Componente | Mínimo | Recomendado |
|---|---|---|
| GPU | — (CPU posible) | NVIDIA ≥4 GB VRAM |
| RAM | 8 GB | 16 GB |
| Almacenamiento | 5 GB | 20 GB |

> **Google Colab**: T4 (16 GB VRAM, gratuito) es suficiente para ambas fases.
> Recomendado: Colab Pro con A100 para datasets > 500 imágenes.

### Software

- Python 3.11
- [uv](https://github.com/astral-sh/uv) ≥ 0.4.0 (gestor de entorno)
- Git ≥ 2.40

---

## Instalación

### 1. Clonar el repositorio

```bash
git clone ttps://github.com/ief-udb/alternaria-vision.git
cd alternaria-vision
```

### 2. Instalar uv (si no está instalado)

```bash
# Linux / macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 3. Crear entorno virtual e instalar dependencias

```bash
# Instalar Python 3.11 y dependencias de producción
uv sync

# Incluir herramientas de desarrollo (pytest, ruff, black, pre-commit)
uv sync --extra dev

# Activar pre-commit hooks
uv run pre-commit install
```

### 4. Verificar la instalación

```bash
uv run pytest tests/ -v
```

Todos los tests deben pasar sin requerir GPU ni datos reales.

---

## Estructura del Repositorio

```mermaid
alternaria-vision/
│
├── .github/workflows/ci.yml # CI: lint + tests automáticos
├── configs/
│ ├── train_clf.yaml # Hiperparámetros Fase 1
│ └── train_seg.yaml # Hiperparámetros Fase 2
│
├── data/
│ ├── raw/ # Imágenes originales (.gitignore)
│ ├── processed/
│ │ ├── classification/ # Estructura binaria para Fase 1
│ │ │ ├── alternaria/
│ │ │ └── otros_hongos/
│ │ └── segmentation/ # Dataset YOLO para Fase 2
│ │ ├── images/{train,val,test}/
│ │ ├── labels/{train,val,test}/
│ │ └── data.yaml
│ └── splits/ # train.txt, val.txt, test.txt
│
├── src/
│ ├── data/
│ │ ├── augmentations.py # Pipelines Albumentations
│ │ └── dataset.py # MicroscopyDataset (PyTorch)
│ ├── models/
│ │ ├── classifier.py # AlternariaCLF (EfficientNet-B2)
│ │ └── segmenter.py # AlternariaSEG (YOLOv11-seg)
│ ├── training/
│ │ ├── train_clf.py # Entrenamiento Fase 1
│ │ └── train_seg.py # Entrenamiento Fase 2
│ ├── evaluation/
│ │ ├── metrics.py # Métricas clínicas + visualizaciones
│ │ └── evaluate.py # CLI de evaluación standalone
│ └── utils/
│ ├── device.py # Detección automática GPU/MPS/CPU
│ ├── logger.py # Logger Rich centralizado
│ ├── prepare_data.py # Organización de datos crudos
│ └── converter.py # X-AnyLabeling JSON → YOLO .txt
│
├── app/
│ └── app.py # Interfaz web Streamlit
│
├── tests/
│ ├── test_dataset.py
│ └── test_model.py
│
├── notebooks/
│ ├── 01_EDA_microscopia.ipynb
│ ├── 02_augmentation_preview.ipynb
│ └── 03_metricas_analisis.ipynb
│
├── checkpoints/
│ ├── classification/ # best_model.pt, last_model.pt
│ └── segmentation/ # best_model.pt
│
├── outputs/ # Métricas, curvas, Grad-CAM
├── logs/ # Archivos de log de entrenamiento
├── pyproject.toml # Dependencias y scripts (uv)
├── Makefile # Comandos de uso frecuente
└── README.md

```
--


---

## Flujo de Trabajo

### 1. Preparación de Datos

#### Clasificación binaria (Fase 1)

Organiza tus imágenes en subcarpetas por especie dentro de `data/raw/`:

```
data/raw/
alternaria/ ← Imágenes de Alternaria alternata
aspergillus_niger/ ← Otros hongos
fusarium_solani/
...


Luego ejecuta:

```bash
make prepare-data
# Equivale a:
# uv run prepare-data --src-dir data/raw/ --dest-dir data/processed/classification/
```

Esto genera `data/processed/classification/inventario_dataset.csv`
con el recuento de imágenes por clase.

#### Segmentación (Fase 2)

1. Anota tus imágenes en **X-AnyLabeling** usando el modelo SAM-base.
   Clases a definir: `conidias`, `conidias_multiseptadas`, `hifas`.
2. Exporta las anotaciones como JSON (una por imagen).
3. Convierte al formato YOLO:

```bash
make convert-ann
# Equivale a:
# uv run convert-ann data/raw/annotations/ \
#     data/processed/segmentation/labels/train/ \
#     --images-dir data/raw/images/ \
#     --images-out data/processed/segmentation/images/train/
```

Esto genera automáticamente `data/processed/segmentation/data.yaml`.

---

### 2. Fase 1 — Clasificación Binaria

#### Entrenamiento

```bash
# Modelo principal: EfficientNet-B2
make train-clf

# Modelo comparativo: ConvNeXt-Tiny
make train-clf-cmp

# Reanudar entrenamiento interrumpido
uv run train-clf --resume checkpoints/classification/last_model.pt
```

El entrenamiento ejecuta automáticamente:

- **Fase A** (5 épocas): Solo la cabeza clasificadora con backbone congelado.
- **Fase B** (≤30 épocas): Fine-tuning de los últimos 3 bloques del backbone
  con CosineAnnealingLR y early stopping (patience=10).

Los resultados se guardan en `outputs/alternaria-clf/`:
- `confusion_matrix.png`
- `roc_curve.png`
- `pr_curve.png`
- `training_history.png`
- `gradcam/` (20 muestras Grad-CAM++)
- `metrics_log.csv`

#### Evaluación independiente

```bash
uv run evaluate \
    --checkpoint checkpoints/classification/best_model.pt \
    --data-dir data/processed/classification/ \
    --tta \
    --name efficientnet_b2_tta
```

---

### 3. Fase 2 — Segmentación de Estructuras

```bash
# Entrenamiento completo
make train-seg

# Reanudar entrenamiento interrumpido
uv run train-seg --resume

# Solo evaluación (sin reentrenar)
uv run train-seg --eval-only \
    --checkpoint outputs/segmentation/alternaria_seg/weights/best.pt
```

Los resultados se guardan en `outputs/segmentation/alternaria_seg/`:
- Curvas mAP50, Precision-Recall por clase
- Matriz de confusión por clase
- Imágenes de validación con máscaras superpuestas

---

### 4. Interfaz Web Streamlit

```bash
make app
# Equivale a: uv run streamlit run app/app.py
```

La interfaz estará disponible en `http://localhost:8501`.

#### En Google Colab

```python
# Instalar dependencias
!pip install -q uv
!uv sync

# Lanzar con túnel público
!uv run streamlit run app/app.py &
!npx localtunnel --port 8501
```

---

## Métricas y Objetivos Clínicos

### Fase 1 — Clasificación

El modelo debe cumplir los siguientes criterios mínimos para uso pedagógico:

| Métrica | Objetivo mínimo | Justificación |
|---|---|---|
| **Sensitivity** | ≥ 0.85 | No perder casos reales de *A. alternata* |
| **Specificity** | ≥ 0.80 | Evitar confusión con otros dematiáceos |
| **AUC-ROC** | ≥ 0.90 | Discriminación global robusta |
| **F1-Score** | ≥ 0.83 | Balance precision/recall |

El umbral óptimo se determina por el **índice de Youden J**
(maximiza Sensitivity + Specificity − 1) y puede diferir del default 0.5.

### Fase 2 — Segmentación

| Métrica | Objetivo mínimo |
|---|---|
| **mAP50-seg** (global) | ≥ 0.70 |
| **mAP50-seg** (conidias) | ≥ 0.75 |
| **mAP50-seg** (hifas) | ≥ 0.65 |
| **mAP50-seg** (conidias_multiseptadas) | ≥ 0.60 |

> Los objetivos de conidias_multiseptadas son más flexibles dada
> la menor frecuencia de esta clase en el dataset inicial.

---

## Decisiones Técnicas

### ¿Por qué EfficientNet-B2 y no ViT o VGG16?

| Criterio | VGG16 | ViT-B/16 | **EfficientNet-B2** |
|---|---|---|---|
| Params | 138 M | 86 M | **9.1 M** |
| VRAM mín. | 6 GB | 8 GB | **2 GB** |
| Datasets pequeños (<500 imgs) | Regular | ⚠️ Sobreajuste | ✅ Robusto |
| Transfer learning | Bueno | Bueno | ✅ Excelente |
| Explicabilidad (Grad-CAM) | Básica | Attention maps | ✅ Grad-CAM++ |

EfficientNet-B2 ofrece el mejor balance entre rendimiento y eficiencia
computacional para datasets de microscopía médica con < 500 imágenes por clase.

### ¿Por qué YOLOv11n-seg y no Mask R-CNN o U-Net?

- **Mask R-CNN**: requiere > 200 imágenes anotadas y GPU ≥ 6 GB.
  Latencia de inferencia ~80 ms (inaceptable para uso interactivo).
- **U-Net**: excelente para segmentación semántica pero no detecta
  instancias individuales (no puede contar conidias separadas).
- **YOLOv11n-seg**: detección + segmentación en una sola pasada,
  ~8 ms por imagen, funciona desde ~50 imágenes anotadas.
  Ideal para datasets pequeños y uso educativo en tiempo real.

### ¿Por qué Streamlit y no Gradio o React?

Streamlit fue seleccionado sobre las alternativas por:
- **Integración nativa** con PyTorch/NumPy sin API REST intermedia.
- **`st.session_state`**: gestión de estado multi-paso para el flujo
  educativo (carga → predicción → segmentación → información).
- **Visualizaciones científicas** con Plotly integrado.
- **Despliegue gratuito** en Streamlit Community Cloud.
- Tiempo de desarrollo estimado: 3–5 días vs. 2–4 semanas (React).

### ¿Por qué Albumentations para aumentación y no torchvision?

- `HueSaturationValue(hue_shift_limit=180)`: rotación completa del
  espectro de color, crítica para invarianza al medio de contraste
  (azul de lactofenol, verde brillante, KOH amarillo).
- `ChannelShuffle`: agnosticismo al color mediante permutación RGB.
- `ToGray`: fuerza al modelo a aprender morfología pura sin color.
- `CLAHE`: realce de contraste local adaptativo, superior a
  `RandomContrast` para microscopia con iluminación no uniforme.
- Velocidad: Albumentations es 3–10× más rápido que torchvision
  para pipelines complejos (Buslaev et al., 2020).

---

## Citación

Si utilizas este trabajo en publicaciones académicas, por favor cita:

```bibtex
@software{alternaria_vision_2026,
  author    = {{Instituto de Estudios del Futuro, Universidad de Boyacá}},
  title     = {Prototipo de Visión Artificial para Identificación de
               \textit{Alternaria alternata} mediante Deep Learning},
  year      = {2026},
  url       = {https://github.com/TU_USUARIO/alternaria-vision},
  version   = {1.0.0},
  note      = {Herramienta pedagógica para la asignatura de Micología,
               Programa de Bacteriología}
}
```

### Referencias bibliográficas clave

- Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking model scaling for
  convolutional neural networks. *ICML 2019*.
- Liu, Z., et al. (2022). A ConvNet for the 2020s. *CVPR 2022*.
- Jocher, G., et al. (2023). *Ultralytics YOLOv8/v11*. GitHub.
- Buslaev, A., et al. (2020). Albumentations: Fast and flexible image
  augmentations. *Information, 11*(2), 125.
- Selvaraju, R. R., et al. (2017). Grad-CAM: Visual explanations from deep
  networks via gradient-based localization. *ICCV 2017*.
- Youden, W. J. (1950). Index for rating diagnostic tests.
  *Cancer, 3*(1), 32–35.
- de Hoog, G. S., et al. (2019). *Atlas of Clinical Fungi* (4th ed.).
  Westerdijk Fungal Biodiversity Institute.

---

## Contacto

**Instituto de Estudios del Futuro** | 
Universidad de Boyacá — Tunja, Colombia

> *"La inteligencia artificial al servicio del diagnóstico micológico
> y la formación de bacteriólogos en Colombia."*
