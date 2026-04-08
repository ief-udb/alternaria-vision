# Alternaria Vision

**Prototipo de visión artificial para clasificación y segmentación de *Alternaria alternata* en microscopía**

*Computer vision prototype for classification and segmentation of *Alternaria alternata* in microscopy*

---

> **Instituto de Estudios del Futuro — Universidad de Boyacá**
---

## 🇪🇸 Español

### Descripción

Alternaria Vision es un sistema de visión artificial de dos fases diseñado para el análisis automatizado de imágenes microscópicas de hongos:

| Fase | Tarea | Modelo | Detalle |
|------|-------|--------|---------|
| **Fase 1** | Clasificación binaria | EfficientNet-B2 (+ comparativo ConvNeXt-Tiny) | *Alternaria alternata* vs. otros hongos |
| **Fase 2** | Segmentación de instancias | YOLOv11n-seg | Conidias · Conidias multiseptadas · Hifas |

### Características principales

- **Fine-tuning en dos etapas** (backbone congelado → descongelamiento progresivo)
- **Augmentación avanzada** con Albumentations (rotaciones, color jitter, ruido gaussiano, etc.)
- **Entrenamiento con precisión mixta** (AMP) y gradient clipping
- **Interpretabilidad** mediante Grad-CAM
- **Exportación a ONNX** para inferencia en producción
- **Aplicación interactiva** con Streamlit
- **CI/CD** con GitHub Actions (lint, format, tests)

### Requisitos

- Python ≥ 3.11
- [uv](https://docs.astral.sh/uv/) (gestor de paquetes)

### Instalación

```bash
# Clonar el repositorio
git clone https://github.com/ief-udb/alternaria-vision.git
cd alternaria-vision

# Instalar dependencias de producción
make setup

# Instalar dependencias de desarrollo (incluye jupyter, pytest, ruff, black, pre-commit)
make setup-dev
```

### Uso

```bash
# Entrenar clasificador (EfficientNet-B2)
make train-clf

# Entrenar clasificador comparativo (ConvNeXt-Tiny)
make train-clf-cmp

# Entrenar segmentador (YOLOv11n-seg)
make train-seg

# Convertir anotaciones X-AnyLabeling → formato YOLO
make convert-ann

# Lanzar aplicación Streamlit
make app

# Lint y formato
make lint
make format

# Ejecutar tests
make test
```

### Estructura del proyecto

```
alternaria-vision/
├── app/                    # Aplicación Streamlit
│   └── assets/             # Recursos estáticos
├── configs/                # Archivos de configuración YAML
│   ├── train_clf.yaml      # Config clasificación
│   └── train_seg.yaml      # Config segmentación
├── data/
│   ├── raw/                # Imágenes y anotaciones originales
│   ├── processed/          # Datos preprocesados
│   └── splits/             # Particiones train/val/test
├── notebooks/              # Notebooks de exploración y análisis
├── src/                    # Código fuente principal
│   ├── data/               # Datasets y data loaders
│   ├── evaluation/         # Métricas y evaluación
│   ├── models/             # Arquitecturas de modelos
│   ├── training/           # Entrenamiento y fine-tuning
│   └── utils/              # Utilidades (conversión de anotaciones, etc.)
├── tests/                  # Tests unitarios
├── .github/workflows/      # CI con GitHub Actions
├── configs/                # Configuraciones de entrenamiento
├── makefile                # Comandos automatizados
└── pyproject.toml          # Dependencias y metadatos del proyecto
```

### Métricas reportadas

**Clasificación:** Accuracy · Sensitivity · Specificity · Precision · F1-Score · AUC-ROC

**Segmentación:** mAP@50 · mAP@50:95 · Precision · Recall · IoU por clase

---

## 🇬🇧 English

### Description

Alternaria Vision is a two-phase computer vision system designed for automated analysis of microscopic fungal images:

| Phase | Task | Model | Detail |
|-------|------|-------|--------|
| **Phase 1** | Binary classification | EfficientNet-B2 (+ ConvNeXt-Tiny benchmark) | *Alternaria alternata* vs. other fungi |
| **Phase 2** | Instance segmentation | YOLOv11n-seg | Conidia · Multiseptate conidia · Hyphae |

### Key Features

- **Two-stage fine-tuning** (frozen backbone → progressive unfreezing)
- **Advanced augmentation** with Albumentations (rotations, color jitter, Gaussian noise, etc.)
- **Mixed precision training** (AMP) with gradient clipping
- **Interpretability** via Grad-CAM
- **ONNX export** for production inference
- **Interactive application** with Streamlit
- **CI/CD** with GitHub Actions (lint, format, tests)

### Requirements

- Python ≥ 3.11
- [uv](https://docs.astral.sh/uv/) (package manager)

### Installation

```bash
# Clone the repository
git clone https://github.com/ief-udb/alternaria-vision.git
cd alternaria-vision

# Install production dependencies
make setup

# Install development dependencies (includes jupyter, pytest, ruff, black, pre-commit)
make setup-dev
```

### Usage

```bash
# Train classifier (EfficientNet-B2)
make train-clf

# Train benchmark classifier (ConvNeXt-Tiny)
make train-clf-cmp

# Train segmentation model (YOLOv11n-seg)
make train-seg

# Convert X-AnyLabeling annotations → YOLO format
make convert-ann

# Launch Streamlit application
make app

# Lint and format
make lint
make format

# Run tests
make test
```

### Project Structure

```
alternaria-vision/
├── app/                    # Streamlit application
│   └── assets/             # Static assets
├── configs/                # YAML configuration files
│   ├── train_clf.yaml      # Classification config
│   └── train_seg.yaml      # Segmentation config
├── data/
│   ├── raw/                # Raw images and annotations
│   ├── processed/          # Preprocessed data
│   └── splits/             # Train/val/test splits
├── notebooks/              # Exploration and analysis notebooks
├── src/                    # Main source code
│   ├── data/               # Datasets and data loaders
│   ├── evaluation/         # Metrics and evaluation
│   ├── models/             # Model architectures
│   ├── training/           # Training and fine-tuning
│   └── utils/              # Utilities (annotation conversion, etc.)
├── tests/                  # Unit tests
├── .github/workflows/      # CI with GitHub Actions
├── makefile                # Automated commands
└── pyproject.toml          # Dependencies and project metadata
```

### Reported Metrics

**Classification:** Accuracy · Sensitivity · Specificity · Precision · F1-Score · AUC-ROC

**Segmentation:** mAP@50 · mAP@50:95 · Precision · Recall · IoU per class

---

## Licencia / License

Este proyecto no cuenta con una licencia de código abierto. Todos los derechos reservados.

*This project does not have an open-source license. All rights reserved.* 
