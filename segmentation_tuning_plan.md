# Mejorar modelo de segmentación YOLOv11 — Desbalanceo de clases

## Diagnóstico del problema

El dataset de segmentación presenta un **desbalanceo severo** entre las 3 clases:

| Clase | Instancias | Porcentaje |
|---|---|---|
| conidia | 352 | 14.5% |
| **conidia-multiseptada** | **1977** | **81.4%** |
| hifa | 99 | **4.1%** |

Con 58 imágenes anotadas y este desbalanceo, el modelo converge rápidamente a predecir solo la clase dominante (`conidia-multiseptada`), ignorando `conidia` y `hifa`.

> [!IMPORTANT]
> La raíz del problema es que `hifa` tiene **20× menos** instancias que `conidia-multiseptada`. Ningún hiperparámetro por sí solo resolverá esto; la solución requiere una **combinación de estrategias** de aumentación, pérdida ponderada y ajuste de hiperparámetros.

## Estrategia propuesta

Crear un sistema de **experimentos automatizados** con múltiples combinaciones de configuraciones, ejecutables secuencialmente. Cada experimento será un archivo YAML independiente y un script runner que los ejecuta en secuencia, comparando resultados.

## Proposed Changes

### Configuraciones de experimentos

Se crearán **5 configuraciones** progresivas, cada una atacando el problema desde un ángulo distinto:

---

#### [NEW] [experiment_1_baseline_adamw.yaml](file:///c:/Users/Sebastian/Documents/GitHub/alternaria-vision/configs/experiments/experiment_1_baseline_adamw.yaml)

**Experimento 1 — Baseline mejorado con AdamW + Cosine LR**
- Cambia de SGD (default) a **AdamW** para mejor generalización con datasets pequeños
- Activa **cosine learning rate scheduler** (`cos_lr=True`)
- Reduce learning rate inicial a `5e-4` (más conservador)
- Aumenta `warmup_epochs` a 5 para estabilizar primeras épocas
- `close_mosaic=15` para estabilizar últimas épocas

---

#### [NEW] [experiment_2_heavy_augmentation.yaml](file:///c:/Users/Sebastian/Documents/GitHub/alternaria-vision/configs/experiments/experiment_2_heavy_augmentation.yaml)

**Experimento 2 — Aumentación agresiva para clases minoritarias**
- Mantiene AdamW + cosine LR
- Sube `copy_paste=0.8` (duplica instancias de todas las clases, beneficia minoritarias)
- Activa `mixup=0.15`
- Mayor `mosaic=0.8` para combinar más imágenes y exponer la red a más variedad
- `scale=0.5` para multi-escala agresivo
- `translate=0.2` para desplazamientos

---

#### [NEW] [experiment_3_larger_model.yaml](file:///c:/Users/Sebastian/Documents/GitHub/alternaria-vision/configs/experiments/experiment_3_larger_model.yaml)

**Experimento 3 — Modelo más grande (yolo11s-seg) + aumentación moderada**
- Usa `yolo11s-seg` en lugar de `yolo11n-seg` (más capacidad, ~3× parámetros)
- Aumentación moderada
- `batch_size=8` para compensar mayor uso de VRAM
- `imgsz=640`

---

#### [NEW] [experiment_4_high_resolution.yaml](file:///c:/Users/Sebastian/Documents/GitHub/alternaria-vision/configs/experiments/experiment_4_high_resolution.yaml)

**Experimento 4 — Alta resolución + modelo nano**
- Sube `imgsz=1024` (las imágenes originales son ~1280×960, más detalle preservado)
- `batch_size=4` por limitación de VRAM
- Mantiene yolo11n-seg pero con más resolución para capturar mejor las hifas (más finas y difíciles)
- Aumentación moderada

---

#### [NEW] [experiment_5_best_combo.yaml](file:///c:/Users/Sebastian/Documents/GitHub/alternaria-vision/configs/experiments/experiment_5_best_combo.yaml)

**Experimento 5 — Combinación agresiva (mejor de todos)**
- `yolo11s-seg` (modelo más grande)
- `imgsz=896` (resolución alta pero aún manejable)
- `copy_paste=0.9` (máximo beneficio para clases raras)
- `mixup=0.2`
- `mosaic=1.0` (siempre activo hasta 15 épocas antes del fin)
- AdamW + cosine LR
- `patience=30` (más paciencia antes de early stopping)
- `epochs=300`
- `batch_size=4`

---

### Script de ejecución

#### [NEW] [run_experiments.py](file:///c:/Users/Sebastian/Documents/GitHub/alternaria-vision/src/training/run_experiments.py)

Script que:
1. Lee todos los YAML de `configs/experiments/`
2. Ejecuta cada experimento secuencialmente
3. Registra métricas (mAP50 global y **por clase**) en un CSV de comparación
4. Genera un resumen de cuál configuración funcionó mejor

---

### Modificaciones al modelo

#### [MODIFY] [segmenter.py](file:///c:/Users/Sebastian/Documents/GitHub/alternaria-vision/src/models/segmenter.py)

- Ampliar el método `train()` para aceptar los nuevos parámetros:
  - `optimizer`, `cos_lr`, `close_mosaic`, `warmup_epochs`
  - `mixup`, `scale`, `translate`
  - `auto_augment`
- Mantener compatibilidad con la configuración actual

---

### Comando makefile

#### [MODIFY] [makefile](file:///c:/Users/Sebastian/Documents/GitHub/alternaria-vision/makefile)

Agregar nuevo target `run-experiments` para facilitar la ejecución.

## User Review Required

> [!IMPORTANT]
> **¿Tienes GPU disponible para ejecutar los experimentos?** Los experimentos 3 y 5 usan `yolo11s-seg` que requiere ~3-4 GB de VRAM. Si no tienes GPU, puedo ajustar todas las configuraciones para CPU (serán más lentos pero funcionarán).

> [!WARNING]
> Los 5 experimentos completos pueden tomar entre **2-6 horas** dependiendo del hardware (GPU vs CPU). ¿Deseas ejecutarlos todos o prefieres empezar con uno o dos?

## Verification Plan

### Automated Tests
- Validar que cada YAML se carga correctamente
- Ejecutar el script de experimentos y verificar que genera resultados

### Manual Verification  
- Comparar mAP50 **por clase** entre experimentos
- El objetivo principal es que `conidia` y `hifa` obtengan mAP50 > 0.0 (mejora desde el baseline que solo detecta `conidia-multiseptada`)
- Objetivo aspiracional: `hifa` ≥ 0.40, `conidia` ≥ 0.50

## Fase 2: Estrategia Data-Centric

Tras evaluar que las técnicas _model-centric_ tocaban techo técnico (~0.11 mAP50 en `conidia` y ~0.04 en `hifa`), se procedió a crear un nuevo paradigma compensando el déficit estadístico de las imágenes de microscopía.

### [NEW] [advanced_augment.py](file:///c:/Users/Sebastian/Documents/GitHub/alternaria-vision/src/data/advanced_augment.py)
Script offline que genera el nuevo entorno `data/processed/segmentation_v2/` mediante dos vías:
- **Tiling:** Recortes traslapados de 640x640 de las fotos HQ originales para ampliar la resolución y cantidad general de imágenes.
- **Oversampling Físico:** Rotaciones y flips aplicados hasta **12 veces** para `hifa` y **5 veces** para `conidia` sobre los tiles, compensando artificialmente el desbalance masivo.

### [NEW] [experiment_6_data_centric.yaml](file:///c:/Users/Sebastian/Documents/GitHub/alternaria-vision/configs/experiments/experiment_6_data_centric.yaml)
- Apunta a `data_v2.yaml`.
- Retoma el modelo masivo `yolo11s-seg` y el tamaño eficiente de `imgsz=640` apalancándose en la inmensa cantidad y variedad generada.
- El objetivo principal es que `conidia` y `hifa` obtengan mAP50 > 0.0 (mejora desde el baseline que solo detecta `conidia-multiseptada`)
- Objetivo aspiracional: `hifa` ≥ 0.40, `conidia` ≥ 0.50
