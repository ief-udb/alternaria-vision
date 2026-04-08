"""
train_clf.py
------------
Script principal de entrenamiento — Fase 1: Clasificación Binaria.

Flujo completo:
  1. Cargar configuración desde configs/train_clf.yaml
  2. Crear splits estratificados (train / val / test) y persistirlos
  3. Fase A: Entrenar solo la cabeza clasificadora (backbone congelado)
  4. Fase B: Fine-tuning parcial de los últimos N bloques del backbone
  5. Evaluación final en test set con métricas completas + Grad-CAM
  6. Guardar checkpoints, curvas de entrenamiento y CSV de métricas

Uso:
    # Entrenamiento estándar
    uv run train-clf --config configs/train_clf.yaml

    # Modelo comparativo
    uv run train-clf --config configs/train_clf.yaml --model convnext_tiny

    # Reanudar entrenamiento interrumpido
    uv run train-clf --config configs/train_clf.yaml \
        --resume checkpoints/classification/last_model.pt
"""

from __future__ import annotations

import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import typer
import yaml
from sklearn.metrics import f1_score as sk_f1
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.augmentations import get_train_transforms, get_val_transforms
from src.data.dataset import MicroscopyDataset
from src.evaluation.metrics import (
    compute_metrics,
    plot_confusion_matrix,
    plot_precision_recall_curve,
    plot_roc_curve,
    plot_training_history,
    save_metrics_csv,
)
from src.models.classifier import AlternariaCLF
from src.utils.device import get_batch_size, get_device
from src.utils.logger import get_logger

app = typer.Typer(help="Entrena el clasificador binario Alternaria vs Otros Hongos.")
logger = get_logger(__name__, log_file=Path("logs/train_clf.log"))


# ── Reproducibilidad ─────────────────────────────────────────────────────────


def set_seed(seed: int) -> None:
    """Fija semilla global para reproducibilidad del experimento."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ── Splits y DataLoaders ──────────────────────────────────────────────────────


def build_dataloaders(
    cfg: dict,
    device: torch.device,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    """
    Crea los tres DataLoaders con splits estratificados.

    La estratificación garantiza que la proporción de clases
    se preserve en train, val y test, crítico con datasets pequeños.
    """
    root = Path(cfg["data"]["root"])
    img_size = cfg["data"]["image_size"]
    split = cfg["data"]["split"]
    seed = cfg["project"]["seed"]

    # Dataset completo para generar índices
    full_ds = MicroscopyDataset(root=root, transform=None)
    labels = full_ds.get_labels()
    indices = list(range(len(full_ds)))

    # Split estratificado train / temp
    train_idx, temp_idx = train_test_split(
        indices,
        test_size=1.0 - split["train"],
        stratify=labels,
        random_state=seed,
    )

    # Split estratificado val / test desde temp
    temp_labels = [labels[i] for i in temp_idx]
    val_ratio = split["val"] / (split["val"] + split["test"])
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=1.0 - val_ratio,
        stratify=temp_labels,
        random_state=seed,
    )

    # Persistir splits para reproducibilidad y evaluación independiente
    splits_dir = Path("data/splits")
    splits_dir.mkdir(parents=True, exist_ok=True)
    for name, idx_list in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        with open(splits_dir / f"{name}.txt", "w") as f:
            for i in idx_list:
                p, _ = full_ds.samples[i]
                f.write(str(p.relative_to(root)) + "\n")

    logger.info(
        f"Splits estratificados | "
        f"train={len(train_idx)} | val={len(val_idx)} | test={len(test_idx)}"
    )

    # DataSets con transformaciones específicas
    train_ds = MicroscopyDataset(root, get_train_transforms(img_size), splits_dir / "train.txt")
    val_ds = MicroscopyDataset(root, get_val_transforms(img_size), splits_dir / "val.txt")
    test_ds = MicroscopyDataset(root, get_val_transforms(img_size), splits_dir / "test.txt")

    batch = get_batch_size(
        cfg["model"]["architecture"],
        device,
        override=cfg["training"].get("batch_size"),
    )
    nw = cfg["training"]["num_workers"]
    pin = cfg["training"]["pin_memory"] and device.type == "cuda"

    kw = dict(num_workers=nw, pin_memory=pin)
    train_dl = DataLoader(train_ds, batch_size=batch, shuffle=True, **kw)
    val_dl = DataLoader(val_ds, batch_size=batch, shuffle=False, **kw)
    test_dl = DataLoader(test_ds, batch_size=batch, shuffle=False, **kw)

    return train_dl, val_dl, test_dl, full_ds.classes


# ── Epoch functions ───────────────────────────────────────────────────────────


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler,
) -> tuple[float, float]:
    """
    Ejecuta una época de entrenamiento.

    Usa AMP (Automatic Mixed Precision) si scaler no es None,
    lo que reduce el uso de VRAM y acelera el entrenamiento en GPUs
    compatibles con float16 (T4, A100, RTX).

    Returns
    -------
    tuple[float, float]
        (loss_promedio, accuracy)
    """
    model.train()
    total_loss, correct, n = 0.0, 0, 0

    for imgs, lbls in tqdm(loader, desc="  train", leave=False):
        imgs, lbls = imgs.to(device), lbls.to(device)
        optimizer.zero_grad()

        if scaler is not None and device.type == "cuda":
            with torch.amp.autocast("cuda"):
                out = model(imgs)
                loss = criterion(out, lbls)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(imgs)
            loss = criterion(out, lbls)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        correct += (out.argmax(1) == lbls).sum().item()
        n += lbls.size(0)

    return total_loss / n, correct / n


@torch.no_grad()
def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Ejecuta una época de evaluación (validación o test).

    Returns
    -------
    tuple[float, float, ndarray, ndarray, ndarray]
        (loss, accuracy, y_true, y_pred, y_prob)
    """
    model.eval()
    total_loss = 0.0
    all_lbl: list = []
    all_pred: list = []
    all_prob: list = []
    n = 0

    for imgs, lbls in tqdm(loader, desc="  eval ", leave=False):
        imgs, lbls = imgs.to(device), lbls.to(device)
        out = model(imgs)
        loss = criterion(out, lbls)

        probs = torch.softmax(out, dim=1)[:, 1]
        total_loss += loss.item() * imgs.size(0)
        all_lbl.extend(lbls.cpu().numpy())
        all_pred.extend(out.argmax(1).cpu().numpy())
        all_prob.extend(probs.cpu().numpy())
        n += lbls.size(0)

    y_true = np.array(all_lbl)
    y_pred = np.array(all_pred)
    y_prob = np.array(all_prob)
    acc = float((y_pred == y_true).mean())

    return total_loss / n, acc, y_true, y_pred, y_prob


# ── Grad-CAM ──────────────────────────────────────────────────────────────────


def generate_gradcam_samples(
    model: AlternariaCLF,
    dataset: MicroscopyDataset,
    device: torch.device,
    output_dir: Path,
    n_samples: int = 20,
) -> None:
    """
    Genera imágenes Grad-CAM++ para las primeras n_samples del test set.

    Los mapas de calor permiten verificar que el modelo activa sobre
    las estructuras morfológicas diagnósticas (conidias muriformes,
    hifas septadas) y no sobre artefactos del preparado.

    Compatible con pytorch-grad-cam >= 1.5.0.
    """
    try:
        import cv2
        from pytorch_grad_cam import GradCAMPlusPlus
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    except ImportError:
        logger.warning(
            "pytorch-grad-cam o opencv no instalados. Saltando Grad-CAM.\n"
            "Instala con: uv add grad-cam opencv-python"
        )
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    target_layer = [model.get_gradcam_target_layer()]

    cam = GradCAMPlusPlus(model=model, target_layers=target_layer)
    model.eval()

    for i in range(min(n_samples, len(dataset))):
        img_tensor, label = dataset[i]
        input_tensor = img_tensor.unsqueeze(0).to(device)

        # Calcular mapa de activación para la clase predicha
        grayscale_cam = cam(
            input_tensor=input_tensor,
            targets=[ClassifierOutputTarget(label)],
        )[0]

        # Desnormalizar imagen para visualización
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = img_tensor.cpu().numpy().transpose(1, 2, 0)
        img_np = np.clip(img_np * std + mean, 0, 1).astype(np.float32)

        # Superponer mapa de calor
        vis = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

        cls_name = dataset.classes[label]
        out_path = output_dir / f"gradcam_{i:03d}_{cls_name}.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

    logger.info(f"Grad-CAM++ generado para {min(n_samples, len(dataset))} imágenes → {output_dir}")
    del cam  # hooks are auto-removed by __del__ in recent pytorch-grad-cam


# ── Entrypoint principal ──────────────────────────────────────────────────────


@app.command()
def main(
    config: Path = typer.Option(
        Path("configs/train_clf.yaml"),
        "--config",
        "-c",
        help="Ruta al archivo de configuración YAML.",
    ),
    model: str = typer.Option(
        None,
        "--model",
        "-m",
        help="Sobreescribe la arquitectura del config (ej. convnext_tiny).",
    ),
    resume: Path = typer.Option(
        None,
        "--resume",
        "-r",
        help="Checkpoint .pt para reanudar el entrenamiento.",
    ),
) -> None:
    """
    Entrena el clasificador binario Alternaria alternata / Otros hongos.

    Ejecuta Fase A (cabeza) + Fase B (fine-tuning parcial) y evalúa
    el mejor checkpoint en el test set con métricas completas.
    """
    # ── Configuración ────────────────────────────────────────────────
    with open(config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if model:
        cfg["model"]["architecture"] = model

    set_seed(cfg["project"]["seed"])
    device = get_device()

    logger.info(f"{'=' * 60}")
    logger.info(f"  Proyecto  : {cfg['project']['name']} v{cfg['project']['version']}")
    logger.info(f"  Modelo    : {cfg['model']['architecture']}")
    logger.info(f"  Dispositivo: {device}")
    logger.info(f"{'=' * 60}")

    # ── DataLoaders ──────────────────────────────────────────────────
    train_dl, val_dl, test_dl, class_names = build_dataloaders(cfg, device)

    # ── Modelo ───────────────────────────────────────────────────────
    mdl = AlternariaCLF(
        model_name=cfg["model"]["architecture"],
        num_classes=cfg["model"]["num_classes"],
        pretrained=cfg["model"]["pretrained"],
        dropout_rate=cfg["model"]["dropout_rate"],
    ).to(device)

    if resume and resume.exists():
        ck = torch.load(resume, map_location=device, weights_only=False)
        mdl.load_state_dict(ck["model_state_dict"])
        logger.info(f"Checkpoint cargado para reanudar: {resume}")

    # ── Función de pérdida con pesos de clase ────────────────────────
    cw = None
    if cfg["data"].get("class_weights_auto", True):
        cw = train_dl.dataset.get_class_weights().to(device)
        logger.info(f"Class weights: {dict(zip(class_names, cw.cpu().tolist()))}")
    criterion = nn.CrossEntropyLoss(weight=cw)

    # ── AMP Scaler (solo CUDA) ───────────────────────────────────────
    use_amp = cfg["training"].get("mixed_precision", True) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    ckpt_dir = Path(cfg["checkpoints"]["save_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_f1 = 0.0
    patience = cfg["training"]["early_stopping"]["patience"]
    pat_cnt = 0
    train_losses: list[float] = []
    val_losses: list[float] = []
    val_f1s: list[float] = []

    # ═════════════════════════════════════════════════════════════════
    # FASE A — Solo cabeza clasificadora (backbone congelado)
    # ═════════════════════════════════════════════════════════════════
    fa = cfg["fine_tuning"]["phase_a"]
    logger.info(f"\n{'=' * 60}")
    logger.info(f"  FASE A | epochs={fa['epochs']} | lr={fa['lr']}")
    logger.info("  Backbone congelado — solo head entrenable")
    logger.info(f"{'=' * 60}")

    mdl.freeze_backbone()
    opt_a = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, mdl.parameters()),
        lr=fa["lr"],
        weight_decay=cfg["optimizer"]["weight_decay"],
    )

    for ep in range(1, fa["epochs"] + 1):
        t0 = time.time()
        tl, ta = train_epoch(mdl, train_dl, opt_a, criterion, device, scaler)
        vl, va, y_true_a, y_pred_a, y_prob_a = eval_epoch(mdl, val_dl, criterion, device)
        vf1 = sk_f1(y_true_a, y_pred_a, zero_division=0) if len(y_true_a) > 0 else 0.0
        elapsed = time.time() - t0
        logger.info(
            f"[A] {ep:02d}/{fa['epochs']} | "
            f"train loss={tl:.4f} acc={ta:.3f} | "
            f"val loss={vl:.4f} acc={va:.3f} f1={vf1:.4f} | "
            f"{elapsed:.1f}s"
        )

    # ═════════════════════════════════════════════════════════════════
    # FASE B — Fine-tuning parcial (últimos N bloques)
    # ═════════════════════════════════════════════════════════════════
    fb = cfg["fine_tuning"]["phase_b"]
    logger.info(f"\n{'=' * 60}")
    logger.info(
        f"  FASE B | epochs={fb['epochs']} | lr={fb['lr']} | "
        f"desbloqueando últimos {fb['unfreeze_last_n_blocks']} bloques"
    )
    logger.info(f"{'=' * 60}")

    mdl.unfreeze_last_n_blocks(fb["unfreeze_last_n_blocks"])
    opt_b = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, mdl.parameters()),
        lr=fb["lr"],
        weight_decay=cfg["optimizer"]["weight_decay"],
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_b,
        T_max=fb["epochs"],
        eta_min=cfg["scheduler"]["eta_min"],
    )

    for ep in range(1, fb["epochs"] + 1):
        t0 = time.time()
        tl, ta = train_epoch(mdl, train_dl, opt_b, criterion, device, scaler)
        vl, va, y_true_v, y_pred_v, y_prob_v = eval_epoch(mdl, val_dl, criterion, device)
        sched.step()

        vf1 = float(sk_f1(y_true_v, y_pred_v, zero_division=0))
        train_losses.append(tl)
        val_losses.append(vl)
        val_f1s.append(vf1)

        elapsed = time.time() - t0
        lr_current = sched.get_last_lr()[0]
        logger.info(
            f"[B] {ep:02d}/{fb['epochs']} | "
            f"train loss={tl:.4f} acc={ta:.3f} | "
            f"val loss={vl:.4f} acc={va:.3f} f1={vf1:.4f} | "
            f"lr={lr_current:.2e} | {elapsed:.1f}s"
        )

        # Early stopping y checkpoint del mejor modelo
        if vf1 > best_f1:
            best_f1 = vf1
            pat_cnt = 0
            mdl.save(ckpt_dir / "best_model.pt", ep, vf1, va)
            logger.info(f"  ✓ Nuevo mejor modelo guardado (F1={vf1:.4f})")
        else:
            pat_cnt += 1
            if pat_cnt >= patience:
                logger.info(
                    f"Early stopping en época {ep} (sin mejora en {patience} épocas consecutivas)."
                )
                break

    # Guardar último checkpoint
    mdl.save(ckpt_dir / "last_model.pt", fb["epochs"], best_f1, 0.0)

    # ═════════════════════════════════════════════════════════════════
    # EVALUACIÓN FINAL — TEST SET
    # ═════════════════════════════════════════════════════════════════
    logger.info(f"\n{'=' * 60}")
    logger.info("  EVALUACIÓN FINAL — TEST SET")
    logger.info(f"{'=' * 60}")

    best_mdl = AlternariaCLF.load(ckpt_dir / "best_model.pt", device)
    _, _, y_true_t, y_pred_t, y_prob_t = eval_epoch(best_mdl, test_dl, criterion, device)
    metrics = compute_metrics(y_true_t, y_pred_t, y_prob_t, class_names)

    results_dir = Path("outputs") / cfg["project"]["name"]
    results_dir.mkdir(parents=True, exist_ok=True)

    plot_confusion_matrix(
        metrics.cm,
        class_names,
        save_path=results_dir / "confusion_matrix.png",
        title=f"Matriz de Confusión — {cfg['model']['architecture']}",
    )
    plot_roc_curve(
        y_true_t,
        y_prob_t,
        metrics.auc_roc,
        save_path=results_dir / "roc_curve.png",
    )
    plot_precision_recall_curve(
        y_true_t,
        y_prob_t,
        metrics.auc_pr,
        save_path=results_dir / "pr_curve.png",
    )
    plot_training_history(
        train_losses,
        val_losses,
        val_f1s,
        save_path=results_dir / "training_history.png",
    )
    save_metrics_csv(
        metrics,
        save_path=results_dir / "metrics_log.csv",
        experiment_name=cfg["model"]["architecture"],
    )

    # Grad-CAM sobre test set
    if cfg["metrics"]["gradcam"]["enabled"]:
        generate_gradcam_samples(
            model=best_mdl,
            dataset=test_dl.dataset,
            device=device,
            output_dir=results_dir / "gradcam",
            n_samples=cfg["metrics"]["gradcam"]["n_samples"],
        )

    logger.info("\n  Entrenamiento finalizado.")
    logger.info(f"  Best val F1  : {best_f1:.4f}")
    logger.info(f"  Test F1      : {metrics.f1:.4f}")
    logger.info(f"  Test AUC-ROC : {metrics.auc_roc:.4f}")
    logger.info(
        f"  Objetivos clínicos: "
        f"{'✓ CUMPLIDOS' if metrics.meets_clinical_targets() else '✗ NO CUMPLIDOS'}"
    )
    logger.info(f"  Resultados en: {results_dir}")


if __name__ == "__main__":
    app()
