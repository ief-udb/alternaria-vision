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
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.data.augmentations import (
    _IMAGENET_MEAN,
    _IMAGENET_STD,
    compute_dataset_stats,
    get_train_transforms,
    get_val_transforms,
)
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

import torch.nn.functional as F

app = typer.Typer(help="Entrena el clasificador binario Alternaria vs Otros Hongos.")
logger = get_logger(__name__, log_file=Path("logs/train_clf.log"))

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, reduction='mean', label_smoothing=0.0):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none', label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


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


# ── Mixup / CutMix ────────────────────────────────────────────────────────────


def _rand_bbox(size: tuple, lam: float) -> tuple[int, int, int, int]:
    """Calcula bounding box aleatorio para CutMix."""
    W, H = size[2], size[3]
    cut_w = int(W * np.sqrt(1.0 - lam))
    cut_h = int(H * np.sqrt(1.0 - lam))
    cx, cy = np.random.randint(W), np.random.randint(H)
    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    return x1, y1, x2, y2


def mixup_batch(
    imgs: torch.Tensor, lbls: torch.Tensor, alpha: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Aplica Mixup: mezcla lineal de pares de imágenes/etiquetas."""
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    idx = torch.randperm(imgs.size(0), device=imgs.device)
    return lam * imgs + (1 - lam) * imgs[idx], lbls, lbls[idx], lam


def cutmix_batch(
    imgs: torch.Tensor, lbls: torch.Tensor, alpha: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Aplica CutMix: reemplaza un parche rectangular con otra imagen."""
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    idx = torch.randperm(imgs.size(0), device=imgs.device)
    x1, y1, x2, y2 = _rand_bbox(imgs.size(), lam)
    imgs = imgs.clone()
    imgs[:, :, x1:x2, y1:y2] = imgs[idx, :, x1:x2, y1:y2]
    lam = 1 - (x2 - x1) * (y2 - y1) / (imgs.size(-1) * imgs.size(-2))
    return imgs, lbls, lbls[idx], lam


def mixed_criterion(
    criterion: nn.Module,
    pred: torch.Tensor,
    ya: torch.Tensor,
    yb: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """Pérdida interpolada para Mixup/CutMix."""
    return lam * criterion(pred, ya) + (1 - lam) * criterion(pred, yb)


# ── Splits y DataLoaders ──────────────────────────────────────────────────────



def build_dataloaders(
    cfg: dict,
    device: torch.device,
    mean: tuple = _IMAGENET_MEAN,
    std: tuple = _IMAGENET_STD,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    """
    Crea los tres DataLoaders con splits estratificados.

    La estratificación garantiza que la proporción de clases
    se preserve en train, val y test, crítico con datasets pequeños.

    Parameters
    ----------
    mean, std : tuple
        Estadísticas de normalización. Por defecto ImageNet.
        Pasa stats del dataset para imágenes de microscopía.
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
    train_ds = MicroscopyDataset(
        root, get_train_transforms(img_size, mean, std), splits_dir / "train.txt"
    )
    val_ds = MicroscopyDataset(
        root, get_val_transforms(img_size, mean, std), splits_dir / "val.txt"
    )
    test_ds = MicroscopyDataset(
        root, get_val_transforms(img_size, mean, std), splits_dir / "test.txt"
    )

    batch = get_batch_size(
        cfg["model"]["architecture"],
        device,
        override=cfg["training"].get("batch_size"),
    )
    nw = cfg["training"]["num_workers"]
    pin = cfg["training"]["pin_memory"] and device.type == "cuda"

    kw = dict(num_workers=nw, pin_memory=pin)
    train_dl = DataLoader(train_ds, batch_size=batch, shuffle=True, drop_last=True, **kw)
    val_dl = DataLoader(val_ds, batch_size=batch, shuffle=False, **kw)
    test_dl = DataLoader(test_ds, batch_size=batch, shuffle=False, **kw)

    return train_dl, val_dl, test_dl, full_ds.classes


def build_train_loader(
    cfg: dict,
    device: torch.device,
    splits_dir: Path,
    image_size: int,
    mean: tuple = _IMAGENET_MEAN,
    std: tuple = _IMAGENET_STD,
) -> DataLoader:
    """Reconstruye solo el DataLoader de entrenamiento con un image_size distinto.

    Usado por Progressive Resizing para cambiar la resolución entre etapas
    sin regenerar los splits ni los loaders de val/test.
    """
    root = Path(cfg["data"]["root"])
    train_ds = MicroscopyDataset(
        root, get_train_transforms(image_size, mean, std), splits_dir / "train.txt"
    )
    batch = get_batch_size(
        cfg["model"]["architecture"],
        device,
        override=cfg["training"].get("batch_size"),
    )
    nw = cfg["training"]["num_workers"]
    pin = cfg["training"]["pin_memory"] and device.type == "cuda"
    return DataLoader(train_ds, batch_size=batch, shuffle=True, drop_last=True, num_workers=nw, pin_memory=pin)


# ── Epoch functions ───────────────────────────────────────────────────────────


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler,
    mixup_alpha: float = 0.0,
    cutmix_alpha: float = 0.0,
) -> tuple[float, float]:
    """
    Ejecuta una época de entrenamiento.

    Usa AMP (Automatic Mixed Precision) si scaler no es None.
    Aplica Mixup o CutMix aleatoriamente si los alphas son > 0.

    Returns
    -------
    tuple[float, float]
        (loss_promedio, accuracy)
    """
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    use_mix = mixup_alpha > 0 or cutmix_alpha > 0

    for imgs, lbls in tqdm(loader, desc="  train", leave=False):
        imgs, lbls = imgs.to(device), lbls.to(device)
        optimizer.zero_grad()

        # Elegir Mixup o CutMix aleatoriamente (50/50 cuando ambos activos)
        ya, yb, lam = lbls, lbls, 1.0
        if use_mix and imgs.size(0) > 1:
            if cutmix_alpha > 0 and (mixup_alpha <= 0 or random.random() < 0.5):
                imgs, ya, yb, lam = cutmix_batch(imgs, lbls, cutmix_alpha)
            elif mixup_alpha > 0:
                imgs, ya, yb, lam = mixup_batch(imgs, lbls, mixup_alpha)

        if scaler is not None and device.type == "cuda":
            with torch.amp.autocast("cuda"):
                out = model(imgs)
                loss = mixed_criterion(criterion, out, ya, yb, lam)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(imgs)
            loss = mixed_criterion(criterion, out, ya, yb, lam)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        # Accuracy con etiqueta original (ya) para monitoreo
        correct += (out.argmax(1) == ya).sum().item()
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


# ── K-Fold Cross-Validation ───────────────────────────────────────────────────


def run_kfold(cfg: dict, device: torch.device, n_folds: int) -> None:
    """
    Ejecuta Stratified K-Fold CV y reporta métricas promedio ± std.

    No reemplaza el entrenamiento estándar: es un modo de evaluación
    que estima la varianza de las métricas dado el tamaño del dataset.
    """
    from src.models.classifier import AlternariaCLF

    root = Path(cfg["data"]["root"])
    img_size = cfg["data"]["image_size"]
    seed = cfg["project"]["seed"]
    tr = cfg["training"]
    mix_a = tr.get("mixup_alpha", 0.0)
    cut_a = tr.get("cutmix_alpha", 0.0)

    mean, std = _IMAGENET_MEAN, _IMAGENET_STD
    norm_cfg = cfg["data"].get("normalize", {})
    if norm_cfg.get("use_dataset_stats", False):
        n_samples = norm_cfg.get("dataset_stats_samples", 500)
        logger.info("K-Fold: calculando estadísticas del dataset...")
        mean, std = compute_dataset_stats(root, img_size, n_samples)
        logger.info(f"  mean={mean}  std={std}")

    full_ds = MicroscopyDataset(root=root, transform=None)
    labels = np.array(full_ds.get_labels())
    indices = np.arange(len(full_ds))

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_metrics: list[dict] = []

    use_amp = tr.get("mixed_precision", True) and device.type == "cuda"
    batch = get_batch_size(cfg["model"]["architecture"], device,
                           override=tr.get("batch_size"))
    nw, pin = tr["num_workers"], tr["pin_memory"] and device.type == "cuda"
    ls = tr.get("label_smoothing", 0.0)

    for fold, (train_idx, val_idx) in enumerate(skf.split(indices, labels), 1):
        logger.info(f"\n{'─' * 50}")
        logger.info(f"  K-Fold {fold}/{n_folds} | train={len(train_idx)} | val={len(val_idx)}")

        train_ds = MicroscopyDataset(root, get_train_transforms(img_size, mean, std))
        train_ds.samples = [full_ds.samples[i] for i in train_idx]
        val_ds = MicroscopyDataset(root, get_val_transforms(img_size, mean, std))
        val_ds.samples = [full_ds.samples[i] for i in val_idx]

        train_dl = DataLoader(train_ds, batch_size=batch, shuffle=True, drop_last=True,
                              num_workers=nw, pin_memory=pin)
        val_dl = DataLoader(val_ds, batch_size=batch, shuffle=False,
                            num_workers=nw, pin_memory=pin)

        mdl = AlternariaCLF(
            model_name=cfg["model"]["architecture"],
            num_classes=cfg["model"]["num_classes"],
            pretrained=cfg["model"]["pretrained"],
            dropout_rate=cfg["model"]["dropout_rate"],
        ).to(device)

        cw = train_ds.get_class_weights().to(device) if cfg["data"].get("class_weights_auto") else None
        criterion = FocalLoss(weight=cw, label_smoothing=ls, gamma=3.0)
        scaler = torch.amp.GradScaler("cuda") if use_amp else None

        # Fase A
        fa = cfg["fine_tuning"]["phase_a"]
        mdl.freeze_backbone()
        opt_a = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, mdl.parameters()),
            lr=fa["lr"], weight_decay=cfg["optimizer"]["weight_decay"],
        )
        for _ in range(fa["epochs"]):
            train_epoch(mdl, train_dl, opt_a, criterion, device, scaler, mix_a, cut_a)

        # Fase B
        fb = cfg["fine_tuning"]["phase_b"]
        mdl.unfreeze_last_n_blocks(fb["unfreeze_last_n_blocks"])
        opt_b = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, mdl.parameters()),
            lr=fb["lr"], weight_decay=cfg["optimizer"]["weight_decay"],
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt_b, T_max=fb["epochs"], eta_min=cfg["scheduler"]["eta_min"]
        )
        best_f1, pat_cnt = 0.0, 0
        patience = cfg["training"]["early_stopping"]["patience"]
        best_state = None

        for ep in range(1, fb["epochs"] + 1):
            train_epoch(mdl, train_dl, opt_b, criterion, device, scaler, mix_a, cut_a)
            _, _, yt, yp, ypr = eval_epoch(mdl, val_dl, criterion, device)
            sched.step()
            vf1 = float(sk_f1(yt, yp, zero_division=0))
            if vf1 > best_f1:
                best_f1 = vf1
                pat_cnt = 0
                best_state = {k: v.cpu().clone() for k, v in mdl.state_dict().items()}
            else:
                pat_cnt += 1
                if pat_cnt >= patience:
                    break

        if best_state:
            mdl.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        _, _, yt, yp, ypr = eval_epoch(mdl, val_dl, criterion, device)
        from src.evaluation.metrics import compute_metrics
        m = compute_metrics(yt, yp, ypr, full_ds.classes)
        fold_metrics.append({"f1": m.f1, "auc": m.auc_roc, "sens": m.sensitivity,
                              "spec": m.specificity})
        logger.info(f"  Fold {fold} → F1={m.f1:.4f} AUC={m.auc_roc:.4f} "
                    f"sens={m.sensitivity:.4f} spec={m.specificity:.4f}")

    # Resumen
    logger.info(f"\n{'=' * 60}")
    logger.info(f"  K-Fold ({n_folds} folds) — Resumen")
    for key in ("f1", "auc", "sens", "spec"):
        vals = [d[key] for d in fold_metrics]
        logger.info(f"  {key:>6}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
    logger.info(f"{'=' * 60}")


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
    kfold: int = typer.Option(
        0,
        "--kfold",
        "-k",
        help="Nº de folds para K-Fold CV. 0 = entrenamiento estándar.",
    ),
) -> None:
    """
    Entrena el clasificador binario Alternaria alternata / Otros hongos.

    Ejecuta Fase A (cabeza) + Fase B (fine-tuning parcial) y evalúa
    el mejor checkpoint en el test set con métricas completas.
    Con --kfold N activa Stratified K-Fold Cross-Validation.
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

    # ── Normalización del dataset ────────────────────────────────────
    norm_cfg = cfg["data"].get("normalize", {})
    mean, std = _IMAGENET_MEAN, _IMAGENET_STD
    if norm_cfg.get("use_dataset_stats", False):
        n_samp = norm_cfg.get("dataset_stats_samples", 500)
        logger.info("Calculando estadísticas reales del dataset...")
        mean, std = compute_dataset_stats(
            cfg["data"]["root"], cfg["data"]["image_size"], n_samp
        )
        logger.info(f"  mean={tuple(f'{v:.4f}' for v in mean)}")
        logger.info(f"  std ={tuple(f'{v:.4f}' for v in std)}")
    else:
        logger.info("Usando estadísticas ImageNet para normalización.")

    # ── Modo K-Fold ──────────────────────────────────────────────────
    if kfold > 1:
        logger.info(f"\n  Modo K-Fold CV ({kfold} folds)")
        run_kfold(cfg, device, kfold)
        return

    # ── DataLoaders ──────────────────────────────────────────────────
    train_dl, val_dl, test_dl, class_names = build_dataloaders(cfg, device, mean, std)

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

    # ── Función de pérdida con pesos de clase y label smoothing ──────
    cw = None
    if cfg["data"].get("class_weights_auto", True):
        cw = train_dl.dataset.get_class_weights().to(device)
        logger.info(f"Class weights: {dict(zip(class_names, cw.cpu().tolist()))}")
    ls = cfg["training"].get("label_smoothing", 0.0)
    criterion = FocalLoss(weight=cw, label_smoothing=ls, gamma=3.0)
    logger.info(f"Label smoothing: {ls} | Usando Focal Loss (gamma=3.0)")

    # ── AMP Scaler (solo CUDA) ───────────────────────────────────────
    use_amp = cfg["training"].get("mixed_precision", True) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    # ── Mixup / CutMix ───────────────────────────────────────────────
    mix_a = cfg["training"].get("mixup_alpha", 0.0)
    cut_a = cfg["training"].get("cutmix_alpha", 0.0)
    if mix_a > 0 or cut_a > 0:
        logger.info(f"Mixup alpha={mix_a}  CutMix alpha={cut_a}")

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
    pr_cfg = cfg["training"].get("progressive_resizing", {})
    pr_enabled = pr_cfg.get("enabled", False)

    # Resolución Fase A (menor si progressive resizing está activo)
    fa_size = pr_cfg.get("phase_a_size", cfg["data"]["image_size"]) if pr_enabled else cfg["data"]["image_size"]
    if pr_enabled and fa_size != cfg["data"]["image_size"]:
        logger.info(f"Progressive Resizing: Fase A a {fa_size}px")
        train_dl = build_train_loader(cfg, device, Path("data/splits"), fa_size, mean, std)

    logger.info(f"\n{'=' * 60}")
    logger.info(f"  FASE A | epochs={fa['epochs']} | lr={fa['lr']} | size={fa_size}px")
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
        # Sin Mixup/CutMix en Fase A (cabeza nueva, necesita gradientes limpios)
        tl, ta = train_epoch(mdl, train_dl, opt_a, criterion, device, scaler)
        vl, va, y_true_a, y_pred_a, _ = eval_epoch(mdl, val_dl, criterion, device)
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

    # ── SWA setup ────────────────────────────────────────────────────
    swa_cfg = cfg["training"].get("swa", {})
    swa_enabled = swa_cfg.get("enabled", False)
    swa_start = swa_cfg.get("start_epoch", 15)
    swa_model = None
    swa_sched = None
    if swa_enabled:
        swa_model = torch.optim.swa_utils.AveragedModel(mdl)
        swa_sched = torch.optim.swa_utils.SWALR(
            opt_b, swa_lr=swa_cfg.get("lr", 1e-5), anneal_epochs=5
        )
        logger.info(f"SWA activado: promediará pesos desde época B={swa_start}")

    # ── Progressive Resizing — etapas Fase B ─────────────────────────
    pb_stages = pr_cfg.get("phase_b_stages", []) if pr_enabled else []
    pb_stage_idx = 0
    pb_epoch_count = 0  # contador de épocas dentro de la etapa actual
    if pb_stages:
        cur_size = pb_stages[0]["size"]
        logger.info(f"Progressive Resizing: Fase B iniciando a {cur_size}px")
        train_dl = build_train_loader(cfg, device, Path("data/splits"), cur_size, mean, std)

    for ep in range(1, fb["epochs"] + 1):
        # ── Cambio de resolución (Progressive Resizing) ───────────────
        if pb_stages and pb_stage_idx < len(pb_stages):
            stage = pb_stages[pb_stage_idx]
            if pb_epoch_count >= stage["epochs"]:
                pb_stage_idx += 1
                pb_epoch_count = 0
                if pb_stage_idx < len(pb_stages):
                    new_size = pb_stages[pb_stage_idx]["size"]
                    logger.info(f"  Progressive Resizing: cambiando a {new_size}px (época B={ep})")
                    train_dl = build_train_loader(
                        cfg, device, Path("data/splits"), new_size, mean, std
                    )
            pb_epoch_count += 1

        t0 = time.time()
        tl, ta = train_epoch(mdl, train_dl, opt_b, criterion, device, scaler, mix_a, cut_a)
        vl, va, y_true_v, y_pred_v, y_prob_v = eval_epoch(mdl, val_dl, criterion, device)

        # SWA: promediar pesos y usar su scheduler a partir de swa_start
        if swa_enabled and ep >= swa_start:
            swa_model.update_parameters(mdl)
            swa_sched.step()
        else:
            sched.step()

        vf1 = float(sk_f1(y_true_v, y_pred_v, zero_division=0))
        train_losses.append(tl)
        val_losses.append(vl)
        val_f1s.append(vf1)

        elapsed = time.time() - t0
        lr_current = opt_b.param_groups[0]["lr"]
        swa_tag = " [SWA]" if swa_enabled and ep >= swa_start else ""
        logger.info(
            f"[B] {ep:02d}/{fb['epochs']}{swa_tag} | "
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

    # ── Finalizar SWA ────────────────────────────────────────────────
    if swa_enabled and swa_model is not None:
        logger.info("SWA: actualizando batch norm stats del modelo promediado...")
        torch.optim.swa_utils.update_bn(train_dl, swa_model, device=device)
        # Guardar modelo SWA como alternativa al best_model
        torch.save(
            {"model_state_dict": swa_model.module.state_dict()},
            ckpt_dir / "swa_model.pt",
        )
        logger.info(f"  SWA model guardado → {ckpt_dir / 'swa_model.pt'}")

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

