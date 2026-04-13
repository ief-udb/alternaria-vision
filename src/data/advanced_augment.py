"""
advanced_augment.py
-------------------
Genera un dataset balanceado V2 aplicando:
1) Tiling (Recortes): Divide las imágenes de 1280x960 en parches superpuestos
   de 640x640 para mejorar la resolución relativa de hifas escasas.
2) Oversampling Físico: Multiplica (clona) mediante rotaciones/flips los
   parches que contengan clases minoritarias (conidia, hifa).

Uso:
    uv run python src/data/advanced_augment.py
"""

from __future__ import annotations

import ast
import shutil
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


SRC_DIR = Path("data/processed/segmentation")
DEST_DIR = Path("data/processed/segmentation_v2")

CLASS_NAMES = ["conidia", "conidia-multiseptada", "hifa"]

# Multiplicadores base para oversampling de tiles generados
OVERSAMPLE_RATIOS = {
    0: 5,  # conidia: duplicar 5x
    1: 1,  # conidia-multiseptada: 1x (no duplicar)
    2: 12, # hifa: duplicar 12x
}

TILE_SIZE = 640
OVERLAP = 128 # Paso de 512


def read_yolo_polygons(txt_path: Path, img_shape: tuple[int, int]) -> list[tuple[int, np.ndarray]]:
    if not txt_path.exists():
        return []

    h, w = img_shape[:2]
    polygons = []
    
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            cls_id = int(parts[0])
            coords = np.array(parts[1:], dtype=np.float32).reshape(-1, 2)
            # Des-normalizar
            coords[:, 0] *= w
            coords[:, 1] *= h
            polygons.append((cls_id, coords.astype(np.int32)))
            
    return polygons


def write_yolo_polygons(txt_path: Path, polygons: list[tuple[int, np.ndarray]], img_shape: tuple[int, int]) -> None:
    h, w = img_shape[:2]
    lines = []
    for cls_id, poly in polygons:
        if len(poly) < 3:
            continue
        # Normalizar
        norm_poly = poly.astype(np.float32)
        norm_poly[:, 0] /= w
        norm_poly[:, 1] /= h
        # Asegurar dentro de [0, 1]
        norm_poly = np.clip(norm_poly, 0.0, 1.0)
        
        poly_str = " ".join([f"{x:.6f} {y:.6f}" for x, y in norm_poly])
        lines.append(f"{cls_id} {poly_str}")
        
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def poly_to_mask(polygons: list[tuple[int, np.ndarray]], shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    for cls_id, poly in polygons:
        cv2.fillPoly(mask, [poly], int(cls_id + 1))
    return mask


def mask_to_polygons(mask: np.ndarray) -> list[tuple[int, np.ndarray]]:
    polygons = []
    unique_ids = np.unique(mask)
    for uid in unique_ids:
        if uid == 0:
            continue
        cls_id = int(uid - 1)
        class_mask = (mask == uid).astype(np.uint8)
        contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            if cv2.contourArea(cnt) > 50:
                poly = cnt.squeeze()
                if poly.ndim == 2 and len(poly) >= 3:
                    polygons.append((cls_id, poly))
    return polygons


def rotate_image_and_mask(img: np.ndarray, mask: np.ndarray, angle_idx: int) -> tuple[np.ndarray, np.ndarray]:
    if angle_idx == 1:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE), cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)
    elif angle_idx == 2:
        return cv2.rotate(img, cv2.ROTATE_180), cv2.rotate(mask, cv2.ROTATE_180)
    elif angle_idx == 3:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE), cv2.rotate(mask, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif angle_idx == 4:
        return cv2.flip(img, 1), cv2.flip(mask, 1)
    elif angle_idx == 5:
        return cv2.flip(img, 0), cv2.flip(mask, 0)
    elif angle_idx == 6:
        return cv2.rotate(cv2.flip(img, 1), cv2.ROTATE_90_CLOCKWISE), cv2.rotate(cv2.flip(mask, 1), cv2.ROTATE_90_CLOCKWISE)
    elif angle_idx == 7:
        return cv2.rotate(cv2.flip(img, 1), cv2.ROTATE_90_COUNTERCLOCKWISE), cv2.rotate(cv2.flip(mask, 1), cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img, mask


def process_dataset(split: str = "train"):
    img_dir_in = SRC_DIR / "images" / split
    lbl_dir_in = SRC_DIR / "labels" / split
    
    img_dir_out = DEST_DIR / "images" / split
    lbl_dir_out = DEST_DIR / "labels" / split
    
    img_dir_out.mkdir(parents=True, exist_ok=True)
    lbl_dir_out.mkdir(parents=True, exist_ok=True)
    
    img_files = list(img_dir_in.glob("*.jpg")) + list(img_dir_in.glob("*.JPG"))
    img_files += list(img_dir_in.glob("*.png")) + list(img_dir_in.glob("*.PNG"))
    
    print(f"--- Procesando {split} ({len(img_files)} imgs) ---")
    
    total_crops_generated = 0
    oversampled_count = 0
    
    for img_path in tqdm(img_files, desc=split):
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
            
        h_orig, w_orig = img_bgr.shape[:2]
        lbl_path = lbl_dir_in / (img_path.stem + ".txt")
        
        polys = read_yolo_polygons(lbl_path, (h_orig, w_orig))
        mask_orig = poly_to_mask(polys, (h_orig, w_orig))
        
        step = TILE_SIZE - OVERLAP
        
        y_starts = list(range(0, h_orig - TILE_SIZE + 1, step))
        if y_starts[-1] + TILE_SIZE < h_orig: y_starts.append(h_orig - TILE_SIZE)
            
        x_starts = list(range(0, w_orig - TILE_SIZE + 1, step))
        if x_starts[-1] + TILE_SIZE < w_orig: x_starts.append(w_orig - TILE_SIZE)
            
        if not y_starts: y_starts = [0]
        if not x_starts: x_starts = [0]
            
        for y in y_starts:
            for x in x_starts:
                x2 = min(x + TILE_SIZE, w_orig)
                y2 = min(y + TILE_SIZE, h_orig)
                
                img_crop = img_bgr[y:y2, x:x2]
                mask_crop = mask_orig[y:y2, x:x2]
                
                has_labels = np.any(mask_crop > 0)
                if not has_labels and np.random.rand() > 0.05:
                    continue  # Guardar un 5% de fondos vacios
                    
                crop_polys = mask_to_polygons(mask_crop)
                if has_labels and not crop_polys:
                    continue

                max_oversample = 1
                if split == "train":
                    classes_in_crop = set([c for c, _ in crop_polys])
                    for cid in classes_in_crop:
                        if OVERSAMPLE_RATIOS[cid] > max_oversample:
                            max_oversample = OVERSAMPLE_RATIOS[cid]
                
                for r in range(max_oversample):
                    final_img = img_crop
                    final_mask = mask_crop
                    
                    if r > 0:
                        aug_idx = (r % 7) + 1
                        final_img, final_mask = rotate_image_and_mask(img_crop, mask_crop, aug_idx)
                        oversampled_count += 1
                        
                    out_polys = mask_to_polygons(final_mask)
                    
                    if not out_polys and has_labels:
                        continue
                        
                    base_name = f"{img_path.stem}_T{y}_{x}_R{r}"
                    out_img_path = img_dir_out / f"{base_name}.jpg"
                    out_lbl_path = lbl_dir_out / f"{base_name}.txt"
                    
                    cv2.imwrite(str(out_img_path), final_img)
                    if out_polys:
                        write_yolo_polygons(out_lbl_path, out_polys, final_img.shape)
                        
                    total_crops_generated += 1

    print(f"  > Generados {total_crops_generated} tiles (incluyendo {oversampled_count} copias extra).")


def create_data_yaml():
    yaml_in = SRC_DIR / "data.yaml"
    yaml_out = DEST_DIR / "data_v2.yaml"
    
    with open(yaml_in, "r", encoding="utf-8") as f:
        content = f.read()
        
    abs_path = str(DEST_DIR.resolve()).replace("\\", "\\\\")
    content = content.replace(str(SRC_DIR.resolve()).replace("\\", "\\\\"), abs_path)
    
    # Just in case the absolutes didn't correctly replace
    content = content.replace("segmentation", "segmentation_v2")
    # Also fix paths relative or absolute inside yaml
    lines = []
    for line in content.splitlines():
        if line.startswith("path:"):
            lines.append(f"path: {abs_path}")
        else:
            lines.append(line)
    
    with open(yaml_out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    
    print(f"\nGenerado data.yaml en {yaml_out}")


if __name__ == "__main__":
    if DEST_DIR.exists():
        shutil.rmtree(DEST_DIR)
        
    process_dataset("train")
    process_dataset("val")
    process_dataset("test")
    
    create_data_yaml()
    print("\n¡Dataset V2 Generado Exitosamente!")
