import os
import shutil
import random
from pathlib import Path

random.seed(42)

def main():
    base_dir = Path("data/processed/segmentation")
    images_train_dir = base_dir / "images/train"
    labels_train_dir = base_dir / "labels/train"
    
    images_val_dir = base_dir / "images/val"
    labels_val_dir = base_dir / "labels/val"
    
    images_test_dir = base_dir / "images/test"
    labels_test_dir = base_dir / "labels/test"
    
    for d in [images_val_dir, labels_val_dir, images_test_dir, labels_test_dir]:
        d.mkdir(parents=True, exist_ok=True)
        
    all_images = list(images_train_dir.glob("*.[jJ][pP][gG]")) + list(images_train_dir.glob("*.[pP][nN][gG]"))
    all_images = sorted(all_images)
    
    random.shuffle(all_images)
    
    total = len(all_images)
    val_size = int(total * 0.15)
    test_size = int(total * 0.15)
    
    val_images = all_images[:val_size]
    test_images = all_images[val_size:val_size + test_size]
    
    def move_files(files, dest_img_dir, dest_lbl_dir):
        for img_path in files:
            lbl_path = labels_train_dir / (img_path.stem + ".txt")
            if lbl_path.exists():
                shutil.move(img_path, dest_img_dir / img_path.name)
                shutil.move(lbl_path, dest_lbl_dir / lbl_path.name)
            else:
                print(f"Warning: No label for {img_path.name}")
                
    move_files(val_images, images_val_dir, labels_val_dir)
    move_files(test_images, images_test_dir, labels_test_dir)
    
    print(f"Moved {len(val_images)} to val and {len(test_images)} to test.")
    print("Done.")

if __name__ == "__main__":
    main()
