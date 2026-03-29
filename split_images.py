import os
import random
import shutil

SOURCE_DIR = "raw_leaf_no_duplicates/dataset_leaf_no_duplicates"
OUTPUT_DIR = "dataset"

# 80% for training, 10% for validation, 10% for testing
SPLIT = (0.8, 0.1, 0.1)

# get class names from source directory
classes = os.listdir(SOURCE_DIR)

for cls in classes:
    class_path = os.path.join(SOURCE_DIR, cls)
    images = os.listdir(class_path)

    random.shuffle(images)

    total = len(images)
    train_end = int(total * SPLIT[0])
    val_end = int(total * (SPLIT[0] + SPLIT[1]))

    splits = {
        "train": images[:train_end],
        "validate": images[train_end:val_end],
        "test": images[val_end:]
    }

    for split_name, split_images in splits.items():
        split_dir = os.path.join(OUTPUT_DIR, split_name, cls)
        os.makedirs(split_dir, exist_ok=True)

        for img in split_images:
            src = os.path.join(class_path, img)
            dst = os.path.join(split_dir, img)
            shutil.copy(src, dst)

print("Done")