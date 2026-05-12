from PIL import Image
import os

folder = "gatekeep_model/2/dataset_labuyo_leaf_checker/non_siling_labuyo_leaf"

for root, dirs, files in os.walk(folder):
    for file in files:
        if file.lower().startswith("lam") and file.lower().endswith((".jpg", ".jpeg")):
            path = os.path.join(root, file)
            try:
                img = Image.open(path)
                img = img.convert("RGB")
                img.save(path, "JPEG", quality=95)
                print(f"Converted {file}")
            except Exception as e:
                print("Error fixing:", path)