import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input as preprocess_v3

# ================= SETTINGS =================

IMG_SIZE = 224
MODEL_PATH = "models/siliguide_model2/2.4/siliguide_model2.4.5_v3small.keras"

# UPDATE THIS: Point this to the folder containing your 15% testing split
TEST_DIR = "dataset/test"  

# Note: The order of these classes must match the alphabetical order of your folders 
# if you want the labels to align correctly, or you can strictly map them.
CLASS_NAMES = [
    "curling",
    "healthy",
    "spots",
    "yellowing"
]

# ================= LOAD MODEL =================

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!\n")

# ================= PREPROCESS FUNCTION =================
# Using your exact preprocessing logic to ensure consistency

def preprocess_image(image_path):
    # Load and convert to array
    image = Image.open(image_path).convert("RGB")
    image = np.array(image)

    # 1. Resize exactly like training
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))

    # 2. Cast to float32 (DO NOT divide by 255.0)
    image = tf.cast(image, tf.float32)
    
    # Add batch dimension
    image = tf.expand_dims(image, axis=0)

    # 3. Apply MobileNetV3 preprocessor
    image = preprocess_v3(image)

    return image

# ================= RUN EVALUATION =================

y_true = []
y_pred = []

print("Processing test images...")

# Loop through each class folder
for class_index, class_name in enumerate(CLASS_NAMES):
    folder_path = os.path.join(TEST_DIR, class_name)
    
    if not os.path.exists(folder_path):
        print(f"Warning: Folder not found -> {folder_path}")
        continue
        
    # Process every image in the folder
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder_path, filename)
            
            try:
                # Preprocess the image
                processed_image = preprocess_image(img_path)
                
                # Predict
                predictions = model.predict(processed_image, verbose=0)
                predicted_class_index = np.argmax(predictions[0])
                
                # Store results
                y_true.append(class_index)
                y_pred.append(predicted_class_index)
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

print("\nEvaluation complete! Generating reports...\n")

# ================= OUTPUT RESULTS =================

# 1. Print Text Report (Precision, Recall, F1-Score)
print("--------------------------------------------------")
print("CLASSIFICATION REPORT")
print("--------------------------------------------------")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

# 2. Generate and Plot Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=CLASS_NAMES, 
            yticklabels=CLASS_NAMES)

plt.title('SiliGuide MobileNetV3 Confusion Matrix')
plt.ylabel('Actual Symptom')
plt.xlabel('Predicted Symptom')

# Save the plot for your capstone documentation
plt.tight_layout()
plt.savefig('siliguide_confusion_matrix_2.4.5_v3small.png', dpi=300)
print("Saved confusion matrix visualization as 'siliguide_confusion_matrix.png'.")

# Show the plot on screen
plt.show()