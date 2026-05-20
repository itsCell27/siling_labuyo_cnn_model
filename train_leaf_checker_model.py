import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input as preprocess_v3
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score

# ================= SETTINGS =================

IMG_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 50
NUM_CLASSES = 2

train_dir = "dataset_split3/train"
val_dir = "dataset_split3/validate"
test_dir = "dataset_split3/test"

# Create the output directory if it doesn't exist
os.makedirs("gatekeep_model/3", exist_ok=True)

# ================= LOAD DATA =================
# Added crop_to_aspect_ratio=True to automatically center-crop rectangular images to 1x1

train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    crop_to_aspect_ratio=True, 
    shuffle=True
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    crop_to_aspect_ratio=True,
    shuffle=False
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    crop_to_aspect_ratio=True,
    shuffle=False
)

class_names = train_ds.class_names
print("Classes:", class_names)

# ================= NORMALIZATION & PREPROCESSING =================

# MobileNetV3 requires a specific preprocessing function
train_ds = train_ds.map(lambda x, y: (preprocess_v3(x), y))
val_ds = val_ds.map(lambda x, y: (preprocess_v3(x), y))
test_ds = test_ds.map(lambda x, y: (preprocess_v3(x), y))

# ================= AUGMENTATION =================

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.1),
    layers.RandomBrightness(0.1),
])

# Apply augmentation ONLY on training
train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))

# Early stopping callback
early_stop = EarlyStopping(
    monitor="val_loss",     # watch validation loss
    patience=5,             # wait 5 epochs before stopping
    restore_best_weights=True,  # rollback to best model
    verbose=1
)

# ================= MODEL =================

base_model = MobileNetV3Small(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze base model (No fine-tuning for Phase 1)
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(
        NUM_CLASSES, # Will create 2 output nodes
        activation="softmax"
    )
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ================= TRAIN =================

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[early_stop]
)

# ================= TEST =================

test_loss, test_acc = model.evaluate(test_ds)
print(f"Test accuracy: {test_acc:.4f}")

# ================= SAVE STANDARD MODEL =================
print("\nSaving standard model...")
model.save("gatekeep_model/3/leaf_checker_model3_v3small.keras")

# ================= EXTRACT WEIGHTS =================
print("\nExtracting Dense Layer weights for Kotlin Heatmap generation...")
# In our Sequential model: layers[0] is MobileNet, layers[1] is GAP, layers[2] is Dense
dense_layer = model.layers[2]
# get_weights()[0] grabs the connection weights between the extracted features and your 2 classes
dense_weights = dense_layer.get_weights()[0] 
np.savetxt("gatekeep_model/3/leaf_checker_model3_dense_weights.csv", dense_weights, delimiter=",")
print("Saved leaf_checker_model3_dense_weights.csv successfully!")

# ================= RE-WIRE TO DUAL-OUTPUT =================
print("\nRe-wiring architecture for Dual-Output...")

# 1. Create a brand new Input Tensor explicitly
inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

# 2. Connect the layers manually to tap into the feature maps
features = model.layers[0](inputs)       # Output 2: The 7x7 Feature Maps (MobileNet base)
x = model.layers[1](features)            # Global Average Pooling
predictions = model.layers[2](x)         # Output 1: The Classification Probabilities

# 3. Create the Dual-Output model
dual_output_model = models.Model(
    inputs=inputs, 
    outputs=[
        predictions, 
        features
    ]
)

print("\nConverting Dual-Output model to TensorFlow Lite for Android...")
# 4. Convert directly to TFLite for your Kotlin application
converter = tf.lite.TFLiteConverter.from_keras_model(dual_output_model)
tflite_model = converter.convert()

with open("gatekeep_model/3/leaf_checker_model3_dual_output.tflite", "wb") as f:
    f.write(tflite_model)
print("Saved leaf_checker_model3_dual_output.tflite successfully!")

# ================= PLOT =================
plt.figure(figsize=(8,6))
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.legend()
plt.title("Model Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.tight_layout()
plt.show()


# =================================================================
# AUTOMATED EVALUATION & CONFUSION MATRIX
# =================================================================
print("\n==================================================")
print("GENERATING CONFUSION MATRIX & METRICS")
print("==================================================")

# 1. Get true labels directly from the test dataset
# (This works perfectly because test_ds was set with shuffle=False)
y_true = np.concatenate([y for x, y in test_ds], axis=0)

# 2. Get predictions from the newly trained model
print("Running predictions on the test dataset...")
predictions = model.predict(test_ds)
y_pred = np.argmax(predictions, axis=1)

# 3. Calculate Overall Capstone Metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

print("\nMODEL RESULTS:")
print(f"Accuracy:  {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall:    {recall * 100:.2f}%")
print(f"F1-Score:  {f1 * 100:.2f}%\n")

print("--------------------------------------------------")
print("DETAILED CLASSIFICATION REPORT")
print("--------------------------------------------------")
# class_names was defined earlier in your training script via train_ds.class_names
print(classification_report(y_true, y_pred, target_names=class_names))

# 4. Generate and Plot Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, 
            yticklabels=class_names)

plt.title('SiliGuide Post-Training Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')

# Save the plot automatically
plt.tight_layout()
matrix_filename = "gatekeep_model/3/leaf_checker_model3_confusion_matrix.png"
plt.savefig(matrix_filename, dpi=300)
print(f"\nSaved confusion matrix visualization as '{matrix_filename}'.")

# Show the matrix on screen
plt.show()