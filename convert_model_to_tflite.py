import tensorflow as tf

# Load your saved model
model = tf.keras.models.load_model("models/gatekeeper2_v3small.keras")

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)

tflite_model = converter.convert()

# Save file
with open("tflite_models/gatekeeper2_v3small.tflite", "wb") as f:
    f.write(tflite_model)

print("TFLite model created successfully!")