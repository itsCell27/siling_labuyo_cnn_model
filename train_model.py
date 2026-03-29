import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV3Small, MobileNetV2
import matplotlib.pyplot as plt

# ================= SETTINGS =================

IMG_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 30
NUM_CLASSES = 4

train_dir = "dataset/train"
val_dir = "dataset/validate"
test_dir = "dataset/test"


# ================= IMAGE PREPROCESS =================
# Convert 3:4 to 1:1 by resizing and cropping

def resize_and_crop(image, label):

    image = tf.image.resize(image, (256, 256), preserve_aspect_ratio=True)

    image = tf.image.resize_with_crop_or_pad(
        image,
        IMG_SIZE,
        IMG_SIZE
    )

    image = tf.cast(image, tf.float32) / 255.0

    return image, label


# ================= LOAD DATA =================

train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=(300, 300),  # load bigger first
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=(300, 300),
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=(300, 300),
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = train_ds.class_names
print("Classes:", class_names)


# apply crop pipeline

train_ds = train_ds.map(resize_and_crop)
val_ds = val_ds.map(resize_and_crop)
test_ds = test_ds.map(resize_and_crop)


# ================= MODEL =================

base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet"
)

# no fine tuning
base_model.trainable = False


model = models.Sequential([

    base_model,

    layers.GlobalAveragePooling2D(),

    layers.Dense(
        NUM_CLASSES,
        activation="softmax"
    )

])


model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]

)


model.summary()


# ================= TRAIN =================

history = model.fit(

    train_ds,

    validation_data=val_ds,

    epochs=EPOCHS

)


# ================= TEST =================

test_loss, test_acc = model.evaluate(test_ds)

print("Test accuracy:", test_acc)


# ================= SAVE MODEL =================

#model.save("models/siling_labuyo_leaf_state_classification_model1.keras")


# ================= PLOT =================

plt.plot(history.history["accuracy"], label="train")

plt.plot(history.history["val_accuracy"], label="val")

plt.legend()

plt.title("Accuracy")

plt.show()