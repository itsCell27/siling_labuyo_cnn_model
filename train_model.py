import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV3Small, MobileNetV2, MobileNetV3Large
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input as preprocess_v3
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# ================= SETTINGS =================

IMG_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 20
NUM_CLASSES = 4

train_dir = "dataset/train"
val_dir = "dataset/validate"
test_dir = "dataset/test"


# ================= LOAD DATA =================

train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = train_ds.class_names
print("Classes:", class_names)

# ================= NORMALIZATION =================

normalization_layer = layers.Rescaling(1./255)

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

    epochs=EPOCHS,

    callbacks=[early_stop]

)


# ================= TEST =================

test_loss, test_acc = model.evaluate(test_ds)

print("Test accuracy:", test_acc)


# ================= SAVE MODEL =================

model.save("models/siliguide_model2/siliguide_model2.4.5_v3small.keras")


# ================= PLOT =================

plt.plot(history.history["accuracy"], label="train")

plt.plot(history.history["val_accuracy"], label="val")

plt.legend()

plt.title("Accuracy")

plt.show()