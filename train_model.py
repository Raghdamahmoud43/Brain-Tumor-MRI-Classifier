
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# =========================
# Settings
# =========================
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15

TRAIN_DIR = "Training"
TEST_DIR = "Testing"

# =========================
# Load Training Dataset
# =========================
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# =========================
# Load Validation Dataset
# =========================
val_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# =========================
# Load Testing Dataset
# =========================
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# Save class names
class_names = train_ds.class_names

print("\nClasses:", class_names)

# =========================
# Improve Performance
# =========================
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.shuffle(1000, seed=42).prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

# =========================
# Data Augmentation
# =========================
augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.08),
])

# =========================
# MobileNetV2 Base Model
# =========================
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze pretrained layers
base_model.trainable = False

# =========================
# Build Model
# =========================
inputs = layers.Input(shape=(224, 224, 3))

x = augmentation(inputs)

x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

x = base_model(x, training=False)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dense(256, activation="relu")(x)

x = layers.Dropout(0.4)(x)

outputs = layers.Dense(
    len(class_names),
    activation="softmax"
)(x)

model = models.Model(inputs, outputs)

# =========================
# Compile
# =========================
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =========================
# Callbacks
# =========================
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "best_model.keras",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    min_lr=0.00001,
    verbose=1
)

# =========================
# Train
# =========================
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[
        checkpoint,
        early_stopping,
        reduce_lr
    ]
)

# =========================
# Load Best Saved Model
# =========================
model = tf.keras.models.load_model("best_model.keras")

print("\nBest model loaded successfully!")

# =========================
# Evaluate on Test Set
# =========================
test_loss, test_accuracy = model.evaluate(test_ds)

print("\nTest Accuracy:", test_accuracy)
print("Test Loss:", test_loss)

# =========================
# Predictions
# =========================
y_true = []
y_pred = []

for images, labels in test_ds:
    predictions = model.predict(images, verbose=0)

    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(predictions, axis=1))

# =========================
# Confusion Matrix
# =========================
cm = confusion_matrix(y_true, y_pred)

print("\nConfusion Matrix:")
print(cm)

# =========================
# Classification Report
# =========================
print("\nClassification Report:")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names
    )
)
