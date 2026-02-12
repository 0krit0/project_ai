import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from datetime import datetime

# ===============================
# CONFIG
# ===============================
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 5
FEEDBACK_DIR = "feedback_images"
MIN_IMAGES = 20
BASE_MODEL_PATH = "damage_model.h5"
CLASS_NAMES = ["low", "medium", "high"]
VALID_EXT = (".jpg", ".jpeg", ".png")

# ===============================
# Count feedback images
# ===============================
def count_feedback_images():
    total = 0
    for level in CLASS_NAMES:
        path = os.path.join(FEEDBACK_DIR, level)
        if os.path.exists(path):
            total += len([
                f for f in os.listdir(path)
                if f.lower().endswith(VALID_EXT)
            ])
    return total

total_images = count_feedback_images()
print(f"Feedback images: {total_images}")

if total_images < MIN_IMAGES:
    print("Not enough data for retraining")
    print(f"Need at least {MIN_IMAGES} images")
    raise SystemExit(0)

# ===============================
# Load base model
# ===============================
print("Loading base model...")
model = load_model(BASE_MODEL_PATH)

# ===============================
# Freeze feature extractor layers
# ===============================
for layer in model.layers[:-3]:
    layer.trainable = False

print("Frozen base layers. Training top layers only.")

# ===============================
# Prepare feedback dataset
# ===============================
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=10,
    zoom_range=0.1,
    horizontal_flip=True
)

train_data = datagen.flow_from_directory(
    FEEDBACK_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

val_data = datagen.flow_from_directory(
    FEEDBACK_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

# ===============================
# Compile model
# ===============================
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ===============================
# Retrain model
# ===============================
print("Starting model retraining...")
model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    verbose=1
)

# ===============================
# Save retrained model
# ===============================
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
new_model_name = f"damage_model_retrained_{timestamp}.h5"
model.save(new_model_name)

print("Retraining complete")
print(f"Saved model: {new_model_name}")
print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
