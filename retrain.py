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
# นับจำนวนรูป feedback ที่ถูกต้อง
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
print(f"📸 จำนวนรูป feedback ทั้งหมด: {total_images}")

if total_images < MIN_IMAGES:
    print("❌ ข้อมูลยังไม่เพียงพอสำหรับการ retrain")
    print(f"ต้องการอย่างน้อย {MIN_IMAGES} รูป")
    exit()

# ===============================
# โหลดโมเดลเดิม
# ===============================
print("🔄 โหลดโมเดลเดิม...")
model = load_model(BASE_MODEL_PATH)

# ===============================
# Freeze feature extractor
# ===============================
for layer in model.layers[:-3]:
    layer.trainable = False

print("🔒 Freeze layers สำเร็จ (train เฉพาะ layer บน)")

# ===============================
# เตรียมข้อมูลจาก feedback_images
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
# Compile โมเดล (incremental learning)
# ===============================
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ===============================
# Re-train
# ===============================
print("🚀 เริ่ม Re-train โมเดลจาก feedback...")
model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    verbose=1
)

# ===============================
# บันทึกโมเดลเวอร์ชันใหม่
# ===============================
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
new_model_name = f"damage_model_retrained_{timestamp}.h5"
model.save(new_model_name)

print("✅ Re-train สำเร็จ")
print(f"📦 บันทึกโมเดลใหม่เป็น {new_model_name}")
print(f"🕒 เวลา: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

