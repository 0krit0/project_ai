import tensorflow as tf
import numpy as np
from PIL import Image

# โหลดโมเดล
model = tf.keras.models.load_model("damage_model.h5")

# label ตามโฟลเดอร์ dataset
labels = ["high", "low", "medium"]  
# หมายเหตุ: TensorFlow เรียงตามชื่อโฟลเดอร์ A-Z

# โหลดรูป
img = Image.open("test.jpg").convert("RGB")
img = img.resize((224, 224))
img = np.array(img) / 255.0
img = np.expand_dims(img, axis=0)

# ทำนายผล
pred = model.predict(img)
pred_index = np.argmax(pred)
confidence = np.max(pred)

print("ผลการทำนาย:", labels[pred_index])
print("ความมั่นใจ:", round(confidence * 100, 2), "%")
