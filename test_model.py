import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model("damage_model.h5")

# Labels from dataset folder order (A-Z)
labels = ["high", "low", "medium"]

# Load image
img = Image.open("test.jpg").convert("RGB")
img = img.resize((224, 224))
img = np.array(img) / 255.0
img = np.expand_dims(img, axis=0)

# Predict
pred = model.predict(img)
pred_index = np.argmax(pred)
confidence = np.max(pred)

print("Prediction:", labels[pred_index])
print("Confidence:", round(confidence * 100, 2), "%")
