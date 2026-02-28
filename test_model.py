import os
import unittest


class ModelSmokeTests(unittest.TestCase):
    def test_predict_with_sample_image(self):
        if not os.path.exists("damage_model.h5"):
            self.skipTest("damage_model.h5 not found")
        if not os.path.exists("test.jpg"):
            self.skipTest("test.jpg not found")

        import numpy as np
        import tensorflow as tf
        from PIL import Image

        labels = ["high", "low", "medium"]
        model = tf.keras.models.load_model("damage_model.h5")

        img = Image.open("test.jpg").convert("RGB")
        img = img.resize((224, 224))
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = np.expand_dims(arr, axis=0)

        pred = model.predict(arr, verbose=0)
        pred_index = int(np.argmax(pred))
        confidence = float(np.max(pred))

        self.assertIn(labels[pred_index], labels)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)


if __name__ == "__main__":
    unittest.main()
