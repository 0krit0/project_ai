import os
import csv
import json
import shutil
import sqlite3
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# ===============================
# CONFIG
# ===============================
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 8
BASE_MODEL_PATH = "damage_model.h5"
CLASS_NAMES = ["low", "medium", "high"]
VALID_EXT = (".jpg", ".jpeg", ".png", ".webp")
DB_NAME = "app.db"

FEEDBACK_DIR = "feedback_images"
AUTOSET_DIR = os.path.join(FEEDBACK_DIR, "_autolabel")
MERGED_DIR = os.path.join(FEEDBACK_DIR, "_retrain_dataset")
MIN_IMAGES = 20


def normalize_path(path):
    return str(path or "").replace("\\", "/").strip()


def valid_label(label):
    lv = (label or "").strip().lower()
    return lv if lv in CLASS_NAMES else None


def is_valid_image_path(path):
    p = normalize_path(path)
    if not p:
        return False
    if not os.path.exists(p):
        return False
    if not p.lower().endswith(VALID_EXT):
        return False
    if "_heatmap" in os.path.basename(p).lower():
        return False
    return True


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def clear_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def collect_from_feedback_dirs():
    samples = []
    for label in CLASS_NAMES:
        folder = os.path.join(FEEDBACK_DIR, label)
        if not os.path.isdir(folder):
            continue
        for name in os.listdir(folder):
            src = os.path.join(folder, name)
            if is_valid_image_path(src):
                samples.append((label, src, "feedback_dir"))
    return samples


def collect_from_feedback_table(conn):
    samples = []
    cur = conn.cursor()
    cur.execute(
        """
        SELECT result, is_correct, image_path
        FROM feedback
        ORDER BY created_at DESC
        """
    )
    for result, is_correct, image_path in cur.fetchall():
        label = valid_label(result)
        if not label:
            continue
        if (is_correct or "").strip().lower() not in {"yes", "true", "1", "correct"}:
            continue
        src = normalize_path(image_path)
        if is_valid_image_path(src):
            samples.append((label, src, "feedback_table"))
    return samples


def collect_from_reviewed_cases(conn):
    samples = []
    cur = conn.cursor()
    cur.execute(
        """
        SELECT c.final_result, ci.image_path
        FROM case_images ci
        JOIN cases c ON c.id = ci.case_id
        WHERE c.status = 'reviewed'
          AND c.final_result IN ('low', 'medium', 'high')
        ORDER BY ci.created_at DESC
        """
    )
    for final_result, image_path in cur.fetchall():
        label = valid_label(final_result)
        if not label:
            continue
        src = normalize_path(image_path)
        if is_valid_image_path(src):
            samples.append((label, src, "reviewed_case"))
    return samples


def dedupe_samples(samples):
    out = []
    seen = set()
    for label, src, source in samples:
        key = (label, os.path.abspath(src).lower())
        if key in seen:
            continue
        seen.add(key)
        out.append((label, src, source))
    return out


def materialize_dataset(samples):
    clear_dir(AUTOSET_DIR)
    clear_dir(MERGED_DIR)

    for root in [AUTOSET_DIR, MERGED_DIR]:
        for label in CLASS_NAMES:
            ensure_dir(os.path.join(root, label))

    manifest_path = os.path.join(AUTOSET_DIR, "manifest.csv")
    with open(manifest_path, "w", encoding="utf-8", newline="") as mf:
        writer = csv.writer(mf)
        writer.writerow(["label", "source", "src_path", "copied_path"])
        serial = {k: 0 for k in CLASS_NAMES}
        for label, src, source in samples:
            serial[label] += 1
            ext = os.path.splitext(src)[1].lower() or ".jpg"
            name = f"{label}_{serial[label]:06d}{ext}"
            dst_auto = os.path.join(AUTOSET_DIR, label, name)
            dst_merged = os.path.join(MERGED_DIR, label, name)
            try:
                shutil.copy2(src, dst_auto)
                shutil.copy2(src, dst_merged)
                writer.writerow([label, source, src, dst_auto])
            except OSError:
                continue

    for label in CLASS_NAMES:
        src_dir = os.path.join(FEEDBACK_DIR, label)
        dst_dir = os.path.join(MERGED_DIR, label)
        if not os.path.isdir(src_dir):
            continue
        for name in os.listdir(src_dir):
            src = os.path.join(src_dir, name)
            if not is_valid_image_path(src):
                continue
            base = os.path.basename(src)
            dst = os.path.join(dst_dir, f"base_{base}")
            try:
                shutil.copy2(src, dst)
            except OSError:
                continue


def count_dataset_images(path):
    counts = {}
    total = 0
    for label in CLASS_NAMES:
        folder = os.path.join(path, label)
        if not os.path.isdir(folder):
            counts[label] = 0
            continue
        cnt = len([f for f in os.listdir(folder) if f.lower().endswith(VALID_EXT)])
        counts[label] = cnt
        total += cnt
    return counts, total


def compute_class_weights(y_classes, num_classes):
    if len(y_classes) == 0:
        return None
    counts = np.bincount(y_classes, minlength=num_classes).astype(np.float64)
    nonzero = counts[counts > 0]
    if len(nonzero) == 0:
        return None
    total = float(np.sum(counts))
    weights = {}
    for i, c in enumerate(counts):
        if c <= 0:
            continue
        weights[i] = total / (num_classes * c)
    return weights


def confusion_matrix_np(y_true, y_pred, num_classes):
    mat = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            mat[t, p] += 1
    return mat


def main():
    all_samples = []
    all_samples.extend(collect_from_feedback_dirs())

    if os.path.exists(DB_NAME):
        conn = sqlite3.connect(DB_NAME)
        all_samples.extend(collect_from_feedback_table(conn))
        all_samples.extend(collect_from_reviewed_cases(conn))
        conn.close()

    all_samples = dedupe_samples(all_samples)
    materialize_dataset(all_samples)

    class_counts, total_images = count_dataset_images(MERGED_DIR)
    print(f"Merged training images: {total_images} | counts={class_counts}")

    if total_images < MIN_IMAGES:
        print("Not enough data for retraining")
        print(f"Need at least {MIN_IMAGES} images")
        raise SystemExit(0)

    if any(class_counts.get(c, 0) < 3 for c in CLASS_NAMES):
        print("Not enough balanced data per class (need >= 3 each)")
        raise SystemExit(0)

    print("Loading base model...")
    model = load_model(BASE_MODEL_PATH)

    for layer in model.layers[:-5]:
        layer.trainable = False
    print("Frozen base layers. Fine-tuning top layers.")

    datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.2,
        rotation_range=14,
        zoom_range=0.18,
        width_shift_range=0.08,
        height_shift_range=0.08,
        shear_range=0.08,
        horizontal_flip=True,
        brightness_range=(0.8, 1.2),
    )

    train_data = datagen.flow_from_directory(
        MERGED_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training",
        shuffle=True,
    )

    val_data = datagen.flow_from_directory(
        MERGED_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
    )

    model.compile(
        optimizer=Adam(learning_rate=8e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    class_weights = compute_class_weights(train_data.classes, len(train_data.class_indices))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    new_model_name = f"damage_model_retrained_{timestamp}.h5"

    callbacks = [
        EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
        ModelCheckpoint(new_model_name, monitor="val_accuracy", save_best_only=True),
    ]

    print("Starting model retraining...")
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=EPOCHS,
        verbose=1,
        class_weight=class_weights,
        callbacks=callbacks,
    )

    model.save(new_model_name)

    val_data.reset()
    pred = model.predict(val_data, verbose=0)
    y_pred = np.argmax(pred, axis=1)
    y_true = np.array(val_data.classes, dtype=np.int64)
    inv_map = {v: k for k, v in val_data.class_indices.items()}
    labels = [inv_map[i] for i in range(len(inv_map))]

    cm = confusion_matrix_np(y_true, y_pred, len(labels))
    cm_dict = {
        labels[r]: {labels[c]: int(cm[r, c]) for c in range(len(labels))}
        for r in range(len(labels))
    }

    val_loss, val_acc = model.evaluate(val_data, verbose=0)

    report = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_file": new_model_name,
        "dataset_total": total_images,
        "dataset_class_counts": class_counts,
        "class_indices": val_data.class_indices,
        "val_accuracy": float(val_acc),
        "val_loss": float(val_loss),
        "confusion_matrix": cm_dict,
        "history": {k: [float(x) for x in v] for k, v in history.history.items()},
    }

    report_path = f"retrain_report_{timestamp}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("Retraining complete")
    print(f"Saved model: {new_model_name}")
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
