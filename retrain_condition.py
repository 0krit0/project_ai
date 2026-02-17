import os
import json
import sqlite3
from datetime import datetime

# ===============================
# CONFIG
# ===============================
FEEDBACK_DIR = "feedback_images"
STATUS_FILE = "retrain_status.json"
MIN_IMAGES = 20
DAYS_LIMIT = 5
VALID_EXT = (".jpg", ".jpeg", ".png")
CLASS_NAMES = ["low", "medium", "high"]
DB_NAME = "app.db"


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


def count_reviewed_case_images():
    if not os.path.exists(DB_NAME):
        return 0
    try:
        conn = sqlite3.connect(DB_NAME)
        cur = conn.cursor()
        cur.execute(
            """
            SELECT COUNT(*)
            FROM case_images ci
            JOIN cases c ON c.id = ci.case_id
            WHERE c.status = 'reviewed'
              AND c.final_result IN ('low', 'medium', 'high')
            """
        )
        value = int(cur.fetchone()[0] or 0)
        conn.close()
        return value
    except Exception:
        return 0


# ===============================
# Load last retrain status
# ===============================
def load_status():
    if not os.path.exists(STATUS_FILE):
        return None
    with open(STATUS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


# ===============================
# Save retrain status
# ===============================
def save_status(image_count):
    status = {
        "last_retrain": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "image_count": image_count,
    }
    with open(STATUS_FILE, "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2, ensure_ascii=False)


# ===============================
# Check retrain condition
# ===============================
def should_retrain():
    current_images = count_feedback_images() + count_reviewed_case_images()
    status = load_status()

    if status is None:
        return current_images >= MIN_IMAGES, current_images

    last_time = datetime.strptime(
        status["last_retrain"], "%Y-%m-%d %H:%M:%S"
    )
    days_passed = (datetime.now() - last_time).days

    if current_images >= MIN_IMAGES:
        return True, current_images

    if days_passed >= DAYS_LIMIT:
        return True, current_images

    return False, current_images
