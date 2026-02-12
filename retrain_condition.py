import os
import json
from datetime import datetime

# ===============================
# CONFIG
# ===============================
FEEDBACK_DIR = "feedback_images"
STATUS_FILE = "retrain_status.json"
MIN_IMAGES = 20          # ครบ 20 รูป
DAYS_LIMIT = 5           # หรือครบ 5 วัน
VALID_EXT = (".jpg", ".jpeg", ".png")
CLASS_NAMES = ["low", "medium", "high"]

# ===============================
# นับจำนวน feedback images
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

# ===============================
# โหลดสถานะ retrain ล่าสุด
# ===============================
def load_status():
    if not os.path.exists(STATUS_FILE):
        return None
    with open(STATUS_FILE, "r") as f:
        return json.load(f)

# ===============================
# บันทึกสถานะ retrain
# ===============================
def save_status(image_count):
    status = {
        "last_retrain": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "image_count": image_count
    }
    with open(STATUS_FILE, "w") as f:
        json.dump(status, f, indent=2)

# ===============================
# ตรวจว่าควร retrain หรือไม่
# ===============================
def should_retrain():
    current_images = count_feedback_images()
    status = load_status()

    # ยังไม่เคย retrain มาก่อน
    if status is None:
        return current_images >= MIN_IMAGES, current_images

    last_time = datetime.strptime(
        status["last_retrain"], "%Y-%m-%d %H:%M:%S"
    )
    days_passed = (datetime.now() - last_time).days

    # เงื่อนไขที่ 1: รูปครบ
    if current_images >= MIN_IMAGES:
        return True, current_images

    # เงื่อนไขที่ 2: เวลาครบ
    if days_passed >= DAYS_LIMIT:
        return True, current_images

    return False, current_images
