import os
import io
import csv
import json
import uuid
import re
import logging
import subprocess
from queue import Queue, Empty
from functools import wraps
from datetime import datetime
from threading import Lock, Thread
from collections import defaultdict, deque
from logging.handlers import RotatingFileHandler

import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError

from flask import (
    Flask,
    request,
    redirect,
    session,
    render_template,
    send_from_directory,
    Response,
    jsonify,
)
from werkzeug.security import check_password_hash, generate_password_hash

from rules import RULES
from retrain_condition import should_retrain, save_status
from db import (
    init_db,
    migrate_from_csv,
    get_or_create_user,
    get_user_auth,
    create_user,
    set_user_password,
    set_user_avatar,
    add_history,
    add_feedback,
    get_user_history,
    get_profile_summary,
    get_profile_insights,
    get_dashboard_data,
    get_admin_metrics,
    get_recent_feedback,
    get_recent_audit_logs,
    count_admin_users,
    list_users_basic,
    update_user_role_status,
    log_audit,
    create_case,
    list_user_cases,
    list_cases_for_review,
    list_recent_reviewed_cases,
    add_case_image,
    delete_case_image,
    get_case_detail,
    get_case_for_review,
    review_case,
    create_notification,
    get_notifications,
    mark_notification_read,
    create_analysis_job,
    update_analysis_job,
    get_analysis_job,
    get_analysis_job_stats,
    restore_database_from_upload,
)

# ================== APP SETUP ==================
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-change-me")


def env_int(name, default):
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return int(default)


LOG_PATH = os.getenv("APP_LOG_PATH", "run.app.log")
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
LOGIN_PASSWORD = os.getenv("APP_LOGIN_PASSWORD", "")
MODEL_VERSION = os.getenv("MODEL_VERSION", "damage_model.h5")
SESSION_TIMEOUT_MIN = env_int("SESSION_TIMEOUT_MIN", 480)
QUEUE_BACKEND = os.getenv("QUEUE_BACKEND", "local").strip().lower()
REDIS_URL = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")
INLINE_WORKER_ENABLED = os.getenv("INLINE_WORKER_ENABLED", "true").lower() == "true"
TOP_K_OUTCOMES = max(1, min(10, env_int("TOP_K_OUTCOMES", 3)))
ANALYZE_RATE_LIMIT_COUNT = env_int("ANALYZE_RATE_LIMIT_COUNT", 8)
ANALYZE_RATE_LIMIT_WINDOW_SEC = env_int("ANALYZE_RATE_LIMIT_WINDOW_SEC", 60)
NON_CAR_GUARD_ENABLED = os.getenv("NON_CAR_GUARD_ENABLED", "true").lower() == "true"
NON_CAR_MIN_CONFIDENCE = float(os.getenv("NON_CAR_MIN_CONFIDENCE", "72"))
NON_CAR_MIN_SCORE_GAP = float(os.getenv("NON_CAR_MIN_SCORE_GAP", "18"))
NON_CAR_MAX_ENTROPY = float(os.getenv("NON_CAR_MAX_ENTROPY", "0.85"))
NON_CAR_MODEL_ENABLED = os.getenv("NON_CAR_MODEL_ENABLED", "true").lower() == "true"
NON_CAR_MODEL_PATH = os.getenv("NON_CAR_MODEL_PATH", "car_noncar_model.h5")
NON_CAR_MODEL_MIN_CAR_PROB = float(os.getenv("NON_CAR_MODEL_MIN_CAR_PROB", "0.6"))
NON_CAR_MODEL_CAR_INDEX = env_int("NON_CAR_MODEL_CAR_INDEX", 1)
NON_CAR_HARD_BLOCK_MAX_CAR_PROB = float(os.getenv("NON_CAR_HARD_BLOCK_MAX_CAR_PROB", "0.2"))
NON_CAR_RULE_BLOCK_MIN_RISK = env_int("NON_CAR_RULE_BLOCK_MIN_RISK", 5)
REVIEWER_USERNAMES = {
    name.strip().lower()
    for name in os.getenv("REVIEWER_USERNAMES", "reviewer").split(",")
    if name.strip()
}
MANUAL_REVIEW_SCORE_GAP = float(os.getenv("MANUAL_REVIEW_SCORE_GAP", "20"))
MANUAL_REVIEW_MAX_ENTROPY = float(os.getenv("MANUAL_REVIEW_MAX_ENTROPY", "0.9"))
MANUAL_REVIEW_MAX_TTA_DISPERSION = float(os.getenv("MANUAL_REVIEW_MAX_TTA_DISPERSION", "0.085"))
MAX_MULTI_IMAGES = max(1, min(8, env_int("MAX_MULTI_IMAGES", 4)))
TTA_ENABLED = os.getenv("TTA_ENABLED", "true").lower() == "true"
TTA_VARIANTS = max(1, min(5, env_int("TTA_VARIANTS", 4)))

ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_UPLOAD_BYTES = env_int("MAX_UPLOAD_BYTES", 5 * 1024 * 1024)

RETRAIN_CHECK_LOCK = Lock()
RATE_LIMIT_LOCK = Lock()
MODEL_LOAD_LOCK = Lock()
WORKER_LOCK = Lock()
ANALYZE_HITS = defaultdict(deque)
_retrain_check_running = False
ANALYSIS_JOB_QUEUE = Queue()
WORKER_STARTED = False
REDIS_CLIENT = None
REDIS_QUEUE_KEY = "car_damage:analysis_jobs"

init_db()
migrate_from_csv()

# ================== MODEL ==================
model = None
non_car_model = None
NON_CAR_MODEL_LOAD_ATTEMPTED = False
labels = ["high", "low", "medium"]


def configure_logging():
    handler = RotatingFileHandler(LOG_PATH, maxBytes=1_000_000, backupCount=3, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    app.logger.setLevel(logging.INFO)
    app.logger.addHandler(handler)


configure_logging()


def get_queue_backend():
    return "redis" if REDIS_CLIENT is not None else "local"


def get_redis_client():
    global REDIS_CLIENT
    if QUEUE_BACKEND != "redis":
        return None
    if REDIS_CLIENT is not None:
        return REDIS_CLIENT
    try:
        import redis  # type: ignore

        client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
        client.ping()
        REDIS_CLIENT = client
        app.logger.info("analysis queue backend=redis url=%s", REDIS_URL)
        return REDIS_CLIENT
    except Exception as err:
        app.logger.warning("redis queue unavailable, fallback local: %s", err)
        REDIS_CLIENT = None
        return None


def enqueue_analysis_job(payload):
    redis_client = get_redis_client()
    if redis_client is not None:
        redis_client.rpush(REDIS_QUEUE_KEY, json.dumps(payload, ensure_ascii=False))
        return "redis"
    ANALYSIS_JOB_QUEUE.put(payload)
    return "local"


def get_analysis_queue_size():
    redis_client = get_redis_client()
    if redis_client is not None:
        try:
            return int(redis_client.llen(REDIS_QUEUE_KEY))
        except Exception:
            return 0
    return ANALYSIS_JOB_QUEUE.qsize()


def get_model():
    global model
    if model is not None:
        return model

    with MODEL_LOAD_LOCK:
        if model is None:
            app.logger.info("loading model: damage_model.h5")
            model = tf.keras.models.load_model("damage_model.h5")
            app.logger.info("model loaded successfully")
    return model


def get_non_car_model():
    global non_car_model, NON_CAR_MODEL_LOAD_ATTEMPTED
    if not NON_CAR_MODEL_ENABLED:
        return None
    if non_car_model is not None:
        return non_car_model
    if NON_CAR_MODEL_LOAD_ATTEMPTED:
        return None

    with MODEL_LOAD_LOCK:
        if non_car_model is not None:
            return non_car_model
        if NON_CAR_MODEL_LOAD_ATTEMPTED:
            return None
        NON_CAR_MODEL_LOAD_ATTEMPTED = True
        if not os.path.exists(NON_CAR_MODEL_PATH):
            app.logger.warning("non-car model file missing: %s", NON_CAR_MODEL_PATH)
            return None
        try:
            app.logger.info("loading non-car model: %s", NON_CAR_MODEL_PATH)
            non_car_model = tf.keras.models.load_model(NON_CAR_MODEL_PATH)
            app.logger.info("non-car model loaded successfully")
            return non_car_model
        except Exception as err:
            app.logger.warning("failed to load non-car model: %s", err)
            non_car_model = None
            return None


def request_meta():
    return {
        "ip_address": request.headers.get("X-Forwarded-For", request.remote_addr),
        "user_agent": request.headers.get("User-Agent", ""),
    }


def audit(action, user=None, target=None, details=None):
    meta = request_meta()
    log_audit(
        action=action,
        user_id=user.get("id") if user else None,
        username=user.get("name") if user else None,
        target=target,
        details=details,
        ip_address=meta["ip_address"],
        user_agent=meta["user_agent"],
    )


def touch_session():
    if not session.get("user"):
        return False
    now_ts = datetime.now().timestamp()
    last_seen = session.get("last_seen_ts")
    if last_seen and (now_ts - float(last_seen)) > (SESSION_TIMEOUT_MIN * 60):
        session.clear()
        return False
    session["last_seen_ts"] = now_ts
    return True


def require_login(view_func):
    @wraps(view_func)
    def wrapper(*args, **kwargs):
        if not touch_session():
            return redirect("/login")
        user = get_session_user()
        if user is None:
            return redirect("/login")
        return view_func(user, *args, **kwargs)

    return wrapper


def require_admin(view_func):
    @wraps(view_func)
    def wrapper(*args, **kwargs):
        if not touch_session():
            return redirect("/login")
        user = get_session_user()
        if user is None:
            return redirect("/login")
        if not is_admin_user(user):
            return redirect("/")
        return view_func(user, *args, **kwargs)

    return wrapper


def api_require_login(view_func):
    @wraps(view_func)
    def wrapper(*args, **kwargs):
        if not touch_session():
            return jsonify({"ok": False, "error": "unauthorized"}), 401
        user = get_session_user()
        if user is None:
            return jsonify({"ok": False, "error": "unauthorized"}), 401
        return view_func(user, *args, **kwargs)

    return wrapper


def api_require_admin(view_func):
    @wraps(view_func)
    def wrapper(*args, **kwargs):
        if not touch_session():
            return jsonify({"ok": False, "error": "unauthorized"}), 401
        user = get_session_user()
        if user is None:
            return jsonify({"ok": False, "error": "unauthorized"}), 401
        if not is_admin_user(user):
            return jsonify({"ok": False, "error": "forbidden"}), 403
        return view_func(user, *args, **kwargs)

    return wrapper


def api_require_reviewer(view_func):
    @wraps(view_func)
    def wrapper(*args, **kwargs):
        if not touch_session():
            return jsonify({"ok": False, "error": "unauthorized"}), 401
        user = get_session_user()
        if user is None:
            return jsonify({"ok": False, "error": "unauthorized"}), 401
        if not is_reviewer_user(user):
            return jsonify({"ok": False, "error": "forbidden"}), 403
        return view_func(user, *args, **kwargs)

    return wrapper


def validate_upload(file_storage):
    if file_storage is None:
        return "เนเธกเนเธเธเนเธเธฅเนเธฃเธนเธเธ—เธตเนเธญเธฑเธเนเธซเธฅเธ”"

    filename = (file_storage.filename or "").strip()
    if not filename:
        return "เธเธทเนเธญเนเธเธฅเนเนเธกเนเธ–เธนเธเธ•เนเธญเธ"

    ext = os.path.splitext(filename.lower())[1]
    if ext not in ALLOWED_IMAGE_EXTENSIONS:
        return "เธฃเธญเธเธฃเธฑเธเน€เธเธเธฒเธฐเนเธเธฅเน .jpg .jpeg .png .webp"

    mimetype = (file_storage.mimetype or "").lower()
    if not mimetype.startswith("image/"):
        return "เนเธเธฅเนเธ—เธตเนเธญเธฑเธเนเธซเธฅเธ”เธ•เนเธญเธเน€เธเนเธเธฃเธนเธเธ เธฒเธเน€เธ—เนเธฒเธเธฑเนเธ"

    pos = file_storage.stream.tell()
    file_storage.stream.seek(0, os.SEEK_END)
    size = file_storage.stream.tell()
    file_storage.stream.seek(pos, os.SEEK_SET)

    if size <= 0:
        return "เนเธเธฅเนเธฃเธนเธเธงเนเธฒเธเธซเธฃเธทเธญเธญเนเธฒเธเธเนเธญเธกเธนเธฅเนเธกเนเนเธ”เน"
    if size > MAX_UPLOAD_BYTES:
        return f"เนเธเธฅเนเนเธซเธเนเน€เธเธดเธเธเธณเธซเธเธ” (เธชเธนเธเธชเธธเธ” {MAX_UPLOAD_BYTES // (1024 * 1024)}MB)"
    return None


def save_avatar_file(file_storage, username):
    if file_storage is None or not (file_storage.filename or "").strip():
        return None, None
    err = validate_upload(file_storage)
    if err:
        return None, err

    safe_name = re.sub(r"[^a-zA-Z0-9_-]+", "_", (username or "user")).strip("_") or "user"
    avatar_dir = os.path.join("feedback_images", "avatars")
    os.makedirs(avatar_dir, exist_ok=True)
    filename = f"{safe_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    avatar_path = os.path.join(avatar_dir, filename).replace("\\", "/")
    img = Image.open(file_storage.stream).convert("RGB")
    # Center-crop to square then resize for consistent avatar rendering and smaller file size.
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    img = img.crop((left, top, left + side, top + side))
    img = img.resize((256, 256), Image.Resampling.LANCZOS)
    img.save(avatar_path, format="JPEG", quality=86, optimize=True)
    return avatar_path, None


def cleanup_avatar_file(path):
    norm = (path or "").replace("\\", "/")
    if not norm.startswith("feedback_images/avatars/"):
        return
    try:
        if os.path.exists(norm):
            os.remove(norm)
    except OSError:
        pass


def assess_image_quality(img_pil):
    img_rgb = img_pil.convert("RGB")
    arr = np.array(img_rgb, dtype=np.float32)
    h, w = arr.shape[:2]

    gray = arr.mean(axis=2)
    brightness = float(gray.mean())
    contrast = float(gray.std())

    grad_x = np.abs(np.diff(gray, axis=1)).mean() if w > 1 else 0.0
    grad_y = np.abs(np.diff(gray, axis=0)).mean() if h > 1 else 0.0
    edge_strength = float((grad_x + grad_y) / 2.0)

    if min(h, w) < 160:
        return "เธฃเธนเธเธกเธตเธเธเธฒเธ”เน€เธฅเนเธเน€เธเธดเธเนเธ (เธเธฑเนเธเธ•เนเธณเนเธเธฐเธเธณ 160x160)", []
    if brightness < 18:
        return "เธฃเธนเธเธกเธทเธ”เน€เธเธดเธเนเธเธเธเธฃเธฐเธเธเธเธฃเธฐเน€เธกเธดเธเนเธ”เนเนเธกเนเนเธกเนเธเธขเธณ", []
    if brightness > 248:
        return "เธฃเธนเธเธชเธงเนเธฒเธเธเนเธฒเน€เธเธดเธเนเธเธเธเธฃเธฒเธขเธฅเธฐเน€เธญเธตเธขเธ”เธซเธฒเธข", []

    notes = []
    if brightness < 45:
        notes.append("เธ เธฒเธเธเนเธญเธเธเนเธฒเธเธกเธทเธ”")
    if brightness > 220:
        notes.append("เธ เธฒเธเธเนเธญเธเธเนเธฒเธเธชเธงเนเธฒเธเธเนเธฒ")
    if contrast < 22:
        notes.append("เธเธญเธเธ—เธฃเธฒเธชเธ•เนเธ•เนเธณ เธญเธฒเธเนเธขเธเธเธธเธ”เน€เธชเธตเธขเธซเธฒเธขเนเธ”เนเธขเธฒเธ")
    if edge_strength < 10:
        notes.append("เธ เธฒเธเธญเธฒเธเน€เธเธฅเธญเธซเธฃเธทเธญเธชเธฑเนเธ")

    return None, notes


def get_last_conv_layer_name():
    current_model = get_model()
    for layer in reversed(current_model.layers):
        out = getattr(layer, "output_shape", None)
        if out is not None and isinstance(out, tuple) and len(out) == 4:
            return layer.name
    return None


def generate_heatmap_overlay(img_batch, class_index):
    current_model = get_model()
    layer_name = get_last_conv_layer_name()
    if layer_name is None:
        return None

    try:
        grad_model = tf.keras.models.Model(
            [current_model.inputs],
            [current_model.get_layer(layer_name).output, current_model.output],
        )
        with tf.GradientTape() as tape:
            conv_out, predictions = grad_model(img_batch)
            class_channel = predictions[:, class_index]

        grads = tape.gradient(class_channel, conv_out)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_out = conv_out[0]

        heatmap = tf.reduce_sum(conv_out * pooled_grads, axis=-1)
        heatmap = tf.maximum(heatmap, 0)
        max_val = tf.reduce_max(heatmap)
        if float(max_val) == 0.0:
            return None

        heatmap = heatmap / max_val
        heatmap = np.array(heatmap * 255, dtype=np.uint8)

        base = np.array((img_batch[0] * 255).clip(0, 255), dtype=np.uint8)
        heat = np.array(Image.fromarray(heatmap).resize((base.shape[1], base.shape[0])), dtype=np.uint8)

        overlay = np.zeros_like(base)
        overlay[..., 0] = heat
        overlay[..., 1] = (heat * 0.45).astype(np.uint8)

        blend = (base * 0.62 + overlay * 0.38).clip(0, 255).astype(np.uint8)
        return Image.fromarray(blend)
    except Exception as err:
        app.logger.warning("heatmap generation failed: %s", err)
        return None


def confidence_policy(confidence):
    if confidence >= 85:
        return {
            "tier": "เธเธฃเนเธญเธกเนเธเนเธ•เธฑเธ”เธชเธดเธเนเธเน€เธเธทเนเธญเธเธ•เนเธ",
            "advice": "เนเธเนเธเธฅเธเธตเนเธเนเธงเธขเธเธฃเธฐเน€เธกเธดเธเธเนเธฒเน€เธชเธตเธขเธซเธฒเธขเน€เธเธทเนเธญเธเธ•เนเธเนเธ”เน เนเธ•เนเธขเธฑเธเธเธงเธฃเธกเธตเธเธฒเธฃเธ•เธฃเธงเธเธซเธเนเธฒเธเธฒเธ",
            "flag": "high",
        }
    if confidence >= 65:
        return {
            "tier": "เธเธงเธฃเธขเธทเธเธขเธฑเธเธเนเธณ",
            "advice": "เนเธเนเธเธฅเธเธตเนเธฃเนเธงเธกเธเธฑเธเธเธฒเธฃเธ•เธฃเธงเธเธ”เนเธงเธขเธ•เธฒเนเธฅเธฐเธ เธฒเธเน€เธเธดเนเธกเน€เธ•เธดเธกเธเนเธญเธเธเธฃเธฐเน€เธกเธดเธเธเนเธฒเนเธเนเธเนเธฒเธข",
            "flag": "medium",
        }
    return {
        "tier": "เธเธงเธฒเธกเนเธกเนเนเธเนเธเธญเธเธชเธนเธ",
        "advice": "เนเธกเนเธเธงเธฃเนเธเนเธเธฅเธเธตเนเน€เธ”เธตเนเธขเธงเน เนเธเธฐเธเธณเธ–เนเธฒเธขเธ เธฒเธเนเธซเธกเนเธซเธฃเธทเธญเนเธซเนเธเนเธฒเธเธ•เธฃเธงเธเธชเธญเธเธเนเธญเธเธชเธฃเธธเธ",
        "flag": "low",
    }


def estimate_cost_range(part, level):
    base = {
        "low": (1500, 4500),
        "medium": (4500, 12000),
        "high": (12000, 35000),
    }.get(level, (2000, 8000))
    part_multiplier = {
        "เนเธเธซเธเนเธฒ": 1.15,
        "เธเธฃเธฐเธเธเธฃเธ–": 1.25,
        "เธเธฑเธเธเธเธซเธเนเธฒ": 1.0,
        "เธเธฑเธเธเธเธซเธฅเธฑเธ": 1.0,
        "เธเธฃเธฐเธ•เธน": 1.1,
        "เธเธฒเธเธฃเธฐเนเธเธฃเธเธซเธเนเธฒ": 1.2,
        "เนเธเนเธกเธเนเธฒเธเธฃเธ–": 1.05,
    }.get(part, 1.0)
    return int(base[0] * part_multiplier), int(base[1] * part_multiplier)


def _normalize_ratio_map(ratios):
    total = float(sum(max(0.0, float(v)) for v in ratios.values()))
    if total <= 0.0:
        equal = 1.0 / max(1, len(ratios))
        return {k: equal for k in ratios}
    return {k: max(0.0, float(v)) / total for k, v in ratios.items()}


def _split_range_by_ratios(cost_min, cost_max, ratios):
    keys = list(ratios.keys())
    min_parts = {k: int(round(cost_min * ratios[k])) for k in keys}
    max_parts = {k: int(round(cost_max * ratios[k])) for k in keys}
    min_delta = int(cost_min - sum(min_parts.values()))
    max_delta = int(cost_max - sum(max_parts.values()))
    if keys:
        min_parts[keys[0]] += min_delta
        max_parts[keys[0]] += max_delta
    return min_parts, max_parts


def build_cost_breakdown(part, level, cost_min, cost_max, confidence=None, score_gap=None, uncertainty=None):
    base_ratios = {
        "low": {
            "parts": 0.18,
            "labor": 0.33,
            "paint": 0.31,
            "calibration": 0.06,
            "misc": 0.12,
        },
        "medium": {
            "parts": 0.29,
            "labor": 0.30,
            "paint": 0.21,
            "calibration": 0.08,
            "misc": 0.12,
        },
        "high": {
            "parts": 0.42,
            "labor": 0.27,
            "paint": 0.14,
            "calibration": 0.07,
            "misc": 0.10,
        },
    }.get(level, {"parts": 0.28, "labor": 0.30, "paint": 0.22, "calibration": 0.08, "misc": 0.12})

    part_adjustments = {
        "ไฟหน้า": {"parts": 0.08, "paint": -0.06, "labor": -0.02},
        "กระจกรถ": {"parts": 0.14, "paint": -0.14},
        "กันชนหน้า": {"paint": 0.04, "parts": -0.03, "misc": -0.01},
        "กันชนหลัง": {"paint": 0.03, "parts": -0.02, "misc": -0.01},
        "ประตู": {"labor": 0.04, "parts": 0.02, "misc": -0.06},
        "ฝากระโปรงหน้า": {"parts": 0.03, "labor": 0.03, "misc": -0.06},
        "แก้มข้างรถ": {"labor": 0.05, "paint": 0.02, "misc": -0.07},
    }.get(part, {})

    adjusted = dict(base_ratios)
    for key, delta in part_adjustments.items():
        adjusted[key] = adjusted.get(key, 0.0) + float(delta)
    ratios = _normalize_ratio_map(adjusted)
    min_parts, max_parts = _split_range_by_ratios(cost_min, cost_max, ratios)

    item_defs = [
        ("parts", "ค่าอะไหล่"),
        ("labor", "ค่าแรงซ่อม/รื้อประกอบ"),
        ("paint", "ค่าทำสี/เก็บผิว"),
        ("calibration", "ค่าตรวจตั้ง/คาลิเบรต"),
        ("misc", "ค่าใช้จ่ายจิปาถะ"),
    ]
    items = []
    for key, label in item_defs:
        items.append(
            {
                "key": key,
                "label": label,
                "ratio_pct": round(ratios.get(key, 0.0) * 100, 1),
                "cost_min": int(min_parts.get(key, 0)),
                "cost_max": int(max_parts.get(key, 0)),
            }
        )

    expected = int(round((cost_min + cost_max) / 2.0))
    fast_track = int(round(expected * 1.1))
    planned = int(round(expected * 0.94))

    factors = [
        "ราคานี้เป็นการประเมินจากภาพและประเภทความเสียหาย ไม่ใช่ใบเสนอราคาจริงจากอู่",
        "ราคาอาจเปลี่ยนตามยี่ห้ออะไหล่ รุ่นรถ และเรทค่าแรงของพื้นที่",
    ]
    if confidence is not None:
        factors.append(f"ความมั่นใจของโมเดล: {float(confidence):.2f}%")
    if score_gap is not None:
        factors.append(f"ช่องว่างคะแนนอันดับ 1-2: {float(score_gap):.2f}%")
    if uncertainty:
        factors.append(f"ระดับความไม่แน่นอน: {uncertainty}")

    return {
        "cost_min": int(cost_min),
        "cost_max": int(cost_max),
        "expected_cost": expected,
        "scenarios": {
            "economy": int(round(cost_min)),
            "planned": planned,
            "fast_track": fast_track,
            "worst_case": int(round(cost_max)),
        },
        "items": items,
        "factors": factors,
    }


def build_assessment_options(part, raw_scores):
    top_indices = np.argsort(raw_scores)[::-1][:TOP_K_OUTCOMES]
    options = []
    weighted_cost = 0.0
    for idx in top_indices:
        level = labels[int(idx)]
        probability = float(raw_scores[int(idx)])
        score_pct = round(probability * 100, 2)
        cost_min, cost_max = estimate_cost_range(part, level)
        detail = RULES.get(part, {}).get(
            level,
            {"description": "เนเธกเนเธกเธตเธเนเธญเธกเธนเธฅ", "repair": "เนเธกเนเธกเธตเธเนเธญเธกเธนเธฅ"},
        )
        expected_cost = (cost_min + cost_max) / 2.0
        weighted_cost += expected_cost * probability
        options.append(
            {
                "label": level,
                "score": score_pct,
                "probability": round(probability, 4),
                "cost_min": cost_min,
                "cost_max": cost_max,
                "cost_breakdown": build_cost_breakdown(part, level, cost_min, cost_max),
                "description": detail["description"],
                "repair": detail["repair"],
            }
        )

    top_gap = 0.0
    if len(options) >= 2:
        top_gap = round(options[0]["score"] - options[1]["score"], 2)
    uncertainty = "high" if top_gap < 12 else "medium" if top_gap < 25 else "low"
    return options, int(round(weighted_cost)), top_gap, uncertainty


def prediction_entropy(raw_scores):
    p = np.clip(np.array(raw_scores, dtype=np.float64), 1e-9, 1.0)
    p = p / p.sum()
    return float(-(p * np.log(p)).sum())


def clamp01(val):
    return max(0.0, min(1.0, float(val)))


def compute_image_advanced_metrics(img_pil):
    arr = np.array(img_pil.convert("RGB"), dtype=np.float32)
    h, w = arr.shape[:2]
    gray = arr.mean(axis=2)

    brightness = float(gray.mean())
    contrast = float(gray.std())

    dx = np.abs(np.diff(gray, axis=1))
    dy = np.abs(np.diff(gray, axis=0))
    grad_mean = float(((dx.mean() if dx.size else 0.0) + (dy.mean() if dy.size else 0.0)) / 2.0)

    # Cheap Laplacian proxy for focus/sharpness without extra deps.
    c = gray[1:-1, 1:-1]
    lap = (
        -4.0 * c
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
        + gray[:-2, 1:-1]
        + gray[2:, 1:-1]
    ) if min(h, w) >= 3 else np.array([0.0], dtype=np.float32)
    sharpness = float(np.var(lap))

    edge_strength = (np.abs(lap) if isinstance(lap, np.ndarray) else np.array([0.0], dtype=np.float32))
    edge_threshold = float(np.percentile(edge_strength, 75)) if edge_strength.size else 0.0
    edge_density = float((edge_strength > edge_threshold).mean()) if edge_strength.size else 0.0

    ch1 = h // 4
    cw1 = w // 4
    center = edge_strength if min(h, w) < 4 else np.abs(
        (
            -4.0 * gray[ch1 : h - ch1, cw1 : w - cw1]
            + gray[ch1 : h - ch1, cw1 - 1 : w - cw1 - 1]
            + gray[ch1 : h - ch1, cw1 + 1 : w - cw1 + 1]
            + gray[ch1 - 1 : h - ch1 - 1, cw1 : w - cw1]
            + gray[ch1 + 1 : h - ch1 + 1, cw1 : w - cw1]
        )
    )
    center_edge_density = float((center > edge_threshold).mean()) if center.size else edge_density

    # Normalized quality score [0,100]
    exposure = 1.0 - clamp01(abs(brightness - 128.0) / 128.0)
    contrast_n = clamp01(contrast / 64.0)
    sharpness_n = clamp01(sharpness / 300.0)
    edges_n = clamp01(edge_density / 0.28)
    quality_score = round(
        (0.30 * exposure + 0.25 * contrast_n + 0.30 * sharpness_n + 0.15 * edges_n) * 100.0,
        2,
    )

    return {
        "resolution": f"{w}x{h}",
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
        "gradient_strength": round(grad_mean, 3),
        "sharpness": round(sharpness, 3),
        "edge_density": round(edge_density, 4),
        "center_edge_density": round(center_edge_density, 4),
        "quality_score": quality_score,
    }


def build_ai_insights(result, confidence, score_gap, entropy, quality_metrics, domain_gate):
    sev_weight = {"low": 0.22, "medium": 0.56, "high": 0.9}.get((result or "").lower(), 0.5)
    uncertainty_penalty = clamp01(entropy / 1.1) * 35.0 + clamp01((20.0 - score_gap) / 20.0) * 25.0
    confidence_bonus = clamp01(confidence / 100.0) * 25.0
    quality_bonus = clamp01((quality_metrics.get("quality_score", 0.0) / 100.0)) * 15.0
    reliability_score = round(max(0.0, min(100.0, confidence_bonus + quality_bonus + (25.0 - uncertainty_penalty))), 2)
    severity_index = round(min(100.0, max(0.0, sev_weight * 100.0 + clamp01(confidence / 100.0) * 8.0)), 2)

    if severity_index >= 78:
        urgency = "urgent"
        urgency_label = "เน€เธฃเนเธเธ”เนเธงเธ"
    elif severity_index >= 52:
        urgency = "moderate"
        urgency_label = "เธเธงเธฃเธ”เธณเน€เธเธดเธเธเธฒเธฃเน€เธฃเนเธง"
    else:
        urgency = "normal"
        urgency_label = "เธ•เธดเธ”เธ•เธฒเธกเนเธ”เน"

    actions = []
    if domain_gate.get("mode") == "model" and not domain_gate.get("is_car"):
        actions.append("เธ•เธฃเธงเธเธงเนเธฒเน€เธเนเธเธ เธฒเธเธฃเธ–เธเธฃเธดเธเธเนเธญเธเธชเนเธเน€เธเนเธฒเธเธฃเธฐเน€เธกเธดเธ")
    if reliability_score < 45:
        actions.append("เธ–เนเธฒเธขเธ เธฒเธเนเธซเธกเนเนเธเนเธชเธเธเธฃเธฃเธกเธเธฒเธ•เธดเนเธฅเธฐเนเธซเนเธญเธขเธนเนเนเธเธฅเนเธเธธเธ”เน€เธชเธตเธขเธซเธฒเธขเธกเธฒเธเธเธถเนเธ")
    if entropy > 0.9 or score_gap < 15:
        actions.append("เธเธฅเธขเธฑเธเนเธกเนเน€เธชเธ–เธตเธขเธฃ เนเธเธฐเธเธณเนเธซเนเธ•เธฃเธงเธเธเนเธณเธซเธฃเธทเธญเธชเนเธ reviewer เธขเธทเธเธขเธฑเธ")
    if (quality_metrics.get("edge_density") or 0) < 0.08:
        actions.append("เธ เธฒเธเธญเธฒเธเน€เธเธฅเธญ เนเธซเนเธ–เธทเธญเธเธฅเนเธญเธเธเธดเนเธเธเธถเนเธเธซเธฃเธทเธญเน€เธเธดเนเธกเธเธงเธฒเธกเธเธกเธเธฑเธ”")
    if not actions:
        actions.append("เธชเธฒเธกเธฒเธฃเธ–เนเธเนเธเธฅเธเธตเนเน€เธเนเธ baseline estimate เนเธ”เน เนเธฅเธฐเธ•เธฃเธงเธเธซเธเนเธฒเธเธฒเธเธเนเธญเธเธเธดเธ”เธเธฒเธ")

    return {
        "reliability_score": reliability_score,
        "severity_index": severity_index,
        "urgency": urgency,
        "urgency_label": urgency_label,
        "actions": actions[:4],
    }


def summarize_multi_angle(predictions):
    if not predictions:
        return {"image_count": 0, "consistency_score": 0.0, "agreement_ratio": 0.0, "angles": []}
    arr = np.array(predictions, dtype=np.float64)
    labels_idx = np.argmax(arr, axis=1)
    counts = np.bincount(labels_idx, minlength=arr.shape[1])
    agreement_ratio = float(np.max(counts) / max(1, len(labels_idx)))
    mean_vec = arr.mean(axis=0)
    dispersion = float(np.mean(np.linalg.norm(arr - mean_vec, axis=1)))
    consistency_score = round(max(0.0, min(100.0, agreement_ratio * 70.0 + (1.0 - min(dispersion, 1.0)) * 30.0)) * 100.0 / 100.0, 2)

    angles = []
    for i, row in enumerate(arr, start=1):
        idx = int(np.argmax(row))
        score_map = {labels[j]: round(float(row[j]) * 100.0, 2) for j in range(arr.shape[1])}
        angles.append(
            {
                "index": i,
                "top_label": labels[idx],
                "top_score": round(float(row[idx]) * 100.0, 2),
                "scores": score_map,
            }
        )
    return {
        "image_count": int(len(predictions)),
        "consistency_score": consistency_score,
        "agreement_ratio": round(agreement_ratio, 4),
        "angles": angles,
    }


def normalize_summary_tone(value):
    tone = (value or "").strip().lower()
    if tone in {"customer", "technical", "insurance"}:
        return tone
    return "customer"


def build_incident_summary(
    part,
    result,
    confidence,
    est_min,
    est_max,
    ai_insights,
    multi_angle,
    quality_metrics,
    tone="customer",
):
    image_count = int((multi_angle or {}).get("image_count", 1))
    consistency = float((multi_angle or {}).get("consistency_score", 0.0))
    urgency = (ai_insights or {}).get("urgency_label", "เธ•เธดเธ”เธ•เธฒเธกเนเธ”เน")
    reliability = float((ai_insights or {}).get("reliability_score", 0.0))
    quality_score = float((quality_metrics or {}).get("quality_score", 0.0))
    mode = "เธซเธฅเธฒเธขเธกเธธเธก" if image_count > 1 else "เธกเธธเธกเน€เธ”เธตเธขเธง"
    tone = normalize_summary_tone(tone)
    if tone == "technical":
        return (
            f"เธชเธฃเธธเธเน€เธเธดเธเน€เธ—เธเธเธดเธ: part={part}, class={result}, conf={confidence:.2f}%, "
            f"images={image_count} ({mode}), consistency={consistency:.2f}%, reliability={reliability:.2f}%, "
            f"quality_score={quality_score:.2f}, cost_range={est_min:,}-{est_max:,} THB, urgency={urgency}."
        )
    if tone == "insurance":
        return (
            f"เธชเธฃเธธเธเธชเธณเธซเธฃเธฑเธเธเธฒเธเน€เธเธฅเธก: เธ•เธฃเธงเธเธเธเธเธงเธฒเธกเน€เธชเธตเธขเธซเธฒเธขเธเธฃเธดเน€เธงเธ“ {part} เธฃเธฐเธ”เธฑเธ {result} เธเธฒเธเธ เธฒเธ{mode} {image_count} เธ เธฒเธ "
            f"(เธเธงเธฒเธกเธชเธญเธ”เธเธฅเนเธญเธ {consistency:.2f}%, เธเธงเธฒเธกเธเนเธฒเน€เธเธทเนเธญเธ–เธทเธญ {reliability:.2f}%). "
            f"เธเธฃเธฐเน€เธกเธดเธเธเนเธฒเนเธเนเธเนเธฒเธขเน€เธเธทเนเธญเธเธ•เนเธ {est_min:,}-{est_max:,} เธเธฒเธ— เนเธฅเธฐเธเธฑเธ”เธฃเธฐเธ”เธฑเธเธเธงเธฒเธกเน€เธฃเนเธเธ”เนเธงเธเน€เธเนเธ \"{urgency}\"."
        )
    return (
        f"เธชเธฃเธธเธเน€เธซเธ•เธธเธเธฒเธฃเธ“เน: เธฃเธฐเธเธเธเธฃเธฐเน€เธกเธดเธเธเธงเธฒเธกเน€เธชเธตเธขเธซเธฒเธขเธ•เธณเนเธซเธเนเธ {part} เธญเธขเธนเนเนเธเธฃเธฐเธ”เธฑเธ {result} "
        f"(confidence {confidence:.2f}%) เธเธฒเธเธ เธฒเธ{mode}เธเธณเธเธงเธ {image_count} เธ เธฒเธ "
        f"เนเธ”เธขเธกเธตเธเธงเธฒเธกเธชเธญเธ”เธเธฅเนเธญเธเธฃเธฐเธซเธงเนเธฒเธเธกเธธเธกเธกเธญเธ {consistency:.2f}% เนเธฅเธฐเธเธงเธฒเธกเธเนเธฒเน€เธเธทเนเธญเธ–เธทเธญเธเธฅเธฅเธฑเธเธเน {reliability:.2f}%. "
        f"เธเธธเธ“เธ เธฒเธเธ เธฒเธเธฃเธงเธก {quality_score:.2f} เธเธฐเนเธเธ; เธเธฃเธฐเธกเธฒเธ“เธเนเธฒเนเธเนเธเนเธฒเธข {est_min:,}-{est_max:,} เธเธฒเธ—. "
        f"เธฃเธฐเธ”เธฑเธเธเธงเธฒเธกเน€เธฃเนเธเธ”เนเธงเธ: {urgency}."
    )


def incident_summary_from_history_row(row, tone="customer"):
    confidence = float(row.get("confidence") or 0.0)
    result = (row.get("result") or "").lower()
    urgency_label = (
        "เน€เธฃเนเธเธ”เนเธงเธ" if result == "high" else "เธเธงเธฃเธ”เธณเน€เธเธดเธเธเธฒเธฃเน€เธฃเนเธง" if result == "medium" else "เธ•เธดเธ”เธ•เธฒเธกเนเธ”เน"
    )
    ai_insights = {
        "reliability_score": max(0.0, min(100.0, confidence)),
        "urgency_label": urgency_label,
    }
    est_min = int(row.get("est_cost_min") or 0)
    est_max = int(row.get("est_cost_max") or 0)
    if est_min <= 0 and est_max <= 0:
        est_min, est_max = estimate_cost_range(row.get("part"), result or "medium")

    return build_incident_summary(
        part=row.get("part") or "-",
        result=result or "unknown",
        confidence=confidence,
        est_min=est_min,
        est_max=est_max,
        ai_insights=ai_insights,
        multi_angle={"image_count": 1, "consistency_score": 100.0},
        quality_metrics={"quality_score": 70.0},
        tone=tone,
    )


def evaluate_vehicle_domain(img_batch):
    gate_model = get_non_car_model()
    if gate_model is None:
        return {"mode": "rule_fallback", "enabled": NON_CAR_GUARD_ENABLED}

    try:
        pred = gate_model.predict(img_batch, verbose=0)
        values = np.ravel(pred).astype(np.float64)
        if values.size == 0:
            return {"mode": "model_error", "error": "empty_prediction"}
        if values.size == 1:
            car_prob = float(np.clip(values[0], 0.0, 1.0))
        else:
            idx = NON_CAR_MODEL_CAR_INDEX if 0 <= NON_CAR_MODEL_CAR_INDEX < values.size else 0
            if np.min(values) < 0 or np.max(values) > 1.0:
                exps = np.exp(values - np.max(values))
                probs = exps / np.sum(exps)
                car_prob = float(probs[idx])
            else:
                probs = values / max(np.sum(values), 1e-9)
                car_prob = float(probs[idx])
        return {
            "mode": "model",
            "car_probability": round(car_prob, 4),
            "min_car_probability": NON_CAR_MODEL_MIN_CAR_PROB,
            "is_car": car_prob >= NON_CAR_MODEL_MIN_CAR_PROB,
        }
    except Exception as err:
        app.logger.warning("vehicle gate prediction failed: %s", err)
        return {"mode": "model_error", "error": str(err)}


def build_tta_batches(img_pil):
    resized = img_pil.resize((224, 224))
    base = np.array(resized, dtype=np.float32) / 255.0
    variants = [base]
    if not TTA_ENABLED:
        return [np.expand_dims(base, axis=0)]

    if TTA_VARIANTS >= 2:
        variants.append(np.flip(base, axis=1))
    if TTA_VARIANTS >= 3:
        variants.append(np.clip(base * 1.08, 0.0, 1.0))
    if TTA_VARIANTS >= 4:
        variants.append(np.clip(base * 0.92, 0.0, 1.0))
    if TTA_VARIANTS >= 5:
        variants.append(np.clip((base - 0.5) * 1.1 + 0.5, 0.0, 1.0))

    return [np.expand_dims(v, axis=0) for v in variants]


def should_route_manual_review(confidence, score_gap, entropy, domain_suspicious=False, tta_dispersion=0.0):
    return (
        score_gap < MANUAL_REVIEW_SCORE_GAP
        or entropy > MANUAL_REVIEW_MAX_ENTROPY
        or confidence < 65
        or bool(domain_suspicious)
        or float(tta_dispersion) > MANUAL_REVIEW_MAX_TTA_DISPERSION
    )


def calibrate_damage_level(level, confidence, score_gap, entropy):
    lv = (level or "").lower()
    note = None
    if lv == "high" and (confidence < 82 or score_gap < 22 or entropy > 0.72):
        note = "เธเธฅ high เธ–เธนเธเธเธฃเธฑเธเน€เธเนเธ medium เน€เธเธฃเธฒเธฐเธเธงเธฒเธกเน€เธเธทเนเธญเธกเธฑเนเธ/เธเธงเธฒเธกเธเธฑเธ”เน€เธเธเธขเธฑเธเนเธกเนเธเธญ"
        return "medium", note
    if lv == "medium" and (confidence < 56 and score_gap < 12 and entropy > 0.92):
        note = "เธเธฅ medium เธ–เธนเธเธเธฃเธฑเธเน€เธเนเธ low เน€เธเธฃเธฒเธฐเธเธงเธฒเธกเนเธกเนเนเธเนเธเธญเธเธชเธนเธ"
        return "low", note
    return lv, note


def run_analysis_pipeline(user, part, file_storage, extra_files=None, summary_tone="customer"):
    if not part or not file_storage:
        return None, "missing part or file"
    if part not in RULES:
        return None, "invalid part"

    files = [file_storage] + [f for f in (extra_files or []) if f is not None]
    valid_files = [f for f in files if (getattr(f, "filename", "") or "").strip()]
    if not valid_files:
        return None, "missing part or file"
    if len(valid_files) > MAX_MULTI_IMAGES:
        return None, f"เธฃเธญเธเธฃเธฑเธเธเธฒเธฃเธงเธดเน€เธเธฃเธฒเธฐเธซเนเธซเธฅเธฒเธขเธกเธธเธกเธชเธนเธเธชเธธเธ” {MAX_MULTI_IMAGES} เธฃเธนเธเธ•เนเธญเธเธฃเธฑเนเธ"

    current_model = get_model()
    per_image_predictions = []
    per_image_batches = []
    per_image_pils = []
    per_image_metrics = []
    per_image_tta_dispersion = []
    quality_notes = []
    representative_domain_gate = None
    domain_suspicious = False
    domain_suspicion_notes = []

    for i, f in enumerate(valid_files, start=1):
        upload_error = validate_upload(f)
        if upload_error:
            return None, f"เธ เธฒเธเธ—เธตเน {i}: {upload_error}"
        img_pil = Image.open(f.stream).convert("RGB")
        quality_error, notes = assess_image_quality(img_pil)
        if quality_error:
            return None, f"เธ เธฒเธเธ—เธตเน {i}: {quality_error}"
        quality_notes.extend(notes or [])
        metrics = compute_image_advanced_metrics(img_pil)
        tta_batches = build_tta_batches(img_pil)
        img_batch = tta_batches[0]

        domain_gate = evaluate_vehicle_domain(img_batch)
        if representative_domain_gate is None:
            representative_domain_gate = domain_gate
        elif domain_gate.get("mode") == "model":
            if representative_domain_gate.get("mode") != "model":
                representative_domain_gate = domain_gate
            else:
                rep_prob = float(representative_domain_gate.get("car_probability") or 1.0)
                cur_prob = float(domain_gate.get("car_probability") or 1.0)
                if cur_prob < rep_prob:
                    representative_domain_gate = domain_gate
        if domain_gate.get("mode") == "model" and not domain_gate.get("is_car"):
            car_prob = float(domain_gate.get("car_probability") or 0.0)
            if car_prob <= NON_CAR_HARD_BLOCK_MAX_CAR_PROB:
                return (
                    None,
                    f"เธ เธฒเธเธ—เธตเน {i}: เธฃเธฐเธเธเธ•เธฃเธงเธเธเธเธงเนเธฒเธญเธฒเธเนเธกเนเนเธเนเธ เธฒเธเธฃเธ– เธเธฃเธธเธ“เธฒเธญเธฑเธเนเธซเธฅเธ”เธ เธฒเธเธฃเธ–เธ—เธตเนเน€เธซเนเธเธเธดเนเธเธชเนเธงเธเธเธฑเธ”เน€เธเธ",
                )
            domain_suspicious = True
            domain_suspicion_notes.append(
                f"เธ เธฒเธเธ—เธตเน {i} เธญเธฒเธเนเธกเนเน€เธซเนเธเธเธดเนเธเธชเนเธงเธเธฃเธ–เธเธฑเธ” (car probability {car_prob * 100:.1f}%) เธฃเธฐเธเธเธเธฐเธชเนเธเนเธซเนเธเธเธเนเธงเธขเธฃเธตเธงเธดเธง"
            )

        tta_scores = []
        for tta_batch in tta_batches:
            pred = current_model.predict(tta_batch, verbose=0)
            tta_scores.append(np.array(pred[0], dtype=np.float64))
        tta_arr = np.stack(tta_scores, axis=0)
        raw_scores = np.mean(tta_arr, axis=0)
        dispersion = float(np.mean(np.linalg.norm(tta_arr - raw_scores, axis=1)))
        per_image_tta_dispersion.append(dispersion)
        per_image_predictions.append(raw_scores)
        per_image_batches.append(img_batch)
        per_image_pils.append(img_pil)
        per_image_metrics.append(metrics)

    raw_scores = np.mean(np.stack(per_image_predictions, axis=0), axis=0)
    idx = int(np.argmax(raw_scores))
    confidence = round(float(raw_scores[idx]) * 100, 2)
    level = labels[idx]
    domain_gate = representative_domain_gate or {"mode": "rule_fallback", "enabled": NON_CAR_GUARD_ENABLED}
    unique_quality_notes = []
    for note in quality_notes:
        if note not in unique_quality_notes:
            unique_quality_notes.append(note)
    for note in domain_suspicion_notes:
        if note not in unique_quality_notes:
            unique_quality_notes.append(note)
    quality_notes = unique_quality_notes
    if per_image_metrics:
        quality_metrics = {
            "resolution": per_image_metrics[0]["resolution"],
            "brightness": round(float(np.mean([m["brightness"] for m in per_image_metrics])), 2),
            "contrast": round(float(np.mean([m["contrast"] for m in per_image_metrics])), 2),
            "gradient_strength": round(float(np.mean([m["gradient_strength"] for m in per_image_metrics])), 3),
            "sharpness": round(float(np.mean([m["sharpness"] for m in per_image_metrics])), 3),
            "edge_density": round(float(np.mean([m["edge_density"] for m in per_image_metrics])), 4),
            "center_edge_density": round(float(np.mean([m["center_edge_density"] for m in per_image_metrics])), 4),
            "quality_score": round(float(np.mean([m["quality_score"] for m in per_image_metrics])), 2),
            "tta_dispersion": round(float(np.mean(per_image_tta_dispersion)) if per_image_tta_dispersion else 0.0, 4),
        }
    else:
        quality_metrics = {
            "resolution": "-",
            "brightness": 0,
            "contrast": 0,
            "gradient_strength": 0,
            "sharpness": 0,
            "edge_density": 0,
            "center_edge_density": 0,
            "quality_score": 0,
            "tta_dispersion": 0.0,
        }

    evidence_items = [
        {"label": labels[j], "score": round(float(raw_scores[j]) * 100, 2)}
        for j in np.argsort(raw_scores)[::-1][:TOP_K_OUTCOMES]
    ]
    assessment_options, weighted_cost, score_gap, uncertainty = build_assessment_options(
        part, raw_scores
    )
    entropy = prediction_entropy(raw_scores)
    tta_dispersion = float(quality_metrics.get("tta_dispersion") or 0.0)
    rule_risk = 0
    if confidence < NON_CAR_MIN_CONFIDENCE:
        rule_risk += 1
    if score_gap < NON_CAR_MIN_SCORE_GAP:
        rule_risk += 1
    if entropy > NON_CAR_MAX_ENTROPY:
        rule_risk += 1
    if confidence < max(30.0, NON_CAR_MIN_CONFIDENCE - 15.0):
        rule_risk += 1
    if score_gap < max(3.0, NON_CAR_MIN_SCORE_GAP * 0.6):
        rule_risk += 1
    if entropy > min(1.2, NON_CAR_MAX_ENTROPY + 0.12):
        rule_risk += 1
    if tta_dispersion > MANUAL_REVIEW_MAX_TTA_DISPERSION:
        rule_risk += 1

    should_block_by_rules = (
        NON_CAR_GUARD_ENABLED
        and domain_gate.get("mode") != "model"
        and rule_risk >= NON_CAR_RULE_BLOCK_MIN_RISK
    )
    if should_block_by_rules:
        return (
            None,
            "เนเธกเนเธชเธฒเธกเธฒเธฃเธ–เธขเธทเธเธขเธฑเธเธงเนเธฒเน€เธเนเธเธ เธฒเธเธเธงเธฒเธกเน€เธชเธตเธขเธซเธฒเธขเธเธญเธเธฃเธ–เนเธ”เน เธเธฃเธธเธ“เธฒเธญเธฑเธเนเธซเธฅเธ”เธ เธฒเธเธฃเธ–เธ—เธตเนเน€เธซเนเธเธเธดเนเธเธชเนเธงเธเธเธฑเธ”เน€เธเธ",
        )
    if NON_CAR_GUARD_ENABLED and not should_block_by_rules and rule_risk >= 3:
        domain_suspicious = True
        if "เธ เธฒเธเธเนเธณเธเธฑเนเธเธญเธฒเธเธ—เธณเนเธซเนเธเธฅเธเธฅเธฒเธ”เน€เธเธฅเธทเนเธญเธ เธฃเธฐเธเธเธเธฐเธชเนเธเธฃเธตเธงเธดเธงเน€เธเธดเนเธก" not in quality_notes:
            quality_notes.append("เธ เธฒเธเธเนเธณเธเธฑเนเธเธญเธฒเธเธ—เธณเนเธซเนเธเธฅเธเธฅเธฒเธ”เน€เธเธฅเธทเนเธญเธ เธฃเธฐเธเธเธเธฐเธชเนเธเธฃเธตเธงเธดเธงเน€เธเธดเนเธก")
    if tta_dispersion > MANUAL_REVIEW_MAX_TTA_DISPERSION:
        quality_notes.append(
            f"TTA dispersion สูง ({tta_dispersion:.4f}) อาจเป็นเคสกำกวม แนะนำให้ผู้เชี่ยวชาญตรวจทาน"
        )

    adjusted_level, calibration_note = calibrate_damage_level(level, confidence, score_gap, entropy)
    if adjusted_level != level:
        level = adjusted_level
    policy = confidence_policy(confidence)
    ai_insights = build_ai_insights(
        result=level,
        confidence=confidence,
        score_gap=score_gap,
        entropy=entropy,
        quality_metrics=quality_metrics,
        domain_gate=domain_gate,
    )
    detail = RULES.get(part, {}).get(
        level,
        {"description": "เนเธกเนเธกเธตเธเนเธญเธกเธนเธฅ", "repair": "เนเธกเนเธกเธตเธเนเธญเธกเธนเธฅ"},
    )
    est_min, est_max = estimate_cost_range(part, level)
    cost_breakdown = build_cost_breakdown(
        part,
        level,
        est_min,
        est_max,
        confidence=confidence,
        score_gap=score_gap,
        uncertainty=uncertainty,
    )
    multi_angle = summarize_multi_angle(per_image_predictions)
    normalized_tone = normalize_summary_tone(summary_tone)
    incident_summaries = {
        "customer": build_incident_summary(
            part=part,
            result=level,
            confidence=confidence,
            est_min=est_min,
            est_max=est_max,
            ai_insights=ai_insights,
            multi_angle=multi_angle,
            quality_metrics=quality_metrics,
            tone="customer",
        ),
        "technical": build_incident_summary(
            part=part,
            result=level,
            confidence=confidence,
            est_min=est_min,
            est_max=est_max,
            ai_insights=ai_insights,
            multi_angle=multi_angle,
            quality_metrics=quality_metrics,
            tone="technical",
        ),
        "insurance": build_incident_summary(
            part=part,
            result=level,
            confidence=confidence,
            est_min=est_min,
            est_max=est_max,
            ai_insights=ai_insights,
            multi_angle=multi_angle,
            quality_metrics=quality_metrics,
            tone="insurance",
        ),
    }
    incident_summary = incident_summaries.get(normalized_tone, incident_summaries["customer"])
    incident_summary_labels = {
        "customer": "เธฅเธนเธเธเนเธฒ",
        "technical": "เธเนเธฒเธ/เน€เธ—เธเธเธดเธ",
        "insurance": "เธเธฃเธฐเธเธฑเธ",
    }

    save_dir = os.path.join("feedback_images", level)
    os.makedirs(save_dir, exist_ok=True)

    filename = datetime.now().strftime("%Y%m%d_%H%M%S") + ".jpg"
    image_path = os.path.join(save_dir, filename).replace("\\", "/")
    representative_idx = int(
        np.argmax([float(pred[idx]) for pred in per_image_predictions])
    ) if per_image_predictions else 0
    rep_img = per_image_pils[representative_idx] if per_image_pils else Image.open(file_storage.stream).convert("RGB")
    rep_batch = per_image_batches[representative_idx] if per_image_batches else np.expand_dims(np.array(rep_img.resize((224, 224)), dtype=np.float32) / 255.0, axis=0)
    rep_img.save(image_path, format="JPEG")

    heatmap_path = None
    heatmap_img = generate_heatmap_overlay(rep_batch, idx)
    if heatmap_img is not None:
        heatmap_filename = filename.replace(".jpg", "_heatmap.jpg")
        heatmap_path = os.path.join(save_dir, heatmap_filename).replace("\\", "/")
        heatmap_img.save(heatmap_path, format="JPEG")

    add_history(
        user_id=user["id"],
        part=part,
        result=level,
        confidence=confidence,
        trust=policy["tier"],
        image_path=image_path,
        model_version=MODEL_VERSION,
        est_cost_min=est_min,
        est_cost_max=est_max,
    )
    create_notification(
        user["id"],
        "เธงเธดเน€เธเธฃเธฒเธฐเธซเนเน€เธชเธฃเนเธเนเธฅเนเธง",
        f"{part}: {level} ({confidence}%) เธเธฃเธฐเธกเธฒเธ“เธเนเธฒเนเธเนเธเนเธฒเธข {est_min:,}-{est_max:,} เธเธฒเธ—",
    )
    manual_review_reasons = []
    if score_gap < MANUAL_REVIEW_SCORE_GAP:
        manual_review_reasons.append(f"ช่องว่างคะแนนอันดับ 1-2 ต่ำกว่าเกณฑ์ ({score_gap:.2f}% < {MANUAL_REVIEW_SCORE_GAP:.2f}%)")
    if entropy > MANUAL_REVIEW_MAX_ENTROPY:
        manual_review_reasons.append(
            f"ความไม่แน่นอนของโมเดลสูง ({entropy:.4f} > {MANUAL_REVIEW_MAX_ENTROPY:.4f})"
        )
    if confidence < 65:
        manual_review_reasons.append(f"ความมั่นใจต่ำกว่าเกณฑ์ (confidence {confidence:.2f}% < 65%)")
    if tta_dispersion > MANUAL_REVIEW_MAX_TTA_DISPERSION:
        manual_review_reasons.append(
            f"ผลจาก TTA แกว่งสูง ({tta_dispersion:.4f} > {MANUAL_REVIEW_MAX_TTA_DISPERSION:.4f})"
        )
    if domain_suspicious:
        manual_review_reasons.append("ผ่านการตรวจรถแบบก้ำกึ่ง ระบบจึงให้คนตรวจทานยืนยันผล")

    needs_manual_review = should_route_manual_review(
        confidence,
        score_gap,
        entropy,
        domain_suspicious=domain_suspicious,
        tta_dispersion=tta_dispersion,
    )
    if needs_manual_review and not manual_review_reasons:
        manual_review_reasons.append("เงื่อนไขความเสี่ยงรวมเข้าเกณฑ์ตรวจทานโดยผู้เชี่ยวชาญ")
    review_case_id = None
    if needs_manual_review:
        review_case_id = create_case(
            user_id=user["id"],
            title=f"Auto review: {part} {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            vehicle_info=f"confidence={confidence}% entropy={round(entropy, 4)} gap={score_gap}",
            status="needs_review",
            predicted_result=level,
            predicted_confidence=confidence,
        )
        create_notification(
            user["id"],
            "เธชเนเธเน€เธเธชเนเธซเนเธเธนเนเธ•เธฃเธงเธเนเธฅเนเธง",
            f"เน€เธเธช #{review_case_id} เธ–เธนเธเธชเนเธเธ•เธฃเธงเธเธเนเธณเนเธ”เธข reviewer เน€เธเธทเนเธญเธเธเธฒเธเธเธฅเนเธกเนเนเธเนเธเธญเธ",
        )
        app.logger.info("manual review queued case_id=%s user=%s", review_case_id, user["name"])

    Thread(target=dispatch_retrain_check, daemon=True).start()

    return {
        "part": part,
        "result": level,
        "confidence": confidence,
        "trust": policy["tier"],
        "policy": policy,
        "decision": policy["advice"],
        "detail": detail,
        "quality_notes": quality_notes,
        "quality_metrics": quality_metrics,
        "ai_insights": ai_insights,
        "calibration_note": calibration_note,
        "multi_angle": multi_angle,
        "incident_summaries": incident_summaries,
        "incident_summary_labels": incident_summary_labels,
        "selected_summary_tone": normalized_tone,
        "incident_summary": incident_summary,
        "evidence_items": evidence_items,
        "assessment_options": assessment_options,
        "image_path": image_path,
        "heatmap_path": heatmap_path,
        "model_version": MODEL_VERSION,
        "est_cost_min": est_min,
        "est_cost_max": est_max,
        "expected_cost": weighted_cost,
        "cost_breakdown": cost_breakdown,
        "score_gap": score_gap,
        "prediction_entropy": round(entropy, 4),
        "uncertainty": uncertainty,
        "top_k_outcomes": TOP_K_OUTCOMES,
        "domain_gate": domain_gate,
        "diagnostics": {
            "non_car_rule_risk": rule_risk,
            "non_car_rule_block_threshold": NON_CAR_RULE_BLOCK_MIN_RISK,
            "domain_suspicious": domain_suspicious,
            "tta_dispersion": round(tta_dispersion, 4),
            "tta_enabled": TTA_ENABLED,
            "tta_variants": TTA_VARIANTS,
            "manual_review_reasons": manual_review_reasons,
            "thresholds": {
                "manual_review_score_gap": MANUAL_REVIEW_SCORE_GAP,
                "manual_review_max_entropy": MANUAL_REVIEW_MAX_ENTROPY,
                "manual_review_max_tta_dispersion": MANUAL_REVIEW_MAX_TTA_DISPERSION,
                "non_car_min_confidence": NON_CAR_MIN_CONFIDENCE,
                "non_car_min_score_gap": NON_CAR_MIN_SCORE_GAP,
                "non_car_max_entropy": NON_CAR_MAX_ENTROPY,
                "non_car_hard_block_max_car_probability": NON_CAR_HARD_BLOCK_MAX_CAR_PROB,
            },
        },
        "needs_manual_review": needs_manual_review,
        "review_case_id": review_case_id,
    }, None


def process_analysis_job_payload(payload):
    job_id = payload.get("job_id")
    user = payload.get("user")
    part = payload.get("part")
    file_path = payload.get("file_path")
    try:
        update_analysis_job(job_id, status="running")
        with open(file_path, "rb") as f:
            class LocalFile:
                filename = os.path.basename(file_path)
                mimetype = "image/jpeg"
                stream = f

            result, err = run_analysis_pipeline(user, part, LocalFile())
        if err:
            update_analysis_job(job_id, status="failed", error_text=err)
        else:
            update_analysis_job(
                job_id,
                status="done",
                image_path=result.get("image_path"),
                result_json=result,
            )
    except Exception as err:
        update_analysis_job(job_id, status="failed", error_text=str(err))
    finally:
        try:
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
        except OSError:
            pass


def analysis_worker_loop_local():
    while True:
        try:
            payload = ANALYSIS_JOB_QUEUE.get(timeout=1)
        except Empty:
            continue
        try:
            process_analysis_job_payload(payload)
        finally:
            ANALYSIS_JOB_QUEUE.task_done()


def analysis_worker_loop_redis():
    redis_client = get_redis_client()
    if redis_client is None:
        analysis_worker_loop_local()
        return
    while True:
        popped = redis_client.blpop(REDIS_QUEUE_KEY, timeout=1)
        if not popped:
            continue
        _, raw_payload = popped
        try:
            payload = json.loads(raw_payload)
        except Exception:
            continue
        process_analysis_job_payload(payload)


def ensure_worker_started():
    global WORKER_STARTED
    if not INLINE_WORKER_ENABLED:
        return
    if WORKER_STARTED:
        return
    with WORKER_LOCK:
        if WORKER_STARTED:
            return
        if get_redis_client() is not None:
            Thread(target=analysis_worker_loop_redis, daemon=True).start()
        else:
            Thread(target=analysis_worker_loop_local, daemon=True).start()
        WORKER_STARTED = True


def run_worker_forever():
    if get_redis_client() is not None:
        app.logger.info("starting dedicated worker backend=redis")
        analysis_worker_loop_redis()
        return
    app.logger.info("starting dedicated worker backend=local")
    analysis_worker_loop_local()


def dispatch_retrain_check():
    global _retrain_check_running
    with RETRAIN_CHECK_LOCK:
        if _retrain_check_running:
            return
        _retrain_check_running = True

    try:
        need_retrain, img_count = should_retrain()
        if need_retrain:
            subprocess.Popen(["python", "retrain.py"])
            save_status(img_count)
            app.logger.info("retrain dispatched with img_count=%s", img_count)
    except Exception as retrain_error:
        app.logger.warning("retrain check warning: %s", retrain_error)
    finally:
        with RETRAIN_CHECK_LOCK:
            _retrain_check_running = False


def get_session_user():
    user = session.get("user")
    if not user:
        return None

    if "id" in user and "name" in user and "role" in user:
        return user

    if "name" in user and "id" not in user:
        normalized = get_or_create_user(user["name"])
        session["user"] = normalized
        return normalized

    if "name" in user:
        auth_row = get_user_auth(user["name"])
        if auth_row:
            normalized = {
                "id": auth_row["id"],
                "name": auth_row["username"],
                "role": auth_row["role"] or "user",
                "is_active": int(auth_row["is_active"]) == 1,
                "avatar_path": auth_row.get("avatar_path"),
            }
            session["user"] = normalized
            return normalized

    return None


def is_admin_user(user):
    if not user:
        return False
    role = (user.get("role") or "").strip().lower()
    return role == "admin"


def is_reviewer_user(user):
    if not user:
        return False
    if is_admin_user(user):
        return True
    role = (user.get("role") or "").strip().lower()
    if role == "reviewer":
        return True
    return user.get("name", "").strip().lower() in REVIEWER_USERNAMES


def resolve_user_role(username):
    username_lc = (username or "").strip().lower()
    if username_lc in REVIEWER_USERNAMES:
        return "reviewer"
    return "user"


def normalize_role_name(value):
    role = (value or "").strip().lower()
    if role in {"user", "reviewer", "admin"}:
        return role
    return None


def normalize_ui_depth_mode(value):
    mode = (value or "").strip().lower()
    if mode in {"easy", "pro", "engineer"}:
        return mode
    return "easy"


def validate_password_policy(password):
    if len(password or "") < 8:
        return "รหัสผ่านต้องยาวอย่างน้อย 8 ตัวอักษร"
    if not re.search(r"[A-Za-z]", password or ""):
        return "รหัสผ่านต้องมีตัวอักษรอย่างน้อย 1 ตัว"
    if not re.search(r"\d", password or ""):
        return "รหัสผ่านต้องมีตัวเลขอย่างน้อย 1 ตัว"
    return None


def check_analyze_rate_limit(user):
    key = f"uid:{user['id']}"
    now = datetime.now().timestamp()

    with RATE_LIMIT_LOCK:
        hits = ANALYZE_HITS[key]
        while hits and (now - hits[0]) > ANALYZE_RATE_LIMIT_WINDOW_SEC:
            hits.popleft()

        if len(hits) >= ANALYZE_RATE_LIMIT_COUNT:
            wait_sec = int(ANALYZE_RATE_LIMIT_WINDOW_SEC - (now - hits[0])) + 1
            return f"เธชเนเธเธเธณเธเธญเธ–เธตเนเน€เธเธดเธเนเธ เธเธฃเธธเธ“เธฒเธฃเธญเธเธฃเธฐเธกเธฒเธ“ {max(wait_sec, 1)} เธงเธดเธเธฒเธ—เธตเนเธฅเนเธงเธฅเธญเธเนเธซเธกเน"

        hits.append(now)
    return None


def render_index(user, **kwargs):
    quick_stats, recent_records = get_dashboard_data(user["id"])
    selected_part = kwargs.get("selected_part") or next(iter(RULES.keys()))
    selected_summary_tone = normalize_summary_tone(kwargs.get("selected_summary_tone"))
    selected_ui_depth_mode = normalize_ui_depth_mode(
        kwargs.get("selected_ui_depth_mode") or session.get("analysis_ui_depth_mode")
    )
    payload = {
        "user": user,
        "is_admin": is_admin_user(user),
        "is_reviewer": is_reviewer_user(user),
        "quick_stats": quick_stats,
        "recent_records": recent_records,
        "selected_part": selected_part,
        "selected_summary_tone": selected_summary_tone,
        "selected_ui_depth_mode": selected_ui_depth_mode,
        "model_version": MODEL_VERSION,
        "max_multi_images": MAX_MULTI_IMAGES,
        "selected_case_image_id": kwargs.get("selected_case_image_id"),
    }
    payload.update(kwargs)
    return render_template("index.html", **payload)


@app.context_processor
def inject_nav_roles():
    user = get_session_user()
    return {
        "nav_is_admin": is_admin_user(user),
        "nav_is_reviewer": is_reviewer_user(user),
    }


@app.template_filter("case_code")
def case_code_filter(value):
    try:
        return f"{int(value):06d}"
    except Exception:
        return str(value or "-")


# ================== STATIC IMAGE ROUTE ==================
@app.route("/feedback_images/<path:filename>")
def feedback_images(filename):
    return send_from_directory("feedback_images", filename)


# ================== LOGIN ==================
@app.route("/login", methods=["GET", "POST"])
def login():
    warning = None

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        mode = request.form.get("mode", "login")
        avatar_file = request.files.get("avatar")

        if not username:
            warning = "เธเธฃเธธเธ“เธฒเธฃเธฐเธเธธเธเธทเนเธญเธเธนเนเนเธเนเธเธฒเธ"
        elif mode == "register":
            password_error = validate_password_policy(password)
            if password_error:
                warning = password_error
            else:
                role = resolve_user_role(username)
                username_lc = username.strip().lower()
                if username_lc == ADMIN_USERNAME.strip().lower():
                    warning = "ชื่อนี้สงวนไว้สำหรับผู้ดูแลระบบ กรุณาใช้ชื่ออื่น"
                    return render_template("login.html", warning=warning, require_password=True)
                avatar_path, avatar_error = save_avatar_file(avatar_file, username)
                if avatar_error:
                    warning = avatar_error
                    return render_template("login.html", warning=warning, require_password=True)
                user, err = create_user(
                    username=username,
                    password_hash=generate_password_hash(password),
                    role=role,
                    avatar_path=avatar_path,
                )
                if err == "username_exists":
                    warning = "เธเธทเนเธญเธเธนเนเนเธเนเธเธตเนเธกเธตเธญเธขเธนเนเนเธฅเนเธง"
                else:
                    session["user"] = user
                    session["last_seen_ts"] = datetime.now().timestamp()
                    audit("auth.register", user=user, target="users", details={"username": username})
                    app.logger.info("register user=%s role=%s", username, role)
                    return redirect("/")
        else:
            auth_row = get_user_auth(username)
            if auth_row:
                if int(auth_row["is_active"]) != 1:
                    warning = "เธเธฑเธเธเธตเธ–เธนเธเธเธดเธ”เธเธฒเธฃเนเธเนเธเธฒเธ"
                elif not auth_row.get("password_hash"):
                    warning = "บัญชีนี้ยังไม่ได้ตั้งรหัสผ่าน กรุณาติดต่อแอดมินเพื่อรีเซ็ตรหัสผ่าน"
                elif not password:
                    warning = "เธเธฃเธธเธ“เธฒเธเธฃเธญเธเธฃเธซเธฑเธชเธเนเธฒเธ"
                else:
                    try:
                        valid_password = check_password_hash(auth_row["password_hash"], password)
                    except ValueError:
                        valid_password = False
                    if not valid_password:
                        warning = "เธฃเธซเธฑเธชเธเนเธฒเธเนเธกเนเธ–เธนเธเธ•เนเธญเธ"
                    else:
                        user = {
                            "id": auth_row["id"],
                            "name": auth_row["username"],
                            "role": auth_row["role"] or "user",
                            "is_active": int(auth_row["is_active"]) == 1,
                            "avatar_path": auth_row.get("avatar_path"),
                        }
                        session["user"] = user
                        session["last_seen_ts"] = datetime.now().timestamp()
                        audit("auth.login", user=user, target="session")
                        app.logger.info("login user=%s role=%s", username, user["role"])
                        return redirect("/")
            else:
                warning = "ไม่พบบัญชีผู้ใช้ กรุณาสมัครสมาชิกก่อนเข้าสู่ระบบ"

    return render_template(
        "login.html",
        warning=warning,
        require_password=True,
    )


# ================== LOGOUT ==================
@app.route("/logout")
def logout():
    user = get_session_user()
    if user:
        audit("auth.logout", user=user, target="session")
    session.clear()
    return redirect("/login")


# ================== INDEX (ANALYZE) ==================
@app.route("/", methods=["GET", "POST"])
@require_login
def index(user):
    if request.method == "POST":
        rate_limit_error = check_analyze_rate_limit(user)
        summary_tone = normalize_summary_tone(request.form.get("summary_tone"))
        ui_depth_mode = normalize_ui_depth_mode(
            request.form.get("ui_depth_mode") or session.get("analysis_ui_depth_mode")
        )
        session["analysis_ui_depth_mode"] = ui_depth_mode
        case_id_raw = (request.form.get("case_id") or "").strip()
        case_image_id_raw = (request.form.get("case_image_id") or "").strip()

        selected_case = None
        selected_case_image_id = None
        selected_case_image = None
        if case_id_raw.isdigit():
            selected_case = get_case_detail(int(case_id_raw), user["id"])
        if selected_case and case_image_id_raw.isdigit():
            selected_case_image_id = int(case_image_id_raw)
            selected_case_image = next(
                (img for img in selected_case.get("images", []) if int(img.get("id", 0)) == selected_case_image_id),
                None,
            )

        if rate_limit_error:
            return render_index(
                user,
                warning=rate_limit_error,
                selected_summary_tone=summary_tone,
                selected_ui_depth_mode=ui_depth_mode,
                selected_case=selected_case,
                selected_case_image_id=selected_case_image_id,
            )

        part = request.form.get("part")
        files = [f for f in request.files.getlist("file") if (f.filename or "").strip()]
        file = files[0] if files else None

        if file is None and selected_case_image:
            img_path = selected_case_image.get("image_path")
            if img_path and os.path.exists(img_path):
                with open(img_path, "rb") as f:
                    raw = f.read()

                class CaseImageFile:
                    filename = os.path.basename(img_path)
                    mimetype = "image/jpeg"
                    stream = io.BytesIO(raw)

                file = CaseImageFile()
                files = [file]
            else:
                return render_index(
                    user,
                    warning="??????????????????????????",
                    selected_part=part,
                    selected_summary_tone=summary_tone,
                    selected_ui_depth_mode=ui_depth_mode,
                    selected_case=selected_case,
                    selected_case_image_id=selected_case_image_id,
                )

        if not part or not file:
            return render_index(
                user,
                warning="???????????????????????????????????????",
                selected_part=part,
                selected_summary_tone=summary_tone,
                selected_ui_depth_mode=ui_depth_mode,
                selected_case=selected_case,
                selected_case_image_id=selected_case_image_id,
            )
        if part not in RULES:
            return render_index(
                user,
                warning="????????????????????????????",
                selected_part=part,
                selected_summary_tone=summary_tone,
                selected_ui_depth_mode=ui_depth_mode,
                selected_case=selected_case,
                selected_case_image_id=selected_case_image_id,
            )

        upload_error = validate_upload(file)
        if upload_error:
            return render_index(
                user,
                warning=upload_error,
                selected_part=part,
                selected_summary_tone=summary_tone,
                selected_ui_depth_mode=ui_depth_mode,
                selected_case=selected_case,
                selected_case_image_id=selected_case_image_id,
            )

        try:
            result_payload, pipeline_error = run_analysis_pipeline(
                user, part, file, extra_files=files[1:], summary_tone=summary_tone
            )
            if pipeline_error:
                return render_index(
                    user,
                    warning=pipeline_error,
                    selected_part=part,
                    selected_summary_tone=summary_tone,
                    selected_ui_depth_mode=ui_depth_mode,
                    selected_case=selected_case,
                    selected_case_image_id=selected_case_image_id,
                )

            if selected_case and result_payload.get("image_path"):
                note = (
                    f"analysis:{part} result={result_payload.get('result')} "
                    f"confidence={result_payload.get('confidence')}%"
                )
                add_case_image(selected_case["id"], user["id"], result_payload["image_path"], note=note)

            audit(
                "analyze.submit",
                user=user,
                target="history",
                details={
                    "part": part,
                    "result": result_payload["result"],
                    "confidence": result_payload["confidence"],
                    "model_version": MODEL_VERSION,
                    "case_id": selected_case["id"] if selected_case else None,
                    "case_image_id": selected_case_image_id,
                },
            )

            app.logger.info(
                "analyze user=%s part=%s result=%s confidence=%.2f",
                user["name"],
                part,
                result_payload["result"],
                result_payload["confidence"],
            )

            return render_index(
                user,
                selected_part=part,
                selected_ui_depth_mode=ui_depth_mode,
                selected_case=selected_case,
                selected_case_image_id=selected_case_image_id,
                **result_payload,
            )

        except UnidentifiedImageError:
            return render_index(
                user,
                warning="?????????????????????????????????????????",
                selected_part=part,
                selected_summary_tone=summary_tone,
                selected_ui_depth_mode=ui_depth_mode,
                selected_case=selected_case,
                selected_case_image_id=selected_case_image_id,
            )
        except Exception as err:
            app.logger.exception("analyze error: %s", err)
            return render_index(
                user,
                warning=f"????????????????????????????: {err}",
                selected_part=part,
                selected_summary_tone=summary_tone,
                selected_ui_depth_mode=ui_depth_mode,
                selected_case=selected_case,
                selected_case_image_id=selected_case_image_id,
            )

    case_id_raw = (request.args.get("case_id") or "").strip()
    case_image_id_raw = (request.args.get("case_image_id") or "").strip()
    selected_case = None
    selected_case_image_id = None
    if case_id_raw.isdigit():
        selected_case = get_case_detail(int(case_id_raw), user["id"])
    if case_image_id_raw.isdigit():
        selected_case_image_id = int(case_image_id_raw)
    ui_depth_mode = normalize_ui_depth_mode(
        request.args.get("ui_depth_mode") or session.get("analysis_ui_depth_mode")
    )
    session["analysis_ui_depth_mode"] = ui_depth_mode
    return render_index(
        user,
        selected_ui_depth_mode=ui_depth_mode,
        selected_case=selected_case,
        selected_case_image_id=selected_case_image_id,
    )


# ================== HISTORY ==================
@app.route("/history")
@require_login
def history(user):
    part = (request.args.get("part") or "").strip()
    result = (request.args.get("result") or "").strip()
    date_from = (request.args.get("date_from") or "").strip()
    date_to = (request.args.get("date_to") or "").strip()

    records = get_user_history(
        user["id"],
        part=part or None,
        result=result or None,
        date_from=date_from or None,
        date_to=date_to or None,
    )
    return render_template(
        "history.html",
        username=user["name"],
        is_admin=is_admin_user(user),
        records=records,
        filters={
            "part": part,
            "result": result,
            "date_from": date_from,
            "date_to": date_to,
        },
        filter_options={
            "parts": list(RULES.keys()),
            "results": ["low", "medium", "high"],
        },
    )


# ================== PROFILE ==================
@app.route("/profile")
@require_login
def profile(user):
    total, last_time = get_profile_summary(user["id"])
    insights = get_profile_insights(user["id"])
    recent = get_user_history(user["id"], limit=5)

    return render_template(
        "profile.html",
        username=user["name"],
        is_admin=is_admin_user(user),
        total=total,
        last_time=last_time,
        insights=insights,
        recent_records=recent,
    )


@app.route("/profile/avatar", methods=["POST"])
@require_login
def profile_avatar_upload(user):
    avatar_file = request.files.get("avatar")
    old_avatar = user.get("avatar_path")
    avatar_path, avatar_error = save_avatar_file(avatar_file, user["name"])
    if avatar_error:
        total, last_time = get_profile_summary(user["id"])
        insights = get_profile_insights(user["id"])
        recent = get_user_history(user["id"], limit=5)
        return render_template(
            "profile.html",
            username=user["name"],
            is_admin=is_admin_user(user),
            total=total,
            last_time=last_time,
            insights=insights,
            recent_records=recent,
            avatar_warning=avatar_error,
        )
    if avatar_path:
        set_user_avatar(user["id"], avatar_path)
        user["avatar_path"] = avatar_path
        session["user"] = user
        if old_avatar and old_avatar != avatar_path:
            cleanup_avatar_file(old_avatar)
        audit("profile.avatar.update", user=user, target="users")
    return redirect("/profile")


# ================== ADMIN ==================
@app.route("/reviewer")
@require_login
def reviewer_dashboard(user):
    if not is_reviewer_user(user):
        return redirect("/")
    pending_cases = list_cases_for_review(limit=200)
    reviewed_cases = list_recent_reviewed_cases(limit=200)
    audit_rows = [r for r in get_recent_audit_logs(limit=200) if r.get("action") == "case.review"][:40]
    return render_template(
        "reviewer.html",
        username=user["name"],
        is_admin=is_admin_user(user),
        is_reviewer=True,
        pending_cases=pending_cases,
        reviewed_cases=reviewed_cases,
        audit_rows=audit_rows,
    )


# ================== ADMIN ==================
def render_admin_page(user, admin_warning=None, admin_success=None):
    metrics = get_admin_metrics()
    feedback_rows = get_recent_feedback(limit=25)
    audit_rows = get_recent_audit_logs(limit=25)
    user_rows = list_users_basic(limit=300)
    return render_template(
        "admin.html",
        username=user["name"],
        is_admin=True,
        metrics=metrics,
        feedback_rows=feedback_rows,
        audit_rows=audit_rows,
        user_rows=user_rows,
        admin_warning=admin_warning,
        admin_success=admin_success,
    )


@app.route("/admin")
@require_admin
def admin_dashboard(user):
    return render_admin_page(user)


@app.route("/admin/backup")
@require_admin
def admin_backup_db(user):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"app_backup_{stamp}.db"
    with open("app.db", "rb") as f:
        payload = f.read()
    audit("admin.backup", user=user, target="app.db", details={"filename": filename})

    return Response(
        payload,
        mimetype="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.route("/admin/restore", methods=["POST"])
@require_admin
def admin_restore_db(user):
    if request.form.get("confirm_restore") != "YES_RESTORE":
        return render_admin_page(
            user,
            admin_warning="กรุณาพิมพ์คำยืนยันให้ถูกต้องก่อนกู้คืนฐานข้อมูล",
            admin_success=None,
        )

    restore_error = restore_database_from_upload(request.files.get("db_file"))
    audit(
        "admin.restore",
        user=user,
        target="app.db",
        details={"status": "failed" if restore_error else "success"},
    )
    return render_admin_page(
        user,
        admin_warning=restore_error,
        admin_success=None if restore_error else "กู้คืนฐานข้อมูลสำเร็จ",
    )


@app.route("/admin/users/<int:user_id>/update", methods=["POST"])
@require_admin
def admin_update_user(user, user_id):
    role = normalize_role_name(request.form.get("role"))
    is_active_raw = (request.form.get("is_active") or "").strip()
    is_active = int(is_active_raw) if is_active_raw in {"0", "1"} else None

    if role is None and is_active is None:
        return render_admin_page(user, admin_warning="ต้องระบุข้อมูลที่ต้องการอัปเดต", admin_success=None)

    user_rows = list_users_basic(limit=300)
    target = next((r for r in user_rows if int(r.get("id", 0)) == int(user_id)), None)
    if not target:
        return render_admin_page(user, admin_warning="ไม่พบบัญชีผู้ใช้ที่ต้องการแก้ไข", admin_success=None)

    target_role = normalize_role_name(target.get("role")) or "user"
    target_active = 1 if int(target.get("is_active", 0)) == 1 else 0

    if int(user_id) == int(user["id"]):
        if role is not None and role != "admin":
            return render_admin_page(user, admin_warning="ไม่สามารถลดสิทธิ์บัญชีแอดมินของตัวเองได้", admin_success=None)
        if is_active == 0:
            return render_admin_page(user, admin_warning="ไม่สามารถปิดการใช้งานบัญชีตัวเองได้", admin_success=None)

    will_drop_admin = target_role == "admin" and target_active == 1 and (
        (role is not None and role != "admin") or (is_active == 0)
    )
    if will_drop_admin and count_admin_users() <= 1:
        return render_admin_page(
            user,
            admin_warning="ไม่สามารถลบหรือปิดแอดมินคนสุดท้ายของระบบได้",
            admin_success=None,
        )

    changed = update_user_role_status(user_id, role=role, is_active=is_active)
    if not changed:
        return render_admin_page(user, admin_warning="ไม่มีการเปลี่ยนแปลงข้อมูล", admin_success=None)

    audit(
        "admin.user_update",
        user=user,
        target=f"user:{user_id}",
        details={
            "before_role": target_role,
            "before_is_active": target_active,
            "after_role": role if role is not None else target_role,
            "after_is_active": is_active if is_active is not None else target_active,
        },
    )
    return render_admin_page(user, admin_warning=None, admin_success="อัปเดตสิทธิ์ผู้ใช้สำเร็จ")


@app.route("/admin/users/<int:user_id>/password", methods=["POST"])
@require_admin
def admin_reset_user_password(user, user_id):
    new_password = request.form.get("new_password", "")
    password_error = validate_password_policy(new_password)
    if password_error:
        return render_admin_page(user, admin_warning=password_error, admin_success=None)

    user_rows = list_users_basic(limit=300)
    target = next((r for r in user_rows if int(r.get("id", 0)) == int(user_id)), None)
    if not target:
        return render_admin_page(user, admin_warning="ไม่พบบัญชีผู้ใช้ที่ต้องการรีเซ็ตรหัสผ่าน", admin_success=None)

    set_user_password(target["username"], generate_password_hash(new_password))
    audit(
        "admin.user_password_reset",
        user=user,
        target=f"user:{user_id}",
        details={"username": target["username"]},
    )
    return render_admin_page(user, admin_warning=None, admin_success=f"รีเซ็ตรหัสผ่านให้ {target['username']} สำเร็จ")


# ================== EXPORT CSV ==================
@app.route("/export_csv")
@require_login
def export_csv(user):
    summary_tone = normalize_summary_tone(request.args.get("summary_tone"))
    rows = get_user_history(user["id"])
    audit(
        "history.export_csv",
        user=user,
        target="history",
        details={"summary_tone": summary_tone},
    )

    def generate():
        header = io.StringIO()
        writer = csv.writer(header)
        writer.writerow(
            [
                "เธงเธฑเธเธ—เธตเน",
                "เธเธทเนเธญเธเธนเนเนเธเน",
                "เธ•เธณเนเธซเธเนเธ",
                "เธเธฅ",
                "เธเธงเธฒเธกเธกเธฑเนเธเนเธ",
                "เธฃเธฐเธ”เธฑเธเธเธงเธฒเธกเน€เธเธทเนเธญเธกเธฑเนเธ",
                "เธฃเธนเธ",
                "เนเธ—เธเธชเธฃเธธเธ",
                "เธชเธฃเธธเธเน€เธซเธ•เธธเธเธฒเธฃเธ“เน",
            ]
        )
        yield "\ufeff" + header.getvalue()
        for row in rows:
            line = io.StringIO()
            writer = csv.writer(line)
            writer.writerow(
                [
                    row.get("datetime", ""),
                    user["name"],
                    row.get("part", ""),
                    row.get("result", ""),
                    row.get("confidence", ""),
                    row.get("trust", ""),
                    row.get("image_path", ""),
                    summary_tone,
                    incident_summary_from_history_row(row, summary_tone),
                ]
            )
            yield line.getvalue()

    return Response(
        generate(),
        mimetype="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename=history_{user["name"]}_{summary_tone}.csv'
        },
    )


@app.route("/report/latest")
@require_login
def report_latest(user):
    summary_tone = normalize_summary_tone(request.args.get("summary_tone"))
    rows = get_user_history(user["id"], limit=1)
    if not rows:
        return render_template(
            "report.html",
            username=user["name"],
            is_admin=is_admin_user(user),
            row=None,
            summary_tone=summary_tone,
            incident_summary=None,
        )
    incident_summary = incident_summary_from_history_row(rows[0], summary_tone)
    audit(
        "report.latest.view",
        user=user,
        target="history",
        details={"summary_tone": summary_tone},
    )
    return render_template(
        "report.html",
        username=user["name"],
        is_admin=is_admin_user(user),
        row=rows[0],
        summary_tone=summary_tone,
        incident_summary=incident_summary,
    )


def render_report_pdf_bytes(user_name, row, summary_tone="customer", incident_summary=None):
    lines = [
        "CarDamage AI - Latest Analysis Report",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"User: {user_name}",
        f"Summary Tone: {summary_tone}",
        f"Date: {row.get('datetime', '-')}",
        f"Part: {row.get('part', '-')}",
        f"Result: {row.get('result', '-')}",
        f"Confidence: {row.get('confidence', '-')}",
        f"Trust: {row.get('trust', '-')}",
        f"Model: {row.get('model_version', '-')}",
        f"Estimated Cost (THB): {row.get('est_cost_min', 0)} - {row.get('est_cost_max', 0)}",
        "Incident Summary:",
        incident_summary or "-",
        "Note: This report is for preliminary assessment.",
    ]

    # Render text to an image first, then export as PDF so Thai text is preserved via TrueType glyphs.
    page_w, page_h = 1240, 1754  # Approx A4 portrait at 150 DPI.
    canvas = Image.new("RGB", (page_w, page_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    def load_font(size):
        candidates = [
            "C:/Windows/Fonts/THSarabunNew.ttf",
            "C:/Windows/Fonts/tahoma.ttf",
            "C:/Windows/Fonts/arial.ttf",
        ]
        for path in candidates:
            if os.path.exists(path):
                try:
                    return ImageFont.truetype(path, size=size)
                except Exception:
                    continue
        return ImageFont.load_default()

    title_font = load_font(36)
    body_font = load_font(24)

    y = 70
    draw.text((70, y), lines[0], fill=(20, 40, 36), font=title_font)
    y += 70
    max_text_width = page_w - 140
    for line in lines[1:]:
        text = str(line or "")
        # Wrap line manually to keep content inside page width.
        chunk = ""
        for ch in text:
            probe = chunk + ch
            w = draw.textlength(probe, font=body_font)
            if w > max_text_width and chunk:
                draw.text((70, y), chunk, fill=(35, 55, 49), font=body_font)
                y += 36
                chunk = ch
            else:
                chunk = probe
        if chunk:
            draw.text((70, y), chunk, fill=(35, 55, 49), font=body_font)
            y += 36
        else:
            y += 36

    import io as _io

    buf = _io.BytesIO()
    canvas.save(buf, format="PDF", resolution=150.0)
    return buf.getvalue()


@app.route("/report/latest.pdf")
@require_login
def report_latest_pdf(user):
    summary_tone = normalize_summary_tone(request.args.get("summary_tone"))
    rows = get_user_history(user["id"], limit=1)
    if not rows:
        return Response("no report data", status=404, mimetype="text/plain")
    incident_summary = incident_summary_from_history_row(rows[0], summary_tone)
    payload = render_report_pdf_bytes(
        user["name"],
        rows[0],
        summary_tone=summary_tone,
        incident_summary=incident_summary,
    )
    audit(
        "report.latest.pdf",
        user=user,
        target="history",
        details={"summary_tone": summary_tone},
    )
    return Response(
        payload,
        mimetype="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="latest_report_{summary_tone}.pdf"'
        },
    )


# ================== FEEDBACK ==================
@app.route("/feedback", methods=["POST"])
@require_login
def feedback(user):
    raw_conf = request.form.get("confidence")
    try:
        conf_value = float(raw_conf) if raw_conf not in (None, "") else None
    except ValueError:
        conf_value = None

    add_feedback(
        user_id=user["id"],
        result=request.form.get("result"),
        confidence=conf_value,
        is_correct=request.form.get("is_correct"),
        comment=request.form.get("comment", ""),
        image_path=request.form.get("image_path"),
    )
    audit("feedback.submit", user=user, target="feedback")

    return redirect("/")


# ================== CASES / NOTIFICATIONS ==================
@app.route("/cases")
@require_login
def cases_page(user):
    cases = list_user_cases(user["id"], limit=100)
    review_cases = list_cases_for_review(limit=150) if is_reviewer_user(user) else []
    return render_template(
        "cases.html",
        username=user["name"],
        is_admin=is_admin_user(user),
        is_reviewer=is_reviewer_user(user),
        cases=cases,
        review_cases=review_cases,
    )


@app.route("/cases/<int:case_id>")
@require_login
def case_detail_page(user, case_id):
    case_data = get_case_detail(case_id, user["id"])
    if not case_data:
        return redirect("/cases")
    return render_template(
        "case_detail.html",
        username=user["name"],
        is_admin=is_admin_user(user),
        is_reviewer=is_reviewer_user(user),
        case_data=case_data,
    )


@app.route("/api/cases", methods=["GET", "POST"])
@api_require_login
def api_cases(user):
    if request.method == "POST":
        title = (request.form.get("title") or "").strip()
        vehicle_info = (request.form.get("vehicle_info") or "").strip()
        if not title:
            return jsonify({"ok": False, "error": "title_required"}), 400
        case_id = create_case(user["id"], title, vehicle_info)
        create_notification(user["id"], "เธชเธฃเนเธฒเธเน€เธเธชเนเธซเธกเน", f"เน€เธเธช #{case_id} เธ–เธนเธเธชเธฃเนเธฒเธเนเธฅเนเธง")
        audit("case.create", user=user, target=f"case:{case_id}")
        return jsonify({"ok": True, "case_id": case_id, "case_code": f"{case_id:06d}"}), 201

    if request.args.get("scope") == "review":
        if not is_reviewer_user(user):
            return jsonify({"ok": False, "error": "forbidden"}), 403
        rows = list_cases_for_review(limit=200)
        return jsonify({"ok": True, "items": rows, "count": len(rows)})
    rows = list_user_cases(user["id"], limit=200)
    return jsonify({"ok": True, "items": rows, "count": len(rows)})


@app.route("/api/cases/<int:case_id>", methods=["GET"])
@api_require_login
def api_case_detail(user, case_id):
    data = get_case_detail(case_id, user["id"])
    if not data:
        return jsonify({"ok": False, "error": "not_found"}), 404
    return jsonify({"ok": True, "case": data})


@app.route("/api/cases/<int:case_id>/images", methods=["POST"])
@api_require_login
def api_case_add_image(user, case_id):
    file = request.files.get("file")
    upload_error = validate_upload(file)
    if upload_error:
        return jsonify({"ok": False, "error": upload_error}), 400

    case_row = get_case_detail(case_id, user["id"])
    if not case_row:
        return jsonify({"ok": False, "error": "not_found"}), 404

    folder = os.path.join("feedback_images", "cases", str(case_id))
    os.makedirs(folder, exist_ok=True)
    name = datetime.now().strftime("%Y%m%d_%H%M%S") + ".jpg"
    path = os.path.join(folder, name).replace("\\", "/")
    Image.open(file.stream).convert("RGB").save(path, format="JPEG")
    add_case_image(case_id, user["id"], path, request.form.get("note"))
    audit("case.image.add", user=user, target=f"case:{case_id}")
    return jsonify({"ok": True, "image_path": path})


@app.route("/api/cases/<int:case_id>/images/<int:image_id>/delete", methods=["POST"])
@api_require_login
def api_case_delete_image(user, case_id, image_id):
    case_row = get_case_detail(case_id, user["id"])
    if not case_row:
        return jsonify({"ok": False, "error": "not_found"}), 404
    deleted_path = delete_case_image(case_id, user["id"], image_id)
    if not deleted_path:
        return jsonify({"ok": False, "error": "image_not_found"}), 404
    try:
        if os.path.exists(deleted_path):
            os.remove(deleted_path)
    except OSError:
        pass
    audit("case.image.delete", user=user, target=f"case:{case_id}", details={"image_id": image_id})
    return jsonify({"ok": True, "deleted_image_id": image_id})


@app.route("/api/cases/<int:case_id>/review", methods=["POST"])
@api_require_reviewer
def api_case_review(user, case_id):
    final_result = (request.form.get("final_result") or "").strip().lower()
    reviewer_note = (request.form.get("reviewer_note") or "").strip()
    if final_result not in {"low", "medium", "high"}:
        return jsonify({"ok": False, "error": "invalid_final_result"}), 400

    case_row = get_case_for_review(case_id)
    if not case_row:
        return jsonify({"ok": False, "error": "not_found"}), 404
    if case_row.get("status") != "needs_review":
        return jsonify({"ok": False, "error": "case_not_pending_review"}), 409

    ok = review_case(case_id, user["id"], final_result, reviewer_note)
    if not ok:
        return jsonify({"ok": False, "error": "review_update_failed"}), 409

    create_notification(
        case_row["user_id"],
        "เธเธฅเธฃเธตเธงเธดเธงเน€เธเธชเธเธฃเนเธญเธกเนเธฅเนเธง",
        f"เน€เธเธช #{case_id} เนเธ”เนเธเธฅเธขเธทเธเธขเธฑเธ: {final_result}",
    )
    audit(
        "case.review",
        user=user,
        target=f"case:{case_id}",
        details={
            "owner_user_id": case_row["user_id"],
            "status_before": case_row.get("status"),
            "status_after": "reviewed",
            "predicted_result": case_row.get("predicted_result"),
            "predicted_confidence": case_row.get("predicted_confidence"),
            "final_result_before": case_row.get("final_result"),
            "final_result": final_result,
            "reviewer_note": reviewer_note,
            "reviewed_by_user_id": user["id"],
            "reviewed_by_username": user["name"],
        },
    )
    return jsonify({"ok": True, "case_id": case_id, "final_result": final_result})


@app.route("/api/notifications")
@api_require_login
def api_notifications(user):
    unread_only = request.args.get("unread") == "1"
    rows = get_notifications(user["id"], unread_only=unread_only, limit=100)
    return jsonify({"ok": True, "items": rows, "count": len(rows)})


@app.route("/api/notifications/<int:notification_id>/read", methods=["POST"])
@api_require_login
def api_notifications_read(user, notification_id):
    ok = mark_notification_read(notification_id, user["id"])
    if not ok:
        return jsonify({"ok": False, "error": "not_found"}), 404
    return jsonify({"ok": True})


# ================== API ==================
@app.route("/api/health")
def api_health():
    return jsonify(
        {
            "ok": True,
            "service": "car-damage-ai",
            "model_version": MODEL_VERSION,
            "non_car_guard": {
                "enabled": NON_CAR_GUARD_ENABLED,
                "min_confidence": NON_CAR_MIN_CONFIDENCE,
                "min_score_gap": NON_CAR_MIN_SCORE_GAP,
                "max_entropy": NON_CAR_MAX_ENTROPY,
                "model_enabled": NON_CAR_MODEL_ENABLED,
                "model_path": NON_CAR_MODEL_PATH,
                "model_min_car_probability": NON_CAR_MODEL_MIN_CAR_PROB,
                "model_car_index": NON_CAR_MODEL_CAR_INDEX,
                "hard_block_max_car_probability": NON_CAR_HARD_BLOCK_MAX_CAR_PROB,
                "rule_block_min_risk": NON_CAR_RULE_BLOCK_MIN_RISK,
            },
            "manual_review": {
                "score_gap_threshold": MANUAL_REVIEW_SCORE_GAP,
                "max_entropy_threshold": MANUAL_REVIEW_MAX_ENTROPY,
            },
            "multi_angle": {"max_images": MAX_MULTI_IMAGES},
            "summary_tones": ["customer", "technical", "insurance"],
            "time": datetime.now().isoformat(),
        }
    )


@app.route("/api/history")
@api_require_login
def api_history(user):
    limit = request.args.get("limit", "20")
    try:
        limit = max(1, min(200, int(limit)))
    except ValueError:
        limit = 20
    rows = get_user_history(
        user["id"],
        part=(request.args.get("part") or None),
        result=(request.args.get("result") or None),
        date_from=(request.args.get("date_from") or None),
        date_to=(request.args.get("date_to") or None),
        limit=limit,
    )
    return jsonify({"ok": True, "items": rows, "count": len(rows)})


@app.route("/api/analyze", methods=["POST"])
@api_require_login
def api_analyze(user):
    rate_limit_error = check_analyze_rate_limit(user)
    if rate_limit_error:
        return jsonify({"ok": False, "error": rate_limit_error}), 429

    part = request.form.get("part")
    summary_tone = normalize_summary_tone(request.form.get("summary_tone"))
    files = [f for f in request.files.getlist("file") if (f.filename or "").strip()]
    file = files[0] if files else None
    if not part or not file:
        return jsonify({"ok": False, "error": "missing part or image file"}), 400
    if part not in RULES:
        return jsonify({"ok": False, "error": "invalid part"}), 400

    if request.args.get("async") == "1":
        if len(files) > 1:
            return jsonify({"ok": False, "error": "async mode เธฃเธญเธเธฃเธฑเธเธเธฃเธฑเนเธเธฅเธฐ 1 เธฃเธนเธเน€เธ—เนเธฒเธเธฑเนเธ"}), 400
        ensure_worker_started()
        job_id = uuid.uuid4().hex
        upload_error = validate_upload(file)
        if upload_error:
            return jsonify({"ok": False, "error": upload_error}), 400

        tmp_dir = "feedback_images/_jobs"
        os.makedirs(tmp_dir, exist_ok=True)
        ext = os.path.splitext((file.filename or "upload.jpg").lower())[1] or ".jpg"
        tmp_path = os.path.join(tmp_dir, f"{job_id}{ext}").replace("\\", "/")
        file.save(tmp_path)
        create_analysis_job(job_id, user["id"], part)
        backend = enqueue_analysis_job(
            {"job_id": job_id, "user": user, "part": part, "file_path": tmp_path}
        )
        audit("api.analyze.queued", user=user, target="analysis_jobs", details={"job_id": job_id})
        return jsonify({"ok": True, "job_id": job_id, "status": "queued", "backend": backend}), 202

    try:
        result_payload, pipeline_error = run_analysis_pipeline(
            user,
            part,
            file,
            extra_files=files[1:],
            summary_tone=summary_tone,
        )
        if pipeline_error:
            return jsonify({"ok": False, "error": pipeline_error}), 400
        audit(
            "api.analyze.submit",
            user=user,
            target="history",
            details={
                "part": part,
                "result": result_payload["result"],
                "confidence": result_payload["confidence"],
            },
        )
        return jsonify({"ok": True, **result_payload})
    except UnidentifiedImageError:
        return jsonify({"ok": False, "error": "invalid image data"}), 400
    except Exception as err:
        app.logger.exception("api analyze error: %s", err)
        return jsonify({"ok": False, "error": "internal_error"}), 500


@app.route("/api/analyze/jobs/<job_id>")
@api_require_login
def api_analyze_job_status(user, job_id):
    job = get_analysis_job(job_id, user["id"])
    if not job:
        return jsonify({"ok": False, "error": "not_found"}), 404
    return jsonify({"ok": True, "job": job})


@app.route("/api/admin/metrics")
@api_require_admin
def api_admin_metrics(user):
    metrics = get_admin_metrics()
    audit("api.admin.metrics", user=user, target="metrics")
    return jsonify({"ok": True, "metrics": metrics})


@app.route("/api/admin/monitor")
@api_require_admin
def api_admin_monitor(user):
    metrics = get_admin_metrics()
    jobs = get_analysis_job_stats()
    monitor = {
        "metrics": metrics,
        "jobs": jobs,
        "queue_size": get_analysis_queue_size(),
        "queue_backend": get_queue_backend(),
        "inline_worker_enabled": INLINE_WORKER_ENABLED,
        "worker_started": WORKER_STARTED,
    }
    audit("api.admin.monitor", user=user, target="monitor")
    return jsonify({"ok": True, "monitor": monitor})


@app.route("/healthz")
def healthz():
    return (
        jsonify(
            {
                "ok": True,
                "service": "car-damage-ai",
                "time": datetime.utcnow().isoformat() + "Z",
                "queue_backend": get_queue_backend(),
                "queue_size": get_analysis_queue_size(),
            }
        ),
        200,
    )


# ================== RUN ==================
if __name__ == "__main__":
    port = int(os.getenv("PORT", os.getenv("FLASK_PORT", "5000")))
    app.run(
        host=os.getenv("FLASK_HOST", "0.0.0.0"),
        port=port,
        debug=os.getenv("FLASK_DEBUG", "false").lower() == "true",
    )




