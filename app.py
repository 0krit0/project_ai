import os
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
from PIL import Image, UnidentifiedImageError

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
    log_audit,
    create_case,
    list_user_cases,
    add_case_image,
    get_case_detail,
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


def validate_upload(file_storage):
    if file_storage is None:
        return "ไม่พบไฟล์รูปที่อัปโหลด"

    filename = (file_storage.filename or "").strip()
    if not filename:
        return "ชื่อไฟล์ไม่ถูกต้อง"

    ext = os.path.splitext(filename.lower())[1]
    if ext not in ALLOWED_IMAGE_EXTENSIONS:
        return "รองรับเฉพาะไฟล์ .jpg .jpeg .png .webp"

    mimetype = (file_storage.mimetype or "").lower()
    if not mimetype.startswith("image/"):
        return "ไฟล์ที่อัปโหลดต้องเป็นรูปภาพเท่านั้น"

    pos = file_storage.stream.tell()
    file_storage.stream.seek(0, os.SEEK_END)
    size = file_storage.stream.tell()
    file_storage.stream.seek(pos, os.SEEK_SET)

    if size <= 0:
        return "ไฟล์รูปว่างหรืออ่านข้อมูลไม่ได้"
    if size > MAX_UPLOAD_BYTES:
        return f"ไฟล์ใหญ่เกินกำหนด (สูงสุด {MAX_UPLOAD_BYTES // (1024 * 1024)}MB)"
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
        return "รูปมีขนาดเล็กเกินไป (ขั้นต่ำแนะนำ 160x160)", []
    if brightness < 18:
        return "รูปมืดเกินไปจนระบบประเมินได้ไม่แม่นยำ", []
    if brightness > 248:
        return "รูปสว่างจ้าเกินไปจนรายละเอียดหาย", []

    notes = []
    if brightness < 45:
        notes.append("ภาพค่อนข้างมืด")
    if brightness > 220:
        notes.append("ภาพค่อนข้างสว่างจ้า")
    if contrast < 22:
        notes.append("คอนทราสต์ต่ำ อาจแยกจุดเสียหายได้ยาก")
    if edge_strength < 10:
        notes.append("ภาพอาจเบลอหรือสั่น")

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
            "tier": "พร้อมใช้ตัดสินใจเบื้องต้น",
            "advice": "ใช้ผลนี้ช่วยประเมินค่าเสียหายเบื้องต้นได้ แต่ยังควรมีการตรวจหน้างาน",
            "flag": "high",
        }
    if confidence >= 65:
        return {
            "tier": "ควรยืนยันซ้ำ",
            "advice": "ใช้ผลนี้ร่วมกับการตรวจด้วยตาและภาพเพิ่มเติมก่อนประเมินค่าใช้จ่าย",
            "flag": "medium",
        }
    return {
        "tier": "ความไม่แน่นอนสูง",
        "advice": "ไม่ควรใช้ผลนี้เดี่ยวๆ แนะนำถ่ายภาพใหม่หรือให้ช่างตรวจสอบก่อนสรุป",
        "flag": "low",
    }


def estimate_cost_range(part, level):
    base = {
        "low": (1500, 4500),
        "medium": (4500, 12000),
        "high": (12000, 35000),
    }.get(level, (2000, 8000))
    part_multiplier = {
        "ไฟหน้า": 1.15,
        "กระจกรถ": 1.25,
        "กันชนหน้า": 1.0,
        "กันชนหลัง": 1.0,
        "ประตู": 1.1,
        "ฝากระโปรงหน้า": 1.2,
        "แก้มข้างรถ": 1.05,
    }.get(part, 1.0)
    return int(base[0] * part_multiplier), int(base[1] * part_multiplier)


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
            {"description": "ไม่มีข้อมูล", "repair": "ไม่มีข้อมูล"},
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


def run_analysis_pipeline(user, part, file_storage):
    if not part or not file_storage:
        return None, "missing part or file"
    if part not in RULES:
        return None, "invalid part"

    upload_error = validate_upload(file_storage)
    if upload_error:
        return None, upload_error

    img_pil = Image.open(file_storage.stream).convert("RGB")
    quality_error, quality_notes = assess_image_quality(img_pil)
    if quality_error:
        return None, quality_error

    resized = img_pil.resize((224, 224))
    img_np = np.array(resized, dtype=np.float32) / 255.0
    img_batch = np.expand_dims(img_np, axis=0)

    current_model = get_model()
    pred = current_model.predict(img_batch, verbose=0)
    raw_scores = pred[0]
    idx = int(np.argmax(raw_scores))
    confidence = round(float(raw_scores[idx]) * 100, 2)
    level = labels[idx]

    evidence_items = [
        {"label": labels[i], "score": round(float(raw_scores[i]) * 100, 2)}
        for i in np.argsort(raw_scores)[::-1][:TOP_K_OUTCOMES]
    ]
    assessment_options, weighted_cost, score_gap, uncertainty = build_assessment_options(
        part, raw_scores
    )
    entropy = prediction_entropy(raw_scores)
    if NON_CAR_GUARD_ENABLED and (
        confidence < NON_CAR_MIN_CONFIDENCE
        or score_gap < NON_CAR_MIN_SCORE_GAP
        or entropy > NON_CAR_MAX_ENTROPY
    ):
        return (
            None,
            "ไม่สามารถยืนยันว่าเป็นภาพความเสียหายของรถได้ กรุณาอัปโหลดภาพรถที่เห็นชิ้นส่วนชัดเจน",
        )

    policy = confidence_policy(confidence)
    detail = RULES.get(part, {}).get(
        level,
        {"description": "ไม่มีข้อมูล", "repair": "ไม่มีข้อมูล"},
    )
    est_min, est_max = estimate_cost_range(part, level)

    save_dir = os.path.join("feedback_images", level)
    os.makedirs(save_dir, exist_ok=True)

    filename = datetime.now().strftime("%Y%m%d_%H%M%S") + ".jpg"
    image_path = os.path.join(save_dir, filename).replace("\\", "/")
    img_pil.save(image_path, format="JPEG")

    heatmap_path = None
    heatmap_img = generate_heatmap_overlay(img_batch, idx)
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
        "วิเคราะห์เสร็จแล้ว",
        f"{part}: {level} ({confidence}%) ประมาณค่าใช้จ่าย {est_min:,}-{est_max:,} บาท",
    )
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
        "evidence_items": evidence_items,
        "assessment_options": assessment_options,
        "image_path": image_path,
        "heatmap_path": heatmap_path,
        "model_version": MODEL_VERSION,
        "est_cost_min": est_min,
        "est_cost_max": est_max,
        "expected_cost": weighted_cost,
        "score_gap": score_gap,
        "prediction_entropy": round(entropy, 4),
        "uncertainty": uncertainty,
        "top_k_outcomes": TOP_K_OUTCOMES,
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
    if role == "admin":
        return True
    return user.get("name", "").strip().lower() == ADMIN_USERNAME.strip().lower()


def check_analyze_rate_limit(user):
    key = f"uid:{user['id']}"
    now = datetime.now().timestamp()

    with RATE_LIMIT_LOCK:
        hits = ANALYZE_HITS[key]
        while hits and (now - hits[0]) > ANALYZE_RATE_LIMIT_WINDOW_SEC:
            hits.popleft()

        if len(hits) >= ANALYZE_RATE_LIMIT_COUNT:
            wait_sec = int(ANALYZE_RATE_LIMIT_WINDOW_SEC - (now - hits[0])) + 1
            return f"ส่งคำขอถี่เกินไป กรุณารอประมาณ {max(wait_sec, 1)} วินาทีแล้วลองใหม่"

        hits.append(now)
    return None


def render_index(user, **kwargs):
    quick_stats, recent_records = get_dashboard_data(user["id"])
    selected_part = kwargs.get("selected_part") or next(iter(RULES.keys()))
    payload = {
        "user": user,
        "is_admin": is_admin_user(user),
        "quick_stats": quick_stats,
        "recent_records": recent_records,
        "selected_part": selected_part,
        "model_version": MODEL_VERSION,
    }
    payload.update(kwargs)
    return render_template("index.html", **payload)


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
            warning = "กรุณาระบุชื่อผู้ใช้งาน"
        elif mode == "register":
            if len(password) < 6:
                warning = "รหัสผ่านต้องมีอย่างน้อย 6 ตัวอักษร"
            else:
                role = (
                    "admin"
                    if username.strip().lower() == ADMIN_USERNAME.strip().lower()
                    else "user"
                )
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
                    warning = "ชื่อผู้ใช้นี้มีอยู่แล้ว"
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
                    warning = "บัญชีถูกปิดการใช้งาน"
                elif not auth_row.get("password_hash"):
                    if LOGIN_PASSWORD and password != LOGIN_PASSWORD:
                        warning = "รหัสผ่านระบบไม่ถูกต้อง"
                    else:
                        user = {
                            "id": auth_row["id"],
                            "name": auth_row["username"],
                            "role": auth_row["role"] or "user",
                            "is_active": int(auth_row["is_active"]) == 1,
                            "avatar_path": auth_row.get("avatar_path"),
                        }
                        if password and len(password) >= 6:
                            set_user_password(username, generate_password_hash(password))
                        session["user"] = user
                        session["last_seen_ts"] = datetime.now().timestamp()
                        audit("auth.login_legacy_row", user=user, target="session")
                        app.logger.info("legacy-row login user=%s", username)
                        return redirect("/")
                elif not password:
                    warning = "กรุณากรอกรหัสผ่าน"
                else:
                    try:
                        valid_password = check_password_hash(auth_row["password_hash"], password)
                    except ValueError:
                        valid_password = False
                    if not valid_password:
                        warning = "รหัสผ่านไม่ถูกต้อง"
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
                # Backward compatibility: allow old username-only flow if system password matches.
                if LOGIN_PASSWORD and password != LOGIN_PASSWORD:
                    warning = "รหัสผ่านระบบไม่ถูกต้อง"
                else:
                    user = get_or_create_user(username)
                    auth_after = get_user_auth(username) or {}
                    if not auth_after.get("password_hash"):
                        # Bootstrap password for legacy user if user enters one.
                        if password:
                            set_user_password(username, generate_password_hash(password))
                    session["user"] = user
                    session["last_seen_ts"] = datetime.now().timestamp()
                    audit("auth.login_legacy", user=user, target="session")
                    app.logger.info("legacy login user=%s", username)
                    return redirect("/")

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
        if rate_limit_error:
            return render_index(user, warning=rate_limit_error)

        part = request.form.get("part")
        file = request.files.get("file")

        if not part or not file:
            return render_index(
                user,
                warning="กรุณาเลือกตำแหน่งและรูปภาพก่อนวิเคราะห์",
                selected_part=part,
            )
        if part not in RULES:
            return render_index(
                user,
                warning="ตำแหน่งความเสียหายไม่ถูกต้อง",
                selected_part=part,
            )

        upload_error = validate_upload(file)
        if upload_error:
            return render_index(user, warning=upload_error, selected_part=part)

        try:
            result_payload, pipeline_error = run_analysis_pipeline(user, part, file)
            if pipeline_error:
                return render_index(user, warning=pipeline_error, selected_part=part)
            audit(
                "analyze.submit",
                user=user,
                target="history",
                details={
                    "part": part,
                    "result": result_payload["result"],
                    "confidence": result_payload["confidence"],
                    "model_version": MODEL_VERSION,
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
                **result_payload,
            )

        except UnidentifiedImageError:
            return render_index(
                user,
                warning="ไฟล์รูปไม่ถูกต้องหรือระบบไม่รองรับไฟล์นี้",
                selected_part=part,
            )
        except Exception as err:
            app.logger.exception("analyze error: %s", err)
            return render_index(
                user,
                warning=f"เกิดข้อผิดพลาดในการวิเคราะห์: {err}",
                selected_part=part,
            )

    return render_index(user)


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
@app.route("/admin")
@require_admin
def admin_dashboard(user):
    metrics = get_admin_metrics()
    feedback_rows = get_recent_feedback(limit=25)
    audit_rows = get_recent_audit_logs(limit=25)
    return render_template(
        "admin.html",
        username=user["name"],
        is_admin=True,
        metrics=metrics,
        feedback_rows=feedback_rows,
        audit_rows=audit_rows,
    )


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
        metrics = get_admin_metrics()
        feedback_rows = get_recent_feedback(limit=25)
        audit_rows = get_recent_audit_logs(limit=25)
        return render_template(
            "admin.html",
            username=user["name"],
            is_admin=True,
            metrics=metrics,
            feedback_rows=feedback_rows,
            audit_rows=audit_rows,
            admin_warning="กรุณาพิมพ์คำยืนยันให้ถูกต้องก่อนกู้คืนฐานข้อมูล",
            admin_success=None,
        )

    restore_error = restore_database_from_upload(request.files.get("db_file"))
    metrics = get_admin_metrics()
    feedback_rows = get_recent_feedback(limit=25)
    audit_rows = get_recent_audit_logs(limit=25)
    audit(
        "admin.restore",
        user=user,
        target="app.db",
        details={"status": "failed" if restore_error else "success"},
    )

    return render_template(
        "admin.html",
        username=user["name"],
        is_admin=True,
        metrics=metrics,
        feedback_rows=feedback_rows,
        audit_rows=audit_rows,
        admin_warning=restore_error,
        admin_success=None if restore_error else "กู้คืนฐานข้อมูลสำเร็จ",
    )


# ================== EXPORT CSV ==================
@app.route("/export_csv")
@require_login
def export_csv(user):
    rows = get_user_history(user["id"])
    audit("history.export_csv", user=user, target="history")

    def generate():
        yield "\ufeffวันที่,ชื่อผู้ใช้,ตำแหน่ง,ผล,ความมั่นใจ,ระดับความเชื่อมั่น,รูป\n"
        for row in rows:
            yield (
                f'{row["datetime"]},'
                f'{user["name"]},'
                f'{row["part"]},'
                f'{row["result"]},'
                f'{row["confidence"]},'
                f'{row["trust"]},'
                f'{row["image_path"]}\n'
            )

    return Response(
        generate(),
        mimetype="text/csv",
        headers={"Content-Disposition": f'attachment; filename=history_{user["name"]}.csv'},
    )


@app.route("/report/latest")
@require_login
def report_latest(user):
    rows = get_user_history(user["id"], limit=1)
    if not rows:
        return render_template(
            "report.html",
            username=user["name"],
            is_admin=is_admin_user(user),
            row=None,
        )
    audit("report.latest.view", user=user, target="history")
    return render_template(
        "report.html",
        username=user["name"],
        is_admin=is_admin_user(user),
        row=rows[0],
    )


def _escape_pdf_text(text):
    if text is None:
        return ""
    s = str(text).replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    return s.encode("latin-1", errors="replace").decode("latin-1")


def render_report_pdf_bytes(user_name, row):
    lines = [
        "CarDamage AI - Latest Analysis Report",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"User: {user_name}",
        f"Date: {row.get('datetime', '-')}",
        f"Part: {row.get('part', '-')}",
        f"Result: {row.get('result', '-')}",
        f"Confidence: {row.get('confidence', '-')}",
        f"Trust: {row.get('trust', '-')}",
        f"Model: {row.get('model_version', '-')}",
        f"Estimated Cost (THB): {row.get('est_cost_min', 0)} - {row.get('est_cost_max', 0)}",
        "Note: This report is for preliminary assessment.",
    ]
    text_cmd = ["BT", "/F1 12 Tf", "50 790 Td", "14 TL"]
    for i, line in enumerate(lines):
        if i == 0:
            text_cmd.append(f"({_escape_pdf_text(line)}) Tj")
        else:
            text_cmd.append(f"T* ({_escape_pdf_text(line)}) Tj")
    text_cmd.append("ET")
    stream = "\n".join(text_cmd).encode("latin-1")

    objects = []
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
    objects.append(
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>"
    )
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    objects.append(
        b"<< /Length "
        + str(len(stream)).encode("ascii")
        + b" >>\nstream\n"
        + stream
        + b"\nendstream"
    )

    out = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for idx, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out.extend(f"{idx} 0 obj\n".encode("ascii"))
        out.extend(obj)
        out.extend(b"\nendobj\n")

    xref_start = len(out)
    out.extend(f"xref\n0 {len(objects)+1}\n".encode("ascii"))
    out.extend(b"0000000000 65535 f \n")
    for off in offsets[1:]:
        out.extend(f"{off:010d} 00000 n \n".encode("ascii"))
    out.extend(
        (
            f"trailer\n<< /Size {len(objects)+1} /Root 1 0 R >>\n"
            f"startxref\n{xref_start}\n%%EOF\n"
        ).encode("ascii")
    )
    return bytes(out)


@app.route("/report/latest.pdf")
@require_login
def report_latest_pdf(user):
    rows = get_user_history(user["id"], limit=1)
    if not rows:
        return Response("no report data", status=404, mimetype="text/plain")
    payload = render_report_pdf_bytes(user["name"], rows[0])
    audit("report.latest.pdf", user=user, target="history")
    return Response(
        payload,
        mimetype="application/pdf",
        headers={"Content-Disposition": 'attachment; filename="latest_report.pdf"'},
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
    return render_template(
        "cases.html",
        username=user["name"],
        is_admin=is_admin_user(user),
        cases=cases,
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
        create_notification(user["id"], "สร้างเคสใหม่", f"เคส #{case_id} ถูกสร้างแล้ว")
        audit("case.create", user=user, target=f"case:{case_id}")
        return jsonify({"ok": True, "case_id": case_id}), 201

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
            },
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
    file = request.files.get("file")
    if not part or not file:
        return jsonify({"ok": False, "error": "missing part or image file"}), 400
    if part not in RULES:
        return jsonify({"ok": False, "error": "invalid part"}), 400

    if request.args.get("async") == "1":
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
        result_payload, pipeline_error = run_analysis_pipeline(user, part, file)
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


# ================== RUN ==================
if __name__ == "__main__":
    app.run(
        host=os.getenv("FLASK_HOST", "0.0.0.0"),
        port=int(os.getenv("FLASK_PORT", "5000")),
        debug=os.getenv("FLASK_DEBUG", "false").lower() == "true",
    )



