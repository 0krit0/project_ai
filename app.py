import os
import logging
import subprocess
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
)

from rules import RULES
from retrain_condition import should_retrain, save_status
from db import (
    init_db,
    migrate_from_csv,
    get_or_create_user,
    add_history,
    add_feedback,
    get_user_history,
    get_profile_summary,
    get_profile_insights,
    get_dashboard_data,
    get_admin_metrics,
    get_recent_feedback,
    restore_database_from_upload,
)

# ================== APP SETUP ==================
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-change-me")

LOG_PATH = os.getenv("APP_LOG_PATH", "run.app.log")
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
LOGIN_PASSWORD = os.getenv("APP_LOGIN_PASSWORD", "")
ANALYZE_RATE_LIMIT_COUNT = int(os.getenv("ANALYZE_RATE_LIMIT_COUNT", "8"))
ANALYZE_RATE_LIMIT_WINDOW_SEC = int(os.getenv("ANALYZE_RATE_LIMIT_WINDOW_SEC", "60"))

ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(5 * 1024 * 1024)))

RETRAIN_CHECK_LOCK = Lock()
RATE_LIMIT_LOCK = Lock()
ANALYZE_HITS = defaultdict(deque)
_retrain_check_running = False

init_db()
migrate_from_csv()

# ================== LOAD MODEL ==================
model = tf.keras.models.load_model("damage_model.h5")
labels = ["high", "low", "medium"]


def configure_logging():
    handler = RotatingFileHandler(LOG_PATH, maxBytes=1_000_000, backupCount=3, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    app.logger.setLevel(logging.INFO)
    app.logger.addHandler(handler)


configure_logging()


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
    for layer in reversed(model.layers):
        out = getattr(layer, "output_shape", None)
        if out is not None and isinstance(out, tuple) and len(out) == 4:
            return layer.name
    return None


def generate_heatmap_overlay(img_batch, class_index):
    layer_name = get_last_conv_layer_name()
    if layer_name is None:
        return None

    try:
        grad_model = tf.keras.models.Model(
            [model.inputs],
            [model.get_layer(layer_name).output, model.output],
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

    if "id" in user and "name" in user:
        return user

    if "name" in user:
        normalized = get_or_create_user(user["name"])
        session["user"] = normalized
        return normalized

    return None


def is_admin_user(user):
    if not user:
        return False
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

        if not username:
            warning = "กรุณาระบุชื่อผู้ใช้งาน"
        elif LOGIN_PASSWORD and password != LOGIN_PASSWORD:
            warning = "รหัสผ่านไม่ถูกต้อง"
        else:
            session["user"] = get_or_create_user(username)
            app.logger.info("login user=%s", username)
            return redirect("/")

    return render_template(
        "login.html",
        warning=warning,
        require_password=bool(LOGIN_PASSWORD),
    )


# ================== LOGOUT ==================
@app.route("/logout")
def logout():
    session.clear()
    return redirect("/login")


# ================== INDEX (ANALYZE) ==================
@app.route("/", methods=["GET", "POST"])
def index():
    user = get_session_user()
    if user is None:
        return redirect("/login")

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
            img_pil = Image.open(file).convert("RGB")
            quality_error, quality_notes = assess_image_quality(img_pil)
            if quality_error:
                return render_index(user, warning=quality_error, selected_part=part)

            resized = img_pil.resize((224, 224))
            img_np = np.array(resized, dtype=np.float32) / 255.0
            img_batch = np.expand_dims(img_np, axis=0)

            pred = model.predict(img_batch, verbose=0)
            raw_scores = pred[0]
            idx = int(np.argmax(raw_scores))
            confidence = round(float(raw_scores[idx]) * 100, 2)
            level = labels[idx]

            top_indices = np.argsort(raw_scores)[::-1][:3]
            evidence_items = [
                {
                    "label": labels[int(i)],
                    "score": round(float(raw_scores[int(i)]) * 100, 2),
                }
                for i in top_indices
            ]

            policy = confidence_policy(confidence)
            trust = policy["tier"]
            decision = policy["advice"]

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

            detail = RULES.get(part, {}).get(
                level,
                {"description": "ไม่มีข้อมูล", "repair": "ไม่มีข้อมูล"},
            )

            add_history(
                user_id=user["id"],
                part=part,
                result=level,
                confidence=confidence,
                trust=trust,
                image_path=image_path,
            )

            Thread(target=dispatch_retrain_check, daemon=True).start()

            app.logger.info(
                "analyze user=%s part=%s result=%s confidence=%.2f",
                user["name"],
                part,
                level,
                confidence,
            )

            return render_index(
                user,
                result=level,
                selected_part=part,
                confidence=confidence,
                trust=trust,
                decision=decision,
                detail=detail,
                quality_notes=quality_notes,
                evidence_items=evidence_items,
                image_path=image_path,
                heatmap_path=heatmap_path,
                policy=policy,
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
def history():
    user = get_session_user()
    if user is None:
        return redirect("/login")

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
def profile():
    user = get_session_user()
    if user is None:
        return redirect("/login")

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


# ================== ADMIN ==================
@app.route("/admin")
def admin_dashboard():
    user = get_session_user()
    if user is None:
        return redirect("/login")
    if not is_admin_user(user):
        return redirect("/")

    metrics = get_admin_metrics()
    feedback_rows = get_recent_feedback(limit=25)
    return render_template(
        "admin.html",
        username=user["name"],
        is_admin=True,
        metrics=metrics,
        feedback_rows=feedback_rows,
    )


@app.route("/admin/backup")
def admin_backup_db():
    user = get_session_user()
    if user is None:
        return redirect("/login")
    if not is_admin_user(user):
        return redirect("/")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"app_backup_{stamp}.db"
    with open("app.db", "rb") as f:
        payload = f.read()

    return Response(
        payload,
        mimetype="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.route("/admin/restore", methods=["POST"])
def admin_restore_db():
    user = get_session_user()
    if user is None:
        return redirect("/login")
    if not is_admin_user(user):
        return redirect("/")

    restore_error = restore_database_from_upload(request.files.get("db_file"))
    metrics = get_admin_metrics()
    feedback_rows = get_recent_feedback(limit=25)

    return render_template(
        "admin.html",
        username=user["name"],
        is_admin=True,
        metrics=metrics,
        feedback_rows=feedback_rows,
        admin_warning=restore_error,
        admin_success=None if restore_error else "กู้คืนฐานข้อมูลสำเร็จ",
    )


# ================== EXPORT CSV ==================
@app.route("/export_csv")
def export_csv():
    user = get_session_user()
    if user is None:
        return redirect("/login")

    rows = get_user_history(user["id"])

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


# ================== FEEDBACK ==================
@app.route("/feedback", methods=["POST"])
def feedback():
    user = get_session_user()
    if user is None:
        return redirect("/login")

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

    return redirect("/")


# ================== RUN ==================
if __name__ == "__main__":
    app.run(
        host=os.getenv("FLASK_HOST", "127.0.0.1"),
        port=int(os.getenv("FLASK_PORT", "5000")),
        debug=os.getenv("FLASK_DEBUG", "false").lower() == "true",
    )
