import os
import csv
from datetime import datetime

import tensorflow as tf
import numpy as np
from PIL import Image
import subprocess

from rules import RULES
from retrain_condition import should_retrain, save_status

from flask import (
    Flask,
    request,
    redirect,
    session,
    render_template,
    send_from_directory,
    Response
)

# ================== APP SETUP ==================
app = Flask(__name__)
app.secret_key = "car_damage_ai_secret_123456"

# ================== STATIC IMAGE ROUTE ==================
@app.route("/feedback_images/<path:filename>")
def feedback_images(filename):
    return send_from_directory("feedback_images", filename)

# ================== LOAD MODEL ==================
model = tf.keras.models.load_model("damage_model.h5")
labels = ["high", "low", "medium"]

# ================== LOGIN ==================
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        if username:
            session["user"] = {"name": username}
            return redirect("/")
    return render_template("login.html")

# ================== LOGOUT ==================
@app.route("/logout")
def logout():
    session.clear()
    return redirect("/login")

# ================== INDEX (ANALYZE) ==================
@app.route("/", methods=["GET", "POST"])
def index():
    if "user" not in session:
        return redirect("/login")

    if request.method == "POST":
        part = request.form.get("part")
        file = request.files.get("file")

        if not part or not file:
            return render_template("index.html", user=session["user"])

        try:
            # 1. Load image
            img_pil = Image.open(file).convert("RGB")

            # 2. Prepare for model
            img = img_pil.resize((224, 224))
            img = np.array(img) / 255.0
            img = np.expand_dims(img, axis=0)

            # 3. Predict
            pred = model.predict(img)
            idx = np.argmax(pred)
            confidence = round(float(np.max(pred)) * 100, 2)
            level = labels[idx]

            # 4. Save image
            save_dir = os.path.join("feedback_images", level)
            os.makedirs(save_dir, exist_ok=True)

            filename = datetime.now().strftime("%Y%m%d_%H%M%S") + ".jpg"
            image_path = os.path.join(save_dir, filename)
            img_pil.save(image_path, format="JPEG")

            # 5. Get rule detail
            detail = RULES.get(part, {}).get(level, {
                "description": "ไม่มีข้อมูล",
                "repair": "ไม่มีข้อมูล"
            })

            # 6. Trust level
            if confidence >= 70:
                trust = "ความมั่นใจสูง"
            elif confidence >= 40:
                trust = "ความมั่นใจปานกลาง"
            else:
                trust = "ความมั่นใจต่ำ"

            # 7. Decision
            if confidence >= 80:
                decision = "สามารถใช้ผลลัพธ์นี้ประกอบการตัดสินใจเบื้องต้นได้"
            elif confidence >= 60:
                decision = "ควรใช้ร่วมกับการตรวจสอบเพิ่มเติม"
            else:
                decision = "ไม่แนะนำให้ใช้ผลลัพธ์นี้เพียงอย่างเดียว"

            # 8. Save experience log
            log_file = "experience_log.csv"
            file_exists = os.path.isfile(log_file)

            with open(log_file, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow([
                        "datetime",
                        "username",
                        "part",
                        "result",
                        "confidence",
                        "trust",
                        "image_path"
                    ])

                writer.writerow([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    session["user"]["name"],
                    part,
                    level,
                    confidence,
                    trust,
                    image_path
                ])

            # 9. Retrain check
            need_retrain, img_count = should_retrain()
            if need_retrain:
                subprocess.Popen(["python", "retrain.py"])
                save_status(img_count)

            return render_template(
                "index.html",
                user=session["user"],
                result=level,
                confidence=confidence,
                trust=trust,
                decision=decision,
                detail=detail,
                image_path=image_path
            )

        except Exception as e:
            print("ERROR:", e)
            return render_template(
                "index.html",
                user=session["user"],
                warning="❌ เกิดข้อผิดพลาดในการวิเคราะห์"
            )

    return render_template("index.html", user=session["user"])

# ================== HISTORY ==================
@app.route("/history")
def history():
    if "user" not in session:
        return redirect("/login")

    username = session["user"]["name"]
    records = []

    if os.path.isfile("experience_log.csv"):
        with open("experience_log.csv", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["username"] == username:
                    records.append(row)

    return render_template(
        "history.html",
        username=username,
        records=records
    )

# ================== PROFILE ==================
@app.route("/profile")
def profile():
    if "user" not in session:
        return redirect("/login")

    username = session["user"]["name"]
    total = 0
    last_time = "-"

    if os.path.isfile("experience_log.csv"):
        with open("experience_log.csv", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = [r for r in reader if r["username"] == username]
            total = len(rows)
            if rows:
                rows.sort(key=lambda r: r["datetime"])
                last_time = rows[-1]["datetime"]

    return render_template(
        "profile.html",
        username=username,
        total=total,
        last_time=last_time
    )

# ================== EXPORT CSV ==================
@app.route("/export_csv")
def export_csv():
    if "user" not in session:
        return redirect("/login")

    username = session["user"]["name"]

    def generate():
        yield "\ufeffวันที่,ชื่อผู้ใช้,ตำแหน่ง,ผล,ความมั่นใจ,ความเชื่อมั่น,รูป\n"
        if os.path.isfile("experience_log.csv"):
            with open("experience_log.csv", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row["username"] == username:
                        yield (
                            f'{row["datetime"]},'
                            f'{row["username"]},'
                            f'{row["part"]},'
                            f'{row["result"]},'
                            f'{row["confidence"]},'
                            f'{row["trust"]},'
                            f'{row["image_path"]}\n'
                        )

    return Response(
        generate(),
        mimetype="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename=history_{username}.csv"
        }
    )

# ================== FEEDBACK ==================
@app.route("/feedback", methods=["POST"])
def feedback():
    if "user" not in session:
        return redirect("/login")

    feedback_file = "feedback_log.csv"
    file_exists = os.path.isfile(feedback_file)

    with open(feedback_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "datetime",
                "username",
                "result",
                "confidence",
                "is_correct",
                "comment",
                "image_path"
            ])

        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            session["user"]["name"],
            request.form.get("result"),
            request.form.get("confidence"),
            request.form.get("is_correct"),
            request.form.get("comment", ""),
            request.form.get("image_path")
        ])

    return redirect("/")

# ================== RUN ==================
if __name__ == "__main__":
    app.run(debug=True)
