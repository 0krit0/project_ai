import sqlite3
from datetime import datetime
import csv
import os
import tempfile

DB_NAME = "app.db"


def normalize_image_path(path):
    if not path:
        return path
    return str(path).replace("\\", "/")


def get_db():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            created_at TEXT NOT NULL,
            last_login TEXT NOT NULL
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            part TEXT NOT NULL,
            result TEXT NOT NULL,
            confidence REAL NOT NULL,
            trust TEXT NOT NULL,
            image_path TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            result TEXT,
            confidence REAL,
            is_correct TEXT,
            comment TEXT,
            image_path TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
        """
    )

    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_history_user_created ON history(user_id, created_at)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_feedback_user_created ON feedback(user_id, created_at)"
    )

    conn.commit()
    conn.close()


def get_or_create_user(username):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = get_db()
    cur = conn.cursor()

    cur.execute("SELECT id, username FROM users WHERE username = ?", (username,))
    row = cur.fetchone()

    if row is None:
        cur.execute(
            "INSERT INTO users (username, created_at, last_login) VALUES (?, ?, ?)",
            (username, now, now),
        )
        user_id = cur.lastrowid
    else:
        user_id = row["id"]
        cur.execute(
            "UPDATE users SET last_login = ? WHERE id = ?",
            (now, user_id),
        )

    conn.commit()
    cur.execute("SELECT id, username FROM users WHERE id = ?", (user_id,))
    user = cur.fetchone()
    conn.close()
    return {"id": user["id"], "name": user["username"]}


def add_history(user_id, part, result, confidence, trust, image_path):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO history (user_id, part, result, confidence, trust, image_path, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (user_id, part, result, confidence, trust, normalize_image_path(image_path), now),
    )
    conn.commit()
    conn.close()


def add_feedback(user_id, result, confidence, is_correct, comment, image_path):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO feedback (user_id, result, confidence, is_correct, comment, image_path, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            user_id,
            result,
            confidence,
            is_correct,
            comment,
            normalize_image_path(image_path),
            now,
        ),
    )
    conn.commit()
    conn.close()


def get_user_history(
    user_id, part=None, result=None, date_from=None, date_to=None, limit=None
):
    conn = get_db()
    cur = conn.cursor()
    query = """
        SELECT created_at AS datetime, part, result, confidence, trust, image_path
        FROM history
        WHERE user_id = ?
    """
    params = [user_id]

    if part:
        query += " AND part = ?"
        params.append(part)
    if result:
        query += " AND result = ?"
        params.append(result)
    if date_from:
        query += " AND created_at >= ?"
        params.append(f"{date_from} 00:00:00")
    if date_to:
        query += " AND created_at <= ?"
        params.append(f"{date_to} 23:59:59")

    query += " ORDER BY created_at DESC"
    if limit is not None:
        query += " LIMIT ?"
        params.append(int(limit))
    cur.execute(query, tuple(params))
    rows = [dict(r) for r in cur.fetchall()]
    for row in rows:
        row["image_path"] = normalize_image_path(row.get("image_path"))
    conn.close()
    return rows


def get_profile_summary(user_id):
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        "SELECT COUNT(*) AS total, MAX(created_at) AS last_time FROM history WHERE user_id = ?",
        (user_id,),
    )
    row = cur.fetchone()
    conn.close()
    total = int(row["total"]) if row and row["total"] is not None else 0
    last_time = row["last_time"] if row and row["last_time"] else "-"
    return total, last_time


def get_profile_insights(user_id):
    conn = get_db()
    cur = conn.cursor()

    cur.execute(
        "SELECT AVG(confidence) AS avg_conf FROM history WHERE user_id = ?",
        (user_id,),
    )
    avg_row = cur.fetchone()
    avg_conf = (
        round(float(avg_row["avg_conf"]), 2)
        if avg_row and avg_row["avg_conf"] is not None
        else 0.0
    )

    cur.execute(
        """
        SELECT part, COUNT(*) AS cnt
        FROM history
        WHERE user_id = ?
        GROUP BY part
        ORDER BY cnt DESC
        LIMIT 1
        """,
        (user_id,),
    )
    part_row = cur.fetchone()
    top_part = part_row["part"] if part_row else "-"

    conn.close()
    return {"avg_conf": avg_conf, "top_part": top_part}


def get_dashboard_data(user_id):
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT
            COUNT(*) AS total,
            AVG(confidence) AS avg_conf,
            MAX(created_at) AS last_time
        FROM history
        WHERE user_id = ?
        """,
        (user_id,),
    )
    summary = cur.fetchone()

    cur.execute(
        """
        SELECT created_at AS datetime, part, result, confidence, trust, image_path
        FROM history
        WHERE user_id = ?
        ORDER BY created_at DESC
        LIMIT 3
        """,
        (user_id,),
    )
    recent = [dict(r) for r in cur.fetchall()]
    for row in recent:
        row["image_path"] = normalize_image_path(row.get("image_path"))
    conn.close()

    quick_stats = {
        "total": int(summary["total"]) if summary and summary["total"] is not None else 0,
        "avg_conf": round(float(summary["avg_conf"]), 2)
        if summary and summary["avg_conf"] is not None
        else 0.0,
        "last_time": summary["last_time"] if summary and summary["last_time"] else "-",
    }
    return quick_stats, recent


def get_admin_metrics():
    conn = get_db()
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) AS c FROM users")
    total_users = int(cur.fetchone()["c"])

    cur.execute("SELECT COUNT(*) AS c FROM history")
    total_analyses = int(cur.fetchone()["c"])

    cur.execute("SELECT AVG(confidence) AS a FROM history")
    avg_row = cur.fetchone()
    avg_conf = (
        round(float(avg_row["a"]), 2) if avg_row and avg_row["a"] is not None else 0.0
    )

    cur.execute(
        """
        SELECT COUNT(*) AS c
        FROM history
        WHERE datetime(created_at) >= datetime('now', '-1 day')
        """
    )
    last_24h = int(cur.fetchone()["c"])

    cur.execute(
        """
        SELECT result, COUNT(*) AS cnt
        FROM history
        GROUP BY result
        ORDER BY cnt DESC
        """
    )
    distribution = [dict(r) for r in cur.fetchall()]

    conn.close()
    return {
        "total_users": total_users,
        "total_analyses": total_analyses,
        "avg_conf": avg_conf,
        "last_24h": last_24h,
        "distribution": distribution,
    }


def get_recent_feedback(limit=20):
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT
            f.created_at AS datetime,
            u.username,
            f.result,
            f.confidence,
            f.is_correct,
            f.comment,
            f.image_path
        FROM feedback f
        JOIN users u ON u.id = f.user_id
        ORDER BY f.created_at DESC
        LIMIT ?
        """,
        (int(limit),),
    )
    rows = [dict(r) for r in cur.fetchall()]
    for row in rows:
        row["image_path"] = normalize_image_path(row.get("image_path"))
    conn.close()
    return rows


def restore_database_from_upload(file_storage):
    if file_storage is None:
        return "ไม่พบไฟล์ฐานข้อมูลที่อัปโหลด"
    filename = (file_storage.filename or "").lower()
    if not filename.endswith(".db"):
        return "รองรับเฉพาะไฟล์ .db เท่านั้น"

    with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp:
        temp_path = tmp.name
        file_storage.save(temp_path)

    try:
        src = sqlite3.connect(temp_path)
        dst = sqlite3.connect(DB_NAME)
        src.backup(dst)
        src.close()
        dst.close()
    except Exception as e:
        return f"restore ไม่สำเร็จ: {e}"
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass

    return None


def migrate_from_csv():
    conn = get_db()
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) AS cnt FROM history")
    has_history = int(cur.fetchone()["cnt"]) > 0
    cur.execute("SELECT COUNT(*) AS cnt FROM feedback")
    has_feedback = int(cur.fetchone()["cnt"]) > 0

    if (has_history and has_feedback) or (
        not os.path.isfile("experience_log.csv") and not os.path.isfile("feedback_log.csv")
    ):
        conn.close()
        return

    user_cache = {}

    def get_user_id(username):
        if username in user_cache:
            return user_cache[username]
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cur.execute("SELECT id FROM users WHERE username = ?", (username,))
        row = cur.fetchone()
        if row is None:
            cur.execute(
                "INSERT INTO users (username, created_at, last_login) VALUES (?, ?, ?)",
                (username, now, now),
            )
            user_id = cur.lastrowid
        else:
            user_id = row["id"]
        user_cache[username] = user_id
        return user_id

    if not has_history and os.path.isfile("experience_log.csv"):
        with open("experience_log.csv", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                username = (row.get("username") or "").strip()
                if not username:
                    continue
                user_id = get_user_id(username)
                try:
                    conf = float(row.get("confidence", 0))
                except (TypeError, ValueError):
                    conf = 0.0
                cur.execute(
                    """
                    INSERT INTO history (user_id, part, result, confidence, trust, image_path, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        user_id,
                        row.get("part", ""),
                        row.get("result", ""),
                        conf,
                        row.get("trust", ""),
                        row.get("image_path", ""),
                        row.get("datetime") or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    ),
                )

    if not has_feedback and os.path.isfile("feedback_log.csv"):
        with open("feedback_log.csv", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                username = (row.get("username") or "").strip()
                if not username:
                    continue
                user_id = get_user_id(username)
                try:
                    conf = float(row.get("confidence")) if row.get("confidence") else None
                except (TypeError, ValueError):
                    conf = None
                cur.execute(
                    """
                    INSERT INTO feedback (user_id, result, confidence, is_correct, comment, image_path, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        user_id,
                        row.get("result"),
                        conf,
                        row.get("is_correct"),
                        row.get("comment", ""),
                        row.get("image_path"),
                        row.get("datetime") or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    ),
                )

    conn.commit()
    conn.close()
