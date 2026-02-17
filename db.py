import csv
import json
import os
import sqlite3
import tempfile
from datetime import datetime

DB_NAME = os.getenv("DB_NAME", "app.db")


def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def normalize_image_path(path):
    if not path:
        return path
    return str(path).replace("\\", "/")


def get_db():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn


def table_columns(conn, table_name):
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table_name})")
    return {row["name"] for row in cur.fetchall()}


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
        """
        CREATE TABLE IF NOT EXISTS audit_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            username TEXT,
            action TEXT NOT NULL,
            target TEXT,
            details TEXT,
            ip_address TEXT,
            user_agent TEXT,
            created_at TEXT NOT NULL
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS model_registry (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            version TEXT UNIQUE NOT NULL,
            source TEXT,
            notes TEXT,
            is_active INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL
        )
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS cases (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            title TEXT NOT NULL,
            vehicle_info TEXT,
            status TEXT NOT NULL DEFAULT 'open',
            predicted_result TEXT,
            predicted_confidence REAL,
            final_result TEXT,
            reviewer_note TEXT,
            reviewed_by INTEGER,
            reviewed_at TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id),
            FOREIGN KEY(reviewed_by) REFERENCES users(id)
        )
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS case_images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            case_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            image_path TEXT NOT NULL,
            note TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY(case_id) REFERENCES cases(id),
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS notifications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            title TEXT NOT NULL,
            body TEXT NOT NULL,
            is_read INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS analysis_jobs (
            id TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL,
            status TEXT NOT NULL,
            part TEXT,
            image_path TEXT,
            result_json TEXT,
            error_text TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS password_reset_tokens (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            token_hash TEXT UNIQUE NOT NULL,
            expires_at TEXT NOT NULL,
            consumed_at TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS email_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            to_email TEXT NOT NULL,
            subject TEXT NOT NULL,
            status TEXT NOT NULL,
            error_text TEXT,
            meta TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS email_verify_codes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            code_hash TEXT NOT NULL,
            expires_at TEXT NOT NULL,
            consumed_at TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS case_favorites (
            user_id INTEGER NOT NULL,
            case_id INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY(user_id, case_id),
            FOREIGN KEY(user_id) REFERENCES users(id),
            FOREIGN KEY(case_id) REFERENCES cases(id)
        )
        """
    )

    # Online migrations for existing databases.
    user_cols = table_columns(conn, "users")
    if "password_hash" not in user_cols:
        cursor.execute("ALTER TABLE users ADD COLUMN password_hash TEXT")
    if "role" not in user_cols:
        cursor.execute("ALTER TABLE users ADD COLUMN role TEXT NOT NULL DEFAULT 'user'")
    if "is_active" not in user_cols:
        cursor.execute("ALTER TABLE users ADD COLUMN is_active INTEGER NOT NULL DEFAULT 1")
    if "avatar_path" not in user_cols:
        cursor.execute("ALTER TABLE users ADD COLUMN avatar_path TEXT")
    if "email" not in user_cols:
        cursor.execute("ALTER TABLE users ADD COLUMN email TEXT")
    if "age" not in user_cols:
        cursor.execute("ALTER TABLE users ADD COLUMN age INTEGER")
    if "gender" not in user_cols:
        cursor.execute("ALTER TABLE users ADD COLUMN gender TEXT")
    if "onboarding_done" not in user_cols:
        cursor.execute("ALTER TABLE users ADD COLUMN onboarding_done INTEGER NOT NULL DEFAULT 0")
    if "email_verified" not in user_cols:
        cursor.execute("ALTER TABLE users ADD COLUMN email_verified INTEGER NOT NULL DEFAULT 0")

    history_cols = table_columns(conn, "history")
    if "model_version" not in history_cols:
        cursor.execute(
            "ALTER TABLE history ADD COLUMN model_version TEXT NOT NULL DEFAULT 'damage_model.h5'"
        )
    if "est_cost_min" not in history_cols:
        cursor.execute("ALTER TABLE history ADD COLUMN est_cost_min INTEGER")
    if "est_cost_max" not in history_cols:
        cursor.execute("ALTER TABLE history ADD COLUMN est_cost_max INTEGER")
    case_cols = table_columns(conn, "cases")
    if "predicted_result" not in case_cols:
        cursor.execute("ALTER TABLE cases ADD COLUMN predicted_result TEXT")
    if "predicted_confidence" not in case_cols:
        cursor.execute("ALTER TABLE cases ADD COLUMN predicted_confidence REAL")
    if "final_result" not in case_cols:
        cursor.execute("ALTER TABLE cases ADD COLUMN final_result TEXT")
    if "reviewer_note" not in case_cols:
        cursor.execute("ALTER TABLE cases ADD COLUMN reviewer_note TEXT")
    if "reviewed_by" not in case_cols:
        cursor.execute("ALTER TABLE cases ADD COLUMN reviewed_by INTEGER")
    if "reviewed_at" not in case_cols:
        cursor.execute("ALTER TABLE cases ADD COLUMN reviewed_at TEXT")

    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_history_user_created ON history(user_id, created_at)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_feedback_user_created ON feedback(user_id, created_at)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_audit_created ON audit_logs(created_at)"
    )
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_role ON users(role)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_onboarding_done ON users(onboarding_done)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_cases_user_created ON cases(user_id, created_at)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_cases_status_updated ON cases(status, updated_at)")
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_notifications_user_read ON notifications(user_id, is_read, created_at)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_jobs_user_status ON analysis_jobs(user_id, status, created_at)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_reset_tokens_user_exp ON password_reset_tokens(user_id, expires_at)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_email_logs_user_created ON email_logs(user_id, created_at)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_email_verify_user_exp ON email_verify_codes(user_id, expires_at)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_case_fav_case ON case_favorites(case_id)"
    )

    cursor.execute(
        """
        INSERT OR IGNORE INTO model_registry (version, source, notes, is_active, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        ("damage_model.h5", "local", "default base model", 1, now_str()),
    )

    conn.commit()
    conn.close()


def serialize_details(details):
    if details is None:
        return None
    if isinstance(details, str):
        return details
    try:
        return json.dumps(details, ensure_ascii=False)
    except Exception:
        return str(details)


def log_audit(
    action,
    user_id=None,
    username=None,
    target=None,
    details=None,
    ip_address=None,
    user_agent=None,
):
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO audit_logs (
            user_id, username, action, target, details, ip_address, user_agent, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            user_id,
            username,
            action,
            target,
            serialize_details(details),
            ip_address,
            user_agent,
            now_str(),
        ),
    )
    conn.commit()
    conn.close()


def get_recent_audit_logs(limit=50):
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT created_at, username, action, target, details, ip_address
        FROM audit_logs
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (int(limit),),
    )
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def count_admin_users():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) AS c FROM users WHERE role = 'admin' AND is_active = 1")
    value = int(cur.fetchone()["c"] or 0)
    conn.close()
    return value


def list_users_basic(limit=300):
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, username, role, is_active, created_at, last_login
        FROM users
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (int(limit),),
    )
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def update_user_role_status(user_id, role=None, is_active=None):
    if role is None and is_active is None:
        return False
    conn = get_db()
    cur = conn.cursor()
    sets = []
    params = []
    if role is not None:
        sets.append("role = ?")
        params.append(role)
    if is_active is not None:
        sets.append("is_active = ?")
        params.append(1 if int(is_active) == 1 else 0)
    params.append(int(user_id))
    cur.execute(f"UPDATE users SET {', '.join(sets)} WHERE id = ?", tuple(params))
    changed = cur.rowcount
    conn.commit()
    conn.close()
    return changed > 0


def get_or_create_user(username):
    now = now_str()
    conn = get_db()
    cur = conn.cursor()

    cur.execute(
        "SELECT id, username, role, is_active, avatar_path, email, age, gender, onboarding_done, email_verified FROM users WHERE username = ?",
        (username,),
    )
    row = cur.fetchone()

    if row is None:
        cur.execute(
            """
            INSERT INTO users (username, created_at, last_login, role, is_active, onboarding_done)
            VALUES (?, ?, ?, 'user', 1, 0)
            """,
            (username, now, now),
        )
        user_id = cur.lastrowid
    else:
        user_id = row["id"]
        cur.execute("UPDATE users SET last_login = ? WHERE id = ?", (now, user_id))

    conn.commit()
    cur.execute(
        "SELECT id, username, role, is_active, avatar_path, email, age, gender, onboarding_done, email_verified FROM users WHERE id = ?",
        (user_id,),
    )
    user = cur.fetchone()
    conn.close()
    return {
        "id": user["id"],
        "name": user["username"],
        "role": user["role"] or "user",
        "is_active": int(user["is_active"]) == 1,
        "avatar_path": normalize_image_path(user["avatar_path"]),
        "email": user["email"],
        "age": user["age"],
        "gender": user["gender"],
        "onboarding_done": int(user["onboarding_done"] or 0) == 1,
        "email_verified": int(user["email_verified"] or 0) == 1,
    }


def get_user_auth(username):
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, username, password_hash, role, is_active, avatar_path, email, age, gender, onboarding_done, email_verified
        FROM users
        WHERE username = ?
        """,
        (username,),
    )
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None


def create_user(username, password_hash, role="user", avatar_path=None, email=None, age=None, gender=None):
    now = now_str()
    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute(
            """
            INSERT INTO users (username, password_hash, role, is_active, avatar_path, email, age, gender, onboarding_done, email_verified, created_at, last_login)
            VALUES (?, ?, ?, 1, ?, ?, ?, ?, 0, 0, ?, ?)
            """,
            (
                username,
                password_hash,
                role,
                normalize_image_path(avatar_path),
                (email or "").strip().lower() or None,
                int(age) if age is not None else None,
                (gender or "").strip().lower() or None,
                now,
                now,
            ),
        )
        user_id = cur.lastrowid
        conn.commit()
        cur.execute(
            "SELECT id, username, role, is_active, avatar_path, email, age, gender, onboarding_done, email_verified FROM users WHERE id = ?",
            (user_id,),
        )
        row = cur.fetchone()
        return {
            "id": row["id"],
            "name": row["username"],
            "role": row["role"] or "user",
            "is_active": int(row["is_active"]) == 1,
            "avatar_path": normalize_image_path(row["avatar_path"]),
            "email": row["email"],
            "age": row["age"],
            "gender": row["gender"],
            "onboarding_done": int(row["onboarding_done"] or 0) == 1,
            "email_verified": int(row["email_verified"] or 0) == 1,
        }, None
    except sqlite3.IntegrityError:
        return None, "username_exists"
    finally:
        conn.close()


def set_user_password(username, password_hash):
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        "UPDATE users SET password_hash = ? WHERE username = ?",
        (password_hash, username),
    )
    changed = cur.rowcount
    conn.commit()
    conn.close()
    return changed > 0


def set_user_avatar(user_id, avatar_path):
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        "UPDATE users SET avatar_path = ? WHERE id = ?",
        (normalize_image_path(avatar_path), user_id),
    )
    changed = cur.rowcount
    conn.commit()
    conn.close()
    return changed > 0


def get_user_profile(user_id):
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, username, role, is_active, avatar_path, email, age, gender, onboarding_done, email_verified
        FROM users
        WHERE id = ?
        """,
        (int(user_id),),
    )
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    return {
        "id": row["id"],
        "username": row["username"],
        "role": row["role"] or "user",
        "is_active": int(row["is_active"]) == 1,
        "avatar_path": normalize_image_path(row["avatar_path"]),
        "email": row["email"],
        "age": row["age"],
        "gender": row["gender"],
        "onboarding_done": int(row["onboarding_done"] or 0) == 1,
        "email_verified": int(row["email_verified"] or 0) == 1,
    }


def email_in_use(email, exclude_user_id=None):
    norm_email = (email or "").strip().lower()
    if not norm_email:
        return False
    conn = get_db()
    cur = conn.cursor()
    if exclude_user_id is None:
        cur.execute("SELECT 1 FROM users WHERE lower(email) = lower(?) LIMIT 1", (norm_email,))
    else:
        cur.execute(
            "SELECT 1 FROM users WHERE lower(email) = lower(?) AND id <> ? LIMIT 1",
            (norm_email, int(exclude_user_id)),
        )
    row = cur.fetchone()
    conn.close()
    return row is not None


def set_user_profile(user_id, email=None, age=None, gender=None):
    sets = []
    params = []
    if email is not None:
        sets.append("email = ?")
        params.append((email or "").strip().lower() or None)
    if age is not None:
        sets.append("age = ?")
        params.append(int(age))
    if gender is not None:
        sets.append("gender = ?")
        params.append((gender or "").strip().lower() or None)
    if not sets:
        return False
    conn = get_db()
    cur = conn.cursor()
    params.append(int(user_id))
    cur.execute(f"UPDATE users SET {', '.join(sets)} WHERE id = ?", tuple(params))
    changed = cur.rowcount
    conn.commit()
    conn.close()
    return changed > 0


def set_onboarding_done(user_id, done=True):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("UPDATE users SET onboarding_done = ? WHERE id = ?", (1 if done else 0, int(user_id)))
    changed = cur.rowcount
    conn.commit()
    conn.close()
    return changed > 0


def get_user_by_email(email):
    value = (email or "").strip().lower()
    if not value:
        return None
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, username, role, is_active, avatar_path, email, age, gender, onboarding_done, email_verified
        FROM users
        WHERE lower(email) = lower(?)
        LIMIT 1
        """,
        (value,),
    )
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    return {
        "id": row["id"],
        "name": row["username"],
        "role": row["role"] or "user",
        "is_active": int(row["is_active"]) == 1,
        "avatar_path": normalize_image_path(row["avatar_path"]),
        "email": row["email"],
        "age": row["age"],
        "gender": row["gender"],
        "onboarding_done": int(row["onboarding_done"] or 0) == 1,
        "email_verified": int(row["email_verified"] or 0) == 1,
    }


def create_password_reset_token(user_id, token_hash, expires_at):
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO password_reset_tokens (user_id, token_hash, expires_at, consumed_at, created_at)
        VALUES (?, ?, ?, NULL, ?)
        """,
        (int(user_id), token_hash, expires_at, now_str()),
    )
    conn.commit()
    conn.close()


def get_password_reset_token(token_hash):
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, user_id, token_hash, expires_at, consumed_at, created_at
        FROM password_reset_tokens
        WHERE token_hash = ?
        LIMIT 1
        """,
        (token_hash,),
    )
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None


def consume_password_reset_token(token_hash):
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        "UPDATE password_reset_tokens SET consumed_at = ? WHERE token_hash = ? AND consumed_at IS NULL",
        (now_str(), token_hash),
    )
    changed = cur.rowcount
    conn.commit()
    conn.close()
    return changed > 0


def add_email_log(user_id, to_email, subject, status, error_text=None, meta=None):
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO email_logs (user_id, to_email, subject, status, error_text, meta, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            int(user_id) if user_id is not None else None,
            (to_email or "").strip(),
            (subject or "").strip(),
            (status or "").strip().lower(),
            error_text,
            serialize_details(meta),
            now_str(),
        ),
    )
    conn.commit()
    conn.close()


def list_user_email_logs(user_id, limit=20):
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, to_email, subject, status, error_text, meta, created_at
        FROM email_logs
        WHERE user_id = ?
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (int(user_id), int(limit)),
    )
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def get_last_user_email_log(user_id):
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, to_email, subject, status, error_text, meta, created_at
        FROM email_logs
        WHERE user_id = ?
        ORDER BY created_at DESC
        LIMIT 1
        """,
        (int(user_id),),
    )
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None


def create_email_verify_code(user_id, code_hash, expires_at):
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO email_verify_codes (user_id, code_hash, expires_at, consumed_at, created_at)
        VALUES (?, ?, ?, NULL, ?)
        """,
        (int(user_id), code_hash, expires_at, now_str()),
    )
    conn.commit()
    conn.close()


def get_latest_active_email_verify_code(user_id):
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, user_id, code_hash, expires_at, consumed_at, created_at
        FROM email_verify_codes
        WHERE user_id = ? AND consumed_at IS NULL
        ORDER BY created_at DESC
        LIMIT 1
        """,
        (int(user_id),),
    )
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None


def consume_email_verify_code(code_id):
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        "UPDATE email_verify_codes SET consumed_at = ? WHERE id = ? AND consumed_at IS NULL",
        (now_str(), int(code_id)),
    )
    changed = cur.rowcount
    conn.commit()
    conn.close()
    return changed > 0


def set_email_verified(user_id, verified=True):
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        "UPDATE users SET email_verified = ? WHERE id = ?",
        (1 if verified else 0, int(user_id)),
    )
    changed = cur.rowcount
    conn.commit()
    conn.close()
    return changed > 0


def is_case_favorite(user_id, case_id):
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        "SELECT 1 FROM case_favorites WHERE user_id = ? AND case_id = ? LIMIT 1",
        (int(user_id), int(case_id)),
    )
    row = cur.fetchone()
    conn.close()
    return row is not None


def toggle_case_favorite(user_id, case_id):
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        "SELECT 1 FROM case_favorites WHERE user_id = ? AND case_id = ? LIMIT 1",
        (int(user_id), int(case_id)),
    )
    row = cur.fetchone()
    if row:
        cur.execute(
            "DELETE FROM case_favorites WHERE user_id = ? AND case_id = ?",
            (int(user_id), int(case_id)),
        )
        conn.commit()
        conn.close()
        return False
    cur.execute(
        "INSERT INTO case_favorites (user_id, case_id, created_at) VALUES (?, ?, ?)",
        (int(user_id), int(case_id), now_str()),
    )
    conn.commit()
    conn.close()
    return True


def list_user_favorite_case_ids(user_id):
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        "SELECT case_id FROM case_favorites WHERE user_id = ? ORDER BY created_at DESC",
        (int(user_id),),
    )
    rows = [int(r["case_id"]) for r in cur.fetchall()]
    conn.close()
    return rows


def add_history(
    user_id,
    part,
    result,
    confidence,
    trust,
    image_path,
    model_version="damage_model.h5",
    est_cost_min=None,
    est_cost_max=None,
):
    now = now_str()
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO history (
            user_id, part, result, confidence, trust, image_path, model_version, est_cost_min, est_cost_max, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            user_id,
            part,
            result,
            confidence,
            trust,
            normalize_image_path(image_path),
            model_version,
            est_cost_min,
            est_cost_max,
            now,
        ),
    )
    conn.commit()
    conn.close()


def add_feedback(user_id, result, confidence, is_correct, comment, image_path):
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
            now_str(),
        ),
    )
    conn.commit()
    conn.close()


def _build_history_filters(user_id, part=None, result=None, date_from=None, date_to=None):
    where = " WHERE user_id = ?"
    params = [int(user_id)]
    if part:
        where += " AND part = ?"
        params.append(part)
    if result:
        where += " AND result = ?"
        params.append(result)
    if date_from:
        where += " AND created_at >= ?"
        params.append(f"{date_from} 00:00:00")
    if date_to:
        where += " AND created_at <= ?"
        params.append(f"{date_to} 23:59:59")
    return where, params


def get_user_history(
    user_id,
    part=None,
    result=None,
    date_from=None,
    date_to=None,
    limit=None,
    offset=None,
    sort_by="date_desc",
):
    conn = get_db()
    cur = conn.cursor()
    query = """
        SELECT
            id,
            created_at AS datetime,
            part,
            result,
            confidence,
            trust,
            image_path,
            model_version,
            est_cost_min,
            est_cost_max
        FROM history
    """
    where, params = _build_history_filters(
        user_id,
        part=part,
        result=result,
        date_from=date_from,
        date_to=date_to,
    )
    query += where

    if sort_by == "confidence_desc":
        query += " ORDER BY confidence DESC, created_at DESC"
    elif sort_by == "confidence_asc":
        query += " ORDER BY confidence ASC, created_at DESC"
    elif sort_by == "severity_desc":
        query += (
            " ORDER BY CASE result "
            "WHEN 'high' THEN 3 WHEN 'medium' THEN 2 WHEN 'low' THEN 1 ELSE 0 END DESC, created_at DESC"
        )
    elif sort_by == "severity_asc":
        query += (
            " ORDER BY CASE result "
            "WHEN 'high' THEN 3 WHEN 'medium' THEN 2 WHEN 'low' THEN 1 ELSE 0 END ASC, created_at DESC"
        )
    elif sort_by == "date_asc":
        query += " ORDER BY created_at ASC"
    else:
        query += " ORDER BY created_at DESC"

    if limit is not None:
        query += " LIMIT ?"
        params.append(int(limit))
    if offset is not None:
        query += " OFFSET ?"
        params.append(int(offset))
    cur.execute(query, tuple(params))
    rows = [dict(r) for r in cur.fetchall()]
    for row in rows:
        row["image_path"] = normalize_image_path(row.get("image_path"))
    conn.close()
    return rows


def count_user_history(user_id, part=None, result=None, date_from=None, date_to=None):
    conn = get_db()
    cur = conn.cursor()
    where, params = _build_history_filters(
        user_id,
        part=part,
        result=result,
        date_from=date_from,
        date_to=date_to,
    )
    cur.execute(f"SELECT COUNT(*) AS c FROM history{where}", tuple(params))
    row = cur.fetchone()
    conn.close()
    return int((row or {"c": 0})["c"] or 0)


def get_user_history_summary(user_id, part=None, date_from=None, date_to=None):
    conn = get_db()
    cur = conn.cursor()
    where, params = _build_history_filters(
        user_id,
        part=part,
        result=None,
        date_from=date_from,
        date_to=date_to,
    )
    cur.execute(
        f"""
        SELECT
            COUNT(*) AS total,
            SUM(CASE WHEN result = 'high' THEN 1 ELSE 0 END) AS high_count,
            SUM(CASE WHEN result = 'medium' THEN 1 ELSE 0 END) AS medium_count,
            SUM(CASE WHEN result = 'low' THEN 1 ELSE 0 END) AS low_count
        FROM history
        {where}
        """,
        tuple(params),
    )
    row = dict(cur.fetchone() or {})
    conn.close()
    return {
        "total": int(row.get("total") or 0),
        "high": int(row.get("high_count") or 0),
        "medium": int(row.get("medium_count") or 0),
        "low": int(row.get("low_count") or 0),
    }


def get_user_history_item(user_id, history_id):
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT
            id,
            created_at AS datetime,
            part,
            result,
            confidence,
            trust,
            image_path,
            model_version,
            est_cost_min,
            est_cost_max
        FROM history
        WHERE id = ? AND user_id = ?
        LIMIT 1
        """,
        (int(history_id), int(user_id)),
    )
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    payload = dict(row)
    payload["image_path"] = normalize_image_path(payload.get("image_path"))
    return payload


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

    cur.execute(
        """
        SELECT
            SUM(CASE WHEN result = 'high' THEN 1 ELSE 0 END) AS high_count,
            SUM(CASE WHEN result = 'medium' THEN 1 ELSE 0 END) AS medium_count,
            SUM(CASE WHEN result = 'low' THEN 1 ELSE 0 END) AS low_count
        FROM history
        WHERE user_id = ?
        """,
        (user_id,),
    )
    result_row = cur.fetchone() or {}

    conn.close()
    return {
        "avg_conf": avg_conf,
        "top_part": top_part,
        "high_count": int(result_row["high_count"] or 0),
        "medium_count": int(result_row["medium_count"] or 0),
        "low_count": int(result_row["low_count"] or 0),
    }


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
        SELECT
            created_at AS datetime,
            part,
            result,
            confidence,
            trust,
            image_path,
            model_version,
            est_cost_min,
            est_cost_max
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

    cur.execute("SELECT COUNT(*) AS c FROM audit_logs")
    audit_events = int(cur.fetchone()["c"])
    cur.execute("SELECT COUNT(*) AS c FROM notifications WHERE is_read = 0")
    unread_notifications = int(cur.fetchone()["c"])
    cur.execute("SELECT COUNT(*) AS c FROM analysis_jobs WHERE status = 'queued'")
    queued_jobs = int(cur.fetchone()["c"])
    cur.execute("SELECT COUNT(*) AS c FROM analysis_jobs WHERE status = 'running'")
    running_jobs = int(cur.fetchone()["c"])

    conn.close()
    return {
        "total_users": total_users,
        "total_analyses": total_analyses,
        "avg_conf": avg_conf,
        "last_24h": last_24h,
        "distribution": distribution,
        "audit_events": audit_events,
        "unread_notifications": unread_notifications,
        "queued_jobs": queued_jobs,
        "running_jobs": running_jobs,
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


def create_case(
    user_id,
    title,
    vehicle_info=None,
    status="open",
    predicted_result=None,
    predicted_confidence=None,
):
    now = now_str()
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO cases (
            user_id, title, vehicle_info, status, predicted_result, predicted_confidence, created_at, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            user_id,
            title,
            vehicle_info,
            status or "open",
            predicted_result,
            predicted_confidence,
            now,
            now,
        ),
    )
    case_id = cur.lastrowid
    conn.commit()
    conn.close()
    return case_id


def list_user_cases(user_id, limit=50):
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT
            id,
            title,
            vehicle_info,
            status,
            predicted_result,
            predicted_confidence,
            final_result,
            reviewer_note,
            reviewed_by,
            reviewed_at,
            created_at,
            updated_at,
            (
                SELECT COUNT(1)
                FROM case_images ci
                WHERE ci.case_id = cases.id
            ) AS image_count
        FROM cases
        WHERE user_id = ?
        ORDER BY updated_at DESC
        LIMIT ?
        """,
        (user_id, int(limit)),
    )
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def delete_case(user_id, case_id):
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id
        FROM cases
        WHERE id = ? AND user_id = ?
        """,
        (int(case_id), int(user_id)),
    )
    row = cur.fetchone()
    if not row:
        conn.close()
        return None

    cur.execute(
        """
        SELECT image_path
        FROM case_images
        WHERE case_id = ? AND user_id = ?
        """,
        (int(case_id), int(user_id)),
    )
    image_paths = [normalize_image_path(r["image_path"]) for r in cur.fetchall()]

    cur.execute("DELETE FROM case_images WHERE case_id = ? AND user_id = ?", (int(case_id), int(user_id)))
    cur.execute("DELETE FROM case_favorites WHERE case_id = ? AND user_id = ?", (int(case_id), int(user_id)))
    cur.execute("DELETE FROM cases WHERE id = ? AND user_id = ?", (int(case_id), int(user_id)))
    conn.commit()
    conn.close()
    return {"case_id": int(case_id), "image_paths": image_paths}


def list_cases_for_review(limit=100):
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT
            c.id,
            c.user_id,
            u.username,
            c.title,
            c.vehicle_info,
            c.status,
            c.predicted_result,
            c.predicted_confidence,
            c.final_result,
            c.reviewer_note,
            c.reviewed_by,
            c.reviewed_at,
            c.created_at,
            c.updated_at
        FROM cases c
        JOIN users u ON u.id = c.user_id
        WHERE c.status = 'needs_review'
        ORDER BY c.updated_at DESC
        LIMIT ?
        """,
        (int(limit),),
    )
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def list_recent_reviewed_cases(limit=100):
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT
            c.id,
            c.user_id,
            owner.username AS owner_username,
            c.title,
            c.status,
            c.predicted_result,
            c.predicted_confidence,
            c.final_result,
            c.reviewer_note,
            c.reviewed_by,
            reviewer.username AS reviewer_username,
            c.reviewed_at,
            c.updated_at
        FROM cases c
        JOIN users owner ON owner.id = c.user_id
        LEFT JOIN users reviewer ON reviewer.id = c.reviewed_by
        WHERE c.status = 'reviewed'
        ORDER BY c.reviewed_at DESC, c.updated_at DESC
        LIMIT ?
        """,
        (int(limit),),
    )
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def add_case_image(case_id, user_id, image_path, note=None):
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO case_images (case_id, user_id, image_path, note, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (case_id, user_id, normalize_image_path(image_path), note, now_str()),
    )
    cur.execute("UPDATE cases SET updated_at = ? WHERE id = ?", (now_str(), case_id))
    conn.commit()
    conn.close()


def set_case_prediction(case_id, user_id, predicted_result, predicted_confidence):
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE cases
        SET predicted_result = ?, predicted_confidence = ?, updated_at = ?
        WHERE id = ? AND user_id = ?
        """,
        (
            (predicted_result or "").strip().lower() or None,
            float(predicted_confidence) if predicted_confidence is not None else None,
            now_str(),
            int(case_id),
            int(user_id),
        ),
    )
    conn.commit()
    conn.close()


def delete_case_image(case_id, user_id, image_id):
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT image_path
        FROM case_images
        WHERE id = ? AND case_id = ? AND user_id = ?
        """,
        (image_id, case_id, user_id),
    )
    row = cur.fetchone()
    if not row:
        conn.close()
        return None

    image_path = normalize_image_path(row["image_path"])
    cur.execute(
        "DELETE FROM case_images WHERE id = ? AND case_id = ? AND user_id = ?",
        (image_id, case_id, user_id),
    )
    cur.execute("UPDATE cases SET updated_at = ? WHERE id = ?", (now_str(), case_id))
    conn.commit()
    conn.close()
    return image_path


def get_case_detail(case_id, user_id):
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT
            id,
            user_id,
            title,
            vehicle_info,
            status,
            predicted_result,
            predicted_confidence,
            final_result,
            reviewer_note,
            reviewed_by,
            reviewed_at,
            created_at,
            updated_at
        FROM cases
        WHERE id = ? AND user_id = ?
        """,
        (case_id, user_id),
    )
    case_row = cur.fetchone()
    if not case_row:
        conn.close()
        return None

    cur.execute(
        """
        SELECT id, image_path, note, created_at
        FROM case_images
        WHERE case_id = ? AND user_id = ?
        ORDER BY created_at DESC
        """,
        (case_id, user_id),
    )
    images = [dict(r) for r in cur.fetchall()]
    for row in images:
        row["image_path"] = normalize_image_path(row.get("image_path"))
    conn.close()

    payload = dict(case_row)
    payload["images"] = images
    return payload


def get_case_for_review(case_id):
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT
            c.id,
            c.user_id,
            u.username,
            c.title,
            c.vehicle_info,
            c.status,
            c.predicted_result,
            c.predicted_confidence,
            c.final_result,
            c.reviewer_note,
            c.reviewed_by,
            c.reviewed_at,
            c.created_at,
            c.updated_at
        FROM cases c
        JOIN users u ON u.id = c.user_id
        WHERE c.id = ?
        """,
        (case_id,),
    )
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None


def review_case(case_id, reviewer_user_id, final_result, reviewer_note=None):
    now = now_str()
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE cases
        SET
            status = 'reviewed',
            final_result = ?,
            reviewer_note = ?,
            reviewed_by = ?,
            reviewed_at = ?,
            updated_at = ?
        WHERE id = ? AND status = 'needs_review'
        """,
        (final_result, reviewer_note, reviewer_user_id, now, now, case_id),
    )
    changed = cur.rowcount
    conn.commit()
    conn.close()
    return changed > 0


def create_notification(user_id, title, body):
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO notifications (user_id, title, body, is_read, created_at)
        VALUES (?, ?, ?, 0, ?)
        """,
        (user_id, title, body, now_str()),
    )
    conn.commit()
    conn.close()


def get_notifications(user_id, unread_only=False, limit=50):
    conn = get_db()
    cur = conn.cursor()
    query = """
        SELECT id, title, body, is_read, created_at
        FROM notifications
        WHERE user_id = ?
    """
    params = [user_id]
    if unread_only:
        query += " AND is_read = 0"
    query += " ORDER BY created_at DESC LIMIT ?"
    params.append(int(limit))
    cur.execute(query, tuple(params))
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def mark_notification_read(notification_id, user_id):
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        "UPDATE notifications SET is_read = 1 WHERE id = ? AND user_id = ?",
        (notification_id, user_id),
    )
    changed = cur.rowcount
    conn.commit()
    conn.close()
    return changed > 0


def create_analysis_job(job_id, user_id, part):
    now = now_str()
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO analysis_jobs (
            id, user_id, status, part, created_at, updated_at
        ) VALUES (?, ?, 'queued', ?, ?, ?)
        """,
        (job_id, user_id, part, now, now),
    )
    conn.commit()
    conn.close()


def update_analysis_job(job_id, status=None, image_path=None, result_json=None, error_text=None):
    conn = get_db()
    cur = conn.cursor()
    sets = ["updated_at = ?"]
    params = [now_str()]
    if status is not None:
        sets.append("status = ?")
        params.append(status)
    if image_path is not None:
        sets.append("image_path = ?")
        params.append(normalize_image_path(image_path))
    if result_json is not None:
        sets.append("result_json = ?")
        if isinstance(result_json, str):
            params.append(result_json)
        else:
            params.append(json.dumps(result_json, ensure_ascii=False))
    if error_text is not None:
        sets.append("error_text = ?")
        params.append(error_text)
    params.append(job_id)
    cur.execute(f"UPDATE analysis_jobs SET {', '.join(sets)} WHERE id = ?", tuple(params))
    conn.commit()
    conn.close()


def get_analysis_job(job_id, user_id):
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, status, part, image_path, result_json, error_text, created_at, updated_at
        FROM analysis_jobs
        WHERE id = ? AND user_id = ?
        """,
        (job_id, user_id),
    )
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    payload = dict(row)
    if payload.get("result_json"):
        try:
            payload["result"] = json.loads(payload["result_json"])
        except Exception:
            payload["result"] = payload["result_json"]
    payload["image_path"] = normalize_image_path(payload.get("image_path"))
    return payload


def get_analysis_job_stats():
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT status, COUNT(*) AS cnt
        FROM analysis_jobs
        GROUP BY status
        """
    )
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


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
        now = now_str()
        cur.execute("SELECT id FROM users WHERE username = ?", (username,))
        row = cur.fetchone()
        if row is None:
            cur.execute(
                """
                INSERT INTO users (username, created_at, last_login, role, is_active)
                VALUES (?, ?, ?, 'user', 1)
                """,
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
                    INSERT INTO history (
                        user_id, part, result, confidence, trust, image_path, model_version, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        user_id,
                        row.get("part", ""),
                        row.get("result", ""),
                        conf,
                        row.get("trust", ""),
                        row.get("image_path", ""),
                        row.get("model_version", "damage_model.h5"),
                        row.get("datetime") or now_str(),
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
                        row.get("datetime") or now_str(),
                    ),
                )

    conn.commit()
    conn.close()
