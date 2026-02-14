import io
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
from PIL import Image

import app
import db


class IntegrationFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app.init_db()
        app.app.config["TESTING"] = True

    def setUp(self):
        app.ADMIN_USERNAME = "it_admin_root"
        self.client = app.app.test_client()
        self._cleanup_integration_users()

    def tearDown(self):
        self._cleanup_integration_users()

    def _cleanup_integration_users(self):
        conn = db.get_db()
        cur = conn.cursor()
        cur.execute("SELECT id FROM users WHERE username LIKE 'it_%'")
        user_ids = [r[0] for r in cur.fetchall()]
        for uid in user_ids:
            cur.execute("DELETE FROM feedback WHERE user_id = ?", (uid,))
            cur.execute("DELETE FROM history WHERE user_id = ?", (uid,))
            cur.execute("DELETE FROM users WHERE id = ?", (uid,))
        conn.commit()
        conn.close()

    def _make_image_bytes(self):
        img = Image.new("RGB", (224, 224), (120, 120, 120))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)
        return buf

    def _valid_part(self):
        return next(iter(app.RULES.keys()))

    def test_full_user_flow(self):
        mock_model = SimpleNamespace(
            predict=mock.Mock(return_value=np.array([[0.1, 0.8, 0.1]]))
        )
        with mock.patch.object(
            app, "get_model", return_value=mock_model
        ), mock.patch.object(app, "should_retrain", return_value=(False, 0)), mock.patch.object(
            app, "generate_heatmap_overlay", return_value=None
        ):
            register_resp = self.client.post(
                "/login",
                data={"mode": "register", "username": "it_flow_user", "password": "secret123"},
                follow_redirects=True,
            )
            self.assertEqual(register_resp.status_code, 200)

            analyze_resp = self.client.post(
                "/",
                data={
                    "part": self._valid_part(),
                    "file": (self._make_image_bytes(), "sample.jpg"),
                },
                content_type="multipart/form-data",
                follow_redirects=True,
            )
            self.assertEqual(analyze_resp.status_code, 200)
            text = analyze_resp.data.decode("utf-8", errors="replace")
            self.assertIn("Top-3", text)

            history_resp = self.client.get("/history")
            profile_resp = self.client.get("/profile")
            export_resp = self.client.get("/export_csv")
            feedback_resp = self.client.post(
                "/feedback",
                data={
                    "result": "low",
                    "confidence": "88.5",
                    "is_correct": "yes",
                    "comment": "looks good",
                    "image_path": "feedback_images/low/test.jpg",
                },
            )

            self.assertEqual(history_resp.status_code, 200)
            self.assertEqual(profile_resp.status_code, 200)
            self.assertEqual(export_resp.status_code, 200)
            self.assertEqual(feedback_resp.status_code, 302)

            conn = db.get_db()
            cur = conn.cursor()
            cur.execute("SELECT id FROM users WHERE username = ?", ("it_flow_user",))
            user_row = cur.fetchone()
            self.assertIsNotNone(user_row)
            user_id = user_row[0]
            cur.execute("SELECT COUNT(*) FROM history WHERE user_id = ?", (user_id,))
            self.assertEqual(cur.fetchone()[0], 1)
            cur.execute("SELECT COUNT(*) FROM feedback WHERE user_id = ?", (user_id,))
            self.assertEqual(cur.fetchone()[0], 1)
            conn.close()

    def test_upload_invalid_extension(self):
        self.client.post(
            "/login",
            data={"mode": "register", "username": "it_bad_ext", "password": "secret123"},
            follow_redirects=True,
        )
        bad_file = io.BytesIO(b"not-image")
        resp = self.client.post(
            "/",
            data={"part": self._valid_part(), "file": (bad_file, "bad.txt")},
            content_type="multipart/form-data",
            follow_redirects=True,
        )
        self.assertEqual(resp.status_code, 200)
        text = resp.data.decode("utf-8", errors="replace")
        self.assertIn(".jpg", text)

    def test_admin_page_access(self):
        self.client.post(
            "/login",
            data={"mode": "register", "username": "it_admin", "password": "secret123"},
            follow_redirects=True,
        )
        resp = self.client.get("/admin", follow_redirects=False)
        self.assertEqual(resp.status_code, 302)

        self.client.get("/logout", follow_redirects=True)
        self.client.post(
            "/login",
            data={"mode": "register", "username": "it_admin_root", "password": "secret123"},
            follow_redirects=True,
        )
        admin_resp = self.client.get("/admin", follow_redirects=False)
        self.assertEqual(admin_resp.status_code, 200)

    def test_api_health_and_history(self):
        self.client.post(
            "/login",
            data={"mode": "register", "username": "it_api", "password": "secret123"},
            follow_redirects=True,
        )
        h = self.client.get("/api/health")
        self.assertEqual(h.status_code, 200)

        history = self.client.get("/api/history")
        self.assertEqual(history.status_code, 200)
        self.assertIn("ok", history.get_json())


if __name__ == "__main__":
    unittest.main()
