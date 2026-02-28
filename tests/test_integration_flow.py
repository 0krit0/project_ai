import io
import json
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
from PIL import Image
from werkzeug.datastructures import MultiDict

import app
import db


class IntegrationFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app.init_db()
        app.app.config["TESTING"] = True

    def setUp(self):
        app.ADMIN_USERNAME = "it_admin_root"
        app.REVIEWER_USERNAMES = {"it_reviewer"}
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
            cur.execute("DELETE FROM case_images WHERE user_id = ?", (uid,))
            cur.execute("DELETE FROM cases WHERE user_id = ? OR reviewed_by = ?", (uid, uid))
            cur.execute("DELETE FROM notifications WHERE user_id = ?", (uid,))
            cur.execute("DELETE FROM analysis_jobs WHERE user_id = ?", (uid,))
            cur.execute("DELETE FROM audit_logs WHERE user_id = ?", (uid,))
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

    def _create_user(self, username, password="secret123", role="user"):
        user, err = db.create_user(
            username=username,
            password_hash=app.generate_password_hash(password),
            role=role,
        )
        self.assertIsNone(err)
        self.assertIsNotNone(user)
        return user

    def _login_user(self, username, password="secret123"):
        return self.client.post(
            "/login",
            data={"username": username, "password": password},
            follow_redirects=True,
        )

    def _create_and_login(self, username, password="secret123", role="user"):
        self._create_user(username, password=password, role=role)
        return self._login_user(username, password=password)

    def test_full_user_flow(self):
        mock_model = SimpleNamespace(
            predict=mock.Mock(return_value=np.array([[0.1, 0.8, 0.1]]))
        )
        with mock.patch.object(
            app, "get_model", return_value=mock_model
        ), mock.patch.object(app, "should_retrain", return_value=(False, 0)), mock.patch.object(
            app, "generate_heatmap_overlay", return_value=None
        ):
            login_resp = self._create_and_login("it_flow_user")
            self.assertEqual(login_resp.status_code, 200)

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
        self._create_and_login("it_bad_ext")
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

    def test_login_requires_registered_account(self):
        resp = self.client.post(
            "/login",
            data={"username": "it_not_registered", "password": "abc12345"},
            follow_redirects=True,
        )
        self.assertEqual(resp.status_code, 200)
        conn = db.get_db()
        cur = conn.cursor()
        cur.execute("SELECT id FROM users WHERE username = ?", ("it_not_registered",))
        self.assertIsNone(cur.fetchone())
        conn.close()

    def test_register_password_policy(self):
        resp = self.client.post(
            "/register",
            data={
                "username": "it_weak_pass",
                "password": "12345678",
                "confirm_password": "12345678",
                "email": "it_weak_pass@example.com",
                "age": "25",
                "gender": "male",
                "accept_terms": "1",
                "avatar": (self._make_image_bytes(), "avatar.jpg"),
            },
            follow_redirects=True,
        )
        self.assertEqual(resp.status_code, 200)
        conn = db.get_db()
        cur = conn.cursor()
        cur.execute("SELECT id FROM users WHERE username = ?", ("it_weak_pass",))
        self.assertIsNone(cur.fetchone())
        conn.close()

    def test_admin_page_access(self):
        self._create_and_login("it_admin")
        resp = self.client.get("/admin", follow_redirects=False)
        self.assertEqual(resp.status_code, 302)

        self.client.get("/logout", follow_redirects=True)
        self._create_user("it_admin_root", role="admin")
        self._login_user("it_admin_root")
        admin_resp = self.client.get("/admin", follow_redirects=False)
        self.assertEqual(admin_resp.status_code, 200)

    def test_admin_can_reset_user_password(self):
        self._create_and_login("it_reset_target_user")
        self.client.get("/logout", follow_redirects=True)

        self._create_user("it_admin_root", role="admin")
        self._login_user("it_admin_root")

        conn = db.get_db()
        cur = conn.cursor()
        cur.execute("SELECT id FROM users WHERE username = ?", ("it_reset_target_user",))
        target_id = cur.fetchone()[0]
        conn.close()

        reset_resp = self.client.post(
            f"/admin/users/{target_id}/password",
            data={"new_password": "newpass123"},
            follow_redirects=True,
        )
        self.assertEqual(reset_resp.status_code, 200)

        self.client.get("/logout", follow_redirects=True)
        login_resp = self.client.post(
            "/login",
            data={"username": "it_reset_target_user", "password": "newpass123"},
            follow_redirects=False,
        )
        self.assertEqual(login_resp.status_code, 302)

    def test_api_health_and_history(self):
        self._create_and_login("it_api")
        h = self.client.get("/api/health")
        self.assertEqual(h.status_code, 200)

        history = self.client.get("/api/history")
        self.assertEqual(history.status_code, 200)
        self.assertIn("ok", history.get_json())

    def test_reviewer_workflow_override(self):
        ambiguous_model = SimpleNamespace(
            predict=mock.Mock(return_value=np.array([[0.34, 0.33, 0.33]]))
        )
        with mock.patch.object(
            app, "get_model", return_value=ambiguous_model
        ), mock.patch.object(app, "should_retrain", return_value=(False, 0)), mock.patch.object(
            app, "generate_heatmap_overlay", return_value=None
        ), mock.patch.object(app, "NON_CAR_GUARD_ENABLED", False):
            self._create_and_login("it_case_owner")
            analyze_resp = self.client.post(
                "/api/analyze",
                data={
                    "part": self._valid_part(),
                    "file": (self._make_image_bytes(), "sample.jpg"),
                },
                content_type="multipart/form-data",
            )
            self.assertEqual(analyze_resp.status_code, 200)
            payload = analyze_resp.get_json()
            self.assertTrue(payload.get("needs_manual_review"))
            case_id = payload.get("review_case_id")
            self.assertIsNotNone(case_id)

            self.client.get("/logout", follow_redirects=True)
            self._create_and_login("it_reviewer")

            review_queue = self.client.get("/api/cases?scope=review")
            self.assertEqual(review_queue.status_code, 200)
            queue_payload = review_queue.get_json()
            self.assertTrue(
                any(int(item.get("id")) == int(case_id) for item in queue_payload.get("items", []))
            )

            review_resp = self.client.post(
                f"/api/cases/{case_id}/review",
                data={"final_result": "medium", "reviewer_note": "manual override by reviewer"},
            )
            self.assertEqual(review_resp.status_code, 200)
            reviewed = review_resp.get_json()
            self.assertEqual(reviewed.get("final_result"), "medium")

            conn = db.get_db()
            cur = conn.cursor()
            cur.execute("SELECT status, final_result FROM cases WHERE id = ?", (case_id,))
            row = cur.fetchone()
            self.assertEqual(row["status"], "reviewed")
            self.assertEqual(row["final_result"], "medium")

            cur.execute(
                "SELECT details FROM audit_logs WHERE action = 'case.review' ORDER BY id DESC LIMIT 1"
            )
            audit_row = cur.fetchone()
            conn.close()
            self.assertIsNotNone(audit_row)
            details = json.loads(audit_row["details"])
            self.assertEqual(details.get("status_before"), "needs_review")
            self.assertEqual(details.get("status_after"), "reviewed")
            self.assertEqual(details.get("final_result"), "medium")

    def test_non_car_model_gate_blocks_request(self):
        non_car_gate_model = SimpleNamespace(
            predict=mock.Mock(return_value=np.array([[0.95, 0.05]]))
        )
        with mock.patch.object(
            app, "get_non_car_model", return_value=non_car_gate_model
        ), mock.patch.object(
            app,
            "get_model",
            return_value=SimpleNamespace(predict=mock.Mock(return_value=np.array([[0.1, 0.8, 0.1]]))),
        ), mock.patch.object(app, "should_retrain", return_value=(False, 0)):
            self._create_and_login("it_non_car_gate")
            resp = self.client.post(
                "/api/analyze",
                data={
                    "part": self._valid_part(),
                    "file": (self._make_image_bytes(), "sample.jpg"),
                },
                content_type="multipart/form-data",
            )
            self.assertEqual(resp.status_code, 400)
            payload = resp.get_json()
            self.assertFalse(payload.get("ok"))
            self.assertTrue(payload.get("error"))

    def test_non_car_model_borderline_blocks_request(self):
        borderline_non_car_model = SimpleNamespace(
            predict=mock.Mock(return_value=np.array([[0.35, 0.65]]))
        )
        with mock.patch.object(
            app, "get_non_car_model", return_value=borderline_non_car_model
        ), mock.patch.object(
            app,
            "get_model",
            return_value=SimpleNamespace(predict=mock.Mock(return_value=np.array([[0.05, 0.9, 0.05]]))),
        ), mock.patch.object(app, "should_retrain", return_value=(False, 0)), mock.patch.object(
            app, "generate_heatmap_overlay", return_value=None
        ):
            self._create_and_login("it_non_car_border")
            resp = self.client.post(
                "/api/analyze",
                data={
                    "part": self._valid_part(),
                    "file": (self._make_image_bytes(), "sample.jpg"),
                },
                content_type="multipart/form-data",
            )
            self.assertEqual(resp.status_code, 400)
            payload = resp.get_json()
            self.assertFalse(payload.get("ok"))
            self.assertTrue(payload.get("error"))

    def test_multi_angle_analysis_summary(self):
        model = SimpleNamespace(
            predict=mock.Mock(return_value=np.array([[0.22, 0.68, 0.1]]))
        )
        with mock.patch.object(
            app, "get_model", return_value=model
        ), mock.patch.object(app, "should_retrain", return_value=(False, 0)), mock.patch.object(
            app, "generate_heatmap_overlay", return_value=None
        ), mock.patch.object(app, "NON_CAR_GUARD_ENABLED", False):
            self._create_and_login("it_multi_angle")
            resp = self.client.post(
                "/api/analyze",
                data=MultiDict([
                    ("part", self._valid_part()),
                    ("summary_tone", "insurance"),
                    ("file", (self._make_image_bytes(), "a1.jpg")),
                    ("file", (self._make_image_bytes(), "a2.jpg")),
                ]),
                content_type="multipart/form-data",
            )
            self.assertEqual(resp.status_code, 200)
            payload = resp.get_json()
            self.assertTrue(payload.get("ok"))
            self.assertIn("multi_angle", payload)
            self.assertEqual(payload["multi_angle"]["image_count"], 2)
            self.assertIn("incident_summary", payload)
            self.assertIn("incident_summaries", payload)
            self.assertEqual(payload.get("selected_summary_tone"), "insurance")
            self.assertTrue(payload.get("incident_summary"))


if __name__ == "__main__":
    unittest.main()
