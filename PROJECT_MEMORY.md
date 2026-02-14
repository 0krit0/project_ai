# Project Memory

Last updated: 2026-02-14
Owner: project_ai team

## Current Goal
- Move CarDamage AI from demo-level UX to an operational tool with safer prediction flow.
- Support user/account workflow with profile avatar and richer admin/monitor capabilities.
- Keep local-first developer workflow while adding optional docker + worker deployment path.

## Current Status
- End-to-end web flow works: login/register -> analyze -> result -> feedback -> history/profile/admin/report/cases.
- SQLite schema has online migrations and now includes auth/audit/model metadata/job queue/case/notification data.
- Multi-outcome analysis is active (top-k, weighted expected cost, uncertainty summary).
- API layer is available for health, analyze (sync/async), history, cases, notifications, admin monitor.
- Integration tests pass (`python -m unittest tests.test_integration_flow -v`).

## Major Changes Implemented Today (2026-02-14)
1. Auth, role, session, audit
- Added password-hash login/register flow with role-based admin checks.
- Added session timeout handling and richer session user payload.
- Added audit log events for auth, analyze, feedback, export, admin routes, api monitor.

2. Data model and migration expansion
- `users`: added `password_hash`, `role`, `is_active`, `avatar_path`.
- `history`: added `model_version`, `est_cost_min`, `est_cost_max`.
- New tables: `audit_logs`, `model_registry`, `cases`, `case_images`, `notifications`, `analysis_jobs`.

3. API + operations
- Added `/api/health`, `/api/history`, `/api/analyze` (sync + async), `/api/analyze/jobs/<id>`.
- Added `/api/cases`, `/api/cases/<id>`, `/api/cases/<id>/images`.
- Added `/api/notifications`, `/api/notifications/<id>/read`.
- Added `/api/admin/metrics` and `/api/admin/monitor`.

4. Worker + queue path
- Added in-process queue worker for async analysis.
- Added optional Redis-backed queue mode via env (`QUEUE_BACKEND=redis`).
- Added separate worker process entrypoint (`worker.py`) with worker health endpoint (`/health`).

5. Report and deploy
- Added report page (`/report/latest`) and server-side PDF export (`/report/latest.pdf`).
- Added container deployment assets: `Dockerfile`, `.dockerignore`, `docker-compose.yml`, `requirements.txt`.

6. UX/UI updates
- Added login register mode controls and avatar upload/preview in login form.
- Added profile avatar upload endpoint and sidebar avatar display in key pages.
- Added cases/report navigation links across web pages.
- Added non-car warning modal on analyze page.

7. Prediction quality controls
- Added multi-outcome assessment block with per-outcome score + repair + cost range.
- Added weighted expected-cost estimate.
- Added non-car guard gate (confidence + score-gap + entropy thresholds) before accepting prediction.

## Key Config / Env
- Core
  - `FLASK_SECRET_KEY`
  - `FLASK_HOST` (default `0.0.0.0`)
  - `FLASK_PORT` (default `5000`)
  - `APP_LOG_PATH`
- Auth / Admin
  - `ADMIN_USERNAME`
  - `APP_LOGIN_PASSWORD` (legacy fallback)
  - `SESSION_TIMEOUT_MIN`
- Analyze / Guard
  - `ANALYZE_RATE_LIMIT_COUNT`
  - `ANALYZE_RATE_LIMIT_WINDOW_SEC`
  - `MAX_UPLOAD_BYTES`
  - `TOP_K_OUTCOMES`
  - `NON_CAR_GUARD_ENABLED`
  - `NON_CAR_MIN_CONFIDENCE`
  - `NON_CAR_MIN_SCORE_GAP`
  - `NON_CAR_MAX_ENTROPY`
- Model
  - `MODEL_VERSION`
- Queue / Worker
  - `QUEUE_BACKEND` (`local` or `redis`)
  - `REDIS_URL`
  - `INLINE_WORKER_ENABLED`
  - `WORKER_HEALTH_HOST`
  - `WORKER_HEALTH_PORT`

## Main Files Involved
- `app.py`
- `db.py`
- `worker.py`
- `requirements.txt`
- `Dockerfile`
- `docker-compose.yml`
- `.dockerignore`
- `templates/index.html`
- `templates/login.html`
- `templates/history.html`
- `templates/profile.html`
- `templates/admin.html`
- `templates/cases.html`
- `templates/report.html`
- `static/ui.css`
- `tests/test_integration_flow.py`

## Known Risks / Follow-ups (Next Work)
1. Non-car detection is rule-based over current classifier outputs; still can pass false positives.
2. Need true domain gate model (`car vs non-car`) before damage-level inference.
3. `templates/cases.html` and `templates/report.html` have text-encoding corruption and should be normalized to UTF-8 clean text.
4. Runtime artifacts (feedback images, logs, retrain status) are frequently generated in working tree.
5. Async worker queue currently has basic reliability only; no retry policy/dead-letter queue yet.
6. Server-side PDF is minimal text layout; can be improved for branded report format.

## Suggested Next Engineering Steps
1. Add dedicated non-car classifier (or detector) and route hard block before damage model.
2. Add manual override workflow + reviewer role for ambiguous predictions.
3. Normalize template text encoding for `cases/report` and add localization consistency checks.
4. Add queue retry/backoff + failed-job admin view + requeue endpoint.
5. Add `.gitignore` tightening for runtime artifacts and generated image outputs.
