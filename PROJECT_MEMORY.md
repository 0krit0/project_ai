# Project Memory

Last updated: 2026-02-12
Owner: project_ai team

## Current Goal
- Move CarDamage AI from demo-level UX to a usable operational tool.
- Keep analysis flow simple for users while adding operational controls for admin.
- Preserve fast iteration speed with local Flask + TensorFlow stack.

## Current Status
- End-to-end flow works: login -> analyze -> result -> feedback -> history/profile/export.
- Storage migrated to SQLite (`app.db`) with CSV migration helper retained.
- UI is now unified via `static/ui.css` across login/index/history/profile/admin.
- Integration tests pass for main flow and admin access guard.
- Server currently reachable at `http://127.0.0.1:5000/login`.

## Major Changes Implemented (Latest)
1. Backend hardening and operations
- Added app logging via rotating file handler (`run.app.log`).
- Added analyze request rate limiting (per user, window-based).
- Added optional login password check via env `APP_LOGIN_PASSWORD`.
- Added admin role check (current rule: username == `ADMIN_USERNAME`, default `admin`).

2. Explainability and decision policy
- Added Grad-CAM style heatmap overlay generation for analyzed images.
- Added confidence policy mapping into 3 tiers with actionable guidance.
- Kept top-3 evidence panel and quality notes in result view.

3. Admin features
- Added `/admin` dashboard with:
  - total users
  - total analyses
  - analyses in last 24h
  - average confidence
  - result distribution
  - recent feedback table
- Added database maintenance routes:
  - `/admin/backup` download SQLite backup
  - `/admin/restore` upload and restore `.db`

4. Profile upgrades
- Added richer profile dashboard with:
  - total analyses
  - average confidence
  - top damaged part
  - last activity time
  - recent 5 records

5. UI/UX updates
- Added admin menu visibility on index/history/profile for admin users.
- Added confidence policy card and heatmap panel on result section.
- Improved login form to support optional password field + warning message.
- Added style blocks for policy/heatmap/profile/admin in `static/ui.css`.

6. Testing and verification
- Updated integration tests (`tests/test_integration_flow.py`) to cover:
  - full user flow
  - invalid upload extension
  - admin access behavior
- Ran tests successfully: `python -m unittest tests.test_integration_flow -v`.
- Ran syntax check: `python -m py_compile app.py db.py`.

## Key Config / Env
- `FLASK_SECRET_KEY`
- `APP_LOGIN_PASSWORD` (optional)
- `ADMIN_USERNAME` (default `admin`)
- `ANALYZE_RATE_LIMIT_COUNT` (default `8`)
- `ANALYZE_RATE_LIMIT_WINDOW_SEC` (default `60`)
- `MAX_UPLOAD_BYTES` (default `5MB`)
- `APP_LOG_PATH` (default `run.app.log`)

## Main Files Involved
- `app.py`
- `db.py`
- `templates/index.html`
- `templates/login.html`
- `templates/history.html`
- `templates/profile.html`
- `templates/admin.html` (new)
- `static/ui.css`
- `tests/test_integration_flow.py`

## Known Risks / Follow-ups
- Current admin identification by username is simple; should move to explicit role table.
- Runtime artifacts (feedback images, logs, status files) are present in working tree.
- Model confidence remains probabilistic; should add calibration if using for cost estimate.
- Restore DB is powerful; consider adding second confirmation and audit trail.

## Suggested Next Engineering Steps
1. Add proper auth (password hash + roles in DB).
2. Add `.gitignore` policy for runtime artifacts and logs.
3. Add audit log table for admin actions (backup/restore).
4. Add model/version metadata to each history record.
5. Add API endpoint layer (for mobile or external integration).
