# Project Memory

Last updated: 2026-02-12
Owner: project_ai team

## Current Goal
- Stabilize and polish CarDamage AI before active development.
- Ensure Thai text is readable across UI/backend.
- Upgrade UX/UI so the app feels production-like (consistent, clear, usable on mobile).

## Current Status
- Core flow works: login -> analyze image -> show result -> feedback -> history/profile/export CSV.
- Major text encoding cleanup completed (garbled Thai removed in core files).
- UX/UI redesign completed across login/index/history/profile with one shared design system.
- Changes are local and not committed yet.

## Completed This Session (Detailed)
1. Baseline code cleanup
- Removed broken template wrappers and stray triple-quote markers.
- Updated Flask secret key usage from hardcoded string to env-based fallback.

2. Thai encoding cleanup (UTF-8 rewrite)
- Rewrote garbled Thai text in backend and templates.
- Rebuilt `rules.py` with clean Thai keys/values and aligned with UI part names.
- Standardized CSV export header Thai text.

3. UX/UI full redesign
- Added shared stylesheet: `static/ui.css`.
- Migrated all core pages to shared visual language:
  - `templates/login.html`
  - `templates/index.html`
  - `templates/history.html`
  - `templates/profile.html`
- Improved information hierarchy on result view (decision-first with KPI blocks).
- Added UX states:
  - loading state on analyze button
  - warning/error notice block on index
  - empty state on history page
- Kept responsive behavior for desktop/mobile.

4. Validation
- Python syntax check passed (`py_compile`) for key python files.
- Emoji scan run: no emoji left in code/templates/config files.

## Files Touched (Current Working Tree)
- `app.py`
- `rules.py`
- `retrain.py`
- `retrain_condition.py`
- `train.py/train.py`
- `test_model.py`
- `db.py`
- `templates/index.html`
- `templates/login.html`
- `templates/history.html`
- `templates/profile.html`
- `static/ui.css` (new)
- `static/manifest.json`
- `static/service_worker.js`

## Important Design Decisions
- Keep Flask + CSV flow unchanged for now (no DB migration yet) to avoid scope creep.
- Prioritize UX consistency and readability first.
- Preserve existing functional behavior while redesigning UI.

## Known Risks / Gaps
- UI is redesigned from code but still needs live browser smoke test.
- `debug=True` still enabled in `app.py` (ok for dev, not for production).
- Secret key fallback still dev-like if env var is not set.
- `db.py` exists but app currently uses CSV logs.
- Retrained model files are versioned, but serving model switch strategy is still undefined.

## Pending Tasks
- [ ] Run live functional smoke test in browser (all pages + analyze + feedback + export).
- [ ] Decide whether to keep CSV or migrate to SQLite (`db.py`).
- [ ] Add `.gitignore` for `__pycache__/` and model/temp artifacts.
- [ ] Clean pycache artifacts currently appearing in git status.
- [ ] Commit and push current changes in logical chunks.

## Suggested Commit Plan
1. `chore: baseline template and security cleanup`
- app secret key env fallback + template quote cleanup

2. `fix: normalize thai utf-8 text across app and rules`
- app/rules/templates/scripts text cleanup

3. `feat: redesign UI with shared style system`
- new `static/ui.css` + all page redesigns + loading/empty/error states

## Next Immediate Step
- Review app visually in browser and capture any final spacing/text issues.
- Then clean pycache + create `.gitignore` + commit/push.

## Quick Start for Next Session
1. Read this file first.
2. Run `git status --short` to verify uncommitted changes.
3. Start from "Next Immediate Step".
