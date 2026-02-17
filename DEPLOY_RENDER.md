# Deploy To Render (24/7-ready)

This guide prepares this project for a Render web service.

## 1) Preconditions

- Your code is pushed to GitHub.
- `requirements.txt` includes `waitress`.
- App has a health endpoint: `/healthz`.
- Your model file `damage_model.h5` is in repo (or fetched during build).

## 2) Create Web Service

1. Open Render dashboard.
2. New -> Web Service -> Connect your GitHub repo.
3. Use these settings:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `waitress-serve --host=0.0.0.0 --port=$PORT app:app`
   - Health Check Path: `/healthz`

You can also use `render.yaml` from this repo (Blueprint deploy).

## 3) Environment Variables (minimum)

Set these in Render:

- `FLASK_SECRET_KEY` (required; random long value)
- `FLASK_DEBUG=false`
- `DB_NAME=/var/data/app.db` (recommended for persistent disk)
- `MODEL_VERSION=damage_model.h5`
- `INLINE_WORKER_ENABLED=true`
- `QUEUE_BACKEND=local`
- `NON_CAR_MODEL_ENABLED=false` (unless non-car model file exists on server)

Optional tuning:

- `TTA_ENABLED=true`
- `TTA_VARIANTS=4`
- `MANUAL_REVIEW_MAX_TTA_DISPERSION=0.085`

## 4) Deploy and Validate

After deploy:

1. Open `https://<your-service>.onrender.com/healthz` and confirm `ok: true`.
2. Login and run one analyze flow.
3. Check logs for model-load errors.

## 5) 24-hour runtime notes

- Use a paid Render plan for true always-on uptime.
- Free plan can sleep after inactivity.
- If you need persistent history/uploads with SQLite, create a Render Disk and keep `DB_NAME` on mounted path (for example `/var/data/app.db`).

## 6) Important behavior after code changes

Editing code locally does **not** update the live website automatically.

To update live site:

1. Commit changes.
2. Push to GitHub branch connected to Render.
3. Wait for Render auto-deploy (or click manual deploy).

Only after redeploy will the running website use new code.
