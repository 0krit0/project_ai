# AutoScope AI

ระบบเว็บสำหรับวิเคราะห์ความเสียหายรถยนต์จากภาพด้วย AI พร้อม workflow การทำงานแบบใช้งานจริง (ผู้ใช้, เคสงาน, รีวิว, รายงาน, แอดมิน)

## Features

- วิเคราะห์ความเสียหายจากภาพ (single/multi-image)
- แสดงผลแบบ Top-K พร้อมช่วงค่าใช้จ่ายโดยประมาณ
- ระบบเคสงาน (Cases) และจัดการรูปในแต่ละเคส
- ประวัติการวิเคราะห์ + ส่งออก CSV/PDF
- ระบบผู้ใช้: สมัคร, ล็อกอิน, โปรไฟล์, อัปโหลดรูปโปรไฟล์
- Reviewer/Admin dashboard และ API monitoring
- รองรับ queue แบบ local และ Redis (optional)

## Tech Stack

- Python 3.10+
- Flask
- TensorFlow (CPU)
- SQLite
- Waitress (production serve)

## Project Structure

- `app.py` - Flask application หลัก
- `db.py` - database layer + migration helpers
- `serve.py` - production entrypoint (Waitress)
- `worker.py` - optional worker process
- `templates/` - HTML templates
- `static/` - frontend assets
- `tests/` - integration tests

## Quick Start (Local)

1. ติดตั้ง dependencies

```bash
pip install -r requirements.txt
```

2. สร้างไฟล์ environment

```bash
copy .env.example .env
```

3. ตั้งค่าอย่างน้อยใน `.env`

- `FLASK_SECRET_KEY`
- `APP_LOGIN_PASSWORD` (หรือใช้งานผ่านระบบ register)
- `MODEL_VERSION=damage_model.h5`

4. รันแอป

```bash
python serve.py
```

แอปจะรันที่ `http://127.0.0.1:5000` (หรือพอร์ตจาก `PORT`/`FLASK_PORT`)

## Run Tests

```bash
python -m unittest tests.test_integration_flow -v
```

## Key API Endpoints

- `GET /api/health`
- `POST /api/analyze`
- `GET /api/analyze/jobs/<job_id>`
- `GET /api/history`
- `GET|POST /api/cases`
- `GET /api/admin/metrics`

## Deployment

มีไฟล์สำหรับ deploy พร้อมใช้งาน:

- `Dockerfile`
- `docker-compose.yml`
- `render.yaml`
- รายละเอียดเพิ่มเติม: `DEPLOY_RENDER.md`

## Notes

- model หลักเริ่มต้นคือ `damage_model.h5`
- หากต้องการใช้ Redis queue ให้ตั้ง `QUEUE_BACKEND=redis` และ `REDIS_URL`
- ควรเก็บไฟล์ runtime เช่น logs/db ใน storage ที่เหมาะสมเมื่อขึ้น production
