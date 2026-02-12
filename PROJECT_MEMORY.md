# Project Memory

Last updated: 2026-02-12
Owner: project_ai team

## Current Goal
- Build and maintain a web app that analyzes car damage severity from uploaded images.

## Current Status
- Core system is working end-to-end: upload image -> AI prediction -> show guidance -> log history/feedback.
- Retrain pipeline exists and can be triggered automatically when conditions are met.

## Completed Since Last Session
- Created persistent memory file for cross-session handoff.
- Documented core project workflow and architecture summary.

## Pending Tasks
- [ ] Validate Thai text encoding in UI/messages (some text appears garbled).
- [ ] Decide whether to migrate logs from CSV to SQLite (`db.py`) for consistency.
- [ ] Add basic test checklist for prediction, history, feedback, and export flow.

## Next Immediate Step
- Run a quick functional check of the full user flow on local server and fix any UI/text issues.

## Core Workflow (Project Principle)
- User logs in (`/login`) and enters the main analysis page (`/`).
- User selects a car part and uploads an image.
- Backend preprocesses image to 224x224 and runs TensorFlow model (`damage_model.h5`).
- Model predicts severity class: `low`, `medium`, or `high`.
- App maps prediction + selected part to repair guidance via `RULES` in `rules.py`.
- Result, confidence, trust level, and decision text are shown on UI.
- Uploaded image and analysis record are saved (`feedback_images/*`, `experience_log.csv`).
- User can send correctness feedback (`/feedback`) stored in `feedback_log.csv`.
- Retrain condition checks feedback image volume/time (`retrain_condition.py`).
- If threshold met, app triggers `retrain.py` to fine-tune and save a new model file.

## Key Components
- `app.py`: Flask routes, prediction flow, history/profile/export/feedback.
- `rules.py`: Rule-based descriptions, risk, and repair advice by part + severity.
- `train.py/train.py`: Initial training script (MobileNetV2 transfer learning).
- `retrain.py`: Incremental retrain from `feedback_images`.
- `retrain_condition.py`: Rules for when retrain should run.
- `templates/index.html`: Main user UI for analyze + feedback.

## Files Touched
- `PROJECT_MEMORY.md`: initialized and filled with project operation summary.

## Commands Used
```bash
Get-Content app.py -TotalCount 260
Get-Content rules.py -TotalCount 260
Get-Content train.py\\train.py -TotalCount 260
Get-Content retrain.py -TotalCount 260
Get-Content retrain_condition.py -TotalCount 220
Set-Content PROJECT_MEMORY.md
```

## Risks / Blockers
- Thai string encoding appears inconsistent in several files.
- Retrained model is saved as a new file, but serving model auto-switch strategy is not defined.

## Notes for Next Session
- Start by reading this file.
- Continue from "Next Immediate Step".
- If asked to update memory, overwrite this file with latest concise status.
