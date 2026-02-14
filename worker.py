import os
import threading
from datetime import datetime

from flask import Flask, jsonify

import app as core_app

health_app = Flask(__name__)
WORKER_HEALTH_PORT = int(os.getenv("WORKER_HEALTH_PORT", "8001"))
WORKER_HEALTH_HOST = os.getenv("WORKER_HEALTH_HOST", "0.0.0.0")

worker_state = {
    "started_at": datetime.now().isoformat(),
    "last_error": None,
}
worker_thread = None


def worker_main():
    try:
        core_app.app.logger.info(
            "worker process booting (QUEUE_BACKEND=%s, INLINE_WORKER_ENABLED=%s)",
            core_app.QUEUE_BACKEND,
            core_app.INLINE_WORKER_ENABLED,
        )
        core_app.run_worker_forever()
    except Exception as err:
        worker_state["last_error"] = str(err)
        core_app.app.logger.exception("worker crashed: %s", err)


@health_app.route("/health")
def health():
    alive = bool(worker_thread and worker_thread.is_alive())
    return (
        jsonify(
            {
                "ok": alive and worker_state["last_error"] is None,
                "worker_alive": alive,
                "queue_backend": core_app.get_queue_backend(),
                "queue_size": core_app.get_analysis_queue_size(),
                "started_at": worker_state["started_at"],
                "last_error": worker_state["last_error"],
                "time": datetime.now().isoformat(),
            }
        ),
        200 if alive else 503,
    )


def main():
    global worker_thread
    worker_thread = threading.Thread(target=worker_main, daemon=True)
    worker_thread.start()
    health_app.run(host=WORKER_HEALTH_HOST, port=WORKER_HEALTH_PORT, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
