import os

from waitress import serve

from app import app


if __name__ == "__main__":
    host = os.getenv("FLASK_HOST", "0.0.0.0")
    port = int(os.getenv("PORT", os.getenv("FLASK_PORT", "5000")))
    serve(app, host=host, port=port)
