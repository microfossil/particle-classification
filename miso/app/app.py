import json
import os
import traceback

from flask import Flask
from flask_smorest import Api
from flask_cors import CORS
from miso.app.tasks.celery_init import celery_init_app
from werkzeug.exceptions import InternalServerError

from miso.app.blueprints.train_blueprint import blp as train_blp
from miso.app.blueprints.tasks_blueprint import blp as tasks_blp


app = Flask(__name__)
app.config["JSONIFY_PRETTYPRINT_REGULAR"] = True
app.config["API_TITLE"] = "MISO"
app.config["API_VERSION"] = "v1"
app.config["OPENAPI_URL_PREFIX"] = "/"
app.config["OPENAPI_VERSION"] = "3.0.2"
app.config["OPENAPI_SWAGGER_UI_PATH"] = "/swagger"
app.config["OPENAPI_SWAGGER_UI_URL"] = "https://cdn.jsdelivr.net/npm/swagger-ui-dist/"
app.config["CELERY"] = dict(
    broker_url=os.getenv("VISAGE_RABBITMQ_URL", "amqp://guest:guest@localhost:5672/"),
    result_backend=os.getenv("VISAGE_REDIS_URL", "redis://localhost:6379/0"),
    task_ignore_result=False,
)
app.secret_key = "super secret key"

@app.errorhandler(InternalServerError)
def internal_server_error_handler(e):
    return {
        "code": e.code,
        "message": str(traceback.format_exc()),
        "status": e.description
    }, 500

api = Api(app)
app.register_error_handler(500, internal_server_error_handler)
CORS(app)

api.register_blueprint(train_blp, url_prefix="/api/v1")
api.register_blueprint(tasks_blp, url_prefix="/api/v1")

celery_app = celery_init_app(app)

with open("openapi.json", "w") as outfile:
    json.dump(api.spec.to_dict(), outfile, indent=4)


if __name__ == "__main__":
    app.run()

