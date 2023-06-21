from datetime import datetime

from flask.views import MethodView
from flask_smorest import Blueprint
from marshmallow_dataclass import class_schema

from miso.app.blueprints.schemas import api_schemas
from miso.app.tasks.celery_models import CeleryTask
from miso.app.tasks.gpu_tasks import train_model_task
from miso.training.parameters import MisoConfig

blp = Blueprint("train",
                __name__,
                description="Operations to train classifiers")


@blp.route("/train")
class CvatProjects(MethodView):
    @blp.doc(operationId='TrainModelAsTask')
    @blp.arguments(api_schemas.train_schema)
    @blp.response(201, api_schemas.celery_task_schema)
    def post(self, config: MisoConfig):
        result = train_model_task.delay(class_schema(MisoConfig)().dump(config))
        return CeleryTask(name="Training model",
                          description=f"Training model {config.cnn.type} on {config.dataset.source}",
                          task_id=result.task_id,
                          result_id=result.id)
