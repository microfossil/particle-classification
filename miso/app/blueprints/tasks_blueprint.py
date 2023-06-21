from flask.views import MethodView
from flask_smorest import Blueprint

from miso.app.blueprints.schemas import api_schemas
from miso.app.tasks.service import TaskService

blp = Blueprint("task",
                __name__,
                description="Celery tasks")


@blp.route("/task")
class Tasks(MethodView):
    @blp.doc(operationId='TaskList')
    @blp.response(200, api_schemas.celery_queue_status_schema)
    def get(self):
        service = TaskService()
        status = service.view_queue()
        return status


@blp.route("/task/<task_id>")
class TasksById(MethodView):
    @blp.doc(operationId='TaskDelete')
    @blp.response(204)
    def delete(self, task_id):
        service = TaskService()
        return service.task_delete(task_id)


@blp.route("/task_status/<task_id>")
@blp.doc(operationId='TaskStatus')
@blp.response(200, api_schemas.celery_task_progress_schema)
def task_status(task_id):
    service = TaskService()
    return service.task_progress(task_id)
