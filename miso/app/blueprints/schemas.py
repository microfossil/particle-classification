from marshmallow_dataclass import class_schema
from miso.app.tasks.celery_models import CeleryTask, CeleryTaskStatus, CeleryQueueStatus, CeleryTaskProgress

from miso.training.parameters import MisoConfig


class MisoSchemas:
    train_schema = class_schema(MisoConfig)

    celery_task_schema = class_schema(CeleryTask)
    celery_task_status_schema = class_schema(CeleryTaskStatus)
    celery_queue_status_schema = class_schema(CeleryQueueStatus)
    celery_task_progress_schema = class_schema(CeleryTaskProgress)

api_schemas = MisoSchemas()