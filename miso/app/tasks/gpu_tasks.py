from celery import shared_task
from celery.contrib.abortable import AbortableTask
from marshmallow_dataclass import class_schema
from miso.app.tasks.monitor import TaskMonitor
from miso.training.parameters import MisoConfig
from miso.training.trainer import train_image_classification_model


@shared_task(bind=True, base=AbortableTask)
def train_model_task(self, config):
    monitor = TaskMonitor(self)
    config = class_schema(MisoConfig)().load(config)
    train_image_classification_model(config)