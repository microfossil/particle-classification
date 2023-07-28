from miso.app.tasks.celery_models import CeleryQueueStatus, CeleryTaskProgress

from celery import current_app
from celery.contrib.abortable import AbortableAsyncResult
from celery.result import AsyncResult


class TaskService:
    def view_queue(self) -> CeleryQueueStatus:
        i = current_app.control.inspect()
        status = CeleryQueueStatus(
            scheduled=[item for sublist in list(i.scheduled().values()) for item in sublist],
            active=[item for sublist in list(i.active().values()) for item in sublist],
            reserved=[item for sublist in list(i.reserved().values()) for item in sublist]
        )
        return status

    def task_progress(self, id) -> CeleryTaskProgress:
        res = AsyncResult(id, app=current_app)
        if not isinstance(res.info, Exception) and res.info is not None:
            return CeleryTaskProgress(id=res.task_id,
                                      state=res.state,
                                      message=res.info.get("message", "none"),
                                      count=res.info.get("count", 0),
                                      total=res.info.get("total", 1))
        else:
            return CeleryTaskProgress(id=res.task_id,
                                      state=res.state,
                                      message="Cancelling",
                                      count=0,
                                      total=1)

    def task_delete(self, id):
        current_app.control.revoke(id, terminate=True)
        res = AbortableAsyncResult(id, app=current_app)
        res.abort()
