

class TaskMonitor:
    def __init__(self, celery_obj):
        self.celery_obj = celery_obj
        self.message = ""
        self.count = 0
        self.total = 1

    def _update(self, state, message, count, total):
        self.message = message
        self.count = count
        self.total = total
        self.celery_obj.update_state(
            state=state,
            meta={
                "message": message,
                "count": count,
                "total": total
            })

    def progress(self, message, count=0, total=1):
        self._update("STARTED", message, count, total)

    def succeeded(self, message):
        self._update("SUCCEEDED", message, 1, 1)

    def failed(self, message):
        self._update("FAILED", message, 0, 1)
        raise Exception(message)

    def is_cancelled(self):
        if self.celery_obj.is_aborted():
            self._update("CANCELLED", "Task cancelled", 0, 1)
            return True
        return False