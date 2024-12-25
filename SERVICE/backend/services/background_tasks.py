from uuid import UUID

from serializers.background_tasks import BGTask
from settings.app_config import AppConfig


class BGTasksService:
    def __init__(
        self,
        app_config: AppConfig,
        bg_tasks_store: dict[UUID, BGTask]
    ):
        self.app_config = app_config
        self.bg_tasks = bg_tasks_store

    async def get_tasks(self) -> list[BGTask]:
        return list(self.bg_tasks.values())

    def rotate_tasks(self):
        if len(self.bg_tasks) > self.app_config.max_saved_bg_tasks:
            removable_tasks = sorted(
                (
                    task for task in self.bg_tasks.values()
                    if task.status in ("success", "failure")
                ),
                key=lambda task: task.updated_at,
                reverse=True
            )

            excess_count = (
                len(self.bg_tasks) - self.app_config.max_saved_bg_tasks
            )
            task_ids_to_remove = {
                task.uuid for task in removable_tasks[:excess_count]
            }
            for task_id in task_ids_to_remove:
                del self.bg_tasks[task_id]
