from repository import BgTasksRepository
from serializers import BGTaskSchema
from settings import app_config


class BGTasksService:
    """Service for managing background tasks."""

    def __init__(
        self,
        bg_tasks_repo: BgTasksRepository,
    ):
        self.bg_tasks_repo = bg_tasks_repo

    async def get_tasks(self) -> list[BGTaskSchema]:
        """Get a list of background tasks."""
        tasks = await self.bg_tasks_repo.get_tasks()
        return [BGTaskSchema.model_validate(task) for task in tasks]

    async def rotate_tasks(self):
        """Rotate the background tasks."""
        bg_tasks = await self.bg_tasks_repo.get_tasks()
        if len(bg_tasks) > app_config.max_saved_bg_tasks:
            excess_count = len(bg_tasks) - app_config.max_saved_bg_tasks
            task_ids_to_remove = [
                task.uuid for task in bg_tasks
                if task.status in ("success", "failure")
            ][-excess_count:]

            await self.bg_tasks_repo.delete_tasks_by_uuid(task_ids_to_remove)
