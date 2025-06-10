from typing import Annotated

from fastapi import Depends, APIRouter

from dependency import get_bg_tasks_service
from serializers import BGTaskSchema
from services import BGTasksService

router = APIRouter()


@router.get(
    "/",
    response_model=list[BGTaskSchema],
    description="Получение списка задач"
)
async def get_tasks(
    bg_tasks_service: Annotated[BGTasksService, Depends(get_bg_tasks_service)]
):
    """Endpoint to get tasks"""
    return await bg_tasks_service.get_tasks()
