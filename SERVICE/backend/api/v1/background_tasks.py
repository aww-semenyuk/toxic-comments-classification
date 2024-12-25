from typing import Annotated

from fastapi import Depends, APIRouter

from dependency import get_bg_tasks_service
from serializers.background_tasks import BGTask
from services.background_tasks import BGTasksService

router = APIRouter()


@router.get("/", response_model=list[BGTask])
async def get_tasks(
    bg_tasks_service: Annotated[BGTasksService, Depends(get_bg_tasks_service)]
):
    return await bg_tasks_service.get_tasks()
