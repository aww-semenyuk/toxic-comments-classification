from uuid import UUID

from serializers.background_tasks import BGTask
from serializers.trainer import MLModel

models: dict[str, MLModel] = {}
loaded_models = {}
bg_tasks: dict[UUID, BGTask] = {}
