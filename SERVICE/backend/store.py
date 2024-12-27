from uuid import UUID

from serializers.background_tasks import BGTask
from serializers.trainer import MLModel, MLModelType

DEFAULT_MODELS_INFO = {
    "default_logistic_regression": {
        "type": MLModelType.logistic_regression,
        "filename": "model_lr_e.cloudpickle"
    },
    "default_linear_svc": {
        "type": MLModelType.linear_svc,
        "filename": "model_svc_e.cloudpickle"
    },
    "default_multinomial_naive_bayes": {
        "type": MLModelType.multinomial_nb,
        "filename": "model_mnb_e.cloudpickle"
    }
}

models: dict[str, MLModel] = {}
loaded_models = {}
bg_tasks: dict[UUID, BGTask] = {}
