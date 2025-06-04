from serializers.trainer import MLModelType

DEFAULT_MODELS_INFO = {
    "default_logistic_regression": {
        "type": MLModelType.logistic_regression,
        "filename": "model_lr_e.cloudpickle",
        "is_dl_model": False
    },
    "default_linear_svc": {
        "type": MLModelType.linear_svc,
        "filename": "model_svc_e.cloudpickle",
        "is_dl_model": False
    },
    "default_multinomial_naive_bayes": {
        "type": MLModelType.multinomial_nb,
        "filename": "model_mnb_e.cloudpickle",
        "is_dl_model": False
    },
    "default_distilbert": {
        "type": MLModelType.distilbert,
        "filename": "distilbert",
        "is_dl_model": True,
        "tokenizer": "distilbert-base-uncased"
    }
}

loaded_models = {}
