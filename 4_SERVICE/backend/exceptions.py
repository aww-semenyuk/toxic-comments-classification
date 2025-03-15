class ModelNotFoundError(Exception):
    def __init__(self, model_id: str):
        self.detail = (
            f"Модель '{model_id}' не найдена. Возможно она еще обучается."
        )
        super().__init__(self.detail)


class ModelIDAlreadyExistsError(Exception):
    def __init__(self, model_id: str):
        self.detail = f"Модель '{model_id}' уже существует."
        super().__init__(self.detail)


class ModelAlreadyLoadedError(Exception):
    def __init__(self, model_id: str):
        self.detail = f"Модель '{model_id}' уже загружена."
        super().__init__(self.detail)


class ModelNotTrainedError(Exception):
    def __init__(self, model_id: str):
        self.detail = f"Модель '{model_id}' еще не обучилась."
        super().__init__(self.detail)


class ModelNotLoadedError(Exception):
    def __init__(self, model_id: str):
        self.detail = f"Модель '{model_id}' не загружена в память."
        super().__init__(self.detail)


class ModelsLimitExceededError(Exception):
    detail = "Превышен лимит моделей для инференса."


class InvalidFitPredictDataError(Exception):
    def __init__(self, message: str):
        self.detail = message
        super().__init__(self.detail)


class DefaultModelRemoveUnloadError(Exception):
    detail = "Нельзя удалить или выгрузить из памяти модели по умолчанию."


class ActiveProcessesLimitExceededError(Exception):
    detail = "Превышен лимит активных процессов."
