class ModelNotFoundError(Exception):
    def __init__(self, model_id: str):
        self.detail = f"Model '{model_id}' not found."
        super().__init__(self.detail)


class ModelIDAlreadyExistsError(Exception):
    def __init__(self, model_id: str):
        self.detail = f"Model with ID={model_id} already exists."
        super().__init__(self.detail)


class ModelNotLoadedError(Exception):
    def __init__(self, model_id: str):
        self.detail = f"Model '{model_id}' not loaded."
        super().__init__(self.detail)


class ModelsLimitExceededError(Exception):
    detail = "Models limit exceeded."


class InvalidFitPredictDataError(Exception):
    def __init__(self, message: str):
        self.detail = message
        super().__init__(self.detail)


class ActiveProcessesLimitExceededError(Exception):
    detail = "Active processes limit exceeded."
