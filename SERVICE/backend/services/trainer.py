import asyncio
import os
from concurrent.futures import ProcessPoolExecutor

import unicodedata
import spacy
import nltk
nltk.download('stopwords')

import cloudpickle

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.base import clone, TransformerMixin, BaseEstimator
from sklearn.pipeline import Pipeline

from exceptions import (
    ModelIDAlreadyExistsError,
    ModelNotFoundError,
    ModelNotLoadedError,
    ModelsLimitExceededError,
    InvalidFitPredictDataError,
    ActiveProcessesLimitExceededError
)
from serializers.trainer import (
    FitRequest,
    MessageResponse,
    GetStatusResponse,
    PredictRequest,
    ModelListResponse,
    MLModelType,
    VectorizerType,
    PredictResponse
)
from settings.app_config import AppConfig, active_processes

available_models = {MLModelType.LogisticRegression: LogisticRegression(), 
                    MLModelType.MultinomialNB: MultinomialNB()}

class FunctionWrapper:
    # https://stackoverflow.com/a/75720040

    def __init__(self, fn):
        self.fn_ser = cloudpickle.dumps(fn)

    def __call__(self, *args, **kwargs):
        fn = cloudpickle.loads(self.fn_ser)
        return fn(*args, **kwargs)

default_vec_params = {
    'tokenizer': FunctionWrapper(lambda x: x.split('\t')),
    'strip_accents': None,
    'lowercase': False,
    'preprocessor': None,
    'stop_words': None,
    'token_pattern': None
}

available_vectorizers = {VectorizerType.CountVectorizer: CountVectorizer(**default_vec_params), 
                         VectorizerType.TfidfVectorizer: TfidfVectorizer(**default_vec_params)}

class CustomTokenizer(BaseEstimator, TransformerMixin):
    nlp = spacy.load('en_core_web_sm')
    stopwords = set(nltk.corpus.stopwords.words('english'))

    def __init__(self, batch_size=64, sep='\t'):
        self.batch_size = batch_size
        self.sep = sep

    @staticmethod
    def normalize_text(doc):
        return unicodedata.normalize('NFKD', doc).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        results = []

        corpus_normalized = [self.normalize_text(doc) for doc in X]
        pipe = self.nlp.pipe(corpus_normalized, disable=['ner', 'parser'], batch_size=self.batch_size)

        for doc in pipe:
            results.append(self.sep.join([token.lemma_.lower() for token in doc if not (token.lemma_.lower() in self.stopwords or token.is_space or token.is_punct)]))

        return results

class TrainerService:
    def __init__(
        self,
        app_config: AppConfig,
        models_store: dict,
        loaded_models: dict
    ):
        self.models = models_store
        self.loaded_models = loaded_models
        self.app_config = app_config

        if not app_config.models_dir_path.exists():
            app_config.models_dir_path.mkdir(parents=True)

    async def fit_models(
        self, fit_data: list[FitRequest]
    ) -> list[MessageResponse]:
        unique_fit_items = {}
        for item in fit_data:
            if item.config.id not in unique_fit_items:
                unique_fit_items[item.config.id] = item
        unique_fit_items_list = list(unique_fit_items.values())

        for item in unique_fit_items_list:
            model_id = item.config.id
            if model_id in self.models:
                raise ModelIDAlreadyExistsError(model_id)

        fit_items_cnt = len(unique_fit_items_list)
        free_cores_cnt = self.app_config.cores_cnt - active_processes.value
        if fit_items_cnt > free_cores_cnt:
            raise ActiveProcessesLimitExceededError()

        active_processes.value += fit_items_cnt
        executor = ProcessPoolExecutor(max_workers=free_cores_cnt)
        loop = asyncio.get_running_loop()
        tasks = [
            loop.run_in_executor(executor, self._train_model, item)
            for item in unique_fit_items_list
        ]

        response_list = []
        for task, item in zip(
            asyncio.as_completed(tasks),
            unique_fit_items_list
        ):
            config = item.config
            model_id = config.id
            model_type = config.ml_model_type
            try:
                model = await task
                active_processes.value -= 1

                model_file_path = (
                    self.app_config.models_dir_path / f"{model_id}.cloudpickle"
                )
                with model_file_path.open('wb') as f:
                    cloudpickle.dump(model, f)

                self.models[model_id] = {
                    "type": model_type,
                    "saved_model_file_path": model_file_path
                }

                response_list.append(MessageResponse(
                    message=f"Model '{item.config.id}' trained and saved."
                ))
            except InvalidFitPredictDataError as e:
                response_list.append(MessageResponse(
                    message=f"Model '{item.config.id}' failed: {e.detail}"
                ))
            except Exception as e:
                response_list.append(MessageResponse(
                    message=(
                        f"Model '{item.config.id}' unexpected error: {str(e)}"
                    )
                ))

        return response_list

    def _train_model(self, data: FitRequest):
        config = data.config
        model_type = config.ml_model_type
        vec_type = config.vectorizer_type
        try:
            estimator = clone(available_models[model_type]).set_params(**config.ml_model_params)
            vec = clone(available_vectorizers[vec_type]).set_params(**config.vectorizer_params)
            pipe = Pipeline(steps=[('preproc', CustomTokenizer()), 
                                   ('vec', vec), 
                                   ('estimator', estimator)])
            pipe.fit(data.X, data.y)
            print(pipe['vec'].get_params())
        except ValueError as e:
            raise InvalidFitPredictDataError(e.args[0])

        return pipe

    async def load_model(self, model_id: str) -> list[MessageResponse]:
        if len(self.loaded_models) == self.app_config.models_max_cnt:
            raise ModelsLimitExceededError()
        if model_id not in self.models:
            raise ModelNotFoundError(model_id)
        
        with self.models[model_id].get("saved_model_file_path").open('rb') as f:
            self.loaded_models[model_id] = cloudpickle.load(f)

        return [MessageResponse(message=f"Model '{model_id}' loaded.")]

    async def get_status(self) -> list[GetStatusResponse]:
        return [
            GetStatusResponse(status=f"Model '{model_id}' Status Ready")
            for model_id in self.loaded_models.keys()
        ]

    async def unload_model(self, model_id: str) -> list[MessageResponse]:
        if model_id not in self.models:
            raise ModelNotFoundError(model_id)

        self.loaded_models.pop(model_id)
        return [MessageResponse(message=f"Model '{model_id}' unloaded.")]

    async def predict(
        self,
        predict_data: list[PredictRequest]
    ) -> list[PredictResponse]:
        unique_predict_items = {}
        for item in predict_data:
            if item.id not in unique_predict_items:
                unique_predict_items[item.id] = item
        unique_predict_items_list = list(unique_predict_items.values())

        for item in unique_predict_items_list:
            model_id = item.id
            if model_id not in self.models:
                raise ModelNotFoundError(model_id)
            if model_id not in self.loaded_models:
                raise ModelNotLoadedError(model_id)

        model_preds = []
        for item in unique_predict_items_list:
            model = self.loaded_models.get(item.id)
            model_preds.append(PredictResponse(
                predictions=model.predict(item.X).tolist()
            ))

        return model_preds

    async def list_models(self) -> list[ModelListResponse]:
        models_list = [
            {"id": model_id, "type": model_info["type"]}
            for model_id, model_info in self.models.items()
        ]
        return [ModelListResponse(models=models_list)]

    async def remove_model(self, model_id: str) -> list[MessageResponse]:
        if model_id not in self.models:
            raise ModelNotFoundError(model_id)

        saved_model_filepath = self.models[model_id]["saved_model_file_path"]
        self.models.pop(model_id)
        self.loaded_models.pop(model_id, None)
        if os.path.isfile(saved_model_filepath):
            os.remove(saved_model_filepath)

        return [MessageResponse(message=f"Model '{model_id}' removed.")]

    async def remove_all_models(self) -> list[MessageResponse]:
        saved_model_file_paths = [
            model_info["saved_model_file_path"]
            for model_info in self.models.values()
        ]
        self.models.clear()
        self.loaded_models.clear()
        for file_path in saved_model_file_paths:
            if os.path.isfile(file_path):
                os.remove(file_path)

        return [MessageResponse(message="All models removed.")]
