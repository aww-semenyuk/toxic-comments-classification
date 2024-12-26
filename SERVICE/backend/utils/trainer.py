from pathlib import Path

import cloudpickle
import pandas as pd

import spacy
import nltk
import unicodedata

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from background_tasks import prepare_fit_data
from exceptions import InvalidFitPredictDataError
from serializers.trainer import MLModelType, VectorizerType, MLModelConfig


class FunctionWrapper:
    """
    Wrapper to support serialization in multiprocessing
    credit: https://stackoverflow.com/a/75720040
    """
    def __init__(self, fn):
        self.fn_ser = cloudpickle.dumps(fn)

    def __call__(self, *args, **kwargs):
        fn = cloudpickle.loads(self.fn_ser)
        return fn(*args, **kwargs)

AVAILABLE_ESTIMATORS = {
    MLModelType.LogisticRegression: LogisticRegression(), 
    MLModelType.MultinomialNB: MultinomialNB(),
    MLModelType.LinearSVC: LinearSVC()
}

AVAILABLE_VECTORIZERS = {
    VectorizerType.CountVectorizer: CountVectorizer(), 
    VectorizerType.TfidfVectorizer: TfidfVectorizer()
}

VEC_PARAMS_FOR_SPACY_TOKENIZED_TEXT = {
    'tokenizer': FunctionWrapper(lambda x: x.split('\t')),
    'strip_accents': None,
    'lowercase': False,
    'preprocessor': None,
    'stop_words': None,
    'token_pattern': None
}

class SpacyTokenizer(BaseEstimator, TransformerMixin):
    nlp = spacy.load('en_core_web_sm')
    stopwords = set(nltk.corpus.stopwords.words('english'))

    def __init__(self, sep='\t', n_process=1, batch_size=64):
        self.sep = sep
        self.n_process = n_process
        self.batch_size = batch_size

    @staticmethod
    def normalize_text(doc):
        return unicodedata.normalize('NFKD', doc).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        results = []

        pipe = self.nlp.pipe(map(self.normalize_text, X), disable=['ner', 'parser'], batch_size=self.batch_size, n_process=self.n_process)

        for doc in pipe:
            results.append(self.sep.join([token.lemma_.lower() for token in doc if not (token.lemma_.lower() in self.stopwords or token.is_space or token.is_punct)]))

        return results


def make_pipeline_from_config(config: MLModelConfig):
    estimator = clone(AVAILABLE_ESTIMATORS[config.ml_model_type]).set_params(**config.ml_model_params)
    vec = clone(AVAILABLE_VECTORIZERS[config.vectorizer_type]).set_params(**config.vectorizer_params)

    if config.custom_tokenizer == "spacy_lemma":
        vec = vec.set_params(**VEC_PARAMS_FOR_SPACY_TOKENIZED_TEXT)
        pipe = Pipeline(steps=[('tok', SpacyTokenizer()), 
                               ('vec', vec), 
                               ('estimator', estimator)])
    else:
        pipe = Pipeline(steps=[('vec', vec), 
                               ('estimator', estimator)])

    return pipe


def train_and_save_model_task(
    models_dir_path: Path,
    model_config: MLModelConfig,
    fit_dataset: pd.DataFrame
) -> Path:
    # TODO: Заменить затычку на настоящую функцию (1)
    X, y = prepare_fit_data()

    model_id = model_config.id
    model_type = model_config.ml_model_type
    try:
        pipe = make_pipeline_from_config(model_config)
        pipe.fit(X, y)
    except ValueError as e:
        raise InvalidFitPredictDataError(e.args[0])

    safe_model_id = "".join(
        char for char in model_id if char.isalnum() or char in ('-', '_')
    ).rstrip()
    model_file_path = models_dir_path / f"{safe_model_id}.cloudpickle"
    with model_file_path.open('wb') as file:
        cloudpickle.dump(pipe, file)

    return model_file_path
