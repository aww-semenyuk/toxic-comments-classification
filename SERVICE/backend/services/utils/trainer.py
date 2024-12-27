from pathlib import Path
from typing import Any

import cloudpickle
import numpy as np
import pandas as pd

import spacy
import nltk
import unicodedata

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from serializers.trainer import MLModelType, VectorizerType, MLModelConfig

AVAILABLE_ESTIMATORS = {
    MLModelType.logistic_regression: LogisticRegression,
    MLModelType.multinomial_nb: MultinomialNB,
    MLModelType.linear_svc: LinearSVC
}
AVAILABLE_VECTORIZERS = {
    VectorizerType.count_vectorizer: CountVectorizer,
    VectorizerType.tfidf_vectorizer: TfidfVectorizer
}


class FunctionWrapper:
    """Wrapper to support serialization in multiprocessing."""

    def __init__(self, fn):
        self.fn_ser = cloudpickle.dumps(fn)

    def __call__(self, *args, **kwargs):
        fn = cloudpickle.loads(self.fn_ser)
        return fn(*args, **kwargs)


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


def make_pipeline_from_config(config: MLModelConfig) -> tuple[Pipeline, dict, dict]:
    estimator = AVAILABLE_ESTIMATORS[config.ml_model_type]().set_params(**config.ml_model_params)
    vectorizer = AVAILABLE_VECTORIZERS[config.vectorizer_type]().set_params(**config.vectorizer_params)

    if config.spacy_lemma_tokenizer:
        vectorizer = vectorizer.set_params(
            tokenizer=FunctionWrapper(lambda x: x.split('\t')),
            strip_accents=None,
            lowercase=False,
            preprocessor=None,
            stop_words=None,
            token_pattern=None
        )
        pipe = Pipeline(steps=[
            ('tok', SpacyTokenizer()),
            ('vec', vectorizer),
            ('estimator', estimator)
        ])
    else:
        pipe = Pipeline(steps=[
            ('vec', vectorizer),
            ('estimator', estimator)
        ])

    return pipe, estimator.get_params(), vectorizer.get_params()


def train_and_save_model_task(
    models_dir_path: Path,
    model_config: MLModelConfig,
    fit_dataset: pd.DataFrame
) -> tuple[Path, dict, dict]:
    model_id = model_config.id

    X = fit_dataset["comment_text"]
    y = fit_dataset["toxic"]

    pipe, model_params, vectorizer_params = make_pipeline_from_config(model_config)
    pipe.fit(X, y)

    safe_model_id = "".join(
        char for char in model_id if char.isalnum() or char in ('-', '_')
    ).rstrip()
    model_file_path = models_dir_path / f"{safe_model_id}.cloudpickle"
    with model_file_path.open('wb') as file:
        cloudpickle.dump(pipe, file)

    return model_file_path, model_params, vectorizer_params


def serialize_params(obj: Any) -> Any:
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    elif isinstance(obj, (list, tuple)):
        return [serialize_params(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: serialize_params(value) for key, value in obj.items()}
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, type):
        return obj.__name__
    elif callable(obj):
        return obj.__name__
    else:
        return str(obj)
