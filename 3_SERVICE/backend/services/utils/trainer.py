from pathlib import Path
import unicodedata

import cloudpickle
import pandas as pd

import spacy
import nltk
from sklearn import pipeline as sk_pipeline

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from transformers import pipeline as tf_pipeline

from serializers import MLModelType, VectorizerType, MLModelConfig
from serializers.utils.trainer import serialize_params
from settings import MODELS_DIR

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
    """Tokenizer using Spacy."""
    nlp = spacy.load('en_core_web_sm')
    stopwords = set(nltk.corpus.stopwords.words('english'))

    def __init__(self, sep='\t', n_process=1, batch_size=64):
        self.sep = sep
        self.n_process = n_process
        self.batch_size = batch_size

    @staticmethod
    def normalize_text(doc):
        """Normalize the text."""
        return unicodedata.normalize(
            'NFKD', doc
        ).encode(
            'ascii',
            'ignore'
        ).decode(
            'utf-8',
            'ignore'
        )

    def fit(self, X, y=None):
        """Fit the tokenizer."""
        return self

    def transform(self, X):
        """Transform the input data."""
        results = []

        pipe = self.nlp.pipe(
            map(self.normalize_text, X),
            disable=['ner', 'parser'],
            batch_size=self.batch_size,
            n_process=self.n_process
        )

        for doc in pipe:
            results.append(self.sep.join([
                token.lemma_.lower() for token in doc
                if not (
                    token.lemma_.lower() in self.stopwords
                    or token.is_space
                    or token.is_punct
                )
            ]))

        return results


def make_pipeline_from_config(
    config: MLModelConfig
) -> tuple[Pipeline, dict, dict]:
    """Make a pipeline from the configuration."""
    estimator = AVAILABLE_ESTIMATORS[config.ml_model_type]().set_params(
        **config.ml_model_params
    )
    vectorizer = AVAILABLE_VECTORIZERS[config.vectorizer_type]().set_params(
        **config.vectorizer_params
    )

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
    model_config: MLModelConfig,
    fit_dataset: pd.DataFrame
) -> tuple[Path, dict, dict]:
    """Train and save a model."""
    model_name = model_config.name

    X = fit_dataset["comment_text"]
    y = fit_dataset["toxic"]

    pipe, model_params, vectorizer_params = make_pipeline_from_config(
        model_config
    )
    pipe.fit(X, y)

    model_file_path = MODELS_DIR / f"{model_name}.cloudpickle"
    with model_file_path.open('wb') as file:
        cloudpickle.dump(pipe, file)

    return model_file_path, model_params, vectorizer_params


def get_dl_model_predictions(
    model,
    texts: list[str],
    return_scores: bool = False
) -> list[int]:
    predictions = []
    if return_scores:
        results = model(texts, top_k=2)
        for result in results:
            for pred in result:
                if pred["label"] == "LABEL_1":
                    predictions.append(pred["score"])
    else:
        results = model(texts)
        for result in results:
            predictions.append(1 if result["label"] == "LABEL_1" else 0)

    return predictions


def get_dl_model(
    saved_model_path: str,
    tokenizer_name: str
) -> tuple[tf_pipeline, dict, dict]:
    pipe = tf_pipeline(
        "text-classification",
        model=saved_model_path,
        tokenizer=tokenizer_name
    )
    model_params = serialize_params(pipe.model.config.to_dict())
    vectorizer_params = serialize_params(pipe.tokenizer.init_kwargs)
    return pipe, model_params, vectorizer_params


def get_ml_model(saved_model_path: str) -> tuple[sk_pipeline, dict, dict]:
    with open(saved_model_path, "rb") as f:
        pipe = cloudpickle.load(f)

    model_params = serialize_params(
        pipe.named_steps["classifier"].get_params()
    )
    vectorizer_params = serialize_params(
        pipe.named_steps["vectorizer"].get_params()
    )

    return pipe, model_params, vectorizer_params
