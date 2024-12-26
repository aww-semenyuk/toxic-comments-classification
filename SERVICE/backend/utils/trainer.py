import cloudpickle

import spacy
import nltk
import unicodedata

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from serializers.trainer import MLModelType, VectorizerType, FitConfig


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

def make_pipeline(config: FitConfig):
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
