import re
import pandas as pd
import streamlit as st
from pandarallel import pandarallel
import nltk
import string
import plotly.subplots as sp
import plotly.graph_objects as go
import plotly.io as pio
from joblib import Parallel, delayed
from time import time
from collections import Counter
import unicodedata
from itertools import chain
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import base64
import io
import random

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('universal_tagset')


  

# Цвета для «обычных» и «токсичных» данных
COLOR_ORDINARY = "powderblue"
COLOR_TOXIC = "crimson"

st.set_page_config(layout="wide")
pandarallel.initialize(progress_bar=False)

class PreprocessLemmaTokenizer:
    def __init__(self):
        self.wnl = nltk.stem.WordNetLemmatizer()
        self.tknzr = nltk.tokenize.TweetTokenizer()
        self.stopwords = set(nltk.corpus.stopwords.words('english'))
        self.punctuation = set(string.punctuation)

        self.url_re = re.compile(r'(https?://\S+|www\.\S+)', re.IGNORECASE)
        self.symbols_re = re.compile(r'[^a-zA-Zа-яА-Я0-9\s]')
        self.spaces_re = re.compile(r'\s+')

    @staticmethod
    def _normalize(doc):
        return unicodedata.normalize('NFKD', doc).encode('ascii', 'ignore').decode('utf-8', 'ignore')

    @staticmethod
    def _get_wordnet_pos(treebank_tag):
        if treebank_tag.startswith('J'):
            return nltk.corpus.wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return nltk.corpus.wordnet.VERB
        elif treebank_tag.startswith('N'):
            return nltk.corpus.wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return nltk.corpus.wordnet.ADV
        else:
            return nltk.corpus.wordnet.NOUN

    def _clean_txt(self, text):
        c_t = self.url_re.sub('', text.lower())
        c_t = self.symbols_re.sub(' ', c_t)
        c_text = self.spaces_re.sub(' ', c_t).strip()
        return c_text

    def __call__(self, doc):
        clean_doc = self._normalize(self._clean_txt(doc))
        tokens = self.tknzr.tokenize(clean_doc)
        pos_tags = nltk.pos_tag(tokens)
        return [
            self.wnl.lemmatize(token, pos=self._get_wordnet_pos(pos_tag))
            for token, pos_tag in pos_tags
            if token.lower() not in self.stopwords and token not in self.punctuation
        ]

pos_tags_dict = {
    'NN': 'noun (sg.)',
    'JJ': 'adj.',
    'NNS': 'noun (pl.)',
    'RB': 'adv.',
    'VBP': 'verb (pres., not 3rd)',
    'VBG': 'gerund/pres.part.',
    'VBD': 'verb (past)',
    'VB': 'verb (inf.)',
    'CD': 'numeral',
    'VBN': 'verb (p.p.)',
    'IN': 'prep./conj.',
    'VBZ': 'verb (3rd sg.)',
    'MD': 'modal',
    'JJR': 'adj. (comp.)',
    'NNP': 'prop. noun (sg.)',
    'PRP': 'pers. pron.',
    'JJS': 'adj. (super.)',
    'DT': 'determiner',
    'RBR': 'adv. (comp.)',
    'FW': 'foreign word',
    'CC': 'coord. conj.',
    'RP': 'particle',
    'WDT': 'wh-det.',
    'UH': 'interj.',
    'WP$': 'wh-poss. pron.',
    'RBS': 'adv. (super.)',
    '$': 'dollar sign',
    'WRB': 'wh-adv.',
    'NNPS': 'prop. noun (pl.)',
    'WP': 'wh-pron.',
    'PRP$': 'poss. pron.',
    'POS': 'poss. end.',
    'EX': 'exist. "there"',
    'SYM': 'symbol',
    'TO': 'prep. "to"',
    "''": 'cl. quote',
    '``': 'op. quote',
    'PDT': 'predet.',
    'LS': 'list marker'
}

def process_tokens(tokens):
    pos_tags = nltk.pos_tag(tokens, lang='eng')
    return [tag for _, tag in pos_tags]

def extract_features(tokens_series, ngram_sizes=(2, 3)):
    unique_words = set(chain.from_iterable(tokens_series))
    unique_ngrams = {n: set() for n in ngram_sizes}

    def extract_ngrams(tokens, n):
        return {' '.join(gram) for gram in nltk.ngrams(tokens, n)}

    for n in ngram_sizes:
        ngrams_for_size = Parallel(n_jobs=-1)(
            delayed(extract_ngrams)(tokens, n)
            for tokens in tokens_series if len(tokens) >= n
        )
        unique_ngrams[n].update(chain.from_iterable(ngrams_for_size))

    return unique_words, unique_ngrams

def get_tfd(unique_tokens, cleaned_com_t, gram):
    if gram >= 2:
        vectorizer = CountVectorizer(vocabulary=unique_tokens, ngram_range=(gram, gram))
    else:
        vectorizer = CountVectorizer(vocabulary=unique_tokens)
    
    def process_chunk(chunk):
        X_chunk = vectorizer.fit_transform(chunk)
        return np.asarray(X_chunk.sum(axis=0))[0]
    
    n_jobs = -1
    if len(cleaned_com_t) < abs(n_jobs):
        chunk_size = len(cleaned_com_t)
    else:
        chunk_size = len(cleaned_com_t) // abs(n_jobs)
    
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_chunk)(cleaned_com_t[i:i + chunk_size])
        for i in range(0, len(cleaned_com_t), chunk_size)
    )

    total_counts = np.sum(results, axis=0)
    return dict(zip(vectorizer.get_feature_names_out(), total_counts))

#WordCloud генерация
COLOR_ORDINARY_RGB = (176, 224, 230) 
COLOR_TOXIC_RGB = (220, 20, 60)

def generate_wordcloud_base64(frequency_dict):
    palette = [COLOR_ORDINARY_RGB, COLOR_TOXIC_RGB]
    
    def random_color_func(*args, **kwargs):
        r, g, b = random.choice(palette)
        return f"rgb({r},{g},{b})"
    
    wordcloud = WordCloud(
        width=900, 
        height=450,
        background_color=None, 
        color_func=random_color_func
    ).generate_from_frequencies(frequency_dict)

    buffer = io.BytesIO()
    wordcloud.to_image().save(buffer, format="PNG")
    buffer.seek(0)
    return f"data:image/png;base64,{base64.b64encode(buffer.read()).decode()}"

# Функция отрисовки графика частей речи
def plot_pos_in_plotly(most_common_pos_o, most_common_pos_t):
    fig = sp.make_subplots(
        rows=1, cols=2,
        subplot_titles=("Ordinary comments", "Toxic comments"),
        horizontal_spacing=0.1
    )

    x_ord = [count for _, count in most_common_pos_o]
    y_ord = [pos for pos, _ in most_common_pos_o]
    x_tox = [count for _, count in most_common_pos_t]
    y_tox = [pos for pos, _ in most_common_pos_t]

    # Ordinary
    fig.add_trace(
        go.Bar(
            x=x_ord, 
            y=y_ord,
            orientation='h',
            marker_color=COLOR_ORDINARY,
            name="Ordinary POS"
        ),
        row=1, col=1
    )
    # Toxic
    fig.add_trace(
        go.Bar(
            x=x_tox, 
            y=y_tox,
            orientation='h',
            marker_color=COLOR_TOXIC,
            name="Toxic POS"
        ),
        row=1, col=2
    )

    fig.update_layout(
        title="Frequency of parts of speech",
        showlegend=True,
        height=600,
        width=1000,
    )

    fig.update_xaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="Part of speech", row=1, col=1)

    
    fig.update_yaxes(autorange='reversed')

    return fig

# Начало приложения

st.title('Classification of toxic comments service')
st.header('Data')

uploaded_file = st.file_uploader('Upload your CSV file', type=['csv'])
if uploaded_file is not None:
    start_time = time()
    data = pd.read_csv(uploaded_file, index_col=0)
    
    data[['text_length', 'num_words', 'num_sent', 'num_punct']] = (
        data['comment_text']
        .parallel_apply(
            lambda text: (
                len(text),
                len(nltk.tokenize.word_tokenize(text)),
                len(nltk.tokenize.sent_tokenize(text)),
                sum(ch in string.punctuation for ch in text)
            )
        )
        .apply(pd.Series)
    )
    
    data['tokens_ws'] = data['comment_text'].apply(PreprocessLemmaTokenizer())
    data['ctws'] = data.tokens_ws.apply(lambda x: ' '.join(x))

    # Обычные POS
    all_pos_o = Parallel(n_jobs=-1)(
        delayed(process_tokens)(tokens) 
        for tokens in data[data.toxic == 0]['tokens_ws']
    )
    all_pos_o = [tag for pos_list in all_pos_o for tag in pos_list]
    pos_cnt_o = Counter(all_pos_o)
    translated_counter_o = Counter({
        pos_tags_dict.get(tag, tag): count for tag, count in pos_cnt_o.items()
    })
    most_common_pos_o = translated_counter_o.most_common(20)

    # Токсичные POS
    all_pos_t = Parallel(n_jobs=-1)(
        delayed(process_tokens)(tokens) 
        for tokens in data[data.toxic == 1]['tokens_ws']
    )
    all_pos_t = [tag for pos_list in all_pos_t for tag in pos_list]
    pos_cnt_t = Counter(all_pos_t)
    translated_counter_t = Counter({
        pos_tags_dict.get(tag, tag): count for tag, count in pos_cnt_t.items()
    })
    most_common_pos_t = translated_counter_t.most_common(20)

    # Извлечение N-грамм
    tokens_non_toxic = data.loc[data.toxic == 0, 'tokens_ws']
    tokens_toxic = data.loc[data.toxic == 1, 'tokens_ws']
    uawo, ngrams_non_toxic = extract_features(tokens_non_toxic)
    uawt, ngrams_toxic = extract_features(tokens_toxic)

    bos = list(ngrams_non_toxic[2])
    tos = list(ngrams_non_toxic[3])
    bts = list(ngrams_toxic[2])
    tts = list(ngrams_toxic[3])

    uawo = list(uawo)
    uawt = list(uawt)

    # Частотные словари
    data_wf = {
        "Unigram": {
            "ordinary": get_tfd(uawo, data.loc[data.toxic == 0, 'ctws'], 1),
            "toxic": get_tfd(uawt, data.loc[data.toxic == 1, 'ctws'], 1)
        },
        "Bigram": {
            "ordinary": get_tfd(bos, data.loc[data.toxic == 0, 'ctws'], 2),
            "toxic": get_tfd(bts, data.loc[data.toxic == 1, 'ctws'], 2)
        },
        "Trigram": {
            "ordinary": get_tfd(tos, data.loc[data.toxic == 0, 'ctws'], 3),
            "toxic": get_tfd(tts, data.loc[data.toxic == 1, 'ctws'], 3)
        }
    }

    st.write(f"Data has been successfully uploaded and processed. Execution time: {time() - start_time:.2f} sec.")
    st.header("EDA")

    #  Графики распределения длины
    st.subheader("Distribution of length analysis")

    metrics = ["text_length", "num_words", "num_sent", "num_punct"]
    metric_names = {
        "text_length": "Character Length",
        "num_words": "Word Length",
        "num_sent": "Sentence Length",
        "num_punct": "Number of Punctuations"
    }

    def calculate_statistics(df, metric):
        return {
            "mean": df[metric].mean(),
            "median": df[metric].median(),
            "max": df[metric].max(),
            "min": df[metric].min()
        }
# распределение слов
    fig_len = sp.make_subplots(
        rows=1, cols=2,
        subplot_titles=["Ordinary Comments", "Toxic Comments"],
        horizontal_spacing=0.1
    )

    def add_histograms(metric):
        toxic_data = data[data["toxic"] == 1][metric]
        non_toxic_data = data[data["toxic"] == 0][metric]

        fig_len.add_trace(go.Histogram(
            x=non_toxic_data,
            name=f'Ordinary ({metric_names[metric]})',
            marker_color=COLOR_ORDINARY,
            opacity=0.75,
            visible=(metric == metrics[0]),  
            xbins=dict(size=5)
        ), row=1, col=1)

        fig_len.add_trace(go.Histogram(
            x=toxic_data,
            name=f'Toxic ({metric_names[metric]})',
            marker_color=COLOR_TOXIC,
            opacity=0.75,
            visible=(metric == metrics[0]),
            xbins=dict(size=5)
        ), row=1, col=2)

    for m in metrics:
        add_histograms(m)

    # Кнопки переключения
    buttons = []
    for i, m in enumerate(metrics):
        visible = [False] * len(metrics) * 2
        visible[i*2] = True
        visible[i*2 + 1] = True

        non_toxic_stats = calculate_statistics(data[data["toxic"] == 0], m)
        toxic_stats = calculate_statistics(data[data["toxic"] == 1], m)

        annotations = [
            dict(
                x=0.0, y=-0.2, xref="paper", yref="paper", showarrow=False,
                text=(
                    f"Ordinary:\nMean: {non_toxic_stats['mean']:.2f}, "
                    f"Median: {non_toxic_stats['median']:.2f}, "
                    f"Max: {non_toxic_stats['max']}, Min: {non_toxic_stats['min']}"
                ),
                font=dict(size=12),
                align="center"
            ),
            dict(
                x=0.64, y=-0.2, xref="paper", yref="paper", showarrow=False,
                text=(
                    f"Toxic:\nMean: {toxic_stats['mean']:.2f}, "
                    f"Median: {toxic_stats['median']:.2f}, "
                    f"Max: {toxic_stats['max']}, Min: {toxic_stats['min']}"
                ),
                font=dict(size=12),
                align="center"
            )
        ]

        buttons.append(dict(
            label=metric_names[m],
            method="update",
            args=[
                {"visible": visible},
                {
                    "annotations": annotations,
                    "yaxis.autorange": True,
                    "yaxis2.autorange": True,
                    "title": f"Distribution of {metric_names[m]} by Toxicity"
                }
            ]
        ))

    fig_len.update_layout(
        updatemenus=[
            dict(
                active=0,
                buttons=buttons,
                direction="down",
                showactive=True,
                x=1.165,
                y=1.12
            )
        ],
        title="Distribution of Comment Lengths",
        xaxis_title="Length",
        yaxis_title="Count",
        height=600,
        width=1100,
        showlegend=True,
        legend=dict(
            x=1.05,
            y=1.0,
            xanchor="left",
            yanchor="auto"
        )
    )
    fig_len.update_layout(
        title={
            "text": "Distribution of Comment Lengths",
            "x": 0.0,
            "xanchor": "left",
            "yanchor": "top"
        }
    )

    # Вывод графика
    st.plotly_chart(fig_len, use_container_width=True)

    # Частота частей речи
    st.subheader("Frequency of parts of speech")
    fig_pos = plot_pos_in_plotly(most_common_pos_o, most_common_pos_t)
    st.plotly_chart(fig_pos, use_container_width=True)

    # Частота слов и N-грамм
    st.subheader("Bar Plots and Word Clouds for N-grams")

    fig_ngrams = sp.make_subplots(
        rows=2, cols=2,
        column_widths=[0.5, 0.5],
        row_heights=[0.5, 0.5],
        specs=[
            [{"type": "bar"}, {"type": "bar"}],
            [{"type": "image"}, {"type": "image"}]
        ],
        subplot_titles=("Ordinary", "Toxic", "", "")
    )

    image_configs = []
    ngram_keys = list(data_wf.keys()) 

    for i, (ngram_type, freqs) in enumerate(data_wf.items()):
        ordinary_df = (
            pd.DataFrame(list(freqs["ordinary"].items()), columns=["word", "count"])
            .sort_values(by="count", ascending=False)
            .head(30)
        )
        toxic_df = (
            pd.DataFrame(list(freqs["toxic"].items()), columns=["word", "count"])
            .sort_values(by="count", ascending=False)
            .head(30)
        )

        bar_ordinary = go.Bar(
            x=ordinary_df["count"],
            y=ordinary_df["word"],
            orientation="h",
            marker=dict(color=COLOR_ORDINARY),
            name=f"{ngram_type} Ordinary",
            visible=(i == 0)
        )
        bar_toxic = go.Bar(
            x=toxic_df["count"],
            y=toxic_df["word"],
            orientation="h",
            marker=dict(color=COLOR_TOXIC),
            name=f"{ngram_type} Toxic",
            visible=(i == 0)
        )

        fig_ngrams.add_trace(bar_ordinary, row=1, col=1)
        fig_ngrams.add_trace(bar_toxic,   row=1, col=2)

        ordinary_wc = generate_wordcloud_base64(freqs["ordinary"])
        toxic_wc = generate_wordcloud_base64(freqs["toxic"])

        image_configs.append([
            dict(
                source=ordinary_wc,
                xref="paper", yref="paper",
                x=0.0, y=0.1,
                sizex=0.7, sizey=0.4,
                xanchor="left", yanchor="bottom",
                layer="above"
            ),
            dict(
                source=toxic_wc,
                xref="paper", yref="paper",
                x=0.55, y=0.1,
                sizex=0.7, sizey=0.4,
                xanchor="left", yanchor="bottom",
                layer="above"
            )
        ])

    # По умолчанию показываем для первого слова
    fig_ngrams.update_layout(images=image_configs[0])

    # Кнопки переключения
    buttons_ngrams = []
    for i, ngram_type in enumerate(ngram_keys):
        visibility = [False] * len(fig_ngrams.data)
        visibility[i * 2] = True
        visibility[i * 2 + 1] = True

        buttons_ngrams.append(
            dict(
                label=ngram_type,
                method="update",
                args=[
                    {"visible": visibility},
                    {"images": image_configs[i]}
                ]
            )
        )

    fig_ngrams.update_layout(
        updatemenus=[
            dict(
                buttons=buttons_ngrams,
                direction="down",
                showactive=True,
                x=1.1,
                y=1.05,
                xanchor="center",
                yanchor="top"
            )
        ],
        #plot_bgcolor="white",
        #paper_bgcolor="white",
        height=1000,
        width=1200,
        title="Bar Plots and Word Clouds for N-grams",
        showlegend=True,
        legend=dict(
            x=1.05,
            y=1.0,
            xanchor="left",
            yanchor="auto"
        )
    )

    fig_ngrams.update_yaxes(autorange='reversed')
    st.plotly_chart(fig_ngrams, use_container_width=True)

else:
    st.write("Please upload your CSV file")