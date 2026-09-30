import string
import itertools

import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from nltk import tokenize
from nltk.corpus import stopwords
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle
from wordcloud import WordCloud

# ---------------------------------------------------------------------------
# Page config & styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .block-container { padding-top: 2rem; max-width: 1200px; }

    .hero {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #db2777 100%);
        padding: 2.6rem 2.2rem;
        border-radius: 22px;
        color: white;
        margin-bottom: 1.6rem;
        box-shadow: 0 18px 40px rgba(79, 70, 229, 0.28);
    }
    .hero h1 { margin: 0; font-size: 2.6rem; font-weight: 800; letter-spacing: -0.5px; }
    .hero p  { margin: .6rem 0 0 0; font-size: 1.08rem; opacity: .92; max-width: 760px; }

    .card {
        background: #ffffff;
        border: 1px solid #eceef5;
        border-radius: 18px;
        padding: 1.3rem 1.4rem;
        box-shadow: 0 6px 22px rgba(30, 41, 59, 0.06);
        height: 100%;
    }
    .card h4 { margin: 0 0 .35rem 0; color: #1e1b4b; font-weight: 700; }
    .card p  { margin: 0; color: #475569; font-size: .95rem; }

    .stat {
        background: linear-gradient(180deg, #ffffff 0%, #f6f5ff 100%);
        border: 1px solid #e4e1ff;
        border-radius: 18px;
        padding: 1.2rem 1.3rem;
        box-shadow: 0 6px 18px rgba(79, 70, 229, 0.08);
    }
    .stat .label { color: #6b7280; font-size: .82rem; font-weight: 600; text-transform: uppercase; letter-spacing: .06em; }
    .stat .value { color: #3730a3; font-size: 2rem; font-weight: 800; margin-top: .15rem; }

    .result-real {
        background: linear-gradient(135deg, #10b981, #059669);
        color: white; padding: 1.6rem; border-radius: 18px; text-align: center;
        box-shadow: 0 12px 28px rgba(16, 185, 129, .3);
    }
    .result-fake {
        background: linear-gradient(135deg, #f43f5e, #be123c);
        color: white; padding: 1.6rem; border-radius: 18px; text-align: center;
        box-shadow: 0 12px 28px rgba(244, 63, 94, .3);
    }
    .result-real h2, .result-fake h2 { margin: 0; font-size: 2rem; font-weight: 800; }
    .result-real p, .result-fake p   { margin: .3rem 0 0 0; opacity: .95; }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e1b4b 0%, #312e81 100%);
    }
    section[data-testid="stSidebar"] * { color: #e0e7ff !important; }

    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background: #f1f0ff; border-radius: 12px; padding: 10px 20px; font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
        color: white !important;
    }
    .stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }

    .stButton > button {
        background: linear-gradient(135deg, #4f46e5, #7c3aed);
        color: white; border: none; border-radius: 12px;
        padding: .65rem 1.6rem; font-weight: 700;
        box-shadow: 0 8px 20px rgba(79, 70, 229, .3);
    }
    .stButton > button:hover { transform: translateY(-1px); color: white; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Original code (unchanged logic), wrapped into cached functions
# ---------------------------------------------------------------------------
def punctuation_removal(text):
    all_list = [char for char in text if char not in string.punctuation]
    clean_str = ''.join(all_list)
    return clean_str


@st.cache_data(show_spinner=False)
def load_and_prepare():
    nltk.download('stopwords', quiet=True)
    stop = stopwords.words('english')

    # Read datasets
    fake = pd.read_csv("Fake\\Fake.csv")
    true = pd.read_csv("True\\True.csv")

    raw_shapes = (fake.shape, true.shape)

    # Add flag to track fake and real
    fake['target'] = 'fake'
    true['target'] = 'true'

    # Concatenate dataframes
    data = pd.concat([fake, true]).reset_index(drop=True)

    # Shuffle the data
    data = shuffle(data)
    data = data.reset_index(drop=True)


    # Removing the date
    data.drop(["date"], axis=1, inplace=True)

    # Removing the title
    data.drop(["title"], axis=1, inplace=True)

    # Convert to lowercase
    data['text'] = data['text'].apply(lambda x: x.lower())

    # Remove punctuation
    data['text'] = data['text'].apply(punctuation_removal)

    # Removing stopwords
    data['text'] = data['text'].apply(
        lambda x: ' '.join([word for word in x.split() if word not in (stop)])
    )
    return data, raw_shapes


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


@st.cache_resource(show_spinner=False)
def train_model(_data):
    data = _data
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        data['text'], data.target, test_size=0.2, random_state=42
    )

    # Decision Tree Classifier: vectorizing and applying TF-IDF
    pipe = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('model', DecisionTreeClassifier(criterion='entropy',
                                                      max_depth=20,
                                                      splitter='best',
                                                      random_state=42))])
    # Fitting the model
    model = pipe.fit(X_train, y_train)

    # Accuracy
    prediction = model.predict(X_test)
    acc = round(accuracy_score(y_test, prediction) * 100, 2)
    cm = metrics.confusion_matrix(y_test, prediction)
    return model, acc, cm, len(X_train), len(X_test)


token_space = tokenize.WhitespaceTokenizer()


def counter(text, column_text, quantity, color):
    all_words = ' '.join([text for text in text[column_text]])
    token_phrase = token_space.tokenize(all_words)
    frequency = nltk.FreqDist(token_phrase)
    df_frequency = pd.DataFrame({"Word": list(frequency.keys()),
                                 "Frequency": list(frequency.values())})
    df_frequency = df_frequency.nlargest(columns="Frequency", n=quantity)
    fig = plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=df_frequency, x="Word", y="Frequency", color=color)
    ax.set(ylabel="Count")
    plt.xticks(rotation='vertical')
    plt.tight_layout()
    return fig


def make_wordcloud(df, label, colormap):
    subset = df[df["target"] == label]
    all_words = ' '.join([text for text in subset.text])
    wordcloud = WordCloud(width=800, height=500,
                          max_font_size=110,
                          collocations=False,
                          background_color="white",
                          colormap=colormap).generate(all_words)
    fig = plt.figure(figsize=(10, 7))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.tight_layout()
    return fig


def clean_input(text):
    """Same preprocessing as the training data."""
    stop = stopwords.words('english')
    text = text.lower()
    text = punctuation_removal(text)
    return ' '.join([word for word in text.split() if word not in (stop)])


# ---------------------------------------------------------------------------
# Load data + model
# ---------------------------------------------------------------------------
try:
    with st.spinner("Loading and cleaning dataset..."):
        data, (fake_shape, true_shape) = load_and_prepare()
except FileNotFoundError:
    st.error(
        "Could not find `Fake\\Fake.csv` and `True\\True.csv`. "
        "Run this app from the project folder that contains the `Fake` and `True` folders."
    )
    st.stop()

with st.spinner("Training Decision Tree model (first run only)..."):
    model, acc, cm, n_train, n_test = train_model(data)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 📰 Fake News Detector")
    st.markdown("Classify news articles as **real** or **fake** using NLP and machine learning.")
    st.markdown("---")
    st.markdown("**Pipeline**")
    st.markdown("1. Lowercase\n2. Remove punctuation\n3. Remove stopwords\n4. CountVectorizer\n5. TF-IDF\n6. Decision Tree")
    st.markdown("---")
    st.markdown("**Model settings**")
    st.markdown("criterion: `entropy`  \nmax_depth: `20`  \nsplitter: `best`  \nrandom_state: `42`")
    st.markdown("---")
    st.caption("Built with Streamlit and scikit-learn")

# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>📰 Fake News Detection</h1>
        <p>Fake news are stories that are false, manipulated, have no solid proof or come from
        unreliable sources. Explore the dataset, inspect the model and check your own articles.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_overview, tab_explore, tab_model, tab_predict = st.tabs(
    ["🏠 Overview", "📊 Data Exploration", "🤖 Model Performance", "🔍 Try It Yourself"]
)

# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------
with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    stats = [
        ("Total Articles", f"{len(data):,}"),
        ("Fake Articles", f"{fake_shape[0]:,}"),
        ("Real Articles", f"{true_shape[0]:,}"),
        ("Model Accuracy", f"{acc}%"),
    ]
    for col, (label, value) in zip([c1, c2, c3, c4], stats):
        col.markdown(
            f'<div class="stat"><div class="label">{label}</div><div class="value">{value}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    a, b, c = st.columns(3)
    a.markdown(
        '<div class="card"><h4>🧹 Clean</h4><p>Articles are lowercased, stripped of punctuation '
        'and stopwords before modelling.</p></div>', unsafe_allow_html=True)
    b.markdown(
        '<div class="card"><h4>🔢 Vectorize</h4><p>Text is converted to word counts and weighted '
        'with TF-IDF.</p></div>', unsafe_allow_html=True)
    c.markdown(
        '<div class="card"><h4>🌳 Classify</h4><p>A Decision Tree with entropy criterion and '
        'depth 20 predicts fake vs real.</p></div>', unsafe_allow_html=True)

    st.markdown("### Dataset preview")
    st.dataframe(data.head(10), use_container_width=True)

# ---------------------------------------------------------------------------
# Data exploration
# ---------------------------------------------------------------------------
with tab_explore:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Articles per subject")
        subj = data.groupby(['subject'])['text'].count()
        fig = plt.figure(figsize=(6, 4))
        subj.plot(kind="bar", color="#6366f1")
        plt.tight_layout()
        st.pyplot(fig)
        st.dataframe(subj.rename("count"), use_container_width=True)

    with col2:
        st.markdown("#### Fake vs real articles")
        tgt = data.groupby(['target'])['text'].count()
        fig = plt.figure(figsize=(6, 4))
        tgt.plot(kind="bar", color=["#f43f5e", "#10b981"])
        plt.tight_layout()
        st.pyplot(fig)
        st.dataframe(tgt.rename("count"), use_container_width=True)

    st.markdown("---")
    st.markdown("#### Word clouds")
    w1, w2 = st.columns(2)
    with w1:
        st.markdown("**Fake news**")
        st.pyplot(make_wordcloud(data, "fake", "Reds"))
    with w2:
        st.markdown("**Real news**")
        st.pyplot(make_wordcloud(data, "true", "Greens"))

    st.markdown("---")
    st.markdown("#### Most frequent words")
    top_n = st.slider("Number of words", 10, 50, 20)
    f1, f2 = st.columns(2)
    with f1:
        st.markdown("**Fake news**")
        st.pyplot(counter(data[data["target"] == "fake"], "text", top_n, "#f43f5e"))
    with f2:
        st.markdown("**Real news**")
        st.pyplot(counter(data[data["target"] == "true"], "text", top_n, "#10b981"))

# ---------------------------------------------------------------------------
# Model performance
# ---------------------------------------------------------------------------
with tab_model:
    m1, m2, m3 = st.columns(3)
    for col, (label, value) in zip(
        [m1, m2, m3],
        [("Accuracy", f"{acc}%"), ("Training samples", f"{n_train:,}"), ("Test samples", f"{n_test:,}")],
    ):
        col.markdown(
            f'<div class="stat"><div class="label">{label}</div><div class="value">{value}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    left, right = st.columns([1, 1])
    with left:
        st.markdown("#### Confusion matrix")
        fig = plt.figure(figsize=(5.5, 4.5))
        plot_confusion_matrix(cm, classes=['Fake', 'Real'])
        st.pyplot(fig)
    with right:
        st.markdown("#### Model details")
        st.markdown(
            """
            <div class="card">
            <p><b>Pipeline</b><br>CountVectorizer → TfidfTransformer → DecisionTreeClassifier</p><br>
            <p><b>Hyperparameters</b><br>criterion = entropy<br>max_depth = 20<br>
            splitter = best<br>random_state = 42</p><br>
            <p><b>Split</b><br>80% train / 20% test (random_state = 42)</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------------
# Try it yourself
# ---------------------------------------------------------------------------
with tab_predict:
    st.markdown("#### Paste a news article")
    user_text = st.text_area(
        "Article text",
        height=240,
        placeholder="Paste the body of a news article here...",
        label_visibility="collapsed",
    )

    if st.button("Analyze article"):
        if not user_text.strip():
            st.warning("Please paste some text first.")
        else:
            cleaned = clean_input(user_text)
            if not cleaned:
                st.warning("The text has no meaningful words after cleaning. Try a longer article.")
            else:
                pred = model.predict([cleaned])[0]
                if pred == "true":
                    st.markdown(
                        '<div class="result-real"><h2>✅ Looks like REAL news</h2>'
                        '<p>The model classified this article as real.</p></div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        '<div class="result-fake"><h2>🚨 Looks like FAKE news</h2>'
                        '<p>The model classified this article as fake.</p></div>',
                        unsafe_allow_html=True,
                    )
                with st.expander("See the cleaned text fed to the model"):
                    st.write(cleaned)

    st.caption("Predictions come from a Decision Tree and should not replace fact-checking.")