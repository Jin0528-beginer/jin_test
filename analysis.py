"""AI Insight Engine 분석 로직 모음.

이 모듈은 Streamlit 같은 UI 프레임워크를 import하지 않는다.
(UI 코드는 streamlit_app.py에서만 다루고, 여기는 순수하게
'CSV -> 분석 결과' 를 만드는 함수만 모아둔다. 이렇게 나눠두면
UI 없이도 함수 단위로 테스트하거나, 나중에 다른 UI로 바꿀 때도
이 파일은 그대로 재사용할 수 있다.)
"""

import random
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# 기본 stopword 목록 (필요하면 이 리스트에 단어를 추가/삭제해서 조정)
DEFAULT_STOPWORDS = [
    "청년", "지역", "광주", "전남", "정보", "경우", "부분", "요즘",
    "실제로", "개인적으로", "생각합니다", "좋겠습니다", "어렵다",
    "어렵습니다", "필요하다", "필요합니다", "있으면",
]


def load_model(model_name=MODEL_NAME):
    # SentenceTransformer 로딩은 몇 초~수십 초가 걸리는 무거운 작업이다.
    # 이 함수 자체는 캐싱을 모른다 -> 캐싱(st.cache_resource)은
    # streamlit_app.py 쪽에서 이 함수를 감싸서 처리한다.
    return SentenceTransformer(model_name)


def read_and_clean_csv(file):
    # file: 파일 경로(str) 또는 파일 객체(BytesIO, Streamlit UploadedFile 등) 모두 허용
    df = None
    for enc in ["utf-8-sig", "utf-8", "cp949"]:
        try:
            if hasattr(file, "seek"):
                # 파일 객체는 이전 인코딩 시도에서 읽은 위치가 남아있으므로
                # 다시 처음부터 읽을 수 있도록 위치를 되돌려야 한다
                file.seek(0)
            df = pd.read_csv(file, encoding=enc)
            break
        except UnicodeDecodeError:
            pass

    if df is None:
        raise ValueError("CSV 인코딩을 읽지 못했습니다.")

    if "text" not in df.columns:
        raise ValueError(f"'text' 컬럼이 필요합니다. 현재 컬럼: {list(df.columns)}")

    df = df.copy()
    df["text"] = df["text"].astype("string").str.strip()
    df = df.dropna(subset=["text"])
    df = df[df["text"] != ""]
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)

    return df


def get_cluster_keywords(df, text_col="text", cluster_col="cluster", top_n=6, extra_stopwords=None):
    stopwords = list(DEFAULT_STOPWORDS)
    if extra_stopwords:
        stopwords += list(extra_stopwords)

    # cluster별로 문장을 전부 이어 붙여 '문서 1개'로 만든다
    # (TF-IDF는 문서들 사이에서 상대적으로 중요한 단어를 찾는 방식이라
    #  cluster 하나 = 문서 하나로 취급해야 cluster를 대표하는 키워드를 뽑을 수 있다)
    cluster_ids = sorted(df[cluster_col].unique())
    cluster_documents = [
        " ".join(df.loc[df[cluster_col] == cid, text_col].astype(str))
        for cid in cluster_ids
    ]

    vectorizer = TfidfVectorizer(
        token_pattern=r"[가-힣]{2,}",  # 완성형 한글 2글자 이상만 단어로 인정
        ngram_range=(1, 2),  # unigram + bigram
        stop_words=stopwords,
    )

    tfidf_matrix = vectorizer.fit_transform(cluster_documents)
    feature_names = vectorizer.get_feature_names_out()

    rows = []
    for row_idx, cid in enumerate(cluster_ids):
        row_scores = tfidf_matrix[row_idx].toarray().ravel()
        top_indices = row_scores.argsort()[::-1][:top_n]
        top_keywords = [feature_names[i] for i in top_indices]
        rows.append({"cluster": cid, "keywords": ", ".join(top_keywords)})

    return pd.DataFrame(rows)


def get_representative_sentence(df, embeddings, kmeans, cluster_col="cluster", text_col="text"):
    # cluster별 대표 문장 = 해당 cluster 중심점(centroid)에 가장 가까운 실제 문장
    # embeddings가 정규화되어 있으므로 내적(dot product) = cosine similarity
    representative = {}

    for cluster_id, centroid in enumerate(kmeans.cluster_centers_):
        cluster_mask = (df[cluster_col] == cluster_id).to_numpy()
        cluster_indices = np.where(cluster_mask)[0]
        cluster_embeddings = embeddings[cluster_indices]

        similarities = cluster_embeddings @ centroid
        best_local_idx = similarities.argmax()
        best_row_idx = cluster_indices[best_local_idx]

        representative[cluster_id] = df.iloc[best_row_idx][text_col]

    return representative


def semantic_search(query, df, embeddings, model, top_k=5, text_col="text", cluster_col="cluster"):
    # 검색어도 문장들과 같은 방식(normalize_embeddings=True)으로 인코딩해야
    # 내적만으로 cosine similarity를 정확히 계산할 수 있다
    query_embedding = model.encode([query], normalize_embeddings=True)[0]
    similarities = embeddings @ query_embedding

    top_indices = similarities.argsort()[::-1][:top_k]

    return pd.DataFrame({
        "rank": range(1, len(top_indices) + 1),
        "score": similarities[top_indices].round(4),
        "cluster": df.iloc[top_indices][cluster_col].values,
        "text": df.iloc[top_indices][text_col].values,
    })


def build_analysis(file, n_clusters, model, seed=SEED):
    """
    CSV 파일 하나로 아래 파이프라인을 전부 실행한다.
    정제 -> embedding -> clustering -> keywords -> 대표 의견 -> topic map

    model은 (재로딩 비용이 크므로) 호출하는 쪽에서 이미 로드/캐싱된
    SentenceTransformer 인스턴스를 넘겨받아 사용한다.
    """

    # 1) CSV 정제
    df = read_and_clean_csv(file)

    # 2) Embedding
    texts = df["text"].tolist()
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=False,
        normalize_embeddings=True,
    )

    # 3) Clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init="auto")
    df["cluster"] = kmeans.fit_predict(embeddings)

    # 4) Cluster별 keyword
    cluster_keywords_df = get_cluster_keywords(df)

    # 5) 대표 의견 + Topic Summary
    opinion_counts = df["cluster"].value_counts().sort_index()
    representative_sentences = get_representative_sentence(df, embeddings, kmeans)

    topic_summary = cluster_keywords_df.copy()
    topic_summary["opinion_count"] = topic_summary["cluster"].map(opinion_counts)
    topic_summary["representative_text"] = topic_summary["cluster"].map(representative_sentences)
    topic_summary = topic_summary[["cluster", "opinion_count", "keywords", "representative_text"]]

    # 6) Topic Map (PCA 2D + Plotly)
    pca = PCA(n_components=2, random_state=seed)
    coords_2d = pca.fit_transform(embeddings)
    df["x"] = coords_2d[:, 0]
    df["y"] = coords_2d[:, 1]
    df["cluster_label"] = df["cluster"].astype(str)  # 숫자 그대로 두면 색이 그라데이션으로 오해됨

    topic_map_fig = px.scatter(
        df,
        x="x",
        y="y",
        color="cluster_label",
        hover_data={"text": True, "x": False, "y": False, "cluster_label": False},
        title="Interactive Topic Map (PCA 2D)",
    )
    topic_map_fig.update_layout(legend_title_text="cluster")

    return {
        "df": df,
        "embeddings": embeddings,
        "kmeans": kmeans,
        "n_clusters": n_clusters,
        "topic_summary": topic_summary,
        "topic_map_fig": topic_map_fig,
    }
