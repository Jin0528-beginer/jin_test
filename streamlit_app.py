"""AI Insight Engine - Streamlit entrypoint.

실행: streamlit run streamlit_app.py

이 파일은 UI(위젯 배치, 캐싱, 세션 상태)만 담당하고,
실제 분석 로직은 전부 analysis.py의 함수를 호출해서 처리한다.
"""

import streamlit as st

from analysis import MODEL_NAME, build_analysis, load_model, semantic_search

st.set_page_config(page_title="AI Insight Engine", page_icon="🔎", layout="wide")


@st.cache_resource
def get_model():
    # 요구사항: SentenceTransformer는 st.cache_resource로 감싸서
    # 사용자가 앱을 여러 번 조작해도 모델을 한 번만 로드한다
    return load_model(MODEL_NAME)


def init_session_state():
    # 요구사항: 분석 결과는 st.session_state로 유지
    # (Streamlit은 위젯을 조작할 때마다 스크립트 전체를 다시 실행하므로,
    #  session_state에 저장해두지 않으면 Analyze 결과가 바로 사라진다)
    st.session_state.setdefault("analysis", None)
    st.session_state.setdefault("search_results", None)


def render_analyze_section():
    st.subheader("1. Analyze")

    col_input, col_result = st.columns([1, 2])

    with col_input:
        uploaded_file = st.file_uploader("CSV Upload", type=["csv"])
        n_clusters = st.slider("Number of Topics (k)", min_value=2, max_value=15, value=7, step=1)
        analyze_clicked = st.button("✨ Analyze", type="primary", use_container_width=True)

    if analyze_clicked:
        if uploaded_file is None:
            st.error("CSV 파일을 먼저 업로드하세요.")
        else:
            with st.spinner("분석 중입니다... (처음 실행 시 모델 다운로드로 시간이 더 걸릴 수 있습니다)"):
                model = get_model()
                st.session_state["analysis"] = build_analysis(uploaded_file, n_clusters, model=model)
                st.session_state["search_results"] = None  # 새로 분석했으면 이전 검색 결과는 초기화

    analysis = st.session_state["analysis"]
    if analysis is None:
        with col_result:
            st.info("CSV를 업로드하고 Analyze를 눌러주세요.")
        return

    with col_result:
        st.markdown(f"**분석된 의견 수:** {len(analysis['df']):,}개 (cluster {analysis['n_clusters']}개)")
        st.markdown("#### Topic Summary")
        st.dataframe(analysis["topic_summary"], use_container_width=True)

        # 요구사항: 결과 CSV download
        export_df = analysis["df"].drop(columns=["cluster_label"], errors="ignore")
        csv_bytes = export_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="⬇️ 분석 결과 CSV 다운로드",
            data=csv_bytes,
            file_name="analysis_result.csv",
            mime="text/csv",
        )

    st.markdown("#### Topic Map")
    st.plotly_chart(analysis["topic_map_fig"], use_container_width=True)


def render_search_section():
    st.subheader("2. Semantic Search")

    analysis = st.session_state["analysis"]
    if analysis is None:
        st.info("먼저 CSV를 업로드하고 Analyze를 실행하세요.")
        return

    query = st.text_input("Query", placeholder="예: 버스 배차간격이 너무 길어요")
    top_k = st.slider("Top-K", min_value=1, max_value=20, value=5, step=1)

    if st.button("🔍 Search"):
        if not query.strip():
            st.error("검색어를 입력하세요.")
        else:
            model = get_model()
            st.session_state["search_results"] = semantic_search(
                query,
                analysis["df"],
                analysis["embeddings"],
                model,
                top_k=top_k,
            )

    # session_state에 저장해두었기 때문에, 이후 다른 위젯을 조작해서
    # 스크립트가 다시 실행되어도 검색 결과가 그대로 남아있다
    if st.session_state["search_results"] is not None:
        st.dataframe(st.session_state["search_results"], use_container_width=True)


def main():
    st.title("🔎 AI Insight Engine")
    st.caption("CSV 의견 데이터를 업로드하면 Topic으로 자동 정리하고, 자연어로 검색할 수 있습니다.")

    init_session_state()
    render_analyze_section()
    st.divider()
    render_search_section()


if __name__ == "__main__":
    main()
