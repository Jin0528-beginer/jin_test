# AI Insight Engine

설문/의견 CSV를 업로드하면 문장을 embedding → clustering해서 Topic으로 자동 정리하고,
자연어로 의견을 검색할 수 있는 Streamlit 앱입니다.

## 기능

- CSV 업로드 + Topic 개수(k) 조절
- Topic별 keyword, 대표 의견, 의견 수를 담은 Topic Summary 표
- PCA 2D 기반 Interactive Topic Map (Plotly)
- 자연어 Semantic Search (Top-K)
- 분석 결과 CSV 다운로드

## 파일 구성

| 파일 | 역할 |
|---|---|
| `streamlit_app.py` | entrypoint. UI, `st.cache_resource`(모델 캐싱), `st.session_state`(결과 유지)만 담당 |
| `analysis.py` | UI 프레임워크에 의존하지 않는 순수 분석 함수 (CSV 정제, embedding, clustering, keyword, 대표 의견, topic map, semantic search) |
| `requirements.txt` | 의존 패키지 목록 |
| `ai_insight_engine_youth_comments.csv` | 샘플 데이터 |

## 실행 방법

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

브라우저에서 `http://localhost:8501`이 자동으로 열립니다.

## 사용 방법

1. **Analyze**: CSV 업로드 → Topic 개수(k) 조절 → `Analyze` 클릭
   - CSV에는 `text` 컬럼이 반드시 있어야 합니다.
2. 분석이 끝나면 Topic Summary 표, Topic Map, CSV 다운로드 버튼이 나타납니다.
3. **Semantic Search**: 검색어와 Top-K를 입력하고 `Search` 클릭 → 의미 기반으로 유사한 의견을 찾아줍니다.

## 참고

- Embedding 모델: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (다국어 지원, 한국어 포함)
- 외부 LLM API 키는 사용하지 않습니다. 모든 분석은 로컬에서 로드한 embedding 모델로만 수행됩니다.
- 첫 실행 시 embedding 모델을 다운로드하므로 시간이 다소 걸릴 수 있습니다. 이후에는 `st.cache_resource`에 의해 캐싱되어 재사용됩니다.
