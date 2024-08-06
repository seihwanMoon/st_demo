import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage

# 페이지 설정
st.set_page_config(page_title="서울시 택배 물류 분석", page_icon="📦", layout="wide")

# 폰트 설정
# plt.rcParams['font.family'] = 'AppleGothic'  # 맥
plt.rcParams['font.family'] = 'Malgun Gothic'  # 윈도우

# --- 데이터 로드 및 전처리 ---
# 1. CSV 데이터 업로드
uploaded_file = st.sidebar.file_uploader("📦 CSV 파일 업로드", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, encoding='cp949')
    df['총 택배량'] = df.iloc[:, 8:].sum(axis=1)

    # 품목 컬럼명 변경
    df.columns = [col.replace('대분류_착지물동량 ', '') for col in df.columns]
    
    # '배송년월일' 컬럼을 datetime 형식으로 변환
    df['배송년월일'] = pd.to_datetime(df['배송년월일'], format='%Y%m%d')
else:
    st.sidebar.warning("분석할 CSV 파일을 업로드 해주세요.")
    st.stop()

st.sidebar.markdown("---")
# --- 사이드바 설정 ---
# 2. 출발지 컬럼 선택 (복수 선택, 초기값: 전체)
st.sidebar.header("🔍 분석항목 설정")
sido_list = ['전체'] + df['송하인_시명'].unique().tolist()
selected_sido = st.sidebar.multiselect("출발지 선택", sido_list, default=['전체'])

# 3. 도착지 컬럼 선택 (복수 선택, 초기값: 전체)
gu_list = ['전체'] + df['수하인_구명'].unique().tolist()
selected_gu = st.sidebar.multiselect("도착지 선택", gu_list, default=['전체'])

# 선택된 지역 데이터 필터링
if '전체' not in selected_sido:
    df = df[df['송하인_시명'].isin(selected_sido)]
if '전체' not in selected_gu:
    df = df[df['수하인_구명'].isin(selected_gu)]

# --- 메인 화면 ---
st.title("📊 서울시 택배 물류 분석")

# 4. 분석 기능 탭
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🗺️ 지역별 택배량", "📦 품목별 분석", 
                                      "🏘️ 구별 택배량 분석", "📅 일자별 택배량 분석", 
                                      "👥 군집 분석"])

# --- 탭1: 지역별 택배량 ---
with tab1:
    st.header("🗺️ 지역별 택배량 분석")
    if len(selected_sido) == 0 or len(selected_gu) == 0:
        st.warning("왼쪽 사이드바에서 출발지와 도착지를 선택해주세요.")
    else:
        # 선택된 지역별 택배량 시각화 (피벗 테이블 사용)
        sido_gu_total = df.pivot_table(values='총 택배량', index='송하인_시명', 
                                        columns='수하인_구명', aggfunc='sum', fill_value=0)
        st.bar_chart(sido_gu_total)
        st.write(sido_gu_total.head(3)) # 상위 3개 값 표시

# --- 탭2: 품목별 분석 ---
with tab2:
    st.header("📦 품목별 분석")
    
    category_total = df.iloc[:, 8:-1].sum().sort_values(ascending=False)
    st.bar_chart(category_total)
    # 품목명 컬럼 이름 변경
    st.write(category_total.to_frame().rename(columns={0: '총 택배량'}).head(3)) 

    selected_category = st.selectbox("품목 선택", category_total.index)

    if '전체' in selected_sido:
        fashion_df = df[['송하인_시명', '수하인_구명', selected_category]]
        fashion_pivot = fashion_df.pivot_table(values=selected_category, index='송하인_시명', 
                                                columns='수하인_구명', aggfunc='sum', fill_value=0)
        st.write(sns.heatmap(fashion_pivot, cmap="YlGnBu", annot=True, fmt=".0f", linewidths=.5))
    else:
        sido_gu_category_total = df.groupby('수하인_구명')[selected_category].sum().sort_values(ascending=False)
        st.bar_chart(sido_gu_category_total)
        # st.write(sido_gu_category_total.to_frame().head(3)) 
        st.write(sido_gu_category_total.head(3)) # 상위 3개 값 표시
# --- 탭3: 구별 택배량 분석 ---
with tab3:
    st.header("🏘️ 구별 택배량 분석")
    if len(selected_sido) == 0 or len(selected_gu) == 0:
        st.warning("왼쪽 사이드바에서 출발지와 도착지를 선택해주세요.")
    else:
        gu_total = df.groupby('수하인_구명')['총 택배량'].sum().sort_values(ascending=False)
        st.bar_chart(gu_total)
        st.write(gu_total.to_frame().head(3))

# --- 탭4: 일자별 택배량 분석 ---
with tab4:
    st.header("📅 일자별 택배량 분석")
    if len(selected_sido) == 0 or len(selected_gu) == 0:
        st.warning("왼쪽 사이드바에서 출발지와 도착지를 선택해주세요.")
    else:
        daily_total = df.groupby('배송년월일')['총 택배량'].count()
        # daily_total 데이터프레임을 리셋하여 '배송년월일'을 컬럼으로 변환
        daily_total = daily_total.reset_index() 
        # x축 눈금을 1일부터 31일까지 표시
        st.line_chart(daily_total, x='배송년월일', y='총 택배량', use_container_width=True)
        st.write(daily_total.head(3))

# --- 탭 5: 군집 분석 ---
with tab5:
    st.header("👥 군집 분석")

    # 5-1. 계층적 군집 분석 (서울시 구별)
    st.subheader("🏘️ 계층적 군집 분석 (서울시 구별)")
    if len(df['수하인_구명'].unique()) >= 2:
        gu_category = df.drop(columns=['배송년월일']).groupby('수하인_구명').sum().iloc[:, 6:].transpose()
        linked = linkage(gu_category.transpose(), 'ward')
        fig, ax = plt.subplots(figsize=(10, 7))
        dendrogram(linked, orientation='top', labels=gu_category.transpose().index,
                   distance_sort='descending', show_leaf_counts=True, ax=ax)
        ax.set_title("서울시 구별 택배 배송 패턴 (Hierarchical Clustering)", fontsize=15)
        ax.set_xlabel("구", fontsize=12)
        ax.set_ylabel("거리", fontsize=12)
        st.pyplot(fig)
    else:
        st.warning("계층적 군집 분석을 위해 2개 이상의 도착지를 선택해주세요.")

    # 5-2. K-means Clustering (지역별)
    st.subheader("🗺️ K-means Clustering (지역별)")
    if len(df['송하인_시명'].unique()) >= 3:
        sido_gu = df.pivot_table(values='총 택배량', index='송하인_시명', columns='수하인_구명', aggfunc='sum', fill_value=0)
        scaler = StandardScaler()
        sido_gu_scaled = scaler.fit_transform(sido_gu)
        n_clusters = st.slider("클러스터 개수 선택", 1, 10, 3)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(sido_gu_scaled)
        cluster_labels = kmeans.labels_
        sido_gu['클러스터'] = cluster_labels
        for i in range(n_clusters):
            st.write(f"--- Cluster {i} ---")
            st.write(sido_gu[sido_gu['클러스터'] == i].index.tolist())
    else:
        st.warning("K-means Clustering을 위해 3개 이상의 출발지를 선택해주세요.")
