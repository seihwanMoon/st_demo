import os
import streamlit as st
from pandasai import Agent
# from pandasai import SmartDataframe
from pandasai.connectors import PandasConnector
import pandas as pd
from dotenv import load_dotenv
from langchain_groq.chat_models import ChatGroq
import pandasai as pai
pai.clear_cache()

# 한글처리
import matplotlib as plt
plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False


field_descriptions = {
    'Level (레벨)': 'BOM 구성 요소의 계층 구조를 나타내는 숫자입니다. 값이 1 이면 최상위 입니다',
    'Name (이름)': '부품 또는 자재의 이름을 나타냅니다',
    'Revision (버전)': 'BOM 구성 요소의 버전 또는 수정 번호를 표시하는 숫자입니다. 최소값은 1 이고 수정이 되면 숫자값이 1씩 증가함',
    '제목 (Title)': '부품 또는 모듈의 명칭 입니다',
    'Description (설명)': '부품 또는 자재에 대한 자세한 설명을 제공합니다',
    'Type (유형)': 'BOM 구성 요소의 유형을 정의합니다',
    '단계 (Phase)': '프로젝트 또는 제품 개발의 현재 단계를 나타냅니다',
    'Not Last Revision (최종 수정 아님)': '구성 요소가 최종 수정인지 여부를 나타냅니다',
    'State (상태)': 'BOM 구성 요소의 현재 상태가 릴리스 상태인지 나타냅니다',
    'F/N (Find Number) (찾기 번호)': '부품이나 항목을 참조하는 번호입니다',
    'Qty (Quantity) (수량)': '필요한 부품의 수를 표시하는 숫자',
    'Unit Of Measure (측정 단위)': '부품의 수량을 측정하는 단위입니다',
    'Change Required (변경 필요)': '부품에 변경이 필요한지 여부를 나타냅니다',
    'Ref Des (Reference Designator) (참조 설계자)': '회로도에서 구성 요소의 위치를 나타내는 식별자입니다',
    'Part Specification (부품 사양)': '부품의 기술 사양을 포함한 명칭',
    'Maker P/N (Maker Part Number) (제조사 부품 번호)': '제조업체가 부여한 부품 번호입니다',
    'Maker Name (제조사 이름)': '부품을 제조한 회사 또는 공급업체의 이름입니다',
    'Design Collaboration (설계 협업)': '설계 과정에서의 협업 요구 사항이나 상태를 나타냅니다',
    'Collaborative Policy (협업 정책)': '협업에 대한 정책이나 지침을 보여줍니다',
    'Comp. Loc. (Component Location) (부품 위치)': '제품 내에서 부품이 위치한 곳을 나타냅니다',
    'Usage (사용)': '부품이 어떻게 사용되는지를 나타냅니다',
    'Design Revision (설계 수정)': '설계의 수정 번호를 표시합니다',
    'Design Maturity (설계 성숙도)': '구성 요소 설계의 성숙도 수준을 나타냅니다',
}



# load_dotenv()
GROQ_API_KEY=os.getenv('GROQ_API_KEY')

# llm = ChatGroq(model_name='llama3-70b-8192', api_key=GROQ_API_KEY)
# llm = ChatGroq(
#     model_name="llama-3.1-8b-instant", # llama-3.1-8b-instant, gemma2-9b-it
#     # api_key = os.environ["GROQ_API_KEY"]
#     )
st.set_page_config(page_title="SmartEnterAI", page_icon="🐼", layout="wide")
st.title("BOM Data 분석 🙌")

#Function to get LLM
def get_LLM(llm_type):
    #Creating LLM object based on the llm type selected:
    try:
        if llm_type == 'llama3.1_8B':
            llm = ChatGroq(model_name="llama-3.1-8b-instant", api_key=GROQ_API_KEY)

        elif llm_type =='llama3.1_70B':            
            llm = ChatGroq(model_name="llama-3.1-70b-versatile", api_key=GROQ_API_KEY)

        elif llm_type =='gemma2-9b':            
            llm = ChatGroq(model_name="gemma2-9b-it", api_key=GROQ_API_KEY)
        return llm
    except Exception as e:
        #st.error(e)
        st.error("No/Incorrect API key provided! ")




with st.sidebar:
    st.title("설정화면:⚙️")
    #Activating Demo Data
    # st.text("Data Setup: 📝")
    uploaded_file = st.file_uploader("Upload your Data",accept_multiple_files=False,type = ['csv','xls','xlsx'])

    st.markdown(":green[*첫번째 행에는 열이름이 있는지 확인!.*]")

    #selecting LLM to use
    llm_type = st.selectbox(
                        "언어모델(LLM)을 선택 하세요",
                        ('llama3.1_8B','llama3.1_70B','gemma2-9b'),index=0)




if uploaded_file is not None:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.info("등록 완료")
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            data = pd.read_excel(uploaded_file)
        st.dataframe(data, use_container_width=True)
        connector = PandasConnector({"original_df": data}, field_descriptions=field_descriptions)
        df = Agent(connector, config={"llm": get_LLM(llm_type)})

    with col2:
        st.info("채팅 시작하기💬")
        prompt = st.text_area("질문 입력:")

        if st.button("생성"):
            if prompt:
                with st.spinner("결과 생성중, 잠시만..."):
                    result = df.chat(prompt)
                    st.write(result)  # Display the returned result

                    # Check if the result is a valid image path
                    if isinstance(result, str) and os.path.exists(result):
                        if result.lower().endswith(('.png', '.jpg', '.jpeg')):
                            st.image(result, caption="Generated Chart")
                        else:
                            st.warning("The returned result is not an image path.")
                    else:
                        st.warning("차트이미지는 생성되지 않았습니다..")
            else:
                st.warning("Please enter a prompt")

