import streamlit as st
import requests
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Travel ABSA System", layout="wide")
st.title("Hệ thống Phân tích Du lịch ")

API_URL = "https://ntdat232-hotel-absa-api.hf.space"

with st.sidebar:
    model_choice = st.selectbox("Chọn Model thông qua API:", ["Logistic Regression", "PhoBERT Transformer"])
    m_type = "logistic" if model_choice == "Logistic Regression" else "phobert"

user_input = st.text_area("Nhập đánh giá:", height=150)

if st.button("Gửi tới Backend API"):
    if user_input:
        with st.spinner("Đang gọi API..."):
            # Gửi yêu cầu POST tới FastAPI
            payload = {"text": user_input, "model_type": m_type}
            response = requests.post(API_URL, json=payload)
            
            if response.status_code == 200:
                data = response.json()
                preds = data["predictions"]
                cats = data["categories"]
                
                # Hiển thị kết quả
                cols = st.columns(3)
                res_list = []
                for i, cat in enumerate(cats):
                    neg, pos = preds[i*2], preds[i*2+1]
                    status = "⚪ None"
                    if pos == 1: 
                        status = "🟢 Tích cực"; res_list.append({"Aspect": cat, "Sentiment": "Positive"})
                    elif neg == 1: 
                        status = "🔴 Tiêu cực"; res_list.append({"Aspect": cat, "Sentiment": "Negative"})
                    
                    with cols[i % 3]:
                        st.metric(cat, status)
                
                if res_list:
                    st.plotly_chart(px.pie(pd.DataFrame(res_list), names='Sentiment', color='Sentiment',
                                           color_discrete_map={'Positive':'#2ecc71', 'Negative':'#e74c3c'}))
            else:
                st.error("Lỗi kết nối API Backend!")