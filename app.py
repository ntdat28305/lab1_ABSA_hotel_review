import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
import io

st.set_page_config(page_title="Hotel review ABSA", page_icon="👑", layout="wide")

API_URL = "https://ntdat232-hotel-absa-api.hf.space/predict"
CATEGORIES_VN = ['Room_Facilities', 'Service_Staff', 'Location', 'Food_Beverage', 'Price_Value', 'General']

# Khởi tạo Session State
if 'history' not in st.session_state:
    st.session_state.history = []
if 'batch_history' not in st.session_state:
    st.session_state.batch_history = []

tab_home, tab_history = st.tabs(["Trang Chủ", "Lịch Sử"])

# --- TAB 1: TRANG CHỦ ---
with tab_home:
    st.markdown("### Phân tích cảm xúc đa khía cạnh (ABSA)")
    
    col_txt, col_up = st.columns([1.5, 1])
    
    with col_txt:
        user_input = st.text_area("Nhập nội dung đánh giá:", placeholder="Ví dụ: Phòng đẹp nhưng phục vụ chậm...", height=100)
    
    with col_up:
        uploaded_file = st.file_uploader("Hoặc upload file (CSV, XLSX):", type=["csv", "xlsx"])
    
    c_model, c_col = st.columns(2)
    with c_model:
        model_choice = st.selectbox("Chọn mô hình xử lý:", ["PhoBERT Transformer", "Logistic Regression"])
    
    target_column = None
    if uploaded_file:
        with c_col:
            df_preview = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
            target_column = st.selectbox("Chọn cột chứa văn bản:", df_preview.columns)

    # Nút xử lý Backend
    btn_process = st.button("Tiến hành phân tích", use_container_width=True)

    if btn_process:
        m_type = "phobert" if "PhoBERT" in model_choice else "logistic"
        
        # TRƯỜNG HỢP 1: PHÂN TÍCH CÂU ĐƠN
        if user_input and not uploaded_file:
            with st.spinner("Đang tiến hành phân tích..."):
                try:
                    payload = {"text": user_input, "model_type": m_type}
                    resp = requests.post(API_URL, json=payload, timeout=20).json()
                    preds = resp["predictions"]
                    
                    st.success("Kết quả phân tích:")
                    cols = st.columns(6)
                    detected_aspects = []
                    for i, cat in enumerate(CATEGORIES_VN):
                        neg, pos = preds[i*2], preds[i*2+1]
                        label = "🟢 Pos" if pos == 1 else ("🔴 Neg" if neg == 1 else "⚪ None")
                        cols[i].metric(cat, label)

                        if pos == 1: detected_aspects.append(f"{cat}: Positive")
                        elif neg == 1: detected_aspects.append(f"{cat}: Negative")

                    st.session_state.history.insert(0, {
                        "Thời gian": datetime.now().strftime("%H:%M:%S"),
                        "Văn bản": user_input,
                        "Kết quả chi tiết": ", ".join(detected_aspects) if detected_aspects else "Không phát hiện khía cạnh nào"
                    })
                except Exception as e:
                    st.error(f"Lỗi API: {e}")

# TRƯỜNG HỢP 2: PHÂN TÍCH FILE
        elif uploaded_file and target_column:
            with st.spinner("Đang xử lý file..."):
                all_results = []
                detailed_rows = []
                prog = st.progress(0)
                
                for i, txt in enumerate(df_preview[target_column]):
                    try:
                        r = requests.post(API_URL, json={"text": str(txt), "model_type": m_type}, timeout=10).json()["predictions"]
                        all_results.append(r)
                        
                        sentiments = []
                        for idx, cat in enumerate(CATEGORIES_VN):
                            neg, pos = r[idx*2], r[idx*2+1]
                            
                            if pos == 1:
                                sentiments.append(f"{cat}: Tích cực")
                            elif neg == 1:
                                sentiments.append(f"{cat}: Tiêu cực")
                        
                        detailed_rows.append({
                            "Văn bản gốc": txt, 
                            "Kết quả phân tích": " | ".join(sentiments) if sentiments else "Không phát hiện khía cạnh"
                        })
                    except Exception as e:
                        all_results.append([0]*12)
                        detailed_rows.append({"Văn bản gốc": txt, "Kết quả phân tích": "Lỗi xử lý"})
                        
                    prog.progress((i+1)/len(df_preview))
                
                res_np = np.array(all_results)
                stats = []
                for i, cat in enumerate(CATEGORIES_VN):
                    pos_count = np.sum(res_np[:, i*2+1])
                    neg_count = np.sum(res_np[:, i*2])
                    total_mentions = pos_count + neg_count
                    
                    if total_mentions > 0:
                        p_pct = (pos_count / total_mentions) * 100
                        n_pct = (neg_count / total_mentions) * 100
                    else:
                        p_pct, n_pct = 0, 0
                    
                    stats.append({"Khía cạnh": cat, "Tích cực (%)": p_pct, "Tiêu cực (%)": n_pct})
                
                report_df = pd.DataFrame(stats)
                detail_df = pd.DataFrame(detailed_rows)
                
                # --- HIỂN THỊ ---
                st.subheader("Biểu đồ phân tích xu hướng")
                fig = px.bar(report_df, x="Khía cạnh", y=["Tích cực (%)", "Tiêu cực (%)"], 
                             barmode="group",
                             color_discrete_map={"Tích cực (%)": "#2ecc71", "Tiêu cực (%)": "#e74c3c"})
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("Bảng đối soát chi tiết")
                st.dataframe(detail_df.head(10), use_container_width=True)
                if len(detail_df) > 10:
                    st.info(f"💡 Lưu ý: Hệ thống chỉ hiển thị 10/{len(detail_df)} dòng đầu tiên. Vui lòng tải file bên dưới để xem đầy đủ.")
                # --- XUẤT FILE REPORT ---
                # Xuất file chứa chi tiết từng câu để doanh nghiệp kiểm tra
                csv = detail_df.to_csv(index=False).encode('utf-8-sig')
                st.download_button(
                    label="📥 Tải Báo cáo chi tiết (.csv)",
                    data=csv,
                    file_name=f"Report_ABSA_{datetime.now().strftime('%d%m%Y')}.csv",
                    mime="text/csv"
                )

# --- TAB 2: LỊCH SỬ ---
with tab_history:
    st.subheader("Lịch sử phân tích câu đơn")
    if st.session_state.history:
        st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True)
    else:
        st.info("Chưa có lịch sử câu đơn.")