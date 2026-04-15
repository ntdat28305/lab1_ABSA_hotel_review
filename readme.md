# Hotel Review Aspect-Based Sentiment Analysis (ABSA)

Hệ thống phân tích cảm xúc đa khía cạnh (ABSA) dành cho đánh giá khách sạn tại Việt Nam, sử dụng các mô hình Machine Learning và Deep Learning. Dự án bao gồm quy trình từ thu thập dữ liệu (crawling), tiền xử lý, huấn luyện mô hình đến triển khai ứng dụng web.

## 📂 Cấu trúc thư mục (Project Structure)

- `data/`: Chứa các tệp dữ liệu gốc (.json) và dữ liệu đã xử lý (.csv).
- `models/`: Lưu trữ các tệp mô hình đã huấn luyện (.joblib, .pt).
- `source/`: Chứa các Notebook (.ipynb) xử lý cốt lõi.
- `api.py`: Backend API (FastAPI) để phục vụ dự báo.
- `app.py`: Frontend Web Interface (Streamlit).
- `Dockerfile`: Cấu hình Docker để triển khai ứng dụng.
- `requirements.txt`: Danh sách các thư viện cần thiết.

## 🚀 Quy trình thực hiện (Workflow)

Để chạy dự án, vui lòng thực hiện theo đúng trình tự các file trong thư mục `source/`:

1.  **Thu thập dữ liệu:** - Chạy `source/data_crawl.ipynb` để thu thập dữ liệu review từ nền tảng Traveloka thông qua API/Web Scraping.
2.  **Tiền xử lý dữ liệu:** - Chạy `source/process.ipynb` để làm sạch dữ liệu, chuẩn hóa văn bản Tiếng Việt và gán nhãn khía cạnh (aspect labeling).
3.  **Huấn luyện mô hình:** - Chạy `source/train_model.ipynb` để thực hiện feature engineering (TF-IDF/Embeddings) và huấn luyện các mô hình (Baseline LR/RF, LSTM, PhoBERT, BARTpho).

## 🛠 Cài đặt và Triển khai

### 1. Cài đặt thư viện
```bash
pip install -r requirements.txt
```

### 2. Chạy Backend API (FastAPI)
```bash
python api.py
```

### 3. Chạy Frontend UI (Streamlit)
```bash
streamlit run app.py
```

## Triển Khai Cloud
* Backend: Đã được triển khai trên Hugging Face Spaces (Docker).

* Frontend: Đã được triển khai trên Streamlit Cloud.

* Tính năng mở rộng: Hỗ trợ phân tích câu đơn, phân tích file hàng loạt (CSV/Excel) và biểu đồ phân tích xu hướng.