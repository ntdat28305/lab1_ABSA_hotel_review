from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from underthesea import word_tokenize
import numpy as np

app = FastAPI(title="Travel ABSA API")

PHOBERT_REPO = "ntdat232/phobert-hotel-absa" 
LR_PATH = "models/baseline_logistic_regression.joblib"
VEC_PATH = "models/baseline_tfidf_vectorizer.joblib"
CATEGORIES = ['Room_Facilities', 'Service_Staff', 'Location', 'Food_Beverage', 'Price_Value', 'General']

print("Đang tải các mô hình...")
lr_models_dict = joblib.load(LR_PATH) 
vectorizer = joblib.load(VEC_PATH)
p_tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
p_model = AutoModelForSequenceClassification.from_pretrained(PHOBERT_REPO)
print("Hệ thống đã sẵn sàng!")

class ReviewInput(BaseModel):
    text: str
    model_type: str

@app.post("/predict")
async def predict(data: ReviewInput):
    text = data.text
    m_type = data.model_type.lower()
    
    try:
        if "logistic" in m_type:
            X_tfidf = vectorizer.transform([text])
            
            final_preds = []
            for cat in CATEGORIES:
                model_for_cat = lr_models_dict[cat]
                prediction = int(model_for_cat.predict(X_tfidf)[0]) 
                
                neg_flag = 1 if prediction == 1 else 0
                pos_flag = 1 if prediction == 2 else 0
                final_preds.extend([neg_flag, pos_flag])
            preds = final_preds

        else:
            text_seg = word_tokenize(text, format="text")
            inputs = p_tokenizer(text_seg, return_tensors="pt", truncation=True, max_length=128)
            with torch.no_grad():
                logits = p_model(**inputs).logits
                probs = torch.sigmoid(logits).cpu().numpy()[0]
                preds = (probs > 0.5).astype(int).tolist()
        
        return {
            "status": "success",
            "model_used": m_type,
            "predictions": preds,
            "categories": CATEGORIES
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
#  uvicorn api:app --reload