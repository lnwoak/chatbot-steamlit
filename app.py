import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# โหลด tokenizer และโมเดล
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("finetuned_wangchanberta_sentiment")
    model = AutoModelForSequenceClassification.from_pretrained("finetuned_wangchanberta_sentiment")
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# แผนที่ ID → label
id2label = {0: "depressed", 1: "neutral"}

# UI
st.title("🧠 วิเคราะห์อารมณ์จากข้อความ (Sentiment Detection)")

st.markdown("ใส่ข้อความหลายบรรทัดเพื่อวิเคราะห์แต่ละข้อความ เช่น:")
st.code("รู้สึกดีมาก\nอยากตาย\nเครียดกับชีวิต", language="text")

user_input = st.text_area("📝 พิมพ์ข้อความ (หลายบรรทัด)", height=200)

if st.button("🔍 วิเคราะห์อารมณ์"):
    # แยกข้อความตามบรรทัด
    texts = [line.strip() for line in user_input.split('\n') if line.strip() != ""]

    if not texts:
        st.warning("❗ กรุณาใส่ข้อความอย่างน้อยหนึ่งบรรทัด")
    else:
        # เตรียม input
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)

        # ทำนาย
        with torch.no_grad():
            outputs = model(**inputs)
            pred_label_ids = torch.argmax(outputs.logits, dim=1).tolist()

        # แสดงผลลัพธ์
        st.subheader("📊 ผลลัพธ์:")
        for text, label_id in zip(texts, pred_label_ids):
            label = id2label.get(label_id, "unknown")
            st.write(f"• '{text}' → **{label}**")
