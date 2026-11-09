from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="ทำนายคะแนนคณิตศาสตร์", page_icon="📘", layout="centered")
st.title("📘 ระบบทำนายคะแนนปลายภาควิชาคณิตศาสตร์")
st.write("กรอกข้อมูลนักเรียน 4 ปัจจัย เพื่อประมาณคะแนนสุดท้ายด้วย Multiple Linear Regression")

MODEL_PATH = Path(__file__).parent / "math_score_model.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

if not MODEL_PATH.is_file():
    st.error("ไม่พบไฟล์ math_score_model.pkl กรุณาเพิ่มไฟล์ไว้ในโฟลเดอร์เดียวกับ app.py")
    st.stop()

try:
    model = load_model()
except Exception as exc:
    st.error("ไม่สามารถโหลดโมเดลได้ โปรดตรวจสอบว่าไฟล์ .pkl ถูกต้องและเวอร์ชัน scikit-learn เข้ากัน")
    st.exception(exc)
    st.stop()

with st.form("prediction_form"):
    g1 = st.number_input("คะแนนคณิตศาสตร์ช่วงเก็บคะแนนครั้งที่ 1 (0–20)", min_value=0.0, max_value=20.0, value=14.0, step=1.0)
    g2 = st.number_input("คะแนนคณิตศาสตร์ช่วงเก็บคะแนนครั้งที่ 2 (0–20)", min_value=0.0, max_value=20.0, value=15.0, step=1.0)
    study = st.selectbox("เวลาอ่านหนังสือต่อสัปดาห์", options=[1, 2, 3, 4], index=1,
        format_func=lambda x: {1:"น้อยกว่า 2 ชั่วโมง", 2:"2–5 ชั่วโมง", 3:"5–10 ชั่วโมง", 4:"มากกว่า 10 ชั่วโมง"}[x])
    failures = st.selectbox("จำนวนครั้งที่เคยสอบตก", options=[0, 1, 2, 3], index=0,
        format_func=lambda x: "3 ครั้งขึ้นไป" if x == 3 else ("ไม่เคย" if x == 0 else f"{x} ครั้ง"))
    submitted = st.form_submit_button("ทำนายคะแนนปลายภาค", use_container_width=True, type="primary")

if submitted:
    student = pd.DataFrame([{"G1": g1, "G2": g2, "studytime": study, "failures": failures}])
    try:
        raw_prediction = float(model.predict(student)[0])
        prediction = max(0.0, min(20.0, raw_prediction))
        st.success("ทำนายคะแนนสำเร็จ")
        st.metric("คะแนนปลายภาคที่คาดการณ์", f"{prediction:.2f} / 20")
        if prediction != raw_prediction:
            st.caption("ค่าทำนายดิบอยู่นอกช่วง 0–20 จึงจำกัดคะแนนที่แสดงให้ตรงกับช่วงคะแนนของวิชา")
    except Exception as exc:
        st.error("ไม่สามารถทำนายได้ โปรดตรวจสอบไฟล์โมเดลและลำดับตัวแปร")
        st.exception(exc)

st.divider()
st.caption("โมเดลฝึกจากข้อมูล Student Performance (UCI) สำหรับการศึกษาเท่านั้น คะแนนเป็นค่าประมาณ ไม่ใช่คะแนนสอบจริง")
