import streamlit as st
import joblib
import numpy as np
import pandas as pd
import base64
import os

# ============================================================
# 🎨 ตั้งค่าหน้าเว็บ
# ============================================================
st.set_page_config(
    page_title="🫀 Heart Disease Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# 🎨 Custom CSS — ดีไซน์สวยงามแบบ Glassmorphism
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;500;600;700&display=swap');

    /* พื้นหลัง gradient */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        font-family: 'Prompt', sans-serif;
    }

    /* Main container */
    .main-container {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        border-radius: 25px;
        padding: 2.5rem;
        box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        margin: 1rem 0;
    }

    /* Header */
    .main-header {
        text-align: center;
        padding: 1.5rem 0;
        background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
        border-radius: 20px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(231, 76, 60, 0.4);
    }
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        opacity: 0.95;
    }

    /* Heart animation */
    .heart-icon {
        display: inline-block;
        animation: heartbeat 1.5s ease-in-out infinite;
        font-size: 2.5rem;
    }
    @keyframes heartbeat {
        0%, 100% { transform: scale(1); }
        25% { transform: scale(1.15); }
        50% { transform: scale(1); }
        75% { transform: scale(1.1); }
    }

    /* Section card */
    .section-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        border-radius: 18px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 5px solid #e74c3c;
        box-shadow: 0 5px 15px rgba(0,0,0,0.08);
    }
    .section-card h3 {
        color: #c0392b;
        margin-top: 0;
        font-weight: 600;
    }

    /* Result boxes */
    .result-healthy {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(17, 153, 142, 0.4);
        animation: fadeIn 0.8s ease-in;
    }
    .result-risk {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(235, 51, 73, 0.4);
        animation: fadeIn 0.8s ease-in;
    }
    .result-healthy h2, .result-risk h2 {
        color: white !important;
        margin: 0;
        font-size: 2rem;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Probability bar */
    .prob-bar-container {
        background: rgba(255,255,255,0.3);
        border-radius: 50px;
        padding: 5px;
        margin: 1rem 0;
        overflow: hidden;
    }
    .prob-bar {
        height: 25px;
        border-radius: 50px;
        background: linear-gradient(90deg, #38ef7d, #f45c43);
        text-align: center;
        color: white;
        font-weight: 600;
        line-height: 25px;
        transition: width 1s ease;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2c3e50 0%, #4a6278 100%);
        color: white;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3, [data-testid="stSidebar"] label {
        color: white !important;
    }
            
    /* แสดงปุ่มเปิด Sidebar */
    button[kind="header"]{
        display:block !important;
    }

    [data-testid="collapsedControl"]{
        display:flex !important;
        visibility:visible !important;
        opacity:1 !important;
    }

    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
        color: white !important;
        border: none;
        border-radius: 50px;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        width: 100%;
        box-shadow: 0 5px 15px rgba(231, 76, 60, 0.4);
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(231, 76, 60, 0.6);
    }

    /* Metric cards */
    .metric-card {
        background: white;
        border-radius: 15px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 3px 10px rgba(0,0,0,0.1);
        border-top: 4px solid #e74c3c;
    }
    .metric-card h4 {
        margin: 0;
        color: #7f8c8d;
        font-size: 0.85rem;
    }
    .metric-card p {
        margin: 0.3rem 0 0 0;
        font-size: 1.5rem;
        font-weight: 700;
        color: #2c3e50;
    }

    /* Footer */
    .footer {
        text-align: center;
        padding: 1rem;
        color: white;
        font-size: 0.9rem;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# 📦 โหลดโมเดล
# ============================================================
@st.cache_resource
def load_model():
    if os.path.exists('heart_disease_model.pkl'):
        model = joblib.load('heart_disease_model.pkl')
        return model
    else:
        st.error("❌ ไม่พบไฟล์ heart_disease_model.pkl")
        st.stop()

model = load_model()

# Feature mapping
feature_mapping = {
    'Sex': {0: 'หญิง', 1: 'ชาย'},
    'ChestPainType': {1: 'Typical Angina (เจ็บหน้าอกทั่วไป)',
                      2: 'Atypical Angina (เจ็บหน้าอกไม่ปกติ)',
                      3: 'Non-Anginal Pain (เจ็บที่ไม่ใช่โรคหัวใจ)',
                      4: 'Asymptomatic (ไม่มีอาการ)'},
    'RestingECG': {0: 'Normal (ปกติ)',
                   1: 'ST-T wave abnormality',
                   2: 'LV Hypertrophy',
                   3: 'Possible LV Hypertrophy'},
    'ExerciseAngina': {0: 'ไม่เจ็บหน้าอก', 1: 'เจ็บหน้าอก'},
    'ST_Slope': {1: 'Upsloping (ชันขึ้น)',
                 2: 'Flat (ราบ)',
                 3: 'Downsloping (ชันลง)'},
}


# ============================================================
# 🎨 Header
# ============================================================
st.markdown("""
<div class="main-header">
    <div class="heart-icon">❤️</div>
    <h1>ระบบทำนายโรคหัวใจ</h1>
    <p>Heart Disease Prediction System — ด้วย Machine Learning</p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# 📝 Sidebar — ข้อมูลผู้ใช้งาน
# ============================================================
with st.sidebar:
    st.markdown("## 🩺 ข้อมูลผู้ตรวจ")
    st.markdown("---")

    st.markdown("### 👤 ข้อมูลพื้นฐาน")
    age = st.number_input("อายุ (ปี)", min_value=20, max_value=100, value=50, step=1)
    sex = st.selectbox("เพศ", options=[1, 0],
                       format_func=lambda x: feature_mapping['Sex'][x])

    st.markdown("---")
    st.markdown("### 💓 อาการและสัญญาณชีพ")
    chest_pain = st.selectbox(
        "ประเภทอาการเจ็บหน้าอก",
        options=[1, 2, 3, 4],
        format_func=lambda x: feature_mapping['ChestPainType'][x]
    )
    resting_bp = st.number_input(
        "ความดันโลหิตขณะพัก (mm Hg)",
        min_value=80, max_value=220, value=120, step=1
    )
    cholesterol = st.number_input(
        "คอเลสเตอรอล (mg/dl)",
        min_value=100, max_value=600, value=200, step=1
    )
    fasting_bs = st.selectbox(
        "น้ำตาลในเลือดขณะอดอาหาร > 120 mg/dl",
        options=[0, 1],
        format_func=lambda x: "ใช่" if x == 1 else "ไม่ใช่"
    )
    resting_ecg = st.selectbox(
        "ผล ECG ขณะพัก",
        options=[0, 1, 2, 3],
        format_func=lambda x: feature_mapping['RestingECG'][x]
    )
    max_hr = st.number_input(
        "อัตราการเต้นหัวใจสูงสุด (bpm)",
        min_value=60, max_value=220, value=140, step=1
    )
    exercise_angina = st.selectbox(
        "อาการเจ็บหน้าอกขณะออกกำลังกาย",
        options=[0, 1],
        format_func=lambda x: feature_mapping['ExerciseAngina'][x]
    )
    oldpeak = st.number_input(
        "ค่า ST depression (Oldpeak)",
        min_value=0.0, max_value=10.0, value=1.0, step=0.1,
        format="%.1f"
    )
    st_slope = st.selectbox(
        "ความชันของ ST segment",
        options=[1, 2, 3],
        format_func=lambda x: feature_mapping['ST_Slope'][x]
    )

    st.markdown("---")
    st.markdown("""
    <div style='text-align:center; padding:1rem; background:rgba(255,255,255,0.1);
                border-radius:10px;'>
        <small>⚠️ ผลการทำนายเป็นเพียงการประเมินเบื้องต้น<br>
        กรุณาปรึกษาแพทย์ผู้เชี่ยวชาญ</small>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# 🔮 ทำนายผล
# ============================================================
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_button = st.button("🔍 ทำนายผลสุขภาพหัวใจ", use_container_width=True)

if predict_button:
    # สร้าง input array
    input_data = np.array([[age, sex, chest_pain, resting_bp, cholesterol,
                            fasting_bs, resting_ecg, max_hr, exercise_angina,
                            oldpeak, st_slope]])

    # ทำนาย
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    st.markdown("---")

    # แสดงผลลัพธ์
    if prediction == 0:
        st.markdown(f"""
        <div class="result-healthy">
            <div style="font-size:4rem;">✅</div>
            <h2>ผลการทำนาย: ไม่มีความเสี่ยง</h2>
            <p style="font-size:1.2rem; margin-top:0.5rem;">
                หัวใจของคุณอยู่ในเกณฑ์ปกติ ดูแลสุขภาพต่อไปนะครับ! 💚
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-risk">
            <div style="font-size:4rem;">⚠️</div>
            <h2>ผลการทำนาย: มีความเสี่ยง</h2>
            <p style="font-size:1.2rem; margin-top:0.5rem;">
                พบสัญญาณเสี่ยงต่อโรคหัวใจ กรุณาปรึกษาแพทย์โดยเร็ว 🏥
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Probability bar
    risk_pct = probability[1] * 100
    st.markdown("### 📊 ระดับความมั่นใจในการทำนาย")
    st.markdown(f"""
    <div class="prob-bar-container">
        <div class="prob-bar" style="width: {risk_pct:.1f}%;">
            {risk_pct:.1f}%
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Metric cards
    st.markdown("### 📋 สรุปข้อมูลที่คุณกรอก")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <h4>อายุ</h4><p>{age} ปี</p>
        </div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <h4>ความดันโลหิต</h4><p>{resting_bp} mmHg</p>
        </div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="metric-card">
            <h4>อัตราการเต้นหัวใจ</h4><p>{max_hr} bpm</p>
        </div>""", unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class="metric-card">
            <h4>คอเลสเตอรอล</h4><p>{cholesterol} mg/dl</p>
        </div>""", unsafe_allow_html=True)

    # คำแนะนำ
    st.markdown("### 💡 คำแนะนำเพื่อสุขภาพหัวใจ")
    if prediction == 1:
        st.warning("""
        **สิ่งที่คุณควรทำ:**
        - 🏥 **พบแพทย์** เพื่อตรวจร่างกายอย่างละเอียด
        - 🥗 **ปรับอาหาร** ลดไขมันอิ่มตัว เกลือ และน้ำตาล
        - 🚶‍♂️ **ออกกำลังกาย** เบาๆ เป็นประจำ อย่างน้อย 30 นาที/วัน
        - 🚭 **งดสูบบุหรี่** และลดแอลกอฮอล์
        - 😌 **จัดการความเครียด** ด้วยการทำสมาธิหรือโยคะ
        - 💊 **ตรวจสุขภาพ** เป็นประจำทุก 6 เดือน
        """)
    else:
        st.success("""
        **รักษาสภาพนี้ไว้:**
        - 🏃‍♂️ ออกกำลังกายสม่ำเสมอ 150 นาที/สัปดาห์
        - 🥦 ทานผักผลไม้ให้หลากหลาย
        - 😴 นอนหลับให้เพียงพอ 7-8 ชั่วโมง
        - 💧 ดื่มน้ำสะอาดวันละ 8 แก้ว
        - 🧘 ทำกิจกรรมผ่อนคลายความเครียด
        - 📅 ตรวจสุขภาพประจำปี
        """)

# ============================================================
# 📖 Footer
# ============================================================
st.markdown("---")
st.markdown("""
<div class="footer">
    <p>🫀 Heart Disease Prediction System | Powered by Decision Tree & Streamlit</p>
    <p><small>© 2026 — ระบบนี้ใช้สำหรับการศึกษาเท่านั้น ไม่ใช่การวินิจฉัยทางการแพทย์</small></p>
</div>
""", unsafe_allow_html=True)