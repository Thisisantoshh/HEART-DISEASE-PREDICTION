import streamlit as st
import sqlite3
import pickle
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import io
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

# Load the pre-trained model
with open("heart.pkl", "rb") as f:
    model = pickle.load(f)

# Connect to the SQLite database
conn = sqlite3.connect("Login.db", check_same_thread=False)
cursor = conn.cursor()

# Default session state
defaults = {
    "logged_in": False,
    "username": "",
    "patient_registered": False,
    "selected_patient_id": None,
    "prefill_age": None,
    "nav_page": "Register Patient"
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

st.markdown("<h1 style='display: inline; font-size: 2rem; padding-bottom:10px;'>💓 Heart Disease Prediction - Admin Dashboard</h1>", unsafe_allow_html=True)


#Admin Login/registration
def register_admin(username, password):
    cursor.execute("SELECT 1 FROM admin WHERE username=?", (username,))
    if cursor.fetchone(): return False
    cursor.execute("INSERT INTO admin(username,password) VALUES(?,?)", (username, password))
    conn.commit()
    return True


def login_admin(username, password):
    cursor.execute("SELECT 1 FROM admin WHERE username=? AND password=?", (username, password))
    return cursor.fetchone()

#Authentication
st.sidebar.subheader("Admin Authentication")
mode = st.sidebar.radio("Mode", ["Login", "Register Admin"])
if not st.session_state.logged_in:
    if mode == "Login":
        user = st.sidebar.text_input("Username", key="login_user")
        pwd = st.sidebar.text_input("Password", type="password", key="login_pass")
        if st.sidebar.button("Login"):
            if login_admin(user, pwd):
                st.session_state.logged_in = True
                st.session_state.username = user
                st.session_state.nav_page = "Register Patient"
                st.success("Logged in successfully!")
                st.rerun()
            else:
                st.error("Invalid credentials")
    else:
        new_user = st.sidebar.text_input("New Username", key="reg_user")
        new_pwd = st.sidebar.text_input("New Password", type="password", key="reg_pass")
        if st.sidebar.button("Register Admin"):
            if register_admin(new_user, new_pwd):
                st.session_state.logged_in = True
                st.session_state.username = new_user
                st.session_state.nav_page = "Register Patient"
                st.success("Admin registered & logged in.")
                st.rerun()
            else:
                st.warning("Username already exists.")
    st.stop()

#navbar
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate to", ["Register Patient", "Predict", "Metrics"],
                         index=["Register Patient","Predict","Metrics"].index(st.session_state.nav_page))
st.session_state.nav_page = page

if page == "Register Patient":
    st.subheader("👥 Register New Patient")
    name = st.text_input("Patient Name", key="pat_name")
    age_val = st.number_input("Age", 1, 120, key="pat_age")
    gender = st.selectbox("Gender", ["Male","Female","Other"], key="pat_gender")
    notes = st.text_area("Additional Notes", key="pat_notes")
    if st.button("Register Patient"):
        if not name.strip():
            st.warning("Enter a valid name.")
        else:
            cursor.execute("INSERT INTO users(name,age,gender,notes) VALUES(?,?,?,?)",
                           (name, age_val, gender, notes))
            conn.commit()
            cursor.execute("SELECT last_insert_rowid()")
            st.session_state.selected_patient_id = cursor.fetchone()[0]
            st.session_state.prefill_age = age_val
            st.session_state.patient_registered = True
            st.session_state.nav_page = "Predict"
            st.rerun()

elif page == "Predict":
    if not st.session_state.patient_registered:
        st.info("Please register a patient first.")
    else:
        st.subheader("Predict Heart Condition")
        # Fetch patients
        cursor.execute("SELECT id,name,age FROM users")
        users = cursor.fetchall()
        user_map = {f"{u[1]} (ID:{u[0]})": (u[0],u[2]) for u in users}
        sel = st.selectbox("Select Patient", list(user_map.keys()), key="sel_pat")
        user_id, stored_age = user_map[sel]
        age_prefill = st.session_state.prefill_age or stored_age
        st.session_state.prefill_age = None
        
        c1,c2 = st.columns(2)
        with c1:
            age = st.number_input("Age",1,120,age_prefill)
            sex = st.selectbox("Sex",["Male","Female"], key="feat_sex"); sex_val=1 if sex=="Male" else 0
            cp = st.selectbox("Chest Pain Type",["Typical Angina (0)","Atypical Angina (1)","Non-anginal Pain (2)","Asymptomatic (3)"], key="feat_cp"); cp_val=int(cp.split("(")[-1][0])
            trestbps = st.slider("Resting BP (mm Hg)",80,200,120, key="feat_trestbps")
            chol = st.slider("Cholesterol (mg/dl)",100,600,240, key="feat_chol")
        with c2:
            fbs = st.selectbox("Fasting BS>120?",["No(0)","Yes(1)"], key="feat_fbs"); fbs_val=int(fbs.split("(")[-1][0])
            restecg = st.selectbox("Resting ECG",["Normal(0)","ST-T Abn(1)","LVH(2)"], key="feat_restecg"); restecg_val=int(restecg.split("(")[-1][0])
            thalach = st.slider("Max HR Achieved",60,250,150, key="feat_thalach")
            exang = st.selectbox("Exercise Angina",["No(0)","Yes(1)"], key="feat_exang"); exang_val=int(exang.split("(")[-1][0])
        c3,c4 = st.columns(2)
        with c3:
            oldpeak = st.slider("ST Dep.(Oldpeak)",0.0,6.0,1.0,step=0.1, key="feat_oldpeak")
            slope = st.selectbox("ST Slope",["Upsloping(0)","Flat(1)","Downsloping(2)"], key="feat_slope"); slope_val=int(slope.split("(")[-1][0])
        with c4:
            ca = st.slider("Vessels(ca)",0,3,0, key="feat_ca")
            thal = st.selectbox("Thalassemia",["Normal(1)","Fixed(2)","Reversible(3)"], key="feat_thal"); thal_val=int(thal.split("(")[-1][0])
        if st.button("🔍 Predict Heart Risk"):
            feats=np.array([[age,sex_val,cp_val,trestbps,chol,fbs_val,restecg_val,thalach,exang_val,oldpeak,slope_val,ca,thal_val]])
            pred=model.predict(feats)[0]; prob=model.predict_proba(feats)[0][1]
            res="Low risk" if pred==1 else "High risk"
            insight=f"Prediction: **{res}** (Confidence {prob:.0%})"
            # Save prediction
            cursor.execute("INSERT INTO predictions(user_id,input_params,prediction,ai_insight) VALUES(?,?,?,?)",
                           (user_id,json.dumps(feats.tolist()),res,insight))
            if pred==0:
                suggestion = "⚠️ Please consult a cardiologist. Recommend a low-sodium diet, regular light exercise, and stress management."
            else:
                suggestion = "✅ Keep up a balanced diet, regular exercise, and regular checkups. Maintain healthy lifestyle habits."
            conn.commit()
            if pred==1: st.success(insight)
            else: st.error(insight)
            st.info(suggestion)
            # Generate PDF report
            buffer = io.BytesIO()
            p = canvas.Canvas(buffer, pagesize=letter)
            p.setFont("Helvetica-Bold", 16)
            p.drawString(50, 750, "Patient Report")
            p.setFont("Helvetica", 12)
            p.drawString(50, 720, f"Patient Name: {sel.split(' (')[0]}")
            p.drawString(50, 700, f"Inputs:")
            inputs = [
                f"Age: {age}", f"Sex: {sex}", f"Chest Pain Type: {cp}", f"Resting BP: {trestbps}",
                f"Cholesterol: {chol}", f"Fasting BS>120: {fbs}", f"Resting ECG: {restecg}",
                f"Max HR: {thalach}", f"Exercise Angina: {exang}", f"Oldpeak: {oldpeak}",
                f"Slope: {slope}", f"Vessels: {ca}", f"Thalassemia: {thal}"
            ]
            y = 680
            for inp in inputs:
                p.drawString(60, y, inp)
                y -= 18
            p.drawString(50, y, f"Prediction: {res} (Confidence {prob:.0%})")
            y -= 30
            p.drawString(50, y, f"Approved by Admin: {st.session_state.username}")
            p.showPage()
            p.save()
            buffer.seek(0)
            st.download_button(
                label="📄 Download Patient Report (PDF)",
                data=buffer,
                file_name=f"{sel.split(' (')[0]}_report.pdf",
                mime="application/pdf"
            )
        if st.button("➕ New Patient"):
            st.session_state.patient_registered=False
            st.session_state.nav_page="Register Patient"
            st.rerun()

#Metrics page
elif page == "Metrics":
    st.subheader("📊 Overall Metrics")
    dfp = pd.read_sql_query("SELECT prediction FROM predictions", conn)
    if not dfp.empty:
        counts = dfp['prediction'].value_counts().reindex(["Low risk","High risk"],fill_value=0)
        fig, ax = plt.subplots()
        bars = ax.bar(counts.index, counts.values)
        ax.set_xlabel("Risk Category"); ax.set_ylabel("Number of Patients"); ax.set_title("Low vs High Risk Distribution")
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h / 2,
                int(h),
                ha='center', va='center', color='white'
            )
        st.pyplot(fig)
    else:
        st.info("Make predictions to view metrics.")
