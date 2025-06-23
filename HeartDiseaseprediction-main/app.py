# app.py

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

# Initialize required tables
cursor.execute('''
    CREATE TABLE IF NOT EXISTS admin (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL UNIQUE,
        password TEXT NOT NULL
    )
''')
cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        age INTEGER,
        gender TEXT,
        notes TEXT
    )
''')
cursor.execute('''
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        input_params TEXT,
        prediction TEXT,
        ai_insight TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
''')
conn.commit()

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

# Admin functions
def register_admin(username, password):
    cursor.execute("SELECT 1 FROM admin WHERE username=?", (username,))
    if cursor.fetchone():
        return False
    cursor.execute("INSERT INTO admin(username,password) VALUES(?,?)", (username, password))
    conn.commit()
    return True

def login_admin(username, password):
    cursor.execute("SELECT 1 FROM admin WHERE username=? AND password=?", (username, password))
    return cursor.fetchone()

# Authentication UI
st.sidebar.subheader("Admin Authentication")
mode = st.sidebar.radio("Mode", ["Login", "Register Admin"])
if not st.session_state.logged_in:
    if mode == "Login":
        user = st.sidebar.text_input("Username")
        pwd = st.sidebar.text_input("Password", type="password")
        if st.sidebar.button("Login"):
            if login_admin(user, pwd):
                st.session_state.logged_in = True
                st.session_state.username = user
                st.success("Login successful")
                st.rerun()
            else:
                st.error("Invalid credentials")
    else:
        new_user = st.sidebar.text_input("New Username")
        new_pwd = st.sidebar.text_input("New Password", type="password")
        if st.sidebar.button("Register"):
            if register_admin(new_user, new_pwd):
                st.success("Admin registered. Please login.")
                st.rerun()
            else:
                st.warning("Username already exists.")
    st.stop()

# Navigation
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate to", ["Register Patient", "Predict", "Metrics"],
                        index=["Register Patient", "Predict", "Metrics"].index(st.session_state.nav_page))
st.session_state.nav_page = page

# Register Patient Page
if page == "Register Patient":
    st.subheader("👥 Register New Patient")
    name = st.text_input("Patient Name")
    age = st.number_input("Age", 1, 120)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    notes = st.text_area("Additional Notes")
    if st.button("Register Patient"):
        if not name.strip():
            st.warning("Name cannot be empty.")
        else:
            cursor.execute("INSERT INTO users (name, age, gender, notes) VALUES (?, ?, ?, ?)", (name, age, gender, notes))
            conn.commit()
            cursor.execute("SELECT last_insert_rowid()")
            st.session_state.selected_patient_id = cursor.fetchone()[0]
            st.session_state.prefill_age = age
            st.session_state.patient_registered = True
            st.session_state.nav_page = "Predict"
            st.rerun()

# Prediction Page
elif page == "Predict":
    if not st.session_state.patient_registered:
        st.info("Please register a patient first.")
    else:
        st.subheader("🔍 Predict Heart Disease Risk")
        cursor.execute("SELECT id, name, age FROM users")
        users = cursor.fetchall()
        user_map = {f"{u[1]} (ID:{u[0]})": (u[0], u[2]) for u in users}
        sel = st.selectbox("Select Patient", list(user_map.keys()))
        user_id, stored_age = user_map[sel]
        age = st.number_input("Age", 1, 120, value=stored_age)

        col1, col2 = st.columns(2)
        with col1:
            sex = st.selectbox("Sex", ["Male", "Female"])
            cp = st.selectbox("Chest Pain Type", ["Typical Angina (0)", "Atypical Angina (1)", "Non-anginal Pain (2)", "Asymptomatic (3)"])
            trestbps = st.slider("Resting Blood Pressure", 80, 200, 120)
            chol = st.slider("Cholesterol", 100, 400, 240)
            fbs = st.selectbox("Fasting BS > 120?", ["No(0)", "Yes(1)"])
        with col2:
            restecg = st.selectbox("Resting ECG", ["Normal(0)", "ST-T Abn(1)", "LVH(2)"])
            thalach = st.slider("Max Heart Rate", 60, 250, 150)
            exang = st.selectbox("Exercise Induced Angina", ["No(0)", "Yes(1)"])
            oldpeak = st.slider("ST Depression", 0.0, 6.0, 1.0)
            slope = st.selectbox("ST Slope", ["Upsloping(0)", "Flat(1)", "Downsloping(2)"])
            ca = st.slider("Major Vessels", 0, 4, 0)
            thal = st.selectbox("Thalassemia", ["Normal(1)", "Fixed(2)", "Reversible(3)"])

        if st.button("Predict"):
            features = np.array([[
                age,
                1 if sex == "Male" else 0,
                int(cp[-2]),
                trestbps,
                chol,
                int(fbs[-2]),
                int(restecg[-2]),
                thalach,
                int(exang[-2]),
                oldpeak,
                int(slope[-2]),
                ca,
                int(thal[-2])
            ]])

            pred = model.predict(features)[0]
            prob = model.predict_proba(features)[0][1]
            result = "Low risk" if pred == 1 else "High risk"
            message = f"Prediction: **{result}** (Confidence: {prob:.0%})"
            st.session_state.patient_registered = True

            cursor.execute("INSERT INTO predictions(user_id, input_params, prediction, ai_insight) VALUES (?, ?, ?, ?)",
                           (user_id, json.dumps(features.tolist()), result, message))
            conn.commit()

            st.success(message if pred == 1 else "")
            st.error(message if pred == 0 else "")

            # PDF Report
            buffer = io.BytesIO()
            p = canvas.Canvas(buffer, pagesize=letter)
            p.drawString(50, 750, f"Patient Name: {sel}")
            p.drawString(50, 730, f"Prediction: {result}")
            p.drawString(50, 710, f"Confidence: {prob:.0%}")
            p.drawString(50, 690, f"Reviewed by Admin: {st.session_state.username}")
            p.showPage()
            p.save()
            buffer.seek(0)

            st.download_button("📄 Download PDF Report", buffer, f"{sel.split()[0]}_report.pdf", "application/pdf")

        if st.button("➕ New Patient"):
            st.session_state.patient_registered = False
            st.rerun()

# Metrics Page
elif page == "Metrics":
    st.subheader("📊 Prediction Metrics")
    df = pd.read_sql_query("SELECT prediction FROM predictions", conn)
    if not df.empty:
        counts = df["prediction"].value_counts().reindex(["Low risk", "High risk"], fill_value=0)
        fig, ax = plt.subplots()
        bars = ax.bar(counts.index, counts.values, color=["green", "red"])
        ax.set_title("Risk Predictions")
        ax.set_ylabel("Patient Count")
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, height/2, f"{int(height)}", ha='center', color='white')
        st.pyplot(fig)
    else:
        st.info("No predictions made yet.")
