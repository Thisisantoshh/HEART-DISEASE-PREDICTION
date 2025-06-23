# 💓 Heart Disease Prediction – Admin Dashboard
A web application to predict heart disease risk, register patients, store results, and download PDF reports.

---

## 🚀 Features
- Admin authentication with secure password hashing
- Register new patients (Name, Age, Gender, Notes)
- Predict heart disease risk using a pre-trained model
- Collects 13 health features (age, chest pain, BP, cholesterol, etc.)
- Classifies as "Low risk" or "High risk", with confidence scores
- PDF report download summarizing input & prediction
- Metrics dashboard visualizing aggregated risk distribution
- Logging for operations and error tracking

---

## 🧰 Tech Stack
- Streamlit – Interactive web interface
- SQLite – Lightweight database (admin, users, predictions tables)
- Python libraries:
     - pickle, scikit-learn – ML model handling
     - hashlib – SHA‑256 password hashing
     - reportlab – PDF generation
     - pandas, numpy, matplotlib – Data processing & plotting
     - logging – Error/operation logs

---

## ⚙️ Setup & Installation
- Clone Repo
```bash
git clone <repo-url>
cd HeartDiseasePrediction
```
- Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate
venv\Scripts\activate #on Windows
```
- Install dependencies
```bash
pip install -r requirements.txt
```
- Run the app
```bash
streamlit run app.py
```
# 🔒 Security Notes
- Passwords are stored securely using SHA‑256 hashing.
- SQL queries use parameterized execution, preventing injection.
- Input validation ensures safe and correct data:
      - Age must be 1–100
      - Feature vector shape enforced before prediction
---
# 🧪 Usage Instructions
- Register or log in as an admin in the sidebar.
- Add a patient with their personal data.
- Predict their heart disease risk and download a PDF summary.
- View aggregate metrics (Low vs High risk distribution).
---
📈 Future Improvements
- Migrate to Postgres or cloud DB to ensure persistence on platforms like Render.
- Add email capability to send reports.
- Expand UI with detailed historical data per patient.
