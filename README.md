# 📊 Ad Click Prediction App

A machine learning web app built using **Python**, **scikit-learn**, and **Streamlit** that predicts whether a user will click on a social media advertisement based on:

- Gender
- Age
- Estimated Salary

---

## 🧠 Features

- Logistic Regression & Decision Tree models
- Input sliders for Age & Salary
- Gender dropdown
- Adjustable prediction threshold
- Real-time prediction with ML model
- Streamlit UI
- Fully deployable on Streamlit Cloud

---

## 📁 Files

| File | Purpose |
|------|---------|
| `app.py` | Streamlit app script |
| `social_ad_click_model.pkl` | Trained ML model |
| `scaler.pkl` | Scaler used during training |
| `requirements.txt` | Python dependencies |

---

## 🚀 How to Run

### 📌 Option 1: Localhost

```bash
pip install -r requirements.txt
streamlit run app.py
