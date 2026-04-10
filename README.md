# 🌸 Personality Prediction from Text — ML Project

Predicts MBTI personality traits (**Introvert/Extrovert** and **Thinking/Feeling**)
from free-form text using **SVM** and **Random Forest** + **TF-IDF** features.

---

## 📁 Project Structure

```
personality_prediction/
├── app.py             ← Main Streamlit application
├── requirements.txt   ← Python dependencies
└── README.md          ← This file
```

> You also need to download and place `mbti_1.csv` in this folder (see step 3 below).

---

## 🚀 Setup & Run

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Download the dataset
1. Go to Kaggle: https://www.kaggle.com/datasets/datasnaek/mbti-type
2. Download `mbti_1.csv`
3. Place it in the **same folder** as `app.py`

### Step 3 — Run the app
```bash
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`

---

## 🧠 How It Works

| Component        | Details |
|------------------|---------|
| Dataset          | MBTI Personality Type Dataset (Kaggle) |
| Preprocessing    | Lowercase → remove URLs → remove special chars |
| Features         | TF-IDF (word + bigrams, top 5,000 features) |
| Task 1           | Binary: **Introvert (I)** vs **Extrovert (E)** |
| Task 2           | Binary: **Thinking (T)** vs **Feeling (F)** |
| Models           | **SVM** (linear kernel, primary) + **Random Forest** (comparison) |
| Confidence       | `predict_proba` from both models |

---

## 🎨 UI Design

- **Color palette:** Baby pink `#FFC0CB` · Light blue `#ADD8E6` · White background  
- **Font:** DM Serif Display (headings) + DM Sans (body)  
- Clean card layout, rounded elements, soft gradients

---

## 💡 Notes

- First run trains the models (~30–60 seconds). Subsequent runs load from cache instantly.
- At least **3–5 sentences** of input text works best for a reliable prediction.
- This is a fun ML experiment, not a clinical personality assessment!
