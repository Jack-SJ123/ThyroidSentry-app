# 🧠 ThyroidSentry — Thyroid Cancer Recurrence Prediction App

A Streamlit-based clinical decision support tool that predicts the **risk of thyroid cancer recurrence** using machine learning models.  
The app allows clinicians, students, and researchers to explore the dataset, visualize clinical patterns, compare multiple ML models, and generate individualized recurrence risk predictions.

🌐 **Live App:** https://thyroidsentry-app.streamlit.app/  
📂 **Dataset:** `Thyroid_Diff.csv`  
🤖 **Pre-trained Models:** Included under `/models/`

## 🚀 Features

### 📊 1. Dataset Overview
- Summary statistics  
- Feature descriptions  
- Dataset preview  

### 🔬 2. Exploratory Data Analysis (EDA)
- Distribution plots  
- Numeric & categorical feature exploration  
- Correlation heatmap  

### 📈 3. Model Performance Comparison
- Trains 3 ML models:
  - Random Forest  
  - XGBoost  
  - Gradient Boosting  
- Evaluates 2 pre-trained PyCaret models  
- Metrics compared:
  - Accuracy, Precision, Recall, F1-Score  
  - ROC-AUC  
- Confusion matrices & ROC curves  
- Feature importance plots  

### 🔮 4. Recurrence Prediction
- User inputs patient features  
- Best-performing model predicts:
  - Recurrence likelihood (0–100%)  
  - Risk label (High / Low)  
- Interactive probability gauge  

## 🏗️ Project Structure

```
ThyroidSentry-app/
│
├── app.py
├── Thyroid_Diff.csv
├── models/
│   ├── best_thyroid_model.pkl
│   └── best_tuned_thyroid_model.pkl
├── requirements.txt
└── README.md
```

## 📦 Installation (Local)

1. Clone the repo:
```bash
git clone https://github.com/Jack-SJ123/ThyroidSentry-app.git
cd ThyroidSentry-app
```

2. Create & activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # on Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the app:
```bash
streamlit run app.py
```

## 📘 Requirements

Key libraries used:

- streamlit  
- pandas  
- numpy  
- scikit-learn  
- xgboost  
- pycaret  
- plotly  
- joblib  

All pinned inside `requirements.txt`.

## 📜 License

This project is intended for **educational and research purposes only**.  
Not intended as a replacement for clinical judgment or medical decision-making.

## 🙌 Acknowledgements

Created as part of **SAIT ARTI-404 — Web Development & Cloud Computing**.  
Dataset and domain concepts inspired by clinical thyroid cancer research.
