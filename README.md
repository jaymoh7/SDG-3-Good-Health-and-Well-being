# SDG-3-Good-Health-and-Well-being
# 🧠 Diabetes Prediction using Machine Learning
This project applies a Random Forest classifier to predict diabetes using the PIMA Indians Diabetes Dataset. It supports **SDG 3: Good Health and Well-being** by promoting early diagnosis and preventive healthcare.

---

## 🔍 Problem Statement

Early detection of diabetes is critical for reducing long-term health complications. This project builds a predictive model using health-related features like glucose, BMI, and age.

---

## 🧰 Tools & Technologies

- Python 3
- Jupyter Notebook / VS Code
- Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
- RandomForestClassifier

---

## 📂 Dataset

- **Source**: [Kaggle - PIMA Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- 768 rows, 9 columns (including outcome)

---

## ⚙️ How to Run

1. Clone this repository
2. Install required packages:
   ```bash
   pip install pandas scikit-learn matplotlib seaborn
Run the Python script:

    python diabetes_prediction.py

📊 Results

    Accuracy: 73.38%

    F1-score (Diabetic class): 0.63

    See full metrics and confusion matrix in the report or screenshots below.

🧩 Screenshots

The screenshots of a jupyter output is uploaded

🧠 Ethical Reflection

This dataset focuses on a narrow population (PIMA Indian women). Applying the model globally without retraining may introduce bias. To ensure fairness:

    Use more diverse datasets

    Interpret model predictions cautiously

    Always supplement with professional medical advice

🌍 SDG Alignment

This work aligns with SDG 3 – Good Health and Well-being, by contributing to accessible predictive tools for early healthcare intervention.
