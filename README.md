# 🛡️ Credit Card Fraud Detection — ML + Flask Web App

> A beginner-friendly Machine Learning project that detects fraudulent credit card transactions using Python, Scikit-learn, and Flask.

---

## 📌 Project Overview

This project uses the [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) to train ML models that classify transactions as **Fraudulent** or **Genuine**. A Flask web app lets users input transaction details and get instant predictions.

---

## 🗂️ Project Structure

```
CreditCardFraudDetection/
│
├── dataset/
│   └── creditcard.csv          ← Download from Kaggle (you add this)
│
├── models/
│   ├── fraud_model.pkl         ← Auto-generated after training
│   ├── scaler.pkl              ← Auto-generated after training
│   ├── feature_names.pkl       ← Auto-generated after training
│   ├── confusion_matrix.png    ← Auto-generated after training
│   └── model_comparison.png    ← Auto-generated after training
│
├── static/
│   └── style.css               ← Website styling
│
├── templates/
│   └── index.html              ← Website HTML
│
├── app.py                      ← Flask web application
├── train_model.py              ← ML model training script
├── requirements.txt            ← Python packages list
├── README.md                   ← This file
└── .gitignore                  ← Files to exclude from GitHub
```

---

## ⚙️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core programming language |
| Pandas | Data loading and manipulation |
| NumPy | Numerical computation |
| Scikit-learn | ML models (LR, DT, RF) |
| Matplotlib & Seaborn | Data visualization |
| Imbalanced-learn (SMOTE) | Handle imbalanced dataset |
| Joblib | Save/load trained model |
| Flask | Web application framework |
| HTML + CSS | Frontend UI |

---

## 🚀 Step-by-Step Setup Guide

### ✅ Step 1 — Install Python

Download Python 3.10+ from: https://www.python.org/downloads/  
During installation, **check "Add Python to PATH"**.

---

### ✅ Step 2 — Open VS Code

1. Download VS Code: https://code.visualstudio.com/
2. Open the project folder: **File → Open Folder → CreditCardFraudDetection**
3. Open terminal: **Terminal → New Terminal**

---

### ✅ Step 3 — Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate it (Windows)
venv\Scripts\activate

# Activate it (Mac/Linux)
source venv/bin/activate
```

You'll see `(venv)` in the terminal — that means it's active ✅

---

### ✅ Step 4 — Install Required Packages

```bash
pip install -r requirements.txt
```

This installs all packages listed in `requirements.txt`. Takes 1–3 minutes.

---

### ✅ Step 5 — Download the Dataset from Kaggle

1. Go to: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. Click **Download** (you need a free Kaggle account)
3. Extract the ZIP file
4. Copy `creditcard.csv` into the `dataset/` folder

Your folder should look like:
```
dataset/
└── creditcard.csv   ← (144 MB file)
```

---

### ✅ Step 6 — Train the Machine Learning Model

```bash
python train_model.py
```

This script will:
- Load and explore the dataset
- Scale features
- Apply SMOTE to balance the data
- Train Logistic Regression, Decision Tree, and Random Forest
- Compare accuracies
- Save the best model to `models/fraud_model.pkl`
- Generate charts in the `models/` folder

⏳ Takes about 2–5 minutes depending on your computer.

---

### ✅ Step 7 — Run the Flask Web App

```bash
python app.py
```

You'll see:
```
🚀 Credit Card Fraud Detection App Running!
👉 http://127.0.0.1:5000
```

Open your browser and go to: **http://127.0.0.1:5000**

---

### ✅ Step 8 — Test the App

1. Enter **Time**: e.g. `3600`
2. Enter **Amount**: e.g. `149.62`
3. Leave **V1–V28** as `0` (for quick testing)
4. Click **Analyze Transaction**
5. See prediction: ✅ Genuine or ⚠️ Fraudulent

---

## 📤 How to Upload to GitHub

```bash
# 1. Initialize git repository
git init

# 2. Add all files
git add .

# 3. Make your first commit
git commit -m "Initial commit: Credit Card Fraud Detection Project"

# 4. Create a new repo on GitHub.com (click New Repository)
# Then connect it:
git remote add origin https://github.com/YOUR_USERNAME/CreditCardFraudDetection.git

# 5. Push code to GitHub
git push -u origin main
```

> ⚠️ Note: The dataset (`creditcard.csv`) is in `.gitignore` because it's 144MB (too large for GitHub). Mention in your README how to download it.

---

## 💼 LinkedIn Project Description

```
🛡️ Credit Card Fraud Detection | ML + Flask Project

Built an end-to-end fraud detection system using Python and machine learning:
• Trained 3 models (Logistic Regression, Decision Tree, Random Forest) on 284K+ transactions
• Applied SMOTE to handle severe class imbalance (fraud is <0.2% of data)
• Achieved 99%+ accuracy with ROC-AUC score evaluation
• Deployed as a Flask web application with a clean dark-theme UI
• Users can enter transaction details and instantly predict fraud vs genuine

Tech: Python | Scikit-learn | Flask | SMOTE | Pandas | Matplotlib | Seaborn

GitHub: [your-link]
```

---

## 📋 Resume Bullet Points

```
• Built Credit Card Fraud Detection ML web app using Python, Flask & Scikit-learn
• Applied SMOTE to balance imbalanced dataset (284K transactions, <0.2% fraud)
• Trained and compared 3 ML models: Logistic Regression, Decision Tree, Random Forest
• Achieved 99%+ classification accuracy; visualized results with Confusion Matrix & ROC-AUC
• Deployed prediction interface as a Flask web application with responsive CSS UI
```

---

## 🎓 Viva Interview Questions & Answers

**Q1: What is the problem being solved?**
> A: We are detecting fraudulent credit card transactions. The dataset is highly imbalanced — most transactions are genuine, so we need special techniques to train a good model.

**Q2: What is SMOTE and why did you use it?**
> A: SMOTE stands for Synthetic Minority Oversampling Technique. Since fraud transactions are very rare (<0.2%), our model would be biased toward predicting "genuine" every time. SMOTE creates synthetic fraud samples to balance the dataset.

**Q3: What is the difference between the 3 models?**
> A: Logistic Regression is a simple linear model that estimates probability. Decision Tree works like a flowchart of yes/no questions. Random Forest combines many decision trees (ensemble method) for better accuracy and generalization.

**Q4: Why is accuracy alone not enough to evaluate fraud detection?**
> A: Because the dataset is imbalanced. A model that predicts "genuine" for everything gets 99.8% accuracy — but misses all fraud! We use ROC-AUC score, Precision, Recall, and F1-score for better evaluation.

**Q5: What is a Confusion Matrix?**
> A: It shows 4 values: True Positives (fraud correctly detected), True Negatives (genuine correctly detected), False Positives (genuine wrongly flagged as fraud), and False Negatives (fraud missed — most dangerous!).

**Q6: What is ROC-AUC score?**
> A: ROC-AUC measures how well the model distinguishes between fraud and genuine across all thresholds. Score closer to 1.0 means better performance.

**Q7: Why did you use StandardScaler?**
> A: The `Amount` and `Time` columns have very different ranges than V1–V28. Scaling puts them on the same scale so the model doesn't bias toward large-value features.

**Q8: What is Joblib used for?**
> A: Joblib saves the trained model to disk as a `.pkl` file. This means we don't need to retrain the model every time the web app runs.

**Q9: How does the Flask web app work?**
> A: Flask listens for HTTP requests. When the user submits the form (POST request), Flask reads the values, passes them to the loaded ML model, gets a prediction, and returns the result to the browser.

**Q10: What are V1–V28 in the dataset?**
> A: They are the result of PCA (Principal Component Analysis) applied to anonymize the original transaction features for privacy. The exact meaning is hidden, but they represent patterns in transaction behavior.

---

## 📊 Model Performance (Expected Results)

| Model | Accuracy | AUC Score |
|-------|----------|-----------|
| Logistic Regression | ~94% | ~0.94 |
| Decision Tree | ~99% | ~0.99 |
| **Random Forest** | **~99.9%** | **~0.999** |

> Random Forest is usually the best performing model for this dataset.

---

## 📬 Contact

Built as part of internship project. Feel free to fork and improve!

---

*Made with ❤️ using Python + Flask + Scikit-learn*
