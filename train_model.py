# =============================================================
# CREDIT CARD FRAUD DETECTION - MODEL TRAINING SCRIPT
# =============================================================
# This script:
#   1. Loads the dataset
#   2. Preprocesses the data
#   3. Handles imbalanced data using SMOTE
#   4. Trains 3 ML models
#   5. Compares their performance
#   6. Saves the best model as fraud_model.pkl
# =============================================================

# ---- Import all required libraries ----
import pandas as pd                          # For loading and managing data
import numpy as np                           # For numerical operations
import matplotlib.pyplot as plt              # For plotting graphs
import seaborn as sns                        # For beautiful plots
import joblib                                # For saving the trained model
import os                                    # For creating folders

# Sklearn tools for preprocessing and evaluation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

# SMOTE - Synthetic Minority Oversampling Technique
# This creates fake fraud samples to balance the dataset
from imblearn.over_sampling import SMOTE

# -----------------------------------------------
# STEP 1: LOAD THE DATASET
# -----------------------------------------------
print("=" * 60)
print("  CREDIT CARD FRAUD DETECTION - TRAINING STARTED")
print("=" * 60)

print("\n[Step 1] Loading dataset...")

# Check if the dataset file exists
dataset_path = "dataset/creditcard.csv"
if not os.path.exists(dataset_path):
    print("\n❌ ERROR: Dataset not found!")
    print("Please download creditcard.csv from:")
    print("https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
    print("Then place it inside the 'dataset/' folder.")
    exit()

# Load the CSV file into a DataFrame (like an Excel table in Python)
df = pd.read_csv(dataset_path)

print(f"✅ Dataset loaded! Shape: {df.shape}")
print(f"   Total rows: {df.shape[0]}, Total columns: {df.shape[1]}")


# -----------------------------------------------
# STEP 2: EXPLORE THE DATA
# -----------------------------------------------
print("\n[Step 2] Exploring the data...")

# Show first 5 rows
print("\nFirst 5 rows of dataset:")
print(df.head())

# Show basic info
print("\nColumn info:")
print(df.dtypes)

# Check how many fraud vs genuine transactions exist
fraud_count = df['Class'].value_counts()
print(f"\nTransaction counts:")
print(f"   Genuine (0): {fraud_count[0]}")
print(f"   Fraud   (1): {fraud_count[1]}")
print(f"   Fraud percentage: {fraud_count[1]/len(df)*100:.4f}%")
# As you can see, fraud is very rare — this is called "imbalanced data"


# -----------------------------------------------
# STEP 3: CHECK FOR MISSING VALUES
# -----------------------------------------------
print("\n[Step 3] Checking for missing values...")

missing = df.isnull().sum().sum()
if missing == 0:
    print("✅ No missing values found! Data is clean.")
else:
    print(f"⚠️  Found {missing} missing values. Filling with column mean...")
    df.fillna(df.mean(), inplace=True)  # Fill missing values with column average


# -----------------------------------------------
# STEP 4: FEATURE SCALING
# -----------------------------------------------
print("\n[Step 4] Scaling features (Amount and Time)...")

# 'Amount' and 'Time' are not scaled like other columns (V1-V28)
# We scale them so all features are on the same range
scaler = StandardScaler()

# Scale 'Amount' — transaction amount varies a lot, so we normalize it
df['Amount'] = scaler.fit_transform(df[['Amount']])

# Scale 'Time' — seconds elapsed since first transaction
df['Time'] = scaler.fit_transform(df[['Time']])

print("✅ Scaling done!")


# -----------------------------------------------
# STEP 5: SPLIT INTO FEATURES (X) AND LABEL (y)
# -----------------------------------------------
print("\n[Step 5] Splitting features and labels...")

# X = all columns EXCEPT 'Class' (these are our input features)
X = df.drop('Class', axis=1)

# y = only the 'Class' column (0 = Genuine, 1 = Fraud)
y = df['Class']

print(f"   Features shape: {X.shape}")
print(f"   Labels shape: {y.shape}")


# -----------------------------------------------
# STEP 6: HANDLE IMBALANCED DATA USING SMOTE
# -----------------------------------------------
print("\n[Step 6] Applying SMOTE to balance the dataset...")
print("   This creates synthetic fraud samples to match genuine ones...")

# SMOTE creates artificial fraud transactions to balance the dataset
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

print(f"✅ After SMOTE:")
print(f"   Genuine (0): {sum(y_resampled == 0)}")
print(f"   Fraud   (1): {sum(y_resampled == 1)}")


# -----------------------------------------------
# STEP 7: TRAIN-TEST SPLIT
# -----------------------------------------------
print("\n[Step 7] Splitting into Train and Test sets...")

# 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled,
    test_size=0.2,       # 20% for testing
    random_state=42      # Fixed seed = reproducible results
)

print(f"   Training samples: {X_train.shape[0]}")
print(f"   Testing samples:  {X_test.shape[0]}")


# -----------------------------------------------
# STEP 8: TRAIN ALL 3 MODELS
# -----------------------------------------------
print("\n[Step 8] Training models...")

# ---- Model 1: Logistic Regression ----
# Simple model, works like a probability calculator
print("\n  Training Logistic Regression...")
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_acc = accuracy_score(y_test, lr_pred)
lr_auc = roc_auc_score(y_test, lr_pred)
print(f"  ✅ Logistic Regression → Accuracy: {lr_acc:.4f}, AUC: {lr_auc:.4f}")

# ---- Model 2: Decision Tree ----
# Works like a flowchart of yes/no questions
print("\n  Training Decision Tree...")
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)
dt_acc = accuracy_score(y_test, dt_pred)
dt_auc = roc_auc_score(y_test, dt_pred)
print(f"  ✅ Decision Tree → Accuracy: {dt_acc:.4f}, AUC: {dt_auc:.4f}")

# ---- Model 3: Random Forest ----
# Combines many decision trees for better accuracy
print("\n  Training Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)
rf_auc = roc_auc_score(y_test, rf_pred)
print(f"  ✅ Random Forest → Accuracy: {rf_acc:.4f}, AUC: {rf_auc:.4f}")


# -----------------------------------------------
# STEP 9: COMPARE MODELS
# -----------------------------------------------
print("\n[Step 9] Model Comparison:")
print("-" * 50)
print(f"{'Model':<25} {'Accuracy':>10} {'AUC Score':>12}")
print("-" * 50)
print(f"{'Logistic Regression':<25} {lr_acc:>10.4f} {lr_auc:>12.4f}")
print(f"{'Decision Tree':<25} {dt_acc:>10.4f} {dt_auc:>12.4f}")
print(f"{'Random Forest':<25} {rf_acc:>10.4f} {rf_auc:>12.4f}")
print("-" * 50)


# -----------------------------------------------
# STEP 10: PICK THE BEST MODEL
# -----------------------------------------------
print("\n[Step 10] Selecting best model...")

# Store all models and their accuracy in a dictionary
models = {
    "Logistic Regression": (lr_model, lr_acc, lr_pred),
    "Decision Tree":       (dt_model, dt_acc, dt_pred),
    "Random Forest":       (rf_model, rf_acc, rf_pred),
}

# Find the model with highest accuracy
best_name = max(models, key=lambda k: models[k][1])
best_model, best_acc, best_pred = models[best_name]

print(f"🏆 Best Model: {best_name} with Accuracy: {best_acc:.4f}")


# -----------------------------------------------
# STEP 11: SHOW EVALUATION METRICS FOR BEST MODEL
# -----------------------------------------------
print(f"\n[Step 11] Detailed Report for '{best_name}':")
print("\nClassification Report:")
print(classification_report(y_test, best_pred, target_names=["Genuine", "Fraud"]))


# -----------------------------------------------
# STEP 12: PLOT CONFUSION MATRIX
# -----------------------------------------------
print("[Step 12] Saving Confusion Matrix plot...")

# Create models folder if it doesn't exist
os.makedirs("models", exist_ok=True)

# Plot confusion matrix
cm = confusion_matrix(y_test, best_pred)
plt.figure(figsize=(7, 5))
sns.heatmap(
    cm,
    annot=True,           # Show numbers in boxes
    fmt='d',              # Integer format
    cmap='Blues',         # Color theme
    xticklabels=['Genuine', 'Fraud'],
    yticklabels=['Genuine', 'Fraud']
)
plt.title(f'Confusion Matrix - {best_name}', fontsize=14, fontweight='bold')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig("models/confusion_matrix.png")  # Save the image
plt.close()
print("✅ Confusion matrix saved as 'models/confusion_matrix.png'")


# -----------------------------------------------
# STEP 13: PLOT MODEL COMPARISON BAR CHART
# -----------------------------------------------
print("[Step 13] Saving Model Comparison chart...")

model_names = list(models.keys())
accuracies = [models[m][1] for m in model_names]

plt.figure(figsize=(8, 5))
bars = plt.bar(model_names, accuracies, color=['#4C72B0', '#DD8452', '#55A868'])
plt.ylim(0.8, 1.0)          # Zoom in on the Y-axis for better visibility
plt.title("Model Accuracy Comparison", fontsize=14, fontweight='bold')
plt.ylabel("Accuracy")
plt.xlabel("Model")

# Add value labels on top of each bar
for bar, acc in zip(bars, accuracies):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.002,
        f"{acc:.4f}",
        ha='center', va='bottom', fontsize=11
    )

plt.tight_layout()
plt.savefig("models/model_comparison.png")
plt.close()
print("✅ Model comparison saved as 'models/model_comparison.png'")


# -----------------------------------------------
# STEP 14: SAVE THE BEST MODEL
# -----------------------------------------------
print("\n[Step 14] Saving the best model...")

model_path = "models/fraud_model.pkl"
joblib.dump(best_model, model_path)
print(f"✅ Model saved to '{model_path}'")

# Also save the scaler so we can use it in the Flask app
scaler_path = "models/scaler.pkl"
joblib.dump(scaler, scaler_path)
print(f"✅ Scaler saved to '{scaler_path}'")

# Save feature column names so Flask knows the input order
feature_path = "models/feature_names.pkl"
joblib.dump(list(X.columns), feature_path)
print(f"✅ Feature names saved to '{feature_path}'")

print("\n" + "=" * 60)
print("  ✅ TRAINING COMPLETE! All files saved in 'models/' folder.")
print("  Now run: python app.py")
print("=" * 60)
