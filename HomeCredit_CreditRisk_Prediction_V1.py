# ============================================
# HOME CREDIT DEFAULT RISK - CASE STUDY
# ============================================

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_auc_score, roc_curve,
    accuracy_score, precision_score, recall_score, f1_score
)

# ============================================
# Step 1: Load Dataset
# ============================================

print("Loading Dataset...")
df = pd.read_csv("application_train.csv")

print("\nDataset Shape:", df.shape)
print("\nDataset Info:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

# ============================================
# Step 2: Data Preprocessing
# ============================================

print("\nPreprocessing Data...")

# Drop ID column
if 'SK_ID_CURR' in df.columns:
    df.drop(columns=['SK_ID_CURR'], inplace=True)

# Drop columns with >50% missing values
threshold = len(df) * 0.5
df = df.dropna(thresh=threshold, axis=1)

# Fill missing values
df.fillna(df.median(numeric_only=True), inplace=True)

for col in df.select_dtypes(include='object'):
    df[col].fillna(df[col].mode()[0], inplace=True)

print("Missing values handled.")

# ============================================
# Step 3: Encoding
# ============================================

print("\nEncoding categorical variables...")
df = pd.get_dummies(df)
print("Encoding completed.")

# ============================================
# Step 4: Feature Engineering
# ============================================

print("\nFeature Engineering...")

if 'AMT_CREDIT' in df.columns and 'AMT_INCOME_TOTAL' in df.columns:
    df['DEBT_TO_INCOME'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']

print("Feature engineering done.")

# ============================================
# Step 5: Exploratory Data Analysis
# ============================================

print("\nPerforming EDA...")

# Target distribution
plt.figure()
sns.countplot(x='TARGET', data=df)
plt.title("Target Distribution")
plt.savefig("target_distribution.png")

# Income distribution
if 'AMT_INCOME_TOTAL' in df.columns:
    plt.figure()
    sns.histplot(df['AMT_INCOME_TOTAL'], kde=True)
    plt.title("Income Distribution")
    plt.savefig("income_distribution.png")

# ============================================
# Step 6: Prepare Data
# ============================================

print("\nPreparing data for model...")

X = df.drop('TARGET', axis=1)
y = df['TARGET']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ============================================
# Step 7: Model Training
# ============================================

print("\nTraining Model...")

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("Model training completed.")

# ============================================
# Step 8: Model Evaluation
# ============================================

print("\nEvaluating Model...")

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

# Classification Report
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ROC-AUC
roc_auc = roc_auc_score(y_test, y_prob)
print("\nROC-AUC Score:", roc_auc)

# ============================================
# Step 9: ROC Curve
# ============================================

fpr, tpr, _ = roc_curve(y_test, y_prob)

plt.figure()
plt.plot(fpr, tpr, label="ROC Curve")
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.savefig("roc_curve.png")

# ============================================
# Step 10: Feature Importance
# ============================================

print("\nGenerating Feature Importance...")

importances = model.feature_importances_
indices = np.argsort(importances)[-10:]

plt.figure()
plt.barh(range(len(indices)), importances[indices])
plt.yticks(range(len(indices)), [X.columns[i] for i in indices])
plt.title("Top 10 Important Features")
plt.savefig("feature_importance.png")

# ============================================
# Step 11: Save Results (Excel with 2 Sheets)
# ============================================

print("\nSaving Results to Excel...")

# Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

metrics_df = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
    "Value": [accuracy, precision, recall, f1, roc_auc]
})

# Predictions
pred_df = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

# Save Excel
with pd.ExcelWriter("HomeCredit_CreditRisk_Prediction_Results.xlsx") as writer:
    pred_df.to_excel(writer, sheet_name="Predictions", index=False)
    metrics_df.to_excel(writer, sheet_name="Metrics", index=False)

print("\nExcel file created successfully!")

# ============================================
# DONE
# ============================================

print("\nAll steps completed successfully!")