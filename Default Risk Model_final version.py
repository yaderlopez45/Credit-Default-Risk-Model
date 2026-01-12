#%% Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Importing specific tools
from joblib import dump, load
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, confusion_matrix, RocCurveDisplay, PrecisionRecallDisplay
from sklearn.metrics import ConfusionMatrixDisplay

#%% Load file

df = pd.read_csv("loan.csv")


#%% CLEANING DATA

# --------------------------------------------------
# Target Variable Creation
# --------------------------------------------------

# Define final loan statuses to keep
final_status = ["Fully Paid", "Charged Off", "Default"]

# Filter dataset to loans with final outcomes only
df = df[df["loan_status"].isin(final_status)].copy()

# Create binary target variable
# 1 = Charged Off or Default
# 0 = Fully Paid
df["default"] = df["loan_status"].isin(["Charged Off", "Default"]).astype(int)



# --------------------------------------------------
# Column Renaming
# --------------------------------------------------

# Rename inquiry column for consistency

renaming_map = {"loan_amnt":"loan_amount",
 "int_rate":"interest_rate", "emp_length":"employment_length", "annual_inc":"annual_income", "open_acc": "open_accounts", "issue_d":"issue_date", "inq_last_6mths":"delinq_last_6mths"}

df = df.rename(columns=renaming_map)


# --------------------------------------------------
# Date Conversions
# --------------------------------------------------

# Convert issue date and earliest credit line to datetime format
df["issue_date"] = pd.to_datetime(df["issue_date"], format="%b-%y")
df["earliest_cr_line"] = pd.to_datetime(df["earliest_cr_line"], format="%b-%y")

# Verify datetime conversion
df[["issue_date", "earliest_cr_line"]].dtypes.head()


# --------------------------------------------------
# Credit Age Feature Engineering
# --------------------------------------------------

# Create credit age in years
df["credit_age"] = (df["issue_date"] - df["earliest_cr_line"]).dt.days / 365

# Preview results
df["credit_age"].head()


# --------------------------------------------------
# Interaction Variables
# --------------------------------------------------

# Create loan burden ratio: compares loan size relative to borrower income
# Higher values indicate greater repayment strain
df["loan_to_income"] = df["loan_amount"] / df["annual_income"]

# Create credit utilization interaction: combines revolving balance and utilization rate
# Captures both how much credit is used and how close the borrower is to their limit
df["utilization_interaction"] = df["revol_bal"] * df["revol_util"]

# Create risk interaction between debt burden and pricing
# Higher values reflect borrowers with both high DTI and high interest rates
df["dti_and_int"] = df["dti"] * df["interest_rate"]

# --------------------------------------------------
# Loan Term Transformation
# --------------------------------------------------

# Convert loan term from string to numeric
df["term"] = df["term"].str.replace("months", "", regex=False)
df["term"] = pd.to_numeric(df["term"], errors="coerce")

# Preview results
df["term"].head()


# --------------------------------------------------
# Delinquency Indicator
# --------------------------------------------------

# Create binary indicator for borrowers who have never been delinquent
# 1 = never delinquent
# 0 = has been delinquent at least once
df["never_delinquent"] = df["mths_since_last_delinq"].isna().astype(int)

# Preview results
df["never_delinquent"].head()

# --------------------------------------------------
# Employment Length Transformation
# --------------------------------------------------

# Convert employment length from string format to numeric years
df["employment_length"] = df["employment_length"].str.replace("10+ years", "10", regex=False)
df["employment_length"] = df["employment_length"].str.replace("< 1 year", "0.5", regex=False)
df["employment_length"] = df["employment_length"].str.replace(" years", "", regex=False)
df["employment_length"] = df["employment_length"].str.replace(" year", "", regex=False)
df["employment_length"] = df["employment_length"].str.strip()

# Convert cleaned values to numeric
df["employment_length"] = pd.to_numeric(df["employment_length"], errors="coerce")

# Preview results
df["employment_length"].head()




# %% Feature Selection
# ============================

# Select the feature columns to be used in the model
features = [
    "loan_amount",
    "term",
    "interest_rate",
    "employment_length",
    "home_ownership",
    "annual_income",
    "verification_status",
    "dti",
    "delinq_2yrs",
    "never_delinquent",
    "credit_age",
    "open_accounts",
    "loan_to_income",
    "utilization_interaction",
    "dti_and_int"
]



 # %% Pipeline Creation
# ============================

# Identify categorical and numerical columns
categorical_columns = [
    "home_ownership",
    "verification_status",
]

numerical_columns = [
    numcol for numcol in features
    if numcol not in categorical_columns
]


# Create preprocessing pipelines

# Numeric pipeline:
# - Impute missing values using the median
# - Scale values for model stability
numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical pipeline:
# - Impute missing values using the most frequent category
# - One-hot encode categorical variables
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])


# %% Column Transformer
# ============================

# Apply the appropriate preprocessing pipeline to each column type
preprocess = ColumnTransformer([
    ("num", numerical_pipeline, numerical_columns),
    ("cat", categorical_pipeline, categorical_columns)
])


# %% Time-based Train/Test Split
# ============================
# Goal: simulate a real-world setting by training on older loans and testing on the most recent year.

# Cutoff = 1 year before the latest issue_date in the dataset
cutoff_date = df.issue_date.max() - pd.DateOffset(years=1)

# Train set: loans issued earlier than the cutoff date
early_data = df[df.issue_date < cutoff_date]

# Test set: loans issued on/after the cutoff date (most recent year)
late_data = df[df.issue_date >= cutoff_date] 

# Build X (features) and y (target) for train and test
X_train = early_data[features]
y_train = early_data["default"]

X_test = late_data[features]
y_test = late_data["default"]





# %% Model: Preprocessing + Logistic Regression
# ============================
# Pipeline ensures the exact same preprocessing is applied during training and testing.

model = Pipeline([
    ("preprocess", preprocess),  # column transformations: encoding, scaling, etc.
    ("classifier", LogisticRegression(
        max_iter=2000,           # higher cap to help convergence
        class_weight="balanced" # handles class imbalance by reweighting classes
    ))
])

# Fit the full pipeline on the training data
model.fit(X_train, y_train)

#%% Test Set Predictions
# ============================
# predictions = hard class labels (0 = non-default, 1 = default)
# predictions_probabilities = predicted probability of default (class = 1)

predictions = model.predict(X_test)
predictions_probabilities = model.predict_proba(X_test)[:, 1]

#%% Performance Metrics
# ============================
# Goal: Evaluate model performance on the test set and produce interpretation-friendly outputs.
# Note:
# - classification_report uses hard class predictions (0/1)
# - ROC AUC uses predicted probabilities (ranking ability, threshold-independent)

# ----------------------------
# TABLES
# ----------------------------

# Classification report (precision, recall, f1-score, support) using hard predictions
print(classification_report(y_test, predictions))

# ROC AUC score using predicted probabilities (class 1 = default)
roc_auc = roc_auc_score(y_test, predictions_probabilities)
print("ROC AUC:", roc_auc)

# ROC curve components (used later for plotting)
fpr, tpr, thresholds = roc_curve(y_test, predictions_probabilities)

# Average Predicted PoD by Actual Class
# ============================
# Calibration sanity check:
# - actual = 0 (no default) should have lower average predicted PoD than actual = 1 (default)

PoD_df = pd.DataFrame({
    "actual": y_test,
    "PoD": predictions_probabilities
})

avg_PoD = PoD_df.groupby("actual")["PoD"].mean()
print(avg_PoD)

# Coefficient Table
# ============================
# Pull the trained logistic regression coefficients and feature names after preprocessing.
# Convert coefficients to odds ratios to make effects easier to interpret.

logreg = model.named_steps["classifier"]
feature_names = model.named_steps["preprocess"].get_feature_names_out()

coef_table = pd.DataFrame({
    "feature": feature_names,
    "coefficient": logreg.coef_[0]
})

# Convert log-odds coefficients to odds ratios:
# - odds_ratio > 1 increases odds of default
# - odds_ratio < 1 decreases odds of default
coef_table["odds_ratio"] = np.exp(coef_table["coefficient"])

# Sort by odds_ratio (largest values correspond to strongest increases in default odds)
coef_table = coef_table.sort_values("odds_ratio", ascending=False)

# Display top 15 features with the largest odds ratios
print(coef_table.head(20))

# PoD Deciles (non-graph computation)
# ============================
# Bin predicted PoD into 10 equal-sized groups (deciles) to examine risk ranking performance.

PoD_df["PoD_decile"] = pd.qcut(PoD_df["PoD"], 10, labels=False)

# For each predicted-risk decile, compute the realized default rate
decile_default_rate = (
    PoD_df
    .groupby("PoD_decile")["actual"]
    .mean()
)

# Feature Importance (non-graph computation)
# ============================
# Identify top positive and top negative coefficient features for plotting later.
# Positive coefficients increase default risk; negative coefficients reduce default risk.

top_coefficients = coef_table.sort_values("coefficient", ascending=False).head(10)
bottom_coefficients = coef_table.sort_values("coefficient").head(10)

# ----------------------------
# GRAPHS
# ----------------------------

# ROC Curve
# ============================
# Visualizes the tradeoff between TPR (recall) and FPR across thresholds.

plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, label=f"Logistic Regression (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – Credit Default Model")
plt.legend()
plt.show()

# Confusion Matrix
# ============================
# Shows counts of TN, FP, FN, TP based on the model’s current class predictions (predictions).

plt.clf()
ConfusionMatrixDisplay.from_predictions(
    y_test,
    predictions,
    cmap="RdYlGn",
    values_format="d"  # show counts as integers
    )
plt.title("Confusion Matrix – Credit Default Model")
plt.tight_layout()
plt.show()

# Distribution of Predicted Probabilities
# ============================
# Shows how spread out the predicted risk scores are (separation vs overlap).

mean_pp = predictions_probabilities.mean()
median_pp = np.median(predictions_probabilities)

plt.clf()
plt.hist(predictions_probabilities, bins=30)
plt.axvline(mean_pp, linestyle="--", linewidth=2, label=f"Mean = {mean_pp:.3f}", color="grey")
plt.axvline(median_pp, linestyle=":", linewidth=2, label=f"Median = {median_pp:.3f}",color="grey")
plt.title("Distribution of predicted probabilities")
plt.xlabel("Predicted Probability of Default")
plt.ylabel("Count")
plt.legend()
plt.show()

# PoD by Actual Outcome (Bar Chart)
# ============================
# Visual check: average predicted PoD should be higher for actual defaults.

plt.clf()
avg_PoD.plot(kind="bar", figsize=(6, 4))
plt.xticks([0, 1], ["No Default", "Default"], rotation=0, color="grey")
plt.ylabel("Average Predicted Probability of Default")
plt.title("Average PoD by Actual Outcome")
plt.show()

# PoD Deciles Plot
# ============================
# Checks ranking: higher predicted-risk deciles should have higher realized default rates.
baseline_default_rate = y_test.mean()

plt.clf()
plt.figure(figsize=(7, 5))
plt.plot(decile_default_rate.index + 1, decile_default_rate.values, marker="o")
plt.xlabel("PoD Decile (Low → High Risk)")
plt.ylabel("Actual Default Rate")
plt.title("Actual Default Rate by PoD Decile")
plt.axhline(
    baseline_default_rate,
    linestyle="--",
    linewidth=2,
    label=f"Baseline Default Rate = {baseline_default_rate:.3f}",
    color="grey"
)
plt.grid(True)
plt.legend()
plt.show()

# Feature Importance Plots (Odds Ratios)
# ============================
# Visualizes features with the largest increase and decrease in default risk (via odds ratios).

plt.clf()
plt.figure(figsize=(7, 5))
plt.barh(top_coefficients["feature"], top_coefficients["odds_ratio"])
plt.xlabel("Odds Ratio")
plt.title("Top Features Increasing Default Risk")
plt.gca().invert_yaxis()
plt.show()

plt.clf()
plt.figure(figsize=(7, 5))
plt.barh(bottom_coefficients["feature"], bottom_coefficients["odds_ratio"])
plt.xlabel("Odds Ratio")
plt.title("Top Features Reducing Default Risk")
plt.gca().invert_yaxis()
plt.show()
# %%
