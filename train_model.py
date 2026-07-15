import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for generating figures
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)

# Import models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier

# Create output directories
os.makedirs("models", exist_ok=True)
os.makedirs("static/images/eda", exist_ok=True)

def perform_eda(df_raw):
    print("Performing Exploratory Data Analysis...")
    df = df_raw.copy()
    
    # Fill NAs temporarily for plotting distributions nicely
    for col in df.columns:
        if df[col].isnull().any():
            if df[col].dtype == 'object':
                df[col] = df[col].fillna(df[col].mode()[0])
            else:
                df[col] = df[col].fillna(df[col].median())

    # Set styles
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({'font.size': 10, 'axes.labelsize': 12, 'axes.titlesize': 14})
    
    # Color palette
    colors = ['#1a365d', '#319795', '#2b6cb0', '#48bb78', '#e53e3e']
    sns.set_palette(sns.color_palette(colors))

    # 1. Pie chart: Loan Status Distribution
    plt.figure(figsize=(6, 6))
    status_counts = df['Loan_Status'].value_counts()
    plt.pie(status_counts, labels=['Approved (Y)', 'Rejected (N)'], autopct='%1.1f%%', 
            colors=['#48bb78', '#e53e3e'], startangle=140, explode=(0.05, 0),
            textprops={'fontsize': 12, 'weight': 'bold'})
    plt.title('Loan Status Distribution')
    plt.tight_layout()
    plt.savefig('static/images/eda/loan_status_pie.png', dpi=150)
    plt.close()

    # 2. Countplot: Gender vs Loan Status
    plt.figure(figsize=(7, 5))
    sns.countplot(x='Gender', hue='Loan_Status', data=df)
    plt.title('Loan Status by Gender')
    plt.xlabel('Gender')
    plt.ylabel('Count')
    plt.legend(title='Approved?')
    plt.tight_layout()
    plt.savefig('static/images/eda/gender_analysis.png', dpi=150)
    plt.close()

    # 3. Countplot: Married status vs Loan Status
    plt.figure(figsize=(7, 5))
    sns.countplot(x='Married', hue='Loan_Status', data=df)
    plt.title('Loan Status by Marital Status')
    plt.xlabel('Married')
    plt.ylabel('Count')
    plt.legend(title='Approved?')
    plt.tight_layout()
    plt.savefig('static/images/eda/marital_status_analysis.png', dpi=150)
    plt.close()

    # 4. Countplot: Education vs Loan Status
    plt.figure(figsize=(7, 5))
    sns.countplot(x='Education', hue='Loan_Status', data=df)
    plt.title('Loan Status by Education')
    plt.xlabel('Education')
    plt.ylabel('Count')
    plt.legend(title='Approved?')
    plt.tight_layout()
    plt.savefig('static/images/eda/education_analysis.png', dpi=150)
    plt.close()

    # 5. Countplot: Property Area vs Loan Status
    plt.figure(figsize=(7, 5))
    sns.countplot(x='Property_Area', hue='Loan_Status', data=df)
    plt.title('Loan Status by Property Area')
    plt.xlabel('Property Area')
    plt.ylabel('Count')
    plt.legend(title='Approved?')
    plt.tight_layout()
    plt.savefig('static/images/eda/property_area_analysis.png', dpi=150)
    plt.close()

    # 6. Countplot: Self Employed vs Loan Status
    plt.figure(figsize=(7, 5))
    sns.countplot(x='Self_Employed', hue='Loan_Status', data=df)
    plt.title('Loan Status by Employment Type')
    plt.xlabel('Self Employed')
    plt.ylabel('Count')
    plt.legend(title='Approved?')
    plt.tight_layout()
    plt.savefig('static/images/eda/employment_analysis.png', dpi=150)
    plt.close()

    # 7. Countplot: Credit History vs Loan Status
    plt.figure(figsize=(7, 5))
    sns.countplot(x='Credit_History', hue='Loan_Status', data=df)
    plt.title('Loan Status by Credit History')
    plt.xlabel('Credit History')
    plt.ylabel('Count')
    plt.legend(title='Approved?')
    plt.tight_layout()
    plt.savefig('static/images/eda/credit_history_analysis.png', dpi=150)
    plt.close()

    # 8. Histogram/Distribution: Applicant Income
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    sns.histplot(df['ApplicantIncome'], kde=True, color='#2b6cb0')
    plt.title('Applicant Income Distribution')
    plt.xlabel('Applicant Income')

    plt.subplot(1, 2, 2)
    sns.boxplot(y=df['ApplicantIncome'], color='#319795')
    plt.title('Applicant Income Outliers')
    plt.tight_layout()
    plt.savefig('static/images/eda/applicant_income_dist.png', dpi=150)
    plt.close()

    # 9. Histogram/Distribution: Loan Amount
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    sns.histplot(df['LoanAmount'], kde=True, color='#2b6cb0')
    plt.title('Loan Amount Distribution')
    plt.xlabel('Loan Amount (in Thousands)')

    plt.subplot(1, 2, 2)
    sns.boxplot(y=df['LoanAmount'], color='#319795')
    plt.title('Loan Amount Outliers')
    plt.tight_layout()
    plt.savefig('static/images/eda/loan_amount_dist.png', dpi=150)
    plt.close()

    # 10. Violin Plot: Loan Amount by Education & Loan Status
    plt.figure(figsize=(8, 5))
    sns.violinplot(x='Education', y='LoanAmount', hue='Loan_Status', data=df, split=True, inner="quart")
    plt.title('Loan Amount Distribution by Education and Approval Status')
    plt.tight_layout()
    plt.savefig('static/images/eda/loan_amount_violin.png', dpi=150)
    plt.close()

    # 11. Pair Plot (Numerical features colored by Loan Status)
    # Select small subset to avoid rendering bottlenecks
    pair_df = df[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Status']].copy()
    pair_df = pair_df.dropna()
    g = sns.pairplot(pair_df, hue='Loan_Status', palette={'Y': '#48bb78', 'N': '#e53e3e'}, diag_kind='kde')
    g.fig.suptitle('Pairwise Feature Plot', y=1.02)
    g.savefig('static/images/eda/pair_plot.png', dpi=150)
    plt.close()

    # 12. Correlation Heatmap
    # For correlation, encode objects temporarily
    corr_df = df.copy()
    for col in corr_df.select_dtypes(include='object').columns:
        corr_df[col] = LabelEncoder().fit_transform(corr_df[col].astype(str))
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_df.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Matrix of All Variables')
    plt.tight_layout()
    plt.savefig('static/images/eda/correlation_heatmap.png', dpi=150)
    plt.close()
    
    print("EDA Visualizations saved successfully!")

def main():
    # 1. Load Data
    data_path = "dataset/loan_data.csv"
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Run data_generator.py first.")
        return
        
    df = pd.read_csv(data_path)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # 2. Duplicate Removal
    duplicates = df.duplicated(subset=['Loan_ID'])
    num_dups = duplicates.sum()
    if num_dups > 0:
        print(f"Removing {num_dups} duplicate records based on Loan_ID...")
        df = df.drop_duplicates(subset=['Loan_ID'], keep='first').reset_index(drop=True)
    
    # 3. Perform EDA on cleaned raw data
    perform_eda(df)
    
    # Drop Loan_ID as it's not a model feature
    df = df.drop(columns=['Loan_ID'])
    
    # 4. Missing Value Imputation (Preprocess from scratch)
    print("Handling missing values...")
    categorical_cols = ['Gender', 'Married', 'Dependents', 'Self_Employed', 'Credit_History']
    numerical_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
    
    imputers = {}
    
    # Categorical -> Mode
    for col in categorical_cols:
        mode_val = df[col].mode()[0]
        imputers[col] = mode_val
        df[col] = df[col].fillna(mode_val)
        
    # Numerical -> Median
    for col in numerical_cols:
        median_val = df[col].median()
        imputers[col] = median_val
        df[col] = df[col].fillna(median_val)
        
    # 5. Outlier Detection and log transformation (Feature Engineering)
    # Log transformation reduces skewness of income and loan amount
    print("Applying transformations to handle outliers...")
    df['ApplicantIncome_Log'] = np.log1p(df['ApplicantIncome'])
    df['CoapplicantIncome_Log'] = np.log1p(df['CoapplicantIncome'])
    df['LoanAmount_Log'] = np.log1p(df['LoanAmount'])
    
    # 6. Encoding Categorical Variables
    print("Encoding categorical features...")
    label_encoders = {}
    binary_cols = ['Gender', 'Married', 'Education', 'Self_Employed']
    
    # Label encode binary variables
    for col in binary_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
        
    # One-Hot Encode multi-class variables: Property_Area & Dependents
    # Map Dependents manually to numerical class representation
    dep_map = {'0': 0, '1': 1, '2': 2, '3+': 3}
    df['Dependents'] = df['Dependents'].map(dep_map)
    # Fill any map errors if any just in case
    df['Dependents'] = df['Dependents'].fillna(0).astype(int)
    
    # One-hot encode Property_Area
    df = pd.get_dummies(df, columns=['Property_Area'], prefix='Property', drop_first=False)
    
    # Map Loan_Status to binary target
    status_map = {'Y': 1, 'N': 0}
    df['Loan_Status'] = df['Loan_Status'].map(status_map)
    
    # Prepare features and target
    # Select columns to train on
    feature_cols = [
        'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
        'ApplicantIncome_Log', 'CoapplicantIncome_Log', 'LoanAmount_Log', 'Loan_Amount_Term',
        'Credit_History', 'Property_Rural', 'Property_Semiurban', 'Property_Urban'
    ]
    
    X = df[feature_cols]
    y = df['Loan_Status']
    
    # 7. Scale Features
    print("Scaling features...")
    scaler = StandardScaler()
    
    # Scale all numerical/continuous columns
    scale_cols = ['ApplicantIncome_Log', 'CoapplicantIncome_Log', 'LoanAmount_Log', 'Loan_Amount_Term']
    X_scaled = X.copy()
    X_scaled[scale_cols] = scaler.fit_transform(X[scale_cols])
    
    # 8. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.20, random_state=42, stratify=y)
    print(f"Training shape: {X_train.shape}, Testing shape: {X_test.shape}")
    
    # 9. Train Multiple Models
    models = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=5),
        "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100, max_depth=6),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB(),
        "Support Vector Machine": SVC(random_state=42, probability=True, C=1.0),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42, n_estimators=100, learning_rate=0.05, max_depth=4),
        "XGBoost": XGBClassifier(random_state=42, n_estimators=100, learning_rate=0.05, max_depth=4, eval_metric='logloss')
    }
    
    results = {}
    comparison_data = []
    
    print("\nTraining and evaluating models...")
    for name, model in models.items():
        # Fit model
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_test)
        
        # Calculate Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        # CV Score
        cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')
        cv_mean = cv_scores.mean()
        
        # ROC AUC
        roc_auc = roc_auc_score(y_test, y_prob)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred).tolist()
        
        # Store results
        results[name] = {
            "model_object": model,
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1),
            "cv_accuracy": float(cv_mean),
            "roc_auc": float(roc_auc),
            "confusion_matrix": cm,
            "classification_report": classification_report(y_test, y_pred, output_dict=True)
        }
        
        comparison_data.append({
            "Model": name,
            "Accuracy": float(acc),
            "Precision": float(prec),
            "Recall": float(rec),
            "F1-Score": float(f1),
            "CV Accuracy": float(cv_mean),
            "ROC-AUC": float(roc_auc)
        })
        
        print(f"| {name:<25} | Acc: {acc:.4f} | F1: {f1:.4f} | CV: {cv_mean:.4f} | AUC: {roc_auc:.4f} |")
        
    # Generate ROC Curve Plot and Confusion Matrix Plot for Best Model
    # Compare based on Testing F1 Score (better balance for loan eligibility)
    comparison_df = pd.DataFrame(comparison_data)
    best_model_name = comparison_df.sort_values(by="F1-Score", ascending=False).iloc[0]["Model"]
    print(f"\nBest Model selected: {best_model_name}")
    
    # Save Model Visualizations
    plt.figure(figsize=(10, 8))
    for name, model in models.items():
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_test)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {results[name]['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('static/images/eda/roc_curves_comparison.png', dpi=150)
    plt.close()
    
    # Plot best model confusion matrix
    plt.figure(figsize=(6, 5))
    best_cm = np.array(results[best_model_name]["confusion_matrix"])
    sns.heatmap(best_cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Rejected', 'Approved'], 
                yticklabels=['Rejected', 'Approved'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {best_model_name}')
    plt.tight_layout()
    plt.savefig('static/images/eda/best_model_confusion_matrix.png', dpi=150)
    plt.close()

    # Feature Importance for Best Model (if tree-based/linear)
    best_model_obj = results[best_model_name]["model_object"]
    importances = None
    if hasattr(best_model_obj, "feature_importances_"):
        importances = best_model_obj.feature_importances_
    elif hasattr(best_model_obj, "coef_"):
        importances = np.abs(best_model_obj.coef_[0])
        
    if importances is not None:
        feat_imp_df = pd.DataFrame({
            "Feature": feature_cols,
            "Importance": importances
        }).sort_values(by="Importance", ascending=True)
        
        plt.figure(figsize=(8, 6))
        plt.barh(feat_imp_df["Feature"], feat_imp_df["Importance"], color='#2b6cb0')
        plt.title(f"Feature Importance ({best_model_name})")
        plt.xlabel("Importance Score")
        plt.tight_layout()
        plt.savefig("static/images/eda/feature_importance.png", dpi=150)
        plt.close()
    else:
        # Generate dummy feature importance graph for KNN or SVM
        # Using Random Forest feature importance as system representative
        rf_model = models["Random Forest"]
        feat_imp_df = pd.DataFrame({
            "Feature": feature_cols,
            "Importance": rf_model.feature_importances_
        }).sort_values(by="Importance", ascending=True)
        
        plt.figure(figsize=(8, 6))
        plt.barh(feat_imp_df["Feature"], feat_imp_df["Importance"], color='#2b6cb0')
        plt.title("System Feature Importance (Random Forest Reference)")
        plt.xlabel("Importance Score")
        plt.tight_layout()
        plt.savefig("static/images/eda/feature_importance.png", dpi=150)
        plt.close()

    # Save comparison data to JSON
    # Remove numpy datatypes from reports for JSON serialization
    json_results = []
    for item in comparison_data:
        json_results.append({
            "Model": item["Model"],
            "Accuracy": item["Accuracy"],
            "Precision": item["Precision"],
            "Recall": item["Recall"],
            "F1-Score": item["F1-Score"],
            "CV Accuracy": item["CV Accuracy"],
            "ROC-AUC": item["ROC-AUC"]
        })
        
    with open("static/model_comparison.json", "w") as f:
        json.dump({
            "comparison": json_results,
            "best_model": best_model_name
        }, f, indent=4)
        
    # 10. Save the selected best model pipeline and other elements
    joblib.dump(best_model_obj, "models/best_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump({
        "imputers": imputers,
        "label_encoders": label_encoders,
        "binary_cols": binary_cols,
        "dep_map": dep_map
    }, "models/encoders.pkl")
    joblib.dump(feature_cols, "models/feature_columns.pkl")
    
    print("\nModels and preprocessing parameters saved to 'models/' successfully!")
    print("Done!")

if __name__ == "__main__":
    main()
