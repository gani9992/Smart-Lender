# SMART LENDER - PROJECT TECHNICAL REPORT

**Project Name**: SMART LENDER – Loan Eligibility Prediction System  
**Framework**: Machine Learning, Flask Web Server, Bootstrap 5 UI Design  
**Classification Types**: Multi-Model Binary Classification  

---

## 1. ABSTRACT
In the modern banking industry, credit risk assessment is a critical operation. Traditional credit checks are often slow and manual, requiring extensive review of financial parameters. This project presents **SmartLender**, an intelligent web application that predicts customer loan eligibility in real-time. By utilizing applicant demographic and financial parameters, the system executes predictions across eight supervised classification algorithms (Logistic Regression, Decision Tree, Random Forest, K-Nearest Neighbors, Naive Bayes, Support Vector Machines, Gradient Boosting, and XGBoost). The optimal performing model is automatically selected, packaged, and serialized for real-time inference within a Flask backend. The UI is custom-designed using premium glassmorphic, mobile-friendly styles with full dark mode support, offering structured explaining logs and printable pre-approval PDF reports.

---

## 2. PROJECT OBJECTIVES
The key goals of the SmartLender system are:
1. **Automated Risk Profiling**: Replace slow credit reviews with instant predictions.
2. **Multi-Model Ensemble Comparison**: Implement, evaluate, and compare 8 ML algorithms using precision, recall, F1, and cross-validation accuracy.
3. **Explainable AI Integration**: Provide natural language reasoning explaining decision indicators (e.g. debt-to-income limits, credit score delinquencies).
4. **CSV Bulk Processing**: Enable banking officers to upload CSV tables containing applicant profiles and perform batch predictions.
5. **IBM SkillsBuild Readiness**: Deliver an architecture meeting security validation standards, and ready for cloud deployment.

---

## 3. SYSTEM METHODOLOGY
### 3.1 Preprocessing Pipeline (from scratch)
Real-world datasets contain anomalies, missing values, duplicates, and skewed values. SmartLender implements a custom preprocessing pipeline:
- **Duplicate Removal**: Scans and removes duplicate rows based on unique IDs.
- **Median/Mode Imputation**: Missing continuous fields are imputed with training medians to prevent outlier skew. Missing categorical fields are imputed with mode values.
- **Logarithmic Normalization**: High-skew continuous columns (`ApplicantIncome`, `LoanAmount`) undergo $log(x+1)$ transformation to reduce skewness and stabilize variance.
- **Categorical Mappings**: Multi-class variables (`Property_Area`) are converted to separate one-hot features. Binary categories are transformed via `LabelEncoder`.
- **Z-Score Scaling**: Numerical variables are scaled using `StandardScaler` to bring all values onto a matching standard deviation space ($mean = 0, std = 1$), improving convergence in algorithms like SVM and KNN.

### 3.2 Model Selection and Evaluation
The system evaluates 8 classifiers:
1. **Logistic Regression**: Serves as a baseline statistical classifier.
2. **Decision Tree**: Provides hierarchical splits.
3. **Random Forest**: Aggregates multiple decision trees to minimize overfitting.
4. **K-Nearest Neighbors (KNN)**: Measures distance-based neighborhood votes.
5. **Naive Bayes**: Uses probabilistic assumptions.
6. **Support Vector Machine (SVM)**: Projects features into a higher-dimensional hyperplane to maximize class separation.
7. **Gradient Boosting Classifier**: Trains sequential models to minimize residual errors.
8. **XGBoost Classifier**: An optimized, high-performance gradient boosting library.

The optimal model is selected based on **F1-Score** (harmonic mean of precision and recall) because loan approvals represent an imbalanced setting where false positives and false negatives carry significant business costs.

---

## 4. COMPARATIVE ANALYSIS FINDINGS
During pipeline training, tree-based models and SVM tend to achieve the highest performance.
- **Credit History Score**: Represents the single most crucial indicator. Applicants without a credit score have a low approval probability (~10%).
- **Debt-to-Income (DTI)**: A ratio of monthly obligation to pooled income. Ratios above 45% trigger significant risk flags.
- **Semi-Urban Premium**: Properties located in semi-urban areas correlate with slightly higher historical approval rates.

---

## 5. DEVELOPER API DOCUMENTATION
SmartLender features a REST endpoint for external integration.

### Request Payload
`POST /api/predict`
```json
{
  "Gender": "Male",
  "Married": "Yes",
  "Dependents": "1",
  "Education": "Graduate",
  "Self_Employed": "No",
  "ApplicantIncome": 4500,
  "CoapplicantIncome": 1200,
  "LoanAmount": 130,
  "Loan_Amount_Term": 360,
  "Credit_History": 1.0,
  "Property_Area": "Semiurban"
}
```

### JSON Response
`200 OK`
```json
{
  "status": "success",
  "prediction": "Y",
  "probability": 0.884,
  "risk_level": "Low",
  "reasoning": "Applicant has a positive credit history, demonstrating low historical default risk. Strong Debt-to-Income (DTI) ratio.",
  "suggestions": ["Submit official income slips for final verification."],
  "prediction_id": 4,
  "is_fallback": false
}
```

---

## 6. CLOUD DEPLOYMENT CONFIGURATIONS

### Procfile (for PaaS platforms like Heroku/Render)
```text
web: gunicorn app:app
```

### runtime.txt
```text
python-3.10.11
```

### manifest.yml (for IBM Cloud Foundry)
```yaml
applications:
  - name: smartlender-app
    memory: 256M
    instances: 1
    buildpack: python_buildpack
```
To deploy on IBM Cloud:
1. Build dependencies and verify locally: `python app.py`.
2. Install IBM Cloud CLI and target region.
3. Push using: `ibmcloud cf push`.
