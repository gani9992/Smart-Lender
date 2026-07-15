import os
import joblib
import numpy as np
import pandas as pd

class LoanPredictor:
    def __init__(self, models_dir="models"):
        self.models_dir = models_dir
        self.model = None
        self.scaler = None
        self.encoders = None
        self.feature_cols = None
        self.is_loaded = False
        self.load_artifacts()

    def load_artifacts(self):
        try:
            model_path = os.path.join(self.models_dir, "best_model.pkl")
            scaler_path = os.path.join(self.models_dir, "scaler.pkl")
            encoders_path = os.path.join(self.models_dir, "encoders.pkl")
            feat_cols_path = os.path.join(self.models_dir, "feature_columns.pkl")
            
            if (os.path.exists(model_path) and os.path.exists(scaler_path) and 
                    os.path.exists(encoders_path) and os.path.exists(feat_cols_path)):
                self.model = joblib.load(model_path)
                self.scaler = joblib.load(scaler_path)
                self.encoders = joblib.load(encoders_path)
                self.feature_cols = joblib.load(feat_cols_path)
                self.is_loaded = True
                print("Predictor successfully loaded all models and preprocess scaling artifacts.")
            else:
                print("Warning: Prediction model artifacts not found. Please train models first.")
                self.is_loaded = False
        except Exception as e:
            print(f"Error loading model artifacts: {str(e)}")
            self.is_loaded = False

    def predict(self, input_data):
        """
        input_data: dictionary containing applicant features:
          Gender, Married, Dependents, Education, Self_Employed,
          ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term,
          Credit_History, Property_Area
        """
        if not self.is_loaded:
            # Fallback heuristic prediction if model not trained yet
            return self._heuristic_fallback(input_data)
            
        try:
            # Prepare data
            data = input_data.copy()
            
            # Fill missing entries with encoders' imputers
            imputers = self.encoders.get("imputers", {})
            for key, val in imputers.items():
                if key not in data or data[key] is None or pd.isna(data[key]):
                    data[key] = val
                    
            # Numerical default values if missing
            num_defaults = {
                "ApplicantIncome": 5000.0,
                "CoapplicantIncome": 0.0,
                "LoanAmount": 140.0,
                "Loan_Amount_Term": 360.0,
                "Credit_History": 1.0
            }
            for key, val in num_defaults.items():
                if key not in data or data[key] is None or pd.isna(data[key]):
                    data[key] = val
            
            # Feature engineering: log transform
            app_inc = float(data["ApplicantIncome"])
            coapp_inc = float(data["CoapplicantIncome"])
            loan_amt = float(data["LoanAmount"])
            term = float(data["Loan_Amount_Term"])
            credit = float(data["Credit_History"])
            
            app_log = np.log1p(app_inc)
            coapp_log = np.log1p(coapp_inc)
            loan_log = np.log1p(loan_amt)
            
            # Encode binary categoricals
            label_encoders = self.encoders.get("label_encoders", {})
            
            # Binary variables encoding
            gender_enc = label_encoders["Gender"].transform([str(data["Gender"])])[0] if "Gender" in label_encoders else (1 if data["Gender"] == "Male" else 0)
            married_enc = label_encoders["Married"].transform([str(data["Married"])])[0] if "Married" in label_encoders else (1 if data["Married"] == "Yes" else 0)
            edu_enc = label_encoders["Education"].transform([str(data["Education"])])[0] if "Education" in label_encoders else (0 if data["Education"] == "Graduate" else 1)
            self_emp_enc = label_encoders["Self_Employed"].transform([str(data["Self_Employed"])])[0] if "Self_Employed" in label_encoders else (1 if data["Self_Employed"] == "Yes" else 0)
            
            # Map Dependents
            dep_val = str(data["Dependents"])
            dep_map = self.encoders.get("dep_map", {'0': 0, '1': 1, '2': 2, '3+': 3})
            dependents_enc = dep_map.get(dep_val, 0)
            
            # Property Area one-hot encoding
            prop_area = str(data["Property_Area"])
            prop_rural = 1 if prop_area == "Rural" else 0
            prop_semi = 1 if prop_area == "Semiurban" else 0
            prop_urban = 1 if prop_area == "Urban" else 0
            
            # Construct feature vector
            feature_vector = pd.DataFrame([{
                'Gender': gender_enc,
                'Married': married_enc,
                'Dependents': dependents_enc,
                'Education': edu_enc,
                'Self_Employed': self_emp_enc,
                'ApplicantIncome_Log': app_log,
                'CoapplicantIncome_Log': coapp_log,
                'LoanAmount_Log': loan_log,
                'Loan_Amount_Term': term,
                'Credit_History': credit,
                'Property_Rural': prop_rural,
                'Property_Semiurban': prop_semi,
                'Property_Urban': prop_urban
            }])
            
            # Apply scaling
            scale_cols = ['ApplicantIncome_Log', 'CoapplicantIncome_Log', 'LoanAmount_Log', 'Loan_Amount_Term']
            feature_vector[scale_cols] = self.scaler.transform(feature_vector[scale_cols])
            
            # Align features with training columns
            feature_vector = feature_vector[self.feature_cols]
            
            # Run model prediction
            pred_class = self.model.predict(feature_vector)[0]
            
            # Get probability
            if hasattr(self.model, "predict_proba"):
                prob_approve = self.model.predict_proba(feature_vector)[0][1]
            else:
                # SVMS without probability = True, or similar
                decision = self.model.decision_function(feature_vector)[0]
                prob_approve = 1 / (1 + np.exp(-decision))  # Sigmoid scaling
                
            prediction = "Y" if pred_class == 1 else "N"
            
            # Generate reasoning and suggestions
            reasoning, risk, suggestions = self._generate_analysis(
                prediction, prob_approve, app_inc, coapp_inc, loan_amt, term, credit,
                data["Education"], data["Self_Employed"], data["Property_Area"]
            )
            
            return {
                "status": "success",
                "prediction": prediction,
                "probability": float(prob_approve),
                "risk_level": risk,
                "reasoning": reasoning,
                "suggestions": suggestions,
                "is_fallback": False
            }
            
        except Exception as e:
            print(f"Error during ML prediction: {str(e)}")
            return self._heuristic_fallback(input_data, error_msg=str(e))

    def _generate_analysis(self, prediction, prob, income, coincome, loan_amt, term, credit, edu, self_emp, prop_area):
        total_income = income + coincome
        dti = (loan_amt * 1000) / (total_income * (term / 12) if term > 0 else 360)
        
        reasons = []
        suggestions = []
        
        # Credit history evaluation
        if credit == 1.0:
            reasons.append("Applicant has a positive credit history, demonstrating low historical default risk.")
        else:
            reasons.append("Applicant lacks a credit history or has a record of delinquent payments, which represents a high risk for standard lending metrics.")
            suggestions.append("Establish a credit history by paying utility bills or using secured credit cards before reapplying.")
            
        # Debt to income evaluation
        if dti < 0.28:
            reasons.append(f"Strong Debt-to-Income (DTI) ratio of {dti:.1%}. Monthly obligations are well within typical financial limits.")
        elif dti > 0.45:
            reasons.append(f"Elevated Debt-to-Income (DTI) ratio of {dti:.1%}, signaling potential budget strain in matching monthly payments.")
            suggestions.append("Apply for a lower loan amount or extend the term period to reduce monthly amortization limits.")
            
        # Education and Employment
        if edu == "Graduate":
            reasons.append("Graduate educational status correlates with stable professional mobility and steady income prospects.")
        else:
            reasons.append("Undergraduate status represents a slightly lower average income potential in the financial scoring index.")
            
        if self_emp == "Yes":
            reasons.append("Self-employed profile introduces higher volatility scores, necessitating stricter credit verification.")
            suggestions.append("Provide verified tax returns from the past 2 years to confirm stable independent income streams.")
            
        # Property Area benefit
        if prop_area == "Semiurban":
            reasons.append("Property located in a Semiurban area, benefiting from higher real estate appreciation and lower local LTV caps.")
            
        # Determine risk and summarize
        if prediction == "Y":
            risk = "Low" if prob > 0.82 else "Medium"
            if not reasons:
                reasons.append("Applicant matches standard underwriting criteria with average risk scores.")
            reasoning_summary = " ".join(reasons[:3])
            suggestions.append("Proceed with submitting official salary slips and properties documentation for final processing.")
        else:
            risk = "High" if prob < 0.35 else "Medium"
            reasoning_summary = " ".join(reasons)
            if not suggestions:
                suggestions.append("Consider applying with a co-applicant who has a steady, verifiable source of income.")
                
        return reasoning_summary, risk, suggestions

    def _heuristic_fallback(self, data, error_msg=None):
        """
        A rule-based predictor to run in case the machine learning model is not yet compiled or failed.
        """
        print("Running heuristic predictor fallback...")
        
        # Parse inputs
        try:
            credit = float(data.get("Credit_History", 1.0))
        except:
            credit = 1.0
            
        try:
            income = float(data.get("ApplicantIncome", 5000.0))
        except:
            income = 5000.0
            
        try:
            coincome = float(data.get("CoapplicantIncome", 0.0))
        except:
            coincome = 0.0
            
        try:
            loan_amt = float(data.get("LoanAmount", 140.0))
        except:
            loan_amt = 140.0
            
        try:
            term = float(data.get("Loan_Amount_Term", 360.0))
        except:
            term = 360.0
            
        edu = data.get("Education", "Graduate")
        
        # Rule system
        score = 0.0
        if credit == 1.0:
            score += 0.60
        else:
            score -= 0.20
            
        total_income = income + coincome
        ratio = (loan_amt * 1000) / total_income if total_income > 0 else 100
        
        if ratio < 30:
            score += 0.20
        elif ratio > 60:
            score -= 0.15
            
        if edu == "Graduate":
            score += 0.05
            
        prob = max(0.05, min(0.95, score + 0.10))
        prediction = "Y" if prob >= 0.50 else "N"
        
        reasoning, risk, suggestions = self._generate_analysis(
            prediction, prob, income, coincome, loan_amt, term, credit,
            edu, data.get("Self_Employed", "No"), data.get("Property_Area", "Urban")
        )
        
        if error_msg:
            reasoning = f"[System Warning: ML Fallback Active - {error_msg}] " + reasoning
            
        return {
            "status": "success",
            "prediction": prediction,
            "probability": float(prob),
            "risk_level": risk,
            "reasoning": reasoning,
            "suggestions": suggestions,
            "is_fallback": True
        }
