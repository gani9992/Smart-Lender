import os
import numpy as np
import pandas as pd

def generate_loan_data(num_samples=1000, seed=42):
    np.random.seed(seed)
    
    # Generate Loan_ID
    loan_ids = [f"LP{1000 + i:06d}" for i in range(num_samples)]
    
    # Categorical distributions
    gender_choices = ["Male", "Female"]
    gender_probs = [0.80, 0.20]
    
    married_choices = ["Yes", "No"]
    married_probs = [0.65, 0.35]
    
    dependents_choices = ["0", "1", "2", "3+"]
    dependents_probs = [0.57, 0.17, 0.17, 0.09]
    
    education_choices = ["Graduate", "Not Graduate"]
    education_probs = [0.78, 0.22]
    
    self_employed_choices = ["No", "Yes"]
    self_employed_probs = [0.86, 0.14]
    
    property_area_choices = ["Semiurban", "Urban", "Rural"]
    property_area_probs = [0.38, 0.33, 0.29]
    
    # Arrays to store values
    genders = np.random.choice(gender_choices, size=num_samples, p=gender_probs)
    marrieds = np.random.choice(married_choices, size=num_samples, p=married_probs)
    dependents = np.random.choice(dependents_choices, size=num_samples, p=dependents_probs)
    educations = np.random.choice(education_choices, size=num_samples, p=education_probs)
    self_employeds = np.random.choice(self_employed_choices, size=num_samples, p=self_employed_probs)
    property_areas = np.random.choice(property_area_choices, size=num_samples, p=property_area_probs)
    
    # Numerical distributions
    # Base income using log-normal distribution to look realistic
    applicant_income = np.random.lognormal(mean=8.3, sigma=0.6, size=num_samples)
    applicant_income = np.clip(applicant_income, 1500, 81000).astype(int)
    
    # Coapplicant income: 45% have no coapplicant income, rest have some
    has_coapplicant = np.random.choice([0, 1], size=num_samples, p=[0.45, 0.55])
    coapplicant_income = np.zeros(num_samples)
    for i in range(num_samples):
        if has_coapplicant[i] == 1:
            coapplicant_income[i] = np.random.lognormal(mean=7.4, sigma=0.5)
    coapplicant_income = np.clip(coapplicant_income, 0, 41000).astype(int)
    
    # Loan Amount: Correlated with applicant and co-applicant income
    total_income = applicant_income + coapplicant_income
    loan_amount = (total_income * np.random.uniform(0.015, 0.035, size=num_samples)).astype(int)
    loan_amount = np.clip(loan_amount, 9, 700) # Min 9k, Max 700k
    
    # Loan Amount Term
    term_choices = [12, 36, 60, 84, 120, 180, 240, 300, 360, 480]
    term_probs = [0.01, 0.01, 0.01, 0.01, 0.01, 0.07, 0.01, 0.02, 0.84, 0.01]
    loan_amount_term = np.random.choice(term_choices, size=num_samples, p=term_probs)
    
    # Credit History: 85% have good history
    credit_history = np.random.choice([1.0, 0.0], size=num_samples, p=[0.84, 0.16])
    
    # Predict Loan Status based on logic (Approved = Y, Rejected = N)
    loan_status = []
    for i in range(num_samples):
        # Calculate risk score
        score = 0.0
        
        # Credit history is the strongest predictor
        if credit_history[i] == 1.0:
            score += 0.65
        else:
            score -= 0.30
            
        # Income vs Loan Amount ratio
        income = total_income[i]
        amt = loan_amount[i]
        debt_to_income = (amt * 1000) / (income * (loan_amount_term[i] / 12) if loan_amount_term[i] > 0 else 360)
        
        if debt_to_income < 0.25:
            score += 0.15
        elif debt_to_income > 0.45:
            score -= 0.20
            
        # Graduate status
        if educations[i] == "Graduate":
            score += 0.05
            
        # Property Area
        if property_areas[i] == "Semiurban":
            score += 0.10 # Semiurban has higher historical approval rates
        elif property_areas[i] == "Rural":
            score -= 0.05
            
        # Married applicants
        if marrieds[i] == "Yes":
            score += 0.05
            
        # Add random noise
        score += np.random.normal(0, 0.08)
        
        # Decide status
        if score >= 0.40:
            loan_status.append("Y")
        else:
            loan_status.append("N")
            
    # Combine into a DataFrame
    df = pd.DataFrame({
        "Loan_ID": loan_ids,
        "Gender": genders,
        "Married": marrieds,
        "Dependents": dependents,
        "Education": educations,
        "Self_Employed": self_employeds,
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_amount_term,
        "Credit_History": credit_history,
        "Property_Area": property_areas,
        "Loan_Status": loan_status
    })
    
    # Introduce random missing values (NaNs) to simulate real-world data (2% to 4% missing values)
    mask_cols = ["Gender", "Married", "Dependents", "Self_Employed", "LoanAmount", "Loan_Amount_Term", "Credit_History"]
    for col in mask_cols:
        mask = np.random.rand(num_samples) < 0.03 # 3% missing values
        df.loc[mask, col] = np.nan
        
    # Introduce duplicates (e.g., duplicate some rows)
    dup_indices = np.random.choice(range(num_samples), size=10, replace=False)
    dup_rows = df.iloc[dup_indices].copy()
    # Modify Loan_ID slightly or keep it identical. Let's keep it identical to make it a true duplicate.
    df = pd.concat([df, dup_rows], ignore_index=True)
    
    # Shuffle dataset
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    return df

if __name__ == "__main__":
    print("Generating synthetic loan prediction dataset...")
    df = generate_loan_data()
    
    os.makedirs("dataset", exist_ok=True)
    dataset_path = "dataset/loan_data.csv"
    df.to_csv(dataset_path, index=False)
    print(f"Dataset saved to: {dataset_path}")
    print(f"Shape: {df.shape}")
    print(f"Duplicates: {df.duplicated().sum()}")
    print("Missing values per column:")
    print(df.isnull().sum())
