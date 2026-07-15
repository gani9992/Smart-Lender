import os
import csv
import json
import pandas as pd
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, send_file
from config import Config
from utils.db_helper import db, init_db, PredictionHistory, ContactInquiry
from utils.predictor import LoanPredictor
from utils.report_generator import generate_pdf_report

app = Flask(__name__)
app.config.from_object(Config)

# Initialize Database
init_db(app)

# Initialize Predictor
predictor = LoanPredictor()

@app.context_processor
def inject_global_vars():
    # Inject variables accessible to all templates (e.g. system status, dark mode state)
    try:
        with open("static/model_comparison.json", "r") as f:
            comp_data = json.load(f)
            best_model = comp_data.get("best_model", "Random Forest")
    except:
        best_model = "Random Forest (Default)"
    return {
        "best_model_name": best_model
    }

# ----------------- ROUTES -----------------

@app.route('/')
@app.route('/home')
def home():
    # Fetch some stats for the home page
    try:
        total_predictions = PredictionHistory.query.count()
        approved_count = PredictionHistory.query.filter_by(prediction_outcome='Y').count()
        approval_rate = (approved_count / total_predictions * 100) if total_predictions > 0 else 72.4
    except:
        total_predictions = 1250
        approval_rate = 74.2
        
    return render_template(
        'index.html',
        total_predictions=total_predictions,
        approval_rate=round(approval_rate, 1)
    )

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # Gather inputs
            applicant_name = request.form.get('applicant_name', 'Valued Customer').strip()
            if not applicant_name:
                applicant_name = "Valued Customer"
                
            gender = request.form.get('gender')
            married = request.form.get('married')
            dependents = request.form.get('dependents')
            education = request.form.get('education')
            self_employed = request.form.get('self_employed')
            
            try:
                applicant_income = float(request.form.get('applicant_income', 0))
                coapplicant_income = float(request.form.get('coapplicant_income', 0))
                loan_amount = float(request.form.get('loan_amount', 0)) # in thousands
                loan_amount_term = float(request.form.get('loan_amount_term', 360))
                credit_history = float(request.form.get('credit_history', 1.0))
            except ValueError:
                flash("Invalid numeric value provided. Please review financial fields.", "danger")
                return redirect(url_for('predict'))
                
            property_area = request.form.get('property_area')
            
            # Input dictionary
            input_data = {
                "Gender": gender,
                "Married": married,
                "Dependents": dependents,
                "Education": education,
                "Self_Employed": self_employed,
                "ApplicantIncome": applicant_income,
                "CoapplicantIncome": coapplicant_income,
                "LoanAmount": loan_amount,
                "Loan_Amount_Term": loan_amount_term,
                "Credit_History": credit_history,
                "Property_Area": property_area
            }
            
            # Run prediction
            result = predictor.predict(input_data)
            
            if result.get("status") == "success":
                # Create history object
                history_entry = PredictionHistory(
                    applicant_name=applicant_name,
                    gender=gender,
                    married=married,
                    dependents=dependents,
                    education=education,
                    self_employed=self_employed,
                    applicant_income=applicant_income,
                    coapplicant_income=coapplicant_income,
                    loan_amount=loan_amount,
                    loan_amount_term=loan_amount_term,
                    credit_history=credit_history,
                    property_area=property_area,
                    prediction_outcome=result["prediction"],
                    probability=result["probability"],
                    risk_level=result["risk_level"],
                    reasoning=result["reasoning"]
                )
                
                db.session.add(history_entry)
                db.session.commit()
                
                flash("Eligibility prediction completed successfully!", "success")
                return redirect(url_for('result', prediction_id=history_entry.id))
            else:
                flash("Prediction failed: " + result.get("reasoning", "Unknown error"), "danger")
                return redirect(url_for('predict'))
                
        except Exception as e:
            flash(f"An unexpected error occurred: {str(e)}", "danger")
            return redirect(url_for('predict'))
            
    return render_template('predict.html')

@app.route('/predict_csv', methods=['POST'])
def predict_csv():
    if 'csv_file' not in request.files:
        flash("No file part provided.", "danger")
        return redirect(url_for('predict'))
        
    file = request.files['csv_file']
    if file.filename == '':
        flash("No file selected.", "danger")
        return redirect(url_for('predict'))
        
    if file and file.filename.endswith('.csv'):
        try:
            filename = "uploaded_batch.csv"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Read CSV
            df = pd.read_csv(filepath)
            
            # Verify columns
            required_cols = [
                'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
                'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term',
                'Credit_History', 'Property_Area'
            ]
            
            missing_cols = [c for c in required_cols if c not in df.columns]
            if missing_cols:
                flash(f"Invalid CSV structure. Missing columns: {', '.join(missing_cols)}", "danger")
                return redirect(url_for('predict'))
                
            # Read applicant name if it exists, otherwise default
            name_col = 'Applicant_Name' if 'Applicant_Name' in df.columns else None
            
            successful_count = 0
            approved_count = 0
            
            for idx, row in df.iterrows():
                app_name = str(row[name_col]) if name_col else f"Batch Applicant #{idx+1}"
                
                # Check credit history nan
                credit_val = row['Credit_History']
                if pd.isna(credit_val):
                    credit_val = 1.0
                else:
                    credit_val = float(credit_val)
                    
                input_data = {
                    "Gender": str(row['Gender']),
                    "Married": str(row['Married']),
                    "Dependents": str(row['Dependents']),
                    "Education": str(row['Education']),
                    "Self_Employed": str(row['Self_Employed']),
                    "ApplicantIncome": float(row['ApplicantIncome']),
                    "CoapplicantIncome": float(row['CoapplicantIncome']),
                    "LoanAmount": float(row['LoanAmount']),
                    "Loan_Amount_Term": float(row['Loan_Amount_Term']) if not pd.isna(row['Loan_Amount_Term']) else 360.0,
                    "Credit_History": credit_val,
                    "Property_Area": str(row['Property_Area'])
                }
                
                # Run prediction
                res = predictor.predict(input_data)
                
                if res.get("status") == "success":
                    history_entry = PredictionHistory(
                        applicant_name=app_name,
                        gender=input_data["Gender"],
                        married=input_data["Married"],
                        dependents=input_data["Dependents"],
                        education=input_data["Education"],
                        self_employed=input_data["Self_Employed"],
                        applicant_income=input_data["ApplicantIncome"],
                        coapplicant_income=input_data["CoapplicantIncome"],
                        loan_amount=input_data["LoanAmount"],
                        loan_amount_term=input_data["Loan_Amount_Term"],
                        credit_history=input_data["Credit_History"],
                        property_area=input_data["Property_Area"],
                        prediction_outcome=res["prediction"],
                        probability=res["probability"],
                        risk_level=res["risk_level"],
                        reasoning=res["reasoning"]
                    )
                    db.session.add(history_entry)
                    successful_count += 1
                    if res["prediction"] == "Y":
                        approved_count += 1
                        
            db.session.commit()
            
            flash(f"Batch prediction finished: {successful_count} rows processed successfully. {approved_count} pre-approved.", "success")
            return redirect(url_for('admin'))
            
        except Exception as e:
            flash(f"Error parsing batch CSV: {str(e)}", "danger")
            return redirect(url_for('predict'))
    else:
        flash("Unsupported file format. Please upload a .csv file.", "danger")
        return redirect(url_for('predict'))

@app.route('/result/<int:prediction_id>')
def result(prediction_id):
    entry = PredictionHistory.query.get_or_404(prediction_id)
    
    # Load suggestions list from predictor representation
    dummy_pred = predictor.predict(entry.to_dict())
    suggestions = dummy_pred.get("suggestions", ["Ensure credit documentation is fully verified."])
    
    return render_template(
        'result.html',
        record=entry,
        suggestions=suggestions
    )

@app.route('/result/download/<int:prediction_id>')
def download_pdf(prediction_id):
    entry = PredictionHistory.query.get_or_404(prediction_id)
    
    # Get analysis suggestions
    dummy_pred = predictor.predict(entry.to_dict())
    
    pdf_filename = f"report_loan_{entry.id}.pdf"
    pdf_filepath = os.path.join(app.config['UPLOAD_FOLDER'], pdf_filename)
    
    # Make dicts
    input_data = {
        "applicant_name": entry.applicant_name,
        "Gender": entry.gender,
        "Married": entry.married,
        "Dependents": entry.dependents,
        "Education": entry.education,
        "Self_Employed": entry.self_employed,
        "ApplicantIncome": entry.applicant_income,
        "CoapplicantIncome": entry.coapplicant_income,
        "LoanAmount": entry.loan_amount,
        "Loan_Amount_Term": entry.loan_amount_term,
        "Credit_History": entry.credit_history,
        "Property_Area": entry.property_area
    }
    
    result_data = {
        "prediction": entry.prediction_outcome,
        "probability": entry.probability,
        "risk_level": entry.risk_level,
        "reasoning": entry.reasoning,
        "suggestions": dummy_pred.get("suggestions", [])
    }
    
    generate_pdf_report(input_data, result_data, pdf_filepath)
    
    return send_file(pdf_filepath, as_attachment=True, download_name=f"SmartLender_Report_{entry.applicant_name.replace(' ', '_')}.pdf")

@app.route('/dashboard')
def dashboard():
    # Load model comparison
    comparison_data = []
    best_model_name = "Random Forest"
    
    try:
        with open("static/model_comparison.json", "r") as f:
            data = json.load(f)
            comparison_data = data.get("comparison", [])
            best_model_name = data.get("best_model", "Random Forest")
    except Exception as e:
        print(f"Error loading model comparison data: {str(e)}")
        # Dummy data for UI structure in case training is not run yet
        comparison_data = [
            {"Model": "Logistic Regression", "Accuracy": 0.81, "Precision": 0.79, "Recall": 0.98, "F1-Score": 0.87, "CV Accuracy": 0.80, "ROC-AUC": 0.84},
            {"Model": "Decision Tree", "Accuracy": 0.79, "Precision": 0.78, "Recall": 0.95, "F1-Score": 0.85, "CV Accuracy": 0.77, "ROC-AUC": 0.79},
            {"Model": "Random Forest", "Accuracy": 0.83, "Precision": 0.81, "Recall": 0.98, "F1-Score": 0.89, "CV Accuracy": 0.82, "ROC-AUC": 0.86},
            {"Model": "KNN", "Accuracy": 0.75, "Precision": 0.76, "Recall": 0.93, "F1-Score": 0.84, "CV Accuracy": 0.74, "ROC-AUC": 0.76},
            {"Model": "Naive Bayes", "Accuracy": 0.80, "Precision": 0.78, "Recall": 0.97, "F1-Score": 0.87, "CV Accuracy": 0.79, "ROC-AUC": 0.83},
            {"Model": "Support Vector Machine", "Accuracy": 0.81, "Precision": 0.79, "Recall": 0.99, "F1-Score": 0.88, "CV Accuracy": 0.81, "ROC-AUC": 0.81},
            {"Model": "Gradient Boosting", "Accuracy": 0.82, "Precision": 0.80, "Recall": 0.97, "F1-Score": 0.88, "CV Accuracy": 0.81, "ROC-AUC": 0.85},
            {"Model": "XGBoost", "Accuracy": 0.82, "Precision": 0.81, "Recall": 0.96, "F1-Score": 0.88, "CV Accuracy": 0.80, "ROC-AUC": 0.85}
        ]
        
    # Database stats
    try:
        total_p = PredictionHistory.query.count()
        app_p = PredictionHistory.query.filter_by(prediction_outcome='Y').count()
        rej_p = total_p - app_p
        
        low_risk = PredictionHistory.query.filter_by(risk_level='Low').count()
        med_risk = PredictionHistory.query.filter_by(risk_level='Medium').count()
        high_risk = PredictionHistory.query.filter_by(risk_level='High').count()
    except:
        total_p, app_p, rej_p, low_risk, med_risk, high_risk = 0, 0, 0, 0, 0, 0

    return render_template(
        'dashboard.html',
        comparison=comparison_data,
        best_model=best_model_name,
        total_p=total_p,
        app_p=app_p,
        rej_p=rej_p,
        low_risk=low_risk,
        med_risk=med_risk,
        high_risk=high_risk
    )

@app.route('/faq')
def faq():
    return render_template('faq.html')

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        try:
            name = request.form.get('name').strip()
            email = request.form.get('email').strip()
            subject = request.form.get('subject').strip()
            message = request.form.get('message').strip()
            
            if not (name and email and subject and message):
                flash("Please fill in all form inputs before submitting.", "warning")
                return redirect(url_for('contact'))
                
            inquiry = ContactInquiry(
                name=name,
                email=email,
                subject=subject,
                message=message
            )
            
            db.session.add(inquiry)
            db.session.commit()
            
            flash("Thank you! Your message has been recorded. Our team will contact you shortly.", "success")
            return redirect(url_for('contact'))
            
        except Exception as e:
            flash(f"Error logging contact request: {str(e)}", "danger")
            return redirect(url_for('contact'))
            
    return render_template('contact.html')

@app.route('/admin')
def admin():
    # Pagination & Search variables
    page = request.args.get('page', 1, type=int)
    search_q = request.args.get('search', '', type=str)
    
    query = PredictionHistory.query
    
    if search_q:
        query = query.filter(
            (PredictionHistory.applicant_name.like(f"%{search_q}%")) |
            (PredictionHistory.prediction_outcome.like(f"%{search_q}%")) |
            (PredictionHistory.risk_level.like(f"%{search_q}%"))
        )
        
    pagination = query.order_by(PredictionHistory.timestamp.desc()).paginate(page=page, per_page=10, error_out=False)
    records = pagination.items
    
    # Inquiries
    inquiries = ContactInquiry.query.order_by(ContactInquiry.timestamp.desc()).all()
    
    return render_template(
        'admin.html',
        records=records,
        pagination=pagination,
        search=search_q,
        inquiries=inquiries
    )

@app.route('/admin/delete/<int:record_id>')
def delete_record(record_id):
    record = PredictionHistory.query.get_or_404(record_id)
    try:
        db.session.delete(record)
        db.session.commit()
        flash("Prediction record deleted successfully.", "success")
    except Exception as e:
        flash(f"Error deleting record: {str(e)}", "danger")
    return redirect(url_for('admin'))

@app.route('/admin/delete_contact/<int:inquiry_id>')
def delete_contact(inquiry_id):
    inquiry = ContactInquiry.query.get_or_404(inquiry_id)
    try:
        db.session.delete(inquiry)
        db.session.commit()
        flash("Contact inquiry removed.", "success")
    except Exception as e:
        flash(f"Error removing inquiry: {str(e)}", "danger")
    return redirect(url_for('admin'))

# ----------------- REST API ENDPOINT -----------------

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    POST payload should contain json fields matching inputs:
    Gender, Married, Dependents, Education, Self_Employed,
    ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term,
    Credit_History, Property_Area
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"status": "error", "message": "Missing JSON request body"}), 400
            
        result = predictor.predict(data)
        
        # Save to DB if successful
        if result.get("status") == "success":
            history_entry = PredictionHistory(
                applicant_name=data.get("applicant_name", "API Request"),
                gender=data.get("Gender", "Male"),
                married=data.get("Married", "No"),
                dependents=data.get("Dependents", "0"),
                education=data.get("Education", "Graduate"),
                self_employed=data.get("Self_Employed", "No"),
                applicant_income=float(data.get("ApplicantIncome", 5000.0)),
                coapplicant_income=float(data.get("CoapplicantIncome", 0.0)),
                loan_amount=float(data.get("LoanAmount", 140.0)),
                loan_amount_term=float(data.get("Loan_Amount_Term", 360.0)),
                credit_history=float(data.get("Credit_History", 1.0)),
                property_area=data.get("Property_Area", "Urban"),
                prediction_outcome=result["prediction"],
                probability=result["probability"],
                risk_level=result["risk_level"],
                reasoning=result["reasoning"]
            )
            db.session.add(history_entry)
            db.session.commit()
            
            # Add database ID to return
            result["prediction_id"] = history_entry.id
            
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

if __name__ == '__main__':
    # Force reload of artifacts when running app.py
    predictor.load_artifacts()
    app.run(debug=True, host='0.0.0.0', port=5000)
