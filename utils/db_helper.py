from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class PredictionHistory(db.Model):
    __tablename__ = 'prediction_history'
    
    id = db.Column(db.Integer, primary_key=True)
    applicant_name = db.Column(db.String(100), nullable=True, default="Valued Customer")
    gender = db.Column(db.String(10), nullable=False)
    married = db.Column(db.String(10), nullable=False)
    dependents = db.Column(db.String(10), nullable=False)
    education = db.Column(db.String(20), nullable=False)
    self_employed = db.Column(db.String(10), nullable=False)
    applicant_income = db.Column(db.Float, nullable=False)
    coapplicant_income = db.Column(db.Float, nullable=False)
    loan_amount = db.Column(db.Float, nullable=False)
    loan_amount_term = db.Column(db.Float, nullable=False)
    credit_history = db.Column(db.Float, nullable=False)
    property_area = db.Column(db.String(20), nullable=False)
    
    # Prediction results
    prediction_outcome = db.Column(db.String(5), nullable=False) # 'Y' or 'N'
    probability = db.Column(db.Float, nullable=False)
    risk_level = db.Column(db.String(15), nullable=False) # 'Low', 'Medium', 'High'
    reasoning = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "applicant_name": self.applicant_name,
            "gender": self.gender,
            "married": self.married,
            "dependents": self.dependents,
            "education": self.education,
            "self_employed": self.self_employed,
            "applicant_income": self.applicant_income,
            "coapplicant_income": self.coapplicant_income,
            "loan_amount": self.loan_amount,
            "loan_amount_term": self.loan_amount_term,
            "credit_history": self.credit_history,
            "property_area": self.property_area,
            "prediction_outcome": self.prediction_outcome,
            "probability": self.probability,
            "risk_level": self.risk_level,
            "reasoning": self.reasoning,
            "timestamp": self.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        }

class ContactInquiry(db.Model):
    __tablename__ = 'contact_inquiries'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), nullable=False)
    subject = db.Column(db.String(200), nullable=False)
    message = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "email": self.email,
            "subject": self.subject,
            "message": self.message,
            "timestamp": self.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        }

def init_db(app):
    db.init_app(app)
    with app.app_context():
        db.create_all()
