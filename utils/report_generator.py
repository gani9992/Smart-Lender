import os
from fpdf import FPDF
from datetime import datetime

class LoanReportPDF(FPDF):
    def header(self):
        # Draw top banner
        self.set_fill_color(26, 54, 93) # Dark Navy Blue
        self.rect(0, 0, 210, 35, 'F')
        
        # Logo or Title Text
        self.set_text_color(255, 255, 255)
        self.set_font('helvetica', 'B', 16)
        self.cell(0, 15, 'SMART LENDER - ELIGIBILITY ANALYSIS', border=0, ln=1, align='C')
        
        # Sub-title
        self.set_font('helvetica', 'I', 10)
        self.cell(0, -2, 'Automated Loan Risk & Pre-Approval Assessment Report', border=0, ln=1, align='C')
        self.ln(12)

    def footer(self):
        # Position at 1.5 cm from bottom
        self.set_y(-15)
        self.set_font('helvetica', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'Page {self.page_no()}/{{nb}} | SmartLender Automated underwriting systems | Run on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', align='C')

def generate_pdf_report(data, result_data, filepath):
    pdf = LoanReportPDF()
    pdf.alias_nb_pages()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # 1. Customer Details & Application Summary
    pdf.set_y(40)
    pdf.set_font('helvetica', 'B', 14)
    pdf.set_text_color(26, 54, 93)
    pdf.cell(0, 10, "Application Assessment Overview", ln=True)
    pdf.line(10, 48, 200, 48)
    pdf.ln(5)
    
    # Status Banner Card
    pdf.set_font('helvetica', 'B', 12)
    is_approved = result_data["prediction"] == "Y"
    status_text = "PRE-APPROVED" if is_approved else "PROVISIONALLY DECLINED"
    
    if is_approved:
        pdf.set_fill_color(223, 240, 216) # Light green
        pdf.set_text_color(60, 118, 61)   # Dark green text
        pdf.set_draw_color(214, 233, 198)
    else:
        pdf.set_fill_color(242, 222, 222) # Light red
        pdf.set_text_color(169, 68, 66)   # Dark red text
        pdf.set_draw_color(235, 204, 209)
        
    # Draw status card
    pdf.cell(0, 15, f"  DECISION: {status_text}", border=1, ln=True, fill=True)
    pdf.ln(5)
    
    # Score details
    pdf.set_text_color(51, 51, 51)
    pdf.set_font('helvetica', '', 10)
    
    # Grid of details
    # Left column
    x_start = 10
    col_w = 90
    
    pdf.set_font('helvetica', 'B', 10)
    pdf.cell(col_w, 6, f"Applicant Name: {data.get('applicant_name', 'Valued Customer')}", ln=False)
    pdf.cell(col_w, 6, f"Assessment Confidence: {result_data['probability']:.1%}", ln=True)
    
    pdf.cell(col_w, 6, f"Risk Evaluation: {result_data['risk_level']} Risk", ln=False)
    credit_hist_text = "Good" if float(data.get('Credit_History', 1.0)) == 1.0 else "Poor/None"
    pdf.cell(col_w, 6, f"Credit History: {credit_hist_text}", ln=True)
    pdf.ln(4)
    
    # 2. Financial & Demographic Details Table
    pdf.set_font('helvetica', 'B', 12)
    pdf.set_text_color(26, 54, 93)
    pdf.cell(0, 8, "Financial & Demographic Profile", ln=True)
    pdf.ln(1)
    
    # Table headers
    pdf.set_font('helvetica', 'B', 10)
    pdf.set_fill_color(240, 240, 240)
    pdf.set_text_color(26, 54, 93)
    pdf.cell(95, 8, " Parameter Name", border=1, fill=True)
    pdf.cell(95, 8, " Value Provided", border=1, fill=True, ln=True)
    
    # Table body
    pdf.set_font('helvetica', '', 9)
    pdf.set_text_color(51, 51, 51)
    
    profile_rows = [
        ("Gender", data.get("Gender", "Male")),
        ("Marital Status", "Married" if data.get("Married") == "Yes" else "Unmarried"),
        ("Dependents", str(data.get("Dependents", "0"))),
        ("Education", data.get("Education", "Graduate")),
        ("Self-Employed", data.get("Self_Employed", "No")),
        ("Applicant Monthly Income", f"${float(data.get('ApplicantIncome', 0)):,.2f}"),
        ("Co-Applicant Monthly Income", f"${float(data.get('CoapplicantIncome', 0)):,.2f}"),
        ("Requested Loan Amount", f"${float(data.get('LoanAmount', 0))*1000:,.2f}"),
        ("Amortization Term Period", f"{int(float(data.get('Loan_Amount_Term', 360)))} Months"),
        ("Location Property Area", data.get("Property_Area", "Urban"))
    ]
    
    for label, val in profile_rows:
        pdf.cell(95, 7, f" {label}", border=1)
        pdf.cell(95, 7, f" {val}", border=1, ln=True)
        
    pdf.ln(8)
    
    # 3. Model Analysis & Explanations
    pdf.set_font('helvetica', 'B', 12)
    pdf.set_text_color(26, 54, 93)
    pdf.cell(0, 8, "Underwriting Reasoning & Explanation", ln=True)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(3)
    
    pdf.set_font('helvetica', '', 9.5)
    pdf.set_text_color(51, 51, 51)
    pdf.multi_cell(0, 5, result_data["reasoning"])
    pdf.ln(5)
    
    # 4. Actionable Suggestions
    pdf.set_font('helvetica', 'B', 12)
    pdf.set_text_color(26, 54, 93)
    pdf.cell(0, 8, "Actionable Financial Suggestions", ln=True)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(3)
    
    pdf.set_font('helvetica', '', 9.5)
    pdf.set_text_color(51, 51, 51)
    for sug in result_data["suggestions"]:
        pdf.cell(5, 5, chr(149), ln=False, align='C') # bullet point
        pdf.multi_cell(0, 5, f" {sug}")
        
    # Draw disclaimer
    pdf.ln(10)
    pdf.set_font('helvetica', 'I', 8)
    pdf.set_text_color(100, 100, 100)
    pdf.multi_cell(0, 4, "Disclaimer: This assessment is generated automatically by the SmartLender machine learning framework based on input parameters. It does not constitute a formal, binding contract or a final credit guarantee. Official credit review processes apply.")
    
    # Ensure folder path exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    pdf.output(filepath)
    print(f"PDF report generated and written to {filepath}")
