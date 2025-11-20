# 🏦 Bank Loan Eligibility Prediction

A complete Machine Learning project that predicts whether a bank should approve a loan for a customer based on financial and demographic features.

---

## 📌 Features
- ML Model: Logistic Regression  
- Dataset: 1000 synthetic loan applicant records  
- Web App: Streamlit app for real-time predictions  
- Fully interactive input form  
- Probability-based output  

---

## 📁 Project Structure

loan_eligibility_project/
│
├── data/
│ └── loan_data.csv
│
├── model/
│ ├── train_model.py
│ ├── loan_model.pkl
│ └── scaler.pkl
│
├── app/
│ ├── app.py
│ └── requirements.txt
│
└── README.md

yaml
Copy code

---

## 🚀 How to Run the Project

### 1️⃣ Install dependencies  
pip install -r app/requirements.txt

bash
Copy code

### 2️⃣ Train the Model  
cd model
python train_model.py

markdown
Copy code

This generates:
- `loan_model.pkl`
- `scaler.pkl`

### 3️⃣ Run the Streamlit App  
cd ../app
streamlit run app.py

yaml
Copy code

Your app will open in the browser.

---

## 🔮 Future Enhancements
- Add loan purpose
- Add customer salary slips / credit history PDF upload
- Deploy app on cloud (Render / Hugging Face / Streamlit Cloud)

---

## 👨‍💻 Author
Akshay – AI/ML Developer  
