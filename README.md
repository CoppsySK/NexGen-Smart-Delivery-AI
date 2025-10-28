# 🚚 NexGen Smart Delivery AI
### Predictive, Prescriptive, and Explainable AI for Logistics Optimization

---

## 🧠 Overview
**NexGen Smart Delivery AI** is an intelligent logistics optimization platform built for the **OFI Placement Challenge**.  
It uses **Machine Learning, Prescriptive Analytics, and Explainable AI (XAI)** to predict delivery delays, recommend corrective actions, and provide transparent insights through an **interactive Streamlit dashboard**.

This project combines **data science + design thinking** to improve operational efficiency, customer satisfaction, and cost control in logistics.

---

## 🎯 Problem Statement
**Selected Problem:** Predictive Delivery Optimizer (Option 1)

> Design an AI-driven solution that predicts potential delivery delays and provides prescriptive recommendations for improving delivery performance.

The system aims to:
- Predict delivery delays before they occur.
- Suggest actionable interventions (e.g., change carrier, adjust delivery priority).
- Provide transparent explanations using Explainable AI (SHAP).
- Enable managers to make data-driven, cost-efficient decisions.

---

## ⚙️ Features

| Feature | Description |
|----------|-------------|
| 🧩 **Predictive AI** | Uses an XGBoost classifier to predict delivery delays with 87.5% accuracy. |
| 💡 **Prescriptive AI** | Provides actionable recommendations to reduce delay risk. |
| 📊 **Interactive Dashboard** | Visualizes KPIs, delay trends, costs, and customer satisfaction. |
| 🔔 **Alerts & Export** | Flags high-risk deliveries and enables one-click CSV export. |
| 🔍 **Explainable AI** | SHAP visualizations for transparency and decision confidence. |
| 🌿 **Scalable & Sustainable** | Extendable to real-time logistics APIs or eco-routing systems. |

---
## 📊 Model Performance
| Metric | Score |
|---------|--------|
| Accuracy | 0.875 |
| F1 Score | 0.828 |
| ROC-AUC | 0.953 |

**Top Predictive Features:**
- Delivery Status  
- Rating Category  
- Delay Days  
- Total Cost  
- Delivery Cost (INR)

---

## 🧰 Technology Stack
- **Programming Language:** Python  
- **Framework:** Streamlit  
- **Libraries:** XGBoost, Pandas, NumPy, Scikit-learn, Plotly, SHAP, Matplotlib  
- **Development Tools:** VS Code, Jupyter Notebook  

---

## ⚙️ How to Run the Project (Google Drive Version)

### 1️⃣ Using Google Drive
- Download the project folder as a **ZIP file**.  
- Extract it to your desktop.

---

### 2️⃣ Install Dependencies
Ensure you have **Python 3.10+** installed, then open a terminal in the extracted folder and run:
```bash
pip install -r requirements.txt
```
If you don’t have a `requirements.txt` file, install manually:

```bash
pip install streamlit pandas numpy scikit-learn xgboost shap plotly matplotlib
```

### 3️⃣ Run the Streamlit Dashboard

Launch the app with:

```bash
streamlit run your_app.py
```
This will start a local server.
You’ll see a message like:

```bash
Local URL: http://localhost:8501
```
Open this link in your browser to access the interactive dashboard.

---

### 4️⃣ Explore the Dashboard Modes
- Dashboard Mode: View delivery performance and cost analytics.

- Predict & Recommend Mode: Input new order data to predict delay risk and get AI-driven recommendations.

- Alerts & Export Mode: Flag and download high-risk deliveries.

- Explainable AI Mode: View SHAP plots that explain model predictions.

Below is a snapshot of the **mode selection panel** in Streamlit, showing all the available modes clearly visible for easy access 👇  

<p align="center">
  <img src="https://github.com/CoppsySK/NexGen-Smart-Delivery-AI/blob/main/mode_selector.png" alt="Mode Selector Screenshot" width="700">
</p>

*This ensures that reviewers can easily navigate through all implemented functionalities without relying on default selections.*
---

5️⃣ (Optional) Run Data Preparation or EDA Scripts
If you want to regenerate datasets or plots:

```bash
python data_preparation.py
python eda_analysis.py
```
---

### ✅ Troubleshooting
If XGBoost or Scikit-learn throws a version error:

```bash
pip install xgboost==2.0.3 scikit-learn==1.5.1
```
- Ensure .pkl model files are present inside the /model directory.

- Run all commands from the root project folder.

---

### 📈 Key Insights from EDA
- 35% of deliveries faced delays.

- Higher fuel, packaging, and total costs strongly correlated with delay probability.

- Low customer ratings increased delay likelihood.

- Certain carriers showed consistent underperformance.

---

### 📂 Access Links
📘 Final Report (PDF): Google Drive Link Here

💻 Source Code (GitHub): GitHub Repository Link

---

### 🧠 Future Enhancements
- Integrate live API data and GPS tracking.

- Add real-time route and cost optimization.

- Include fleet-level predictive maintenance and eco-efficiency modules.

---

### 🙌 Developer
Developed by: Shivam Khosla
Degree: B.Tech CSE, Manipal University Jaipur
Under: OFI Placement Challenge — NextGen Logistics Track

⭐ “Turning Data into Decisions — Faster, Smarter, and Explainable.”
