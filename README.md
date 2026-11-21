# 📦 Vendor Invoice Intelligence System

An end-to-end Machine Learning system and interactive Streamlit web portal designed to forecast freight shipping costs and detect anomalous or high-risk vendor invoices for procurement and finance operations.

---

## 🎯 Business Problem

In enterprise procurement and accounts payable:
- **Unpredictable Shipping Costs**: Freight charges often fluctuate, leading to budgeting inaccuracies.
- **Invoice Discrepancies & Fraud**: Manual invoice auditing is slow and prone to overlooking invoice dollar mismatches or prolonged receiving delays.
- **Financial Leakage**: Inaccuracies in payments lead to revenue loss and strained vendor relationships.

This project provides an AI-driven solution featuring:
1. **Freight Cost Prediction Module**: Predicts freight expenses from invoice parameters with high precision.
2. **Invoice Risk Flagging Module**: Flags potential invoice risks (e.g., mismatch between invoice dollars and purchase order items, excessive receiving delays) using trained classification models.
3. **Interactive Portal**: A Streamlit dashboard (`app.py`) for finance and procurement teams to run real-time inference and review predictions.

---

## 📁 Repository Structure

```text
Vendor-Invoice-Intelligence-System/
├── app.py                                # Streamlit Web Portal
├── README.md                             # Project Documentation
├── .gitignore                            # Git Ignore Rules
│
├── freight_cost_prediction/              # Module 1: Freight Cost Regression
│   ├── data_preprocessing.py             # Data extraction and feature engineering
│   ├── model_evaluation.py               # Model training (Linear, Decision Tree, RF)
│   ├── train.py                          # Training and best model selection script
│   ├── Predicting Freight Cost.ipynb     # Exploratory Data Analysis & experiments
│   └── models/                           # Serialized regression models (.pkl, .joblib)
│
├── invoice_flagging/                     # Module 2: Invoice Risk Classification
│   ├── data_preprocessing.py             # SQL queries, labeling rules, feature scaling
│   ├── modeling_evaluation.py            # GridSearchCV Random Forest tuning & evaluation
│   ├── train.py                          # Classification training pipeline
│   ├── Invoice Flagging.ipynb            # Exploratory Data Analysis & experiments
│   └── models/                           # Scaler and classifier models
│
├── inference/                            # Standalone Prediction Services
│   ├── predict_freight.py                # Freight inference utility
│   └── predict_invoice_flag.py           # Invoice risk inference utility
│
├── models/                               # Top-level production models
│   ├── predict_flag_invoice.pkl          # Trained classifier model
│   ├── random_forest_model.pkl           # Best Random Forest model
│   └── scaler.pkl                        # Feature scaler
│
└── data/                                 # Local database storage
    └── inventory.db                      # SQLite database (ignored by git due to size)
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/avi-ooo1/Vendor-Invoice-Intelligence-System.git
cd Vendor-Invoice-Intelligence-System
```

### 2. Set Up Virtual Environment
```bash
# Using conda
conda create -n invoice_env python=3.10 -y
conda activate invoice_env

# OR using venv
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install pandas numpy scikit-learn joblib streamlit plotly matplotlib seaborn
```

---

## 🚀 Usage

### Run the Streamlit Portal
```bash
streamlit run app.py
```

### Retrain Freight Cost Models
```bash
cd freight_cost_prediction
python train.py
```

### Retrain Invoice Risk Flagging Models
```bash
cd invoice_flagging
python train.py
```

---

## 📊 Models & Performance

| Module | Model | Primary Metric | Result |
|---|---|---|---|
| Freight Cost Prediction | Linear Regression | $R^2$ Score | **96.99%** |
| Freight Cost Prediction | Random Forest Regressor | $R^2$ Score | 96.48% |
| Freight Cost Prediction | Decision Tree Regressor | $R^2$ Score | 95.63% |
| Invoice Risk Flagging | Tuned Random Forest Classifier | F1 Score / Accuracy | **High Precision** |

---

## 👤 Author
- **GitHub**: [@avi-ooo1](https://github.com/avi-ooo1)
