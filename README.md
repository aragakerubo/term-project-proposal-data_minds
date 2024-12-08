# Credit Card Fraud Detection

This repository contains the code and documentation for a project on **Credit Card Fraud Detection**, completed as part of the CIS 635: Knowledge Discovery and Data Mining course. The project demonstrates the use of machine learning techniques to identify fraudulent credit card transactions in a highly imbalanced dataset.

---

## **Overview**
Fraud detection in credit card transactions is a critical real-world problem. This project explores various machine learning algorithms to improve the accuracy and reliability of fraud detection. The pipeline includes data preprocessing, resampling with SMOTE, model training, and evaluation using advanced metrics suited for imbalanced datasets.

---

## **Features**
- Preprocessing with **Synthetic Minority Over-sampling Technique (SMOTE)** to address class imbalance.
- Comparison of multiple models:
  - Logistic Regression
  - Random Forest
  - XGBoost
  - Neural Networks
  - Ensemble Learning (Voting Classifier)
- Metrics beyond accuracy, including **Precision**, **Recall**, **F1-score**, **ROC-AUC**, and **AUPRC**, to ensure effective fraud detection.

---

## **Results**

| **Model**           | **Precision** | **Recall** | **F1 Score** | **ROC-AUC** | **AUPRC** | **Key Insights**                                                                 |
|----------------------|---------------|------------|--------------|-------------|-----------|----------------------------------------------------------------------------------|
| Logistic Regression  | 0.14          | 0.93       | 0.24         | 0.9825      | 0.8086    | High recall but low precision, leading to many false positives. Suitable for initial screening. |
| Random Forest        | 0.80          | 0.88       | 0.84         | 0.9900      | 0.881     | Strong balance of precision and recall with high reliability. Robust for operational environments. |
| XGBoost              | 0.78          | 0.86       | 0.82         | 0.9911      | 0.8858    | Slightly better ROC-AUC and AUPRC than Random Forest, effectively captures complex interactions. |
| Neural Network       | 0.18          | 0.90       | 0.30         | 0.9684      | 0.823     | High recall but poor precision, leading to frequent false positives.                             |
| Ensemble (Voting)    | 0.80          | 0.87       | 0.83         | 0.9912      | 0.8812    | Combines strengths of Random Forest and XGBoost, achieving a balanced and robust performance.   |

---

## **Usage**
1. Open the notebook directly on **Google Colab**:
   - [Credit Card Fraud Detection Notebook](https://github.com/aragakerubo/term-project-proposal-data_minds/blob/main/Credit_Card_Fraud_Detection.ipynb)

2. All dependencies are specified within the notebook and can be installed directly in the Colab environment.

---

## **Dependencies**
This project uses the following Python libraries:
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `imblearn`
- `scikit-learn`
- `xgboost`

Google Colab will handle most library installations automatically. For any missing libraries, use:
```python
!pip install library_name
```

---


## **Resources**
- **Dataset**: [Credit Card Fraud Dataset](https://github.com/GVSU-CIS635/Datasets/raw/refs/heads/master/creditcard.csv.zip)
- **Project Notebook**: [Credit_Card_Fraud_Detection.ipynb](https://github.com/aragakerubo/term-project-proposal-data_minds/blob/main/Credit_Card_Fraud_Detection.ipynb)

---

## **Future Directions**
- Experiment with unsupervised learning methods to detect novel fraud patterns.
- Incorporate Explainable AI (XAI) for enhanced transparency.
- Extend the analysis for real-time fraud detection with temporal and geospatial features.

