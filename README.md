# N-BaIoT Explainable AI

The N-BaIoT (Network-Based Anomaly IoT) dataset is designed for anomaly detection and classification in Internet of Things (IoT) networks. It is commonly used to identify malicious network traffic on IoT devices and to study machine learning techniques and security analysis in network environments.

This project utilizes the SHAP library in conjunction with the XGBoost classification algorithm to analyze the dataset. The script computes performance metrics and generates various types of explainability graphs to enhance the understanding of model predictions. Additionally, the tool.py script consolidates all dataset files into a single file and labels the rows with one of three classes: mirai, gafgyt, or benign.

# Features

    - Data Processing: Consolidates and labels dataset files to streamline analysis.
    - Anomaly Detection: Uses XGBoost for classifying and detecting anomalies in IoT network traffic.
    - Explainability Analysis: Computes SHAP values and generates graphs to explain model predictions.
    - Performance Metrics: Evaluates the model using various performance metrics to ensure accuracy and effectiveness.

# How to Use

    - Prepare the Dataset: Ensure that all dataset files are available and correctly formatted.
    - Run tool.py: This script will combine the dataset files and label the data with one of the three classes.
    - Analyze with SHAP and XGBoost: Execute the provided scripts to process the dataset and generate explainability graphs and performance metrics.


MEIDAN et al. N-BaIoT—Network-Based Detection of IoT Botnet Attacks Using Deep Autoencoders. IEEE Pervasive Computing, [S.l.], v.17, p.12–22, 2018.
