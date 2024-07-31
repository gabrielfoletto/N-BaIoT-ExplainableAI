import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Load the dataset
df = pd.read_csv('')

# Separate features and labels
X = df.drop('label', axis=1)
y = df['label']

# Encode the labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Train the XGBoost model with cross-validation
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

# Fit the model on the complete training set
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Get class names
class_names = label_encoder.classes_

# Calculate SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Calculate and display classification metrics
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))
print("Accuracy:", accuracy_score(y_test, y_pred))

# Function to calculate metrics
def calculate_metrics(y_true, y_pred):
    conf_matrix = confusion_matrix(y_true, y_pred)
    VN = conf_matrix[0, 0]
    FN = conf_matrix[0, 1:].sum()
    FP = conf_matrix[1:, 0].sum() + (conf_matrix[:, 1:].sum(axis=0) - np.diag(conf_matrix)[1:]).sum()
    VP = np.diag(conf_matrix)[1:].sum()

    accuracy = (VP + VN) / (VP + VN + FP + FN)
    precision = VP / (VP + FP) if (VP + FP) > 0 else 0
    recall = VP / (VP + FN) if (VP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"VN: {VN}, FN: {FN}, VP: {VP}, FP: {FP}")
    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")

# Sum the absolute SHAP values for each feature
if isinstance(shap_values, list):
    # For each class, sum the absolute SHAP values
    shap_sum = np.sum([np.abs(shap_values[i]).mean(axis=0) for i in range(len(shap_values))], axis=0)
else:
    shap_sum = np.abs(shap_values).mean(axis=0)

# Create a DataFrame of feature importance
importance_df = pd.DataFrame({'feature': X.columns, 'importance': shap_sum})
importance_df = importance_df.sort_values(by='importance', ascending=False)
top_features = importance_df['feature'].tolist()

# Remove top features and re-evaluate the model
num_features_to_remove = 10  # e.g., remove the 10 most important features
features_to_remove = top_features[:num_features_to_remove]
X_train_reduced = X_train.drop(columns=features_to_remove)
X_test_reduced = X_test.drop(columns=features_to_remove)

# Train a new model without the most important features
model_reduced = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
model_reduced.fit(X_train_reduced, y_train)

# Make predictions with the reduced model
y_pred_reduced = model_reduced.predict(X_test_reduced)

# Calculate and display classification metrics for the reduced model
print("\nClassification Report (Model with Reduced Features):")
print(classification_report(y_test, y_pred_reduced, target_names=class_names))
print("Accuracy (Model with Reduced Features):", accuracy_score(y_test, y_pred_reduced))

# Calculate metrics with the adapted function
print("\nMetrics for Model with Reduced Features:")
calculate_metrics(y_test, y_pred_reduced)

# Calculate SHAP values for the reduced model
explainer_reduced = shap.TreeExplainer(model_reduced)
shap_values_reduced = explainer_reduced.shap_values(X_test_reduced)

# Create a modified label vector
y_labels = np.array([f"Class {idx} - {class_name}" for idx, class_name in enumerate(class_names)])

# Index this vector with y_test values to get the corresponding labels
y_test_labels = y_labels[y_test]

# Plot the SHAP summary plot split by classes
shap.summary_plot(shap_values_reduced, X_test_reduced, feature_names=X_test_reduced.columns, class_names=y_labels)
plt.title("SHAP Summary Plot (Model with Reduced Features)")
plt.show()

# Plot SHAP summary plot for each class individually without features with SHAP value = 0
for i, class_name in enumerate(class_names):
    print(f"Plotting for class: {class_name}")
    shap_values_class = shap_values_reduced[i]
    non_zero_shap_indices = np.where(np.abs(shap_values_class).mean(axis=0) > 0)[0]
    shap.summary_plot(shap_values_class[:, non_zero_shap_indices], X_test_reduced.iloc[:, non_zero_shap_indices], feature_names=X_test_reduced.columns[non_zero_shap_indices])
    plt.title(f"SHAP Summary Plot for {class_name} (Model with Reduced Features)")
    plt.show()
