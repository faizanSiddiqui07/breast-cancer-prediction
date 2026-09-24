import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DATA_PATH = "data/data.csv"
MODEL_PATH = "model/model-config.js"
RESULTS_DIR = "results"

LEARNING_RATE = 0.2
ITERATIONS = 12000
THRESHOLD = 0.25


# load the dataset
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Could not find {DATA_PATH}. Put the dataset in data/data.csv first."
    )

df = pd.read_csv(DATA_PATH)
print("Shape:", df.shape)
print(df.head())

# these columns are not model features
# id is just an identifier and Unnamed: 32 is an empty column in this dataset
df = df.drop(["id", "Unnamed: 32"], axis=1)

print("\nMissing values:", df.isnull().sum().sum())
print("Duplicates:", df.duplicated().sum())

# M = malignant, B = benign
df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

print("\nTarget distribution:")
print(df["diagnosis"].value_counts())

X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]

# keep the same class proportion in train and test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print("\nTraining shape:", X_train.shape)
print("Testing shape:", X_test.shape)

# fit the scaler only on training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_np = X_train_scaled
y_train_np = y_train.to_numpy()
X_test_np = X_test_scaled
y_test_np = y_test.to_numpy()

# start with zero weights
W = np.zeros(X_train_np.shape[1])
b = 0.0


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


loss_history = []

for i in range(ITERATIONS):
    # forward pass
    z = X_train_np @ W + b
    p = sigmoid(z)

    # avoid log(0) in the loss
    p_clipped = np.clip(p, 1e-15, 1 - 1e-15)

    loss = -np.mean(
        y_train_np * np.log(p_clipped)
        + (1 - y_train_np) * np.log(1 - p_clipped)
    )

    # gradient of the loss with respect to W and b
    error = p - y_train_np
    dW = (X_train_np.T @ error) / len(y_train_np)
    db = np.mean(error)

    # gradient descent
    W -= LEARNING_RATE * dW
    b -= LEARNING_RATE * db

    loss_history.append(loss)

    if i % 100 == 0:
        print(f"Iteration {i}, Loss: {loss:.4f}")

# test probabilities
z_test = X_test_np @ W + b
y_prob = sigmoid(z_test)
y_pred = (y_prob >= THRESHOLD).astype(int)

cm = confusion_matrix(y_test_np, y_pred)
accuracy = accuracy_score(y_test_np, y_pred)
precision = precision_score(y_test_np, y_pred)
recall = recall_score(y_test_np, y_pred)
f1 = f1_score(y_test_np, y_pred)
auc = roc_auc_score(y_test_np, y_prob)

print("\nConfusion Matrix:")
print(cm)
print("\nAccuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)
print("ROC AUC  :", auc)

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# save the same parameters used by the browser app
config = {
    "weights": W.tolist(),
    "bias": float(b),
    "means": scaler.mean_.tolist(),
    "scales": scaler.scale_.tolist(),
    "threshold": THRESHOLD,
}

with open(MODEL_PATH, "w") as f:
    f.write("const MODEL_CONFIG = ")
    json.dump(config, f, indent=2)
    f.write(";\n")

with open(os.path.join(RESULTS_DIR, "metrics.txt"), "w") as f:
    f.write("Model: Logistic Regression from scratch\n")
    f.write("Dataset: Breast Cancer Wisconsin (Diagnostic)\n")
    f.write("Train/test split: 80/20\n")
    f.write("Random state: 42\n")
    f.write(f"Classification threshold: {THRESHOLD}\n\n")
    f.write(f"Accuracy : {accuracy}\n")
    f.write(f"Precision: {precision}\n")
    f.write(f"Recall   : {recall}\n")
    f.write(f"F1 Score : {f1}\n")
    f.write(f"ROC AUC  : {auc}\n\n")
    f.write("Confusion Matrix:\n")
    f.write(str(cm))

# training loss
plt.figure(figsize=(8, 5))
plt.plot(loss_history)
plt.xlabel("Iteration")
plt.ylabel("Binary cross entropy")
plt.title("Training Loss")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "training_loss.png"), dpi=300)
plt.close()

# confusion matrix
plt.figure(figsize=(7, 6))
plt.imshow(cm)
plt.title("Confusion Matrix", fontsize=16, pad=35)
plt.xticks([0, 1], ["Benign", "Malignant"], fontsize=11)
plt.yticks([0, 1], ["Benign", "Malignant"], fontsize=11)
plt.gca().xaxis.tick_top()
plt.gca().xaxis.set_label_position("top")
plt.xlabel("Predicted Label", fontsize=12, labelpad=15)
plt.ylabel("Actual Label", fontsize=12)

labels = [
    (0, 0, "True Negative"),
    (1, 0, "False Positive"),
    (0, 1, "False Negative"),
    (1, 1, "True Positive"),
]
for x, y_pos, name in labels:
    plt.text(x, y_pos, f"{cm[y_pos, x]}\n{name}", ha="center", va="center", fontsize=11)

plt.colorbar()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"), dpi=300, bbox_inches="tight")
plt.close()

# metrics
metrics = {
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1 Score": f1,
    "ROC AUC": auc,
}

plt.figure(figsize=(8, 5))
plt.bar(metrics.keys(), metrics.values())
plt.ylim(0, 1.05)
plt.ylabel("Score")
plt.title("Model Performance")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "metrics.png"), dpi=300, bbox_inches="tight")
plt.close()

# ROC curve
fpr, tpr, _ = roc_curve(y_test_np, y_prob)
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "roc_curve.png"), dpi=300)
plt.close()

# precision-recall curve
precision_values, recall_values, _ = precision_recall_curve(y_test_np, y_prob)
plt.figure(figsize=(7, 5))
plt.plot(recall_values, precision_values)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "precision_recall_curve.png"), dpi=300)
plt.close()

# probability distribution
plt.figure(figsize=(8, 5))
plt.hist(y_prob[y_test_np == 0], bins=20, alpha=0.6, label="Benign")
plt.hist(y_prob[y_test_np == 1], bins=20, alpha=0.6, label="Malignant")
plt.axvline(THRESHOLD, linestyle="--", label=f"Threshold = {THRESHOLD}")
plt.xlabel("Predicted Probability")
plt.ylabel("Number of Samples")
plt.title("Prediction Probability Distribution")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "probability_distribution.png"), dpi=300)
plt.close()

print("\nModel parameters saved to:", MODEL_PATH)
print("Results saved to:", RESULTS_DIR)
