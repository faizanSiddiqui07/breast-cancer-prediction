# Breast Cancer Prediction

A breast cancer classification project using logistic regression trained from scratch with NumPy.

The project has two parts: the Python training code and a small browser app that uses the trained parameters for prediction.

## Live demo

https://breast-cancer-prediction-cyan.vercel.app

## Model

The model uses the 30 diagnostic measurements from the Breast Cancer Wisconsin (Diagnostic) dataset.

`M` is encoded as `1` (malignant) and `B` as `0` (benign).

Before training, the features are standardized with `StandardScaler`. Logistic regression is then trained with batch gradient descent.

```text
z = XW + b
p = sigmoid(z)
```

The loss is binary cross entropy. The weights and bias are updated from the gradient of that loss.

There is no `sklearn.LogisticRegression` in the training code. The actual model is the NumPy implementation in `src/train.py`.

## Training setup

- train/test split: 80/20
- stratified split
- random state: 42
- learning rate: 0.2
- iterations: 12,000
- classification threshold used by the app: 0.25

The threshold is lower than 0.5 because the project puts more emphasis on catching malignant samples. It is part of the deployed model configuration.

## Results

Using the published model parameters:

| Metric | Score |
| --- | ---: |
| Accuracy | 97.37% |
| Precision | 97.56% |
| Recall | 95.24% |
| F1 Score | 96.39% |
| ROC-AUC | 98.48% |

Confusion matrix:

```text
[[71  1]
 [ 2 40]]
```

Rows are actual labels and columns are predicted labels.

## Run the training code

Install the dependencies:

```bash
pip install -r requirements.txt
```

Put the dataset at:

```text
data/data.csv
```

Then run:

```bash
python src/train.py
```

The script trains the model, evaluates it, saves the model parameters used by the frontend, and creates the evaluation plots in `results/`.

## Project structure

```text
breast-cancer-prediction/
│
├── data/
│   └── README.md
│
├── model/
│   └── model-config.js
│
├── results/
│   ├── confusion_matrix.png
│   ├── metrics.png
│   ├── precision_recall_curve.png
│   ├── probability_distribution.png
│   ├── roc_curve.png
│   ├── training_loss.png
│   └── metrics.txt
│
├── src/
│   └── train.py
│
├── index.html
├── style.css
├── app.js
├── model-config.js
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## Web app

The frontend is plain HTML, CSS and JavaScript. It standardizes the entered values using the same saved means and scales, calculates the logistic regression score, applies the sigmoid function and then uses the 0.25 threshold.

The model parameters are stored in `model-config.js`.

## Dataset

The project uses the Breast Cancer Wisconsin (Diagnostic) dataset. The original CSV is intentionally not included in the repository; add your own copy as `data/data.csv` before running the training script.

## Disclaimer

This is an educational machine learning project. It is not a medical diagnostic tool and should not be used for clinical decisions.

## Author

Faizan Siddiqui

GitHub: https://github.com/faizanSiddiqui07
