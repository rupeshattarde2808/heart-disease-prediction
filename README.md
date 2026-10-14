#  Cardio-Scope — Heart Disease Prediction System

<p align="center">
  <strong>An AI-powered heart disease classification dashboard built with Python, Scikit-learn, XGBoost, and Streamlit.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/Machine%20Learning-Scikit--learn-orange" alt="Machine Learning">
  <img src="https://img.shields.io/badge/XGBoost-Model-green" alt="XGBoost">
  <img src="https://img.shields.io/badge/Status-Active-success" alt="Project Status">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="MIT License">
</p>

---

##  Overview

**Cardio-Scope** is a machine learning project that classifies heart disease status using patient health information. It provides an interactive web dashboard where users can enter clinical measurements, generate predictions, and compare the outputs of multiple trained machine learning models.

> **Disclaimer:** This project is intended for educational and research purposes only. It is not a medical diagnostic tool, and its model scores are not clinically validated heart disease or heart attack risk estimates.

##  Features

*  **Patient Information Form** — Enter demographic information and clinical measurements.
*  **Multiple ML Models** — Compare Logistic Regression, Random Forest, and XGBoost when their trained model files are available.
*  **Binary Classification** — Predict the negative or positive heart disease class.
*  **Model Score Comparison** — View positive-class and negative-class model scores.
*  **Interactive Visualizations** — Explore model comparisons using bar charts, line charts, and progress indicators.

##  Dashboard Preview

The dashboard is designed around three main sections:

| Section             | Description                                                        |
| ------------------- | ------------------------------------------------------------------ |
| Patient Information | Enter the input features used by the trained models.               |
| Risk Assessment     | Review predictions, class scores, and model comparisons.           |
| Model Information   | Inspect the available trained models and understand their outputs. |


##  Tech Stack

| Technology                    | Purpose                                       |
| ----------------------------- | --------------------------------------------- |
| Python                        | Core programming language                     |
| Streamlit                     | Interactive web application                   |
| Pandas                        | Data manipulation and input preparation       |
| NumPy                         | Numerical operations                          |
| Scikit-learn                  | Machine learning pipelines and classification |
| XGBoost                       | Gradient-boosting classification              |
| Joblib                        | Saving and loading trained models             |
| Matplotlib / Streamlit Charts | Visualization, where applicable               |

## Machine Learning Models

The project supports three classification algorithms:

### 1. Logistic Regression

A classification algorithm that estimates the probability of belonging to a class. It also serves as a useful baseline model.

### 2. Random Forest

An ensemble algorithm that combines predictions from multiple decision trees to classify patient data.

### 3. XGBoost

A gradient-boosting algorithm that builds decision trees sequentially to improve predictive performance.

The application loads available trained models from the `models/` directory and identifies the model selected during training using `models/best_model.txt`.

### Model Evaluation

The following are the previously observed test-set accuracy results from this project. They are not guaranteed to reproduce identically across different environments or training configurations.

| Model               | Test Accuracy |
| ------------------- | ------------: |
| Logistic Regression |        83.70% |
| Random Forest       |        84.78% |
| XGBoost             |        85.87% |

**Evaluation note:** Accuracy alone does not establish clinical suitability. Model evaluation should also consider precision, recall, F1-score, confusion matrices, calibration, and performance on representative independent data.

##  Installation and Setup

Follow these steps to run the project locally.

### Prerequisites

* Python installed on your system
* Git
* Visual Studio Code or another Python editor

### 1. Clone the repository

Replace the example URL with your repository URL.

```bash
git clone https://github.com/YOUR_USERNAME/heart-disease-prediction.git
cd heart-disease-prediction
```

### 2. Create a virtual environment

**Windows PowerShell**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can use Command Prompt instead:

```cmd
venv\Scripts\activate.bat
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Prepare the dataset

Place the dataset at:

```text
dataset/heart_disease_uci.csv
```

Ensure that the CSV contains the required input columns and the `num` target column expected by `train.py`.

### 5. Train the models

```bash
python train.py
```

This step trains the configured classification models and saves the generated model files in the `models/` directory. It also creates or updates `best_model.txt` according to your training script.

### 6. Launch the Streamlit application

```bash
python -m streamlit run app.py
```

Open the local URL displayed in your terminal, typically:

```text
http://localhost:8501
```

##  How to Use

1. Launch the Streamlit application.
2. Enter the patient information using the sidebar.
3. Click **Assess Model Scores**.
4. Review the predicted class from each available model.
5. Compare the positive-class and negative-class scores.
6. Explore the charts and open the Model Information tab to inspect the available models.

## Dataset and Input Features

The project uses the UCI Heart Disease dataset in CSV format.

The current application uses these 14 input features:

| Feature    | Description                           |
| ---------- | ------------------------------------- |
| `age`      | Age in years                          |
| `sex`      | Recorded sex category                 |
| `dataset`  | Dataset source category               |
| `cp`       | Chest pain type                       |
| `trestbps` | Resting blood pressure                |
| `chol`     | Serum cholesterol                     |
| `fbs`      | Fasting blood sugar indicator         |
| `restecg`  | Resting electrocardiographic result   |
| `thalch`   | Maximum heart rate achieved           |
| `exang`    | Exercise-induced angina indicator     |
| `oldpeak`  | ST depression measurement             |
| `slope`    | Slope of the peak exercise ST segment |
| `ca`       | Number of major vessels               |
| `thal`     | Thalassemia-related test category     |

The original `num` target is converted into a binary classification target during training:

* **Class 0:** Negative class in the dataset.
* **Class 1:** Positive class in the dataset.

These classes represent the training target, not a confirmed clinical diagnosis. The application does not identify a specific heart disease type.

##  Understanding Model Scores

The dashboard displays model-estimated probabilities for the two classes.

* **Negative-class score:** The model's estimated probability for class 0.
* **Positive-class score:** The model's estimated probability for class 1.
* **Combined score:** The simple average of the positive-class scores from the available models.

The combined score is a dashboard summary, not a separately trained or clinically calibrated ensemble. A higher positive-class score does not establish an individual's actual probability of having a heart attack.

##  Contributing

Contributions, suggestions, and bug reports are welcome.

1. Fork the repository.
2. Create a feature branch.
3. Make your changes and test them.
4. Commit your changes with a descriptive message.
5. Open a pull request.

##  License

This project is distributed under the MIT License. See the [`LICENSE`](LICENSE) file for details.

##  Author

**Rupesh Attarde**

* GitHub: [@rupeshattarde2808](https://github.com/rupeshattarde2808)

---

<p align="center">
  <strong>Built with Python, machine learning, and Streamlit </strong>
</p>

<p align="center">
  If you find this project useful, consider giving the repository a ⭐ on GitHub.
</p>
