# EGT309 Titanic Prediction Project

## Overview
The **EGT309 Titanic Prediction Project** is a machine learning pipeline for preprocessing, exploratory data analysis (EDA), modeling, and inference on the Titanic dataset. It is structured into four main components:

1. **Preprocessing (`309_preprocess.py`)**: Cleans and transforms raw Titanic data.
2. **Exploratory Data Analysis (`egt309proj.py`)**: A Flask web application that visualizes data insights.
3. **Model Training (`EGT309_Data_Modelling_Code.py`)**: Trains machine learning models and selects the best one.
4. **Inference (`egt309_inf.py`)**: Uses the trained model to make predictions on new data.

Each component has its own dependencies listed in `requirements.txt` files.

## Installation & Setup
To set up the project, follow these steps:

1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd EGT309_Titanic_Prediction
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run each module individually as per the need.

## Running the Modules

### 1. Preprocessing
```bash
python 309_preprocess.py --train data/train.csv --test data/test.csv --train_output data/preprocessed_train.csv --test_output data/preprocessed_test.csv
```

### 2. Exploratory Data Analysis (EDA)
```bash
python egt309proj.py
```
This will start a Flask server where visualizations can be accessed.

### 3. Model Training
```bash
python EGT309_Data_Modelling_Code.py
```
This script trains multiple models, selects the best one, and saves it as `best_titanic_model.pkl`.

### 4. Inference
```bash
python egt309_inf.py
```
This script loads the trained model and makes predictions on `preprocessed_test.csv`.

## Dependencies
The project requires the following Python libraries:
```txt
pandas==2.2.3
numpy==2.2.3
matplotlib==3.10.0
seaborn==0.13.2
Flask
scikit-learn==1.6.1
scipy==1.15.1
joblib==1.4.2
```
Install these via `pip install -r requirements.txt`.



