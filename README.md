# AIDP Project

## Overview
The **AIDP Project** is a data pipeline that processes, analyzes, models, and performs inference on datasets. It is structured into four main components:

1. **Preprocessing (`preprocess/`)**: Cleans and transforms raw data into `preprocessed_train.csv` and `preprocessed_test.csv`.
2. **Exploratory Data Analysis (`eda/`)**: A Flask web application that visualizes data insights using graphs.
3. **Modeling (`modelling/`)**: Trains a machine learning model and saves it as a `.pkl` file.
4. **Inference (`inference/`)**: Uses the trained model to make predictions on new data.

Each component has its own `Dockerfile` and `requirements.txt` to ensure modularity and ease of deployment.

## Installation & Setup
To set up the project, follow these steps:

1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd AIDP
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run each module individually as per the need.

## Running the Modules

### 1. Preprocessing
```bash
cd preprocess
python preprocess.py
```

### 2. Exploratory Data Analysis (EDA)
```bash
cd eda
python egt309proj.py
```
This will start a Flask server where visualizations can be accessed.

### 3. Model Training
```bash
cd modelling
python model.py
```

### 4. Inference
```bash
cd inference
python egt309_inf.py
```

## Dependencies
The project requires the following Python libraries:
```txt
numpy==2.2.3
pandas==2.2.3
matplotlib==3.10.0
seaborn==0.13.2
Flask
```
These can be installed via `pip install -r requirements.txt`.

## Docker Support
Each module contains a `Dockerfile` for containerization. To build and run a module, use:
```bash
docker build -t <module-name> .
docker run -p 5000:5000 <module-name>
```

## Contact
For any queries, reach out to the project contributors.

