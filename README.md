# Diabetes Risk Prediction 🩺
* Diabetes risk prediction for analyzing paitents health and predicting the risk level.
* Built using HTML, Css and bootstrap for frontend and machine learning algorithms using python
   libraries like scikit-learn, numpy, pandas, joblib and fastapi for model serving.

# Project Structure 
📦 Diabetes-Prediction-App
├── data/
|   └── diabetes.zip
|
|
├── 📂 extracted_data/
|   ├── data_ingestion.py
│   └── diabetes.csv
|
│ 
├── 📁 app
|   ├── risk.py
|
|
├── 📂 Database/
|      ├── medical_db.sql
|      ├── analysis_queries.sql
|      └── clean_view.sql
|
|
├── 📂 src/
|      ├── db_connection.py
|      ├── fetch_data.py
|      ├── load_csv_to_mysql.ipynb
|      ├── preprocessing.py
|      ├── model.py
|      ├── random_forest.py
|      └── rf_model.joblib
|
├──  main.py  
├── requirements.txt
└── README.md

# Structure Info

### 📂 data/
Collected dataset from kaggle (PIMA) diabetes dataset.

### 📂 extracted_data/
performed data ingestion to extract data from .zip file in data file.

- `data_ingestion.py` – used python logic to extract the .zip file
### 📂 app/
Implemented logic for diabetes risk.

- `risk.py` – used python to calculate risk level
### 📂 Database/
Created logic to perform and load the dataset into MySQL Workbench

- `medical_db` – created a database and used to create a PIMA dataset Table
- `analysis_queries.sql` – analyzed the data like identifying null values and statistics
- `clean_view.sql` – Replaced zero values with Nan
  
### 📂 src/
used logic to perform basic data cleaning, preprocessing logic and model Training & Evaluation.

- `db_connection.py` – connected to database and verified to cleaned data.
- `fetch_data.py` – fetched cleaned data from MySQl
- `preprocessing.py` – Handled missing values with mean,splitted train and test datasets, performed standardscaler
- `model.py` – Used Logistic Regression as a base model and evaluated model using classification report, accuracy_score and confusion matrix.
- `rf_model.py` – Used Random Forest as a advanced model and evaluated model using classification report, accuracy_score and confusion matrix.
- `load_csv_to_mysql.py` – Visualized the results for logistic regression and random forest classifier models and saved to joblib
- `rf_model.joblib` – Saved the model with high accuracy (Random Forest Classifier) with 79% accuracy
**main.py**
Served model using FastAPI

## ⚙️ How It Works

1. The user enters health parameters such as pregnancies,mGlucose level, BMI, Age, Blood Pressure, SkinThickness and Insulin .

2. The submitted data is sent to the FastAPI backend via an HTTP POST request.

3. The backend preprocesses the input data by:
   - Converting it into a numerical feature array
   - Applying the same scaling and transformations used during model training

4. The preprocessed data is passed to the trained machine learning model to generate:
   - A binary prediction (Diabetic / Non-Diabetic)
   - A probability score indicating risk confidence

5. Based on the predicted probability:
   - Risk level is classified as Low, Medium, or High
   - Personalized health recommendations are generated using rule-based logic

6. The prediction result, probability score, risk level, and recommendations are rendered back to the user.

### Risk Level Logic
- Probability < 0.30 → Low Risk
- Probability between 0.30 and 0.60 → Medium Risk
- Probability > 0.60 → High Risk


# Tech Stack ⚙️

| Technology | Usage |
|-----------|--------|
| Python | Backend logic |
| Scikit-learn | Machine learning |
| FastAPI | API framework |
| Pandas | Data processing |
| NumPy | Numerical computation ||


## 🚀 Run Locally

### 1. Clone the Repository
```bash
https://github.com/Ajju-03/diabetes-risk-prediction-sql-python.git
 
### 2. Create a Virtual Environment

python -m venv venv

### 3. Activate it

**Windows**

venv\Scripts\activate

**MacOS**

source venv/bin/activate

### 3. install dependencies

pip install -r requirements.txt

### 4. Start the Application
uvicorn app.main:app --reload

### API Documentation
FastAPI automatically generates interactive API docs:

- Swagger UI: http://127.0.0.1:8000/docs
- ReDoc: http://127.0.0.1:8000/redoc
