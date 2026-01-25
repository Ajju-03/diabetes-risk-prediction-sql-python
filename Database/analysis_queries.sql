# few rows of the dataset
SELECT * FROM diabetes_data LIMIT 5;

# row count
SELECT COUNT(*) AS total_records FROM diabetes_data;

# class distribution
SELECT outcome, COUNT(*) AS total 
FROM diabetes_data
GROUP BY outcome;

# Identifying invalid zero values
SELECT 
   SUM(glucose = 0)  AS zero_glucose,
   SUM(bloodpressure = 0) AS zero_bp,
   SUM(skinthickness = 0) AS zero_skinthickness,
   SUM(insulin = 0) AS zero_insulin,
   SUM(bmi = 0) AS zero_bmi
FROM diabetes_data;   

# statsistics
SELECT outcome,
       AVG(glucose) AS avg_glucose,
       AVG(bmi) AS avg_bmi,
       AVG(age) AS avg_age
FROM diabetes_data
GROUP BY outcome;  

# age risk 
SELECT 
     CASE
         WHEN age < 30 THEN 'Young'
         WHEN age BETWEEN 30 AND 50 THEN 'Middle'
         ELSE 'Senior'
     END AS age_group,
     COUNT(*) As total_patients,
     SUM(outcome) AS diabetic_cases
FROM diabetes_data
GROUP BY age_group;     