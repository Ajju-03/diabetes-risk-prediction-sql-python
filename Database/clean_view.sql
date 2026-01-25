CREATE OR REPLACE VIEW
clean_diabetes_data AS
SELECT
      pregnancies,
      NULLIF(glucose, 0) AS glucose,
      NULLIF(bloodpressure, 0) AS bloodpressure,
      NULLIF(skinthickness, 0) AS skinthickness,
      NULLIF(insulin, 0) AS insulin,
      NULLIF(bmi, 0) AS bmi,
      diabetespedigreefunction,
      age,
      outcome
FROM diabetes_data;    

SELECT * FROM clean_diabetes_data LIMIT 5;  